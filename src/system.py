"""Симуляция системы: такт управления + RK4 по state_array самолётов.

Такт dt:
    1. каждый самолёт получает u = (a_tang, omega, a_z) от своего Controls
       (u замораживается на весь такт — zero-order hold);
    2. state_array самолёта (x, y, z, V_xy, psi, v_z) интегрируется RK4 за dt (substeps шагов);
    3. дрон сдвигается по своей модели движения (position_at(dt));
    4. положения записываются в траекторию.
"""

import csv
import json
from abc import ABC, abstractmethod

import numpy as np

from flying_objects import (
    AttentionObject,
    Drone,
    DroneConfig,
    DroneTarget,
    Plane,
    PlaneConfig,
)

State = np.typing.NDArray[np.float64]  # (6,): x, y, z, V_xy, psi, v_z
Control = np.typing.NDArray[np.float64]  # (3,): a_tang, omega, a_z


# ------------------------------------------------------------------ RK4


def rhs(s: State, u: Control) -> State:
    """Правая часть: ẋ = V cos ψ, ẏ = V sin ψ, ż = v_z, V̇ = a_tang, ψ̇ = ω, v̇_z = a_z."""
    return np.array([
        s[3] * np.cos(s[4]),
        s[3] * np.sin(s[4]),
        s[5],
        u[0],
        u[1],
        u[2],
    ])


def rk4_step(s: State, u: Control, h: float) -> State:
    k1 = rhs(s, u)
    k2 = rhs(s + 0.5 * h * k1, u)
    k3 = rhs(s + 0.5 * h * k2, u)
    k4 = rhs(s + h * k3, u)
    return s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# ------------------------------------------------------------------ управление


class Controls(ABC):
    """Интерфейс управления одним самолётом. Настоящий controls — отдельным модулем."""

    @abstractmethod
    def decide(self, plane: Plane, others: list[AttentionObject], t: float) -> Control:
        """Команда (a_tang, omega, a_z) на ближайший такт."""


class StraightControls(Controls):
    """Заглушка: все летят прямо (шаг «0» — без реакции на угрозы)."""

    def decide(self, plane: Plane, others: list[AttentionObject], t: float) -> Control:
        return np.zeros(3)


# ------------------------------------------------------------------ система


class System:
    def __init__(self, planes: list[Plane], drone: Drone, target: DroneTarget,
                 controls: dict[str, Controls], dt: float = 1.0, substeps: int = 4,
    ):
        self.planes = planes
        self.drone = drone
        self.target = target
        self.controls = controls  # id самолёта -> его Controls
        self.dt = dt
        self.substeps = substeps
        self.t = 0.0
        # траектория: времена и положения (K, N, 3) / скорости самолётов, положения дрона (K, 3)
        self.times: list[float] = [0.0]
        self.positions: list[np.ndarray] = [np.array([p.r for p in planes])]
        self.velocities: list[np.ndarray] = [np.array([p.v for p in planes])]
        self.drone_positions: list[np.ndarray] = [drone.r.copy()]

    def objects(self) -> list[AttentionObject]:
        return [*self.planes, self.drone, self.target]

    def step(self) -> None:
        objects = self.objects()
        # 1. решения по одному и тому же снимку (никто не видит чужих решений этого такта)
        commands = [
            self.controls[p.config.id].decide(p, [o for o in objects if o is not p], self.t)
            for p in self.planes
        ]
        # 2. RK4 по state_array каждого самолёта
        h = self.dt / self.substeps
        for p, u in zip(self.planes, commands):
            s = p.get_state_array()
            for _ in range(self.substeps):
                s = rk4_step(s, u, h)
            p.set_state_array(s)
        # 3. дрон — по своей модели движения
        new_r = self.drone.position_at(self.dt)
        if np.allclose(new_r, self.drone.config.aim):
            self.drone.v = np.zeros(3)
        self.drone.r = np.array(new_r, dtype=float)
        # 4. запись
        self.t += self.dt
        self.times.append(self.t)
        self.positions.append(np.array([p.r for p in self.planes]))
        self.velocities.append(np.array([p.v for p in self.planes]))
        self.drone_positions.append(self.drone.r.copy())

    def run(self, duration: float) -> None:
        for _ in range(int(round(duration / self.dt))):
            self.step()

    def trajectory(self) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
        """(times (K,), ids (N,), positions (K, N, 3), velocities (K, N, 3))."""
        return (
            np.array(self.times),
            [p.config.id for p in self.planes],
            np.array(self.positions),
            np.array(self.velocities),
        )


# ------------------------------------------------------------------ загрузка сценария


def load_system(
    scenario_path: str,
    flights_path: str,
    controls_factory=StraightControls,
    dt: float = 1.0,
    substeps: int = 4,
) -> System:
    with open(scenario_path, encoding="utf-8") as f:
        sc = json.load(f)
    aim = np.array([sc["target"]["x_m"], sc["target"]["y_m"], 0.0])
    h = sc["hostile_object"]
    drone_cfg = DroneConfig({
        "id": "HOSTILE",
        "r0": [h["x_m"], h["y_m"], h["alt_m"]],
        "v0": [h["vx_mps"], h["vy_mps"], h["vz_mps"]],
        "aim": aim,
    })
    drone = Drone(drone_cfg.r0.copy(), drone_cfg.v0.copy(), drone_cfg)
    target = DroneTarget(aim)

    planes = []
    with open(flights_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cfg = PlaneConfig({
                "id": row["flight_id"],
                "r0": [float(row["x_m"]), float(row["y_m"]), float(row["alt_m"])],
                "v0": [float(row["vx_mps"]), float(row["vy_mps"]), float(row["vz_mps"])],
            })
            planes.append(Plane(cfg.r0.copy(), cfg.v0.copy(), cfg))

    controls = {p.config.id: controls_factory() for p in planes}
    return System(planes, drone, target, controls, dt, substeps)