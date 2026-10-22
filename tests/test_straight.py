"""Сценарии «все летят прямо» (StraightControls) и проверки RK4 на нашей классовой структуре."""

from pathlib import Path

import numpy as np
import pytest

from flying_objects import Plane, PlaneConfig
from system import StraightControls, System, load_system, rk4_step

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenario.json"
FLIGHTS = ROOT / "flights.csv"


def make_plane(pid, r, v):
    cfg = PlaneConfig({"id": pid, "r0": r, "v0": v})
    return Plane(cfg.r0.copy(), cfg.v0.copy(), cfg)


@pytest.fixture(scope="module")
def straight_run():
    """Полный прогон 1200 с на данных задачи, все летят прямо."""
    system = load_system(str(SCENARIO), str(FLIGHTS), StraightControls)
    system.run(1200.0)
    return system


# ------------------------------------------------------------------ RK4


def test_rk4_straight_line_is_exact():
    p = make_plane("A", [0.0, 0.0, 5000.0], [200.0, 100.0, 0.0])
    s = p.get_state_array()
    for _ in range(40):
        s = rk4_step(s, np.zeros(3), 0.25)
    assert np.allclose(s[:3], [2000.0, 1000.0, 5000.0])


def test_rk4_constant_turn_is_circle():
    """ω = 3°/с при 230 м/с: окружность радиуса V/ω ≈ 4.4 км, полный круг за 120 с."""
    w = np.radians(3.0)
    p = make_plane("A", [0.0, 0.0, 5000.0], [230.0, 0.0, 0.0])
    s = p.get_state_array()
    u = np.array([0.0, w, 0.0])
    for _ in range(480):
        s = rk4_step(s, u, 0.25)
    assert 230.0 / w == pytest.approx(4392, rel=1e-3)
    assert np.allclose(s[:2], [0.0, 0.0], atol=1e-2)


def test_state_array_roundtrip():
    p = make_plane("A", [1.0, 2.0, 3000.0], [-221.66, -13.3, 1.0])
    s = p.get_state_array()
    assert s.shape == (6,)
    p.set_state_array(s)
    assert np.allclose(p.v, [-221.66, -13.3, 1.0])


# ------------------------------------------------------------------ система на маленьком примере


def test_two_planes_one_step():
    a = make_plane("A", [0.0, 0.0, 8000.0], [230.0, 0.0, 0.0])
    b = make_plane("B", [50e3, 0.0, 8000.0], [-230.0, 0.0, 0.0])
    sys_full = load_system(str(SCENARIO), str(FLIGHTS))  # берём только дрон и цель
    system = System(
        [a, b], sys_full.drone, sys_full.target,
        {"A": StraightControls(), "B": StraightControls()}, dt=1.0,
    )
    system.step()
    assert system.t == 1.0
    assert np.allclose(a.r, [230.0, 0.0, 8000.0])
    assert np.allclose(b.r, [50e3 - 230.0, 0.0, 8000.0])


# ------------------------------------------------------------------ полный прогон на данных задачи


def test_trajectory_shape(straight_run):
    times, ids, pos, vel = straight_run.trajectory()
    assert times.shape == (1201,)
    assert len(ids) == 40 and ids[0] == "F001"
    assert pos.shape == (1201, 40, 3) and vel.shape == (1201, 40, 3)


def test_straight_planes_reach_destination(straight_run):
    """Без манёвров каждый борт ровно в r0 + v0 · 1200 = config.destination."""
    for p in straight_run.planes:
        assert np.allclose(p.r, p.config.destination, atol=1e-6), p.config.id
        assert np.allclose(p.v, p.config.v0, atol=1e-9), p.config.id


def test_drone_stops_at_target(straight_run):
    drone_pos = np.array(straight_run.drone_positions)
    aim = straight_run.drone.config.aim
    assert np.allclose(drone_pos[-1], aim)  # к 1200 с он в земле в точке цели
    t_hit = np.linalg.norm(aim[:2] - drone_pos[0, :2]) / np.linalg.norm(straight_run.drone.config.v0[:2])
    assert t_hit == pytest.approx(833.33, abs=0.01)
    assert not np.allclose(drone_pos[833], aim)  # на 833 с ещё в воздухе
    assert np.allclose(drone_pos[834], aim)


def _max_potential(system, source, plane_id):
    """Максимум потенциала неподвижного источника в точке самолёта за весь прогон."""
    _, ids, pos, _ = system.trajectory()
    i = ids.index(plane_id)
    return max(source.get_attention_potential(r) for r in pos[:, i])


def test_straight_flight_is_unsafe_target(straight_run):
    """F016 проходит в 64 м от цели — потенциал цели на нём ненулевой (шаг «1»)."""
    assert _max_potential(straight_run, straight_run.target, "F016") > 0.5
    assert _max_potential(straight_run, straight_run.target, "F001") == 0


def test_straight_flight_is_unsafe_planes(straight_run):
    """F018 догоняет F034 на одной высоте 3600 м: минимум ≈2.5 км < 9260 м."""
    _, ids, pos, _ = straight_run.trajectory()
    i, j = ids.index("F018"), ids.index("F034")
    d = np.linalg.norm(pos[:, i, :2] - pos[:, j, :2], axis=1)
    k = int(np.argmin(d))
    assert d[k] == pytest.approx(2517, abs=5)
    other = make_plane("F034", pos[k, j], np.array([0.0, 229.04, 0.0]))
    assert other.get_attention_potential(pos[k, i]) > 0


def test_straight_flight_is_unsafe_drone(straight_run):
    """F034 (3600 м) проходит в ~0.7 км от дрона (4000 м)."""
    _, ids, pos, _ = straight_run.trajectory()
    drone_pos = np.array(straight_run.drone_positions)
    i = ids.index("F034")
    d = np.linalg.norm(pos[:, i, :2] - drone_pos[:, :2], axis=1)
    assert d.min() < 1000.0
    assert abs(pos[int(np.argmin(d)), i, 2] - drone_pos[int(np.argmin(d)), 2]) < 1500.0