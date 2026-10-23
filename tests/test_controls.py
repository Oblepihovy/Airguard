"""SearchControls: расхождение пар, обход цели, полный прогон на данных задачи (шаг «2»)."""

from pathlib import Path

import numpy as np
import pytest

from checker import check_system
from controls import SearchControls, predict_path
from flying_objects import Drone, DroneConfig, DroneTarget, Plane, PlaneConfig
from system import System, load_system, rk4_step

ROOT = Path(__file__).resolve().parents[1]


def make_plane(pid, r, v):
    cfg = PlaneConfig({"id": pid, "r0": r, "v0": v})
    return Plane(cfg.r0.copy(), cfg.v0.copy(), cfg)


def far_drone():
    dc = DroneConfig({"id": "H", "r0": [-190e3, 190e3, 4000], "v0": [0, -1, 0], "aim": [-190e3, -190e3, 0]})
    return Drone(dc.r0.copy(), dc.v0.copy(), dc)


def small_system(planes, target_xy=(190e3, 190e3)):
    return System(planes, far_drone(), DroneTarget(np.array([*target_xy, 0.0])),
                  {p.config.id: SearchControls() for p in planes})


def test_predict_path_matches_rk4():
    taus = np.linspace(0, 120, 13)
    w = np.radians(3.0)
    path = predict_path(np.array([0, 0, 5000.0]), 230.0, 0.3, np.array([0.3 + np.radians(60)]),
                        np.array([5000.0]), w, 15.0, taus)[0]
    s = np.array([0, 0, 5000, 230, 0.3, 0.0])
    t_turn = np.radians(60) / w
    for k in range(1, len(taus)):
        for j in range(40):
            u = np.array([0.0, w if taus[k - 1] + j * 0.25 < t_turn else 0.0, 0.0])
            s = rk4_step(s, u, 0.25)
        assert np.allclose(path[k, :2], s[:2], atol=1e-3)


@pytest.mark.parametrize("axis", ["east-west", "north-south"])
def test_head_on_resolves_without_mirroring(axis):
    """Встречные на одном эшелоне расходятся, а не повторяют манёвр друг друга."""
    if axis == "east-west":
        a = make_plane("A", [-60e3, 0, 8000], [230, 0, 0])  # курс 90° -> вверх
        b = make_plane("B", [60e3, 0, 8000], [-230, 0, 0])  # курс 270° -> вниз
    else:
        a = make_plane("A", [0, -60e3, 8000], [0, 230, 0])  # курс 0° -> вверх
        b = make_plane("B", [0, 60e3, 8000], [0, -230, 0])  # курс 180° -> вниз
    system = small_system([a, b])
    system.run(400.0)
    report = check_system(system)
    assert "separation" not in report.counts, report.summary()
    _, _, pos, _ = system.trajectory()
    assert pos[:, 0, 2].max() > 8200 and pos[:, 1, 2].min() < 7800  # разошлись по высоте в разные стороны


def test_plane_avoids_target_column():
    a = make_plane("A", [-60e3, 0, 9000], [230, 0, 0])
    system = small_system([a], target_xy=(0.0, 0.0))
    system.run(500.0)
    report = check_system(system)
    assert "target_zone" not in report.counts, report.summary()


def test_returns_to_route_after_avoidance():
    a = make_plane("A", [-60e3, 0, 9000], [230, 0, 0])
    system = small_system([a], target_xy=(0.0, 0.0))
    system.run(800.0)
    assert abs(a.r[1]) < 1500.0  # снова на исходной линии y = 0
    assert abs(a.r[2] - 9000.0) < 1.0  # и на своём эшелоне


def test_full_scenario_passes_checker():
    """Шаг «2»: на данных задачи с SearchControls checker не находит нарушений."""
    system = load_system(str(ROOT / "scenario.json"), str(ROOT / "flights.csv"), SearchControls)
    system.run(1200.0)
    report = check_system(system)
    assert report.passed, report.summary()
    _, _, pos, _ = system.trajectory()
    delays = [
        1200.0 - np.dot(pos[-1, i, :2] - p.config.r0[:2], p.config.v0[:2]) / np.dot(p.config.v0[:2], p.config.v0[:2])
        for i, p in enumerate(system.planes)
    ]
    assert max(delays) < 180.0  # каждый борт отстаёт от графика меньше чем на 3 минуты