"""get_attention_potential с вектором точек и вектором tau = те же числа, что поточечно."""

import numpy as np
import pytest

from flying_objects import Drone, DroneConfig, DroneTarget, Plane, PlaneConfig

TAUS = np.linspace(0.0, 120.0, 13)


def make_objects():
    pc = PlaneConfig({"id": "B", "r0": [20e3, 0, 8000], "v0": [-230, 0, 0]})
    dc = DroneConfig({"id": "H", "r0": [-10e3, 5e3, 4000], "v0": [96, -72, 0], "aim": [20e3, -10e3, 0]})
    return [
        Plane(pc.r0.copy(), pc.v0.copy(), pc),
        Drone(dc.r0.copy(), dc.v0.copy(), dc),
        DroneTarget(np.array([20e3, -10e3, 0.0])),
    ]


@pytest.mark.parametrize("obj", make_objects(), ids=["plane", "drone", "target"])
def test_vector_call_matches_pointwise(obj):
    rng = np.random.default_rng(0)
    pts = rng.uniform([-30e3, -30e3, 3000], [30e3, 30e3, 9000], size=(len(TAUS), 3))
    vec = obj.get_attention_potential(pts, TAUS)
    assert vec.shape == (len(TAUS),)
    one_by_one = [obj.get_attention_potential(p, t) for p, t in zip(pts, TAUS)]
    assert np.allclose(vec, one_by_one)


@pytest.mark.parametrize("obj", make_objects(), ids=["plane", "drone", "target"])
def test_scalar_call_still_scalar(obj):
    u = obj.get_attention_potential(np.array([0.0, 0.0, 8000.0]), 10.0)
    assert np.ndim(u) == 0


def test_same_point_along_tau():
    """Одна точка, вектор tau: потенциал встречного самолёта меняется по мере сближения."""
    plane = make_objects()[0]  # летит на запад с x = 20 км
    u = plane.get_attention_potential(np.array([0.0, 0.0, 8000.0]), TAUS)
    assert u.shape == (len(TAUS),)
    assert u[0] < 1e-6  # сейчас он в 20 км — вне зоны
    assert abs(TAUS[np.argmax(u)] - 20e3 / 230) <= 10.0  # пик около tau ≈ 87 с, когда он над точкой
    assert u.max() > 0.9


def test_drone_position_at_vector_stops_at_aim():
    drone = make_objects()[1]
    p = drone.position_at(np.array([0.0, 100.0, 10_000.0]))
    assert p.shape == (3, 3)
    assert np.allclose(p[0], drone.r)
    assert np.allclose(p[1], drone.r + drone.v * 100.0)
    assert np.allclose(p[2], drone.config.aim)


def test_plane_config_defaults_come_from_rules():
    from rules import rules_v1 as R

    c = PlaneConfig({"id": "X", "r0": [0, 0, 5000], "v0": [230, 0, 0]})
    assert (c.v_min, c.v_max, c.vz_max) == (R.PLANE_V_MIN, R.PLANE_V_MAX, R.PLANE_VZ_MAX)
    assert c.turn_rate_max == pytest.approx(np.radians(R.PLANE_TURN_RATE_MAX_DPS))
    assert c.horizon_s == R.DESTINATION_HORIZON_S
    assert PlaneConfig({"id": "X", "r0": [0, 0, 5000], "v0": [230, 0, 0], "v_max": 250.0}).v_max == 250.0