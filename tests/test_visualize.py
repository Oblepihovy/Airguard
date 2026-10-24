import copy

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from flying_objects import Plane, PlaneConfig  # noqa: E402
from system import StraightControls, System  # noqa: E402
from visualize import animate, field_at  # noqa: E402


def test_field_is_zero_far_and_positive_near(straight_run):
    xs = ys = np.array([-150e3, 0.0, 150e3])
    f = field_at([straight_run.target], xs, ys)
    assert f[0, 0] < 1e-6  # далеко от цели
    near = field_at([straight_run.target], np.array([straight_run.target.r[0]]),
                    np.array([straight_run.target.r[1]]))
    assert near[0, 0] > 0.5


def test_animation_smoke(tmp_path, straight_run):
    cfg = PlaneConfig({"id": "A", "r0": [0, 0, 8000], "v0": [230, 0, 0]})
    drone = copy.deepcopy(straight_run.drone)  # общий прогон из conftest не трогаем
    system = System([Plane(cfg.r0.copy(), cfg.v0.copy(), cfg)], drone, straight_run.target,
                    {"A": StraightControls()})
    system.run(20.0)
    out = tmp_path / "a.gif"
    animate(system, out, step_s=10.0, cell_m=10_000.0)
    assert out.stat().st_size > 0


def test_plot_tracks_smoke(tmp_path, straight_run):
    from visualize import plot_tracks

    out = tmp_path / "tracks.png"
    plot_tracks(straight_run, out)
    assert out.stat().st_size > 0