"""Анимация поля опасности во времени по записанной траектории System.

Фон — суммарный потенциал всех объектов на сетке (x, y) в момент кадра (tau = 0).
Потенциалы считаются теми же get_attention_potential из flying_objects.

Высота, на которой смотрим поле:
  z_view=None — «следы»: каждый объект оценивается на СВОЕЙ высоте, видно горизонтальные
                зоны всех объектов сразу (цель — колонна, на любой высоте);
  z_view=4000 — срез на заданной высоте: видно только то, что опасно на этом эшелоне.

    python src/visualize.py out/straight.gif            # прогон «все прямо» + анимация
"""

import copy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

EXTENT_M = 200e3  # сектор ±200 км


def field_at(objects, xs, ys, z_view=None):
    """Суммарный потенциал на сетке xs × ys. objects — AttentionObject с актуальным r."""
    gx, gy = np.meshgrid(xs, ys)  # (len(ys), len(xs))
    pts = np.stack([gx, gy, np.zeros_like(gx)], axis=-1).reshape(-1, 3)
    f = np.zeros(len(pts))
    for obj in objects:
        pts[:, 2] = obj.r[2] if z_view is None else z_view
        f += obj.get_attention_potential(pts)  # вся сетка одним вызовом
    return f.reshape(gx.shape)


def animate(system, path, step_s=10.0, cell_m=2000.0, z_view=None, fps=10, trail_s=120.0):
    times, ids, pos, vel = system.trajectory()
    drone_pos = np.array(system.drone_positions)
    planes = copy.deepcopy(system.planes)  # копии: исходную систему не трогаем
    drone = copy.deepcopy(system.drone)
    target = system.target
    frames = np.flatnonzero(np.isclose(np.mod(times, step_s), 0.0))

    xs = np.arange(-EXTENT_M, EXTENT_M + cell_m, cell_m)
    ys = xs.copy()
    km = 1e-3

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(np.zeros((len(ys), len(xs))), origin="lower", cmap="magma_r",
                    extent=(xs[0] * km, xs[-1] * km, ys[0] * km, ys[-1] * km), vmin=0, vmax=1.5)
    fig.colorbar(img, ax=ax, shrink=0.8, label="суммарный потенциал опасности")
    trails = [ax.plot([], [], lw=0.8, color="tab:blue", alpha=0.5)[0] for _ in ids]
    planes_sc = ax.scatter([], [], c=[], cmap="viridis", vmin=3000, vmax=12000, s=18, zorder=3)
    drone_sc = ax.scatter([], [], marker="x", s=60, color="crimson", zorder=4, label="дрон")
    ax.scatter(*target.r[:2] * km, marker="D", s=40, color="purple", zorder=4, label="цель")
    title = ax.set_title("")
    ax.set_xlabel("x, км (восток)")
    ax.set_ylabel("y, км (север)")
    ax.set_xlim(xs[0] * km, xs[-1] * km)
    ax.set_ylim(ys[0] * km, ys[-1] * km)
    ax.set_aspect("equal")
    ax.legend(loc="lower left")
    mode = "следы на своих высотах" if z_view is None else f"срез на {z_view:.0f} м"

    def draw(k):
        for p, r, v in zip(planes, pos[k], vel[k]):
            p.r, p.v = r.copy(), v.copy()
        drone.r = drone_pos[k].copy()
        img.set_data(field_at([*planes, drone, target], xs, ys, z_view))
        k0 = max(0, k - int(trail_s / (times[1] - times[0])))
        for n, line in enumerate(trails):
            line.set_data(pos[k0:k + 1, n, 0] * km, pos[k0:k + 1, n, 1] * km)
        planes_sc.set_offsets(pos[k, :, :2] * km)
        planes_sc.set_array(pos[k, :, 2])
        drone_sc.set_offsets([drone_pos[k, :2] * km])
        title.set_text(f"t = {times[k]:.0f} с   ({mode})")
        return [img, planes_sc, drone_sc, title, *trails]

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    writer = FFMpegWriter(fps=fps) if str(path).endswith(".mp4") else PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path

    from system import load_system

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/straight.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    s = load_system(str(root / "scenario.json"), str(root / "flights.csv"))
    s.run(1200.0)
    animate(s, out)
    print(f"-> {out}")