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
    fig.colorbar(img, ax=ax, shrink=0.8, label="total danger potential")
    trails = [ax.plot([], [], lw=0.8, color="tab:blue", alpha=0.5)[0] for _ in ids]
    planes_sc = ax.scatter([], [], c=[], cmap="viridis", vmin=3000, vmax=12000, s=18, zorder=3)
    drone_sc = ax.scatter([], [], marker="x", s=60, color="crimson", zorder=4, label="drone")
    ax.scatter(*target.r[:2] * km, marker="D", s=40, color="purple", zorder=4, label="target")
    title = ax.set_title("")
    ax.set_xlabel("x, km (east)")
    ax.set_ylabel("y, km (north)")
    ax.set_xlim(xs[0] * km, xs[-1] * km)
    ax.set_ylim(ys[0] * km, ys[-1] * km)
    ax.set_aspect("equal")
    ax.legend(loc="lower left")
    mode = "each object at its own altitude" if z_view is None else f"slice at {z_view:.0f} m"

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
        title.set_text(f"t = {times[k]:.0f} s   ({mode})")
        return [img, planes_sc, drone_sc, title, *trails]

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    writer = FFMpegWriter(fps=fps) if str(path).endswith(".mp4") else PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=90)
    plt.close(fig)


def plot_tracks(system, path, title=""):
    """Статичный график (x, y): исходные прямые пунктиром, фактические трассы цветом по эшелону,
    зона цели, путь дрона и его зона в нескольких моментах."""
    from matplotlib.patches import Circle

    from rules import rules_v1 as R

    times, ids, pos, _ = system.trajectory()
    drone_pos = np.array(system.drone_positions)
    km = 1e-3
    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = plt.get_cmap("viridis")
    for n, p in enumerate(system.planes):
        c = p.config
        end = c.r0 + c.v0 * (times[-1] - times[0])
        ax.plot([c.r0[0] * km, end[0] * km], [c.r0[1] * km, end[1] * km],
                color="0.8", lw=0.8, ls="--", zorder=1)
        color = cmap((c.r0[2] - R.ALT_MIN) / (R.ALT_MAX - R.ALT_MIN))
        ax.plot(pos[:, n, 0] * km, pos[:, n, 1] * km, color=color, lw=1.4, zorder=2)
        ax.plot(pos[-1, n, 0] * km, pos[-1, n, 1] * km, "o", ms=3, color=color, zorder=3)
    ax.plot(drone_pos[:, 0] * km, drone_pos[:, 1] * km, color="crimson", lw=2, zorder=4, label="drone")
    for k in (0, len(times) // 4, len(times) // 2):
        ax.add_patch(Circle((drone_pos[k, 0] * km, drone_pos[k, 1] * km), R.DRONE_DANGER_RADIUS * km,
                            fill=False, ec="crimson", ls=":", zorder=4))
    t = system.target.r
    ax.add_patch(Circle((t[0] * km, t[1] * km), R.TARGET_DANGER_RADIUS * km,
                        color="purple", alpha=0.25, zorder=4))
    ax.plot(t[0] * km, t[1] * km, "D", color="purple", zorder=5, label="target")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(R.ALT_MIN, R.ALT_MAX))
    fig.colorbar(sm, ax=ax, shrink=0.8, label="initial flight level, m")
    ax.set_xlim(-EXTENT_M * km, EXTENT_M * km)
    ax.set_ylim(-EXTENT_M * km, EXTENT_M * km)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x, km (east)")
    ax.set_ylabel("y, km (north)")
    ax.set_title(title or f"tracks over {times[-1]:.0f} s (dashed: original straight routes)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
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