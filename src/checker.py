"""Независимая проверка траектории по нормативам из rules_v1.

Checker не пользуется потенциалами и controls: он смотрит только на записанные
положения и сравнивает их с нормативами. Между отсчётами движение считается
линейным, поэтому для пар самолётов минимум расстояния на отрезке берётся точно
(сближение не «проскочит» между тактами).

Проверки:
  separation    — самолёт-самолёт: горизонталь < PLANE_SEPARATION_H и |dz| < PLANE_SEPARATION_V
  drone_zone    — самолёт-дрон:   горизонталь < DRONE_DANGER_RADIUS и |dz| < DRONE_DANGER_HALF_HEIGHT
  target_zone   — колонна над целью: горизонталь < TARGET_DANGER_RADIUS
  speed / turn_rate / climb_rate / altitude — ЛТХ по конечным разностям (лимиты из PlaneConfig)
"""

from dataclasses import dataclass, field

import numpy as np

from rules import rules_v1 as R

V_TOL = R.PLANE_SEPARATION_V_TOL  # соседние эшелоны ровно в 300 м — округление не нарушение
ALT_MIN = R.ALT_MIN
ALT_MAX = R.ALT_MAX
KIN_TOL = 0.01  # 1% на конечные разности


@dataclass
class Violation:
    rule: str
    flight: str
    other: str
    t: float
    value: float
    limit: float


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)  # худший случай по (правило, пара)
    counts: dict[str, int] = field(default_factory=dict)  # сколько отсчётов с нарушением
    min_distance: dict[str, float] = field(default_factory=dict)  # минимум по горизонтали, м

    @property
    def passed(self) -> bool:
        return not self.counts

    def summary(self) -> str:
        lines = ["PASS" if self.passed else "FAIL"]
        for rule, n in sorted(self.counts.items()):
            lines.append(f"  {rule:<12} {n:>6} отсчётов с нарушением")
        for kind, d in self.min_distance.items():
            lines.append(f"  мин. расстояние до {kind:<7} {d / 1000:7.2f} км")
        return "\n".join(lines)


def _segment_min(a, b):
    """min_{s∈[0,1]} |a + s(b - a)| по последней оси (линейное движение между отсчётами)."""
    d = b - a
    dd = np.sum(d * d, axis=-1)
    s = np.where(dd > 1e-12, -np.sum(a * d, axis=-1) / np.maximum(dd, 1e-12), 0.0)
    s = np.clip(s, 0.0, 1.0)
    return np.linalg.norm(a + d * s[..., None], axis=-1)


def _add(report, rule, mask, flights, others, times, values, limit, worst_is_min=True):
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return
    report.counts[rule] = report.counts.get(rule, 0) + len(idx)
    worst = {}
    for k, n in idx:
        key = (str(flights[n]), str(others[n]))
        v = values[k, n]
        if key not in worst or (v < worst[key][1] if worst_is_min else v > worst[key][1]):
            worst[key] = (k, v)
    for (f, o), (k, v) in worst.items():
        report.violations.append(Violation(rule, f, o, float(times[k]), float(v), limit))


def check(times, ids, positions, drone_positions, target, configs=None) -> Report:
    """
    times (K,), ids (N,), positions (K, N, 3), drone_positions (K, 3), target (3,).
    configs: {id: PlaneConfig} — для проверки ЛТХ; None — ЛТХ не проверяются.
    """
    times = np.asarray(times, dtype=float)
    pos = np.asarray(positions, dtype=float)
    drone = np.asarray(drone_positions, dtype=float)
    ids = np.asarray(ids)
    report = Report()
    n = pos.shape[1]

    # --- самолёт-самолёт, точный минимум на каждом отрезке между отсчётами
    ii, jj = np.triu_indices(n, k=1)
    rel = pos[:, ii, :] - pos[:, jj, :]  # (K, P, 3)
    h_min = _segment_min(rel[:-1, :, :2], rel[1:, :, :2])  # (K-1, P)
    dz0, dz1 = rel[:-1, :, 2], rel[1:, :, 2]
    dz_min = np.where(np.sign(dz0) != np.sign(dz1), 0.0, np.minimum(abs(dz0), abs(dz1)))
    same_level = dz_min < R.PLANE_SEPARATION_V - V_TOL
    _add(report, "separation", (h_min < R.PLANE_SEPARATION_H) & same_level,
         ids[ii], ids[jj], times[1:], h_min, R.PLANE_SEPARATION_H)
    report.min_distance["самолёта"] = float(h_min[same_level].min()) if same_level.any() else np.inf

    # --- самолёт-дрон
    d_h = np.linalg.norm(pos[:, :, :2] - drone[:, None, :2], axis=-1)  # (K, N)
    near_level = np.abs(pos[:, :, 2] - drone[:, None, 2]) < R.DRONE_DANGER_HALF_HEIGHT
    hostile = np.full(n, "HOSTILE")
    _add(report, "drone_zone", (d_h < R.DRONE_DANGER_RADIUS) & near_level,
         ids, hostile, times, d_h, R.DRONE_DANGER_RADIUS)
    report.min_distance["дрона"] = float(d_h[near_level].min()) if near_level.any() else np.inf

    # --- колонна над целью
    d_t = np.linalg.norm(pos[:, :, :2] - np.asarray(target)[:2], axis=-1)
    _add(report, "target_zone", d_t < R.TARGET_DANGER_RADIUS,
         ids, np.full(n, "TARGET"), times, d_t, R.TARGET_DANGER_RADIUS)
    report.min_distance["цели"] = float(d_t.min())

    # --- ЛТХ по конечным разностям
    _add(report, "altitude", (pos[:, :, 2] < ALT_MIN - 0.5) | (pos[:, :, 2] > ALT_MAX + 0.5),
         ids, ids, times, pos[:, :, 2], ALT_MIN)
    if configs is not None:
        dt = np.diff(times)[:, None]
        chord = np.diff(pos, axis=0) / dt[..., None]
        v_h = np.hypot(chord[..., 0], chord[..., 1])
        v_min = np.array([configs[i].v_min for i in ids])
        v_max = np.array([configs[i].v_max for i in ids])
        vz_max = np.array([configs[i].vz_max for i in ids])
        _add(report, "speed", (v_h < v_min * (1 - KIN_TOL)) | (v_h > v_max * (1 + KIN_TOL)),
             ids, ids, times[1:], v_h, float(v_max.max()), worst_is_min=False)
        _add(report, "climb_rate", np.abs(chord[..., 2]) > vz_max * (1 + KIN_TOL),
             ids, ids, times[1:], np.abs(chord[..., 2]), float(vz_max.max()), worst_is_min=False)
        heading = np.arctan2(chord[..., 1], chord[..., 0])
        turn = np.abs((np.diff(heading, axis=0) + np.pi) % (2 * np.pi) - np.pi) / dt[1:]
        v_mid = 0.5 * (v_h[1:] + v_h[:-1])
        w_max = np.array([[min(configs[i].turn_rate_max, configs[i].a_norm_max / max(v, 1.0))
                           for i, v in zip(ids, row)] for row in v_mid])
        _add(report, "turn_rate", turn > w_max * (1 + KIN_TOL) + 1e-9,
             ids, ids, times[2:], np.degrees(turn), float(np.degrees(w_max.max())), worst_is_min=False)

    report.violations.sort(key=lambda v: (v.rule, v.t))
    return report


def check_system(system) -> Report:
    """Удобная обёртка: проверить System после run()."""
    times, ids, pos, _ = system.trajectory()
    configs = {p.config.id: p.config for p in system.planes}
    return check(times, ids, pos, system.drone_positions, system.target.r, configs)