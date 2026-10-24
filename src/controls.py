"""Управление самолётом: перебор манёвров со свёрткой опасности по прогнозному пути.

Каждый такт, для одного самолёта (решает только по наблюдаемому: r, v остальных):
  1. кандидаты: курс = направление на исходную трассу («морковка» на carrot_s вперёд) + Δψ,
                высота = свой эшелон + Δz;
  2. для каждого кандидата прогноз своего пути на горизонт T: поворот с максимальной ω
     до нужного курса, дальше прямо; по высоте — не быстрее vz_max;
  3. стоимость:
       опасность  Σ_obj Σ_τ w(τ) · obj.get_attention_potential(путь(τ), τ) · Δτ
     + уход с исходной трассы в конце горизонта
     + потеря продвижения вдоль трассы (иначе развернуться назад «бесплатно»)
     + размер манёвра и смена решения (против дребезга);
  4. минимум стоимости; при равенстве — правый поворот;
  5. команда (a_tang, ω, a_z) к выбранному кандидату, в пределах ЛТХ из PlaneConfig.
"""

import numpy as np

from flying_objects import AttentionObject, Plane
from rules import rules_v1 as R
from system import Control, Controls

ALT_MIN = R.ALT_MIN
ALT_MAX = R.ALT_MAX


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def predict_path(r0, speed, psi, psi_goal, z_goal, omega_max, vz_max, taus):
    """Свои прогнозные пути (C, T, 3) для C кандидатов сразу: разворот с ±omega_max
    до psi_goal (C,), дальше прямо; высота идёт к z_goal (C,) не быстрее vz_max."""
    psi_goal = np.atleast_1d(psi_goal)[:, None]  # (C, 1)
    z_goal = np.atleast_1d(z_goal)[:, None]
    turn = wrap_pi(psi_goal - psi)
    omega = np.sign(turn) * omega_max
    t_turn = np.abs(turn) / omega_max if omega_max > 0 else np.zeros_like(turn)
    tc = np.minimum(taus[None, :], t_turn)  # (C, T)
    psi_end = psi + omega * tc
    # дуга: при omega -> 0 предел (sin(psi + omega tc) - sin psi) / omega = tc cos psi
    safe = np.where(np.abs(omega) > 1e-12, omega, 1.0)
    arc_x = np.where(np.abs(omega) > 1e-12, (np.sin(psi_end) - np.sin(psi)) / safe, tc * np.cos(psi))
    arc_y = np.where(np.abs(omega) > 1e-12, -(np.cos(psi_end) - np.cos(psi)) / safe, tc * np.sin(psi))
    rest = taus[None, :] - tc
    dz = np.clip(z_goal - r0[2], -vz_max * taus, vz_max * taus)
    return np.stack([r0[0] + speed * (arc_x + np.cos(psi_end) * rest),
                     r0[1] + speed * (arc_y + np.sin(psi_end) * rest),
                     r0[2] + dz], axis=-1)


class SearchControls(Controls):
    def __init__(
        self,
        heading_options_deg=R.HEADING_OPTIONS_DEG,  # минус — вправо
        altitude_options_m=R.ALTITUDE_OPTIONS_M,
        carrot_s=R.CARROT_S,
        danger_weight=R.CONTROL_DANGER_WEIGHT,  # опасность важнее маршрута
        route_weight=R.CONTROL_ROUTE_WEIGHT,
        lag_weight=R.CONTROL_LAG_WEIGHT,
        maneuver_penalty=R.CONTROL_MANEUVER_PENALTY,
        switch_penalty=R.CONTROL_SWITCH_PENALTY,
        vertical_rule_penalty=R.CONTROL_VERTICAL_RULE_PENALTY,
        dt=R.SIM_DT,
    ):
        self.dpsi = np.radians(np.asarray(heading_options_deg, dtype=float))
        self.dz = np.asarray(altitude_options_m, dtype=float)
        self.carrot_s = carrot_s
        self.danger_weight = danger_weight
        self.route_weight = route_weight
        self.lag_weight = lag_weight
        self.maneuver_penalty = maneuver_penalty
        self.switch_penalty = switch_penalty
        self.vertical_rule_penalty = vertical_rule_penalty
        self.dt = dt
        self.prev = None  # индекс прошлого выбора (своё решение — помнить его можно)
        self.last_choice = (0.0, None)  # (Δψ, z_goal) — для логов и тестов

    # ---------------------------------------------------------------- выбор манёвра

    def _route_heading(self, plane: Plane):
        c = plane.config
        v0 = c.v0[:2]
        s_along = float(np.dot(plane.r[:2] - c.r0[:2], v0) / np.dot(v0, v0))  # «время по графику»
        carrot = c.r0[:2] + v0 * (s_along + self.carrot_s)
        return float(np.arctan2(carrot[1] - plane.r[1], carrot[0] - plane.r[0])), s_along

    def choose(self, plane: Plane, others: list[AttentionObject]):
        c = plane.config
        taus = np.linspace(0.0, c.prediction_horizon_s, c.prediction_steps)
        w = 1.0 / (1.0 + taus / c.weight_decay_s)
        dtau = np.gradient(taus)
        speed = float(np.hypot(plane.v[0], plane.v[1]))
        psi = float(np.arctan2(plane.v[1], plane.v[0]))
        psi_route, _ = self._route_heading(plane)
        omega_max = min(c.turn_rate_max, c.a_norm_max / max(speed, 1.0))

        dpsi_all = np.repeat(self.dpsi, len(self.dz))  # все пары (Δψ, Δz)
        dz_all = np.tile(self.dz, len(self.dpsi))
        z_goal_all = np.clip(c.r0[2] + dz_all, ALT_MIN, ALT_MAX)
        labels = list(zip(dpsi_all, dz_all, z_goal_all))
        paths = predict_path(plane.r, speed, psi, psi_route + dpsi_all, z_goal_all,
                             omega_max, c.vz_max, taus)  # (C, T, 3)
        n_c, n_t = paths.shape[:2]

        # опасность: один векторный вызов на объект для всех кандидатов и всех τ
        flat = paths.reshape(-1, 3)
        tt = np.tile(taus, n_c)
        danger = np.zeros(n_c)
        for obj in others:
            u = obj.get_attention_potential(flat, tt).reshape(n_c, n_t)
            danger += (u * w * dtau).sum(axis=1)

        # маршрут: уход с трассы и потеря продвижения к концу горизонта, в единицах 10 км
        v0 = c.v0[:2] / np.linalg.norm(c.v0[:2])
        off = paths[:, -1, :2] - c.r0[:2]
        cross = np.abs(off[:, 0] * -v0[1] + off[:, 1] * v0[0]) / 10e3
        progress = (paths[:, -1, :2] - plane.r[:2]) @ v0
        lag = np.maximum(np.linalg.norm(c.v0[:2]) * taus[-1] - progress, 0.0) / 10e3

        # размер манёвра: смена высоты «дороже» поворота, чтобы встречные расходились вправо,
        # а не лезли одновременно на один и тот же эшелон
        size = np.array([abs(np.degrees(d)) / 30.0 + abs(z) / 300.0 for d, z, _ in labels])
        cost = (self.danger_weight * danger + self.route_weight * cross**2 + self.lag_weight * lag**2
                + self.maneuver_penalty * size)
        # Вертикальное соглашение (как правило полукруга ИКАО): курс 0–179° уходит вверх,
        # 180–359° — вниз. Иначе встречные по одному снимку синхронно лезут на один эшелон.
        course = (90.0 - np.degrees(psi)) % 360.0
        up = 1.0 if course < 180.0 else -1.0
        cost = cost + self.vertical_rule_penalty * (np.sign(dz_all) == -up)
        if self.prev is not None and danger.max() > 0:
            cost = cost + self.switch_penalty * (np.arange(n_c) != self.prev)

        # минимум; при равенстве — правее (меньший Δψ: вправо = отрицательный)
        order = np.lexsort((np.array([d for d, _, _ in labels]), np.round(cost, 9)))
        best = int(order[0])
        self.prev = best
        dpsi, _, z_goal = labels[best]
        self.last_choice = (float(np.degrees(dpsi)), z_goal)
        self.last_costs = (labels, danger, cost)  # для отладки
        return psi_route + dpsi, z_goal

    # ---------------------------------------------------------------- команда

    def decide(self, plane: Plane, others: list[AttentionObject], t: float) -> Control:
        c = plane.config
        psi_goal, z_goal = self.choose(plane, others)
        speed = float(np.hypot(plane.v[0], plane.v[1]))
        psi = float(np.arctan2(plane.v[1], plane.v[0]))

        # курс
        omega_max = min(c.turn_rate_max, c.a_norm_max / max(speed, 1.0))
        omega = float(np.clip(0.4 * wrap_pi(psi_goal - psi), -omega_max, omega_max))

        # скорость: нагнать отставание от исходного графика к моменту horizon_s
        _, s_along = self._route_heading(plane)
        v_req = np.linalg.norm(c.v0[:2]) * (c.horizon_s - s_along) / max(c.horizon_s - t, 60.0)
        v_goal = float(np.clip(v_req, c.v_min, c.v_max))
        a_lo = max(-c.a_tang_max, (c.v_min - speed) / self.dt)
        a_hi = min(c.a_tang_max, (c.v_max - speed) / self.dt)
        a_tang = float(np.clip(0.2 * (v_goal - speed), min(a_lo, 0.0), max(a_hi, 0.0)))

        # высота: захват за конечное время без перелёта (P-закон k·dz недобирает метры,
        # а соседний эшелон ровно в 300 м — это уже нарушение)
        d = z_goal - plane.r[2] - 0.5 * plane.v[2]
        vz_goal = np.sign(d) * min(np.sqrt(2.0 * 0.5 * c.a_vert_max * abs(d)), abs(d) / 2.0)
        vz_goal = float(np.clip(vz_goal, -c.vz_max, c.vz_max))
        a_z = float(np.clip((vz_goal - plane.v[2]) / self.dt, -c.a_vert_max, c.a_vert_max))

        return np.array([a_tang, omega, a_z])