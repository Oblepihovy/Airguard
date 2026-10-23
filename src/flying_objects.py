from abc import ABC, abstractmethod

import numpy as np
import numpy.typing  # без этого np.typing может не подгрузиться

from rules.rules_v1 import *
import math_functions as m


Vec3 = np.typing.NDArray[np.float64]

class AttentionObject(ABC):
    """
    Abstract class to represent an object on the map.
    The object has some "Attention Potential", which is R^3 -> R "function of danger" in the point

    For example, target of a drone (which isn't moving),
    has monotonically decreasing function of F(|r_plane - r_target|)



    """
    def __init__(self, r: Vec3):
        self.r = r

    @abstractmethod
    def get_attention_potential(self, r_object: Vec3, tau=0.0) -> np.ndarray:
        """Returns the attention potential of the object.
        @:param r_object: Another object position tensor (which wants to be awared of (self) object).
                          (3,) or (K, 3).
        @:param tau: time ahead of now, s (self is taken at its predicted position at tau).
                     (1, ) (K,) respectively
        @:return: potential with shape (K,)"""
        pass

    def get_attention_gradient(self, r_object: Vec3, tau: float = 0.0, h: float = 1.0) -> Vec3:
        """Returns the gradient of the attention potential of the object.
        """
        grad = np.zeros(3)
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            up = self.get_attention_potential(r_object + e, tau)
            dn = self.get_attention_potential(r_object - e, tau)
            grad[k] = (up - dn) / (2.0 * h)
        return grad

    @staticmethod
    def _bump(d: float, radius: float) -> float:
        return m.softplus(0, 1 - (d / radius) ** 2)


class FlyingObjectConfig:
    def __init__(self, config_dict: dict):
        self.id: str = config_dict["id"]
        self.r0: Vec3 = np.asarray(config_dict["r0"], dtype=float)
        self.v0: Vec3 = np.asarray(config_dict["v0"], dtype=float)


class DroneConfig(FlyingObjectConfig):
    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.aim: Vec3 = np.asarray(config_dict["aim"], dtype=float)  # точка цели, там дрон останавливается


class FlyingObject(AttentionObject, ABC):
    def __init__(self, r: Vec3, v: Vec3, config: FlyingObjectConfig):
        super().__init__(r)
        self.v = v
        self.config = config

    def get_state_array(self) -> np.typing.NDArray[np.float64]:
        """
        @brief: Returns the state array of the object for RK4. (x, y, z, v_mod(xy), psi, v_z)
        :return:
        """
        v_xy = np.hypot(self.v[0], self.v[1])
        psi = np.arctan2(self.v[1], self.v[0])  # курс от оси x (восток) против часовой
        return np.array([self.r[0], self.r[1], self.r[2], v_xy, psi, self.v[2]])

    def set_state_array(self, s: np.typing.NDArray[np.float64]) -> None:
        self.r = np.array(s[:3], dtype=float)
        self.v = np.array([s[3] * np.cos(s[4]), s[3] * np.sin(s[4]), s[5]])

    def position_at(self, tau) -> Vec3:
        return self.r + self.v * np.asarray(tau, dtype=float)[..., None]


class Drone(FlyingObject):
    def position_at(self, tau) -> Vec3:
        # flies to the target and stops in there (h=0)
        tau = np.asarray(tau, dtype=float)[..., None]
        speed = np.linalg.norm(self.v[:2])
        t_hit = np.linalg.norm(self.config.aim[:2] - self.r[:2]) / speed if speed > 0 else 0.0
        return np.where(tau >= t_hit, self.config.aim, self.r + self.v * tau)

    def get_attention_potential(self, r_object, tau=0.0):
        p = self.position_at(tau)
        d_h = np.linalg.norm(r_object[..., :2] - p[..., :2], axis=-1)
        dz = r_object[..., 2] - p[..., 2]
        return (self._bump(d_h, DRONE_DANGER_RADIUS * DANGER_MARGIN_FACTOR)
                * self._bump(dz, DRONE_DANGER_HALF_HEIGHT))

class DroneTarget(AttentionObject):
    def get_attention_potential(self, r_object, tau=0.0):
        d_h = np.linalg.norm(r_object[..., :2] - self.r[:2], axis=-1)
        u = self._bump(d_h, TARGET_DANGER_RADIUS * DANGER_MARGIN_FACTOR)
        return u * np.ones(np.shape(tau))


class PlaneConfig(FlyingObjectConfig):
    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.horizon_s: float = config_dict.get("horizon_s", 1200.0)  # destination = r0 + v0 * horizon_s
        self.v_min: float = config_dict.get("v_min", 180.0)  # м/с
        self.v_max: float = config_dict.get("v_max", 260.0)
        self.a_tang_max: float = config_dict.get("a_tang_max", 1.0)  # м/с²
        self.a_norm_max: float = config_dict.get("a_norm_max", 12.5)  # м/с²
        self.turn_rate_max: float = np.radians(config_dict.get("turn_rate_max_dps", 3.0))  # рад/с
        self.vz_max: float = config_dict.get("vz_max", 15.0)  # м/с
        self.a_vert_max: float = config_dict.get("a_vert_max", 2.0)  # м/с²
        # для controls: горизонт прогноза и свёртка
        self.prediction_horizon_s: float = config_dict.get("prediction_horizon_s", 120.0)
        self.prediction_steps: int = config_dict.get("prediction_steps", 13)
        self.weight_decay_s: float = config_dict.get("weight_decay_s", 40.0)  # w(tau) = 1 / (1 + tau / decay)

    @property
    def destination(self) -> Vec3:
        return self.r0 + self.v0 * self.horizon_s

class Plane(FlyingObject):

    def __init__(self, r, v, plane_config: PlaneConfig):
        super().__init__(r, v, plane_config)


    def get_attention_potential(self, r_object, tau=0.0):
        p = self.position_at(tau)
        d_h = np.linalg.norm(r_object[..., :2] - p[..., :2], axis=-1)
        dz = r_object[..., 2] - p[..., 2]
        return (self._bump(d_h, PLANE_SEPARATION_H * DANGER_MARGIN_FACTOR)
                * self._bump(dz, PLANE_SEPARATION_V))