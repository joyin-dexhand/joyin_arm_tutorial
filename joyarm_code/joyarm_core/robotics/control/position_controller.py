"""PositionController —— 关节位置伺服（默认控制律，Ch6 占位）。

PD 伺服 ``τ = K_p(q_d − q) + K_d(q̇_d − q̇)``（或直接位置下发，视驱动器模式）；
同族备选（velocity/torque/computed_torque/impedance/admittance）各章落地后注册。
"""
from __future__ import annotations

import numpy as np

from .controller import Controller

__all__ = ["PositionController"]


class PositionController(Controller):
    """关节位置伺服（默认控制律）。"""

    def compute(self, arm, target: np.ndarray, state=None, **kw) -> np.ndarray:
        raise NotImplementedError("PositionController.compute 待 Ch6 实现")
