"""PinDynamicsSolver —— pinocchio 黑盒动力学（默认实现，Ch8 占位）。

内核即 pin 双件套 ``rnea``/``crba``（C/G 由 ``rnea`` 特殊调用导出）；本库不做
物理仿真（无正动力学），整机预演由 ``joyarm_ros2_ws`` 的 rviz2 承担。
"""
from __future__ import annotations

import numpy as np

from .dynamics_solver import DynamicsSolver

__all__ = ["PinDynamicsSolver"]


class PinDynamicsSolver(DynamicsSolver):
    """pinocchio（rnea/crba）黑盒动力学。"""

    def idyn(self, arm, q, dq, ddq, f_ext=None) -> np.ndarray:
        raise NotImplementedError("PinDynamicsSolver.idyn 待 Ch8 实现")

    def mass_matrix(self, arm, q) -> np.ndarray:
        raise NotImplementedError("PinDynamicsSolver.mass_matrix 待 Ch8 实现")

    def coriolis(self, arm, q, dq) -> np.ndarray:
        raise NotImplementedError("PinDynamicsSolver.coriolis 待 Ch8 实现")

    def gravity(self, arm, q) -> np.ndarray:
        raise NotImplementedError("PinDynamicsSolver.gravity 待 Ch8 实现")
