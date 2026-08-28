"""LagrangianDynamicsSolver —— 拉格朗日白盒动力学（教学实现，Ch8 占位）。

由 ``T(q,q̇)=½q̇ᵀM(q)q̇``、``V(q)`` 经欧拉-拉格朗日方程逐项导出 M/C/G，与
PinDynamicsSolver 黑盒互为对照；config ``robotics.dynamics: lagrangian`` 切换。
"""
from __future__ import annotations

import numpy as np

from .dynamics_solver import DynamicsSolver

__all__ = ["LagrangianDynamicsSolver"]


class LagrangianDynamicsSolver(DynamicsSolver):
    """拉格朗日推导白盒动力学。"""

    def idyn(self, arm, q, dq, ddq, f_ext=None) -> np.ndarray:
        raise NotImplementedError("LagrangianDynamicsSolver 待 Ch8 实现")

    def mass_matrix(self, arm, q) -> np.ndarray:
        raise NotImplementedError("LagrangianDynamicsSolver 待 Ch8 实现")

    def coriolis(self, arm, q, dq) -> np.ndarray:
        raise NotImplementedError("LagrangianDynamicsSolver 待 Ch8 实现")

    def gravity(self, arm, q) -> np.ndarray:
        raise NotImplementedError("LagrangianDynamicsSolver 待 Ch8 实现")
