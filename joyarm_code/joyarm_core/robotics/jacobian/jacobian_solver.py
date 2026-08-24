"""JacobianSolver —— 雅可比策略 ABC（模板方法）。

``JoyArm._jacobian_solver`` 的契约：实现经 config ``solvers.jacobian`` 选型。
:meth:`jac` 为求解内核（抽象）；可操作度 / 条件数 / 静力学是 **J 的通用衍生量**，
由基类模板直接给出。Ch4。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

__all__ = ["JacobianSolver"]


class JacobianSolver(ABC):
    """速度运动学策略接口：``V = J(q)q̇``。"""

    @abstractmethod
    def jac(
        self,
        arm,
        q: np.ndarray,
        frame: Optional[Union[str, int]] = None,
        ref: str = "local",
    ) -> np.ndarray:
        """内核：``(6,n)`` 雅可比（前 3 行线速度、后 3 行角速度）；``ref`` 取 local/base（末端帧自身系 / 基座系）。"""

    def manipulability(self, arm, q: np.ndarray, frame=None) -> float:
        """Yoshikawa 可操作度 ``w = sqrt(det(J Jᵀ))``（椭球体积度量）。"""
        J = self.jac(arm, q, frame=frame)
        return float(np.sqrt(np.linalg.det(J @ J.T)))

    def cond_number(self, arm, q: np.ndarray, frame=None) -> float:
        """雅可比条件数（椭球各向异性度量）。"""
        return float(np.linalg.cond(self.jac(arm, q, frame=frame)))

    def statics(self, arm, q: np.ndarray, F: np.ndarray, frame=None) -> np.ndarray:
        """静力学：``τ = JᵀF``，``F`` 为 (6,) 末端六维力（N / N·m）。"""
        return self.jac(arm, q, frame=frame).T @ np.asarray(F, dtype=float)
