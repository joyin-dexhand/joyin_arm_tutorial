"""JacobianSolver —— 雅可比求解器接口（ABC）

求解器子类只需实现
- :meth:`jac`（6×n 雅可比；``ref`` 取 local/base）。

config ``robotics.jacobian`` 段写注册名，即按名实例化装入 ``_jacobian_solvers`` 成员字典。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

__all__ = ["JacobianSolver"]


class JacobianSolver(ABC):
    """速度运动学策略接口：``V = J(q)q̇``。"""

    # ----------------------------------------------------------
    # jacobian 抽象内核（jac参考系：local/base）
    # ----------------------------------------------------------
    @abstractmethod
    def jac(self,
        arm,
        q: np.ndarray,
        frame: Union[str, int],
        ref: str = "base",
    ) -> np.ndarray:
        """``(6,n)`` 雅可比；``ref`` 取 base（默认，基座系）/ local（自身系）。"""

    # ----------------------------------------------------------
    # jacobian 派生量模板（可操作度 / 条件数 / 静力学，基于 self.jac）
    # ----------------------------------------------------------
    def manipulability(self, arm, q: np.ndarray, frame: Union[str, int]) -> float:
        """Yoshikawa 可操作度 ``w = sqrt(det(J Jᵀ))``（椭球体积度量）。"""
        J = self.jac(arm, q, frame)
        return float(np.sqrt(np.linalg.det(J @ J.T)))

    def cond_number(self, arm, q: np.ndarray, frame: Union[str, int]) -> float:
        """雅可比条件数（椭球各向异性度量）。"""
        return float(np.linalg.cond(self.jac(arm, q, frame)))

    def statics(self, arm, q: np.ndarray, F: np.ndarray, frame: Union[str, int]) -> np.ndarray:
        """静力学：``τ = JᵀF``，``F`` 为 (6,) 末端六维力（N / N·m）。"""
        return self.jac(arm, q, frame).T @ np.asarray(F, dtype=float)
