"""DynamicsSolver —— 动力学求解器接口（ABC）

求解器子类需实现内核
- :meth:`idyn`（逆动力学：状态+加速度 → τ）、:meth:`mass_matrix`（惯量 M）、
  :meth:`coriolis`（科氏/向心 C 项）、:meth:`gravity`（重力 G 项）。

config ``robotics.dynamics`` 段写注册名，即按名实例化装入 ``_dynamics_solvers`` 成员字典。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

__all__ = ["DynamicsSolver"]


class DynamicsSolver(ABC):
    """动力学策略接口：``M(q)q̈ + C(q,q̇)q̇ + G(q) = τ``（无正动力学 fdyn）。"""

    @abstractmethod
    def idyn(self,
        arm,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        f_ext: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """逆动力学（RNEA）：状态 + 加速度 → ``τ``，返回 ``(n,)`` N·m。"""

    @abstractmethod
    def mass_matrix(self, arm, q: np.ndarray) -> np.ndarray:
        """关节空间惯量矩阵 ``M(q)``（CRBA），返回 ``(n,n)``。"""

    @abstractmethod
    def coriolis(self, arm, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """科氏 + 向心力项 ``C(q,q̇)q̇``（向量形式），返回 ``(n,)``。"""

    @abstractmethod
    def gravity(self, arm, q: np.ndarray) -> np.ndarray:
        """重力项 ``G(q)``，返回 ``(n,)``。"""

    def cartesian_inertia(self, arm, q: np.ndarray, frame: Union[str, int]) -> np.ndarray:
        """笛卡尔惯量 ``Λ = J⁻ᵀ M J⁻¹``，返回 ``(6,6)``（J 经 ``arm.jac`` 门面取）。"""
        J_inv = np.linalg.inv(arm.jac(q, frame))
        return J_inv.T @ self.mass_matrix(arm, q) @ J_inv
