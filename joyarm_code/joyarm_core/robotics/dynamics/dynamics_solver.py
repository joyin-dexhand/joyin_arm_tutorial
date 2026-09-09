"""DynamicsSolver —— 动力学求解器接口（ABC）

求解器子类只需实现
- :meth:`idyn`（逆动力学：状态+加速度 → τ）
- :meth:`mass_matrix`（惯量 M）
- :meth:`coriolis`（科氏/向心 C 项）
- :meth:`gravity`（重力 G 项）

派生量：正动力学 / 笛卡尔惯量

config ``robotics.dynamics`` 段写注册名，即按名实例化装入 ``_dynamics_solvers`` 成员字典。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

__all__ = ["DynamicsSolver"]


class DynamicsSolver(ABC):
    """动力学策略接口：``M(q)q̈ + C(q,q̇)q̇ + G(q) = τ``。"""

    # ----------------------------------------------------------
    # dynamics 抽象内核（idyn/mass_matrix/coriolis/gravity等）
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # dynamics 派生量：正动力学 / 笛卡尔惯量
    #（基于四个抽象内核；cartesian_inertia 经 ``arm.jac`` 门面取 J）
    # ----------------------------------------------------------
    def fdyn(self, arm, q: np.ndarray, dq: np.ndarray,
             tau: np.ndarray) -> np.ndarray:
        """正动力学 ``q̈ = M⁻¹(τ − h)``（``h = C(q,q̇)q̇ + G(q)`` 偏置力），
        返回 ``(n,)``。末端外力旋量 ``F`` 请先并入力矩（``τ ← τ − JᵀF``）。"""
        M = self.mass_matrix(arm, q)
        h = self.coriolis(arm, q, dq) + self.gravity(arm, q)
        return np.linalg.solve(M, np.asarray(tau, dtype=float) - h)

    def cartesian_inertia(self, arm, q: np.ndarray,
            frame: Union[str, int], ref: str = "base") -> np.ndarray:
        """笛卡尔惯量 ``Λ = J⁺ᵀ M J⁺``，返回 ``(6,6)``（J 经 ``arm.jac``）。"""
        J_pinv = np.linalg.pinv(arm.jac(q, frame, ref=ref))
        return J_pinv.T @ self.mass_matrix(arm, q) @ J_pinv
