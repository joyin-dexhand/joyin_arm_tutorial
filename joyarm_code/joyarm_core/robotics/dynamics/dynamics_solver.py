"""DynamicsSolver —— 动力学策略 ABC（模板方法）。

``JoyArm._dynamics_solver`` 的契约：实现经 config ``robotics.dynamics`` 选型。
逆动力学与 M/C/G 四项为求解内核（抽象）；笛卡尔惯量 ``Λ = J⁻ᵀMJ⁻¹`` 是
M ⊕ J 的通用衍生量，由基类模板给出（J 经 ``arm.jac`` 门面，自动跟随雅可比替换）。
本库不做物理仿真，不设正动力学（fdyn）。Ch8。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

__all__ = ["DynamicsSolver"]


class DynamicsSolver(ABC):
    """动力学策略接口：``M(q)q̈ + C(q,q̇)q̇ + G(q) = τ`` 的求解（无正动力学）。"""

    @abstractmethod
    def idyn(self, arm, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray,
             f_ext: Optional[np.ndarray] = None) -> np.ndarray:
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

    def cartesian_inertia(self, arm, q: np.ndarray,
                          frame: Optional[Union[str, int]] = None) -> np.ndarray:
        """笛卡尔惯量 ``Λ = J⁻ᵀ M J⁻¹``（Ch9 阻抗直接用），返回 ``(6,6)``。"""
        J_inv = np.linalg.inv(arm.jac(q, frame=frame))
        return J_inv.T @ self.mass_matrix(arm, q) @ J_inv
