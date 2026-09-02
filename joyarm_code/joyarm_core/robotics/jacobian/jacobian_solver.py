"""JacobianSolver —— 雅可比求解器接口（ABC：只定骨架，不含算法）。

雅可比描述**关节角速度到末端速度的传递**：``V = J(q)·q̇``（``V`` 为六维：
线速度 + 角速度）。它连接关节空间与笛卡尔空间，是速度级逆解、可操作度
分析与静力学的共同基础。

本文件为通用骨架，求解器子类只需实现内核 :meth:`jac`；可操作度 / 条件数 /
静力学是 **J 的通用衍生量**，由基类模板直接给出。

config ``robotics.jacobian`` 段写注册名，即按名实例化装入 ``_jacobian_solvers`` 成员字典。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

__all__ = ["JacobianSolver"]


class JacobianSolver(ABC):
    """速度运动学策略接口：``V = J(q)q̇``。"""

    @abstractmethod
    def jac(
        self,
        arm,
        q: np.ndarray,
        frame: Union[str, int],
        ref: str = "local",
    ) -> np.ndarray:
        """内核：``(6,n)`` 雅可比（前 3 行线速度、后 3 行角速度）；``ref`` 取 local/base（末端帧自身系 / 基座系）。"""

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
