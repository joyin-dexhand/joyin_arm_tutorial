"""TrajPlanner —— 轨迹规划器策略 ABC。

``JoyArm._traj_planner`` 的契约：实现经 config ``solvers.traj`` 选型。两个入口按
空间划分；``method`` 选**同族内方式**，config 选**整个规划器**——两层选择不混淆。Ch5。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .segments import Trajectory

__all__ = ["TrajPlanner"]

_JOINT_METHODS = ("cubic", "quintic", "lspb")
_CART_METHODS = ("line", "arc")


class TrajPlanner(ABC):
    """轨迹规划策略接口：关节 / 笛卡尔空间的平滑轨迹序列。"""

    @abstractmethod
    def plan_joint(
        self, arm, q0: np.ndarray, qf: np.ndarray,
        *, method: str = "quintic", T: float = 2.0, hz: int = 200, **kw,
    ) -> Trajectory:
        """关节空间点到点：``method`` ∈ cubic/quintic/lspb；``T`` 总时长 s；``hz`` 采样 Hz。"""

    @abstractmethod
    def plan_waypoints(
        self, arm, qs: List[np.ndarray], Ts: List[float],
        *, smooth: bool = True,
    ) -> Trajectory:
        """关节空间多点途经（段间平滑拼接）。"""

    @abstractmethod
    def plan_cart(
        self, arm, *, method: str = "line", T: float = 2.0, hz: int = 200, **kw,
    ) -> Trajectory:
        """笛卡尔空间：``line``（需 ``T0``/``Tf``）/ ``arc``（需 ``center``/``radius``/``T_start``/``angle``/``plane``）。"""
