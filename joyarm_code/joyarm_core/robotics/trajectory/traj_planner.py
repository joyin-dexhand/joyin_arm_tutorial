"""TrajPlanner —— 轨迹规划器策略 ABC。

``JoyArm._traj_planner`` 的契约：实现经 config ``solvers.traj`` 选型。四个入口按
空间（joint/cart）×粒度（p2p/waypoints）划分；``method`` 选**同族内方式**，config
选**整个规划器**——两层选择不混淆。平滑契约：位置与姿态均 C2（位置/速度/加速度
无突变）。Ch5。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .segments import Trajectory

__all__ = ["TrajPlanner", "JOINT_METHODS", "CART_METHODS"]

# 公有参数 method 的合法取值表（plan_joint_p2p / plan_cart_p2p 契约，各实现类共享）
JOINT_METHODS = ("cubic", "quintic", "lspb")
CART_METHODS = ("line", "arc")


class TrajPlanner(ABC):
    """轨迹规划策略接口：关节 / 笛卡尔空间的平滑轨迹序列。"""

    @abstractmethod
    def plan_joint_p2p(
        self, arm, q0: np.ndarray, qf: np.ndarray,
        *, method: str = "quintic", T: float = 2.0, hz: int = 200, **kw,
    ) -> Trajectory:
        """关节空间点到点：``method`` ∈ cubic/quintic/lspb；``T`` 总时长 s；``hz`` 采样 Hz。"""

    @abstractmethod
    def plan_joint_waypoints(
        self, arm, qs: List[np.ndarray], Ts: List[float],
        *, smooth: bool = True,
    ) -> Trajectory:
        """关节空间多点途经（段间五次拼接，接缝位置/速度/加速度连续）。"""

    @abstractmethod
    def plan_cart_p2p(
        self, arm, *, method: str = "line", T: float = 2.0, hz: int = 200, **kw,
    ) -> Trajectory:
        """笛卡尔点到点：``line``（需 ``T0``/``Tf``）/ ``arc``（需 ``center``/``radius``/``T_start``/``angle``/``plane``）；位置沿几何路径、姿态 slerp，均配五次时间律（C2）。"""

    @abstractmethod
    def plan_cart_waypoints(
        self, arm, poses: List[np.ndarray], Ts: List[float],
        *, smooth: bool = True,
    ) -> Trajectory:
        """笛卡尔多点途经：``poses`` 为 ``(4,4)`` 列表（含起止）；段间平滑拼接（C2）。"""
