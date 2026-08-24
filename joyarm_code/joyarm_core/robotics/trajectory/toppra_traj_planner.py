"""ToppraTrajPlanner —— 时间最优规划器（备选实现，Ch5 进阶占位）。

TOPP-RA：先给几何路径，再在速度/加速度约束下求最快时间分配；
config ``solvers.traj: toppra`` 切换。
"""
from __future__ import annotations

from typing import List

import numpy as np

from .traj_planner import TrajPlanner
from .segments import Trajectory

__all__ = ["ToppraTrajPlanner"]


class ToppraTrajPlanner(TrajPlanner):
    """时间最优规划。"""

    def plan_joint_p2p(self, arm, q0, qf, *, method="quintic", T=2.0, hz=200, **kw) -> Trajectory:
        raise NotImplementedError("ToppraTrajPlanner 待 Ch5 实现")

    def plan_joint_waypoints(self, arm, qs: List[np.ndarray], Ts: List[float],
                             *, smooth: bool = True) -> Trajectory:
        raise NotImplementedError("ToppraTrajPlanner 待 Ch5 实现")

    def plan_cart_p2p(self, arm, *, method="line", T=2.0, hz=200, **kw) -> Trajectory:
        raise NotImplementedError("ToppraTrajPlanner 待 Ch5 实现")

    def plan_cart_waypoints(self, arm, poses: List[np.ndarray], Ts: List[float],
                            *, smooth: bool = True) -> Trajectory:
        raise NotImplementedError("ToppraTrajPlanner 待 Ch5 实现")
