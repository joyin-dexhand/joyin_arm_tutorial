"""DefaultTrajPlanner —— 默认轨迹规划器（纯函数族的薄封装）。

对 segments 的多方式函数按 ``method`` 分派，无自有状态；与 ToppraTrajPlanner
（时间最优）等互为可换策略。Ch5。
"""
from __future__ import annotations

from typing import List

import numpy as np

from .traj_planner import TrajPlanner, JOINT_METHODS, CART_METHODS
from .segments import (
    Trajectory,
    joint_cubic,
    joint_quintic,
    joint_lspb,
    joint_waypoints,
    cart_line,
    cart_arc,
)

__all__ = ["DefaultTrajPlanner"]


class DefaultTrajPlanner(TrajPlanner):
    """多方式多项式 / 几何插值规划（默认）。"""

    def plan_joint(self, arm, q0, qf, *, method="quintic", T=2.0, hz=200, **kw) -> Trajectory:
        if method == "cubic":
            return joint_cubic(q0, qf, T, hz=hz)
        elif method == "quintic":
            return joint_quintic(q0, qf, T, hz=hz, **kw)
        elif method == "lspb":
            return joint_lspb(q0, qf, T, hz=hz, **kw)
        raise ValueError(f"未知 method={method!r}（关节空间可用：{JOINT_METHODS}）")

    def plan_waypoints(self, arm, qs: List[np.ndarray], Ts: List[float],
                       *, smooth: bool = True) -> Trajectory:
        return joint_waypoints(qs, Ts, smooth=smooth)

    def plan_cart(self, arm, *, method="line", T=2.0, hz=200, **kw) -> Trajectory:
        if method == "line":
            return cart_line(kw["T0"], kw["Tf"], T, hz=hz)
        elif method == "arc":
            return cart_arc(kw["center"], kw["radius"], kw["T_start"],
                            kw["angle"], plane=kw.get("plane", "xy"), T=T, hz=hz)
        raise ValueError(f"未知 method={method!r}（笛卡尔空间可用：{CART_METHODS}）")
