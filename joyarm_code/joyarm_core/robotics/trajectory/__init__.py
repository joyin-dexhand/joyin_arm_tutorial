"""轨迹域（策略族 + 纯函数零件）。

TrajPlanner(ABC) · DefaultTrajPlanner(``"default"``，默认) · ToppraTrajPlanner
(``"toppra"``，Ch5 进阶占位)；:mod:`segments` 为纯函数族（``Trajectory`` 载体 +
joint_*/cart_* 多方式函数 + 工具），是 planner 的内部零件兼教学函数式 API。
``REGISTRY`` 供 config ``robotics.traj`` 选型；``method`` 选同族方式，两层不混淆。
"""
from .traj_planner import TrajPlanner
from .default_traj_planner import DefaultTrajPlanner
from .toppra_traj_planner import ToppraTrajPlanner
from .segments import (
    Trajectory,
    joint_cubic,
    joint_quintic,
    joint_lspb,
    joint_waypoints,
    cart_line,
    cart_arc,
    cart_waypoints,
    cart_to_joint,
    constant_velocity_retime,
    validate,
)

REGISTRY = {
    "default": DefaultTrajPlanner,
    "toppra": ToppraTrajPlanner,
}

__all__ = [
    "TrajPlanner", "DefaultTrajPlanner", "ToppraTrajPlanner", "REGISTRY",
    "Trajectory",
    "joint_cubic", "joint_quintic", "joint_lspb", "joint_waypoints",
    "cart_line", "cart_arc", "cart_waypoints", "cart_to_joint",
    "constant_velocity_retime", "validate",
    "plan_joint_p2p", "plan_joint_waypoints", "plan_cart_p2p", "plan_cart_waypoints",
]


def plan_joint_p2p(arm, q0, qf, *, method="quintic", **kw):
    """函数式入口（教学用）：委托 ``arm.plan_joint_p2p``，即 ``arm._traj_planner``。"""
    return arm.plan_joint_p2p(q0, qf, method=method, **kw)


def plan_joint_waypoints(arm, qs, Ts, **kw):
    """函数式入口（教学用）：委托 ``arm.plan_joint_waypoints``。"""
    return arm.plan_joint_waypoints(qs, Ts, **kw)


def plan_cart_p2p(arm, *, method="line", **kw):
    """函数式入口（教学用）：委托 ``arm.plan_cart_p2p``。"""
    return arm.plan_cart_p2p(method=method, **kw)


def plan_cart_waypoints(arm, poses, Ts, **kw):
    """函数式入口（教学用）：委托 ``arm.plan_cart_waypoints``。"""
    return arm.plan_cart_waypoints(poses, Ts, **kw)
