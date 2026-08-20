"""轨迹域（策略族 + 纯函数零件）。

TrajPlanner(ABC) · DefaultTrajPlanner(``"default"``，默认) · ToppraTrajPlanner
(``"toppra"``，Ch5 进阶占位)；:mod:`segments` 为纯函数族（``Trajectory`` 载体 +
joint_*/cart_* 多方式函数 + 工具），是 planner 的内部零件兼教学函数式 API。
``REGISTRY`` 供 config ``solvers.traj`` 选型；``method`` 选同族方式，两层不混淆。
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
    "cart_line", "cart_arc", "cart_to_joint",
    "constant_velocity_retime", "validate",
    "plan_joint", "plan_waypoints", "plan_cart",
]


def plan_joint(arm, q0, qf, *, method="quintic", **kw):
    """函数式入口（教学用）：委托 ``arm.plan_joint``，即 ``arm._traj_planner``。"""
    return arm.plan_joint(q0, qf, method=method, **kw)


def plan_waypoints(arm, qs, Ts, **kw):
    """函数式入口（教学用）：委托 ``arm.plan_waypoints``。"""
    return arm.plan_waypoints(qs, Ts, **kw)


def plan_cart(arm, *, method="line", **kw):
    """函数式入口（教学用）：委托 ``arm.plan_cart``。"""
    return arm.plan_cart(method=method, **kw)
