"""动力学域（策略族，一节点一文件）。

DynamicsSolver(ABC，Λ 模板) · PinDynamicsSolver(``"pin"``，默认，Ch8 占位) ·
LagrangianDynamicsSolver(``"lagrangian"``，白盒占位)。``REGISTRY`` 供 config
``solvers.dynamics`` 选型；``fdyn`` / ``idyn`` / ``mass_matrix`` / ``coriolis`` /
``gravity`` / ``cartesian_inertia`` 委托门面（教学用）。
"""
from .dynamics_solver import DynamicsSolver
from .pin_dynamics_solver import PinDynamicsSolver
from .lagrangian_dynamics_solver import LagrangianDynamicsSolver

REGISTRY = {
    "pin": PinDynamicsSolver,
    "lagrangian": LagrangianDynamicsSolver,
}

__all__ = [
    "DynamicsSolver", "PinDynamicsSolver", "LagrangianDynamicsSolver", "REGISTRY",
    "fdyn", "idyn", "mass_matrix", "coriolis", "gravity", "cartesian_inertia",
]


def fdyn(arm, q, dq, tau, f_ext=None):
    """函数式入口（教学用）：委托 ``arm.fdyn``，即 ``arm._dynamics_solver.fdyn``。"""
    return arm.fdyn(q, dq, tau, f_ext=f_ext)


def idyn(arm, q, dq, ddq, f_ext=None):
    """函数式入口（教学用）：委托 ``arm.idyn``。"""
    return arm.idyn(q, dq, ddq, f_ext=f_ext)


def mass_matrix(arm, q):
    """函数式入口（教学用）：委托 ``arm.mass_matrix``。"""
    return arm.mass_matrix(q)


def coriolis(arm, q, dq):
    """函数式入口（教学用）：委托 ``arm.coriolis``。"""
    return arm.coriolis(q, dq)


def gravity(arm, q):
    """函数式入口（教学用）：委托 ``arm.gravity``。"""
    return arm.gravity(q)


def cartesian_inertia(arm, q, frame=None):
    """函数式入口（教学用）：委托 ``arm.cartesian_inertia``。"""
    return arm.cartesian_inertia(q, frame=frame)
