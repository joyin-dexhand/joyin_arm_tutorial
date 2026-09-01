"""动力学域（策略接口）。

DynamicsSolver(ABC，Λ 模板) 为通用接口；具体求解算法为教程 Ch8 教学内容，
实现后经 ``REGISTRY`` 注册接入。``idyn`` / ``mass_matrix`` / ``coriolis`` /
``gravity`` / ``cartesian_inertia`` 委托门面（教学用）。
"""
from .dynamics_solver import DynamicsSolver

REGISTRY: dict = {}

__all__ = ["DynamicsSolver", "REGISTRY",
           "idyn", "mass_matrix", "coriolis", "gravity", "cartesian_inertia"]


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
