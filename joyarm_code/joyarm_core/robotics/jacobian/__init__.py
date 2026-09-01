"""速度运动学与静力学域（策略接口）。

JacobianSolver(ABC，衍生量模板) 为通用接口；具体求解算法为教程 Ch4 教学内容，
实现后经 ``REGISTRY`` 注册接入。``jac`` / ``manipulability`` / ``cond_number`` /
``statics`` 委托门面（教学用）。
"""
from .jacobian_solver import JacobianSolver

REGISTRY: dict = {}

__all__ = ["JacobianSolver", "REGISTRY",
           "jac", "manipulability", "cond_number", "statics"]


def jac(arm, q, frame=None, ref="local"):
    """函数式入口（教学用）：委托 ``arm.jac``，即 ``arm._jacobian_solvers`` 活动成员。"""
    return arm.jac(q, frame=frame, ref=ref)


def manipulability(arm, q, frame=None):
    """函数式入口（教学用）：委托 ``arm.manipulability``。"""
    return arm.manipulability(q, frame=frame)


def cond_number(arm, q, frame=None):
    """函数式入口（教学用）：委托 ``arm.cond_number``。"""
    return arm.cond_number(q, frame=frame)


def statics(arm, q, F, frame=None):
    """函数式入口（教学用）：委托 ``arm.statics``。"""
    return arm.statics(q, F, frame=frame)
