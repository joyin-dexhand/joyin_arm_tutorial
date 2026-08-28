"""速度运动学与静力学域（策略族，一节点一文件）。

JacobianSolver(ABC，衍生量模板) · PinJacobianSolver(``"pin"``，默认，Ch4 占位) ·
GeometricJacobianSolver(``"geometric"``，白盒占位)。``REGISTRY`` 供 config
``robotics.jacobian`` 选型；``jac`` / ``manipulability`` / ``cond_number`` /
``statics`` 委托门面（教学用）。
"""
from .jacobian_solver import JacobianSolver
from .pin_jacobian_solver import PinJacobianSolver
from .geometric_jacobian_solver import GeometricJacobianSolver

REGISTRY = {
    "pin": PinJacobianSolver,
    "geometric": GeometricJacobianSolver,
}

__all__ = [
    "JacobianSolver", "PinJacobianSolver", "GeometricJacobianSolver", "REGISTRY",
    "jac", "manipulability", "cond_number", "statics",
]


def jac(arm, q, frame=None, ref="local"):
    """函数式入口（教学用）：委托 ``arm.jac``，即 ``arm._jacobian_solver.jac``。"""
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
