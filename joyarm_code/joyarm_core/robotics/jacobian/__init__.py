"""速度运动学与静力学子包：求解器接口 + 注册表。

JacobianSolver(ABC，衍生量模板) 为通用接口；具体求解算法为教程 Ch4 教学内容，
实现后经 ``REGISTRY`` 注册接入。用法二选一：直接实例化子类求解（``arm``
鸭子类型），或经门面 ``arm.jac`` 及衍生量方法（活动成员）。
"""
from .jacobian_solver import JacobianSolver

REGISTRY: dict = {}

__all__ = ["JacobianSolver", "REGISTRY"]
