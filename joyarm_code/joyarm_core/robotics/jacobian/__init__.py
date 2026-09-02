"""速度运动学与静力学：求解器接口 + 注册表。

- ``JacobianSolver``：求解器接口（ABC，见 jacobian_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``。

"""
from .jacobian_solver import JacobianSolver

REGISTRY: dict = {}

__all__ = ["JacobianSolver", "REGISTRY"]
