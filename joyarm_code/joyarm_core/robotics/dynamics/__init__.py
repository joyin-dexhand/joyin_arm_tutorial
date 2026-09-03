"""动力学：求解器接口 + 注册表。

- ``DynamicsSolver``：求解器接口（ABC，见 dynamics_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``。

"""
from .dynamics_solver import DynamicsSolver

REGISTRY: dict = {}

__all__ = ["DynamicsSolver", "REGISTRY"]
