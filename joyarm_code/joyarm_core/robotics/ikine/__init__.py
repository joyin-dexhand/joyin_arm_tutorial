"""逆运动学：求解器接口 + 注册表。

- ``IkineSolver``：求解器接口（ABC，见 ikine_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``。

"""
from .ikine_solver import IkineSolver

REGISTRY: dict = {}

__all__ = ["IkineSolver", "REGISTRY"]
