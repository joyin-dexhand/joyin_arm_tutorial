"""正运动学：求解器接口 + 注册表。

- ``FkineSolver``：求解器接口（ABC，见 fkine_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``。

"""
from .fkine_solver import FkineSolver

REGISTRY: dict = {}

__all__ = ["FkineSolver", "REGISTRY"]
