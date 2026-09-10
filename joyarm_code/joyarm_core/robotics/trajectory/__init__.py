"""轨迹规划：规划器接口 + 自动/力规划器 + 注册表。

- ``TrajPlanner``：规划器接口（ABC，见 traj_planner.py）；
- ``REGISTRY``：注册表 ``{注册名: 规划器子类}``。

"""
from .traj_planner import TrajPlanner

REGISTRY: dict = {}

__all__ = ["TrajPlanner", "REGISTRY"]
