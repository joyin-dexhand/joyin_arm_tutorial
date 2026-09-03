"""轨迹规划：规划器接口 + 轨迹载体 + 注册表。

- ``TrajPlanner``：规划器接口（ABC，见 traj_planner.py）；
- ``Trajectory``：轨迹载体（时间戳 + 采样点序列，npz 持久化）；
- ``REGISTRY``：注册表 ``{注册名: 规划器子类}``。

"""
from .traj_planner import TrajPlanner, Trajectory

REGISTRY: dict = {}

__all__ = ["TrajPlanner", "Trajectory", "REGISTRY"]
