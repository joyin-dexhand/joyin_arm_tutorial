"""轨迹规划：规划器接口 + 自动/力规划器 + 规划算法 + 注册表。

- ``TrajPlanner``：规划器接口（ABC，见 traj_planner.py）；
- ``planning``：**规划算法**（关节/笛卡尔空间全部规划方法的纯函数存放）；
- ``REGISTRY``：注册表 ``{注册名: 规划器子类}``。

"""
from .planning import cubic_q, cubic_traj
from .traj_planner import TrajPlanner
from .auto_planner import AutoTrajPlanner
from .force_planner import ForceTrajPlanner

REGISTRY: dict = {}

__all__ = ["TrajPlanner", "AutoTrajPlanner", "ForceTrajPlanner",
           "cubic_q", "cubic_traj", "REGISTRY"]
