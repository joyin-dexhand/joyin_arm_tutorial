"""轨迹规划：基类接口 + 子类注册表。

- ``TrajPlanner``：规划器接口（ABC，见 traj_planner.py）；
- ``REGISTRY``：注册表 ``{注册名: 规划器子类}``

新子类实现步骤：
    1. 在本包内创建 `traj_planner_<new>.py` 子类文件，实现子类 `<New>TrajPlanner`（继承自 `traj_planner.py` 的 `TrajPlanner` 基类）；
    2. 在本文件中 import 新子类，并在 ``REGISTRY`` 显式加一行注册（注册名 = 类名小写+下划线）；
"""
from .traj_planner import TrajPlanner
from .traj_planner_single_joint_hold import SingleJointHoldTrajPlanner
from .traj_planner_single_joint_cubic import SingleJointCubicTrajPlanner

REGISTRY: dict = {
    "single_joint_hold_traj_planner": SingleJointHoldTrajPlanner,
    "single_joint_cubic_traj_planner": SingleJointCubicTrajPlanner,
}

__all__ = ["TrajPlanner", "SingleJointHoldTrajPlanner", "SingleJointCubicTrajPlanner", "REGISTRY"]
