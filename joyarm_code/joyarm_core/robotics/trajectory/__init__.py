"""轨迹规划：规划器接口 + 注册表。

- ``TrajPlanner``：规划器接口（ABC，见 traj_planner.py）；
- ``REGISTRY``：注册表 ``{注册名: 规划器子类}``
        新增实现：import 后在本表加一行（注册名 = 类名小写+下划线）， config 文件 ``robotics.trajectory``段按名引用。
"""
from .traj_planner import TrajPlanner
from .traj_planner_to_joint import ToJointTrajPlanner

REGISTRY: dict = {
    "to_joint_traj_planner": ToJointTrajPlanner,   # 默认实现（首位）
}

__all__ = ["TrajPlanner", "ToJointTrajPlanner", "REGISTRY"]
