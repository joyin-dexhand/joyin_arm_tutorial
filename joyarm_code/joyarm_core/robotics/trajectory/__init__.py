"""轨迹规划：规划器接口 + 注册表。

- ``TrajPlanner``：规划器接口（ABC，见 traj_planner.py）；
- ``REGISTRY``：注册表 ``{注册名: 规划器子类}``
                子类加 ``@register``装饰器即自动入表（注册名 = 类名小写 + 下划线）
                机械臂型号 config 文件的 ``robotics.trajectory`` 段按名引用。
- ``ToJointTrajPlanner``：轨迹规划默认实现（注册名 ``to_joint_traj_planner``，见 ``traj_planner_to_joint.py``）。
"""
from .._registry import register
from .traj_planner import TrajPlanner

REGISTRY: dict = {}

from .traj_planner_to_joint import ToJointTrajPlanner  # noqa: E402 —— 注册须在 REGISTRY 定义后执行

__all__ = ["TrajPlanner", "ToJointTrajPlanner", "REGISTRY", "register"]
