"""``joyarm_ros2.apps`` —— 应用级 ROS2 节点【Ch12/14/15 占位】。

- :mod:`~joyarm_ros2.apps.teleop`：示教记录/回放 + 遥操作映射（Ch12）。
- :mod:`~joyarm_ros2.apps.vision`：相机内参/手眼标定、检测/位姿估计（Ch14）。
- :mod:`~joyarm_ros2.apps.ros2`：核心类型 ↔ ROS2 消息桥接（Ch10）。

可视化走 **rviz2**（不在核心包内做可视化）。各模块当前为占位，按章节落地。
"""
from . import teleop, vision, ros2

__all__ = ["teleop", "vision", "ros2"]
