"""``joyarm.application`` —— 应用层子包。

汇聚面向最终用户的上层能力，含三种架构角色：

- **可视化**（独立横向层级）：:mod:`joyarm.application.viz` —— meshcat
  运动学可视化 / 轨迹预演（已实现）。
- **适配器**：:mod:`joyarm.application.ros2`（ROS2 封装，Ch10 占位）、
  :mod:`joyarm.application.vision`（视觉接入，Ch14 占位）。
- **应用支撑**：:mod:`joyarm.application.teleop`（示教 / 遥操作，Ch12 占位）。
"""
from . import viz, teleop, ros2, vision

__all__ = ["viz", "teleop", "ros2", "vision"]
