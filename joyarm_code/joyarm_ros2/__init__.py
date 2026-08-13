"""``joyarm_ros2`` —— ROS2 封装兄弟包（依赖核心包 ``joyarm`` + ``rclpy``）。

把核心 SDK :mod:`joyarm` 的能力封装为 ROS2 节点，与核心包解耦：

- :mod:`joyarm_ros2.arm_nodes`【Ch10】：每个机器人型号 → 一个 ROS2 node，
  对外暴露话题（``ArmState``/``JointState``）/ 服务（IK、控制）/ 动作（轨迹执行）。
- :mod:`joyarm_ros2.apps`【Ch12/14/15】：应用级 node ——
  :mod:`~joyarm_ros2.apps.teleop`（示教/遥操作）、
  :mod:`~joyarm_ros2.apps.vision`（相机标定/检测）、UI/语音/NFC/UWB/体感等；
  **可视化统一走 rviz2**。

设计要点
--------

- 核心包 :mod:`joyarm` 保持 **ROS2-free**（``import joyarm`` 不需要 ``rclpy``）；
  本包独立、可选，仅在需要 ROS2 通信时安装。
- 各模块当前为**占位**（签名 + ``raise NotImplementedError("ChN 实现")``），
  按章节落地；内部类型仍复用 :mod:`joyarm.utils.types`，不重复定义。
"""
from . import arm_nodes, apps

__version__ = "0.1.0"
__all__ = ["arm_nodes", "apps"]
