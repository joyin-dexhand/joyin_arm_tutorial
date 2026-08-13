"""ROS2 适配器（适配器层，Ch10 占位，可选）。

桥接内部纯 Python 类型（dataclass）与 ROS2 消息。

.. note::

    核心类型保持纯 Python dataclass，ROS2 不污染核心层（架构设计第 9 条）。
    ROS2 部分**先不实现**，仅保留接口签名 + 简短说明 +
    ``raise NotImplementedError("Ch10 实现")``。

对应章节：Ch10（``chapter3_4.md`` 第十章 ROS2）。
"""
from __future__ import annotations

from joyarm.utils.types import ArmState, JointState, Pose

__all__ = ["Ros2Adapter", "to_joint_msg", "from_pose_msg"]

# 【给新手的话】本文件是"占位文件"——函数体目前都是 raise NotImplementedError。
# 这不是 bug，而是教学安排：第十章会真正实现 ROS2 通信（且为可选模块）。
# ROS2 是机器人领域常用的通信中间件，让不同程序/设备之间能互发消息。
# 本模块充当"翻译官"：把库内部的 Python 数据类型转成 ROS2 标准消息格式，反之亦然。
# 核心类型保持纯 Python，不被 ROS2 绑定——不用 ROS2 时完全不受影响。


class Ros2Adapter:
    """ROS2 桥接适配器（占位）。

    发布 / 订阅 ``ArmState`` / ``JointState`` / EE 指令；负责内部类型与
    ROS2 消息互转。
    """

    def __init__(self, node_name: str = "joyarm", **kwargs):
        self.node_name = node_name

    def spin(self) -> None:
        """阻塞运行 ROS2 节点。"""
        # 占位：Ch10 实现——启动节点的事件循环，持续收发消息直到被中断。
        raise NotImplementedError("Ros2Adapter.spin 待 Ch10 实现")


def to_joint_msg(state: ArmState) -> JointState:
    """内部 :class:`ArmState` → ROS2 ``JointState`` 消息（占位）。

    .. note::

        返回类型实际为 ROS2 ``sensor_msgs/JointState``；此处注解保持
        内部类型仅为占位可读性，Ch10 实现时替换。
    """
    # 占位：Ch10 实现——把内部 ArmState 的关节部分转成 ROS2 标准的 JointState 消息发出去。
    raise NotImplementedError("to_joint_msg 待 Ch10 实现")


def from_pose_msg(msg) -> Pose:
    """ROS2 ``PoseStamped`` → 内部 :class:`Pose`（占位）。"""
    # 占位：Ch10 实现——把收到的 ROS2 PoseStamped 消息解析回内部 Pose 对象。
    raise NotImplementedError("from_pose_msg 待 Ch10 实现")
