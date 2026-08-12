"""两指夹爪（设备模型层，Ch13 占位）。

提供通用的两指夹爪接口：开 / 合、位置控制、力度控制、状态读取。
对应 ``chapter1_4`` 位置模式示例与 ``chapter4_3`` 第十三章末端执行器统一接口。

``Gripper`` 与 ``Arm`` 同属**设备模型层**：包装对应的通信后端
（:class:`~joyarm.backends.end_backend.EndBackend`，具体实现如
:class:`~joyarm.backends.gripper_backend.GripperBackend`），把"开/合/位置/
力度"等业务语义翻译为底层 CAN 收发——与 ``Arm`` ↔ ``ArmBackend`` 平行。

.. note::

    未来若需支持其他末端装置（五指灵巧手 / 电磁吸附 / 气压吸附等），
    可作为独立模块新增，共用统一的末端执行器接口（``EndBackend``）。

对应章节：Ch13（``chapter4_3.md`` 第十三章 末端执行器统一接口与自识别）。
当前状态：仅签名 + docstring + ``raise NotImplementedError("Ch13 实现")``。
"""
from __future__ import annotations

from ..backends.end_backend import EndBackend

__all__ = ["Gripper"]

# 【给新手的话】本文件是"占位文件"——方法体目前都是 raise NotImplementedError。
# 这不是 bug，而是教学安排：第十三章会真正实现夹爪控制。
# 夹爪是最常见的末端执行器（装在机械臂末端的"手"），通过两个手指的开合来抓放物体。
# 控制方式分两种：位置控制（开多大）和力度控制（夹多紧），后者能避免把物体夹坏。


class Gripper:
    """两指夹爪。

    通过末端执行器通信后端与夹爪通信。

    :param backend: 末端执行器通信后端实例（如 ``GripperBackend``）。
    :param gripper_id: 夹爪总线 ID。
    """

    def __init__(self, backend: EndBackend, gripper_id: int = 0x06, **kwargs):
        self.backend = backend
        # 夹爪在 CAN 总线上的 ID（与 6 个关节电机 0x00~0x05 区分，夹爪用 0x06）
        self.gripper_id = gripper_id

    def open(self) -> None:
        """通用开合（默认行程 / 力度）。"""
        # 占位：Ch13 实现——下发"张开到最大"的指令给夹爪。
        raise NotImplementedError("Gripper.open 待 Ch13 实现")

    def close(self) -> None:
        """通用闭合（默认行程 / 力度）。"""
        # 占位：Ch13 实现——下发"闭合"指令，夹到默认力度即停。
        raise NotImplementedError("Gripper.close 待 Ch13 实现")

    def set_position(self, width_mm: float) -> None:
        """位置控制（对应 chapter1_4 位置模式）。

        :param width_mm: 目标两指间距，毫米。
        """
        # 占位：Ch13 实现——精确控制两指间距，适合抓已知尺寸的物体。
        raise NotImplementedError("Gripper.set_position 待 Ch13 实现")

    def set_force(self, N: float) -> None:
        """力度控制。

        :param N: 目标夹持力，牛顿。
        """
        # 占位：Ch13 实现——按目标夹持力闭合，夹到即停，适合抓易碎/柔软物体。
        raise NotImplementedError("Gripper.set_force 待 Ch13 实现")

    def get_state(self) -> dict:
        """读取当前状态。

        :return: ``{"width_mm": float, "force_N": float, "is_grasping": bool}``。
        """
        # 占位：Ch13 实现——读回当前两指间距、夹持力、是否正在抓持。
        raise NotImplementedError("Gripper.get_state 待 Ch13 实现")
