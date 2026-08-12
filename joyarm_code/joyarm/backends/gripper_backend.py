"""
``GripperBackend`` —— 两指夹爪真机通信后端。

负责两指夹爪的底层通信（CAN 总线，夹爪电机 ID ``0x07``，与 6 个关节电机
``0x01~0x06`` 区分）。本类实现 :class:`~joyarm.backends.end_backend.EndBackend`
抽象，是应用层 :class:`~joyarm.arm.gripper.Gripper` 的通信载体。
"""

from __future__ import annotations

__all__ = ["GripperBackend"]

from .end_backend import EndBackend


class GripperBackend(EndBackend):
    """两指夹爪真机通信后端（CAN 总线，夹爪 ID ``0x07``）。

    :param can_interface: CAN 接口名（如 ``"can0"``）。
    :param baudrate: 波特率（CAN bps）。
    :param gripper_id: 夹爪在 CAN 总线上的 ID（默认 ``0x07``，
                      与 6 个关节电机 ``0x01~0x06`` 区分）。
    """

    def __init__(
        self,
        can_interface: str = "can0",
        baudrate: int = 1_000_000,
        gripper_id: int = 0x07,
        **kwargs,
    ):
        self.can_interface = can_interface
        self.baudrate = baudrate
        # 夹爪在 CAN 总线上的 ID（与 6 个关节电机 0x01~0x06 区分，夹爪用 0x07）
        self.gripper_id = gripper_id
        self._kwargs = kwargs
        # 真实 SDK 句柄
        self._handle = None

    # ----------------------------------------------------------
    # 连接 / 断开
    # ----------------------------------------------------------
    def connect(self) -> None:
        # 占位：Ch13 实现——打开 CAN 接口、建立与夹爪的通信链路。
        raise NotImplementedError("GripperBackend.connect 待 Ch13 实现")

    def disconnect(self) -> None:
        # 占位：Ch13 实现——关闭 CAN 接口、释放资源。
        raise NotImplementedError("GripperBackend.disconnect 待 Ch13 实现")

    # ----------------------------------------------------------
    # 状态读取
    # ----------------------------------------------------------
    def read_state(self) -> dict:
        # 占位：Ch13 实现——读回当前两指间距、夹持力、是否正在抓持。
        # 返回：{"width_mm": float, "force_N": float, "is_grasping": bool}
        raise NotImplementedError("GripperBackend.read_state 待 Ch13 实现")

    # ----------------------------------------------------------
    # 指令下发
    # ----------------------------------------------------------
    def send_position(self, position: float) -> None:
        # 占位：Ch13 实现——精确控制两指间距，适合抓已知尺寸的物体。
        # position = 目标两指间距（毫米）
        raise NotImplementedError("GripperBackend.send_position 待 Ch13 实现")

    def send_force(self, force: float) -> None:
        # 占位：Ch13 实现——按目标夹持力闭合，夹到即停，适合抓易碎/柔软物体。
        # force = 目标夹持力（牛顿）
        raise NotImplementedError("GripperBackend.send_force 待 Ch13 实现")

    def send_action(self, action: str) -> None:
        # 占位：Ch13 实现——通用开合动作：
        #   "open"  → 张开到最大（默认行程/力度）
        #   "close" → 闭合（夹到默认力度即停）
        raise NotImplementedError("GripperBackend.send_action 待 Ch13 实现")
