"""``BackendEndJoyGripper`` —— JoyArm 两指夹爪真机后端。

基于 CAN 总线驱动两指夹爪（ID ``0x07``，与 6 个关节电机 ``0x01~0x06`` 区分）。
派生自 :class:`~joyarm_core.backends.backend_end.BackendEnd`，Ch13 实现；当前为占位。
是 :class:`~joyarm_core.arms.arm.Arm` 末端执行器的通信载体（``Arm.backend_end``）。
"""
from __future__ import annotations

from .backend_end import BackendEnd

__all__ = ["BackendEndJoyGripper"]


class BackendEndJoyGripper(BackendEnd):
    """JoyArm 两指夹爪真机通信后端（CAN 总线，夹爪 ID ``0x07``）。

    :param can_interface: CAN 接口名（如 ``"can0"``）。
    :param baudrate: 波特率（CAN bps）。
    :param id: 夹爪在 CAN 总线上的 ID（默认 ``7`` = ``0x07``）。
    """

    def __init__(
        self,
        can_interface: str = "can0",
        baudrate: int = 1_000_000,
        id: int = 7,
    ):
        self.can_interface = can_interface
        self.baudrate = baudrate
        self.gripper_id = id

    # ---- 连接 ----
    def connect(self) -> None:
        raise NotImplementedError("BackendEndJoyGripper.connect 待 Ch13 实现")

    def disconnect(self) -> None:
        raise NotImplementedError("BackendEndJoyGripper.disconnect 待 Ch13 实现")

    # ---- 状态读取 ----
    def read_state(self) -> dict:
        raise NotImplementedError("BackendEndJoyGripper.read_state 待 Ch13 实现")

    # ---- 指令下发 ----
    def send_position(self, position: float) -> None:
        raise NotImplementedError("BackendEndJoyGripper.send_position 待 Ch13 实现")

    def send_force(self, force: float) -> None:
        raise NotImplementedError("BackendEndJoyGripper.send_force 待 Ch13 实现")

    def send_action(self, action: str) -> None:
        raise NotImplementedError("BackendEndJoyGripper.send_action 待 Ch13 实现")
