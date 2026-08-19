"""``BackendMasRebotDM`` —— JoyArm（reBot-DevArm，达妙 DM 电机）多轴本体真机后端。

基于 CAN 总线驱动 6 个关节电机（ID ``0x01~0x06``）。派生自
:class:`~joyarm_core.backends.backend_mas.BackendMas`，Ch6 实现；当前为占位。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..utils.types import ArmState
from .backend_mas import BackendMas

__all__ = ["BackendMasRebotDM"]


class BackendMasRebotDM(BackendMas):
    """JoyArm（reBot-DevArm）多轴本体真机通信后端（CAN 总线，电机 ID ``0x01~0x06``）。

    :param can_interface: CAN 接口名（如 ``"can0"``）。
    :param baudrate: 波特率（CAN bps）。
    :param motor_ids: 6 个关节电机 ID 列表（默认 ``[1,2,3,4,5,6]``）。
    """

    def __init__(
        self,
        can_interface: str = "can0",
        baudrate: int = 1_000_000,
        motor_ids: Optional[List[int]] = None,
    ):
        self.can_interface = can_interface
        self.baudrate = baudrate
        self.motor_ids = motor_ids or [1, 2, 3, 4, 5, 6]

    # ---- 连接 / 使能 / 标定 / 扫描 ----
    def connect(self) -> None:
        raise NotImplementedError("BackendMasRebotDM.connect 待 Ch6 实现")

    def disconnect(self) -> None:
        raise NotImplementedError("BackendMasRebotDM.disconnect 待 Ch6 实现")

    def enable(self, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendMasRebotDM.enable 待 Ch6 实现")

    def disable(self, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendMasRebotDM.disable 待 Ch6 实现")

    def set_zero(self, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendMasRebotDM.set_zero 待 Ch6 实现")

    def scan(self) -> List[int]:
        raise NotImplementedError("BackendMasRebotDM.scan 待 Ch6 实现")

    # ---- 状态读取 ----
    def read_state(self) -> ArmState:
        raise NotImplementedError("BackendMasRebotDM.read_state 待 Ch6 实现")

    # ---- 指令下发 ----
    def send_position(self, q: np.ndarray, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendMasRebotDM.send_position 待 Ch6 实现")

    def send_velocity(self, dq: np.ndarray, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendMasRebotDM.send_velocity 待 Ch6 实现")

    def send_torque(self, tau: np.ndarray, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendMasRebotDM.send_torque 待 Ch6 实现")

    def send_mit(
        self, q, dq, tau_ff, kp, kd, joint: Optional[int] = None
    ) -> None:
        raise NotImplementedError("BackendMasRebotDM.send_mit 待 Ch6 实现")
