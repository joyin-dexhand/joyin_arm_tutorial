"""
``JoyArmRebotDMBackend`` —— JoyArm（reBot-DevArm）真机机械臂通信后端。

基于 CAN 总线与达妙（DM）电机驱动机械臂本体（6 个关节电机，ID ``0x01~0x06``）。
末端执行器（夹爪等）的通信不在本类范围，请见
:class:`~joyarm.backends.gripper_backend.GripperBackend` 等末端后端。
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from ..utils.types import ArmState
from .arm_backend import ArmBackend

__all__ = ["JoyArmRebotDMBackend"]

class JoyArmRebotDMBackend(ArmBackend):
    """JoyArm（reBot-DevArm）真机机械臂后端（CAN 总线，电机 ID ``0x01~0x06``）。

    :param can_interface: CAN 接口名（如 ``"can0"``）。
    :param baudrate: 波特率（CAN bps）。
    """

    def __init__(
        self,
        can_interface: str = "can0",
        baudrate: int = 1_000_000,
        **kwargs,
    ):
        self.can_interface = can_interface          # CAN 接口名（Linux 下如 "can0"）
        self.baudrate = baudrate                    # 波特率（bps），1Mbps 是 CAN 常用高速档
        self._kwargs = kwargs
        # 真实 SDK 句柄
        self._handle = None

    # ----------------------------------------------------------
    # 连接 / 使能 / 标定 / 扫描
    # ----------------------------------------------------------
    def connect(self) -> None:
        # 占位：Ch6 实现——打开 CAN 接口、初始化 SDK 句柄，建立与电机的通信链路。
        raise NotImplementedError("JoyArmRebotDMBackend.connect 待 Ch6 实现")

    def disconnect(self) -> None:
        # 占位：Ch6 实现——关闭 CAN 接口、释放资源。
        raise NotImplementedError("JoyArmRebotDMBackend.disconnect 待 Ch6 实现")

    def enable(self, joint: Optional[int] = None) -> None:
        # 占位：Ch6 实现——给电机上电使能（默认失能，使能后才能接收运动指令）。
        raise NotImplementedError("JoyArmRebotDMBackend.enable 待 Ch6 实现")

    def disable(self, joint: Optional[int] = None) -> None:
        # 占位：Ch6 实现——电机失能（断电，可手动转动，用于示教或安全停机）。
        raise NotImplementedError("JoyArmRebotDMBackend.disable 待 Ch6 实现")

    def set_zero(self, joint: Optional[int] = None) -> None:
        # 占位：Ch6 实现——把当前位置标定为零点（标定，每次开机或换装后要做）。
        raise NotImplementedError("JoyArmRebotDMBackend.set_zero 待 Ch6 实现")

    def scan(self) -> List[int]:
        # 占位：Ch6 实现——扫描总线上有哪些电机，返回它们的 ID（应为 0x01~0x06）。
        raise NotImplementedError("JoyArmRebotDMBackend.scan 待 Ch6 实现")

    # ----------------------------------------------------------
    # 状态读取
    # ----------------------------------------------------------
    def read_state(self) -> ArmState:
        # 占位：Ch6 实现——一次性读取所有关节的位置/速度/力矩/温度等，打包成 ArmState。
        raise NotImplementedError("JoyArmRebotDMBackend.read_state 待 Ch6 实现")

    # ----------------------------------------------------------
    # 指令下发
    # ----------------------------------------------------------
    def send_position(self, q: np.ndarray, joint: Optional[int] = None) -> None:
        # 占位：Ch6 实现——把目标关节角打包成 CAN 电文发给电机（位置模式）。
        raise NotImplementedError("JoyArmRebotDMBackend.send_position 待 Ch6 实现")

    def send_velocity(self, dq: np.ndarray, joint: Optional[int] = None) -> None:
        # 占位：Ch6 实现——下发目标关节速度（速度模式）。
        raise NotImplementedError("JoyArmRebotDMBackend.send_velocity 待 Ch6 实现")

    def send_torque(self, tau: np.ndarray, joint: Optional[int] = None) -> None:
        # 占位：Ch6 实现——下发目标关节力矩（力矩模式，最底层）。
        raise NotImplementedError("JoyArmRebotDMBackend.send_torque 待 Ch6 实现")

    def send_mit(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        joint: Optional[int] = None,
    ) -> None:
        # 占位：Ch6 实现——MIT 模式：一次下发目标位置/速度/前馈力矩/两个增益，
        # 电机内部按 τ = kp·(q_des−q) + kd·(dq_des−dq) + tau_ff 自行计算最终力矩。
        raise NotImplementedError("JoyArmRebotDMBackend.send_mit 待 Ch6 实现")
