"""``BackendArmDM`` —— JoyArm（joyarm_dm，达妙 DM 电机）多轴本体真机后端。

USB-CAN 串口桥（如 ``/dev/ttyACM0``）驱动 6 个关节电机
（joint1~3 = 4340P，joint4~6 = 4310）。派生自
:class:`~joyarm_core.backends.backend_arm.BackendArm`，Ch6 实现；当前为占位。
DM 通讯协议**手写实现**（桥串口帧封装 + DM 电机 CAN 帧编解码），不依赖 motorbridge。

**共享总线约束**：本后端与 ``BackendEndGripper`` 共用同一条串口（同 channel 同一条
CAN 总线，电机 ID 0x01~0x07 不重叠）。同 channel 必须只有**一个串口属主**——一个
串口句柄 + 一个 RX 读线程 + 一个 TX 锁；两后端按 channel 复用同一共享总线对象
（引用计数：首个 connect 打开并起 RX 线程，引用归零才真正关闭）。否则双开串口会
竞争同一接收缓冲，应答帧被随机瓜分，协议解析静默错乱。

Ch6 实现约束（依已跑通平台标定的协议级行为）::

    connect()     经共享总线（按 channel 复用/新建）注册本后端电机
                  （motor_id/feedback_id/model 取 config）
    set_mode()    POS_VEL：先写电机寄存器 25~28（vel_kp/vel_ki/pos_kp/pos_ki）
                  再发模式切换帧并确认；VEL / MIT：仅切模式
    read_state()  两段式反馈：按反馈 ID 逐电机发请求帧 → RX 线程收应答、
                  解码（pos/vel/torq/温度/状态字）填充状态槽
    set_zero()    先失能，轮询反馈至状态字==0（无故障）再发标零帧（0xFE）
    disconnect()  失能本后端电机 → 释放共享总线引用（归零才关串口）
    send_position()  → POS_VEL 帧（q + vlim，vlim 取 config POS_VEL 段）
    send_velocity()  → VEL 帧（dq）
    send_mit()       → MIT 帧（q, dq, kp, kd, tau_ff 打包），kp/kd None→config
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..utils.types import ArmState, ControlMode
from .backend_arm import BackendArm

__all__ = ["BackendArmDM"]


class BackendArmDM(BackendArm):
    """JoyArm（joyarm_dm）多轴本体真机通信后端（达妙 DM，USB-CAN 串口桥）。

    :param channel: 串口设备（如 ``"/dev/ttyACM0"``）。
    :param rate: 反馈/下发频率 Hz。
    :param joints: 每关节电机配置字典列表（``motor_id``/``feedback_id``/``model``/
        ``vendor``/``MIT``/``POS_VEL``，对应 yaml ``backend_arm.joints``）。
    """

    def __init__(
        self,
        channel: str = "/dev/ttyACM0",
        rate: int = 500,
        joints: Optional[List[dict]] = None,
    ):
        self.channel = channel
        self.rate = rate
        self.joints = joints or []

    # ---- 连接 / 使能 / 标定 ----
    def connect(self) -> None:
        raise NotImplementedError("BackendArmDM.connect 待 Ch6 实现")

    def disconnect(self) -> None:
        raise NotImplementedError("BackendArmDM.disconnect 待 Ch6 实现")

    def enable(self, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendArmDM.enable 待 Ch6 实现")

    def disable(self, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendArmDM.disable 待 Ch6 实现")

    def set_zero(self, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendArmDM.set_zero 待 Ch6 实现")

    # ---- 模式切换 ----
    def set_mode(self, mode: ControlMode) -> None:
        raise NotImplementedError("BackendArmDM.set_mode 待 Ch6 实现")

    # ---- 状态读取 ----
    def read_state(self) -> ArmState:
        raise NotImplementedError("BackendArmDM.read_state 待 Ch6 实现")

    # ---- 指令下发 ----
    def send_position(self, q: np.ndarray, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendArmDM.send_position 待 Ch6 实现")

    def send_velocity(self, dq: np.ndarray, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendArmDM.send_velocity 待 Ch6 实现")

    def send_mit(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_ff: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        raise NotImplementedError("BackendArmDM.send_mit 待 Ch6 实现")
