"""``BackendEndGripper`` —— JoyArm 两指夹爪真机后端。

USB-CAN 串口桥驱动夹爪电机（``gripper``，ID ``0x07``，4310）。
派生自 :class:`~joyarm_core.backends.backend_end.BackendEnd`，Ch13 实现；当前为占位。
是 :class:`~joyarm_core.joyarms.joyarm.JoyArm` 末端执行器的通信载体（``JoyArm.backend_end``）。

**共享总线约束**：与 ``BackendArmDM`` 同 channel（同一条串口/CAN 总线），必须复用
同一共享总线对象（一个串口句柄 + 一个 RX 线程 + 一个 TX 锁，按电机/反馈 ID 分发；
本后端为 0x07/0x17），不可各自打开串口。DM 通讯协议手写实现，不依赖 motorbridge。

Ch13 实现约束（依已跑通平台标定）：夹爪电机与本体关节同构（达妙 DM）::

    send_position(pos)          → POS_VEL 帧（pos + vlim，vlim 取 config）
    send_action("open"/"close") → 预设开/合位置常量 + send_position
    send_force(force)           → 经 MIT 近似（调 tau_ff/kp 模拟夹持力；
                                  DM 夹爪无力控通道，语义为近似值）
    read_state()                → 按反馈 ID（0x17）请求帧 + 解码，组装
                                  width_mm/is_grasping 等字段
"""
from __future__ import annotations

from typing import List, Optional

from .backend_end import BackendEnd

__all__ = ["BackendEndGripper"]


class BackendEndGripper(BackendEnd):
    """JoyArm 两指夹爪真机通信后端（达妙 DM 4310，USB-CAN 串口桥）。

    :param channel: 串口设备（与本体同 ``/dev/ttyACM0``）。
    :param rate: 反馈/下发频率 Hz。
    :param joints: 夹爪电机配置字典列表（同 backend_arm.joints 结构，name="gripper"）。
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

    # ---- 连接 ----
    def connect(self) -> None:
        raise NotImplementedError("BackendEndGripper.connect 待 Ch13 实现")

    def disconnect(self) -> None:
        raise NotImplementedError("BackendEndGripper.disconnect 待 Ch13 实现")

    # ---- 状态读取 ----
    def read_state(self) -> dict:
        raise NotImplementedError("BackendEndGripper.read_state 待 Ch13 实现")

    # ---- 指令下发 ----
    def send_position(self, position: float) -> None:
        raise NotImplementedError("BackendEndGripper.send_position 待 Ch13 实现")

    def send_force(self, force: float) -> None:
        raise NotImplementedError("BackendEndGripper.send_force 待 Ch13 实现")

    def send_action(self, action: str) -> None:
        raise NotImplementedError("BackendEndGripper.send_action 待 Ch13 实现")
