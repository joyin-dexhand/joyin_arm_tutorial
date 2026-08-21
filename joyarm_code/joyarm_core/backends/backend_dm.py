"""``BackendDM`` —— JoyArm（joyarm_dm，达妙 DM 电机）整机真机后端。

USB-CAN 串口桥（如 ``/dev/ttyACM0``）驱动整机 7 个 DM 电机：本体 6 关节
（joint1~3 = 4340P，joint4~6 = 4310）+ 两指夹爪（``gripper``，4310）。
派生自 :class:`~joyarm_core.backends.backend.Backend`，本体+总线待 Ch6、末端
语义待 Ch13 实现；当前为占位。DM 通讯协议**手写实现**（桥串口帧封装 + DM 电机
CAN 帧编解码），不依赖 motorbridge。

**总线共享规则**（config ``arm.channel`` 与 ``end.channel`` 决定）::

    channel 相同 → 共享单总线：一个串口句柄 + 一个 RX 读线程 + 一个 TX 锁，
                   end 借用 arm 的总线（避免双开串口导致应答帧被随机瓜分）；
    channel 不同 → 两条独立总线，互不干扰（支持末端独立通道的硬件形态）。

Ch6 实现约束（依已跑通平台标定的协议级行为）::

    connect()     按 arm/end 子段打开串口（baud_rate，如 921600）+ 起 RX 线程，
                  注册全部电机（motor_id/feedback_id/model 取 config）
    set_mode_arm/end()  POS_VEL：先写电机寄存器 25~28（vel_kp/vel_ki/pos_kp/
                  pos_ki）再发模式切换帧并确认；VEL / MIT：仅切模式
    read_state_arm/end()  两段式反馈：按反馈 ID 逐电机发请求帧 → RX 线程收
                  应答、解码（pos/vel/torq/温度/状态字）填充状态槽
    set_zero_arm/end()    先失能，轮询反馈至状态字==0（无故障）再发标零帧
    disconnect()  失能电机 → 停 RX 线程 → 关串口（共享总线只关一次）
    send_position_arm()   → POS_VEL 帧（q + vlim，vlim 取 config POS_VEL 段）
    send_velocity_arm()   → VEL 帧（dq）
    send_mit_arm()        → MIT 帧（q, dq, kp, kd, tau_ff 打包），kp/kd None→config

Ch13 实现约束（末端语义）::

    send_action_end("open"/"close") → 预设开/合位置常量 + send_position_end
    send_force_end(force)           → 经 MIT 近似（调 tau_ff/kp 模拟夹持力；
                                      DM 夹爪无力控通道，语义为近似值）
    read_state_end()                → 两段式反馈同上，组装 width_mm/is_grasping
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.types import ArmState, ControlMode
from .backend import Backend

__all__ = ["BackendDM"]


class BackendDM(Backend):
    """JoyArm（joyarm_dm）整机真机通信后端（达妙 DM，USB-CAN 串口桥）。

    :param cfg: yaml ``backend:`` 段字典（``name`` 已由 JoyArm 弹出），含
        ``arm:`` / ``end:`` 子段（``channel`` / ``protocol`` / ``baud_rate`` /
        ``control_rate`` / ``joints``，joints 含 ``motor_id``/``feedback_id``/
        ``model``/``vendor``/``MIT``/``POS_VEL``）。
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        arm = cfg.get("arm", {}) or {}
        end = cfg.get("end", {}) or {}
        self._arm_cfg = arm
        self._end_cfg = end
        # 同 channel → end 借用 arm 总线（实现期决定共享或独立开总线）
        self._shared_bus = bool(end) and arm.get("channel") == end.get("channel")

    # ---- 生命周期 ----
    def connect(self) -> None:
        raise NotImplementedError("BackendDM.connect 待 Ch6 实现")

    def disconnect(self) -> None:
        raise NotImplementedError("BackendDM.disconnect 待 Ch6 实现")

    # ---- 本体：_arm ----
    def enable_arm(self, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendDM.enable_arm 待 Ch6 实现")

    def disable_arm(self, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendDM.disable_arm 待 Ch6 实现")

    def set_zero_arm(self, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendDM.set_zero_arm 待 Ch6 实现")

    def set_mode_arm(self, mode: ControlMode) -> None:
        raise NotImplementedError("BackendDM.set_mode_arm 待 Ch6 实现")

    def read_state_arm(self) -> ArmState:
        raise NotImplementedError("BackendDM.read_state_arm 待 Ch6 实现")

    def send_position_arm(self, q: np.ndarray, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendDM.send_position_arm 待 Ch6 实现")

    def send_velocity_arm(self, dq: np.ndarray, joint: Optional[int] = None) -> None:
        raise NotImplementedError("BackendDM.send_velocity_arm 待 Ch6 实现")

    def send_mit_arm(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_ff: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        raise NotImplementedError("BackendDM.send_mit_arm 待 Ch6 实现")

    # ---- 末端：_end ----
    def enable_end(self) -> None:
        raise NotImplementedError("BackendDM.enable_end 待 Ch13 实现")

    def disable_end(self) -> None:
        raise NotImplementedError("BackendDM.disable_end 待 Ch13 实现")

    def set_zero_end(self) -> None:
        raise NotImplementedError("BackendDM.set_zero_end 待 Ch13 实现")

    def set_mode_end(self, mode: ControlMode) -> None:
        raise NotImplementedError("BackendDM.set_mode_end 待 Ch13 实现")

    def read_state_end(self) -> dict:
        raise NotImplementedError("BackendDM.read_state_end 待 Ch13 实现")

    def send_position_end(self, position: float) -> None:
        raise NotImplementedError("BackendDM.send_position_end 待 Ch13 实现")

    def send_force_end(self, force: float) -> None:
        raise NotImplementedError("BackendDM.send_force_end 待 Ch13 实现")

    def send_action_end(self, action: str) -> None:
        raise NotImplementedError("BackendDM.send_action_end 待 Ch13 实现")
