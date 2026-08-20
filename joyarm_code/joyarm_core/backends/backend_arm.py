"""``BackendArm`` —— 多轴本体通信后端抽象基类。

负责多轴机械臂**本体**（关节电机）的通信；派生自通用 :class:`~joyarm_core.backends.backend.Backend`。
具体型号如 :class:`~joyarm_core.backends.backend_arm_dm.BackendArmDM`。

DM 等模式电机的硬约束：**接收指令前必须先切换到对应控制模式**（:meth:`BackendArm.set_mode`）；
电机按 config 注册，无总线扫描；无力矩指令通道，纯力矩经 MIT（``kp=kd=0``）实现。
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np

from ..utils.types import ArmState, ControlMode
from .backend import Backend

__all__ = ["BackendArm"]


class BackendArm(Backend):
    """多轴本体通信后端抽象基类。

    覆盖关节电机的：连接/断开、使能/失能、模式切换、状态读取、位置/速度/MIT 指令、
    零位标定。
    """

    # ----------------------------------------------------------
    # 使能 / 标定（connect/disconnect 继承自 Backend）
    # ----------------------------------------------------------
    @abstractmethod
    def enable(self, joint: Optional[int] = None) -> None:
        """使能关节电机（默认失能）。

        :param joint: 关节索引；``None`` 表示全部关节。
        """

    @abstractmethod
    def disable(self, joint: Optional[int] = None) -> None:
        """失能关节电机。

        :param joint: 关节索引；``None`` 表示全部关节。
        """

    @abstractmethod
    def set_zero(self, joint: Optional[int] = None) -> None:
        """零位标定。

        :param joint: 关节索引；``None`` 表示全部关节。
        """

    # ----------------------------------------------------------
    # 模式切换
    # ----------------------------------------------------------
    @abstractmethod
    def set_mode(self, mode: ControlMode) -> None:
        """切换本体控制模式（收指令前必须先切到对应模式）。

        ``POSITION → 电机 POS_VEL``、``VELOCITY → 电机 VEL``、``MIT → 电机 MIT``；
        切换所需增益（POS_VEL 闭环参数 / MIT kp·kd）回退 config 对应段。

        :param mode: 目标控制模式（``ControlMode`` 三态）。
        """

    # ----------------------------------------------------------
    # 状态读取
    # ----------------------------------------------------------
    @abstractmethod
    def read_state(self) -> ArmState:
        """读取本体状态快照（关节角/速度/力矩/温度等）。"""

    # ----------------------------------------------------------
    # 指令下发
    # ----------------------------------------------------------
    @abstractmethod
    def send_position(self, q: np.ndarray, joint: Optional[int] = None) -> None:
        """位置指令（实现须按 config ``POS_VEL.vlim`` 限速）。

        :param q: 关节角目标 ``(n,)``，弧度。
        :param joint: 关节索引；``None`` 表示全部关节。
        """

    @abstractmethod
    def send_velocity(self, dq: np.ndarray, joint: Optional[int] = None) -> None:
        """速度指令。

        :param dq: 关节速度目标 ``(n,)``，rad/s。
        """

    @abstractmethod
    def send_mit(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_ff: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        """MIT 阻抗/前馈指令：``τ = kp·(q_des−q) + kd·(dq_des−dq) + tau_ff``。

        :param q: 位置目标 ``(n,)``，弧度。
        :param dq: 速度目标 ``(n,)``，rad/s。
        :param tau_ff: 前馈力矩 ``(n,)``，N·m；纯力矩用 ``kp=kd=0`` + ``tau_ff``。
        :param kp: 位置增益 ``(n,)``；``None`` 回退 config ``MIT.kp``。
        :param kd: 速度阻尼 ``(n,)``；``None`` 回退 config ``MIT.kd``。
        """
