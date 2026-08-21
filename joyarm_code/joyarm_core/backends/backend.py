"""``Backend`` —— 整机硬件通信后端抽象基类（通信层）。

一个后端 = 一台完整设备（多轴本体 + 末端执行器，共用或各用通信总线）；
子类按**电机厂商/型号**派生（结构相同、参数不同），如 :class:`joyarm_core.
backends.backend_dm.BackendDM`。契约方法扁平挂在单类上，以 ``_arm`` / ``_end``
后缀区分本体与末端两组。

配置结构即代码结构（yaml ``backend:`` 段，``name`` 选型键经 ``REGISTRY`` 解析）::

    backend:
      name: backend_dm          # 选型键（JoyArm 弹出后按 REGISTRY 构建子类）
      arm: {channel, protocol, baud_rate, control_rate, joints}   # 本体子段
      end: {channel, protocol, baud_rate, control_rate, joints}   # 末端子段

DM 等模式电机的硬约束：**接收指令前必须先切换到对应控制模式**
（:meth:`Backend.set_mode_arm` / :meth:`Backend.set_mode_end`）；电机按 config
注册，无总线扫描；无力矩指令通道，纯力矩经 MIT（``kp=kd=0``）实现。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..utils.types import ArmState, ControlMode

__all__ = ["Backend"]


class Backend(ABC):
    """整机硬件通信后端抽象基类（本体 + 末端一体）。

    ``_arm`` 族方法的 ``joint`` 形参：关节索引，``None`` 表示全部关节。

    :param cfg: yaml ``backend:`` 段字典（``name`` 已由 JoyArm 弹出），含
        ``arm:`` / ``end:`` 两个子段，各含 ``channel`` / ``protocol`` /
        ``baud_rate`` / ``control_rate`` / ``joints``（电机配置列表）。
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    # ----------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------
    @abstractmethod
    def connect(self) -> None:
        """建立通信链路（本体 + 末端）。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开通信链路（失能电机 → 停接收线程 → 关总线）。"""

    # ----------------------------------------------------------
    # 本体（关节电机）：_arm 后缀
    # ----------------------------------------------------------
    @abstractmethod
    def enable_arm(self, joint: Optional[int] = None) -> None:
        """使能本体关节电机（默认失能）。"""

    @abstractmethod
    def disable_arm(self, joint: Optional[int] = None) -> None:
        """失能本体关节电机。"""

    @abstractmethod
    def set_zero_arm(self, joint: Optional[int] = None) -> None:
        """本体零位标定（先失能，反馈无故障后再标零）。"""

    @abstractmethod
    def set_mode_arm(self, mode: ControlMode) -> None:
        """切换本体控制模式（收指令前必须先切到对应模式）。

        ``POSITION → 电机 POS_VEL``、``VELOCITY → 电机 VEL``、``MIT → 电机 MIT``；
        切换所需增益（POS_VEL 闭环参数 / MIT kp·kd）回退 config 对应段。

        :param mode: 目标控制模式（``ControlMode`` 三态）。
        """

    @abstractmethod
    def read_state_arm(self) -> ArmState:
        """读取本体状态快照（关节角/速度/力矩/温度等）。"""

    @abstractmethod
    def send_position_arm(self, q: np.ndarray, joint: Optional[int] = None) -> None:
        """本体位置指令（实现须按 config ``POS_VEL.vlim`` 限速）。

        :param q: 关节角目标 ``(n,)``，弧度。
        """

    @abstractmethod
    def send_velocity_arm(self, dq: np.ndarray, joint: Optional[int] = None) -> None:
        """本体速度指令。

        :param dq: 关节速度目标 ``(n,)``，rad/s。
        """

    @abstractmethod
    def send_mit_arm(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_ff: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        """本体 MIT 阻抗/前馈指令：``τ = kp·(q_des−q) + kd·(dq_des−dq) + tau_ff``。

        :param q: 位置目标 ``(n,)``，弧度。
        :param dq: 速度目标 ``(n,)``，rad/s。
        :param tau_ff: 前馈力矩 ``(n,)``，N·m；纯力矩用 ``kp=kd=0`` + ``tau_ff``。
        :param kp: 位置增益 ``(n,)``；``None`` 回退 config ``MIT.kp``。
        :param kd: 速度阻尼 ``(n,)``；``None`` 回退 config ``MIT.kd``。
        """

    # ----------------------------------------------------------
    # 末端（执行器电机）：_end 后缀
    # ----------------------------------------------------------
    @abstractmethod
    def enable_end(self) -> None:
        """使能末端执行器电机。"""

    @abstractmethod
    def disable_end(self) -> None:
        """失能末端执行器电机。"""

    @abstractmethod
    def set_zero_end(self) -> None:
        """末端零位标定（先失能，反馈无故障后再标零）。"""

    @abstractmethod
    def set_mode_end(self, mode: ControlMode) -> None:
        """切换末端控制模式（语义同 :meth:`set_mode_arm`，作用于末端电机）。

        :param mode: 目标控制模式（``ControlMode`` 三态）。
        """

    @abstractmethod
    def read_state_end(self) -> dict:
        """读取末端状态快照。

        :return: 状态字典，字段由子类定义。
            例如夹爪：``{"width_mm": float, "force_N": float, "is_grasping": bool}``。
        """

    @abstractmethod
    def send_position_end(self, position: float) -> None:
        """末端位置控制（连续量）。

        :param position: 位置目标，语义由子类约定（如夹爪两指间距 mm）。
        """

    @abstractmethod
    def send_force_end(self, force: float) -> None:
        """末端力度控制（连续量；DM 夹爪经 MIT 近似实现）。

        :param force: 力度目标，语义由子类约定（如夹持力 N）。
        """

    @abstractmethod
    def send_action_end(self, action: str) -> None:
        """末端通用离散动作。

        :param action: 动作名，常见 ``"open"`` / ``"close"``；子类可扩展。
        """
