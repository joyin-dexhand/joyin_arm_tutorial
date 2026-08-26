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

接口按功能分类：生命周期（connect/disconnect）、使能失能（enable/disable/
set_zero）、状态读取（read_state）、模式切换（set_mode）、指令下发（send_*）、
电机参数读写（read_param/write_param）。收录多厂商关节电机的通用功能。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..utils.types import ArmState, ControlMode

__all__ = ["Backend"]


class Backend(ABC):
    """整机硬件通信后端抽象基类（本体 + 末端一体）。

    ``joint`` 形参：电机索引，``None`` 表示全部电机。

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
    def set_mode_arm(self, mode: ControlMode = ControlMode.POSITION,
                     joint: Optional[int] = None) -> None:
        """切换本体控制模式（收指令前必须先切到对应模式；默认位置模式）。

        ``POSITION → 电机 POS_VEL``、``VELOCITY → 电机 VEL``、``MIT → 电机 MIT``；
        切换所需增益（POS_VEL 闭环参数 / MIT kp·kd）回退 config 对应段。

        :param mode: 目标控制模式，默认 ``ControlMode.POSITION``（电机 POS_VEL）。
        :param joint: 关节索引，``None`` 表示全部；仅对未处于目标模式的电机补切。
        """

    @abstractmethod
    def read_state_arm(self, joint: Optional[int] = None) -> ArmState:
        """读取本体状态快照（关节角/速度/力矩/使能与错误状态等）。

        :param joint: 关节索引，``None`` 表示全部（数组 ``(n,)``）；指定关节时
            各数组仅含该关节（长度 1）。
        :return: :class:`ArmState` 快照。
        """

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

    @abstractmethod
    def read_param_arm(self, key: str, joint: Optional[int] = None):
        """读本体关节电机参数。

        :param key: 参数名（字符串，语义由子类映射到厂商寄存器，
            如 DM 的 ``"pos_kp"`` → 寄存器 27）。
        :param joint: 关节索引，``None`` 表示全部电机。
        :return: 指定 ``joint`` 时返回该电机参数值（float 或 int）；
            ``joint=None`` 时返回逐电机参数值列表。
        """

    @abstractmethod
    def write_param_arm(self, key: str, value, joint: Optional[int] = None,
                        persist: bool = False) -> None:
        """写本体关节电机参数。

        :param key: 参数名（语义由子类定义）。
        :param value: 参数值，标量（作用于所选全部电机）或与所选电机数
            一致的列表（逐电机）。
        :param joint: 关节索引，``None`` 表示全部电机。
        :param persist: ``True`` 时写入并持久化到非易失存储（如 DM 需先失能再存闪存）。
        """

    # ----------------------------------------------------------
    # 末端（执行器电机组）：_end 后缀（可多电机，如灵巧手）
    # ----------------------------------------------------------
    @abstractmethod
    def enable_end(self, joint: Optional[int] = None) -> None:
        """使能末端执行器电机（``joint=None`` 全部）。"""

    @abstractmethod
    def disable_end(self, joint: Optional[int] = None) -> None:
        """失能末端执行器电机（``joint=None`` 全部）。"""

    @abstractmethod
    def set_zero_end(self, joint: Optional[int] = None) -> None:
        """末端零位标定（先失能，反馈无故障后再标零；``joint=None`` 全部）。"""

    @abstractmethod
    def set_mode_end(self, mode: ControlMode = ControlMode.POSITION,
                     joint: Optional[int] = None) -> None:
        """切换末端控制模式（语义同 :meth:`set_mode_arm`，默认位置模式）。

        :param mode: 目标控制模式，默认 ``ControlMode.POSITION``（电机 POS_VEL）。
        :param joint: 末端电机索引，``None`` 表示全部。
        """

    @abstractmethod
    def read_state_end(self, joint: Optional[int] = None) -> dict:
        """读取末端状态快照（值为所选电机的逐电机序列）。

        :param joint: 末端电机索引，``None`` 表示全部。
        :return: 状态字典，字段由子类定义；值为逐电机序列（单电机末端为
            单元素序列）。例如 DM 夹爪：``{"q": [...], "dq": [...], "tau": [...],
            "enabled": [...], "error": [...], "comm_ok": [...]}``。
        """

    @abstractmethod
    def send_position_end(self, position, joint: Optional[int] = None) -> None:
        """末端位置控制（连续量）。

        :param position: 位置目标，标量（作用于所选全部电机）或与所选电机数
            一致的序列；单位语义由子类约定（如夹爪电机弧度）。
        :param joint: 末端电机索引，``None`` 表示全部。
        """

    @abstractmethod
    def send_force_end(self, force, joint: Optional[int] = None) -> None:
        """末端力度控制（连续量；DM 夹爪经 MIT 近似实现）。

        :param force: 力度目标，标量（作用于所选全部电机）或与所选电机数一致
            的序列；语义由子类约定（如夹持力 N）。
        :param joint: 末端电机索引，``None`` 表示全部。
        """

    @abstractmethod
    def send_action_end(self, action: str, joint: Optional[int] = None) -> None:
        """末端离散动作（预设目标由子类按 config 定义，作用于所选电机）。

        :param action: 动作名，常见 ``"open"`` / ``"close"``；子类可扩展。
        :param joint: 末端电机索引，``None`` 表示全部。
        """

    @abstractmethod
    def read_param_end(self, key: str, joint: Optional[int] = None):
        """读末端电机参数。

        :param key: 参数名（语义由子类映射到厂商寄存器）。
        :param joint: 末端电机索引，``None`` 表示全部电机。
        :return: 指定 ``joint`` 时返回该电机参数值（float 或 int）；
            ``joint=None`` 时返回逐电机参数值列表。
        """

    @abstractmethod
    def write_param_end(self, key: str, value, joint: Optional[int] = None,
                        persist: bool = False) -> None:
        """写末端电机参数（``persist=True`` 持久化到非易失存储）。

        :param key: 参数名（语义由子类定义）。
        :param value: 参数值，标量（作用于所选全部电机）或与所选电机数
            一致的列表（逐电机）。
        :param joint: 末端电机索引，``None`` 表示全部电机。
        :param persist: ``True`` 时写入并持久化到非易失存储。
        """
