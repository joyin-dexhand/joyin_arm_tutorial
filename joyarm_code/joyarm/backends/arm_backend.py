"""
机械臂真机通信后端抽象基类（diandiandian通信层）。

负责**机械臂本体**（6 个关节电机，ID ``0x01~0x06``）的通信。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..utils.types import ArmState

__all__ = ["ArmBackend"]


class ArmBackend(ABC):
    """机械臂真机通信后端抽象基类。

    所有机械臂真机通信后端都必须继承此类，并实现所有抽象方法。
    """

    # ----------------------------------------------------------
    # 连接 / 使能 / 标定 / 扫描
    # ----------------------------------------------------------
    @abstractmethod
    def connect(self) -> None:
        """建立通信链路（USB-CAN 等）。未连接时自动监测端口并连接，并打印连接成功信息"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开通信链路，打印断开连接信息"""

    @abstractmethod
    def enable(self, joint: Optional[int] = None) -> None:
        """使能关节电机（默认失能，见快速上手 §7.1）。

        :param joint: 关节索引；``None`` 表示全部关节。
        """

    @abstractmethod
    def disable(self, joint: Optional[int] = None) -> None:
        """失能关节电机。

        :param joint: 关节索引；``None`` 表示全部关节。
        """

    @abstractmethod
    def set_zero(self, joint: Optional[int] = None) -> None:
        """零位标定（快速上手 §7.2）。

        :param joint: 关节索引；``None`` 表示全部关节。
        """

    @abstractmethod
    def scan(self) -> List[int]:
        """扫描总线电机 ID（默认 ``0x01~0x06``）。

        :return: 已识别电机 ID 列表。
        """

    # ----------------------------------------------------------
    # 状态读取
    # ----------------------------------------------------------
    @abstractmethod
    def read_state(self) -> ArmState:
        """读取完整状态快照

        :return: :class:`joyarm.utils.types.ArmState` 聚合快照。
        """

    # ----------------------------------------------------------
    # 指令下发（按 ControlMode）
    # ----------------------------------------------------------
    @abstractmethod
    def send_position(self, q: np.ndarray, joint: Optional[int] = None) -> None:
        """位置模式指令（§6.1）。

        :param q: ``(n,)`` 目标关节角，弧度。
        :param joint: 关节索引；``None`` 表示全部关节。
        """

    @abstractmethod
    def send_velocity(self, dq: np.ndarray, joint: Optional[int] = None) -> None:
        """速度模式指令。

        :param dq: ``(n,)`` 目标关节速度，弧度/秒。
        :param joint: 关节索引；``None`` 表示全部关节。
        """

    @abstractmethod
    def send_torque(self, tau: np.ndarray, joint: Optional[int] = None) -> None:
        """力矩模式指令（Ch9 无传感器分支）。

        :param tau: ``(n,)`` 目标关节力矩，N·m。
        :param joint: 关节索引；``None`` 表示全部关节。
        """

    @abstractmethod
    def send_mit(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        joint: Optional[int] = None,
    ) -> None:
        """MIT 阻抗/前馈模式指令（对应 :attr:`ControlMode.MIT`）。

        Cheetah 风格：电机内部按
        ``τ = kp·(q_des−q) + kd·(dq_des−dq) + tau_ff`` 计算最终力矩。

        :param q: ``(n,)`` 目标位置，弧度。
        :param dq: ``(n,)`` 目标速度，弧度/秒。
        :param tau_ff: ``(n,)`` 前馈力矩，N·m。
        :param kp: ``(n,)`` 位置增益。
        :param kd: ``(n,)`` 速度增益。
        :param joint: 关节索引；``None`` 表示全部关节。
        """
