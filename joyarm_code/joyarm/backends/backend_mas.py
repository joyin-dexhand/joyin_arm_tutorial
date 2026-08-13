"""``BackendMas`` —— 多轴本体（MAS）通信后端抽象基类。

负责多轴机械臂**本体**（关节电机）的通信；派生自通用 :class:`~joyarm.backends.backend.Backend`。
具体型号如 :class:`~joyarm.backends.backend_mas_rebot_dm.BackendMasRebotDM`。
"""
from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional

import numpy as np

from ..utils.types import ArmState
from .backend import Backend

__all__ = ["BackendMas"]


class BackendMas(Backend):
    """多轴本体通信后端抽象基类。

    覆盖关节电机的：连接/断开、使能/失能、状态读取、位置/速度/力矩/MIT 指令、
    零位标定、总线扫描。
    """

    # ----------------------------------------------------------
    # 使能 / 标定 / 扫描（connect/disconnect 继承自 Backend）
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

    @abstractmethod
    def scan(self) -> List[int]:
        """扫描总线电机 ID。

        :return: 已识别电机 ID 列表。
        """

    # ----------------------------------------------------------
    # 状态读取
    # ----------------------------------------------------------
    @abstractmethod
    def read_state(self) -> ArmState:
        """读取本体状态快照（关节角/速度/力矩/温度/电压/电流等）。"""

    # ----------------------------------------------------------
    # 指令下发
    # ----------------------------------------------------------
    @abstractmethod
    def send_position(self, q: np.ndarray, joint: Optional[int] = None) -> None:
        """位置指令。

        :param q: 关节角目标 ``(n,)``，弧度。
        :param joint: 关节索引；``None`` 表示全部关节。
        """

    @abstractmethod
    def send_velocity(self, dq: np.ndarray, joint: Optional[int] = None) -> None:
        """速度指令。

        :param dq: 关节速度目标 ``(n,)``，rad/s。
        """

    @abstractmethod
    def send_torque(self, tau: np.ndarray, joint: Optional[int] = None) -> None:
        """力矩指令。

        :param tau: 关节力矩目标 ``(n,)``，N·m。
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
        """MIT 阻抗/前馈指令：``τ = kp·(q_des−q) + kd·(dq_des−dq) + tau_ff``。"""
