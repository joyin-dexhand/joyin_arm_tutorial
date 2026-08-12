"""
末端执行器真机通信后端抽象基类（通信层）。

本基类负责**末端执行器**（夹爪 / 灵巧手 / 电磁吸附 / 气压吸附等）的通信，
与机械臂本体（见 :class:`~joyarm.backends.arm_backend.ArmBackend`）解耦——
二者各成体系、互不继承，便于未来灵活支持多类末端执行器。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["EndBackend"]


class EndBackend(ABC):
    """末端执行器真机通信后端抽象基类。

    所有末端执行器真机通信后端（如 ``GripperBackend``）都必须继承此类，
    并实现所有抽象方法。

    接口设计保持足够通用：``send_action`` / ``send_position`` / ``send_force``
    覆盖了「离散动作」「位置连续控制」「力度连续控制」三类典型末端执行需求，
    使未来灵巧手、电磁/气压吸附等装置都能复用同一抽象。
    """

    # ----------------------------------------------------------
    # 连接 / 断开
    # ----------------------------------------------------------
    @abstractmethod
    def connect(self) -> None:
        """建立末端执行器通信链路。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开末端执行器通信链路。"""

    # ----------------------------------------------------------
    # 状态读取
    # ----------------------------------------------------------
    @abstractmethod
    def read_state(self) -> dict:
        """读取末端执行器状态快照。

        :return: 状态字典，具体字段由子类定义。
            例如夹爪：``{"width_mm": float, "force_N": float, "is_grasping": bool}``。
        """

    # ----------------------------------------------------------
    # 指令下发
    # ----------------------------------------------------------
    @abstractmethod
    def send_position(self, position: float) -> None:
        """位置控制（连续量）。

        :param position: 位置目标，语义由子类约定。
            例如夹爪为两指间距（毫米），灵巧手为某指弯曲角度。
        """

    @abstractmethod
    def send_force(self, force: float) -> None:
        """力度控制（连续量）。

        :param force: 力度目标，语义由子类约定。
            例如夹爪为夹持力（牛顿）。
        """

    @abstractmethod
    def send_action(self, action: str) -> None:
        """通用离散动作（便于多类末端复用）。

        :param action: 动作名，常见取值如 ``"open"`` / ``"close"``。
            子类可扩展自有动作集合（如灵巧手的 ``"pinch"`` / ``"release"``）。
        """
