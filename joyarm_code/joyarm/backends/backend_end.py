"""``BackendEnd`` —— 末端执行器通信后端抽象基类。

负责**末端执行器**（夹爪 / 灵巧手 / 电磁吸附 / 气压吸附等）的通信；
派生自通用 :class:`~joyarm.backends.backend.Backend`。
具体型号如 :class:`~joyarm.backends.backend_end_joygripper.BackendEndJoyGripper`。
"""
from __future__ import annotations

from abc import abstractmethod

from .backend import Backend

__all__ = ["BackendEnd"]


class BackendEnd(Backend):
    """末端执行器通信后端抽象基类。

    ``send_action`` / ``send_position`` / ``send_force`` 覆盖「离散动作」「位置连续控制」
    「力度连续控制」三类典型末端需求，使灵巧手、吸附等装置都能复用同一抽象。
    """

    @abstractmethod
    def read_state(self) -> dict:
        """读取末端执行器状态快照。

        :return: 状态字典，字段由子类定义。
            例如夹爪：``{"width_mm": float, "force_N": float, "is_grasping": bool}``。
        """

    @abstractmethod
    def send_position(self, position: float) -> None:
        """位置控制（连续量）。

        :param position: 位置目标，语义由子类约定（如夹爪两指间距 mm）。
        """

    @abstractmethod
    def send_force(self, force: float) -> None:
        """力度控制（连续量）。

        :param force: 力度目标，语义由子类约定（如夹持力 N）。
        """

    @abstractmethod
    def send_action(self, action: str) -> None:
        """通用离散动作。

        :param action: 动作名，常见 ``"open"`` / ``"close"``；子类可扩展。
        """
