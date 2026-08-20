"""``Backend`` —— 所有硬件通信后端的抽象根（通信层）。

统一各类硬件（多轴本体 / 末端执行器 / …）的通信入口，仅约束最通用的
「连接 / 断开 / 读状态」契约；具体指令接口由硬件类型层扩展。
继承层次见 :mod:`joyarm_core.backends`。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["Backend"]


class Backend(ABC):
    """所有硬件通信后端的抽象根。

    子类先按**硬件类型**派生（:class:`~joyarm_core.backends.backend_arm.BackendArm` /
    :class:`~joyarm_core.backends.backend_end.BackendEnd`），
    再按**具体型号**派生（如 ``BackendArmDM``）。
    """

    @abstractmethod
    def connect(self) -> None:
        """建立通信链路。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开通信链路。"""

    @abstractmethod
    def read_state(self):
        """读取硬件状态快照（返回类型由子类约定）。"""
