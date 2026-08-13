"""``End`` 基类 —— 末端执行器（End-Effector：夹爪 / 灵巧手 / 吸附 …）。

``End`` 持有末端通信后端 ``backend_end``（由 ``Arm`` 组合时按 ``configs/*.yaml``
的 ``backend_end`` 段实例化传入）。``connected=False``（默认）时执行类方法 raise，
``connect()`` 后可用。
"""
from __future__ import annotations

from typing import Optional

from ..backends.backend_end import BackendEnd  # noqa: E402  (type-only, no cycle)

__all__ = ["End"]


class End:
    """末端执行器基类。

    :param backend_cls: 末端通信后端**类**（如 :class:`~joyarm.backends.backend_end_joygripper.BackendEndJoyGripper`）。
    :param backend_params: 传给 ``backend_cls`` 的参数字典（来自 config 的 ``backend_end`` 段）。
    """

    def __init__(
        self,
        backend_cls: Optional[type] = None,
        backend_params: Optional[dict] = None,
    ):
        self.backend_end: Optional[BackendEnd] = (
            backend_cls(**(backend_params or {})) if backend_cls is not None else None
        )
        self.connected: bool = False

    # ----------------------------------------------------------
    # 连接
    # ----------------------------------------------------------
    def connect(self) -> None:
        """连接末端真机（委托 ``backend_end.connect()``）。"""
        if self.backend_end is not None:
            self.backend_end.connect()
        self.connected = True

    def disconnect(self) -> None:
        """断开末端真机。"""
        if self.backend_end is not None:
            self.backend_end.disconnect()
        self.connected = False

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"{'connected' if self.connected else 'offline'})"
        )

    # ----------------------------------------------------------
    # 状态读取
    # ----------------------------------------------------------
    def get_state(self) -> dict:
        """读取末端状态（如夹爪 ``{"width_mm","force_N","is_grasping"}``）。

        :raises RuntimeError: 未连接真机时抛出。
        """
        if not self.connected:
            raise RuntimeError("End 未连接真机（离线）；请先 connect()。")
        return self.backend_end.read_state()

    # ----------------------------------------------------------
    # 指令（夹爪语义；其它末端子类可扩展）
    # ----------------------------------------------------------
    def open(self) -> None:
        """张开到最大（默认行程/力度）。"""
        if not self.connected:
            raise RuntimeError("End 未连接真机（离线）；请先 connect()。")
        self.backend_end.send_action("open")

    def close(self) -> None:
        """闭合（夹到默认力度即停）。"""
        if not self.connected:
            raise RuntimeError("End 未连接真机（离线）；请先 connect()。")
        self.backend_end.send_action("close")

    def set_position(self, position: float) -> None:
        """位置控制（如两指间距 mm）。"""
        if not self.connected:
            raise RuntimeError("End 未连接真机（离线）；请先 connect()。")
        self.backend_end.send_position(position)

    def set_force(self, force: float) -> None:
        """力度控制（如夹持力 N）。"""
        if not self.connected:
            raise RuntimeError("End 未连接真机（离线）；请先 connect()。")
        self.backend_end.send_force(force)
