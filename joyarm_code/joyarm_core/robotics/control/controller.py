"""Controller —— 控制律接口（ABC）

控制律子类只需实现内核
- :meth:`compute`（单拍控制律：期望 + 实测 → 电机控制值）。

config ``robotics.control`` 段写注册名，即按名实例化装入 ``_controllers`` 成员字典。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Controller"]


class Controller(ABC):
    """控制律策略接口：当前状态 + 期望 → 电机控制值。"""

    @abstractmethod
    def compute(self,
        arm,
        target: np.ndarray,
        state=None,
        **kw,
    ) -> np.ndarray:
        """单 tick 控制律：``target`` 期望量（语义随控制律）；``state`` 最新实测快照
        （``arm.get_arm_state()`` 产物，缺省由执行器喂入）；返回 ``(n,)`` 控制值
        （位置 / 速度 / 力矩或 MIT 参数组）。"""
