"""Controller —— 控制律策略 ABC。

``JoyArm._controllers`` 成员字典的契约：实现经 config ``robotics.control`` 选型，运行期可经
``arm.set_controller`` 切换。每拍控制把"期望 + 实测"映射为电机控制值——指令发送前过
限位守卫（裁剪/否决）。

本模块仅定义通用接口；具体控制律与执行循环为教程 Ch6 教学内容，实现后经
``REGISTRY`` 注册接入。Ch6。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Controller"]


class Controller(ABC):
    """控制律策略接口：当前状态 + 期望 → 电机控制值。"""

    @abstractmethod
    def compute(self, arm, target: np.ndarray, state=None, **kw) -> np.ndarray:
        """单 tick 控制律：``target`` 期望量（语义随控制律）；``state`` 最新实测快照
        （``arm.get_arm_state()`` 产物，缺省由执行器喂入）；返回 ``(n,)`` 控制值
        （位置 / 速度 / 力矩或 MIT 参数组）。"""
