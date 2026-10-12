"""TrajPlanner —— 轨迹规划内核基类接口（ABC，纯计算）

已知：带到达时间的目标（单点或序列）；求解：随时间平滑的轨迹与按绝对时间的采样帧。

将单个或一组带到达时间的目标点，加工成随时间平滑的轨迹（缓存于子类内部），并按绝对时间采样当前时刻的期望轨迹帧。

外部调用：
    - :meth:`plan(arm, target)` —— 规划：arm当前状态 + 目标（单点或序列）→ 时间平滑的轨迹（缓存于子类内部）；
    - :meth:`sample(t)` —— 采样：子类内的规划结果 + 当前时间 → 此刻轨迹帧。

子类需实现：
    - :meth:`plan(arm, target)` —— 规划，结果存到对象内；
    - :meth:`sample(t)` —— 采样。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ...utils.types import TrajFrame

__all__ = ["TrajPlanner"]


class TrajPlanner(ABC):

    @abstractmethod
    def plan(self, arm, target: List[TrajFrame]) -> bool:
        """规划：由 arm 当前状态与 target 目标（单点或序列）计算时间平滑的轨迹。

        :param arm: 机械臂对象。
        :param target: 目标帧，单点或序列。
        :return: 本次是否成功规划。

        **规划结果由子类存储到对象内**（打包为单一不可变对象、一次性原子赋值发布），供 :meth:`sample` 采样。
        """

    @abstractmethod
    def sample(self, t: float) -> TrajFrame:
        """采样：基于 :meth:`plan` 得到的轨迹与当前时间，采样出此刻的轨迹帧。

        :param t: 当前绝对时间（Unix 秒）。
        :return: `TrajFrame` 此刻的轨迹帧。
        """
