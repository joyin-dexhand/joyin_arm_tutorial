"""Controller —— 控制律内核基类接口（ABC）

已知：目标轨迹帧与当前关节状态；求解：单步控制指令并下发。

基于目标帧与当前状态，计算控制指令并下发（子类实现）；
类属性 `Controller.MODE` 声明本控制律需要的电机控制模式（子类覆写）。

外部调用：
    - :meth:`step(arm, target, jointstate)` —— 算一步控制指令并经 arm 下发。

子类需实现：
    - :meth:`step(arm, target, jointstate)` —— 计算并下发一步控制指令。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ...utils.types import ControlMode, JointState, TrajFrame

__all__ = ["Controller"]


class Controller(ABC):

    MODE: ControlMode = ControlMode.MIT

    @abstractmethod
    def step(self, arm, target: TrajFrame, jointstate: JointState) -> bool:
        """步进指令：基于目标与当前状态计算单步控制指令并经 arm 下发。

        :param arm: 机械臂对象。
        :param target: 目标轨迹帧（`TrajFrame`，采样产出的轨迹帧）。
        :param jointstate: 当前关节状态（`JointState`）。
        :return: 是否已成功计算并下发本拍指令。
        """
