"""JointPositionController —— 关节位置控制器（默认实现）。

经位置指令（`send_position_arm`）下发；在关节空间跟踪当前轨迹帧的 `q`。
"""
from __future__ import annotations

import numpy as np

from . import Controller
from ...utils.types import ControlMode, JointState, TrajFrame

__all__ = ["JointPositionController"]


class JointPositionController(Controller):
    """控制默认实现：逐周期经 `send_position_arm` 下发位置指令，跟踪当前轨迹帧的关节位置参考。"""

    MODE = ControlMode.POSITION

    def step(self, arm, target: TrajFrame, jointstate: JointState) -> bool:
        """跟踪当前轨迹帧的关节位置参考，经 `send_position_arm` 直接下发。

        :param arm: 机械臂对象。
        :param target: 目标轨迹帧（当前帧）。
        :param jointstate: 当前关节状态。
        :return: 是否已成功计算并下发本拍指令。
        """
        if target.q is None:
            raise ValueError(
                "controller_joint_position.py - JointPositionController."
                "step：当前轨迹帧缺 q（位置跟随需关节位置参考）")
        q_ref = np.asarray(target.q, dtype=float).reshape(-1)
        if not np.all(np.isfinite(q_ref)):
            raise ValueError(                               # NaN 兜底：拒绝下发
                "controller_joint_position.py - JointPositionController."
                "step：当前轨迹帧 q 含非有限值（NaN/inf），拒绝下发")
        arm.send_position_arm(q_ref)
        return True
