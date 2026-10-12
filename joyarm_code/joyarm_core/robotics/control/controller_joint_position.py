"""JointPositionController —— 关节位置控制器（默认实现）。

激活位置模式；在关节空间跟踪当前轨迹帧的 `q`。
"""
from __future__ import annotations

import logging
import time

import numpy as np

from . import Controller
from ...utils.types import ControlMode, JointState, TrajFrame

__all__ = ["JointPositionController"]

logger = logging.getLogger("joyarm_core.controller")

_WARN_INTERVAL: float = 0.5    # 步长裁剪告警节流（秒）


class JointPositionController(Controller):
    """控制默认实现：位置模式逐周期跟踪当前轨迹帧的关节位置参考。"""

    MODE = ControlMode.POSITION

    def __init__(self):
        """初始化：越限告警节流时间戳置 -inf（首次越限必告警）。"""
        self._warn_last = float("-inf")

    def step(self, arm, target: TrajFrame, jointstate: JointState) -> bool:
        """步进：跟踪当前轨迹帧的关节位置参考，越限关节就地裁剪并节流告警。

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
        arm.set_arm_command(ControlMode.POSITION, q=q_ref)
        return True
