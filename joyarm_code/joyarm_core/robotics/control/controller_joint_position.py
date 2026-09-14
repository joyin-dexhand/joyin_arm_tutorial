"""JointPositionController —— 关节位置控制器（默认实现）。

激活即位置模式（``MODE = POSITION``，运动管线按声明自动 ``set_mode_arm``）；
在关节空间跟踪当前轨迹帧的 ``q``：单周期步长超过 ``dq_max / ctrl_hz``
（不超过关节最大速度）的关节就地裁剪并节流告警。
config ``robotics.control`` 段写注册名 ``joint_position_controller`` 即选用。
"""
from __future__ import annotations

import logging
import time
from typing import Tuple

import numpy as np

from . import Controller
from ...utils.types import ArmState, ControlMode, TrajFrame

__all__ = ["JointPositionController"]

logger = logging.getLogger("joyarm_core.controller")

_WARN_INTERVAL: float = 0.5    # 步长裁剪告警节流（秒）


class JointPositionController(Controller):
    """控制默认实现：位置模式逐周期跟踪当前轨迹帧的关节位置参考。"""

    MODE: ControlMode = ControlMode.POSITION

    def __init__(self, ctrl_hz: float = 200.0):
        """:param ctrl_hz: 控制频率（管线节拍；亦是步长上限的除数），Hz。"""
        super().__init__(ctrl_hz=ctrl_hz)
        self._warn_last = float("-inf")

    def compute(self, arm, frame: TrajFrame,
                state: ArmState) -> Tuple[ControlMode, dict]:
        """内核：``q_ref = frame.q``；步长上限 = ``dq_max / ctrl_hz``——
        越限关节裁剪到 ``q_cur ± 上限`` 并节流告警（帧间隔内不超过最大关节
        速度）；硬限位守卫仍归后端基类 ``send_position_arm`` 模板。"""
        if frame.q is None:
            raise ValueError(
                "controller_joint_position.py - JointPositionController."
                "compute：当前轨迹帧缺 q（位置跟随需关节位置参考）")
        q_ref = np.asarray(frame.q, dtype=float).reshape(-1)
        if not np.all(np.isfinite(q_ref)):
            raise ValueError(                               # NaN 兜底：拒绝下发
                "controller_joint_position.py - JointPositionController."
                "compute：当前轨迹帧 q 含非有限值（NaN/inf），拒绝下发")
        q_cur = np.asarray(state.joint.q, dtype=float).reshape(-1)
        q_cmd = q_ref
        limits = arm.arm_limits
        if limits is not None:
            step_max = np.asarray(limits.dq_max, dtype=float) / self.ctrl_hz
            over = np.abs(q_ref - q_cur) > step_max
            if np.any(over):
                step = np.clip(q_ref - q_cur, -step_max, step_max)
                q_cmd = q_cur + step
                now = time.monotonic()
                if now - self._warn_last >= _WARN_INTERVAL:
                    self._warn_last = now
                    logger.warning(
                        "controller_joint_position.py - "
                        "JointPositionController.compute：关节 %s 单周期步长"
                        "超上限（|Δq| > dq_max/ctrl_hz），已裁剪跟踪（帧间"
                        "速度不超过最大关节速度）",
                        np.where(over)[0].tolist())
        return ControlMode.POSITION, {"q": q_cmd}
