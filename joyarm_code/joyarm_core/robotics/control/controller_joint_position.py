"""JointPositionController —— 关节位置控制器（默认实现）。

激活即位置模式（类属性 ``MODE = POSITION`` 声明本控制律所需电机模式，
运动管线据此自动 ``set_mode_arm``）；在关节空间跟踪当前轨迹帧的 ``q``：单周期步长超过
``dq_max / arm.ctrl_hz``（不超过关节最大速度）的关节就地裁剪并节流告警，
指令经 ``arm.set_arm_command`` 下发（硬限位守卫归后端基类模板）。
config ``robotics.control`` 段写注册名 ``joint_position_controller`` 即选用。
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
        self._warn_last = float("-inf")

    def step(self, arm, target: TrajFrame, jointstate: JointState) -> bool:
        """步进：``q_ref = target.q``；步长上限 = ``dq_max / arm.ctrl_hz``——
        越限关节裁剪到 ``q_cur ± 上限`` 并节流告警（帧间隔内不超过最大关节
        速度），指令经 ``arm.set_arm_command`` 下发。

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
        q_cur = np.asarray(jointstate.q, dtype=float).reshape(-1)
        q_cmd = q_ref
        limits = arm.arm_limits
        if limits is not None:
            hz = float(getattr(arm, "ctrl_hz", 200.0))      # 管线起搏频率（JoyArm 公开）
            step_max = np.asarray(limits.dq_max, dtype=float) / hz
            over = np.abs(q_ref - q_cur) > step_max
            if np.any(over):
                step = np.clip(q_ref - q_cur, -step_max, step_max)
                q_cmd = q_cur + step
                now = time.monotonic()
                if now - self._warn_last >= _WARN_INTERVAL:
                    self._warn_last = now
                    logger.warning(
                        "controller_joint_position.py - "
                        "JointPositionController.step：关节 %s 单周期步长"
                        "超上限（|Δq| > dq_max/ctrl_hz），已裁剪跟踪（帧间"
                        "速度不超过最大关节速度）",
                        np.where(over)[0].tolist())
        arm.set_arm_command(ControlMode.POSITION, q=q_cmd)
        return True
