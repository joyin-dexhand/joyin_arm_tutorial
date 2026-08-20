"""基于运动学的控制（Part 1，Ch6 占位）。

单关节闭环 + 整机位置/速度 + 实时循环 + 轨迹回放；ControlLoop 是**执行器**
（tick 循环，逐拍驱动 ``JoyArm._controller``），控制律本身是 controller.py 的策略族。
"""
from __future__ import annotations

import numpy as np

from ...utils.types import ControlMode

__all__ = [
    "joint_position_control",
    "joint_velocity_control",
    "torque_control",
    "arm_position_control",
    "arm_velocity_control",
    "ControlLoop",
    "play_trajectory",
]


def joint_position_control(arm, q_target: np.ndarray, kp: float, kd: float):
    """§6.1 单关节闭环位置控制（``q_target``(n,)；``kp``/``kd`` 位置/速度增益）。"""
    raise NotImplementedError("joint_position_control 待 Ch6 实现")


def joint_velocity_control(arm, dq_target: np.ndarray):
    """§6.1 速度模式（``dq_target``(n,)）。"""
    raise NotImplementedError("joint_velocity_control 待 Ch6 实现")


def torque_control(arm, tau_target: np.ndarray):
    """§6.1 力矩模式（``tau_target``(n,)）。"""
    raise NotImplementedError("torque_control 待 Ch6 实现")


def arm_position_control(arm, T_target: np.ndarray, q0=None):
    """§6.2 整机位置：IK → 关节角 → 下发（``T_target``(4,4)，``q0`` IK 初值）。"""
    raise NotImplementedError("arm_position_control 待 Ch6 实现")


def arm_velocity_control(arm, twist_target):
    """§6.3 整机速度：Jacobian → 关节速度 → 下发（末端目标旋量）。"""
    raise NotImplementedError("arm_velocity_control 待 Ch6 实现")


class ControlLoop:
    """实时控制循环（执行器，非策略）：每 tick 读状态 → ``_controller`` 控制律 → 下发。

    消费 Trajectory，按 ``hz`` 频率、按实际时间（``t = now − t₀`` 对序列求值而非
    数步数）执行，抗 tick 抖动。用法：``ControlLoop(arm, policy, hz=200)`` →
    ``start()`` / ``stop()``。
    """

    def __init__(self, arm, policy=None, hz: int = 200):
        self.arm = arm
        self.policy = policy
        self.hz = hz

    def start(self) -> None:
        """启动循环（独立线程）。"""
        raise NotImplementedError("ControlLoop.start 待 Ch6 实现")

    def stop(self) -> None:
        """停止循环。"""
        raise NotImplementedError("ControlLoop.stop 待 Ch6 实现")


def play_trajectory(arm, traj, mode: ControlMode = ControlMode.POSITION, hz: int = 200):
    """按时间序列回放轨迹（§6 例程 6/7；等价"自动重放示教"）。"""
    raise NotImplementedError("play_trajectory 待 Ch6 实现")
