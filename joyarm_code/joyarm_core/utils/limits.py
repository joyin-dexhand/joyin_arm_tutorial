"""关节限位守卫（指令路径防护底层）。

``clamp_to_limits``：运动指令下发前逐元素裁剪到关节限位内（``JoyArm.clamp_q``
与后续控制律均消费）。状态监测 / 日志 / 安全策略不在核心库——由 ROS2 监测节点
承担（Ch11）。
"""
from __future__ import annotations

import numpy as np

from .types import JointLimits

__all__ = ["clamp_to_limits"]


def clamp_to_limits(targets: np.ndarray, limits: JointLimits) -> np.ndarray:
    """运动指令逐元素裁剪到关节限位内；返回与 ``targets`` 同形状。

    传硬限位（``JoyArm.joint_limits``）裁到硬限位；传软限位
    （``joint_limits_soft``，即 ``qlow/qhigh``）则留缓冲。

    :param targets: ``(n,)`` 或 ``(N,n)`` 目标关节角，弧度。
    :raises ValueError: 限位 ``q_min > q_max``（配置错误）、标量输入、末维与限位不匹配。
    """
    targets = np.asarray(targets, dtype=float)
    q_min = np.asarray(limits.q_min, dtype=float)
    q_max = np.asarray(limits.q_max, dtype=float)

    # q_min > q_max 属配置错误，静默裁剪会掩盖问题
    bad = np.where(q_min > q_max)[0]
    if bad.size > 0:
        raise ValueError(f"限位配置错误：q_min > q_max（关节索引 {bad.tolist()}）")

    # 标量（0-d）拦截：shape[-1] 会 IndexError
    if targets.ndim == 0:
        raise ValueError("目标关节角不能是标量，需为 (n,) 或 (N,n) 数组")

    if targets.shape[-1] != q_min.shape[0]:
        raise ValueError(
            f"目标关节角最后一维 {targets.shape[-1]} 与限位维度 {q_min.shape[0]} 不匹配"
        )
    # np.clip 广播裁剪：低于 q_min 抬到 q_min，高于 q_max 压到 q_max，支持 (n,)/(N,n)
    return np.clip(targets, q_min, q_max)
