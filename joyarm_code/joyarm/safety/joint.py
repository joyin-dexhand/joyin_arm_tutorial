"""关节层安全（§11.1）：关节裁剪与逐关节限位校验。

- :func:`clamp_to_limits`：运动指令逐元素裁剪到关节限位内（**已实现**，
  软/硬限位实例均可传入，是关节层软防护的底层）。
- :func:`joint_limits_check`：逐关节限位校验（位置 / 速度 / 加速度 / 力矩 /
  温度 / 过压过流），Ch11 实现。

对应章节：Ch11（``chapt11_safety.md`` §11.1）。
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..utils.types import ArmState, JointLimits, Violation

__all__ = ["clamp_to_limits", "joint_limits_check"]


def clamp_to_limits(targets: np.ndarray, limits: JointLimits) -> np.ndarray:
    """将运动指令逐元素裁剪到关节限位内（关节层软防护底层）。

    所有运动指令下发前都应过本函数：对 ``q_min/q_max`` 做逐元素裁剪。
    传入硬限位实例（:attr:`Arm.joint_limits`）裁剪到硬限位；传入软限位
    实例（:attr:`Arm.joint_limits_soft`，其 ``q_min/q_max`` 即软限位
    ``qlow/qhigh``）则留缓冲。

    :param targets: ``(n,)`` 或 ``(N,n)`` 目标关节角，弧度。
    :param limits: 关节限位声明。
    :return: 与 ``targets`` 同形状的裁剪后关节角。
    :raises ValueError: ``targets`` 与限位维度不匹配，或限位自身
                        ``q_min > q_max``（配置错误）时抛出。
    """
    targets = np.asarray(targets, dtype=float)
    q_min = np.asarray(limits.q_min, dtype=float)
    q_max = np.asarray(limits.q_max, dtype=float)

    # q_min > q_max 属配置错误，静默裁剪会掩盖问题
    bad = np.where(q_min > q_max)[0]
    if bad.size > 0:
        raise ValueError(
            f"限位配置错误：q_min > q_max（关节索引 {bad.tolist()}）"
        )

    # 标量（0-d）输入拦截：shape 为 ()，shape[-1] 会 IndexError
    if targets.ndim == 0:
        raise ValueError(
            "目标关节角不能是标量，需为 (n,) 或 (N,n) 数组"
        )

    # 维度校验：目标最后一维必须等于关节数（限位数组长度）
    if targets.shape[-1] != q_min.shape[0]:
        raise ValueError(
            f"目标关节角最后一维 {targets.shape[-1]} 与限位维度 "
            f"{q_min.shape[0]} 不匹配"
        )
    # numpy.clip 逐元素裁剪：低于 q_min 的抬到 q_min，高于 q_max 的压到 q_max
    # 支持 (n,) 和 (N,n) 两种形状（广播）
    return np.clip(targets, q_min, q_max)


def joint_limits_check(state: ArmState, limits: JointLimits) -> List[Violation]:
    """纯函数：关节层限位校验。

    校验项：位置 / 速度 / 加速度 / 力矩 / 电机温度 / 驱动器温度 /
    过压 / 过流。违规分级为 :attr:`Severity.ERROR` /
    :attr:`Severity.CRITICAL`（硬限位）。

    :return: 违规列表（``layer="joint"``）。
    """
    # 占位：Ch11 实现——逐关节检查位置/速度/加速度/力矩/温度/电压/电流是否越限。
    raise NotImplementedError("joint_limits_check 待 Ch11 实现")
