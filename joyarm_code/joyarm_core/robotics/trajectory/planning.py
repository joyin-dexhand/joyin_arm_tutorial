"""规划算法（关节空间 + 笛卡尔空间轨迹规划方法的统一存放地）。

本模块只放**纯函数**（无 ``arm`` 依赖、可独立复用与测试）：三次/五次多项式、
速度启发式途经点插值、笛卡尔直线/圆弧等规划方法都在此实现。**轨迹规划器**
（``TrajPlanner`` 子类，Ch5 教学内容）与上层应用（如 ``JoyArm.move_j``）对
本模块**导入并封装委托**——不在求解器内部直接实现规划方法，因为外部会复用
同一批规划函数。

本章先实现关节空间点到点三次多项式（``cubic_q`` / ``cubic_traj``），
后续章节的规划方法陆续加入本模块。
"""
from __future__ import annotations

import numpy as np

__all__ = ["cubic_q", "cubic_traj"]


# ============================================================
# 关节空间：点到点三次多项式（零边界速度）
# ============================================================
def cubic_q(q0, q1, s) -> np.ndarray:
    """零边界速度三次多项式插值：``q(s) = q0 + Δq·(3s² − 2s³)``。

    端点速度为零（``q̇(0) = q̇(1) = 0``），峰值速度 ``1.5·|Δq|/t`` 在中点。

    :param q0: 起始关节角 ``(n,)``。
    :param q1: 目标关节角 ``(n,)``。
    :param s: 归一化时间 ``∈ [0, 1]``（超出按端点外推），标量或 ``(N,)`` 数组。
    :return: 关节角 ``(n,)``（标量 s）或 ``(N, n)``（数组 s）。
    """
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    s = np.asarray(s, dtype=float)
    blend = 3.0 * s ** 2 - 2.0 * s ** 3          # 平滑 blends：s=0→0，s=1→1
    if s.ndim == 0:                              # 标量 s → (n,)
        return q0 + (q1 - q0) * float(blend)
    return q0 + (q1 - q0) * blend[:, None]       # (N,) s → (N, n)


def cubic_traj(q0, q1, t: float, rate: float = 100.0):
    """三次多项式整段采样：``q0 → q1``，总时长 ``t``，采样率 ``rate``。

    :param q0: 起始关节角 ``(n,)``。
    :param q1: 目标关节角 ``(n,)``。
    :param t: 总时长（秒）；``t ≤ 0`` 退化为单帧直发目标。
    :param rate: 采样率（Hz）。
    :return: ``(ts, q, dq)``——时间戳 ``(N,)``、关节角 ``(N, n)``、关节速度
        ``(N, n)``（``dq/dt`` 解析导数 ``6(s−s²)·Δq/t``，供速度前馈/规划器复用）。
    """
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    t, rate = float(t), float(rate)
    if t <= 0 or rate <= 0:
        return (np.zeros(1), q1.reshape(1, -1), np.zeros((1, q0.size)))
    n = max(int(round(t * rate)) + 1, 2)          # 至少首末两帧
    ts = np.arange(n) / (n - 1) * t
    s = ts / t
    q = q0 + (q1 - q0) * (3.0 * s ** 2 - 2.0 * s ** 3)[:, None]
    dq = (q1 - q0) * (6.0 * s * (1.0 - s) / t)[:, None]
    return ts, q, dq
