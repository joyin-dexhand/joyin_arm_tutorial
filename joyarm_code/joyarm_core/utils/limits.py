"""关节限位守卫（指令路径防护底层）+ 限位构建辅助。

``clamp_to_limits``：运动指令下发前逐元素裁剪到关节限位内；``joint_limits_from_model`` / ``soft_limits``：从任意
运动学模型对象（鸭子类型：pinocchio model 或等价结构）解析硬限位、按比例内缩生成软限位。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .types import JointLimits

__all__ = ["clamp_to_limits", "joint_limits_from_model", "soft_limits"]


# ============================================================
# 限位裁剪（clamp_to_limits：指令越限就近裁剪）
# ============================================================
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


# ============================================================
# 限位构建（joint_limits_from_model 硬限位解析 / soft_limits 软限位派生）
# ============================================================
def joint_limits_from_model(model) -> JointLimits:
    """从模型对象解析**硬限位**（鸭子类型，仅访问属性，不依赖 pinocchio）。

    位置限位 ``q_min/q_max`` 与速度/力矩上限（``velocityLimit``/``effortLimit``
    缺失时置 ∞）。

    :param model: pinocchio ``model``（或提供 ``nq``/``nv``/``lowerPositionLimit``/
                    ``upperPositionLimit``/``velocityLimit``/``effortLimit`` 等属性的等价对象）。
    """
    n, nv = model.nq, model.nv
    q_min = np.asarray(model.lowerPositionLimit, dtype=float).reshape(n)
    q_max = np.asarray(model.upperPositionLimit, dtype=float).reshape(n)
    dq_max = (
        np.asarray(model.velocityLimit, dtype=float).reshape(nv)
        if hasattr(model, "velocityLimit") else np.full(nv, np.inf)
    )
    tau_max = (
        np.asarray(model.effortLimit, dtype=float).reshape(nv)
        if hasattr(model, "effortLimit") else np.full(nv, np.inf)
    )
    if nv != n:  # 含非旋转关节等自由度不等臂：逐关节上限退化为 ∞（由 config 精确覆盖）
        dq_max = np.full(n, np.inf)
        tau_max = np.full(n, np.inf)
    return JointLimits(q_min=q_min, q_max=q_max, dq_max=dq_max, tau_max=tau_max)


def soft_limits(hard: JointLimits, margin: float) -> JointLimits:
    """由硬限位按跨度比例 ``margin`` 内缩生成**软限位**（``margin=0`` 时软=硬）。"""
    span = hard.q_max - hard.q_min
    return replace(hard, q_min=hard.q_min + margin * span, q_max=hard.q_max - margin * span)
