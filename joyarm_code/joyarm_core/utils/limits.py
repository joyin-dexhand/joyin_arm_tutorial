"""关节限位守卫（指令路径防护底层）+ 限位构建/采样辅助。

``clamp_to_limits``：运动指令下发前逐元素裁剪到关节限位内；``joint_limits_from_model`` /
``limits_from_joint_cfgs`` / ``soft_limits``：硬限位解析（URDF 模型 / config 关节条目）
与按四种逐关节绝对余量内缩的软限位派生；``rand_within_limits``：限位内均匀采样。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from .types import JointLimits

__all__ = ["clamp_to_limits", "joint_limits_from_model", "limits_from_joint_cfgs",
           "soft_limits", "rand_within_limits"]


# ============================================================
# 限位裁剪（clamp_to_limits：指令越限就近裁剪）
# ============================================================
def clamp_to_limits(targets: np.ndarray, limits: JointLimits) -> np.ndarray:
    """运动指令逐元素裁剪到关节限位内；返回与 ``targets`` 同形状。

    传硬限位（如 ``JoyArm.arm_limits``）裁到硬限位；传软限位
    （如 ``JoyArm.arm_limits_soft``）则留缓冲。

    :param targets: ``(n,)`` 或 ``(N,n)`` 目标关节角，弧度。
    :raises ValueError: 限位 ``q_min > q_max``（配置错误）、标量输入、末维与限位不匹配。
    """
    targets = np.asarray(targets, dtype=float)
    q_min = np.asarray(limits.q_min, dtype=float)
    q_max = np.asarray(limits.q_max, dtype=float)

    # q_min > q_max 属配置错误，静默裁剪会掩盖问题
    bad = np.where(q_min > q_max)[0]
    if bad.size > 0:
        raise ValueError(
            f"limits.py - clamp_to_limits：限位配置错误 q_min > q_max（关节索引 {bad.tolist()}）")

    # 标量（0-d）拦截：shape[-1] 会 IndexError
    if targets.ndim == 0:
        raise ValueError(
            "limits.py - clamp_to_limits：目标关节角不能是标量，需为 (n,) 或 (N,n) 数组")

    if targets.shape[-1] != q_min.shape[0]:
        raise ValueError(
            f"limits.py - clamp_to_limits：目标关节角最后一维 {targets.shape[-1]} "
            f"与限位维度 {q_min.shape[0]} 不匹配"
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


# 软限位 margin 四键（config joyarm.arm_soft_margins / end_soft_margins 的合法键集）
_MARGIN_KEYS = ("q_upper", "q_lower", "dq", "tau")


def _parse_margins(margins: dict, n: int) -> dict:
    """把 config margin 字典解析为 ``{键: (n,) 数组}``（标量广播、缺省键 = 0）。

    :raises ValueError: margins 非字典、含未知键、值含负数或列表长度 ≠ n。
    """
    if not isinstance(margins, dict):
        raise ValueError(
            f"limits.py - soft_limits：margins 需为四键字典（键取 {_MARGIN_KEYS}），"
            f"实际类型 {type(margins).__name__}")
    unknown = [k for k in margins if k not in _MARGIN_KEYS]
    if unknown:
        raise ValueError(
            f"limits.py - soft_limits：margin 含未知键 {unknown}（合法键 {_MARGIN_KEYS}）")
    parsed = {}
    for key in _MARGIN_KEYS:
        raw = margins.get(key, 0.0)
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:                       # 标量：全关节统一
            arr = np.full(n, float(arr))
        else:                                   # 列表：逐关节
            arr = arr.reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(
                f"limits.py - soft_limits：margin[{key!r}] 长度 {arr.shape[0]} "
                f"与关节数 {n} 不匹配（逐关节需 n 元列表，或写标量统一全关节）")
        if np.any(arr < 0):
            raise ValueError(f"limits.py - soft_limits：margin[{key!r}] 不允许负值：{raw!r}")
        parsed[key] = arr
    return parsed


def limits_from_joint_cfgs(joint_cfgs: list) -> Optional[JointLimits]:
    """从 config 关节条目列表解析**硬限位**（arm/end 同构；空列表返回 ``None``）。

    条目四键 ``q_min``/``q_max``/``dq_max``/``tau_max``（rad / rad/s / N·m，
    末端为电机空间行程）；缺键量纲置 ±∞（不参与守卫）。JoyArm 的
    ``arm_limits``/``end_limits`` 均由此解析（arm 条目四键在
    ``JoyArm.check_config`` 中强制齐全，数值须与 URDF limit 标定保持一致）。

    :param joint_cfgs: ``backend.arm.joints`` / ``backend.end.joints`` 条目列表。
    """
    joints = list(joint_cfgs or [])
    if not joints:
        return None

    def _num(j, key, default):
        v = j.get(key)
        return float(v) if v is not None else default

    return JointLimits(
        q_min=np.array([_num(j, "q_min", -np.inf) for j in joints]),
        q_max=np.array([_num(j, "q_max", np.inf) for j in joints]),
        dq_max=np.array([_num(j, "dq_max", np.inf) for j in joints]),
        tau_max=np.array([_num(j, "tau_max", np.inf) for j in joints]),
    )


def rand_within_limits(limits: JointLimits, size: Optional[int] = None,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """在限位内均匀采样关节角（``(n,)``；``size`` 给出 ``(size, n)``）。

    :param limits: 限位（通常传软限位，如 ``JoyArm.arm_limits_soft``）。
    :param size: 采样组数；缺省单组 ``(n,)``。
    :param rng: ``numpy`` 随机生成器；缺省 ``np.random.default_rng()``。
    """
    rng = rng if rng is not None else np.random.default_rng()
    n = np.asarray(limits.q_min).shape[0]
    if size is None:
        return rng.uniform(limits.q_min, limits.q_max)
    return rng.uniform(limits.q_min, limits.q_max, size=(size, n))


def soft_limits(hard: JointLimits, margins: dict) -> JointLimits:
    """由硬限位按四种**逐关节绝对余量**内缩生成**软限位**（全零 margin 时软=硬）。

    位置上下各自内缩、速度/力矩上限各自下调（inf 限位减 margin 仍为 inf）::

        soft.q_max   = hard.q_max   - margins["q_upper"]   # rad
        soft.q_min   = hard.q_min   + margins["q_lower"]   # rad
        soft.dq_max  = hard.dq_max  - margins["dq"]        # rad/s
        soft.tau_max = hard.tau_max - margins["tau"]       # N·m

    :param hard: 硬限位（URDF 或 config 来源，初始化后固定）。
    :param margins: 四键字典 ``{"q_upper", "q_lower", "dq", "tau"}``，每键为标量
        （全关节统一）或 n 元列表（逐关节），缺省键视为 0——对应 config
        ``joyarm.arm_soft_margins``（末端 ``end_soft_margins``）段。
    :raises ValueError: margins 结构非法（非字典/未知键/负值/长度不符），或内缩后
        位置限位交叉（``q_min > q_max``）、速度/力矩上限为负。
    """
    n = np.asarray(hard.q_min).shape[0]
    m = _parse_margins(margins, n)
    soft = replace(hard,
                   q_min=hard.q_min + m["q_lower"],
                   q_max=hard.q_max - m["q_upper"],
                   dq_max=hard.dq_max - m["dq"],
                   tau_max=hard.tau_max - m["tau"])
    bad_q = np.where(soft.q_min > soft.q_max)[0]
    if bad_q.size > 0:
        raise ValueError(
            f"limits.py - soft_limits：位置 margin 过大导致限位交叉（q_min > q_max，"
            f"关节索引 {bad_q.tolist()}）；请减小 q_upper/q_lower")
    if np.any(soft.dq_max < 0) or np.any(soft.tau_max < 0):
        raise ValueError("limits.py - soft_limits：dq/tau margin 不小于对应硬上限，请减小 margin")
    return soft
