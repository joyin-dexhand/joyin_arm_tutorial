"""关节/末端限位守卫（指令路径防护底层）+ 限位构建/采样辅助。

``clamp_to_limits``：指令下发前逐元素裁剪到关节限位（backend 传**硬限位**）；
``limits_from_joint_cfgs``：硬限位解析；``soft_limits_from_cfg``：软限位直配解析；
``tcp_limits_from_cfg``：末端空间限位解析；``rand_within_limits``：限位内均匀采样。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .types import JointLimits, TcpLimits

__all__ = ["clamp_to_limits", "limits_from_joint_cfgs", "soft_limits_from_cfg",
           "tcp_limits_from_cfg", "rand_within_limits"]


# ============================================================
# 限位裁剪（clamp_to_limits：指令越限就近裁剪）
# ============================================================
def clamp_to_limits(targets: np.ndarray, limits: JointLimits) -> np.ndarray:
    """运动指令逐元素裁剪到关节限位内；返回与 ``targets`` 同形状。

    传硬限位（如 ``JoyArm.arm_limits``）裁到硬限位；backend 指令守卫一律传
    硬限位（软限位归上层状态判断，不参与指令裁剪）。

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
# 限位构建（硬限位 / 软限位 / 末端空间限位）
# ============================================================
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

    :param limits: 限位（如 ``JoyArm.arm_limits``）。
    :param size: 采样组数；缺省单组 ``(n,)``。
    :param rng: ``numpy`` 随机生成器；缺省 ``np.random.default_rng()``。
    """
    rng = rng if rng is not None else np.random.default_rng()
    n = np.asarray(limits.q_min).shape[0]
    if size is None:
        return rng.uniform(limits.q_min, limits.q_max)
    return rng.uniform(limits.q_min, limits.q_max, size=(size, n))


def soft_limits_from_cfg(cfg: dict, n: int) -> JointLimits:
    """从 config 四键**直值**字典解析软限位。

    软限位供**上层状态判断**使用（如超软限位 → 状态异常 → 急停恢复）；
    backend 指令守卫只裁硬限位。JoyArm 的 ``arm_limits_soft`` / ``end_limits_soft``
    均由此解析（config ``joyarm.arm_soft_limits`` / ``end_soft_limits`` 段，
    arm/end 分开配置），且须位于对应硬限位内（由 JoyArm 校验）。

    :param cfg: 四键字典 ``{"q_min", "q_max", "dq_max", "tau_max"}``，每键为标量
        （全关节统一）或 n 元列表（逐关节）；缺省键置 ±∞（该量不设软限）。
    :param n: 关节数（列表长度校验基准）。
    :raises ValueError: 结构非法（非字典/未知键/列表长度 ≠ n/负的幅值上限/
        ``q_min > q_max``）。
    """
    if not isinstance(cfg, dict):
        raise ValueError(
            f"limits.py - soft_limits_from_cfg：软限位需为四键字典"
            f"（q_min/q_max/dq_max/tau_max），实际类型 {type(cfg).__name__}")
    known = ("q_min", "q_max", "dq_max", "tau_max")
    unknown = [k for k in cfg if k not in known]
    if unknown:
        raise ValueError(
            f"limits.py - soft_limits_from_cfg：软限位含未知键 {unknown}（合法键 {known}）")

    def _arr(key, default: float) -> np.ndarray:
        raw = cfg.get(key, default)
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:                        # 标量：全关节统一
            return np.full(n, float(arr))
        arr = arr.reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(
                f"limits.py - soft_limits_from_cfg：软限位[{key!r}] 长度 {arr.shape[0]} "
                f"与关节数 {n} 不匹配（逐关节需 n 元列表，或写标量统一全关节）")
        return arr

    out = JointLimits(q_min=_arr("q_min", -np.inf),
                      q_max=_arr("q_max", np.inf),
                      dq_max=_arr("dq_max", np.inf),
                      tau_max=_arr("tau_max", np.inf))
    bad = np.where(out.q_min > out.q_max)[0]
    if bad.size > 0:
        raise ValueError(
            f"limits.py - soft_limits_from_cfg：软限位配置错误 q_min > q_max"
            f"（关节索引 {bad.tolist()}）")
    if np.any(out.dq_max < 0) or np.any(out.tau_max < 0):
        raise ValueError("limits.py - soft_limits_from_cfg：dq_max/tau_max 不允许负值")
    return out


def tcp_limits_from_cfg(tl: dict) -> TcpLimits:
    """从 config ``tcp_limits`` 段解析末端空间限位。

    ``workspace_box`` 两种写法均可：``[[xmin,ymin,zmin],[xmax,ymax,zmax]]``
    （两行，yaml 常用）或每轴一行 ``[min,max]`` 的 ``(3,2)``；内部统一成
    ``(3,2)``。JoyArm / FakeArm 的 ``tcp_limits`` 均由此解析。

    :param tl: config ``joyarm.tcp_limits`` 段（各键可缺省）。
    """
    box = np.asarray(tl.get("workspace_box",
                            [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.8]]), dtype=float)
    if box.shape == (2, 3):          # [min 行, max 行] → 每轴 [min, max]
        box = box.T
    return TcpLimits(
        workspace_box=box,
        v_lin_max=float(tl.get("v_lin_max", 0.0)),
        v_ang_max=float(tl.get("v_ang_max", 0.0)),
        f_max=float(tl.get("f_max", 0.0)),
        t_max=float(tl.get("t_max", 0.0)),
    )
