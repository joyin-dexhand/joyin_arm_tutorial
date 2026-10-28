"""SingleJointCubicTrajPlanner —— 到单点关节目标的三次多项式规划

1. 由当前 `(q0, dq0)` 到目标 `(q1, dq1)`（`target.dq` 缺省则取 0），位置+速度四边界条件解三次多项式系数；
2. 时长 `T` = 目标绝对时刻 − 当前时刻，采样按绝对时间求值、`τ` 钳位 `[0, T]`（超出时刻保持末状态）；目标已过期则直接发布目标为恒定段（维持终点状态）；
3. `q` 为空或维度与 `home_arm` 不符视为无效，回退恒定保持 q_home。
"""
from __future__ import annotations

import logging
import time as _time
from typing import List, Optional, Tuple

import numpy as np

from . import TrajPlanner
from ...utils.types import TrajFrame

__all__ = ["SingleJointCubicTrajPlanner"]

logger = logging.getLogger("joyarm_core.traj_planner_single_joint_cubic")

# 发布的三次段（不可变约定：发布后禁止原地修改）
# (t0, T, c0, c1, c2, c3)，c* 均为 (n,) 系数
_Segment = Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class SingleJointCubicTrajPlanner(TrajPlanner):

    def __init__(self):
        """无参构造：三次段缓存与多帧告警去重标志置空（首次 plan 前采样将抛错）。"""
        self._seg: Optional[_Segment] = None    # 当前三次段（原子发布）
        self._multi_warned: bool = False        # 多帧告警去重（单帧/空目标规划后复位）

    def plan(self, arm, target: List[TrajFrame]) -> bool:
        """规划：单帧关节目标由当前状态解三次多项式（位置+速度四约束）。

        :param arm: 机械臂对象（起点读 `arm.get_state_arm()` 的 q/dq）。
        :param target: 目标帧，单点或序列——仅取单帧的 `q/dq`（`dq` 为终点速度、缺省则取 0）。
        :return: 是否完成规划（无效目标回退恒定 q_home 亦返回 `True`）。
        :raises ValueError: q 有效但目标帧 `time` 为 `None`。

        **实现**：四约束解析系数 `c0=q0`、`c1=dq0`、`c2=(3Δq−(2·dq0+dq1)·T)/T²`、`c3=(−2Δq+(dq0+dq1)·T)/T³`（`T=time−now`）；
            无效目标发布零系数退化段（恒定 q_home），目标已过期直接恒定发布目标 q1。
        """
        targets = [target] if isinstance(target, TrajFrame) else list(target or [])
        q_home = np.asarray(arm.home_arm, dtype=float).reshape(-1)
        q1: Optional[np.ndarray] = None
        if len(targets) == 1 and targets[0].q is not None:
            tq = np.asarray(targets[0].q, dtype=float).reshape(-1)
            if tq.size == q_home.size:
                q1 = tq
        if len(targets) > 1:
            if not self._multi_warned:
                logger.warning(
                    "traj_planner_single_joint_cubic.py - SingleJointCubicTrajPlanner.plan：目标帧数 "
                    f"{len(targets)} > 1，不支持多帧（仅支持单帧且 q 有效），"
                    "已回退恒定 q_home；请每次只写一帧目标")
                self._multi_warned = True
        else:
            self._multi_warned = False      # 单帧/空目标规划后复位告警去重
        now = _time.time()
        if q1 is None:
            # 无效目标：零系数退化段 = 恒定保持 q_home（限速由电机内部 与 backend 限位承担）
            zeros = np.zeros_like(q_home)
            self._seg = (now, 0.0, q_home.copy(), zeros, zeros, zeros)
            return True
        if targets[0].time is None:
            # q 有效但 time 缺失：属使用错误（门面 set_target_joint 的 t 必填校验，
            # 绕过封装手写 TrajFrame 漏填）——显性报错，不得静默选定运动方向
            raise ValueError(
                "traj_planner_single_joint_cubic.py - SingleJointCubicTrajPlanner."
                "plan：目标帧 time=None（cubic 规划需时间字段定 T；"
                "请经 set_target_joint 封装设置或补全 TrajFrame.time）")
        if float(targets[0].time) <= now:
            # 目标已过期：不规划多项式，直接发布目标为恒定段（维持终点状态）
            zeros = np.zeros_like(q_home)
            self._seg = (now, 0.0, q1.copy(), zeros, zeros, zeros)
            return True
        st = arm.get_state_arm()
        if st is None:
            raise RuntimeError(
                "traj_planner_single_joint_cubic.py - SingleJointCubicTrajPlanner."
                "plan：get_state_arm() 返回 None（未连接或读取失败），无法取规划起点")
        q0 = np.asarray(st.q, dtype=float).reshape(-1)
        dq0 = np.asarray(st.dq, dtype=float).reshape(-1)
        dq1 = np.zeros_like(q1) if targets[0].dq is None \
            else np.asarray(targets[0].dq, dtype=float).reshape(-1)
        T = float(targets[0].time) - now                # 已过期分支已拦截，此处 T > 0
        delta = q1 - q0
        c2 = (3.0 * delta - (2.0 * dq0 + dq1) * T) / T ** 2
        c3 = (-2.0 * delta + (dq0 + dq1) * T) / T ** 3
        # 打包不可变、一次性原子发布（并发契约）
        self._seg = (now, T, q0.copy(), dq0.copy(), c2, c3)
        return True

    def sample(self, t: float) -> TrajFrame:
        """采样：按绝对时间求值，`τ` 钳位 `[0, T]`（超末端保持末端状态）。

        :param t: 采样时刻（s，绝对时间）。
        :return: 此刻轨迹帧 `TrajFrame`（q/dq/ddq 为多项式及其一、二阶导数）。
        """
        if self._seg is None:
            raise RuntimeError(
                "traj_planner_single_joint_cubic.py - SingleJointCubicTrajPlanner.sample：尚未规划（先经 plan 规划）")
        t0, T, c0, c1, c2, c3 = self._seg
        tau = min(max(float(t) - t0, 0.0), T)
        return TrajFrame(
            time=float(t),
            q=c0 + c1 * tau + c2 * tau ** 2 + c3 * tau ** 3,
            dq=c1 + 2.0 * c2 * tau + 3.0 * c3 * tau ** 2,
            ddq=2.0 * c2 + 6.0 * c3 * tau,
        )
