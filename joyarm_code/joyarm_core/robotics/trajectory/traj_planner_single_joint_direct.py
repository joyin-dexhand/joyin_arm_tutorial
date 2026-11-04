"""SingleJointDirectTrajPlanner —— 单帧关节域直传轨迹规划

1. target 仅取单帧，`q` 锚定有效性（非 None 且维度与 `home_arm` 一致），有效时该帧 q/dq/ddq/tau 原样透传
2. 空目标、`q` 无效或多帧视为无效，一律回退发布 q_home（此时 dq/ddq/tau 均为 None）。
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from . import TrajPlanner
from ...utils.types import TrajFrame

__all__ = ["SingleJointDirectTrajPlanner"]

logger = logging.getLogger("joyarm_core.traj_planner_single_joint_direct")


class SingleJointDirectTrajPlanner(TrajPlanner):

    def __init__(self):
        """无参构造：直传目标四域缓存与多帧告警去重标志置空（首次 plan 前采样将抛错）。"""
        self._target_q: Optional[np.ndarray] = None    # 直传目标·位置（原子发布：发布后禁止原地修改）
        self._target_dq: Optional[np.ndarray] = None   # 直传目标·速度（None = 未透传，采样原样 None）
        self._target_ddq: Optional[np.ndarray] = None  # 直传目标·加速度（None = 未透传，采样原样 None）
        self._target_tau: Optional[np.ndarray] = None  # 直传目标·力矩（None = 未透传，采样原样 None）
        self._multi_warned: bool = False               # 多帧告警去重（单帧/空目标规划后复位）

    def plan(self, arm, target: List[TrajFrame]) -> bool:
        """规划：单帧关节域直传——q 锚定有效性，q/dq/ddq/tau 原样透传，无效回退 q_home。

        :param arm: 机械臂对象（仅用 `home_arm` 定维与回退，不读当前状态）。
        :param target: 目标帧，单点或序列——仅取单帧；q 非 None 且维度与 `home_arm` 一致才有效，
            有效时该帧 q/dq/ddq/tau 原样透传（缺失为 None，dq/ddq/tau 不做维度校验）。
        :return: 恒为 `True`（直传无需当前状态，无离线失败路径）。

        **实现**：q 有效则四域（非 None 者转一维 float 数组）拷贝后原子发布；空目标、多帧
        （仅支持单帧且 q 有效，同一多帧目标只告警一次）、q 空或维度不符一律回退 q_home（dq/ddq/tau 置 None）。
        """
        targets = [target] if isinstance(target, TrajFrame) else list(target or [])
        q_home = np.asarray(arm.home_arm, dtype=float).reshape(-1)
        q: Optional[np.ndarray] = None
        dq: Optional[np.ndarray] = None
        ddq: Optional[np.ndarray] = None
        tau: Optional[np.ndarray] = None
        if len(targets) == 1 and targets[0].q is not None:
            tq = np.asarray(targets[0].q, dtype=float).reshape(-1)
            if tq.size == q_home.size:
                q = tq
                dq = np.asarray(targets[0].dq, dtype=float).reshape(-1) if targets[0].dq is not None else None
                ddq = np.asarray(targets[0].ddq, dtype=float).reshape(-1) if targets[0].ddq is not None else None
                tau = np.asarray(targets[0].tau, dtype=float).reshape(-1) if targets[0].tau is not None else None
        if len(targets) > 1:
            if not self._multi_warned:
                logger.warning(
                    "traj_planner_single_joint_direct.py - SingleJointDirectTrajPlanner."
                    "plan：目标帧数 "
                    f"{len(targets)} > 1，不支持多帧（仅支持单帧且 q 有效），"
                    "已回退 q_home；请每次只写一帧目标（set_target_trajframe 单帧）")
                self._multi_warned = True
        else:
            self._multi_warned = False      # 单帧/空目标规划后复位告警去重
        # 拷贝后一次性原子发布（并发契约：发布后禁止原地修改）；无效回退 q_home 且三域置 None
        if q is not None:
            self._target_q = q.copy()
            self._target_dq = dq.copy() if dq is not None else None
            self._target_ddq = ddq.copy() if ddq is not None else None
            self._target_tau = tau.copy() if tau is not None else None
        else:
            self._target_q = q_home.copy()
            self._target_dq = None
            self._target_ddq = None
            self._target_tau = None
        return True

    def sample(self, t: float) -> TrajFrame:
        """采样：恒输出直传帧——四关节域用已发布值原样填（缺失为 None，不补 0）。

        :param t: 采样时刻（s，绝对时间，仅回填 `TrajFrame.time`）。
        :return: 此刻轨迹帧 `TrajFrame`（q/dq/ddq/tau 为发布的目标域，缺失即 None；
            单位 rad / rad/s / rad/s² / N·m）。
        """
        if self._target_q is None:
            raise RuntimeError(
                "traj_planner_single_joint_direct.py - SingleJointDirectTrajPlanner.sample：尚未规划（先经 plan 规划）")
        return TrajFrame(time=float(t), q=self._target_q, dq=self._target_dq, ddq=self._target_ddq, tau=self._target_tau)
