"""SingleJointHoldTrajPlanner —— 到单点关节目标的直传轨迹规划

1. target 仅取关节角 `q`（其余字段忽略），plan 将其原子发布为恒定目标，sample 恒输出该 `q`（dq/ddq 为 0）；
2. `q` 为空或维度与`arm_home` 不符视为无效，回退发布 q_home；
3. 运动限速由控制器和关节电机内部进行实现（若控制器和关节电机均无限速，运动将失控！）。
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from . import TrajPlanner
from ...utils.types import TrajFrame

__all__ = ["SingleJointHoldTrajPlanner"]

logger = logging.getLogger("joyarm_core.traj_planner_single_joint_hold")


class SingleJointHoldTrajPlanner(TrajPlanner):

    def __init__(self):
        """无参构造：直传目标缓存与多帧告警去重标志置空（首次 plan 前采样将抛错）。"""
        self._target_q: Optional[np.ndarray] = None   # 恒定目标（原子发布：发布后禁止原地修改）
        self._multi_warned: bool = False              # 多帧告警去重（单帧/空目标规划后复位）

    def plan(self, arm, target: List[TrajFrame]) -> bool:
        """规划：直传——目标仅取单帧 `q`，有效则恒定发布该 q，无效回退 q_home。

        :param arm: 机械臂对象（仅用 `arm_home` 定维与回退，不读当前状态）。
        :param target: 目标帧，单点或序列——仅取单帧的 `q`（time/dq/ddq 等其余字段忽略）；q 为空或维度不符视为无效。
        :return: 恒为 ``True``（直传无需当前状态，无离线失败路径）。

        **实现**：q 有效则拷贝后原子发布为恒定目标；空目标、多帧（不支持多帧，仅支持单帧且 q 有效，同一多帧目标只告警一次）、q 空或维度不符一律回退 q_home。
        """
        targets = [target] if isinstance(target, TrajFrame) else list(target or [])
        q_home = np.asarray(arm.arm_home, dtype=float).reshape(-1)
        q: Optional[np.ndarray] = None
        if len(targets) == 1 and targets[0].q is not None:
            tq = np.asarray(targets[0].q, dtype=float).reshape(-1)
            if tq.size == q_home.size:
                q = tq
        if len(targets) > 1:
            if not self._multi_warned:
                logger.warning(
                    "traj_planner_single_joint_hold.py - SingleJointHoldTrajPlanner."
                    "plan：目标帧数 "
                    f"{len(targets)} > 1，不支持多帧（仅支持单帧且 q 有效），"
                    "已回退 q_home；请每次只写一帧目标（set_target_traj 单帧）")
                self._multi_warned = True
        else:
            self._multi_warned = False      # 单帧/空目标规划后复位告警去重
        # 拷贝后一次性原子发布（并发契约：发布后禁止原地修改）
        self._target_q = q.copy() if q is not None else q_home.copy()
        return True

    def sample(self, t: float) -> TrajFrame:
        """采样：恒输出直传目标 q（dq/ddq 为 0）。

        :param t: 采样时刻（s，绝对时间，仅回填 ``TrajFrame.time``）。
        :return: 此刻轨迹帧 `TrajFrame`（q 恒为发布的目标，dq/ddq 为 0）。
        """
        if self._target_q is None:
            raise RuntimeError(
                "traj_planner_single_joint_hold.py - SingleJointHoldTrajPlanner.sample：尚未规划（先经 plan 规划）")
        return TrajFrame(time=float(t), q=self._target_q,
                         dq=np.zeros_like(self._target_q),
                         ddq=np.zeros_like(self._target_q))
