"""ToJointTrajPlanner —— 到关节目标点的轨迹规划默认实现（五次多项式）。

点到点关节空间规划：由当前 ``(q0, dq0, ddq0)`` 到目标 ``(q1, dq1, ddq1)``
（目标缺省速度/加速度为 0），六边界条件解五次多项式系数；采样按绝对时间输出 ``q/dq/ddq``。

config ``robotics.traj`` 段写注册名 ``to_joint_traj_planner``即选用。
"""
from __future__ import annotations

import logging
import time as _time
from typing import Optional, Tuple

import numpy as np

from . import TrajPlanner
from ...utils import TrajFrame

__all__ = ["ToJointTrajPlanner"]

logger = logging.getLogger("joyarm_core.traj_planner_to_joint")

# 发布的系数包（不可变约定：发布后禁止原地修改）
# (t0, T, c0..c2 (3,n), c3..c5 (3,n))
_Coeffs = Tuple[float, float, np.ndarray, np.ndarray]


class ToJointTrajPlanner(TrajPlanner):
    """到关节目标点的规划默认实现：五次多项式（六边界条件，加速度连续）。"""

    def __init__(self):
        self._coeffs: Optional[_Coeffs] = None   # 原子发布（并发契约）
        self._last_multi_warn: Optional[tuple] = None   # 上次多帧告警的目标时刻序列

    def plan(self, arm, target) -> bool:
        """规划：单帧关节目标由当前状态解五次多项式系数。

        :param arm: 机械臂对象。
        :param target: 目标帧，单点或序列——本规划器仅支持**单帧**关节目标
            （``q``，可带 ``dq``/``ddq``）；空目标或 >1 帧回退规划到 q_home
            （多帧时各帧到达时刻不变则只告警一次）；笛卡尔目标（``pose``）抛
            ``ValueError``。
        :return: 是否完成规划（回退帧构造失败如未连接真机返回 ``False``，
            本周期跳过、沿用上一次发布的系数）。

        当前 ``(q0, dq0)`` 取自 ``arm.get_arm_state()``（硬件无加速度反馈，
        ``ddq0`` 恒 0）；时长 ``T`` = 目标绝对时刻 − 当前时刻（滚动重规划自适应，
        已过期目标以最短时长 ``1e-3`` s 规划、近似立即到位）。多项式系数经
        3×3 边界方程组向量化解算（``c0=q0``、``c1=dq0``、``c2=ddq0/2`` 已知，
        未知 ``c3..c5`` 由末端位置/速度/加速度三条件唯一确定）。
        """
        targets = [target] if isinstance(target, TrajFrame) else list(target or [])
        if len(targets) > 1:
            # 多帧告警（各帧到达时刻序列不变则只告警一次，防滚动重规划刷屏）
            times = tuple(round(float(t.time), 3) for t in targets)
            if times != self._last_multi_warn:
                logger.warning(
                    "traj_planner_to_joint.py - ToJointTrajPlanner.plan：目标"
                    f"帧数 {len(targets)} > 1（各帧到达时刻 {times}），本规划器仅支持"
                    "单帧目标——多目标连续轨迹待教程章节实现，已回退规划到 q_home；"
                    "请每次只写一帧目标（set_target_traj 单帧），多余帧会被忽略")
                self._last_multi_warn = times
            targets = []
        if not targets:
            # 空目标回退：规划到 q_home（到达时间隐含 1 rad/s 的回家速度）
            try:
                st = arm.get_arm_state().joint
                q0 = np.asarray(st.q, dtype=float).reshape(-1)
                q_home = np.asarray(arm.arm_home, dtype=float).reshape(-1)
                targets = [TrajFrame(
                    time=_time.time() + float(np.max(np.abs(q_home - q0))),
                    q=q_home.copy())]
            except Exception as e:
                # 构造回退帧失败（如未连接真机读不到状态）：本周期跳过，
                # 沿用上一次发布的系数；异常细节留 debug 级，避免离线期刷屏
                logger.debug("traj_planner_to_joint.py - ToJointTrajPlanner.plan："
                             "回退帧构造失败，跳过本周期：%s", e)
                return False
        tg = targets[0]
        if tg.q is None:
            raise ValueError(
                "traj_planner_to_joint.py - ToJointTrajPlanner.plan：目标为笛卡尔"
                "位姿型（pose），本规划器仅支持关节目标 q（可带 dq/ddq，其余字段"
                "不使用、保持 None）；笛卡尔目标需 ikine 逆解管线（待教程章节）")
        q1 = np.asarray(tg.q, dtype=float).reshape(-1)
        dq1 = np.zeros_like(q1) if tg.dq is None \
            else np.asarray(tg.dq, dtype=float).reshape(-1)
        ddq1 = np.zeros_like(q1) if tg.ddq is None \
            else np.asarray(tg.ddq, dtype=float).reshape(-1)
        st = arm.get_arm_state().joint
        q0 = np.asarray(st.q, dtype=float).reshape(-1)
        dq0 = np.asarray(st.dq, dtype=float).reshape(-1)
        ddq0 = np.zeros_like(q0)                            # 无加速度反馈
        T = max(float(tg.time) - _time.time(), 1e-3)        # 防零/负时长除零

        # 未知系数 (c3,c4,c5)：末端三条件（每关节同系数矩阵、向量化右端）
        A = np.array([
            [T ** 3, T ** 4, T ** 5],
            [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
            [6 * T, 12 * T ** 2, 20 * T ** 3],
        ])
        b = np.vstack([
            q1 - q0 - dq0 * T - 0.5 * ddq0 * T ** 2,        # q(T) = q1
            dq1 - dq0 - ddq0 * T,                           # q'(T) = dq1
            ddq1 - ddq0,                                    # q''(T) = ddq1
        ])                                                  # (3, n)
        c345 = np.linalg.solve(A, b)                        # (3, n)
        # 打包不可变、一次性原子发布（并发契约）
        self._coeffs = (_time.time(), T,
                        np.vstack([q0, dq0, 0.5 * ddq0]),   # c0..c2 (3,n)
                        c345)
        return True

    def sample(self, t: float) -> TrajFrame:
        """采样：按绝对时间求值，``τ`` 钳位 ``[0, T]``（超末端保持末端状态），
        输出 ``q/dq/ddq``（多项式及其一、二阶导数）。"""
        if self._coeffs is None:
            raise RuntimeError(
                "traj_planner_to_joint.py - ToJointTrajPlanner.sample："
                "尚未规划（先经 plan 规划）")
        t0, T, c012, c345 = self._coeffs
        tau = min(max(float(t) - t0, 0.0), T)
        p = np.array([1.0, tau, tau ** 2])                  # τ^0..τ^2
        dp = np.array([0.0, 1.0, 2.0 * tau])                # d/dτ τ^0..τ^2
        ddp = np.array([0.0, 0.0, 2.0])                     # d²/dτ²
        p3 = np.array([tau ** 3, tau ** 4, tau ** 5])
        dp3 = np.array([3 * tau ** 2, 4 * tau ** 3, 5 * tau ** 4])
        ddp3 = np.array([6 * tau, 12 * tau ** 2, 20 * tau ** 3])
        return TrajFrame(
            time=float(t),
            q=p @ c012 + p3 @ c345,
            dq=dp @ c012 + dp3 @ c345,
            ddq=ddp @ c012 + ddp3 @ c345,
        )
