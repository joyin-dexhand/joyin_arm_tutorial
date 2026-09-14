"""TrajPlanner —— 轨迹规划内核基类（ABC，纯计算、无线程）

内核入口：
- :meth:`plan_once`：读轨迹目标 → 规则/超时校验剔除 → 空则回退 q_home → 委托内核 :meth:`_plan` 计算插值系数；
- :meth:`sample_frame`：按绝对时间评估系数产出当前帧。

周期调度（plan_hz/sample_hz 双线程）与采样发布门控由 JoyArm 运动管线负责
（``start_motion``/``stop_motion`` 启停、运行期 ``set_solver`` 热切换时同步
刷新一拍，当前帧立即可用）。

config ``robotics.traj`` 段写注册名，即按名实例化装入 ``_traj_planners`` 成员字典。
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ...utils.types import TrajFrame

__all__ = ["TrajPlanner"]

logger = logging.getLogger("joyarm_core.traj_planner")


class TrajPlanner(ABC):
    """轨迹规划内核（纯计算）：目标序列（+当前状态）→ 插值系数 → 按绝对时间采样帧。"""

    # ----------------------------------------------------------
    # traj 构造（plan_hz/sample_hz/dt_min_required 经 config 注入）
    # ----------------------------------------------------------
    def __init__(self, plan_hz: float = 5.0, sample_hz: float = 200.0,
                 dt_min_required: float = 1.0):
        """三参数经 config ``robotics.traj`` 的 ``**params`` 注入。

        :param plan_hz: 规划频率（低频重规划：读桥目标 + 校验 + 重算系数）——
            管线 traj-plan 线程按此属性起节拍，Hz。
        :param sample_hz: 采样频率（≈控制频率，按绝对时间采样产当前轨迹帧）——
            管线 traj-sample 线程按此属性起节拍，Hz。
        :param dt_min_required: 目标最小提前量——时间早于``now + dt_min_required`` 的目标帧视为来不及执行，剔除。
        """
        if plan_hz <= 0 or sample_hz <= 0:
            raise ValueError(
                f"traj_planner.py - TrajPlanner.__init__：频率须为正数"
                f"（plan_hz={plan_hz}, sample_hz={sample_hz}）")
        self.plan_hz = float(plan_hz)
        self.sample_hz = float(sample_hz)
        self.dt_min_required = float(dt_min_required)
        self._last_dropped: Optional[str] = None   # 上次剔除摘要（同批问题只告警一次）

    # ----------------------------------------------------------
    # traj 单步内核（公开；可直接调用，亦供外部驱动（如 ROS2 节点）复用）
    # ----------------------------------------------------------
    def plan_once(self, arm) -> bool:
        """单步规划：读目标 → 逐帧校验剔除→ 超时剔除 → 空则回退q_home → 委托内核 :meth:`_plan`。

        :return: 本次是否完成规划（``True`` = ``_plan`` 已被调用；回退帧构造失败等跳过路径返回 ``False``）。
        """
        now = time.time()
        kept: List[TrajFrame] = []
        dropped: List[str] = []
        for i, tg in enumerate(arm.get_target_traj() or []):
            problem = self._check_frame(tg)
            if problem is None and tg.time < now + self.dt_min_required:
                problem = f"time={tg.time:.3f} 早于 now+dt_min_required" \
                          f"={self.dt_min_required}s，来不及执行"
            if problem is None:
                kept.append(tg)
            else:
                dropped.append(f"[{i}] {problem}")
        if dropped:
            # 同一批无效/超时目标只告警一次（死目标源下防每周期 5Hz 刷屏）；
            # 剔除内容变化或中间恢复正常后再次出现时重新告警
            sig = "\n".join(dropped)
            if sig != self._last_dropped:
                logger.warning("traj_planner.py - TrajPlanner.plan_once：剔除无效/超时"
                               "目标帧：\n" + "\n".join(f"  - {d}" for d in dropped))
                self._last_dropped = sig
        else:
            self._last_dropped = None
        if not kept:                           # 空目标回退：q_home 兜底
            try:
                kept = [self._home_fallback(arm, now)]
            except Exception as e:
                # 构造回退帧失败（如未连接真机读不到状态）：本周期跳过，
                # 沿用上一次发布的系数；异常细节留 debug 级，避免离线期刷屏
                logger.debug("traj_planner.py - TrajPlanner.plan_once："
                             "回退帧构造失败，跳过本周期：%s", e)
                return False
        self._plan(arm, kept)
        return True

    # ----------------------------------------------------------
    # traj 抽象内核（_plan 系数 / sample_frame 按绝对时间采样帧）
    # ----------------------------------------------------------
    @abstractmethod
    def _plan(self, arm, targets: List[TrajFrame]) -> None:
        """内核：目标序列 + ``arm.get_arm_state()`` 当前状态 → 插值系数，锚定
        绝对时间（``time.time()`` 基准）。

        **并发契约**：系数须打包为单一不可变对象、一次性原子赋值发布（如
        ``self._coeffs = coeffs``），禁止原地修改已发布对象。
        """

    @abstractmethod
    def sample_frame(self, t_abs: float) -> TrajFrame:
        """内核：按绝对时间（Unix 秒）评估系数产出当前帧；``t_abs`` 超出轨迹
        末端时钳位到末帧（保持）。"""

    # ----------------------------------------------------------
    # traj 内部校验/回退（_check_frame 单帧判别；_home_fallback 兜底帧）
    # ----------------------------------------------------------
    @staticmethod
    def _check_frame(tg: TrajFrame) -> Optional[str]:
        """单帧有效性判别（返回问题描述，``None`` 为有效）：``time`` 必填且为
        有限数（``0.0`` 视为未填，NaN/inf 无法参与超时比较、一并剔除）；
        ``pose``/``q`` 恰一非空；关节目标（``q`` 非空）可携带可选 ``dq``/``ddq``
        速度/加速度参考；``twist``/``tau`` 目标不携带；pose 目标仅可带 ``wrench``。"""
        if not tg.time or not np.isfinite(tg.time):
            return "time 缺失或非有限数（绝对到达时间必填）"
        has_pose, has_q = tg.pose is not None, tg.q is not None
        if has_pose == has_q:
            return "pose/q 应恰有一个非空（当前" \
                   + ("双双为空" if not has_pose else "双双非空") + "）"
        if has_q:
            for name in ("twist", "tau"):
                if getattr(tg, name) is not None:
                    return f"{name} 应为空（关节目标不携带该字段）"
            if tg.wrench is not None:
                return "wrench 应为空（力参考仅随 pose 目标）"
            for name in ("q", "dq", "ddq"):             # NaN/inf 会毒化整条轨迹
                v = getattr(tg, name)
                if v is not None and not np.all(np.isfinite(np.asarray(v, dtype=float))):
                    return f"{name} 含非有限值（NaN/inf）"
        else:
            for name in ("twist", "dq", "ddq", "tau"):
                if getattr(tg, name) is not None:
                    return f"{name} 应为空（pose 目标不携带该字段）"
        return None

    @staticmethod
    def _home_fallback(arm, now: float) -> TrajFrame:
        """空目标回退帧：``q = arm_home``，到达时间 ``now + max|q_home − q0|``
        （即隐含 1 rad/s 的回家速度）。

        :raises RuntimeError: ``arm.get_arm_state()`` 失败（如未连接真机）。
        """
        q0 = np.asarray(arm.get_arm_state().joint.q, dtype=float).reshape(-1)
        q_home = np.asarray(arm.arm_home, dtype=float).reshape(-1)
        return TrajFrame(time=now + float(np.max(np.abs(q_home - q0))),
                         q=q_home.copy())
