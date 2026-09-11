"""TrajPlanner —— 轨迹规划器基类（ABC）

双线程周期机制（循环体与日志节流由 :class:`joyarm_core.utils.loops.PeriodicThread`
统一提供）+ 可独立测试的单步内核：
- :meth:`plan_once`（plan_hz）：读轨迹目标 → 规则/超时校验剔除 → 空则回退 q_home → 委托内核 :meth:`_plan` 计算插值系数；
- :meth:`sample_once`（sample_hz）：规划成功后，按绝对时间采样 → 写轨迹当前帧（控制器读取消费）。

config ``robotics.traj`` 段写注册名，即按名实例化装入 ``_traj_planners`` 成员字典。
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ...utils.loops import PeriodicThread
from ...utils.types import TrajFrame

__all__ = ["TrajPlanner"]

logger = logging.getLogger("joyarm_core.traj_planner")


class TrajPlanner(ABC):
    """轨迹规划策略基类：目标序列（+当前状态）→ 插值系数 → 按绝对时间采样帧。"""

    # ----------------------------------------------------------
    # traj 构造（plan_hz/sample_hz/dt_min_required 经 config 注入）
    # ----------------------------------------------------------
    def __init__(self, plan_hz: float = 5.0, sample_hz: float = 200.0,
                 dt_min_required: float = 1.0):
        """三参数经 config ``robotics.traj`` 的 ``**params`` 注入。

        :param plan_hz: 规划频率（低频重规划：读桥目标 + 校验 + 重算系数），Hz。
        :param sample_hz: 采样频率（≈控制频率，按绝对时间采样产当前轨迹帧），Hz。
        :param dt_min_required: 目标最小提前量——时间早于``now + dt_min_required`` 的目标帧视为来不及执行，剔除。
        """
        self.plan_hz = float(plan_hz)
        self.sample_hz = float(sample_hz)
        self.dt_min_required = float(dt_min_required)
        self._arm = None                       # start(arm) 注入，线程体消费
        self._planned = False                  # 首帧规划成功门控（采样发布前置）
        self._threads: List[PeriodicThread] = []
        self._last_dropped: Optional[str] = None   # 上次剔除摘要（同批问题只告警一次）

    # ----------------------------------------------------------
    # traj 生命周期（双 daemon 线程：plan_hz 规划 + sample_hz 采样）
    # ----------------------------------------------------------
    def start(self, arm) -> None:
        """启动规划线程（plan_hz）与采样线程（sample_hz）；已运行（或上次
        stop 未成功）时抛 RuntimeError。

        注意：首个周期无有效目标即回退规划回 q_home（见 :meth:`plan_once`）——
        规划器启动后机械臂会立即向 home 运动，除非应用已写入有效目标。
        """
        if self._threads:
            raise RuntimeError(
                "traj_planner.py - TrajPlanner.start：规划器已在运行，须先 stop()")
        self._arm = arm
        self._planned = False
        self._last_dropped = None
        for name, hz, step in (("traj-plan", self.plan_hz, self._tick_plan),
                               ("traj-sample", self.sample_hz, self._tick_sample)):
            worker = PeriodicThread(step, hz, name=name)
            worker.start()
            self._threads.append(worker)

    def stop(self) -> None:
        """停止并回收两线程；有线程 1s 内未退出时抛 ``RuntimeError``（其引用
        保留在 ``_threads``、重新 :meth:`start` 会被拒绝，防止双循环）。"""
        stuck = [w.name for w in self._threads
                 if not self._try_stop(w)]
        self._threads = [w for w in self._threads if w.is_alive()]
        self._planned = False
        if stuck:
            raise RuntimeError(
                f"traj_planner.py - TrajPlanner.stop：线程 {stuck} 1s 内未退出"
                f"（单步可能阻塞在总线 IO 上）；在确认退出前请勿重新 start()")

    @staticmethod
    def _try_stop(worker: PeriodicThread) -> bool:
        """停止单个线程，返回是否成功退出（失败仅记日志，异常统一在 stop 汇总）。"""
        try:
            worker.stop()
            return True
        except RuntimeError as e:
            logger.error("traj_planner.py - TrajPlanner.stop：%s", e)
            return False

    def _tick_plan(self) -> None:
        """规划线程单步入口（无参，供 PeriodicThread 调用）。"""
        self.plan_once(self._arm)

    def _tick_sample(self) -> None:
        """采样线程单步入口（无参，供 PeriodicThread 调用）。"""
        self.sample_once(self._arm)

    # ----------------------------------------------------------
    # traj 单步内核（公开；亦可供外部驱动（如 ROS2 节点）复用）
    # ----------------------------------------------------------
    def plan_once(self, arm) -> None:
        """单步规划：读桥目标 → 逐帧规则校验（无效剔除）→ 超时剔除 → 空则回退
        q_home → 委托内核 :meth:`_plan`；成功后置采样发布门控。"""
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
                return
        self._plan(arm, kept)
        self._planned = True

    def sample_once(self, arm) -> None:
        """单步采样：首帧规划成功前不发布（保持桥「首帧前 None」契约）；之后按
        绝对时间采样并写桥当前帧（控制器读取）。"""
        if not self._planned:
            return
        arm.set_current_frame(self.sample_frame(time.time()))

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
        ``pose``/``q`` 恰一非空；``wrench`` 可选；``twist``/``dq``/``tau`` 必须为空。"""
        if not tg.time or not np.isfinite(tg.time):
            return "time 缺失或非有限数（绝对到达时间必填）"
        has_pose, has_q = tg.pose is not None, tg.q is not None
        if has_pose == has_q:
            return "pose/q 应恰有一个非空（当前" \
                   + ("双双为空" if not has_pose else "双双非空") + "）"
        for name in ("twist", "dq", "tau"):
            if getattr(tg, name) is not None:
                return f"{name} 应为空（目标不携带该字段）"
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
