"""TrajPlanner —— 轨迹规划器接口（ABC）

规划器子类需实现内核
- :meth:`_plan`（由目标序列计算插值参数并锚定绝对时间，参数存实例）；
- :meth:`sample_frame`（按绝对时间评估插值参数，产出当前轨迹帧 TrajFrame）。

config ``robotics.traj`` 段写注册名，即按名实例化装入 ``_traj_planners`` 成员字典。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union

from ...utils.types import TrajFrame

__all__ = ["TrajPlanner"]


class TrajPlanner(ABC):
    """轨迹规划策略接口：目标序列（+当前状态） → 计算插值参数 → 按绝对时间采样帧。"""

    # ----------------------------------------------------------
    # traj 构造与模板（plan_hz/sample_hz；plan：校验后委托内核）
    # ----------------------------------------------------------
    def __init__(self, plan_hz: float = 1.0, sample_hz: float = 200.0):
        """两频率为规划器参数，经 config ``robotics.traj`` 的 ``**params`` 注入。

        :param plan_hz: 规划频率（低频重规划），Hz。
        :param sample_hz: 采样频率（≈控制频率，按绝对时间采样产当前轨迹帧），Hz。
        """
        self.plan_hz = float(plan_hz)
        self.sample_hz = float(sample_hz)

    def plan(self, arm, targets: Union[TrajFrame, List[TrajFrame]], **kw) -> None:
        """模板：单帧归一为列表 → 目标有效性判别 → 委托内核 :meth:`_plan`。

        :param targets: 目标帧或目标帧列表（约束见 :meth:`_check_targets`）。
        :raises ValueError: 任一目标无效时抛出，消息列出全部问题。
        """
        targets = [targets] if isinstance(targets, TrajFrame) else list(targets)
        self._check_targets(targets)
        self._plan(arm, targets, **kw)

    # ----------------------------------------------------------
    # traj 抽象内核（_plan 插值参数 / sample_frame 按绝对时间采样帧）
    # ----------------------------------------------------------
    @abstractmethod
    def _plan(self, arm, targets: List[TrajFrame], **kw) -> None:
        """内核：由目标序列计算**插值参数**并锚定绝对时间（参数结构存实例，算法自定）。"""

    @abstractmethod
    def sample_frame(self, t_abs: float) -> TrajFrame:
        """内核：按绝对时间（Unix 秒）评估插值参数，产出当前轨迹帧。"""

    # ----------------------------------------------------------
    # traj 内部校验（_check_targets 目标有效性，收集全部问题一次抛出）
    # ----------------------------------------------------------
    @staticmethod
    def _check_targets(targets: List[TrajFrame]) -> None:
        """目标有效性判别（求解前执行）：``time`` 必填（``0.0`` 视为未填）；
        ``pose``/``q`` 恰一非空；``wrench`` 可选；``twist``/``dq``/``tau`` 必须为空。

        :raises ValueError: 违规时抛出，消息列出全部问题。
        """
        problems: List[str] = []
        for i, tg in enumerate(targets):
            if not tg.time:
                problems.append(f"[{i}] time 缺失（绝对到达时间必填）")
            has_pose, has_q = tg.pose is not None, tg.q is not None
            if has_pose == has_q:
                problems.append(f"[{i}] pose/q 应恰有一个非空（当前"
                                + ("双双为空" if not has_pose else "双双非空") + "）")
            for name in ("twist", "dq", "tau"):
                if getattr(tg, name) is not None:
                    problems.append(f"[{i}] {name} 应为空（目标不携带该字段）")
        if problems:
            raise ValueError(
                "目标序列无效：\n" + "\n".join(f"  - {p}" for p in problems))
