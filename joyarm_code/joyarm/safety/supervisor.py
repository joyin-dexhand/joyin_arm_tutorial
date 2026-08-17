"""跨层监控与安全策略（§11.4）。

:class:`StateMonitor` 消费 :class:`~joyarm.utils.types.ArmState` 输出
:class:`list[Violation]`（内部三层：关节 §11.1 / 末端 §11.2 / 整机 §11.3），
:class:`SafetySupervisor` 据此发
:class:`~joyarm.utils.types.SafetyAction`（阻尼保持 / 构型维持 / 急停等
策略可换）。

对应章节：Ch11（``chapt11_safety.md`` §11.4）。当前状态：仅签名 + docstring +
``raise NotImplementedError("Ch11 实现")``。
"""
from __future__ import annotations

from typing import List, Optional

from ..utils.types import (
    ArmState,
    JointLimits,
    SafetyAction,
    TcpLimits,
    Violation,
)

__all__ = ["StateMonitor", "SafetySupervisor"]


class StateMonitor:
    """状态监控器：注册限位集 → 消费 :class:`ArmState` → 输出违规列表。

    内部三层：关节（§11.1）/ 末端（§11.2）/ 整机（§11.3）。
    """

    def __init__(self, **kwargs):
        self._joint_limits: Optional[JointLimits] = None
        self._tcp_limits: Optional[TcpLimits] = None

    def set_limits(self, joint_limits: JointLimits, tcp_limits: TcpLimits) -> None:
        """注册关节 / 末端限位集。"""
        # 占位：Ch11 实现——把限位阈值存下来，供后续 check 比对。
        raise NotImplementedError("StateMonitor.set_limits 待 Ch11 实现")

    def check(self, state: ArmState) -> List[Violation]:
        """检查一次状态快照，返回违规列表。"""
        # 占位：Ch11 实现——拿当前状态和限位逐项比对，超出的项汇总成违规列表。
        raise NotImplementedError("StateMonitor.check 待 Ch11 实现")


class SafetySupervisor:
    """§11.4 安全策略：订阅违规 → 发 :class:`SafetyAction`。

    内置可换策略：
    - ``DampingHoldStrategy``：阻尼力矩保持。
    - ``ConfigFreezeStrategy``：构型维持。
    - ``EStopStrategy``：紧急停止。
    """

    def __init__(self, monitor: StateMonitor, **kwargs):
        self.monitor = monitor

    def decide(self, violations: List[Violation]) -> SafetyAction:
        """根据违规列表决定响应动作。"""
        # 占位：Ch11 实现——按违规严重程度映射到响应策略（轻微→裁剪，严重→急停）。
        raise NotImplementedError("SafetySupervisor.decide 待 Ch11 实现")
