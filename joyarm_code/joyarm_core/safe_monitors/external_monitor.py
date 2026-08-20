"""外部层安全（§11.3）：外部碰撞检测（关节力矩残差）。

对应章节：Ch11（``chapt11_safety.md`` §11.3）。当前状态：仅签名 + docstring +
``raise NotImplementedError("Ch11 实现")``。
"""
from __future__ import annotations

from typing import List

from ..utils.types import ArmState, Violation

__all__ = ["ExternalCollisionDetector"]


class ExternalCollisionDetector:
    """§11.3 外部碰撞检测：基于关节力矩残差（实测 − Ch8 预期）阈值。"""

    def detect(self, state: ArmState, expected_tau) -> List[Violation]:
        """对比实测与预期力矩，超阈值则报告外部碰撞。"""
        # 占位：Ch11 实现——无外力时实测力矩应等于预期（idyn 算）；偏差过大说明碰到了东西。
        raise NotImplementedError("ExternalCollisionDetector.detect 待 Ch11 实现")
