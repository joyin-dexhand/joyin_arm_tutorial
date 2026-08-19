"""末端层安全（§11.2）：TCP 限位校验。

对应章节：Ch11（``chapt11_safety.md`` §11.2）。当前状态：仅签名 + docstring +
``raise NotImplementedError("Ch11 实现")``。
"""
from __future__ import annotations

from typing import List

from ..utils.types import ArmState, TcpLimits, Violation

__all__ = ["tcp_limits_check"]


def tcp_limits_check(state: ArmState, limits: TcpLimits) -> List[Violation]:
    """纯函数：末端层限位校验。

    校验项：位置范围（工作空间包围盒）/ 线速度 / 角速度 / 力 / 力矩上限。

    :return: 违规列表（``layer="tcp"``）。
    """
    # 占位：Ch11 实现——检查末端位置/速度/力是否超出工作空间和力上限。
    raise NotImplementedError("tcp_limits_check 待 Ch11 实现")
