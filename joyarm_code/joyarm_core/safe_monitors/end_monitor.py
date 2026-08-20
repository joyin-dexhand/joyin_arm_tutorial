"""执行器层监测（§11.x）：末端执行器（夹爪）状态校验。

对应章节：Ch11 / Ch13。当前状态：仅签名 + docstring +
``raise NotImplementedError("Ch11 实现")``。
"""
from __future__ import annotations

from typing import List, Optional

from ..utils.types import Violation

__all__ = ["end_limits_check"]


def end_limits_check(state, limits: Optional[dict] = None) -> List[Violation]:
    """纯函数：末端执行器（夹爪）状态校验。

    校验项（占位）：开度 / 夹持力 / 温度 / 电流上限，违规 ``layer="end"``。

    :param state: 执行器状态（来自 ``JoyArm.get_end_state()``）。
    :return: 违规列表。
    """
    # 占位：Ch11/Ch13 实现——校验夹爪开度/夹持力等是否超限。
    raise NotImplementedError("end_limits_check 待 Ch11 实现")
