"""整机层监测（§11.2 / §11.3）：TCP 状态 + 自碰撞。

对应章节：Ch11（``chapt11_safety.md`` §11.2 / §11.3）。当前状态：仅签名 + docstring +
``raise NotImplementedError("Ch11 实现")``。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..utils.types import ArmState, TcpLimits, Violation

__all__ = ["tcp_limits_check", "CollisionReport", "SelfCollisionChecker"]


def tcp_limits_check(state: ArmState, limits: TcpLimits) -> List[Violation]:
    """纯函数：TCP 限位校验（§11.2）。

    校验项：位置范围（工作空间包围盒）/ 线速度 / 角速度 / 力 / 力矩上限。

    :return: 违规列表（``layer="tcp"``）。
    """
    # 占位：Ch11 实现——检查末端位置/速度/力是否超出工作空间和力上限。
    raise NotImplementedError("tcp_limits_check 待 Ch11 实现")


@dataclass
class CollisionReport:
    """自碰撞检测报告（§11.3）。

    占位：Ch11 实现——碰撞连杆对列表，可扩展最近距离等信息。
    """

    collision_pairs: list[Tuple[str, str]]


class SelfCollisionChecker:
    """§11.3 自碰撞检测：URDF 连杆 + FK（BVH / SDF 可插拔）。"""

    def check(self, arm, q) -> CollisionReport:
        """检查位形 ``q`` 是否自碰撞。

        :return: 碰撞报告（碰撞对列表）。
        """
        # 占位：Ch11 实现——各连杆位姿统一经 arm.frame_placement(q, frame=连杆帧) 获取
        # （JoyArm 基本能力，与 fkine/ikine 同源），再检测任意两段是否相交（BVH/SDF 加速）。
        raise NotImplementedError("SelfCollisionChecker.check 待 Ch11 实现")
