"""整机层安全（§11.3）：自碰撞检测（URDF 连杆 + FK）。

对应章节：Ch11（``chapt11_safety.md`` §11.3）。当前状态：仅签名 + docstring +
``raise NotImplementedError("Ch11 实现")``。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

__all__ = ["CollisionReport", "SelfCollisionChecker"]


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
        # （Arm 基本能力，与 fkine/ikine 同源），再检测任意两段是否相交（BVH/SDF 加速）。
        raise NotImplementedError("SelfCollisionChecker.check 待 Ch11 实现")
