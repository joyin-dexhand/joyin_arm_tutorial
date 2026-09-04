"""AutoTrajPlanner —— 自动规划器（目标驱动，适用多种位置控制）。

按目标类型自动选择插值方式（姿态四元数一律球面插值）：

- 关节目标单值（``q``）：``q0 → q`` 三次多项式插值；
- 关节目标序列（``q`` 列表）：速度启发式三次多项式插值（途经点速度由有限差分估计）；
- 位姿目标单值（``pose``）：xyz 三次多项式插值 + 姿态 slerp；同时 IK 求 ``q``，``q0 → q`` 三次多项式插值（笛卡尔/关节双参考）；
- 位姿目标序列（``pose`` 列表）：xyz 速度启发式三次多项式 + 姿态 slerp（段间）；同时IK 求 ``q`` 序列后速度启发式三次多项式插值。

实现后经 ``REGISTRY`` 注册（``name="auto"``）。
"""
from __future__ import annotations

from typing import List

from ...utils.types import TrajFrame
from .traj_planner import TrajPlanner

__all__ = ["AutoTrajPlanner"]


class AutoTrajPlanner(TrajPlanner):
    """自动规划器：目标类型驱动——关节/位姿 × 单值/序列 四种情形。"""

    def _plan(self, arm, targets: List[TrajFrame], **kw) -> None:
        raise NotImplementedError("AutoTrajPlanner 未实现")

    def sample_frame(self, t_abs: float) -> TrajFrame:
        raise NotImplementedError("AutoTrajPlanner 未实现")

    # def plan_joint_to_point(self, arm, q0, target: TrajFrame) -> None:
    #     """q0→q 三次多项式插值：当前关节角到目标关节角。"""
    # def plan_joint_waypoint(self, arm, q0, targets: List[TrajFrame]) -> None:
    #     """多途经点速度启发式三次插值：途经点速度由有限差分估计。"""
    # def plan_pose_to_point(self, arm, pose0, q0, target: TrajFrame) -> None:
    #     """xyz 三次插值 + 姿态 slerp；IK 求 q 后 q0→q 三次插值。"""
    # def plan_pose_waypoint(self, arm, pose0, q0, targets: List[TrajFrame]) -> None:
    #     """xyz 速度启发式三次 + 姿态 slerp；IK 求 q 序列后速度启发式三次插值。"""
