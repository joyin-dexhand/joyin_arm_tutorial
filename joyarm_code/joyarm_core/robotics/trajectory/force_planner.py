"""ForceTrajPlanner —— 力规划器（阻抗/导纳/力位混合，Ch5/Ch9 教学内容桩）。

目标携带 ``wrench``（末端期望力/力矩）时的规划策略：
- **阻抗**（impedance）：期望柔顺动力学 ``F = K·x̃ + D·ẋ̃``（位控为主，刚度/阻尼可调）；
- **导纳**（admittance）：由测得外力修正期望轨迹 ``ẍ = M⁻¹(F_ext − F_d − D·ẋ − K·x)``；
- **力位混合**（hybrid）：任务空间按选择矩阵划分位控/力控子空间。

Ch5/Ch9 实现后经 ``REGISTRY`` 注册（``name="force"``）。
"""
from __future__ import annotations

from typing import List

from ...utils.types import TrajFrame
from .traj_planner import TrajPlanner

__all__ = ["ForceTrajPlanner"]


class ForceTrajPlanner(TrajPlanner):
    """力规划器：阻抗 / 导纳 / 力位混合（Ch5/Ch9 实现）。"""

    def _plan(self, arm, targets: List[TrajFrame], **kw) -> None:
        raise NotImplementedError("ForceTrajPlanner 未实现")

    def sample_frame(self, t_abs: float) -> TrajFrame:
        raise NotImplementedError("ForceTrajPlanner 未实现")

    # def plan_impedance(self, arm, target: TrajFrame) -> None:
    #     """阻抗规划：期望柔顺动力学参数 → 参考轨迹 + 力矩前馈。"""
    # def plan_admittance(self, arm, target: TrajFrame) -> None:
    #     """导纳规划：由外力测量实时修正期望轨迹（位置偏移）。"""
    # def plan_hybrid(self, arm, target: TrajFrame) -> None:
    #     """力位混合：选择矩阵划分位控/力控子空间，分轴规划。"""
