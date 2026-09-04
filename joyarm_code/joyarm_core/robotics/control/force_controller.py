"""ForceController —— 力控制律（与 ForceTrajPlanner 配套，Ch6/Ch9 教学内容桩）。

两种控制类型经 config ``type`` 字段选择：
1. ``impedance``（末端 OSC 阻抗）：重力补偿 + 末端位姿 OSC 阻抗控制——柔顺效果
   （结合雅可比与动力学参数，可适当简化如略去科氏）；
2. ``hybrid``（力位混合）：在轨迹 ``wrench`` 的力方向做力控与力补偿（位置超限则
   退回，避免空推），其余方向位置控制（有一定柔性、但不明显柔性）。

Ch6/Ch9 实现后经 ``REGISTRY`` 注册（``name="force"``）。
"""
from __future__ import annotations

from ...utils.types import ArmState, ControlMode, TrajFrame
from .controller import Controller

__all__ = ["ForceController"]


class ForceController(Controller):
    """力控制律：config ``type`` 选阻抗/力位混合（Ch6/Ch9 实现）。"""

    def __init__(self, type: str = "impedance", **kw):
        """控制类型经 config ``type`` 选择（两种取值见模块 docstring）；
        其余参数（``ctrl_hz`` 等）透传基类。"""
        super().__init__(**kw)
        self.type = str(type)

    def _compute(self, arm, frame: TrajFrame, state: ArmState,
                 **kw) -> tuple[ControlMode, dict]:
        raise NotImplementedError("ForceController 为 Ch6/Ch9 教学内容")

    # 实现概要（函数头草案，实现时取消注释并补充算法体）：
    # def ctrl_impedance(self, arm, frame, state):
    #     """末端 OSC 阻抗：F = K·(x_ref−x) + D·(ẋ_ref−ẋ)，τ = JᵀF + 重力补偿（可简化）。"""
    # def ctrl_hybrid(self, arm, frame, state):
    #     """力位混合：wrench 力方向力控/力补偿（位置超限退回防空推），其余方向位置控。"""
