"""AutoController —— 自动控制律（与 AutoTrajPlanner 配套，Ch6 教学内容桩）。

五种控制类型经 config ``type`` 字段选择（其余参数同段配置）：
1. ``position_follow``（POSITION 模式）：直接跟随当前帧 ``q``；config 配速度上限
   （backend POS_VEL.vlim），防轨迹跳变引发过速危险；
2. ``velocity_follow``（VELOCITY 模式）：``dq + k·(q_ref − q)`` 下发；config 配
   关节扭矩/电流上限（backend 基类支持才配置，不支持则不配）；
3. ``drag_gravity``（MIT 模式，零力拖动示教）：重力补偿前馈 ``tau`` + 微小阻尼
   ``kd`` + 小 ``kp`` 原位保持——不拖时维持构型不漂移，稍用力即可拖动；
4. ``joint_mit``（MIT 模式，整臂关节控制）：完整 MIT 控制律
   ``tau = tau_ff + kp·(q_ref − q) + kd·(dq_ref − dq)``，``kp``/``kd`` 由 config 配置；
5. ``osc``（MIT 模式，整臂末端控制）：重力补偿 + 末端位姿 OSC 控制（结合雅可比
   与动力学参数，可适当简化如略去科氏；pose 非完全约束时经冗余/零空间规避奇异）。

Ch6 实现后经 ``REGISTRY`` 注册（``name="auto"``）。
"""
from __future__ import annotations

from ...utils.types import ArmState, ControlMode, TrajFrame
from .controller import Controller

__all__ = ["AutoController"]


class AutoController(Controller):
    """自动控制律：config ``type`` 选五种类型之一（Ch6 实现）。"""

    def __init__(self, type: str = "position_follow", **kw):
        """控制类型经 config ``type`` 选择（五种取值见模块 docstring）；
        其余参数（``ctrl_hz`` 等）透传基类。"""
        super().__init__(**kw)
        self.type = str(type)

    def _compute(self, arm, frame: TrajFrame, state: ArmState,
                 **kw) -> tuple[ControlMode, dict]:
        raise NotImplementedError(
            "auto_controller.py - AutoController._compute：Ch6 教学内容，尚未实现"
            "（五种控制律见类 docstring）")

    # Ch6 实现概要（函数头草案，实现时取消注释并补充算法体）：
    # def ctrl_position_follow(self, arm, frame, state):
    #     """POSITION 模式直接跟随：q = frame.q；vlim 速度上限防跳变过速（config）。"""
    # def ctrl_velocity_follow(self, arm, frame, state):
    #     """VELOCITY 模式：dq + k·(q_ref − q)；扭矩/电流上限（backend 支持才配）。"""
    # def ctrl_drag_gravity(self, arm, frame, state):
    #     """MIT 零力拖动：tau = 重力补偿前馈，微小 kd 阻尼 + 小 kp 原位保持。"""
    # def ctrl_joint_mit(self, arm, frame, state):
    #     """MIT 整臂关节控制：tau_ff + kp(q_ref−q) + kd(dq_ref−dq)（kp/kd = config）。"""
    # def ctrl_osc(self, arm, frame, state):
    #     """MIT 整臂末端 OSC：重力补偿 + 末端位姿 OSC（J/动力学可简化；零空间避奇异）。"""
