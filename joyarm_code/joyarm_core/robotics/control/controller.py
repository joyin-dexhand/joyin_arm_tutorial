"""Controller —— 控制律接口（ABC）

控制律子类需实现内核
- :meth:`_compute`（由当前轨迹帧 + 实测状态算控制值，返回 (控制模式, 指令字典)）。

通用模板 :meth:`step` 代劳状态读取、限位守卫与指令下发（经 ``arm.set_arm_command``）。
config ``robotics.control`` 段写注册名，即按名实例化装入 ``_controllers`` 成员字典。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from ...utils.limits import clamp_to_limits
from ...utils.types import ArmState, ControlMode, TrajFrame

__all__ = ["Controller"]


class Controller(ABC):
    """控制律策略接口：当前轨迹帧 + 实测状态 → 电机控制指令。"""

    def __init__(self, ctrl_hz: float = 200.0):
        """控制频率为控制器参数，经 config ``robotics.control`` spec 的 ``**params`` 注入。

        :param ctrl_hz: 控制频率（``step`` 的调用频率），Hz。
        """
        self.ctrl_hz = float(ctrl_hz)

    def step(self, arm, frame: TrajFrame, state: Optional[ArmState] = None) -> None:
        """模板：状态缺省现读 → 内核算控制值 → 限位守卫 → ``arm.set_arm_command`` 下发。

        :param frame: 当前轨迹帧（``arm.get_current_frame()`` 产物）。
        :param state: 实测状态快照；缺省现读 ``arm.get_arm_state()``。
        """
        if state is None:
            state = arm.get_arm_state()
        mode, cmd = self._compute(arm, frame, state)
        if cmd.get("q") is not None:    # 位置指令裁剪到软限位（防轨迹跳变越限）
            cmd["q"] = clamp_to_limits(cmd["q"], arm.joint_limits_soft)
        if cmd.get("dq") is not None:   # 速度指令幅值裁剪（防过速）
            dq_max = arm.joint_limits_soft.dq_max
            cmd["dq"] = np.clip(cmd["dq"], -dq_max, dq_max)
        arm.set_arm_command(mode, **cmd)

    @abstractmethod
    def _compute(self, arm, frame: TrajFrame, state: ArmState,
                 **kw) -> Tuple[ControlMode, dict]:
        """内核：由参考帧与实测状态算控制值。

        :return: ``(控制模式, 指令字典)``——字典键取 ``q``/``dq``/``tau``/``kp``/``kd``
            （与 ``arm.set_arm_command`` 参数对应，未用键缺省）。
        """
