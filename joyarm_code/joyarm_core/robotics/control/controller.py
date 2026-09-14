"""Controller —— 控制律内核基类（ABC，纯计算、无线程）

内核 :meth:`compute`：当前轨迹帧 + 关节状态 → (控制模式, 指令字典)；
类属性 :attr:`Controller.MODE` 声明本控制律需要的电机控制模式。

周期调度（ctrl_hz 线程、运行门控、指令下发、模式管理）由 JoyArm 运动管线
负责（``start_motion``/``stop_motion`` 启停、运行期 ``set_solver`` 热切换、
按 ``MODE`` 自动切换电机模式）。

config ``robotics.control`` 段写注册名，即按名实例化装入 ``_controllers`` 成员字典。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from ...utils.types import ArmState, ControlMode, TrajFrame

__all__ = ["Controller"]


class Controller(ABC):
    """控制律内核（纯计算）：当前轨迹帧 + 关节状态 → 控制指令。

    :cvar MODE: 本控制律需要的电机控制模式——运动管线在启动/热切换时自动``set_mode_arm``；子类按需覆写。
    """

    MODE: ControlMode = ControlMode.MIT

    # ----------------------------------------------------------
    # ctrl 构造（ctrl_hz；经 config 注入）
    # ----------------------------------------------------------
    def __init__(self, ctrl_hz: float = 200.0):
        """控制频率经 config ``robotics.control`` 的 ``**params`` 注入。

        :param ctrl_hz: 控制频率——管线 ctrl-step 线程按此属性起节拍（切换到
            不同频率的控制器时热重整），Hz。
        :raises ValueError: 频率非正（config 手误在构造时立即暴露，避免线程
            静默死亡或零间歇狂发）。
        """
        if ctrl_hz <= 0:
            raise ValueError(
                f"controller.py - Controller.__init__：ctrl_hz 须为正数"
                f"（收到 {ctrl_hz}）")
        self.ctrl_hz = float(ctrl_hz)

    # ----------------------------------------------------------
    # ctrl 计算内核（公开；单步可直接调用，亦供外部驱动（如 ROS2 节点）复用）
    # ----------------------------------------------------------
    @abstractmethod
    def compute(self, arm, frame: TrajFrame,
                state: ArmState) -> Tuple[ControlMode, dict]:
        """内核：由当前轨迹帧与关节状态计算控制指令。

        :return: ``(控制模式, 指令字典)``——模式须与本类 ``MODE`` 声明一致
            （管线已按 ``MODE`` 切好电机模式）；字典键取 ``q``/``dq``/``tau``/
            ``kp``/``kd``（与 ``arm.set_arm_command`` 参数对应，未用键缺省）；
            指令**原样透传**，限位守卫统一在后端基类 ``send_*`` 模板。
        """
