"""Controller —— 控制器基类（ABC）

单线程周期机制（``start(arm)``/``stop()``，频率 ``ctrl_hz``，循环体与日志
节流由 :class:`joyarm_core.utils.loops.PeriodicThread` 统一提供）+ 可独立
测试的单步入口 :meth:`step_once`：**运行状态门控** → 读轨迹当前帧 → 读
``arm.get_arm_state()`` → 委托内核 :meth:`_compute` 算控制指令 →
``arm.set_arm_command`` 下发。

config ``robotics.control`` 段写注册名，即按名实例化装入 ``_controllers`` 成员字典。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from ...utils.loops import PeriodicThread
from ...utils.types import ArmState, ControlMode, TrajFrame

__all__ = ["Controller"]


class Controller(ABC):
    """控制策略基类：当前帧 + 当前状态 → 控制律 → 指令下发（运行状态门控）。"""

    # ----------------------------------------------------------
    # ctrl 构造（ctrl_hz；经 config 注入）
    # ----------------------------------------------------------
    def __init__(self, ctrl_hz: float = 200.0):
        """控制频率经 config ``robotics.control`` 的 ``**params`` 注入。

        :param ctrl_hz: 控制频率（``step_once`` 的执行频率），Hz。
        """
        self.ctrl_hz = float(ctrl_hz)
        self._arm = None                       # start(arm) 注入，线程体消费
        self._thread: Optional[PeriodicThread] = None

    # ----------------------------------------------------------
    # ctrl 生命周期（单 daemon 线程 @ctrl_hz）
    # ----------------------------------------------------------
    def start(self, arm) -> None:
        """启动控制线程（ctrl_hz）；已在运行（或上次 stop 未成功）时抛 RuntimeError。"""
        if self._thread is not None:
            raise RuntimeError(
                "controller.py - Controller.start：控制器已在运行，须先 stop()")
        self._arm = arm
        worker = PeriodicThread(self._tick, self.ctrl_hz, name="ctrl-step")
        worker.start()
        self._thread = worker

    def stop(self) -> None:
        """停止并回收控制线程；线程 1s 内未退出时抛 ``RuntimeError``（此时
        引用保留、重新 :meth:`start` 会被拒绝，防止双循环）。正常停止后可重启。"""
        if self._thread is not None:
            self._thread.stop()
            self._thread = None

    def _tick(self) -> None:
        """线程单步入口（无参，供 PeriodicThread 调用）。"""
        self.step_once(self._arm)

    # ----------------------------------------------------------
    # ctrl 单步内核（公开；亦可供外部驱动（如 ROS2 节点）复用）
    # ----------------------------------------------------------
    def step_once(self, arm) -> None:
        """单步控制：
        非正常运行状态（``arm.is_normal=False``）→ 跳过（不下发、不计算，急停/恢复由直连 safe_* 机制负责）；
        无当前帧（首帧发布前）→ 跳过；否则读当前帧 + 状态 → :meth:`_compute` → ``set_arm_command`` 下发。

        **模式契约**：``set_arm_command`` 要求当前控制模式与指令模式一致（否则
        每周期抛 ``RuntimeError``、指令饥饿）——应用须在启动管线前先
        ``arm.set_mode_arm()`` 切到控制器将下发的模式（谁切：应用，非本类）。
        """
        if not arm.is_normal:
            return
        frame = arm.get_current_frame()
        if frame is None:
            return
        state = arm.get_arm_state()
        mode, cmd = self._compute(arm, frame, state)
        arm.set_arm_command(mode, **cmd)

    # ----------------------------------------------------------
    # ctrl 抽象内核（_compute 控制律）
    # ----------------------------------------------------------
    @abstractmethod
    def _compute(self, arm, frame: TrajFrame,
                 state: ArmState) -> Tuple[ControlMode, dict]:
        """内核：控制律——由当前帧与当前状态计算指令。

        :return: ``(控制模式, 指令字典)``——字典键取 ``q``/``dq``/``tau``/
            ``kp``/``kd``（与 ``arm.set_arm_command`` 参数对应，未用键缺省）；
            指令**原样透传**，限位守卫统一在后端基类 ``send_*`` 模板。
            返回的模式须与应用事先 ``set_mode_arm`` 切换的模式一致（契约见
            :meth:`step_once`）——单周期内混切模式不在本层职责内。
        """
