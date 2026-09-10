"""Controller —— 控制器基类（ABC）

单线程周期机制（``start(arm)``/``stop()``）+ 可独立测试的单步内核：
- :meth:`step_once`（ctrl_hz）：**运行状态门控** → 读轨迹当前帧 → 读 ``arm.get_arm_state()`` 
    → 委托内核 :meth:`_compute` 算控制指令 → ``arm.set_arm_command`` 下发。

config ``robotics.control`` 段写注册名，即按名实例化装入 ``_controllers`` 成员字典。
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from ...utils.types import ArmState, ControlMode, TrajFrame

__all__ = ["Controller"]

logger = logging.getLogger("joyarm_core.controller")


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
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ----------------------------------------------------------
    # ctrl 生命周期（单 daemon 线程 @ctrl_hz）
    # ----------------------------------------------------------
    def start(self, arm) -> None:
        """启动控制线程（ctrl_hz）；已运行时抛 RuntimeError。"""
        if self._thread is not None:
            raise RuntimeError(
                "controller.py - Controller.start：控制器已在运行，须先 stop()")
        self._arm = arm
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="ctrl-step")
        self._thread.start()

    def stop(self) -> None:
        """停止并回收控制线程（join 超时 1s）；停止后可重新 :meth:`start`。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def _loop(self) -> None:
        """线程体：绝对 deadline 累加节拍（不忙自旋）；单步失败记日志下周期重试。"""
        period = 1.0 / self.ctrl_hz
        deadline = time.perf_counter()
        while not self._stop.is_set():
            deadline += period
            now = time.perf_counter()
            if deadline > now:
                self._stop.wait(deadline - now)
            else:                              # 单步超时错过节拍：重新对齐，不追赶
                deadline = now
            try:
                self.step_once(self._arm)
            except Exception:
                logger.exception("controller.py - Controller._loop：单步 step_once "
                                 "失败，下周期重试")

    # ----------------------------------------------------------
    # ctrl 单步内核（公开；亦可供外部驱动（如 ROS2 节点）复用）
    # ----------------------------------------------------------
    def step_once(self, arm) -> None:
        """单步控制：
        非正常运行状态（``arm.is_normal=False``）→ 跳过（不下发、不计算，急停/恢复由直连 safe_* 机制负责）；
        无当前帧（首帧发布前）→ 跳过；否则读当前帧 + 状态 → :meth:`_compute` → ``set_arm_command`` 下发。
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
        """
