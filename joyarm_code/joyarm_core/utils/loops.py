"""loops —— 周期线程的公共实现。

控制器（ctrl-step）、轨迹规划（traj-plan / traj-sample）、状态保活
（joyarm-state）共用同一套「固定频率循环」机制，统一在本模块实现：

- :func:`run_periodic`：循环体——绝对时间累加节拍（单步超时只错过不追赶），
  单步失败记日志后下一周期重试（日志按时间节流，避免高频刷屏）。
- :class:`PeriodicThread`：线程的启动/停止/重启生命周期封装；``stop()`` 会
  确认线程真的退出，未退出时抛错并保留引用，防止重复 start 造成双循环。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

__all__ = ["run_periodic", "PeriodicThread"]

logger = logging.getLogger("joyarm_core.loops")

_ERR_LOG_INTERVAL: float = 0.5   # 单步失败日志的最小间隔（秒）
_JOIN_TIMEOUT: float = 1.0      # stop() 等待线程退出的上限（秒）


def run_periodic(stop: threading.Event, hz: float, step: Callable[[], None],
                 name: str = "") -> None:
    """按 ``hz`` 频率循环执行 ``step()``，直到 ``stop`` 置位。

    节拍用绝对时间累加：某次执行超时只错过该拍、后续重新对齐，不会为赶
    进度连续快跑。单步抛异常时记日志并继续下一周期；同一故障持续出现时
    日志按 0.5 秒节流（期间静默计数，下次记录时附上被省略的次数）。

    :param stop: 停止事件（置位即退出循环）。
    :param hz: 执行频率，Hz。
    :param step: 单步函数（无参数；内部异常视为「本步失败，下周期重试」）。
    :param name: 日志中标识本循环的名字（如线程名）。
    """
    period = 1.0 / float(hz)
    deadline = time.perf_counter()
    last_log = float("-inf")
    suppressed = 0
    while not stop.is_set():
        deadline += period
        now = time.perf_counter()
        if deadline > now:
            stop.wait(deadline - now)
        else:                              # 单步超时错过节拍：重新对齐，不追赶
            deadline = now
        try:
            step()
        except Exception:
            if time.monotonic() - last_log >= _ERR_LOG_INTERVAL:
                detail = f"（此前 {suppressed} 次同类失败已节流省略）" \
                         if suppressed else ""
                logger.exception("loops.py - run_periodic：%s单步 %s 失败，"
                                 "下周期重试%s",
                                 f"[{name}] " if name else "",
                                 getattr(step, "__name__", step), detail)
                last_log = time.monotonic()
                suppressed = 0
            else:
                suppressed += 1


class PeriodicThread:
    """单个周期线程：``start()`` 启动、``stop()`` 停止回收，可重复启停。

    ``stop()`` 最多等 1 秒让线程退出；超时未退出时抛 ``RuntimeError`` 且
    **保留线程引用**（``is_alive()`` 仍为真、再次 ``start()`` 会被拒绝），
    避免在旧线程还卡在单步内时启动新线程、造成两个循环同时运行。
    """

    def __init__(self, step: Callable[[], None], hz: float,
                 name: str = "periodic") -> None:
        """:param step: 单步函数；:param hz: 频率 Hz；:param name: 线程名。"""
        self._step = step
        self._hz = float(hz)
        self._name = str(name)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return self._name

    def is_alive(self) -> bool:
        """线程是否仍在运行（从未启动视为否）。"""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """启动线程；已在运行（或上次 stop 未成功）时抛 ``RuntimeError``。"""
        if self._thread is not None:
            raise RuntimeError(
                f"loops.py - PeriodicThread.start：[{self._name}] 线程仍在运行"
                f"（或上次 stop 未成功退出），须先确认 stop() 完成")
        self._stop.clear()
        self._thread = threading.Thread(
            target=run_periodic,
            args=(self._stop, self._hz, self._step, self._name),
            daemon=True, name=self._name)
        self._thread.start()

    def stop(self) -> None:
        """停止线程；超时（默认 1s）未退出时抛 ``RuntimeError``（引用保留）。"""
        self._stop.set()
        th = self._thread
        if th is None:
            return
        th.join(timeout=_JOIN_TIMEOUT)
        if th.is_alive():
            raise RuntimeError(
                f"loops.py - PeriodicThread.stop：[{self._name}] 线程 "
                f"{_JOIN_TIMEOUT}s 内未退出（单步可能阻塞在总线 IO 上）；"
                f"在确认其退出前请勿重新 start()")
        self._thread = None
