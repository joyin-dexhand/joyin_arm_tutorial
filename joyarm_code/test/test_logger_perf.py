"""logger.warning 性能基准：100 次调用的总消耗时间。

覆盖：①默认状态（WARNING 生效、实际输出）总耗时；②WARNING 被级别
过滤（ERROR 级）时的耗时；③挂自定义 Handler（列表收集）时的耗时。
三组对照用于评估告警在控制回路热路径中的开销。

运行：``python test/test_logger_perf.py`` 或 pytest。
"""
from __future__ import annotations

import logging
import time

_N = 100  # 每场景调用次数


class _Cap(logging.Handler):
    """列表收集 Handler（同 test_backend_base.py 的日志捕获模式）。"""

    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def _time_warnings(logger: logging.Logger, n: int = _N) -> float:
    """计时 n 次 ``logger.warning``（懒 % 格式化，贴合库内用法）→ 总耗时秒。

    先做 1 次不计入的预热，避免首次懒初始化干扰；用 perf_counter 而非库内
    惯用的 time.time()——微秒级基准需要单调高精度时钟。
    """
    logger.warning("性能测试预热 %d/%d：不计入计时", 0, n)
    t0 = time.perf_counter()
    for i in range(n):
        logger.warning("性能测试 %d/%d：模拟越限告警", i + 1, n)
    return time.perf_counter() - t0


def _report(tag: str, total: float, n: int = _N) -> None:
    print(f"  [{tag}] {n} 次 logger.warning 总耗时 {total * 1e3:.3f} ms"
          f"（平均 {total / n * 1e6:.1f} µs/次）")


def test_warning_default():
    """默认状态：WARNING 生效并实际输出（库内未配置 Handler 时经 lastResort 写 stderr）。"""
    log = logging.getLogger("joyarm_core.backend")
    total = _time_warnings(log)
    _report("默认状态", total)
    assert 0.0 < total < 5.0, "『test_logger_perf.py - 默认状态计时：总耗时异常』"


def test_warning_suppressed():
    """级别过滤：临时升到 ERROR 级屏蔽 WARNING，仅剩级别检查开销。"""
    log = logging.getLogger("joyarm_core.backend")
    old = log.level
    log.setLevel(logging.ERROR)
    try:
        total = _time_warnings(log)
    finally:
        log.setLevel(old)
    _report("级别过滤", total)
    assert 0.0 < total < 5.0, "『test_logger_perf.py - 级别过滤计时：总耗时异常』"


def test_warning_custom_handler():
    """自定义 Handler：挂 _Cap 收集并关断传播，仅测该 Handler 的分发开销。"""
    log = logging.getLogger("joyarm_core.backend")
    cap = _Cap()
    old_prop = log.propagate
    log.addHandler(cap)
    log.propagate = False
    try:
        total = _time_warnings(log)
    finally:
        log.propagate = old_prop
        log.removeHandler(cap)
    _report("自定义 Handler", total)
    assert len(cap.msgs) == _N + 1  # 预热 1 次 + 计时 N 次
    assert 0.0 < total < 5.0, "『test_logger_perf.py - 自定义 Handler 计时：总耗时异常』"


# ----------------------------------------------------------
# 直跑
# ----------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"{len(fns)} 项全部通过")
