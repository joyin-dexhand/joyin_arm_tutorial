"""``joyarm_core.safe_monitors`` —— 安全监测层，按监测对象分文件。

- :mod:`joyarm_core.safe_monitors.arm_monitor`       ： arm（本体关节）——关节裁剪 + 限位校验。
- :mod:`joyarm_core.safe_monitors.end_monitor`       ： end（执行器）——夹爪开度/夹持力校验。
- :mod:`joyarm_core.safe_monitors.joyarm_monitor`    ： joyarm（整机）——TCP 状态限位 + 自碰撞检测。
- :mod:`joyarm_core.safe_monitors.external_monitor`  ： external（外部）——外部碰撞检测（力矩残差）。
- :mod:`joyarm_core.safe_monitors.supervisor_monitor` ： 跨层——状态监控（StateMonitor）与安全策略。

"""
from .arm_monitor import clamp_to_limits, joint_limits_check
from .end_monitor import end_limits_check
from .joyarm_monitor import tcp_limits_check, CollisionReport, SelfCollisionChecker
from .external_monitor import ExternalCollisionDetector
from .supervisor_monitor import StateMonitor, SafetySupervisor

__all__ = [
    "clamp_to_limits",
    "joint_limits_check",
    "end_limits_check",
    "tcp_limits_check",
    "CollisionReport",
    "SelfCollisionChecker",
    "ExternalCollisionDetector",
    "StateMonitor",
    "SafetySupervisor",
]
