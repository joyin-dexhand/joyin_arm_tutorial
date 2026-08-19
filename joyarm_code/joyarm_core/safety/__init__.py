"""``joyarm_core.safety`` —— 安全层，按监控层级分文件。

- :mod:`joyarm_core.safety.joint`      ： 关节层——关节裁剪 + 限位校验。
- :mod:`joyarm_core.safety.tcp`        ： 末端层——TCP 限位校验。
- :mod:`joyarm_core.safety.machine`    ： 整机层——自碰撞检测（URDF + FK）。
- :mod:`joyarm_core.safety.external`   ： 外部层——外部碰撞检测（力矩残差）。
- :mod:`joyarm_core.safety.supervisor` ： 跨层——状态监控（StateMonitor）与安全策略。

"""
from .joint import clamp_to_limits, joint_limits_check
from .tcp import tcp_limits_check
from .machine import CollisionReport, SelfCollisionChecker
from .external import ExternalCollisionDetector
from .supervisor import StateMonitor, SafetySupervisor

__all__ = [
    "clamp_to_limits",
    "joint_limits_check",
    "tcp_limits_check",
    "CollisionReport",
    "SelfCollisionChecker",
    "ExternalCollisionDetector",
    "StateMonitor",
    "SafetySupervisor",
]
