"""``joyarm.safety`` —— 安全层子包（Ch11）。

三级安全监控：关节限位、TCP 限位、整机自碰撞（URDF + 正运动学）与外部
碰撞检测（力矩残差）。向上提供状态监控、碰撞检测与安全监督器：

- :class:`StateMonitor`：关节 / TCP 限位监控。
- :class:`SelfCollisionChecker`：整机自碰撞检测（输出 :class:`CollisionReport`）。
- :class:`ExternalCollisionDetector`：外部碰撞检测（力矩残差）。
- :class:`SafetySupervisor`：聚合上述检查器，输出安全动作决策。
"""
from .safety import (
    CollisionReport,
    StateMonitor,
    SelfCollisionChecker,
    ExternalCollisionDetector,
    SafetySupervisor,
    joint_limits_check,
    tcp_limits_check,
)

__all__ = [
    "CollisionReport",
    "StateMonitor",
    "SelfCollisionChecker",
    "ExternalCollisionDetector",
    "SafetySupervisor",
    "joint_limits_check",
    "tcp_limits_check",
]
