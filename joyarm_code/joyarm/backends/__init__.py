"""``joyarm.backends`` —— 真机通信后端子包。

本子包承载**真机通信**实现（架构设计第 4 条），按「机械臂 / 末端执行器」
拆为两条独立抽象，互不继承，便于未来灵活支持多类末端执行器：

**机械臂后端**（机械臂本体 6 个关节电机，ID ``0x01~0x06``）：

- :class:`~joyarm.backends.arm_backend.ArmBackend`：抽象基类。
- :class:`~joyarm.backends.joyarm_rebot_dm_backend.JoyArmRebotDMBackend`：
  JoyArm（reBot-DevArm）真机机械臂后端（CAN 总线），Ch6 实现。

**末端执行器后端**（夹爪 / 灵巧手 / 吸附装置等）：

- :class:`~joyarm.backends.end_backend.EndBackend`：抽象基类。
- :class:`~joyarm.backends.gripper_backend.GripperBackend`：两指夹爪真机
  通信后端（CAN 总线，夹爪 ID ``0x07``），Ch13 实现。

不设物理仿真后端：仿真 / 预演 / 教学走 ``Arm(backend=None)`` 离线模式 +
:mod:`joyarm.application.viz` 运动学可视化。
"""
from .arm_backend import ArmBackend
from .joyarm_rebot_dm_backend import JoyArmRebotDMBackend
from .end_backend import EndBackend
from .gripper_backend import GripperBackend

__all__ = ["ArmBackend", "JoyArmRebotDMBackend", "EndBackend", "GripperBackend"]
