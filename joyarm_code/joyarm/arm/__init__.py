"""``joyarm.arm`` —— 设备模型层子包。

承载机械臂本体与末端执行器的**设备模型**，与 :mod:`joyarm.backends`
通信层平行对应（设备模型包装对应 backend）：

- :class:`Arm`（:mod:`joyarm.arm.arm`）：机械臂基类，持有 pinocchio
  ``model``/``data`` 与可选真机后端（``backend=None`` 离线模式）。
- :class:`JoyArmRebotDM`（:mod:`joyarm.arm.joyarm_rebot_dm`）：JoyArm
  （reBot-DevArm）6 自由度具体型号，固化 URDF / 限位 / home。
- :class:`Gripper`（:mod:`joyarm.arm.gripper`）：两指夹爪设备模型，
  包装 :class:`~joyarm.backends.end_backend.EndBackend`（Ch13）。

平行结构：``Arm`` ↔ ``ArmBackend``、``Gripper`` ↔ ``EndBackend``。
"""
from .arm import Arm
from .joyarm_rebot_dm import JoyArmRebotDM
from .gripper import Gripper

__all__ = ["Arm", "JoyArmRebotDM", "Gripper"]
