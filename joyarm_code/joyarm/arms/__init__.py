"""``joyarm.arms`` —— 设备模型层子包（Mas / End / Arm）。

- :class:`Mas`（:mod:`joyarm.arms.mas`）：多轴本体（不含末端），运动学算法门面。
- :class:`End`（:mod:`joyarm.arms.end`）：末端执行器（夹爪/灵巧手/吸附…）。
- :class:`Arm`（:mod:`joyarm.arms.arm`）：完整臂 = ``Arm(Mas)`` 组合一个 ``End``。
- :class:`JoyArmRebotDM`（:mod:`joyarm.arms.joyarm_rebot_dm`）：具体型号预设。

继承链：``JoyArmRebotDM`` → ``Arm(Mas)``；``Arm`` 组合一个 ``End``。
``arm.fkine()`` 继承自 ``Mas``；``arm.end.open()`` 操作末端。
"""
from .mas import Mas
from .end import End
from .arm import Arm
from .joyarm_rebot_dm import JoyArmRebotDM

__all__ = ["Mas", "End", "Arm", "JoyArmRebotDM"]
