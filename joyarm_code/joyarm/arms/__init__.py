"""``joyarm.arms`` —— 设备模型层子包（Arm / JoyArmRebotDM）。

- :class:`Arm`（:mod:`joyarm.arms.arm`）：完整机械臂基类 = 多轴本体 + 末端执行器；
  直接持有 ``backend_mas`` / ``backend_end``，提供运动学算法门面与本体/末端控制。
- :class:`JoyArmRebotDM`（:mod:`joyarm.arms.joyarm_rebot_dm`）：具体型号预设。

继承链：``JoyArmRebotDM`` → ``Arm``。
``arm.fkine()`` 为本体正运动学；``arm.end_open()`` 操作末端。
"""
from .arm import Arm
from .joyarm_rebot_dm import JoyArmRebotDM

__all__ = ["Arm", "JoyArmRebotDM"]
