"""``joyarm_core.arms`` —— 设备模型层子包。

- :class:`Arm`（:mod:`joyarm_core.arms.arm`）：完整机械臂基类 = 多轴本体 + 末端执行器；
  直接持有 ``backend_mas`` / ``backend_end``，提供运动学算法门面与本体/末端控制。
- :class:`JoyArmRebotDM`（:mod:`joyarm_core.arms.joyarm_rebot_dm`）：具体机械臂型号。

"""
from .arm import Arm
from .joyarm_rebot_dm import JoyArmRebotDM

__all__ = ["Arm", "JoyArmRebotDM"]
