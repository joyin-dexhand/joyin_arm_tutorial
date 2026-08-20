"""``joyarm_core.joyarms`` —— 设备模型层子包。

- :class:`JoyArm`（:mod:`joyarm_core.joyarms.joyarm`）：完整机械臂基类 = 多轴本体 + 末端执行器；
  直接持有 ``backend_arm`` / ``backend_end``，提供运动学算法门面与本体/末端控制。
- :class:`JoyArmDM`（:mod:`joyarm_core.joyarms.joyarm_dm`）：具体机械臂型号。

"""
from .joyarm import JoyArm
from .joyarm_dm import JoyArmDM

__all__ = ["JoyArm", "JoyArmDM"]
