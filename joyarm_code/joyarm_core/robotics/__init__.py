"""``joyarm_core.robotics`` —— 算法层子包（一域一子包，策略接口）。

仅依赖 utils 与 numpy（鸭子类型消费 ``arm``，不 import joyarm）。每域 = 策略
ABC + ``REGISTRY`` 注册表；JoyArm 构造时按 config ``robotics:`` 段查表组装私有
成员字典（``_fkine_solvers`` 等），算法互调走公开门面。

各域仅保留通用接口定义——具体算法为教程各章教学内容（fkine Ch2 / ikine Ch3 /
jacobian Ch4 / trajectory Ch5 / control Ch6、Ch8-9 / dynamics Ch8），章节实现后
经对应 ``REGISTRY`` 注册即接入 JoyArm（换 config 即换算法）：

- fkine：正运动学（Ch2）
- ikine：逆运动学（Ch3）
- jacobian：速度运动学与静力学（Ch4）
- trajectory：轨迹规划（Ch5）
- dynamics：逆动力学与 M/C/G（Ch8）
- control：控制律（Ch6/8/9）
"""
from .trajectory import Trajectory

__all__ = [
    "Trajectory",
]
