"""``joyarm_core.robotics`` —— 算法层子包（一域一子包，策略接口）。

仅依赖 utils 与 numpy（按属性约定消费 ``arm``，不 import joyarm）。每域 = 策略 ABC + ``REGISTRY`` 注册表；
JoyArm 构造时按 config ``robotics:`` 段查表组装私有成员字典（``_fkine_solvers`` 等），算法间互调走公开门面。

子类实现：
继承域 ABC 并加 ``@register`` 装饰器（见 ``_registry.py``）即自动注册，
config 按名引用即可接入 JoyArm（改 config 即换算法）：

- fkine：正运动学
- ikine：逆运动学
- jacobian：速度运动学与静力学
- dynamics：逆动力学与 M/C/G
- trajectory：轨迹规划
- control：控制律
"""
from .fkine import FkineSolver
from .ikine import IkineSolver
from .jacobian import JacobianSolver
from .dynamics import DynamicsSolver
from .trajectory import TrajPlanner
from .control import Controller

__all__ = [
    "FkineSolver",
    "IkineSolver",
    "JacobianSolver",
    "DynamicsSolver",
    "TrajPlanner",
    "Controller",
]
