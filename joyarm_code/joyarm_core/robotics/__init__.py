"""``joyarm_core.robotics`` —— 算法层子包（一域一子包，策略接口）。

仅依赖 utils 与 numpy（Pin 默认实现另需 pinocchio；按属性约定消费 ``arm``，不 import joyarm）。每域 = 策略 ABC + ``REGISTRY`` 注册表；
JoyArm 构造时按 config ``robotics:`` 段查表组装私有成员字典（``_fkine_solvers`` 等），算法间互调走公开门面。

子类实现：
继承域 ABC 后在所在域 ``REGISTRY`` 显式加一行注册（import 类 + 表中一行），
config 按名引用即可接入 JoyArm（改 config 即换算法）：

    - fkine：正运动学
    - ikine：逆运动学
    - jacobian：速度运动学与静力学
    - dynamics：逆动力学与 M/C/G
    - trajectory：轨迹规划
    - control：控制律
"""
from .fkine import FkineSolver, PinFkineSolver
from .ikine import IkineSolver, PinIkineSolver
from .jacobian import JacobianSolver, PinJacobianSolver
from .dynamics import DynamicsSolver, PinDynamicsSolver
from .trajectory import TrajPlanner, ToJointTrajPlanner
from .control import Controller, JointPositionController

__all__ = [
    "FkineSolver",
    "PinFkineSolver",
    "IkineSolver",
    "PinIkineSolver",
    "JacobianSolver",
    "PinJacobianSolver",
    "DynamicsSolver",
    "PinDynamicsSolver",
    "TrajPlanner",
    "ToJointTrajPlanner",
    "Controller",
    "JointPositionController",
]
