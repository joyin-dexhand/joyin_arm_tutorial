"""``joyarm_core.robotics`` —— 算法层（六个子包，策略接口）。

仅依赖 utils 与 numpy（Pin 默认实现另需 pinocchio 等）。
各子包内：策略 ABC + 子类实现 + ``REGISTRY`` 注册表。

各子类实现步骤详见各子包的 ``__init__.py``。
"""
from .fkine import FkineSolver, PinFkineSolver
from .ikine import IkineSolver, PinIkineSolver
from .jacobian import JacobianSolver, PinJacobianSolver
from .dynamics import DynamicsSolver, PinDynamicsSolver
from .trajectory import TrajPlanner, SingleJointHoldTrajPlanner, SingleJointCubicTrajPlanner
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
    "SingleJointHoldTrajPlanner",
    "SingleJointCubicTrajPlanner",
    "Controller",
    "JointPositionController",
]
