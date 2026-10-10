"""动力学：基类接口 + 子类注册表。

- ``DynamicsSolver``：求解器接口（ABC，见 dynamics_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``

新子类实现步骤：
    1. 在本包内创建新的名为``dynamics_solver_<new>.py``的子类文件，实现子类 ``<New>DynamicsSolver``（继承自``dynamics_solver.py``的 ``DynamicsSolver``基类）；
    2. 在本文件中 import 新子类，并在 ``REGISTRY`` 显式加一行注册（注册名 = 类名小写+下划线）；
"""
from .dynamics_solver import DynamicsSolver
from .dynamics_solver_pin import PinDynamicsSolver

REGISTRY: dict = {
    "pin_dynamics_solver": PinDynamicsSolver,
}

__all__ = ["DynamicsSolver", "PinDynamicsSolver", "REGISTRY"]
