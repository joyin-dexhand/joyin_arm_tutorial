"""动力学：求解器接口 + 注册表。

- ``DynamicsSolver``：求解器接口（ABC，见 dynamics_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``
        新增实现：import 后在本表加一行（注册名 = 类名小写+下划线）， config 文件 ``robotics.dynamics``段按名引用。
"""
from .dynamics_solver import DynamicsSolver
from .dynamics_solver_pin import PinDynamicsSolver

REGISTRY: dict = {
    "pin_dynamics_solver": PinDynamicsSolver,   # 默认实现（首位）
}

__all__ = ["DynamicsSolver", "PinDynamicsSolver", "REGISTRY"]
