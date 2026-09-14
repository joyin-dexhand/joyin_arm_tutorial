"""速度运动学与静力学：求解器接口 + 注册表。

- ``JacobianSolver``：求解器接口（ABC，见 jacobian_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``
        新增实现：import 后在本表加一行（注册名 = 类名小写+下划线）， config 文件 ``robotics.jacobian``段按名引用。
"""
from .jacobian_solver import JacobianSolver
from .jacobian_solver_pin import PinJacobianSolver

REGISTRY: dict = {
    "pin_jacobian_solver": PinJacobianSolver,   # 默认实现（首位）
}

__all__ = ["JacobianSolver", "PinJacobianSolver", "REGISTRY"]
