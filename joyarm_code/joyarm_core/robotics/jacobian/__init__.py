"""雅可比：基类接口 + 子类注册表。

- `JacobianSolver`：求解器接口（ABC，见 jacobian_solver.py）；
- `REGISTRY`：注册表 `{注册名: 求解器子类}`

新子类实现步骤：
    1. 在本包内创建 `jacobian_solver_<new>.py` 子类文件，实现子类 `<New>JacobianSolver`（继承自 `jacobian_solver.py` 的 `JacobianSolver` 基类）；
    2. 在本文件中 import 新子类，并在 `REGISTRY` 显式加一行注册（注册名 = 类名小写+下划线）；
"""
from .jacobian_solver import JacobianSolver
from .jacobian_solver_pin import PinJacobianSolver

REGISTRY: dict = {
    "pin_jacobian_solver": PinJacobianSolver,
}

__all__ = ["JacobianSolver", "PinJacobianSolver", "REGISTRY"]
