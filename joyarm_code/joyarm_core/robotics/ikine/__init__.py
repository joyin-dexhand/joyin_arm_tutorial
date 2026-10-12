"""逆运动学：基类接口 + 子类注册表。

- ``IkineSolver``：求解器接口（ABC，见 ikine_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``

新子类实现步骤：
    1. 在本包内创建 `ikine_solver_<new>.py` 子类文件，实现子类 `<New>IkineSolver`（继承自 `ikine_solver.py` 的 `IkineSolver` 基类）；
    2. 在本文件中 import 新子类，并在 ``REGISTRY`` 显式加一行注册（注册名 = 类名小写+下划线）；
"""
from .ikine_solver import IkineSolver
from .ikine_solver_pin import PinIkineSolver

REGISTRY: dict = {
    "pin_ikine_solver": PinIkineSolver,
}

__all__ = ["IkineSolver", "PinIkineSolver", "REGISTRY"]
