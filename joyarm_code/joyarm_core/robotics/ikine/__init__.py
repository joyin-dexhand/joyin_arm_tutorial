"""逆运动学：求解器接口 + 注册表。

- ``IkineSolver``：求解器接口（ABC，见 ikine_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``
        新增实现：import 后在本表加一行（注册名 = 类名小写+下划线）， config 文件 ``robotics.ikine``段按名引用。
"""
from .ikine_solver import IkineSolver
from .ikine_solver_pin import PinIkineSolver

REGISTRY: dict = {
    "pin_ikine_solver": PinIkineSolver,   # 默认实现（首位）
}

__all__ = ["IkineSolver", "PinIkineSolver", "REGISTRY"]
