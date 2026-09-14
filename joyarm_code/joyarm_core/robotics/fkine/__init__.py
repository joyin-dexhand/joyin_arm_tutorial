"""正运动学：求解器接口 + 注册表。

- ``FkineSolver``：求解器接口（ABC，见 fkine_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``
        新增实现：import 后在本表加一行（注册名 = 类名小写+下划线）， config 文件 ``robotics.fkine``段按名引用。
"""
from .fkine_solver import FkineSolver
from .fkine_solver_pin import PinFkineSolver

REGISTRY: dict = {
    "pin_fkine_solver": PinFkineSolver,   # 默认实现（首位）
}

__all__ = ["FkineSolver", "PinFkineSolver", "REGISTRY"]
