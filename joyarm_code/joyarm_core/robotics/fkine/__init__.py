"""正运动学：基类接口 + 子类注册表。

- ``FkineSolver``：求解器接口（ABC，见 fkine_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``

新子类实现步骤：
    1. 在本包内创建 `fkine_solver_<new>.py` 子类文件，实现子类 `<New>FkineSolver`（继承自 `fkine_solver.py` 的 `FkineSolver` 基类）；
    2. 在本文件中 import 新子类，并在 ``REGISTRY`` 显式加一行注册（注册名 = 类名小写+下划线）；
"""
from .fkine_solver import FkineSolver
from .fkine_solver_pin import PinFkineSolver
from .fkine_solver_mdh import MdhFkineSolver

REGISTRY: dict = {
    "pin_fkine_solver": PinFkineSolver,  # 默认实现
    "mdh_fkine_solver": MdhFkineSolver,  # MDH 解析实现
}

__all__ = ["FkineSolver", "PinFkineSolver","MdhFkineSolver", "REGISTRY"]
