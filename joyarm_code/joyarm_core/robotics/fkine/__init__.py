"""正运动学：求解器接口 + 注册表。

- ``FkineSolver``：求解器接口（ABC，见 fkine_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``
                子类加 ``@register``装饰器即自动入表（注册名 = 类名小写 + 下划线）
                机械臂型号 config 文件的 ``robotics.fkine`` 段按名引用。
- ``PinFkineSolver``：正运动学默认实现（注册名 ``pin_fkine_solver``，见 ``fkine_solver_pin.py``）。
"""
from .._registry import register
from .fkine_solver import FkineSolver

REGISTRY: dict = {}

from .fkine_solver_pin import PinFkineSolver  # noqa: E402 —— 注册须在 REGISTRY 定义后执行

__all__ = ["FkineSolver", "PinFkineSolver", "REGISTRY", "register"]
