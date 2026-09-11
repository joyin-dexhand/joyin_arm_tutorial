"""逆运动学：求解器接口 + 注册表。

- ``IkineSolver``：求解器接口（ABC，见 ikine_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``
                子类加 ``@register``装饰器即自动入表（注册名 = 类名小写 + 下划线）
                机械臂型号 config 文件的 ``robotics.ikine`` 段按名引用。
- ``PinIkineSolver``：逆运动学默认实现（注册名 ``pin_ikine_solver``，见 ``ikine_solver_pin.py``）。
"""
from .._registry import register
from .ikine_solver import IkineSolver

REGISTRY: dict = {}

from .ikine_solver_pin import PinIkineSolver  # noqa: E402 —— 注册须在 REGISTRY 定义后执行

__all__ = ["IkineSolver", "PinIkineSolver", "REGISTRY", "register"]
