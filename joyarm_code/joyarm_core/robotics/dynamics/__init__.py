"""动力学：求解器接口 + 注册表。

- ``DynamicsSolver``：求解器接口（ABC，见 dynamics_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``
                子类加 ``@register``装饰器即自动入表（注册名 = 类名小写 + 下划线）
                机械臂型号 config 文件的 ``robotics.dynamics`` 段按名引用。
"""
from .._registry import register
from .dynamics_solver import DynamicsSolver

REGISTRY: dict = {}

__all__ = ["DynamicsSolver", "REGISTRY", "register"]
