"""逆运动学：求解器接口 + 注册表。

- ``IkineSolver``：求解器接口（ABC，见 ikine_solver.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``
                子类加 ``@register``装饰器即自动入表（注册名 = 类名小写 + 下划线）
                机械臂型号 config 文件的 ``robotics.ikine`` 段按名引用。
"""
from .._registry import register
from .ikine_solver import IkineSolver

REGISTRY: dict = {}

__all__ = ["IkineSolver", "REGISTRY", "register"]
