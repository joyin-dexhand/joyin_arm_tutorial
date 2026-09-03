"""控制律：求解器接口 + 注册表。

- ``Controller``：控制律接口（ABC，见 controller.py）；
- ``REGISTRY``：注册表 ``{注册名: 求解器子类}``。

"""
from .controller import Controller

REGISTRY: dict = {}

__all__ = ["Controller", "REGISTRY"]
