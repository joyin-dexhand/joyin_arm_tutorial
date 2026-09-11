"""控制律：控制律接口 + 注册表。

- ``Controller``：控制律接口（ABC，见 controller.py）；
- ``REGISTRY``：注册表 ``{注册名: 控制律子类}``
                子类加 ``@register``装饰器即自动入表（注册名 = 类名小写 + 下划线）
                机械臂型号 config 文件的 ``robotics.control`` 段按名引用。
"""
from .._registry import register
from .controller import Controller

REGISTRY: dict = {}

__all__ = ["Controller", "REGISTRY", "register"]
