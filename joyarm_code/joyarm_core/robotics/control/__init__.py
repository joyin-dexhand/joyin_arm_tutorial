"""控制律：控制律接口 + 自动/力控制器 + 注册表。

- ``Controller``：控制律接口（ABC，见 controller.py）；
- ``REGISTRY``：注册表 ``{注册名: 控制律子类}``。

"""
from .controller import Controller
from .auto_controller import AutoController
from .force_controller import ForceController

REGISTRY: dict = {}

__all__ = ["Controller", "AutoController", "ForceController", "REGISTRY"]
