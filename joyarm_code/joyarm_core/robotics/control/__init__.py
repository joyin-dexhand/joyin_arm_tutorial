"""控制域（策略接口）。

Controller(ABC) 为通用接口；具体控制律与执行循环为教程 Ch6/8/9 教学内容，
实现后经 ``REGISTRY`` 注册接入。``REGISTRY`` 供 config ``robotics.control``
选型；运行期经 ``arm.set_solver("control", ...)`` 切换。
"""
from .controller import Controller

REGISTRY: dict = {}

__all__ = ["Controller", "REGISTRY"]
