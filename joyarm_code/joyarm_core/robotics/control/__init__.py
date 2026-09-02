"""控制域（策略接口）。

Controller(ABC) 为通用接口；具体控制律与执行循环为教程 Ch6/8/9 教学内容，
实现后经 ``REGISTRY`` 注册接入。用法二选一：直接实例化子类（``arm`` 鸭子
类型，课堂/单测用）；或 config ``robotics.control`` 选型由 JoyArm 装入成员
字典，运行期经 ``arm.set_controller`` 切换。
"""
from .controller import Controller

REGISTRY: dict = {}

__all__ = ["Controller", "REGISTRY"]
