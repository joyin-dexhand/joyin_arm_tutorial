"""逆运动学子包：求解器接口 + 注册表。

IkineSolver(ABC) 为通用接口；具体求解算法为教程 Ch3 教学内容（数值解 /
解析解），实现后经 ``REGISTRY`` 注册接入。用法二选一：直接实例化子类求解
（``arm`` 鸭子类型），或经门面 ``arm.ikine(...)``（活动成员）。
"""
from .ikine_solver import IkineSolver

REGISTRY: dict = {}

__all__ = ["IkineSolver", "REGISTRY"]
