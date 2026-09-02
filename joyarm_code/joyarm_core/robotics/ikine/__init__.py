"""逆运动学子包：求解器接口 + 注册表。

IkineSolver(ABC) 为通用接口；实现后经 ``REGISTRY`` 注册接入。

用法一：直接实例化子类求解；
用法二：经 ``arm.ikine`` 门面求解。
"""
from .ikine_solver import IkineSolver

REGISTRY: dict = {}

__all__ = ["IkineSolver", "REGISTRY"]
