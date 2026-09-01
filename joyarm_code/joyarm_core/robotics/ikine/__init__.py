"""逆运动学域（策略接口）。

IkineSolver(ABC) 为通用接口；具体求解算法为教程 Ch3 教学内容（数值解 /
解析解），实现后经 ``REGISTRY`` 注册接入。``ikine`` / ``ikine_constrained``
委托门面（教学用）。
"""
from .ikine_solver import IkineSolver

REGISTRY: dict = {}

__all__ = ["IkineSolver", "REGISTRY", "ikine", "ikine_constrained"]


def ikine(arm, T_target, frame=None, **kw):
    """函数式入口（教学用）：委托 ``arm.ikine``（特有参数经 ``**kw`` 透传求解器）。"""
    return arm.ikine(T_target, frame=frame, **kw)


def ikine_constrained(arm, T_target, frame=None, **kw):
    """函数式入口（教学用）：委托 ``arm.ikine_constrained``。"""
    return arm.ikine_constrained(T_target, frame=frame, **kw)
