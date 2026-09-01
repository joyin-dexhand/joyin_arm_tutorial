"""正运动学域（策略接口）。

FkineSolver(ABC) 为通用接口；具体求解算法为教程 Ch2 教学内容（如 MDH 白盒
递推），实现后经 ``REGISTRY`` 注册接入。``fkine(arm, q)`` 委托门面
``arm.fkine``（教学用）。
"""
from .fkine_solver import FkineSolver

REGISTRY: dict = {}

__all__ = ["FkineSolver", "REGISTRY", "fkine"]


def fkine(arm, q, frame=None, rep="T"):
    """函数式入口（教学用）：委托 ``arm.fkine``（活动 FK 求解器）。"""
    return arm.fkine(q, frame=frame, rep=rep)
