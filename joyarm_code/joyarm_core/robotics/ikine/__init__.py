"""逆运动学域（策略族，一节点一文件）。

IkineSolver(ABC) · PinIkineSolver(``"pin"``，数值解，Ch3 占位) · AnalyticIkine6R
(``"analytic6r"``，闭式解占位)。``REGISTRY`` 供 config ``solvers.ikine`` 选型；
``ikine`` / ``ikine_constrained`` 委托门面（教学用）。
"""
from .ikine_solver import IkineSolver
from .pin_ikine_solver import PinIkineSolver
from .analytic_ikine_6r import AnalyticIkine6R

REGISTRY = {
    "pin": PinIkineSolver,
    "analytic6r": AnalyticIkine6R,
}

__all__ = [
    "IkineSolver", "PinIkineSolver", "AnalyticIkine6R", "REGISTRY",
    "ikine", "ikine_constrained",
]


def ikine(arm, T_target, q0=None, frame=None, **kw):
    """函数式入口（教学用）：委托 ``arm.ikine``，即 ``arm._ikine_solver.solve``。"""
    return arm.ikine(T_target, q0=q0, frame=frame, **kw)


def ikine_constrained(arm, T_target, q0=None, frame=None, **kw):
    """函数式入口（教学用）：委托 ``arm.ikine_constrained``。"""
    return arm.ikine_constrained(T_target, q0=q0, frame=frame, **kw)
