"""正运动学域（策略族，一节点一文件）。

FkineSolver(ABC) · PinFkineSolver(``"pin"``，默认) · MdhFkineSolver(``"mdh"``，Ch2 占位)。
``REGISTRY`` 供 config ``robotics.fkine`` 选型；``fkine(arm, q)`` 委托门面 ``arm.fkine``（教学用）。
"""
from .fkine_solver import FkineSolver
from .pin_fkine_solver import PinFkineSolver
from .mdh_fkine_solver import MdhFkineSolver

REGISTRY = {
    "pin": PinFkineSolver,
    "mdh": MdhFkineSolver,
}

__all__ = ["FkineSolver", "PinFkineSolver", "MdhFkineSolver", "REGISTRY", "fkine"]


def fkine(arm, q, frame=None, rep="T"):
    """函数式入口（教学用）：委托 ``arm.fkine``，即 ``arm._fkine_solver.solve``。"""
    return arm.fkine(q, frame=frame, rep=rep)
