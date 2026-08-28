"""MdhFkineSolver —— MDH 白盒 FK（教学实现，Ch2 占位）。

手推 MDH 递归（连乘 ``T_i = Rot_x(α)·Trans_x(a)·Rot_z(θ+θ₀)·Trans_z(d)``）的归宿，
与 PinFkineSolver 黑盒互为对照；config ``robotics.fkine: mdh`` 切换。
"""
from __future__ import annotations

import numpy as np

from .fkine_solver import FkineSolver

__all__ = ["MdhFkineSolver"]


class MdhFkineSolver(FkineSolver):
    """MDH 参数表递推白盒 FK。"""

    def frame_T(self, arm, q: np.ndarray, frame=None) -> np.ndarray:
        raise NotImplementedError("MdhFkineSolver 待 Ch2 实现")
