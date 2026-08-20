"""PinJacobianSolver —— pinocchio 黑盒雅可比（默认实现，Ch4 占位）。

内核 Ch4 落地：``pin.computeFrameJacobian``（local）或经 ``getFrameJacobian``
转参考系；衍生量由基类模板给出。
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .jacobian_solver import JacobianSolver

__all__ = ["PinJacobianSolver"]


class PinJacobianSolver(JacobianSolver):
    """pinocchio（URDF 驱动）黑盒雅可比。"""

    def jac(
        self,
        arm,
        q: np.ndarray,
        frame: Optional[Union[str, int]] = None,
        ref: str = "local",
    ) -> np.ndarray:
        raise NotImplementedError("PinJacobianSolver.jac 待 Ch4 实现")
