"""GeometricJacobianSolver —— 几何法手推雅可比（教学白盒，Ch4 占位）。

逐列 ``Jᵢ = [zᵢ × (p_ee − pᵢ); zᵢ]``（旋转关节），位姿经 ``arm.fkine`` 门面获取；
config ``solvers.jacobian: geometric`` 切换。
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .jacobian_solver import JacobianSolver

__all__ = ["GeometricJacobianSolver"]


class GeometricJacobianSolver(JacobianSolver):
    """几何法白盒雅可比。"""

    def jac(
        self,
        arm,
        q: np.ndarray,
        frame: Optional[Union[str, int]] = None,
        ref: str = "local",
    ) -> np.ndarray:
        raise NotImplementedError("GeometricJacobianSolver.jac 待 Ch4 实现")
