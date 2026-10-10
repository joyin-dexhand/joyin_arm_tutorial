"""JacobianSolver —— 雅可比求解器基类接口（ABC）

已知：关节角度 q ；求解：雅可比矩阵 J(q)。

外部调用：
    - :meth:`jac(arm, q, frame)` —— `(6,n)` 雅可比本体（基座系）；
    - :meth:`jac_pinv_damped(arm, q, frame, damping)` —— 雅可比的阻尼最小二乘广义逆。

子类需实现：
    - :meth:`jac(arm, q, frame)` —— 6×n 雅可比（基座系）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

__all__ = ["JacobianSolver"]


class JacobianSolver(ABC):

    @abstractmethod
    def jac(self,
        arm,
        q: np.ndarray,
        frame: str,
    ) -> np.ndarray:
        """求雅可比矩阵。

        :param arm: 机械臂对象。
        :param q: ``(n,)`` 关节角，弧度。
        :param frame: 目标帧名（``str``）。
        :return: ``(6,n)`` 基座参考系下的雅可比矩阵 ``J(q)``。
        """

    def jac_pinv_damped(self, arm, q: np.ndarray, frame: str,
                        damping: float = 1e-3) -> np.ndarray:
        """求雅可比的阻尼最小二乘广义逆。

        :param arm: 机械臂对象。
        :param q: ``(n,)`` 关节角，弧度。
        :param frame: 目标帧名（``str``）。
        :param damping: 阻尼系数 ``λ``，默认 ``1e-3``。
        :return: ``(n,6)`` 基座参考系下的广义逆 ``J* = Jᵀ(JJᵀ + λ²I)⁻¹``。

        **λ→0** 退化为最小范数伪逆 ``J⁺``；``σ=0`` 且 ``λ=0`` 时该方向完全不可控，系数取 0。
        """
        J = self.jac(arm, q, frame)
        U, sv, Vt = np.linalg.svd(J, full_matrices=False)
        denom = sv ** 2 + damping ** 2
        scale = np.divide(sv, denom, out=np.zeros_like(sv), where=denom > 0.0)
        return (Vt.T * scale) @ U.T
