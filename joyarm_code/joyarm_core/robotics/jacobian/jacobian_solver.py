"""JacobianSolver —— 雅可比求解器接口（ABC）

求解器子类只需实现
- :meth:`jac`（6×n 雅可比；``ref`` 取 local/base）。

派生量：速度正逆解 / 静力学 / 奇异值谱 / 可操作度族 / 广义逆与零空间

config ``robotics.jacobian`` 段写注册名，即按名实例化装入 ``_jacobian_solvers`` 成员字典。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

__all__ = ["JacobianSolver"]


class JacobianSolver(ABC):
    """速度运动学策略接口：``V = J(q)q̇``。"""

    # ----------------------------------------------------------
    # jacobian 抽象内核（jac参考系：local/base）
    # ----------------------------------------------------------
    @abstractmethod
    def jac(self,
        arm,
        q: np.ndarray,
        frame: Union[str, int],
        ref: str = "base",
    ) -> np.ndarray:
        """``(6,n)`` 雅可比；``ref`` 取 base（默认，基座系）/ local（自身系）。"""

    # ----------------------------------------------------------
    # jacobian 派生量：速度正逆解 / 静力学 / 奇异值谱 / 可操作度族 / 广义逆与零空间
    # ----------------------------------------------------------

    def fkine_vel(self, arm, q: np.ndarray, dq: np.ndarray,
                  frame: Union[str, int], ref: str = "base") -> np.ndarray:
        """速度正解 ``V = J(q)·q̇``，返回 ``(6,)`` 末端速度旋量
        （线速度 m/s + 角速度 rad/s，参考系同 ``ref``）。"""
        return self.jac(arm, q, frame, ref=ref) @ np.asarray(dq, dtype=float)

    def ikine_vel(self, arm, q: np.ndarray, V: np.ndarray,
                  frame: Union[str, int], ref: str = "base",
                  damping: float = 1e-3) -> np.ndarray:
        """速度逆解（微分逆解）``q̇ = J*·V``，返回 ``(n,)`` 关节速度。"""
        return self.damped_pinv(arm, q, frame, ref=ref, damping=damping) @ np.asarray(V, dtype=float)

    def statics(self, arm, q: np.ndarray, F: np.ndarray,
                frame: Union[str, int], ref: str = "base") -> np.ndarray:
        """静力学 ``τ = JᵀF``：``F`` 为 ``(6,)`` 末端六维力旋量（力 N + 力矩
        N·m，参考系同 ``ref``），返回 ``τ ∈ R^n`` 关节力矩（``n`` = 本体关节数）。"""
        return self.jac(arm, q, frame, ref=ref).T @ np.asarray(F, dtype=float)

    def singular_values(self, arm, q: np.ndarray,
                        frame: Union[str, int], ref: str = "base") -> np.ndarray:
        """奇异值谱 ``σ₁ ≥ … ≥ σ_r``（降序，``r = min(6,n)``）；即速度椭球半轴
        长度，力椭球半轴为其倒数 ``1/σ``（速度/力椭球对偶）。"""
        return np.linalg.svd(self.jac(arm, q, frame, ref=ref), compute_uv=False)

    def is_singular(self, arm, q: np.ndarray, frame: Union[str, int],
                    ref: str = "base", tol: float = 1e-6) -> bool:
        """奇异性判别 ``σ_min < tol``（默认 ``1e-6``）。"""
        return bool(self.singular_values(arm, q, frame, ref=ref)[-1] < tol)

    def manipulability(self, arm, q: np.ndarray, frame: Union[str, int], ref: str = "base") -> float:
        """Yoshikawa 可操作度 ``w = Πσᵢ``——全部 ``min(6,n)`` 个奇异值之积（速度椭球体积度量）。
        ``n≥6`` 且满秩时等于 ``sqrt(det(JJᵀ))``；``n<6`` 时恒为 0。
        """
        return float(np.prod(np.linalg.svd(self.jac(arm, q, frame, ref=ref), compute_uv=False)))

    def manipulability_gradient(self, arm, q: np.ndarray,
            frame: Union[str, int], ref: str = "base", eps: float = 1e-6) -> np.ndarray:
        """可操作度梯度 ``∇w(q)``（中心差分数值微分），返回 ``(n,)``。

        冗余臂（n>6）零空间次级任务的常用输入：
        ``q̇_null ∝ N·∇w``（``N`` 为 :meth:`nullspace_projector`）。"""
        q = np.asarray(q, dtype=float).reshape(-1)
        grad = np.zeros(q.size)
        for i in range(q.size):
            qp, qm = q.copy(), q.copy()
            qp[i] += eps
            qm[i] -= eps
            grad[i] = (self.manipulability(arm, qp, frame, ref=ref)
                       - self.manipulability(arm, qm, frame, ref=ref)) / (2.0 * eps)
        return grad

    def damped_pinv(self, arm, q: np.ndarray, frame: Union[str, int],
                    ref: str = "base", damping: float = 1e-3) -> np.ndarray:
        """阻尼最小二乘（DLS）广义逆 ``J* = Jᵀ(JJᵀ + λ²I)⁻¹``，返回 ``(n,6)``。

        ``λ→0`` 退化为最小范数伪逆 ``J⁺``；``σ=0`` 且 ``λ=0`` 时该方向完全不可控，系数取 0。
        """
        J = self.jac(arm, q, frame, ref=ref)
        U, sv, Vt = np.linalg.svd(J, full_matrices=False)
        denom = sv ** 2 + damping ** 2
        scale = np.divide(sv, denom, out=np.zeros_like(sv), where=denom > 0.0)
        return (Vt.T * scale) @ U.T

    def nullspace_projector(self, arm, q: np.ndarray, frame: Union[str, int],
                            ref: str = "base", damping: float = 1e-3) -> np.ndarray:
        """零空间投影矩阵 ``N = I_n − J*·J``，返回 ``(n,n)``。

        冗余臂零空间次级任务 ``q̇_null = N·z``（``z`` 如 ``∇w``、避限位项）；
        非冗余臂（n≤6 满秩）``N`` 近零。"""
        J = self.jac(arm, q, frame, ref=ref)
        return np.eye(J.shape[1]) - self.damped_pinv(
            arm, q, frame, ref=ref, damping=damping) @ J
