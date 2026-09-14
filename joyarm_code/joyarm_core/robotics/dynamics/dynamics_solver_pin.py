"""PinDynamicsSolver —— 基于 pinocchio 的动力学默认实现。

直接调用 pin 原生算法（``rnea`` / ``crba`` / ``nonLinearEffects``），
按属性约定消费 ``arm.pin_model``（各方法私有 ``pin.Data``），适配任意构型。
config ``robotics.dynamics`` 段写注册名 ``pin_dynamics_solver`` 即选用。
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from . import DynamicsSolver

__all__ = ["PinDynamicsSolver"]


def _frame_id(model, frame) -> int:
    """帧名/索引 → 帧索引；未找到抛错（getFrameId 静默返回哨兵，须拦截）。"""
    fid = int(frame) if isinstance(frame, (int, np.integer)) \
        else model.getFrameId(str(frame))
    if not 0 <= fid < model.nframes:   # 拒越界与负索引
        raise ValueError(
            f"dynamics_solver_pin.py - PinDynamicsSolver：帧 {frame!r} 在模型中"
            f"未找到；可用帧：{[f.name for f in model.frames]}")
    return fid


class PinDynamicsSolver(DynamicsSolver):
    """动力学默认实现：M/C/G/RNEA（pinocchio 刚体动力学算法）。"""

    def idyn(self, arm, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray,
             f_ext=None) -> np.ndarray:
        """逆动力学 RNEA：状态 + 加速度 → ``τ``。

        :param f_ext: 外力旋量暂不支持（需帧力映射，待力控章节实现），
            非 ``None`` 时抛 ``NotImplementedError``。
        """
        if f_ext is not None:
            raise NotImplementedError(
                "dynamics_solver_pin.py - PinDynamicsSolver.idyn：外力项 "
                "f_ext 暂不支持（待力控章节实现）")
        model = arm.pin_model
        return pin.rnea(model, pin.Data(model), np.asarray(q, dtype=float),
                        np.asarray(dq, dtype=float),
                        np.asarray(ddq, dtype=float))

    def mass_matrix(self, arm, q: np.ndarray) -> np.ndarray:
        """关节空间惯量矩阵 ``M(q)``（CRBA 只填上三角，此处对称化补全）。"""
        model = arm.pin_model
        M = np.array(pin.crba(model, pin.Data(model), np.asarray(q, dtype=float)),
                     dtype=float)
        return np.triu(M) + np.triu(M, 1).T

    def coriolis(self, arm, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """科氏 + 向心力项 ``C(q,q̇)q̇``（nonLinearEffects − 重力项）。"""
        model = arm.pin_model
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        data = pin.Data(model)                  # 私有草稿纸：无共享可变状态
        return (pin.nonLinearEffects(model, data, q, dq)
                - pin.rnea(model, data, q, dq * 0.0, dq * 0.0))

    def gravity(self, arm, q: np.ndarray) -> np.ndarray:
        """重力项 ``G(q)``（RNEA 零速度零加速度特例）。"""
        model = arm.pin_model
        q = np.asarray(q, dtype=float)
        return pin.rnea(model, pin.Data(model), q, q * 0.0, q * 0.0)

    def cartesian_inertia(self, arm, q: np.ndarray, frame,
                          ref: str = "base") -> np.ndarray:
        """笛卡尔惯量 ``Λ = J⁺ᵀMJ⁺``——直接用 pin 雅可比（不依赖 jacobian
        域是否配置），``J⁺`` 为截断 SVD 伪逆。"""
        if ref not in ("base", "local"):
            raise ValueError(
                f"dynamics_solver_pin.py - PinDynamicsSolver.cartesian_inertia："
                f"未知 ref={ref!r}（仅支持 'base'/'local'）")
        model = arm.pin_model
        fid = _frame_id(model, frame)
        rf = pin.LOCAL if ref == "local" else pin.LOCAL_WORLD_ALIGNED
        J = pin.computeFrameJacobian(model, pin.Data(model),
                                     np.asarray(q, dtype=float), fid, rf)
        M = self.mass_matrix(arm, q)
        J_pinv = np.linalg.pinv(J)
        return J_pinv.T @ M @ J_pinv
