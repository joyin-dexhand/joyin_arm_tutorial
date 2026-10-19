"""PinJacobianSolver —— 基于 pinocchio 的雅可比默认实现。

直接调用 `pin.computeFrameJacobian`，按属性约定消费 `arm.pin_model`。
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from . import JacobianSolver

__all__ = ["PinJacobianSolver"]


def _frame_id(model, frame) -> int:
    """帧名/索引 → 帧索引；未找到抛错（getFrameId 静默返回哨兵，须拦截）。"""
    fid = int(frame) if isinstance(frame, (int, np.integer)) \
        else model.getFrameId(str(frame))
    if not 0 <= fid < model.nframes:   # 拒越界与负索引
        raise ValueError(
            f"jacobian_solver_pin.py - PinJacobianSolver：帧 {frame!r} 在模型中"
            f"未找到；可用帧：{[f.name for f in model.frames]}")
    return fid


class PinJacobianSolver(JacobianSolver):
    """雅可比默认实现：`J(q) = ∂x/∂q`（pinocchio 帧雅可比）。"""

    def jac(self, arm, q: np.ndarray, frame: str) -> np.ndarray:
        """求雅可比矩阵。

        :param arm: 机械臂对象。
        :param q: `(n,)` 关节角，弧度。
        :param frame: 目标帧名（`str`）。
        :return: `(6,n)` 基座系雅可比矩阵 `J(q)`（LOCAL_WORLD_ALIGNED）。
        """
        model = arm.pin_model
        data = pin.Data(model)                  # 私有草稿纸：无共享可变状态
        fid = _frame_id(model, frame)
        return pin.computeFrameJacobian(model, data,
                                        np.asarray(q, dtype=float), fid,
                                        pin.LOCAL_WORLD_ALIGNED)
