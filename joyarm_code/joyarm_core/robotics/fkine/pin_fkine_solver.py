"""PinFkineSolver —— pinocchio 黑盒 FK（默认实现，Ch2 已实现）。

内核 :meth:`frame_T`（原 ``JoyArm.frame_placement`` 下沉）：``forwardKinematics``
+ ``updateFramePlacement``；形状 / rep / 批量由基类模板提供。
"""
from __future__ import annotations

import numpy as np

from .fkine_solver import FkineSolver

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover
    pin = None

__all__ = ["PinFkineSolver"]


class PinFkineSolver(FkineSolver):
    """pinocchio（URDF 驱动）黑盒 FK。"""

    def frame_T(self, arm, q: np.ndarray, frame=None) -> np.ndarray:
        fid = self._resolve_frame(arm, frame)
        q_arr = np.asarray(q, dtype=float).reshape(arm.n)
        pin.forwardKinematics(arm.model, arm.data, q_arr)
        pin.updateFramePlacement(arm.model, arm.data, fid)
        return arm.T_base @ arm.data.oMf[fid].homogeneous

    def _resolve_frame(self, arm, frame) -> int:
        """帧名/索引/None → pinocchio frame id。"""
        if frame is None:
            return arm.ee_frame_id
        if isinstance(frame, (int, np.integer)):
            return int(frame)
        for i, f in enumerate(arm.model.frames):
            if f.name == frame:
                return i
        raise ValueError(f"找不到帧 '{frame}'")
