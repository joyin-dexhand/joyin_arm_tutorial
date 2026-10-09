from __future__ import annotations

import numpy as np

from .fkine_solver import FkineSolver
from ...utils.types import Pose

__all__ = ["MdhFkineSolver"]


def _mdh_transform(row: np.ndarray, q: float) -> np.ndarray:
    alpha, a, d, theta_offset = row
    theta = theta_offset + q

    sa, ca = np.sin(alpha), np.cos(alpha)
    st, ct = np.sin(theta), np.cos(theta)
    return np.array([
        [ct,      -st,       0.0,  a],
        [st * ca,  ct * ca,  -sa, -sa * d],
        [st * sa,  ct * sa,   ca,  ca * d],
        [0.0,      0.0,       0.0, 1.0],
    ])

def _fkine_mdh(mdh: np.ndarray, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)

    mdh = np.asarray(
        mdh,
        dtype=float,
    )

    if q.shape != (6,):
        raise ValueError(f"关节角 q 维度应为 (6,) 而非 {q.shape}")
    if mdh.shape != (6, 4):
        raise ValueError(f"MDH 参数维度应为 (6,4) 而非 {mdh.shape}")
    T_06 = np.eye(4)

    for row, qi in zip(mdh, q):
        T_06 = T_06 @ _mdh_transform(row, qi)

    return T_06

class MdhFkineSolver(FkineSolver):
    def frame_pose(self, arm, q: np.ndarray, frame) -> Pose:
        """计算 frame_end_tcp 相对于基座的位姿。"""
        model = arm.pin_model

        frame_id = (
            int(frame)
            if isinstance(frame, (int, np.integer))
            else model.getFrameId(str(frame))
        )
        end_id = model.getFrameId(arm.ee_frame_name)

        if frame_id != end_id:
            raise ValueError(
                "fkine_solver_mdh.py - MdhFkineSolver.frame_pose："
                f"目前仅支持末端帧 {arm.ee_frame_name!r}，实际为 {frame!r}"
            )

        config = arm.get_config()
        joyarm_config = config["joyarm"]

        mdh = np.asarray(joyarm_config["arm_mdh"], dtype=float)
        T_06 = _fkine_mdh(mdh, q)

        T_6e = np.asarray(
            joyarm_config["T_linkn_endtcp"],
            dtype=float,
        )

        T_0e = T_06 @ T_6e

        return Pose.from_T(T_0e)