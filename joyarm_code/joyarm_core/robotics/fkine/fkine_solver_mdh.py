"""MdhFkineSolver —— MDH 参数白盒正运动学（教学实现）。

1. 由 config `arm_mdh`（6×4：α/a/d/θ）逐关节连乘齐次变换得 T_06，再右乘 `T_linkn_endtcp` 得末端位姿；
2. 仅支持末端帧（`frame` 须为 cfg `joyarm.frame_end`），其余帧抛 `ValueError`。
"""
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
    """MDH 参数正运动学（仅支持末端帧 `frame_end_tcp`）。"""

    def solve(self, arm, q: np.ndarray, frame) -> Pose:
        """正运动学求解：frame_end_tcp 相对于基座的位姿。"""
        model = arm.pin_model
        joyarm_config = arm.get_config()["joyarm"]

        frame_id = (
            int(frame)
            if isinstance(frame, (int, np.integer))
            else model.getFrameId(str(frame))
        )
        ee_frame = str(joyarm_config["frame_end"])   # 末端帧名（cfg joyarm.frame_end）
        end_id = model.getFrameId(ee_frame)

        if frame_id != end_id:
            raise ValueError(
                "fkine_solver_mdh.py - MdhFkineSolver.solve："
                f"目前仅支持末端帧 {ee_frame!r}，实际为 {frame!r}"
            )

        mdh = np.asarray(joyarm_config["arm_mdh"], dtype=float)
        T_06 = _fkine_mdh(mdh, q)

        T_6e = np.asarray(
            joyarm_config["T_linkn_endtcp"],
            dtype=float,
        )

        T_0e = T_06 @ T_6e

        return Pose.from_T(T_0e)