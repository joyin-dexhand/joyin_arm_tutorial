"""MdhFkineSolver —— MDH 参数连乘的正运动学实现。

算法：按 MDH 参数表逐行求相邻连杆系的位姿矩阵，自基座系至末端系连乘得到末端位姿；
数据：经 `arm.get_config()` 读取 `arm_mdh` 与 `T_linkn_endtcp`，按 arm 对象缓存复用；
frame：仅支持末端 frame（cfg `joyarm.frame_end`），其余 frame 抛 `ValueError`。

用法：
    arm.set_fkine_solver("mdh_fkine_solver")   # 设置要使用的正运动学求解器
    arm.solve_fkine(q, frame)                  # 通过 arm 调用, q 默认取 arm.joint_state_arm.q, frame 默认取且只仅能取 frame_end
"""
from __future__ import annotations

import numpy as np

from .fkine_solver import FkineSolver
from ...utils.types import Pose

__all__ = ["MdhFkineSolver"]


def _mdh_transform(row: np.ndarray, q: float) -> np.ndarray:
    """MDH 参数行 + 关节角 → 相邻连杆位姿矩阵。

    :param row: `(4,)` 参数行 `[α_{i-1}, a_{i-1}, d_i, θ_{i0}]`。
    :param q: 该关节的角度，弧度；实际转角 θ_i = q + θ_{i0}。
    :return: `(4, 4)` 相邻连杆系位姿矩阵。
    """
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
    """连乘正运动学：自基座连乘得到连杆n系的位姿矩阵。

    :param mdh: `(n, 4)` MDH 参数表（行数 = 关节数）。
    :param q: `(n,)` 关节角，弧度。
    :return: `(4, 4)` 第 n 连杆系在基座系下的位姿。
    :raises ValueError: `mdh` 形状不是 `(n, 4)` 二维数组，或 `q` 的维度不等于n。
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    mdh = np.asarray(mdh, dtype=float)

    if mdh.ndim != 2 or mdh.shape[1] != 4:
        raise ValueError(f"fkine_solver_mdh.py - _fkine_mdh：MDH 参数表须为 (n, 4) 二维数组，实际为 {mdh.shape}")
    if q.shape != (mdh.shape[0],):
        raise ValueError(f"fkine_solver_mdh.py - _fkine_mdh：q 的维度须为 MDH 参数表的行数 {mdh.shape[0]}，实际为 {q.shape}")

    T_0n = np.eye(4)
    for row, qi in zip(mdh, q):
        T_0n = T_0n @ _mdh_transform(row, qi)
    return T_0n


class MdhFkineSolver(FkineSolver):
    """MDH 参数连乘的正运动学实现：`q` → 末端 frame 在基座系下的位姿。"""

    def __init__(self):
        """缓存 arm 对象的常量参数（MDH 表、TCP 偏移位姿、末端 frame）。"""
        self._prepared = None

    def _prepare(self, arm) -> tuple:
        """读取 cfg 中的 MDH 数据，按 arm 对象缓存复用。

        :param arm: 提供 `get_config()` 与 `pin_model` 的实例。
        :return: `(arm, arm_mdh, T_linkn_endtcp, frame_end, end_id)`，两数组已置只读。
        :raises ValueError: cfg `joyarm` 段缺 `arm_mdh` 或 `T_linkn_endtcp` 键。
        """
        prepared = self._prepared
        if prepared is None or prepared[0] is not arm:
            cfg = arm.get_config()["joyarm"]
            missing = [key for key in ("arm_mdh", "T_linkn_endtcp") if key not in cfg]
            if missing:
                raise ValueError(
                    "fkine_solver_mdh.py - MdhFkineSolver：cfg.joyarm 段缺 {missing} 键，该型号未提供 MDH 数据，不可用 MDH 求解器"
                )
            mdh = np.asarray(cfg["arm_mdh"], dtype=float)
            T_ne = np.asarray(cfg["T_linkn_endtcp"], dtype=float)
            mdh.setflags(write=False)
            T_ne.setflags(write=False)
            frame_end = str(cfg["frame_end"])
            end_id = arm.pin_model.getFrameId(frame_end)
            # 完整准备成功后再整体替换，避免半成品缓存被后续求解读到。
            prepared = (arm, mdh, T_ne, frame_end, end_id)
            self._prepared = prepared
        return prepared

    def solve(self, arm, q: np.ndarray, frame: str) -> Pose:
        """求末端 frame 在基座系下的位姿。

        常量参数 (arm, arm_mdh, T_linkn_endtcp, frame_end, end_id) 按 arm 对象缓存，首次调用后不再重复读取 cfg。

        :param arm: 机械臂对象。
        :param q: `(n,)` 关节角，弧度。
        :param frame: 末端 frame 名或序号（须为 cfg.joyarm.frame_end`）。
        :return: `Pose`（xyz+四元数），末端 frame 在基坐标系下的位姿。
        :raises ValueError: frame 不是末端 frame，或 cfg 缺 MDH 数据。
        """
        _, mdh, T_ne, frame_end, end_id = self._prepare(arm)

        frame_id = (
            int(frame)
            if isinstance(frame, (int, np.integer))
            else arm.pin_model.getFrameId(str(frame))
        )
        if frame_id != end_id:
            raise ValueError(
                "fkine_solver_mdh.py - MdhFkineSolver.solve：仅支持末端 frame {frame_end!r}，实际为 {frame!r}"
            )

        return Pose.from_T(_fkine_mdh(mdh, q) @ T_ne)
