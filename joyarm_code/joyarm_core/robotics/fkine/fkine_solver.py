"""FkineSolver —— 正运动学求解器接口（ABC）

求解器子类只需实现
- :meth:`frame_pose(arm, q, frame)`     # 算出该帧的位姿（Pose：xyz + 四元数）

config的``robotics.fkine``段写注册名，即按名实例化装入 ``_fkine_solvers`` 成员字典。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from ...utils.types import Pose

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover
    pin = None

__all__ = ["FkineSolver"]


class FkineSolver(ABC):
    """正运动学策略接口：关节角 → 末端（或任意帧）位姿。"""

    @abstractmethod
    def frame_pose(self, arm, q: np.ndarray, frame: Union[str, int]) -> Pose:
        """内核：指定帧在基坐标系下的位姿（含 ``T_base`` 偏移）。"""

    def solve(self,
        arm,
        q: np.ndarray,
        frame: Union[str, int],
        rep: str = "pose",
    ):
        """模板：``frame`` 为目标帧；``q`` 为 ``(n,)`` 单点 / ``(N,n)`` 批量；``rep`` 取 ``"pose"``（默认，xyz+四元数）/``"T"``(4×4)/``"se3"``。"""
        if rep not in ("pose", "T", "se3"):
            raise ValueError(f"未知 rep={rep!r}（仅支持 'pose'/'T'/'se3'）")

        q_arr = np.asarray(q, dtype=float)
        if q_arr.ndim == 1:
            return self._to_rep(self.frame_pose(arm, q_arr, frame), rep)
        elif q_arr.ndim == 2:
            return self._solve_batch(arm, q_arr, frame, rep)
        else:
            raise ValueError(f"q 维度需为 1 或 2，收到 q.shape={q_arr.shape}")

    @staticmethod
    def _to_rep(pose: Pose, rep: str):
        """``Pose`` → 指定表示。"""
        if rep == "pose":
            return pose
        elif rep == "T":
            return pose.T
        else:  # se3
            T = pose.T
            return pin.SE3(T[:3, :3], T[:3, 3])

    def _solve_batch(self, arm, Q: np.ndarray, frame: Union[str, int], rep: str = "pose"):
        """批量 FK：逐组调内核。"""
        N = Q.shape[0]
        if rep == "T":
            out = np.empty((N, 4, 4))
            for i in range(N):
                out[i] = self.frame_pose(arm, Q[i], frame).T
            return out
        return [self._to_rep(self.frame_pose(arm, Q[i], frame), rep) for i in range(N)]
