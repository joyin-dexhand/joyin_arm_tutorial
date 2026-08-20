"""FkineSolver —— 正运动学策略 ABC（模板方法）。

``JoyArm._fkine_solver`` 的契约：实现经 config ``solvers.fkine`` 按注册名选型。
:meth:`solve` 为通用模板（形状重载 + rep 转换 + 批量循环），实现只需给出
``(4,4)`` 内核 :meth:`frame_T`——仅消费 arm 公开属性，不 import joyarms。Ch2。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

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
    def frame_T(self, arm, q: np.ndarray, frame=None) -> np.ndarray:
        """内核：指定帧在基坐标系下的 ``(4,4)`` 位姿（含 ``T_base`` 偏移）。"""

    def solve(
        self,
        arm,
        q: np.ndarray,
        frame: Optional[Union[str, int]] = None,
        rep: str = "T",
    ):
        """模板：``q`` 为 ``(n,)`` 单点 / ``(N,n)`` 批量；``rep`` 取 ``"T"``(Pose)/``"pos"``/``"se3"``。"""
        if rep not in ("T", "pos", "se3"):
            raise ValueError(f"未知 rep={rep!r}（仅支持 'T'/'pos'/'se3'）")

        q_arr = np.asarray(q, dtype=float)
        if q_arr.ndim == 1:
            return self._to_rep(self.frame_T(arm, q_arr, frame), rep)
        elif q_arr.ndim == 2:
            return self._solve_batch(arm, q_arr, frame, rep)
        else:
            raise ValueError(f"q 维度需为 1 或 2，收到 q.shape={q_arr.shape}")

    @staticmethod
    def _to_rep(T: np.ndarray, rep: str):
        """``(4,4)`` → 指定表示。"""
        if rep == "T":
            return Pose.from_T(T)
        elif rep == "pos":
            return T[:3, 3]
        else:  # se3
            return pin.SE3(T[:3, :3], T[:3, 3])

    def _solve_batch(self, arm, Q: np.ndarray, frame, rep: str):
        """批量 FK：循环调内核，``data`` 缓冲区在内核内复用。"""
        N = Q.shape[0]
        if rep == "pos":
            out = np.empty((N, 3))
            for i in range(N):
                out[i] = self.frame_T(arm, Q[i], frame)[:3, 3]
            return out
        return [self._to_rep(self.frame_T(arm, Q[i], frame), rep) for i in range(N)]
