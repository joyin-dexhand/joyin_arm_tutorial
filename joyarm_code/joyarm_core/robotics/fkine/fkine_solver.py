"""FkineSolver —— 正运动学求解器基类接口（ABC）

已知：各关节角 q ；求解：末端位姿 pose（xyz+四元数）。

外部调用：
    - :meth:`solve(arm, q, frame)` —— `q` 为 `(n,)` ，返回 `Pose`（xyz+四元数）。

子类需实现：
    - :meth:`solve(arm, q, frame)` —— 正运动学位姿求解。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from ...utils.types import Pose

__all__ = ["FkineSolver"]


class FkineSolver(ABC):

    @abstractmethod
    def solve(self, arm, q: np.ndarray, frame: str) -> Pose:
        """正运动学求解

        :param arm: 机械臂对象。
        :param q: ``(n,)`` 关节角，弧度。
        :param frame: 末端帧名（``str``）。
        :return: `Pose`（xyz+四元数），末端在基坐标系下的位姿。
        """
