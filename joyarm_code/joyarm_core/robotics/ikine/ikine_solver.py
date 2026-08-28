"""IkineSolver —— 逆运动学策略 ABC。

``JoyArm._ikine_solver`` 的契约：实现经 config ``robotics.ikine`` 选型。约定：迭代
内部经 ``arm.fkine`` 门面求位姿/残差（不直接摸 model/data），从而与 FK 求解器任意
替换组合。Ch3。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from ...utils.types import IKResult

__all__ = ["IkineSolver"]


class IkineSolver(ABC):
    """逆运动学策略接口：目标位姿 → 关节角。"""

    @abstractmethod
    def solve(
        self,
        arm,
        T_target: np.ndarray,
        q0: Optional[np.ndarray] = None,
        frame: Optional[Union[str, int]] = None,
        tol: float = 1e-4,
        iters: int = 200,
        **kwargs,
    ) -> IKResult:
        """求 IK：``T_target``(4,4)；``q0`` 缺省 ``arm.q_neutral``；``tol`` 残差范数；``iters`` 上限。"""

    def solve_constrained(
        self,
        arm,
        T_target: np.ndarray,
        q0: Optional[np.ndarray] = None,
        frame: Optional[Union[str, int]] = None,
        tol: float = 1e-4,
        iters: int = 200,
        **kwargs,
    ) -> IKResult:
        """带关节限位约束版（解满足 ``arm.joint_limits``，Ch3 实现）。"""
        raise NotImplementedError("ikine_constrained 待 Ch3 实现")
