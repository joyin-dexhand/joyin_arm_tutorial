"""IkineSolver —— 逆运动学策略 ABC。

``JoyArm._ikine_solvers["pin"]`` 等成员的契约：实现经 config ``robotics.ikine``
按注册名选型。约定：迭代内部经 ``arm.fkine`` / ``arm.jac`` 门面求位姿与雅可比
（不直接摸 model/data），从而与 FK / 雅可比求解器任意替换组合。Ch3。

参数排序契约（架构约束）：**通用参数在前**（任何 IK 实现都需要），**特有参数
keyword-only 在后**（仅特定算法需要，如数值法的迭代参数；解析法可忽略）。
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
        frame: Optional[Union[str, int]] = None,
        *,
        q0: Optional[np.ndarray] = None,
        tol: float = 1e-4,
        iters: int = 200,
        **kwargs,
    ) -> IKResult:
        """求 IK（通用 → 特有参数排序）。

        :param T_target: 目标位姿 ``(4,4)``（与 ``arm.fkine`` 输出同坐标系约定）。
        :param frame: 目标帧名/索引（通用；``None`` = 末端默认帧）。
        :param q0: 迭代初值（数值法特有；缺省 ``arm.q_neutral``，解析法忽略）。
        :param tol: 残差范数收敛容差（数值法特有，默认 ``1e-4``）。
        :param iters: 迭代次数上限（数值法特有，默认 ``200``）。
        """

    def solve_constrained(
        self,
        arm,
        T_target: np.ndarray,
        frame: Optional[Union[str, int]] = None,
        *,
        q0: Optional[np.ndarray] = None,
        tol: float = 1e-4,
        iters: int = 200,
        restarts: int = 32,
        **kwargs,
    ) -> IKResult:
        """带关节软限位约束版（解满足 ``arm.joint_limits_soft``；失败随机重启）。

        :param restarts: 首轮失败后的随机初值重启次数（默认 32）。
        """
        raise NotImplementedError("ikine_constrained 待具体求解器实现")
