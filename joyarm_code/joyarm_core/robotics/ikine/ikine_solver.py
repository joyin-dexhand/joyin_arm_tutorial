"""IkineSolver —— 逆运动学求解器接口（ABC：只定骨架，不含算法）。

逆运动学与正运动学互为逆问题：**给定目标位姿（Pose：xyz + 四元数），求各关节角。**

本文件为通用骨架，求解器子类只需实现 :meth:`solve`（带软限位约束版
:meth:`solve_constrained` 可选，基类默认未实现）。

两条实现约定：
- 迭代内部经 ``arm.fkine`` / ``arm.jac`` 门面求位姿与雅可比（不直接摸
  model/data），从而与 FK / 雅可比求解器任意替换组合；
- 参数排序：**通用参数在前**（任何实现都需要），**特有参数 keyword-only 在后**
  （如数值法的迭代参数；解析法可忽略）。

config ``robotics.ikine`` 段写注册名，即按名实例化装入 ``_ikine_solvers`` 成员字典。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from ...utils.types import IKResult, Pose

__all__ = ["IkineSolver"]


class IkineSolver(ABC):
    """逆运动学策略接口：目标位姿 → 关节角。"""

    @abstractmethod
    def solve(
        self,
        arm,
        target: Pose,
        frame: Union[str, int],
        *,
        q0: Optional[np.ndarray] = None,
        tol: float = 1e-4,
        iters: int = 200,
        **kwargs,
    ) -> IKResult:
        """求 IK（通用 → 特有参数排序）。

        :param target: 目标位姿 ``Pose``（xyz + 四元数；与 ``arm.fkine`` 输出同约定）。
        :param frame: 目标帧名/索引（必填）。
        :param q0: 迭代初值（数值法特有；缺省 ``arm.q_neutral``，解析法忽略）。
        :param tol: 残差范数收敛容差（数值法特有，默认 ``1e-4``）。
        :param iters: 迭代次数上限（数值法特有，默认 ``200``）。
        """

    def solve_constrained(
        self,
        arm,
        target: Pose,
        frame: Union[str, int],
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
