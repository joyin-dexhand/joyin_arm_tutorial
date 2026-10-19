"""IkineSolver —— 逆运动学求解器基类接口（ABC）

已知：末端位姿 pose（xyz+四元数）；求解：各关节角 q。

外部调用：
    - :meth:`solve(arm, pose, frame, q0, tol, iters)` —— 返回**单组解**：数值法从 `q0` 迭代；解析法经限位剔除后取距 `q0` 最近解；
    - :meth:`solve_all(arm, pose)` —— 由解析法实现后可使用，返回**全部解析解**（6R 通常 8 组）。

子类需实现：
    - :meth:`solve(arm, pose, frame, q0, tol, iters)` —— 单解，数值法迭代 或 解析法选解；
    - :meth:`solve_all(arm, pose)` —— 解析法可选实现，求解全部解析解；默认抛 `NotImplementedError`。

参数排序：**通用参数在前**，**特有参数在后**。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...utils.types import IKResult, Pose

__all__ = ["IkineSolver"]


class IkineSolver(ABC):

    @abstractmethod
    def solve(self,
        arm,
        pose: Pose,
        frame: str,
        q0: np.ndarray,
        *,
        tol: float = 1e-3,
        iters: int = 200,
        **kwargs,
    ) -> IKResult:
        """求单组解。

        :param arm: 机械臂对象。
        :param pose: 目标位姿 `Pose`（xyz + 四元数）。
        :param frame: 目标帧名（`str`）。
        :param q0: `(n,)` 关节角参考，弧度——数值法作**迭代起点**；解析法作**选解参考**：先剔超软限位者，再取最接近 `q0` 的解。
        :param tol: 残差范数收敛容差（数值法特有，默认 `1e-3`）。
        :param iters: 迭代次数上限（数值法特有，默认 `200`）。
        :return: `IKResult`，`q` 为 `(n,)` 单解；无可行解时 `success=False`。
        """

    def solve_all(self,
        arm,
        pose: Pose,
        **kwargs,
    ) -> IKResult:
        """求全部解。

        :param arm: 机械臂对象。
        :param pose: 目标位姿 `Pose`。
        :return: `IKResult`，`q` 为 `(K,n)`。

        **不限位剔除**，但逐关节尝试 `±2π` 平移，使每组解尽量落入关节限位范围内（等价角中取离限位区间最近者）。
        """
        raise NotImplementedError(
            "ikine_solver.py - IkineSolver.solve_all：尚未实现求全部解（需解析法实现，求单解请用 solve）")

