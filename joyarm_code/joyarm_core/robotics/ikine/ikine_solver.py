"""IkineSolver —— 逆运动学求解器接口（ABC）

两个公开函数，覆盖两种用法：
- :meth:`solve`——返回**单组解**（数值法从 ``q0`` 迭代；解析法经限位剔除后取``q0`` 最近解）；
- :meth:`solve_all`——返回**全部解析解**（6R 通常 8 组），供分析/比较。

两条实现约定：
- 迭代内部求位姿/雅可比的两条等效路径：
  ① 经 ``arm.fkine`` / ``arm.jac`` 门面（与 FK / 雅可比求解器任意替换组合）；
  ② 直接用 ``arm.pin_model`` 并**每次求解新建私有 ``pin.Data``**（更快）——内部实现无共享可变状态；
  
- 参数排序：**通用参数在前**（任何实现都需要），**特有参数 keyword-only 在后**。

config ``robotics.ikine`` 段写注册名，即按名实例化装入 ``_ikine_solvers`` 成员字典。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from ...utils.types import IKResult, Pose

__all__ = ["IkineSolver"]


class IkineSolver(ABC):
    """逆运动学策略接口：目标位姿 → 关节角。"""

    # ----------------------------------------------------------
    # ikine 抽象内核：solve（单解；数值法迭代 / 解析法选解）
    # ----------------------------------------------------------
    @abstractmethod
    def solve(self,
        arm,
        target: Pose,
        frame: Union[str, int],
        q0: np.ndarray,
        *,
        tol: float = 1e-3,
        iters: int = 200,
        **kwargs,
    ) -> IKResult:
        """求单组解（通用参数 → 特有参数排序）。

        :param target: 目标位姿 ``Pose``（xyz + 四元数；与 ``arm.fkine`` 输出同约定）。
        :param frame: 目标帧名（``str``）或帧索引（``int``）。
        :param q0: ``(n,)`` 关节角参考，弧度——数值法作**迭代起点**；
            解析法作**选解参考**：全部闭式解先剔除超出关节软限位者，再取与``q0`` 各关节角
            偏差平方和最小的一组（即构型上最接近 ``q0`` 的解，避免机械臂大幅换构型）。
        :param tol: 残差范数收敛容差（数值法特有，默认 ``1e-3``）。
        :param iters: 迭代次数上限（数值法特有，默认 ``200``）。
        :return: :class:`IKResult`，``q`` 为 ``(n,)`` 单解；无可行解时``success=False``,``q`` 为空。
        """

    # ----------------------------------------------------------
    # ikine 可选覆盖：solve_all（全解，仅解析法实现提供）
    # ----------------------------------------------------------
    def solve_all(self,
        arm,
        target: Pose,
        frame: Union[str, int],
        **kwargs,
    ) -> IKResult:
        """求全部解析解。

        :param target: 目标位姿 ``Pose``。
        :param frame: 目标帧名（``str``）或帧索引（``int``）。
        :return: :class:`IKResult`，``q`` 为 ``(K,n)``（每行一组解；6R 通常``K=8``）。
        **不做限位剔除**，但逐关节尝试 ``±2π`` 平移，使每组解尽量落入关节限位范围内（等价角中取离限位区间最近者）。
        """
        raise NotImplementedError(
            "ikine_solver.py - IkineSolver.solve_all：默认实现不可用")

    # ----------------------------------------------------------
    # ikine 共享助手（±2π 归位 + 限位剔除选最近；解析法实现直接复用）
    # ----------------------------------------------------------
    @staticmethod
    def _shift_2pi(q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> np.ndarray:
        """逐关节加 ``k·2π``（k 取最优整数），使 ``q`` 尽量落入 ``[q_min, q_max]``。

        关节限位区间窄于 ``2π`` 时可能无整数 k 可入界，此时取离区间最近的等价角。
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        q_min = np.asarray(q_min, dtype=float).reshape(-1)
        q_max = np.asarray(q_max, dtype=float).reshape(-1)
        two_pi = 2.0 * np.pi
        out = q.copy()
        for j in range(q.size):
            best, best_d = q[j], np.inf
            # 以区间中心圆整出基准 k，最优解必在其 ±1 邻域内
            k0 = int(round(((q_min[j] + q_max[j]) / 2.0 - q[j]) / two_pi))
            for k in (k0 - 1, k0, k0 + 1):
                s = q[j] + k * two_pi
                d = max(0.0, q_min[j] - s, s - q_max[j])   # 与限位区间的距离
                if d < best_d:
                    best, best_d = s, d
            out[j] = best
        return out

    @staticmethod
    def _select_nearest(sols: np.ndarray, q0: np.ndarray,
                        q_min: np.ndarray, q_max: np.ndarray) -> IKResult:
        """限位剔除 + 选最近解（解析法 :meth:`solve` 的收尾逻辑）。

        ``sols`` 为 ``(K,n)`` 候选解：剔除任一关节超出 ``[q_min, q_max]`` 的行，
        剩余行中取与 ``q0`` 各关节角偏差平方和最小的一组；全部越限时返回``success=False``、``q`` 为空。
        """
        sols = np.atleast_2d(np.asarray(sols, dtype=float))
        q0 = np.asarray(q0, dtype=float).reshape(-1)
        feas = sols[((sols >= q_min) & (sols <= q_max)).all(axis=1)]
        if feas.size == 0:
            return IKResult(q=np.zeros(0), success=False, err=float("inf"), n_iter=0)
        k = int(np.argmin(((feas - q0) ** 2).sum(axis=1)))
        return IKResult(q=feas[k], success=True, err=0.0, n_iter=0)
