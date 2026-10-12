"""DynamicsSolver —— 动力学求解器基类接口（ABC）

已知：关节状态 q/dq/ddq；求解：关节力矩 τ 及 M/C/G 各项。

外部调用：
    - :meth:`idyn(arm, q, dq, ddq)` —— 逆动力学：状态+加速度 → τ；
    - :meth:`mass_matrix(arm, q)` —— 关节空间惯量矩阵 M；
    - :meth:`coriolis(arm, q, dq)` —— 科氏/向心项 C；
    - :meth:`gravity(arm, q)` —— 重力项 G。

子类需实现：
    - :meth:`idyn(arm, q, dq, ddq)` —— 由关节状态与加速度计算关节力矩 τ；
    - :meth:`mass_matrix(arm, q)` —— 计算关节空间惯量矩阵 M(q)；
    - :meth:`coriolis(arm, q, dq)` —— 计算科氏/向心项 C(q,dq)；
    - :meth:`gravity(arm, q)` —— 计算重力项 G(q)。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["DynamicsSolver"]


class DynamicsSolver(ABC):

    @abstractmethod
    def idyn(self,
        arm,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
    ) -> np.ndarray:
        """求逆动力学（RNEA）。

        :param arm: 机械臂对象。
        :param q: `(n,)` 关节角，弧度。
        :param dq: `(n,)` 关节速度，rad/s。
        :param ddq: `(n,)` 关节加速度，rad/s²。
        :return: `(n,)` 实现该运动所需的各关节力矩 `τ`，N·m。
        """

    @abstractmethod
    def mass_matrix(self, arm, q: np.ndarray) -> np.ndarray:
        """求关节空间惯量矩阵。

        :param arm: 机械臂对象。
        :param q: `(n,)` 关节角，弧度。
        :return: `(n,n)` 惯量矩阵 `M(q)`。
        """

    @abstractmethod
    def coriolis(self, arm, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """求科氏/向心项。

        :param arm: 机械臂对象。
        :param q: `(n,)` 关节角，弧度。
        :param dq: `(n,)` 关节速度，rad/s。
        :return: `(n,)` 科氏/向心项 `C(q,q̇)q̇`（向量形式），N·m。
        """

    @abstractmethod
    def gravity(self, arm, q: np.ndarray) -> np.ndarray:
        """求重力项。

        :param arm: 机械臂对象。
        :param q: `(n,)` 关节角，弧度。
        :return: `(n,)` 重力项 `G(q)`，N·m。
        """

