"""PinDynamicsSolver —— 基于 pinocchio 的动力学默认实现。

调用 pin 原生算法，按属性约定消费 ``arm.pin_model``（各方法私有 ``pin.Data``），适配任意构型。
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from . import DynamicsSolver

__all__ = ["PinDynamicsSolver"]


class PinDynamicsSolver(DynamicsSolver):
    """动力学默认实现：M/C/G/RNEA（pinocchio 刚体动力学算法）。"""

    def idyn(self, arm, q: np.ndarray, dq: np.ndarray,
             ddq: np.ndarray) -> np.ndarray:
        """求逆动力学（RNEA）。

        :param arm: 机械臂对象。
        :param q: ``(n,)`` 关节角，弧度。
        :param dq: ``(n,)`` 关节速度，rad/s。
        :param ddq: ``(n,)`` 关节加速度，rad/s²。
        :return: ``(n,)`` 实现该运动所需的各关节力矩 ``τ``，N·m。
        """
        model = arm.pin_model
        return pin.rnea(model, pin.Data(model), np.asarray(q, dtype=float),
                        np.asarray(dq, dtype=float),
                        np.asarray(ddq, dtype=float))

    def mass_matrix(self, arm, q: np.ndarray) -> np.ndarray:
        """求关节空间惯量矩阵。

        :param arm: 机械臂对象。
        :param q: ``(n,)`` 关节角，弧度。
        :return: ``(n,n)`` 惯量矩阵 ``M(q)``——CRBA 只填上三角，此处对称化补全。
        """
        model = arm.pin_model
        M = np.array(pin.crba(model, pin.Data(model), np.asarray(q, dtype=float)),dtype=float)
        return np.triu(M) + np.triu(M, 1).T

    def coriolis(self, arm, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """求科氏/向心项。

        :param arm: 机械臂对象。
        :param q: ``(n,)`` 关节角，弧度。
        :param dq: ``(n,)`` 关节速度，rad/s。
        :return: ``(n,)`` 科氏/向心项 ``C(q,q̇)q̇``（nonLinearEffects − 重力项），N·m。
        """
        model = arm.pin_model
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        data = pin.Data(model)                  # 私有草稿纸：无共享可变状态
        return (pin.nonLinearEffects(model, data, q, dq)
                - pin.rnea(model, data, q, dq * 0.0, dq * 0.0))

    def gravity(self, arm, q: np.ndarray) -> np.ndarray:
        """求重力项。

        :param arm: 机械臂对象。
        :param q: ``(n,)`` 关节角，弧度。
        :return: ``(n,)`` 重力项 ``G(q)``（RNEA 零速度零加速度特例），N·m。
        """
        model = arm.pin_model
        q = np.asarray(q, dtype=float)
        return pin.rnea(model, pin.Data(model), q, q * 0.0, q * 0.0)

