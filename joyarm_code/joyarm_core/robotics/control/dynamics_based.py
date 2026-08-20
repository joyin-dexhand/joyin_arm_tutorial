"""基于动力学的控制（Part 2，Ch8 占位）。

计算力矩 / 逆动力学控制：经 ``arm._dynamics_solver``（idyn / mass_matrix 等）
做前馈补偿。
"""
from __future__ import annotations

import numpy as np

__all__ = ["computed_torque_control", "inverse_dynamics_control"]


def computed_torque_control(
    arm,
    q: np.ndarray,
    dq: np.ndarray,
    ddq_des: np.ndarray,
    Kp,
    Kd,
) -> np.ndarray:
    """计算力矩控制（§8.3）：``τ = M(q)q̈_d + C(q,q̇)q̇ + G(q) + Kp·e + Kd·ė`` → ``(n,)``。"""
    raise NotImplementedError("computed_torque_control 待 Ch8 实现")


def inverse_dynamics_control(
    arm,
    q: np.ndarray,
    dq: np.ndarray,
    ddq_des: np.ndarray,
    Kp,
    Kd,
) -> np.ndarray:
    """逆动力学控制（别名 / 变体，阻抗前置）→ ``(n,)``。"""
    raise NotImplementedError("inverse_dynamics_control 待 Ch8 实现")
