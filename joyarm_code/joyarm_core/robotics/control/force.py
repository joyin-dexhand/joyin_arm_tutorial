"""力控制（Part 3，Ch9 占位）。

六维力传感器抽象 + 纯力控 / 力位混合 / 阻抗 / 导纳；依赖 ``arm.jac`` /
``arm.cartesian_inertia`` 门面（自动跟随求解器替换）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ...utils.types import Wrench

__all__ = [
    "ForceTorqueSensor",
    "JointTorqueSensor",
    "pure_force_control",
    "HybridForcePosition",
    "ImpedanceControl",
    "AdmittanceControl",
    "compute_cartesian_impedance",
]


class ForceTorqueSensor:
    """六维力传感器抽象：``read`` 读末端力/力矩 · ``bias`` 去零漂 · ``set_mount`` 设腕部安装变换。"""

    def __init__(self, **kwargs):
        self._mount_T: Optional[np.ndarray] = None

    def read(self) -> Wrench:
        """读取末端六维力。"""
        raise NotImplementedError("ForceTorqueSensor.read 待 Ch9 实现")

    def bias(self, n_samples: int = 100) -> None:
        """采集零漂（n_samples 帧平均后作偏置）。"""
        raise NotImplementedError("ForceTorqueSensor.bias 待 Ch9 实现")

    def set_mount(self, T: np.ndarray) -> None:
        """腕部安装变换 ``T``（传感器到末端）。"""
        raise NotImplementedError("ForceTorqueSensor.set_mount 待 Ch9 实现")


class JointTorqueSensor:
    """关节力矩反馈（无力传感器分支的主反馈源）。"""

    def read(self) -> np.ndarray:
        """读取关节力矩 ``(n,)``。"""
        raise NotImplementedError("JointTorqueSensor.read 待 Ch9 实现")


def pure_force_control(arm, F_desired, frame=None):
    """§9.1 纯力控：``τ = Jᵀ F``（经 ``arm.statics``）。"""
    raise NotImplementedError("pure_force_control 待 Ch9 实现")


class HybridForcePosition:
    """§9.2 力位混合：选择矩阵 ``W``（6×6 对角，1=力控 0=位控）划分子空间并合成。"""

    def __init__(self, **kwargs):
        self._W: Optional[np.ndarray] = None

    def set_selection(self, W: np.ndarray) -> None:
        """设置 6×6 对角选择矩阵。"""
        raise NotImplementedError("HybridForcePosition.set_selection 待 Ch9 实现")


class ImpedanceControl:
    """§9.3 阻抗控制：``M_d ẍ + D_d(ẋ−ẋ_d) + K_d(x−x_d) = F_ext``。

    构造参数 ``K_d, D_d, M_d``（6×6 笛卡尔，默认对角）= ComplianceParams 的 K/D/M；
    有力传感器用实测 ``F_ext``，无力传感器用关节力矩估计；需 ``arm.gravity`` /
    ``arm.coriolis`` 做补偿。
    """

    def __init__(self, K_d=None, D_d=None, M_d=None, **kwargs):
        self.K_d = K_d
        self.D_d = D_d
        self.M_d = M_d


class AdmittanceControl:
    """§9.4 导纳控制：``M_a ẍ + D_a ẋ + K_a(x−x_0) = F_ext`` → 位姿偏移 ``Δx``
    喂给 Part1 位置内环；``K_a, D_a, M_a`` = ComplianceParams 的 K/D/M。"""

    def __init__(self, K_a=None, D_a=None, M_a=None, **kwargs):
        self.K_a = K_a
        self.D_a = D_a
        self.M_a = M_a


def compute_cartesian_impedance(arm, K_d=None, D_d=None, **kwargs):
    """底层计算，复用 ``arm.cartesian_inertia``（Λ 模板）。"""
    raise NotImplementedError("compute_cartesian_impedance 待 Ch9 实现")
