"""``joyarm.utils.interfaces`` —— 核心接口协议（依赖倒置）。

定义 :class:`MasProtocol`：``robotics`` / ``safety`` 算法层对多轴本体对象的最小
结构化契约。算法层依赖本 Protocol 而非具体 :class:`~joyarm.arms.mas.Mas`，
从而保证 **单向依赖、无环**——

- ``arms`` import ``robotics`` / ``safety`` 做**门面委托**（``mas.fkine()`` 等）；
- ``robotics`` / ``safety`` **不 import** ``arms``，仅依赖本 Protocol（鸭子类型）。
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, Union, runtime_checkable

import numpy as np

from .types import JointLimits, TcpLimits

__all__ = ["MasProtocol"]


@runtime_checkable
class MasProtocol(Protocol):
    """算法层（``robotics`` / ``safety``）对多轴本体对象的最小接口契约。

    任何具备下列属性/方法的对象均可作为 ``mas`` 传入算法函数（结构化鸭子类型）。
    :class:`~joyarm.arms.mas.Mas` / :class:`~joyarm.arms.arm.Arm` 及其子类即满足本契约
    （``Arm`` 是-a ``Mas``）。
    """

    # ---- pinocchio 模型 / 数据（重依赖，类型标 Any）----
    model: Any
    data: Any

    # ---- 维度 / 帧 / 基坐标系 ----
    n: int
    nv: int
    ee_frame_name: str
    ee_frame_id: int
    T_base: np.ndarray

    # ---- 限位 ----
    joint_limits: JointLimits
    joint_limits_soft: JointLimits
    tcp_limits: TcpLimits
    q_neutral: np.ndarray

    def frame_placement(
        self, q: np.ndarray, frame: Optional[Union[str, int]] = None
    ) -> np.ndarray:
        """底层单次 FK：返回指定帧在基坐标系下的 ``(4,4)`` 位姿。"""
        ...
