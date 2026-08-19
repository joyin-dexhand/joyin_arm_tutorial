"""``joyarm_core.utils.interfaces`` —— 核心接口协议（依赖倒置）。

- :class:`ArmProtocol`：``robotics`` / ``safety`` 算法层对机械臂对象的最小结构化契约。保证 **单向依赖、无环**——

- ``arms`` import ``robotics`` / ``safety`` 做**门面委托**（``arm.fkine()`` 等）；
- ``robotics`` / ``safety`` 不 import ``arms``，仅依赖本 Protocol。

"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Union, runtime_checkable

import numpy as np

from .types import JointLimits, TcpLimits

__all__ = ["ArmProtocol"]


@runtime_checkable
class ArmProtocol(Protocol):
    """算法层（``robotics`` / ``safety``）对机械臂对象的最小接口契约。

    任何具备下列属性/方法的对象均可作为 ``arm`` 传入算法函数（结构化鸭子类型）：

    :class:`~joyarm_core.arms.arm.Arm` 及其子类即满足本契约。
    """

    # ---- pinocchio 模型 / 数据 ----
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
        """底层单次 FK：返回指定帧在基坐标系下的 ``(4,4)`` 位姿。

        协议**唯一方法**：算法层所有 FK 需求的分发点与覆盖缝——机械臂对象承诺
        "给定 q，任意帧在哪"；``fkine``/``ikine``/自碰撞等消费方一律经它求解，
        不自算 pinocchio。子类覆盖本方法即整体替换 FK 实现。
        """
        ...
