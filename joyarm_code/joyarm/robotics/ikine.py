"""逆运动学（运动学模型层，Ch3 占位）。

求解 ``T_target → q``，默认走 **数值解**（Levenberg-Marquardt / pinocchio 黑盒）。
实现约定：迭代内部经 ``arm.frame_placement(q)`` 求当前位姿/残差（Arm 基本能力，
FK 唯一覆盖缝），勿直接调 pinocchio 摸 ``model``/``data``——以尊重子类覆盖。
若需解析解 / 手写白盒，请在 ``Arm`` 子类中覆盖。

对应章节：Ch3。
当前状态：仅签名 + docstring + ``raise NotImplementedError("Ch3 实现")``。
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..utils.types import IKResult

__all__ = ["ikine", "ikine_constrained"]


def ikine(
    arm,
    T_target: np.ndarray,
    q0: Optional[np.ndarray] = None,
    frame: Optional[Union[str, int]] = None,
    tol: float = 1e-4,
    iters: int = 200,
    **kwargs,
) -> IKResult:
    """逆运动学：目标位姿 → 关节角。

    :param arm: :class:`joyarm.arms.arm.Arm` 实例（满足 :class:`~joyarm.utils.interfaces.ArmProtocol`）。
    :param T_target: ``(4,4)`` 目标齐次位姿。
    :param q0: ``(n,)`` 初值；缺省用 ``arm.q_neutral``。
    :param frame: 末端帧名/索引；缺省为 ``arm.ee_frame_name``。
    :param tol: 收敛精度（末端位姿残差范数）。
    :param iters: 最大迭代次数。
    :return: :class:`~joyarm.utils.types.IKResult`。
    """
    raise NotImplementedError(
        "ikine 待实现"
    )


def ikine_constrained(
    arm,
    T_target: np.ndarray,
    q0: Optional[np.ndarray] = None,
    frame: Optional[Union[str, int]] = None,
    tol: float = 1e-4,
    iters: int = 200,
    **kwargs,
) -> IKResult:
    """带关节限位约束的逆运动学。

    在 :func:`ikine` 基础上增加关节软/硬限位约束，确保解在可行域内。

    :return: :class:`~joyarm.utils.types.IKResult`（解满足 ``arm.joint_limits``）。
    """
    raise NotImplementedError("ikine_constrained 待 Ch3 实现")
