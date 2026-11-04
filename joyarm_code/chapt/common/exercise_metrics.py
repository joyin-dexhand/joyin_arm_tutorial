"""各章实践共用的数值校验与误差指标。"""
from __future__ import annotations

import numpy as np

from joyarm_core import Pose, R_to_axis_angle


def frame_end_name(arm) -> str:
    """从 JoyArm 已加载配置读取末端帧名。"""
    return str(arm.get_config()["joyarm"]["frame_end"])


def pose_error(target: Pose, actual: Pose) -> tuple[float, float]:
    """返回位置误差（mm）和姿态误差（deg）。"""
    position_mm = float(
        np.linalg.norm(target.position - actual.position) * 1000.0
    )
    relative_rotation = target.T[:3, :3] @ actual.T[:3, :3].T
    _, angle = R_to_axis_angle(relative_rotation)
    return position_mm, float(np.rad2deg(angle))


def within_joint_limits(q: np.ndarray, limits, tolerance: float = 0.0) -> bool:
    """判断关节角是否位于给定限位内。"""
    values = np.asarray(q, dtype=float).reshape(-1)
    return bool(
        np.all(values >= np.asarray(limits.q_min) - tolerance)
        and np.all(values <= np.asarray(limits.q_max) + tolerance)
    )


def jacobian_metrics(jacobian: np.ndarray) -> tuple[float, float]:
    """返回雅可比的最小奇异值和条件数。"""
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("雅可比必须是有限的二维数组")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    sigma_min = float(singular_values[-1])
    condition = float(
        np.inf if sigma_min == 0.0 else singular_values[0] / sigma_min
    )
    return sigma_min, condition
