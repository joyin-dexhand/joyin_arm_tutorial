"""``joyarm_core.utils`` —— 基础层子包。

承载最底层的共享基础设施（仅依赖 numpy / 标准库，被所有上层依赖）：

- :mod:`joyarm_core.utils.transforms` ：  SE(3)/SO(3) 数学（旋转矩阵、RPY、轴角、四元数、齐次变换、球面插值 slerp）。
- :mod:`joyarm_core.utils.types`      ：  跨层共享的 ``@dataclass`` 数据类型（``ArmState``、``Pose``、``JointLimits`` 等）与枚举（``ControlMode`` 等）。
- :mod:`joyarm_core.utils.interfaces` ：  ``ArmProtocol`` 接口契约（依赖倒置——算法层依赖它而非 ``joyarms``）。

"""

# =======================================================
# 详细注释请跳转源码文件
# =======================================================

from .types import (
    ArmState,
    TcpState,
    JointState,
    Wrench,
    Twist,
    Pose,
    JointLimits,
    TcpLimits,
    Violation,
    IKResult,
    ComplianceParams,
    ControlMode,
    Severity,
    SafetyAction,
    TrajectorySpace,
)
from .transforms import (
    rot_x,
    rot_y,
    rot_z,
    rpy_to_R,
    R_to_rpy,
    rodrigues,
    R_to_axis_angle,
    axis_angle_to_R,
    quat_to_R,
    R_to_quat,
    quat_to_axis_angle,
    axis_angle_to_quat,
    rpy_to_quat,
    quat_to_rpy,
    quat_mul,
    quat_conj,
    quat_norm,
    Rp_to_T,
    T_to_Rp,
    T_inv,
    T_mul,
    adT,
    slerp,
)
from .interfaces import ArmProtocol

__all__ = [
    # types
    "ArmState",
    "TcpState",
    "JointState",
    "Wrench",
    "Twist",
    "Pose",
    "JointLimits",
    "TcpLimits",
    "Violation",
    "IKResult",
    "ComplianceParams",
    "ControlMode",
    "Severity",
    "SafetyAction",
    "TrajectorySpace",
    # transforms
    "rot_x",
    "rot_y",
    "rot_z",
    "rpy_to_R",
    "R_to_rpy",
    "rodrigues",
    "R_to_axis_angle",
    "axis_angle_to_R",
    "quat_to_R",
    "R_to_quat",
    "quat_to_axis_angle",
    "axis_angle_to_quat",
    "rpy_to_quat",
    "quat_to_rpy",
    "quat_mul",
    "quat_conj",
    "quat_norm",
    "Rp_to_T",
    "T_to_Rp",
    "T_inv",
    "T_mul",
    "adT",
    "slerp",
    # interfaces
    "ArmProtocol",
]
