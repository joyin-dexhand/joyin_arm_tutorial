"""`joyarm_core.utils` —— 共享工具层

底层的共享基础设施（仅依赖 numpy / 标准库，被 robotics / backend / joyarm 依赖）：

- :mod:`joyarm_core.utils.transforms` ：  纯 numpy 数学；
- :mod:`joyarm_core.utils.types`      ：  `@dataclass` 数据类型与枚举，全库统一引用、杜绝重复定义。
- :mod:`joyarm_core.utils.limits`     ：  关节限位守卫 + 限位构建解析 + 限位内均匀采样。

"""

# =======================================================
# 为提高可读性，建议由模块进行导入而非从包导入，如：
#    from joyarm_core.utils.types import ArmState, Pose...
# =======================================================

from .types import (
    ArmState,
    TcpState,
    JointState,
    Wrench,
    Twist,
    Accel,
    Pose,
    TrajFrame,
    JointLimits,
    TcpLimits,
    IKResult,
    ComplianceParams,
    ControlMode,
    MotionState,
    SafetyEvent,
    STATE_FROM,
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
from .limits import clamp_to_limits, limits_from_joint_cfgs, rand_within_limits, soft_limits_from_cfg, tcp_limits_from_cfg

__all__ = [
    # types
    "ArmState",
    "TcpState",
    "JointState",
    "Wrench",
    "Twist",
    "Accel",
    "Pose",
    "TrajFrame",
    "JointLimits",
    "TcpLimits",
    "IKResult",
    "ComplianceParams",
    "ControlMode",
    "MotionState",
    "SafetyEvent",
    "STATE_FROM",
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
    # limits
    "clamp_to_limits",
    "limits_from_joint_cfgs",
    "rand_within_limits",
    "soft_limits_from_cfg",
    "tcp_limits_from_cfg",
]
