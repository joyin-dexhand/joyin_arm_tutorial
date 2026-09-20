"""``joyarm_core.utils`` —— 基础层子包。

承载底层的共享基础设施（仅依赖 numpy / 标准库，被 robotics / backend / joyarm 依赖）：

- :mod:`joyarm_core.utils.transforms` ：  纯 numpy SE(3)/SO(3) 数学：旋转矩阵、RPY、轴角、四元数、齐次变换的互转与运算，以及伴随 ``adT``、球面插值 ``slerp``。
- :mod:`joyarm_core.utils.types`      ：  跨层共享的 ``@dataclass`` 数据类型（``ArmState``、``Pose``、``JointLimits`` 等）与枚举（``ControlMode`` 等），全库统一引用此处定义、杜绝重复定义。
- :mod:`joyarm_core.utils.limits`     ：  关节限位守卫（``clamp_to_limits`` 指令下发前裁剪）+ 硬 / 软 / 末端限位构建解析 + 限位内均匀采样。

"""

# =======================================================
# 详细注释请跳转源码文件。建议由模块进行导入而非直接从本包导入（可读性更高）：
#   from joyarm_core.utils.types import ArmState, Pose...
# =======================================================

from .types import (
    ArmState,
    TcpState,
    JointState,
    Wrench,
    Twist,
    Pose,
    TrajFrame,
    JointLimits,
    TcpLimits,
    IKResult,
    ComplianceParams,
    ControlMode,
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
    "Pose",
    "TrajFrame",
    "JointLimits",
    "TcpLimits",
    "IKResult",
    "ComplianceParams",
    "ControlMode",
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
