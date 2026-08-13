"""``joyarm`` —— JoyArm 机械臂教程核心 SDK 库（ROS2-free）。

分层架构（仅允许向下依赖；``arms`` 作富门面 ``mas.xx`` / ``arm.xx``）::

    arms/         设备模型：Mas(多轴本体) · End(末端) · Arm(Mas+End) · joyarm_rebot_dm
    ─────────────────────────────────────────────
    robotics/     算法层：fkine · ikine(Ch3) · jacobian(Ch4) · trajectory(Ch5)
                   · dyn(Ch8) · control(Ch6/8/9)
    safety/       安全层：safety(Ch11)
    backends/     通信层（三层）：Backend → BackendMas/BackendEnd
                   → BackendMasRebotDM / BackendEndJoyGripper
    ─────────────────────────────────────────────
    utils/        基础层：transforms(数学) · types(共享类型) · interfaces(MasProtocol)
    robots/       URDF + meshes 资产    configs/  per-model YAML 配置

> ROS2 封装（arm_nodes / apps / rviz2 可视化）在**兄弟包** ``joyarm_ros2``，
> 本包保持 ROS2-free，``import joyarm`` 不需要 ``rclpy``。
> 详见 ``joyarm/架构设计.md``。

命名约定：类名驼峰（Mas/End/Arm/Backend…），文件名小写 snake_case。

使用方式::

    from joyarm import JoyArmRebotDM

    arm = JoyArmRebotDM()            # 默认未连接（离线）；Arm = Mas + End
    Q  = arm.rand_q(size=100_000)    # 限位内采样 (N,6)
    P  = arm.fkine(Q, rep="pos")     # 门面：arm.fkine → (N,3)
"""
from __future__ import annotations

# ---- 共享类型 / 数学（基础层）----
from .utils.types import (
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
    clamp_to_limits,
)
from .utils.transforms import (
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
    make_T,
    T_to_Rp,
    T_inv,
    T_mul,
    adT,
    slerp,
)
from .utils.interfaces import MasProtocol

# ---- 设备模型层（Mas / End / Arm）----
from .arms.mas import Mas
from .arms.end import End
from .arms.arm import Arm
from .arms.joyarm_rebot_dm import JoyArmRebotDM

# ---- 算法层（robotics；operate on MasProtocol，不 import arms）----
from .robotics.fkine import fkine
from .robotics.ikine import ikine, ikine_constrained
from .robotics.jacobian import jac, manipulability, cond_number, statics
from .robotics.trajectory import (
    Trajectory,
    joint_cubic,
    joint_quintic,
    joint_lspb,
    joint_waypoints,
    cart_line,
    cart_arc,
    cart_to_joint,
    constant_velocity_retime,
    validate,
)
from .robotics.dyn import fdyn, idyn, mass_matrix, coriolis, gravity, cartesian_inertia
from .robotics.control import (
    ControlLoop,
    play_trajectory,
    joint_position_control,
    joint_velocity_control,
    torque_control,
    arm_position_control,
    arm_velocity_control,
    computed_torque_control,
    inverse_dynamics_control,
    ForceTorqueSensor,
    JointTorqueSensor,
    pure_force_control,
    HybridForcePosition,
    ImpedanceControl,
    AdmittanceControl,
    compute_cartesian_impedance,
)

# ---- 安全层 ----
from .safety.safety import (
    StateMonitor,
    SelfCollisionChecker,
    ExternalCollisionDetector,
    SafetySupervisor,
    joint_limits_check,
    tcp_limits_check,
)

# ---- 通信层（三层继承）----
from . import backends
from .backends import (
    Backend,
    BackendMas,
    BackendEnd,
    BackendMasRebotDM,
    BackendEndJoyGripper,
)

__version__ = "0.1.0"

__all__ = [
    # 基础：类型
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
    "clamp_to_limits",
    # 基础：数学 / 接口
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
    "make_T",
    "T_to_Rp",
    "T_inv",
    "T_mul",
    "adT",
    "slerp",
    "MasProtocol",
    # 设备模型层
    "Mas",
    "End",
    "Arm",
    "JoyArmRebotDM",
    # 算法层
    "fkine",
    "ikine",
    "ikine_constrained",
    "jac",
    "manipulability",
    "cond_number",
    "statics",
    "Trajectory",
    "joint_cubic",
    "joint_quintic",
    "joint_lspb",
    "joint_waypoints",
    "cart_line",
    "cart_arc",
    "cart_to_joint",
    "constant_velocity_retime",
    "validate",
    "fdyn",
    "idyn",
    "mass_matrix",
    "coriolis",
    "gravity",
    "cartesian_inertia",
    "ControlLoop",
    "play_trajectory",
    "joint_position_control",
    "joint_velocity_control",
    "torque_control",
    "arm_position_control",
    "arm_velocity_control",
    "computed_torque_control",
    "inverse_dynamics_control",
    "ForceTorqueSensor",
    "JointTorqueSensor",
    "pure_force_control",
    "HybridForcePosition",
    "ImpedanceControl",
    "AdmittanceControl",
    "compute_cartesian_impedance",
    # 安全层
    "StateMonitor",
    "SelfCollisionChecker",
    "ExternalCollisionDetector",
    "SafetySupervisor",
    "joint_limits_check",
    "tcp_limits_check",
    # 通信层（三层）
    "backends",
    "Backend",
    "BackendMas",
    "BackendEnd",
    "BackendMasRebotDM",
    "BackendEndJoyGripper",
    # 版本
    "__version__",
]
