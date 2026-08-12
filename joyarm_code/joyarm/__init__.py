"""``joyarm`` —— JoyArm 机械臂教程核心 SDK 库。

分层架构（仅允许向下依赖，按子包组织）::

    application/   上层应用：可视化 viz · 遥操作 teleop(Ch12) · 适配器 ros2(Ch10)/vision(Ch14)
    ─────────────────────────────────────────────
    safety/        安全层：safety(Ch11) 限位 / 自碰撞 / 外部碰撞监控
    ─────────────────────────────────────────────
    robotics/      算法层：control(Ch6/8/9) · trajectory(Ch5) · dyn(Ch8)
                   · fkine · ikine(Ch3) · jacobian(Ch4)
    ─────────────────────────────────────────────
    arm/           设备模型层：arm · joyarm_rebot_dm · gripper(Ch13)
    ─────────────────────────────────────────────
    backends/      通信层：arm_backend · joyarm_rebot_dm_backend · end_backend · gripper_backend
    ─────────────────────────────────────────────
    utils/         基础层：transforms(数学) · types(共享类型)

使用方式::

    from joyarm import JoyArmRebotDM, fkine, viz

    arm = JoyArmRebotDM()                            # backend=None 离线模式
    Q  = arm.rand_q(size=100_000)                    # 限位内采样 (N,6)
    P  = fkine(arm, Q, rep="pos")                    # 批量 FK → (N,3)
    v  = viz.make_viewer(); viz.plot_points(v, P)    # 点云可视化

详见 ``joyarm/架构设计.md``（架构定稿文档）。
"""
from __future__ import annotations

from typing import Optional

# ---- 共享类型 / 数学（基础层）----
from .utils.types import (
    ArmState,
    TcpState,
    JointState,
    Wrench,
    Twist,
    Pose,
    Transform,
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

# ---- 运动学模型层 ----
from .arm.arm import Arm
from .arm.joyarm_rebot_dm import JoyArmRebotDM
from .robotics.fkine import fkine
from .robotics.ikine import ikine, ikine_constrained
from .robotics.jacobian import jac, manipulability, cond_number, statics

# ---- 轨迹 / 动力学 ----
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

# ---- 控制（合并运动学/动力学/力控）----
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

# ---- 安全 ----
from .safety.safety import (
    StateMonitor,
    SelfCollisionChecker,
    ExternalCollisionDetector,
    SafetySupervisor,
    joint_limits_check,
    tcp_limits_check,
)

# ---- 应用支撑 ----
from .arm.gripper import Gripper
from .application.teleop import (
    Sample,
    JointSample,
    TcpSample,
    Recorder,
    Player,
    TeleopLoop,
    IdentityMapping,
    ScaledJointMapping,
    CartesianMapping,
    VrControllerMapping,
    Source,
    Sink,
)

# ---- 可视化 / 适配器 ----
from .application import viz
from . import backends
from .backends import ArmBackend, JoyArmRebotDMBackend, EndBackend, GripperBackend

__version__ = "0.1.0"

__all__ = [
    # 基础：类型
    "ArmState",
    "TcpState",
    "JointState",
    "Wrench",
    "Twist",
    "Pose",
    "Transform",
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
    # 基础：数学
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
    # 运动学模型层
    "Arm",
    "JoyArmRebotDM",
    "fkine",
    "ikine",
    "ikine_constrained",
    "jac",
    "manipulability",
    "cond_number",
    "statics",
    # 轨迹 / 动力学
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
    # 控制
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
    # 安全
    "StateMonitor",
    "SelfCollisionChecker",
    "ExternalCollisionDetector",
    "SafetySupervisor",
    "joint_limits_check",
    "tcp_limits_check",
    # 应用支撑
    "Gripper",
    "Sample",
    "JointSample",
    "TcpSample",
    "Recorder",
    "Player",
    "TeleopLoop",
    "IdentityMapping",
    "ScaledJointMapping",
    "CartesianMapping",
    "VrControllerMapping",
    "Source",
    "Sink",
    # 子模块 / 通信
    "viz",
    "backends",
    "ArmBackend",
    "JoyArmRebotDMBackend",
    "EndBackend",
    "GripperBackend",
    # 工厂
    "load_arm",
    # 版本
    "__version__",
]


# ============================================================
# 工厂函数：型号 ↔ Arm 子类、backend ↔ 通信正交组合
# ============================================================
def load_arm(
    model: str = "joyarm_rebot_dm",
    backend: Optional[str] = None,
    **kw,
) -> Arm:
    """按型号加载预设机械臂，按需挂载真机后端。

    型号（结构）与 backend（通信）正交组合：

    - ``model`` 决定 URDF / 限位 / home（:class:`Arm` 子类）。
    - ``backend`` 决定能否下发执行（``None`` 或真机）。

    :param model: 机械臂型号：

        - ``"joyarm_rebot_dm"``（默认）→ :class:`JoyArmRebotDM`

    :param backend: 通信后端：

        - ``None``（默认）：离线模式（仅计算 + 可视化，不可下发执行）。
        - ``"real"``：真机后端（:class:`JoyArmRebotDMBackend`，Ch6 实现）。

    :param kw: 透传给 Arm 子类构造函数（如 ``urdf_path``、``ee_frame_name``）。
    :return: 对应型号的 :class:`Arm` 子类实例。

    示例::

        load_arm("joyarm_rebot_dm")                # 离线
        load_arm("joyarm_rebot_dm", "real")        # 真机
    """
    # ---- 型号 → 子类 ----
    # 注册表：型号名 → Arm 子类。新增型号只需在此登记一行
    _registry = {
        "joyarm_rebot_dm": JoyArmRebotDM,
    }
    if model not in _registry:
        raise ValueError(
            f"未知型号 {model!r}；可选：{list(_registry.keys())}"
        )
    arm_cls = _registry[model]                # 取出对应的类（还没实例化）

    # ---- backend → 实例 ----
    # 通信后端与型号正交：同一个型号既能离线（None）也能挂真机（"real"）
    # 统一从 kw 中取出 backend_kwargs（仅真机后端构造需要），避免 backend=None
    # 时该键被原样透传给 Arm 子类构造函数触发意外的关键字错误
    backend_kwargs = kw.pop("backend_kwargs", {})
    backend_inst = None
    if backend == "real":
        backend_inst = JoyArmRebotDMBackend(**backend_kwargs)   # 真机 CAN 后端
    elif backend is not None:
        raise ValueError(
            f"未知 backend={backend!r}；可选：None（离线）/ 'real'（真机）"
        )

    # 把选好的 backend 和其余参数一起传给 Arm 子类构造
    return arm_cls(backend=backend_inst, **kw)
