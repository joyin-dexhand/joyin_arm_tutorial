"""``joyarm_core`` —— JoyArm 机械臂教程核心 SDK 库。

组合根架构（仅向下依赖；robotics 一域一子包，REGISTRY 选型）::

    joyarm/     组合根：JoyArm(单类，门面+成员组装) · JoyArmFactory(型号名选型)
    ─────────────────────────────────────────────
    robotics/    算法层（一域一子包：ABC+REGISTRY，具体算法为教程各章教学内容）：
                 fkine · ikine · jacobian · trajectory · dynamics · control
                                                              ← 鸭子类型消费 arm
    backend/    通信层：Backend(整机) → BackendDM（name 选型：REGISTRY + get_backend）
    ─────────────────────────────────────────────
    utils/       transforms(数学) · types(共享类型) · limits(限位守卫)
    robot_model/ configs/   URDF+meshes 资产 / per-model YAML（basic/joyarm/robotics/backend 等段）

新型号 = ``configs/<型号>.yaml`` + ``robot_model/`` 资产 + ``backend/backend_*.py``
（型号与整机后端 1:1）；无型号子类。
ROS2 封装（节点/launch/rviz2等）在 ``joyarm_ros2_ws/src/joyarm_node``（详见 AGENTS.md）；
监测已裁撤：指令守卫在 utils/limits.py，状态监测归 ROS2 节点（Ch11）。
命名约定：类名驼峰，文件名小写 snake_case。用法::

    from joyarm_core import joyarm_factory   # 推荐：型号名唯一参数

    arm = joyarm_factory("joyarm_dm")  # 离线组合根：加载 configs+URDF，按 robotics: 组装成员
    # 等价直用：from joyarm_core import JoyArm; arm = JoyArm("joyarm_dm")
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
    Rp_to_T,
    T_to_Rp,
    T_inv,
    T_mul,
    adT,
    slerp,
)

# ---- 设备模型层（JoyArm 单类 + 型号工厂）----
from .joyarm.joyarm import JoyArm
from .joyarm import JoyArmFactory, joyarm_factory

# ---- 算法层（robotics；鸭子类型消费 arm，不 import joyarm）----
from .robotics.fkine import fkine
from .robotics.ikine import ikine, ikine_constrained
from .robotics.jacobian import jac, manipulability, cond_number, statics
from .robotics.trajectory import Trajectory
from .robotics.dynamics import idyn, mass_matrix, coriolis, gravity, cartesian_inertia
# 求解器策略接口（JoyArm 六域成员字典的契约；具体实现为教程各章教学内容）
from .robotics.fkine import FkineSolver
from .robotics.ikine import IkineSolver
from .robotics.jacobian import JacobianSolver
from .robotics.dynamics import DynamicsSolver
from .robotics.trajectory import TrajPlanner
from .robotics.control import Controller

# ---- 指令路径守卫（clamp_to_limits；状态监测/日志归 ROS2 节点，Ch11）----
from .utils.limits import clamp_to_limits

# ---- 通信层（整机后端，两层继承 + name 选型注册表）----
from . import backend
from .backend import (
    Backend,
    BackendDM,
    get_backend,
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
    "Rp_to_T",
    "T_to_Rp",
    "T_inv",
    "T_mul",
    "adT",
    "slerp",
    # 设备模型层
    "JoyArm",
    "JoyArmFactory",
    "joyarm_factory",
    # 算法层
    "fkine",
    "ikine",
    "ikine_constrained",
    "jac",
    "manipulability",
    "cond_number",
    "statics",
    "Trajectory",
    "idyn",
    "mass_matrix",
    "coriolis",
    "gravity",
    "cartesian_inertia",
    # 求解器策略接口（各章实现注册后接入）
    "FkineSolver",
    "IkineSolver",
    "JacobianSolver",
    "DynamicsSolver",
    "TrajPlanner",
    "Controller",
    # 指令路径守卫（监测/日志归 ROS2 节点）
    "clamp_to_limits",
    # 通信层（整机后端）
    "backend",
    "Backend",
    "BackendDM",
    "get_backend",
    # 版本
    "__version__",
]
