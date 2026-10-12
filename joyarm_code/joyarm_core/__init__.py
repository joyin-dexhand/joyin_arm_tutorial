"""``joyarm_core`` —— JoyArm 机械臂核心代码库。

代码库分层结构：

    joyarm/       上层应用 JoyArm 类：读`config`配置，将`robot_model`、`robotics`、`backend`装配成机械臂对象
    ─────────────────────────────────────────────
    config/       型号配置层：每个型号一份 YAML 配置文件
    robot_model/  数字模型层：机械臂 URDF + Mesh资产
    backend/      硬件通信层：Backend（整机硬件后端接口）→ Backend<型号>（具体型号后端）
    robotics/     核心算法层：fkine / ikine / jacobian / dynamics / trajectory / control
    ─────────────────────────────────────────────
    utils/        共享工具层：数学 / 共享数据类型 / 限位裁剪 等

命名约定：类名驼峰，文件名小写 snake_case。

机械臂对象创建方式：
```
    from joyarm_core import joyArm   # 导入创建入口

    arm1 = joyArm("joyarm_<型号>")  # 创建对象
``` 
"""

from __future__ import annotations

__version__ = "0.1.0"

# ---- 共享工具 ----
from .utils.types import (
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
from .utils.limits import (
    clamp_to_limits,
    limits_from_joint_cfgs,
    rand_within_limits,
    soft_limits_from_cfg,
    tcp_limits_from_cfg,
)

# ---- 核心算法层（robotics）----
from .robotics.fkine import FkineSolver, PinFkineSolver, MdhFkineSolver
from .robotics.ikine import IkineSolver, PinIkineSolver, AnalyticIkineSolver
from .robotics.jacobian import JacobianSolver, PinJacobianSolver
from .robotics.dynamics import DynamicsSolver, PinDynamicsSolver
from .robotics.trajectory import TrajPlanner, SingleJointHoldTrajPlanner, SingleJointCubicTrajPlanner
from .robotics.control import Controller, JointPositionController

# ---- 整机后端通信层 ----
from . import backend
from .backend import (
    Backend,
    get_backend,
)

# ---- JoyArm 类 + 对象创建入口 ----
from .joyarm.joyarm import JoyArm
from .joyarm import JoyArmFactory, joyArm

__all__ = [
    # 版本
    "__version__",
    # 共享工具：类型
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
    # 共享工具：数学 / 接口
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
    # 共享工具：指令路径守卫 + 限位构建辅助
    "clamp_to_limits",
    "limits_from_joint_cfgs",
    "soft_limits_from_cfg",
    "tcp_limits_from_cfg",
    "rand_within_limits",
    # ---- JoyArm 类 + 对象创建入口 ----
    "JoyArm",
    "JoyArmFactory",
    "joyArm",
    # 求解器接口
    "FkineSolver",
    "IkineSolver",
    "JacobianSolver",
    "DynamicsSolver",
    "TrajPlanner",
    "Controller",
    # 求解器默认实现
    "PinFkineSolver",
    "MdhFkineSolver",
    "PinIkineSolver",
    "AnalyticIkineSolver",
    "PinJacobianSolver",
    "PinDynamicsSolver",
    "SingleJointHoldTrajPlanner",
    "SingleJointCubicTrajPlanner",
    "JointPositionController",
    # 通信层（整机后端）
    "backend",
    "Backend",
    "get_backend",
]
