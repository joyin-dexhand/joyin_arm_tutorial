"""``joyarm_core`` —— JoyArm 机械臂核心 SDK 库。

分层结构（依赖只允许自上而下，下层不知道上层存在）::

    joyarm/     JoyArm：唯一组装入口（读config配置，把算法、通信各部分装配成一个
                可用的机械臂对象）；JoyArmFactory：按型号名创建joyarm对象
    ─────────────────────────────────────────────
    robotics/   算法层：六个子域（正运动学 fkine · 逆运动学 ikine · 雅可比 jacobian · 
                轨迹 trajectory · 动力学 dynamics · 控制 control），每域 = 接口 + 注册表；
                加 ``@register`` 自动注册（只调用 ``arm`` 公开属性/方法，不反向依赖 joyarm）
    backend/    通信层：Backend（整机接口）→ Backend*（具体后端实现）
    ─────────────────────────────────────────────
    utils/      数学 / 共享数据类型 / 限位裁剪 / 关节插值 / 周期线程等
    robot_model/ configs/   URDF+网格资产 / 每型号一份 YAML 配置

接入新型号：复制 ``configs/joyarm_template.yaml`` 填写 + 放入
``robot_model/<robot>/`` 资产 + 新写 ``backend/backend_<型号>.py``（注册表加
一行）即可，无需改动 JoyArm 本身（型号与整机后端一一对应）。
ROS2 封装（节点/launch/rviz2 等）在 ``joyarm_ros2_ws/``（详见 AGENTS.md）；
状态监测归 ROS2 节点（Ch11）。命名约定：类名驼峰，文件名小写 snake_case。用法::

    from joyarm_core import joyarm_factory   # 推荐入口：型号名唯一参数

    arm = joyarm_factory("joyarm_dm")  # 读配置+URDF 完成组装（默认不连接真机）
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

# ---- 设备模型层（JoyArm 单类 + 型号工厂）----
from .joyarm.joyarm import JoyArm
from .joyarm import JoyArmFactory, joyarm_factory

# ---- 算法层（robotics；只按属性约定调用 arm，不 import joyarm）----
from .robotics.fkine import FkineSolver
from .robotics.ikine import IkineSolver
from .robotics.jacobian import JacobianSolver
from .robotics.dynamics import DynamicsSolver
from .robotics.trajectory import TrajPlanner
from .robotics.control import Controller
from .robotics._registry import register

# ---- 指令路径守卫（clamp_to_limits）+ 限位构建辅助 ----
from .utils.limits import clamp_to_limits, joint_limits_from_model, limits_from_joint_cfgs, soft_limits_from_cfg

# ---- 通信层（整机后端，两层继承 + name 选型注册表）----
from . import backend
from .backend import (
    Backend,
    BackendDM,
    BackendDMMujoco,
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
    "TrajFrame",
    "JointLimits",
    "TcpLimits",
    "IKResult",
    "ComplianceParams",
    "ControlMode",
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
    # 求解器策略接口（各章实现注册后接入）+ 注册装饰器
    "FkineSolver",
    "IkineSolver",
    "JacobianSolver",
    "DynamicsSolver",
    "TrajPlanner",
    "Controller",
    "register",
    # 指令路径守卫 + 限位构建辅助
    "clamp_to_limits",
    "joint_limits_from_model",
    "limits_from_joint_cfgs",
    "soft_limits_from_cfg",
    # 通信层（整机后端）
    "backend",
    "Backend",
    "BackendDM",
    "BackendDMMujoco",
    "get_backend",
    # 版本
    "__version__",
]
