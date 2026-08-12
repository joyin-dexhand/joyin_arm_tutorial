"""共享数据类型与枚举。

本模块定义全篇复用的不可变数据类型（``@dataclass``）与枚举（``enum``），
所有模块统一引用此处定义的类型，杜绝重复定义。

"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np
from .transforms import T_to_Rp, R_to_quat, quat_to_R, make_T

__all__ = [
    # 枚举
    "ControlMode",
    "Severity",
    "SafetyAction",
    "TrajectorySpace",
    # 数据类
    "Wrench",
    "Twist",
    "Pose",
    "Transform",
    "JointState",
    "TcpState",
    "ArmState",
    "JointLimits",
    "TcpLimits",
    "Violation",
    "IKResult",
    "ComplianceParams",
    # 纯函数
    "clamp_to_limits",
]


# ============================================================
# 枚举
# ============================================================
class ControlMode(Enum):
    """关节控制模式

    下发指令时，``Arm.command()`` 根据 mode 调用 backend 的不同方法。

    :cvar POSITION: 位置模式，下发目标关节角 ``q``。
    :cvar VELOCITY: 速度模式，下发目标关节速度 ``dq``。
    :cvar TORQUE: 力矩模式，下发目标关节力矩 ``tau``。
    :cvar MIT: MIT 阻抗/前馈模式，下发 ``(q, dq, tau_ff, kp, kd)``，
    """

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    MIT = "mit"


class Severity(Enum):
    """安全违规分级"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SafetyAction(Enum):
    """安全响应策略

    :cvar NONE: 不响应（仅记录）。
    :cvar CLAMP: 裁剪到限位内（软限位违规的默认动作）。
    :cvar DAMPING_HOLD: 阻尼力矩保持。
    :cvar FREEZE: 构型维持。
    :cvar ESTOP: 紧急停止。
    """

    NONE = "none"
    CLAMP = "clamp"
    DAMPING_HOLD = "damping_hold"
    FREEZE = "freeze"
    ESTOP = "estop"


class TrajectorySpace(Enum):
    """轨迹空间标识"""

    JOINT = "joint"
    CARTESIAN = "cartesian"


# ============================================================
# 几何/运动学数据类
# ============================================================
@dataclass
class Wrench:
    """六维力/力矩。

    :ivar force: ``(3,)`` 力向量，单位 N。
    :ivar torque: ``(3,)`` 力矩向量，单位 N·m。
    """

    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Twist:
    """空间速度

    :ivar linear: ``(3,)`` 线速度，单位 m/s。
    :ivar angular: ``(3,)`` 角速度，单位 rad/s。
    """

    linear: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Pose:
    """统一位姿表示（全篇复用）。

    以 position(3) + orientation（单位四元数 4） 存储，与 ROS2
    ``geometry_msgs/Pose`` 语义对齐。正运动学等内部计算仍用 4×4 齐次矩阵 ``T``，
    通过 :meth:`from_T` / :attr:`T` 在两表示间互转。

    :ivar position: ``(3,)`` 位置向量 ``[x, y, z]``。
    :ivar orientation: ``(4,)`` 单位四元数，``(w, x, y, z)`` 实部在前。
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )

    @classmethod
    def from_T(cls, T: np.ndarray) -> Pose:
        """从 4×4 齐次变换矩阵构造。"""
        R, p = T_to_Rp(T)
        return cls(position=p, orientation=R_to_quat(R))

    @property
    def T(self) -> np.ndarray:
        """派生的 4×4 齐次变换矩阵（每次调用现算）。"""
        return make_T(quat_to_R(self.orientation), self.position)


@dataclass
class Transform:
    """坐标系间的刚体变换（与 ROS2 ``geometry_msgs/Transform`` 语义对齐）。

    :ivar translation: ``(3,)`` 平移向量。
    :ivar rotation: ``(4,)`` 单位四元数，``(w, x, y, z)`` 实部在前。
    """

    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )

    @classmethod
    def from_T(cls, T: np.ndarray) -> Transform:
        """从 4×4 齐次变换矩阵构造。"""
        R, p = T_to_Rp(T)
        return cls(translation=p, rotation=R_to_quat(R))

    @property
    def T(self) -> np.ndarray:
        """派生的 4×4 齐次变换矩阵（每次调用现算）。"""
        return make_T(quat_to_R(self.rotation), self.translation)

    def to_pose(self) -> Pose:
        """转成 :class:`Pose`（字段值不变，仅语义包装不同）。"""
        return Pose(position=self.translation, orientation=self.rotation)


# ============================================================
# 状态快照
# ============================================================
@dataclass
class JointState:
    """关节层状态快照（Ch11 §11.1）。

    :ivar q: ``(n,)`` 关节位置，弧度。
    :ivar dq: ``(n,)`` 关节速度，弧度/秒。
    :ivar ddq: ``(n,)`` 关节加速度，弧度/秒²。
    :ivar tau: ``(n,)`` 关节力矩，N·m。
    :ivar status: ``(n,)`` int 关节电机状态码；
                  ``0``=失能，``1``=使能，其他值代表过温/过流/通信异常等信息。
    :ivar temp_coil: ``(n,)`` 线圈温度，单位 ℃。
    :ivar temp_driver: ``(n,)`` 驱动器温度，单位 ℃。
    :ivar voltage: 总线电压，单位 V。
    :ivar current: 总线电流，单位 A。
    """

    q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ddq: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(0))
    status: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    temp_coil: np.ndarray = field(default_factory=lambda: np.zeros(0))
    temp_driver: np.ndarray = field(default_factory=lambda: np.zeros(0))
    voltage: float = 0.0
    current: float = 0.0


@dataclass
class TcpState:
    """末端层状态快照（Ch11 §11.2）。"""

    pose: Pose = field(default_factory=Pose)
    twist: Twist = field(default_factory=Twist)
    wrench: Wrench = field(default_factory=Wrench)


@dataclass
class ArmState:
    """整机状态聚合快照

    :ivar joint: 关节层状态。
    :ivar tcp: 末端层状态。
    :ivar mode: 当前控制模式。
    :ivar timestamp: 时间戳，秒。
    :ivar errors: 错误信息列表。
    """

    joint: JointState = field(default_factory=JointState)
    tcp: TcpState = field(default_factory=TcpState)
    mode: ControlMode = ControlMode.POSITION
    timestamp: float = 0.0
    errors: List[str] = field(default_factory=list)


# ============================================================
# 限位声明
# ============================================================
@dataclass
class JointLimits:
    """关节限位声明（Ch11 §11.1）。

    所有数组维度均为 ``(n,)``，``n`` 为机械臂自由度数。

    - **硬限位**实例（``Arm.joint_limits``）：URDF/电机物理极限；
      ``q_min/q_max`` 为位置硬限位，供 :func:`safety.joint_limits_check`
      监测报警（硬限位违规 → :attr:`Severity.ERROR` / :attr:`Severity.CRITICAL`）。
    - **软限位**实例（``Arm.joint_limits_soft``）：略窄于硬限位、留余量；
      其 ``q_min/q_max`` 即 ``Arm.qlow/qhigh``，供
      :meth:`Arm.rand_q` / :meth:`Arm.clamp_q` / :meth:`Arm.is_q_valid`
      运行时裁剪（软限位违规 → :attr:`SafetyAction.CLAMP`）。

    :ivar q_min: ``(n,)`` 关节角下限，弧度。
    :ivar q_max: ``(n,)`` 关节角上限，弧度。
    :ivar dq_max: ``(n,)`` 关节速度上限，弧度/秒。
    :ivar ddq_max: ``(n,)`` 关节加速度上限，弧度/秒²。
    :ivar tau_max: ``(n,)`` 关节力矩上限，N·m。
    :ivar temp_coil_max: ``(n,)`` 线圈温度上限，℃。
    :ivar temp_driver_max: ``(n,)`` 驱动器温度上限，℃。
    :ivar voltage_min: ``(n,)`` 电压下限，V。
    :ivar voltage_max: ``(n,)`` 电压上限，V。
    :ivar current_max: ``(n,)`` 电流上限，A。
    """

    q_min: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q_max: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dq_max: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ddq_max: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tau_max: np.ndarray = field(default_factory=lambda: np.zeros(0))
    temp_coil_max: np.ndarray = field(default_factory=lambda: np.zeros(0))
    temp_driver_max: np.ndarray = field(default_factory=lambda: np.zeros(0))
    voltage_min: np.ndarray = field(default_factory=lambda: np.zeros(0))
    voltage_max: np.ndarray = field(default_factory=lambda: np.zeros(0))
    current_max: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class TcpLimits:
    """末端限位声明（Ch11 §11.2）。

    :ivar workspace_box: ``(3,2)`` 末端工作空间包围盒，
                         每行 ``[min, max]`` 对应 x/y/z，单位 m。
    :ivar v_lin_max: 末端最大线速度，m/s。
    :ivar v_ang_max: 末端最大角速度，rad/s。
    :ivar f_max: 末端最大力，N。
    :ivar t_max: 末端最大力矩，N·m。
    """

    workspace_box: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 2))
    )
    v_lin_max: float = 0.0
    v_ang_max: float = 0.0
    f_max: float = 0.0
    t_max: float = 0.0


@dataclass
class Violation:
    """安全监控器输出的违规记录（Ch11）。

    :ivar layer: 监控层标识，``"joint"``/``"tcp"``/``"arm"`，
                 分别对应 §11.1/11.2/11.3。
    :ivar joint_idx: 关节索引（末端/整机层违规可置 ``-1``）。
    :ivar metric: 违规指标名（如 ``"q"``/``"dq"``/``"temp_coil"``）。
    :ivar value: 实测值。
    :ivar limit: 限位阈值。
    :ivar severity: 严重程度分级。
    """

    layer: str = ""
    joint_idx: int = -1
    metric: str = ""
    value: float = 0.0
    limit: float = 0.0
    severity: Severity = Severity.WARNING


# ============================================================
# 求解结果
# ============================================================
@dataclass
class IKResult:
    """逆运动学求解结果（Ch3）。

    :ivar q: ``(n,)`` 解算得到的关节角，弧度。
    :ivar success: 是否收敛到满足精度的解。
    :ivar err: 末端位姿误差（残差范数）。
    :ivar n_iter: 迭代次数。
    """

    q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    success: bool = False
    err: float = float("inf")
    n_iter: int = 0


@dataclass
class ComplianceParams:
    """笛卡尔柔顺参数（Ch9 阻抗/导纳共用）。

    阻抗控制器 ``K_d/D_d/M_d`` 与导纳控制器 ``K_a/D_a/M_a``
    均映射到本类的 ``K/D/M`` 三个 6×6 笛卡尔矩阵（默认对角）。

    :ivar K: ``(6,6)`` 刚度矩阵。
    :ivar D: ``(6,6)`` 阻尼矩阵。
    :ivar M: ``(6,6)`` 惯量矩阵。
    """

    K: np.ndarray = field(default_factory=lambda: np.eye(6))
    D: np.ndarray = field(default_factory=lambda: np.eye(6))
    M: np.ndarray = field(default_factory=lambda: np.eye(6))


# ============================================================
# 纯函数
# ============================================================
def clamp_to_limits(targets: np.ndarray, limits: JointLimits) -> np.ndarray:
    """将运动指令裁剪到关节硬限位内（软防护共享底层）。

    所有运动指令下发前都应过本函数：对 ``q_min/q_max`` 做逐元素裁剪。
    若需软限位裁剪，传入 :attr:`Arm.joint_limits_soft` 即可（其
    ``q_min/q_max`` 即软限位 ``qlow/qhigh``）。

    :param targets: ``(n,)`` 或 ``(N,n)`` 目标关节角，弧度。
    :param limits: 关节限位声明。
    :return: 与 ``targets`` 同形状的裁剪后关节角。
    :raises ValueError: ``targets`` 与限位维度不匹配时抛出。
    """
    targets = np.asarray(targets, dtype=float)
    q_min = np.asarray(limits.q_min, dtype=float)
    q_max = np.asarray(limits.q_max, dtype=float)

    # 维度校验：目标最后一维必须等于关节数（限位数组长度）
    if targets.shape[-1] != q_min.shape[0]:
        raise ValueError(
            f"目标关节角最后一维 {targets.shape[-1]} 与限位维度 "
            f"{q_min.shape[0]} 不匹配"
        )
    # numpy.clip 逐元素裁剪：低于 q_min 的抬到 q_min，高于 q_max 的压到 q_max
    # 支持 (n,) 和 (N,n) 两种形状（广播）
    return np.clip(targets, q_min, q_max)
