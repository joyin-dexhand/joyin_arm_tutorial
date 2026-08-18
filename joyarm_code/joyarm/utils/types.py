"""共享数据类型与枚举。

本模块定义全篇复用的数据类型（可变 ``@dataclass`` 快照）与枚举（``enum``），
所有模块统一引用此处定义的类型，杜绝重复定义。
数据类继承 :class:`_ArrayEqMixin`，提供 ndarray 字段安全的相等性比较。

"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum

import numpy as np
from .transforms import T_to_Rp, R_to_quat, quat_to_R, Rp_to_T

__all__ = [
    # 枚举
    "ControlMode",      # 关节控制模式：POSITION / VELOCITY / TORQUE / MIT
    "Severity",         # 安全违规分级：INFO / WARNING / ERROR / CRITICAL
    "SafetyAction",     # 安全响应策略：NONE / CLAMP / DAMPING_HOLD / FREEZE / ESTOP
    "TrajectorySpace",  # 轨迹空间标识：JOINT / CARTESIAN
    # 数据类
    "Wrench",           # 六维力/力矩：force + torque
    "Twist",            # 空间速度：linear + angular
    "Pose",             # 统一位姿表示：position + orientation（单位四元数）
    "JointState",       # 关节状态：control_mode + q/dq/ddq/tau + enabled/error/comm_ok/angle_ok + temp_motor/temp_driver + voltage/current
    "TcpState",         # 工具中心点状态：pose + twist + wrench
    "ArmState",         # 机械臂状态：joint + tcp + mode + timestamp + errors
    "JointLimits",      # 关节限位: q_min/q_max + dq_max + ddq_max + tau_max + temp_motor_max/temp_driver_max + voltage_min/voltage_max + current_max
    "TcpLimits",        # 工具中心点限位: workspace_box + v_lin_max + v_ang_max + f_max + t_max
    "Violation",        # 安全违规: layer + joint_idx + metric + value + limit + severity
    "IKResult",         # 逆运动学结果： q + success + err + n_iter
    "ComplianceParams", # Compliance 参数 : K/D/M
]


# ============================================================
# 枚举
# ============================================================
class ControlMode(Enum):
    """关节控制模式

    下发指令时，``Arm.set_mas_command()`` 根据 mode 调用 backend 的不同方法。

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
# 数据类公共基类
# ============================================================
class _ArrayEqMixin:
    """ndarray 字段安全的 ``__eq__``。

    dataclass 默认 ``__eq__`` 逐字段 ``==`` 比较，遇 ``np.ndarray`` 因布尔
    歧义抛 ``ValueError``；本混入改为 ndarray 字段用
    :func:`np.array_equal`（``equal_nan=True``），其余字段用 ``==``。

    .. note:: 子类须以 ``@dataclass(eq=False)`` 声明，否则 dataclass
       自动生成的 ``__eq__`` 会覆盖本混入。
    """

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        for f in fields(self):
            a = getattr(self, f.name)
            b = getattr(other, f.name)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if not np.array_equal(a, b, equal_nan=True):
                    return False
            elif a != b:
                return False
        return True


# ============================================================
# 几何/运动学数据类
# ============================================================
@dataclass(eq=False)
class Wrench(_ArrayEqMixin):
    """六维力/力矩。

    :ivar force: ``(3,)`` 力向量，单位 N。
    :ivar torque: ``(3,)`` 力矩向量，单位 N·m。
    """

    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass(eq=False)
class Twist(_ArrayEqMixin):
    """空间速度

    :ivar linear: ``(3,)`` 线速度，单位 m/s。
    :ivar angular: ``(3,)`` 角速度，单位 rad/s。
    """

    linear: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass(eq=False)
class Pose(_ArrayEqMixin):
    """统一位姿表示

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
        return Rp_to_T(quat_to_R(self.orientation), self.position)


# ============================================================
# 状态快照
# ============================================================
@dataclass(eq=False)
class JointState(_ArrayEqMixin):
    """关节层状态快照

    :ivar control_mode: 关节当前控制模式（见 :class:`ControlMode`）。
    :ivar q: ``(n,)`` 关节位置，弧度。
    :ivar dq: ``(n,)`` 关节速度，弧度/秒。
    :ivar ddq: ``(n,)`` 关节加速度，弧度/秒²。
    :ivar tau: ``(n,)`` 关节力矩，N·m。
    :ivar enabled: ``(n,)`` bool 使能状态，``True``=使能，``False``=失能。
    :ivar error: ``(n,)`` bool 异常状态，``True``=电机异常（过温/过流等），``False``=正常。
    :ivar comm_ok: ``(n,)`` bool 通讯状态，``True``=正常，``False``=异常。
    :ivar angle_ok: ``(n,)`` bool 角度状态（编码器角度有效性），``True``=正常，``False``=异常。
    :ivar temp_motor: ``(n,)`` 电机温度，单位 ℃。
    :ivar temp_driver: ``(n,)`` 驱动器温度，单位 ℃。
    :ivar voltage: 总线电压，单位 V。
    :ivar current: 总线电流，单位 A。
    """

    control_mode: ControlMode = ControlMode.POSITION
    q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ddq: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(0))
    enabled: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    error: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    comm_ok: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    angle_ok: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    temp_motor: np.ndarray = field(default_factory=lambda: np.zeros(0))
    temp_driver: np.ndarray = field(default_factory=lambda: np.zeros(0))
    voltage: float = 0.0
    current: float = 0.0


@dataclass(eq=False)
class TcpState(_ArrayEqMixin):
    """末端层状态快照"""

    pose: Pose = field(default_factory=Pose)
    twist: Twist = field(default_factory=Twist)
    wrench: Wrench = field(default_factory=Wrench)


@dataclass(eq=False)
class ArmState(_ArrayEqMixin):
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
    errors: list[str] = field(default_factory=list)


# ============================================================
# 限位声明
# ============================================================
@dataclass(eq=False)
class JointLimits(_ArrayEqMixin):
    """关节限位声明

    所有数组维度均为 ``(n,)``，``n`` 为机械臂自由度数。

    - **硬限位**实例（``Arm.joint_limits``）：URDF/电机物理极限；
      ``q_min/q_max`` 为位置硬限位，供 :func:`safety.joint.joint_limits_check`
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
    :ivar temp_motor_max: ``(n,)`` 电机温度上限，℃。
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
    temp_motor_max: np.ndarray = field(default_factory=lambda: np.zeros(0))
    temp_driver_max: np.ndarray = field(default_factory=lambda: np.zeros(0))
    voltage_min: np.ndarray = field(default_factory=lambda: np.zeros(0))
    voltage_max: np.ndarray = field(default_factory=lambda: np.zeros(0))
    current_max: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass(eq=False)
class TcpLimits(_ArrayEqMixin):
    """末端限位声明

    :ivar workspace_box: ``(3,2)`` 末端工作空间包围盒，每行 ``[min, max]`` 对应 x/y/z，单位 m。
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


@dataclass(eq=False)
class Violation(_ArrayEqMixin):
    """安全监控器输出的违规记录

    :ivar layer: 监控层标识，``"joint"``/``"tcp"``/``"arm"`
    :ivar joint_idx: 关节索引（末端/整机层违规可置 ``-1``）。
    :ivar metric: 违规指标名（如 ``"q"``/``"dq"``/``"temp_motor"``）。
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
@dataclass(eq=False)
class IKResult(_ArrayEqMixin):
    """逆运动学求解结果

    :ivar q: ``(n,)`` 解算得到的关节角，弧度。
    :ivar success: 是否收敛到满足精度的解。
    :ivar err: 末端位姿误差（残差范数）。
    :ivar n_iter: 迭代次数。
    """

    q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    success: bool = False
    err: float = float("inf")
    n_iter: int = 0


@dataclass(eq=False)
class ComplianceParams(_ArrayEqMixin):
    """笛卡尔柔顺参数

    阻抗控制器 ``K_d/D_d/M_d`` 与导纳控制器 ``K_a/D_a/M_a``
    均映射到本类的 ``K/D/M`` 三个 6×6 笛卡尔矩阵（默认对角）。

    :ivar K: ``(6,6)`` 刚度矩阵。
    :ivar D: ``(6,6)`` 阻尼矩阵。
    :ivar M: ``(6,6)`` 惯量矩阵。
    """

    K: np.ndarray = field(default_factory=lambda: np.eye(6))
    D: np.ndarray = field(default_factory=lambda: np.eye(6))
    M: np.ndarray = field(default_factory=lambda: np.eye(6))
