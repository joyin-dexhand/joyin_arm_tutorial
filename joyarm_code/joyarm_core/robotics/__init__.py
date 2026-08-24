"""``joyarm_core.robotics`` —— 算法层子包（一域一子包，策略族）。

仅依赖 utils 与 numpy（鸭子类型消费 ``arm``，不 import joyarms）。每域 = 策略 ABC +
实现们（一节点一文件）+ ``REGISTRY`` 注册表 + 函数式教学入口；JoyArm 构造时按
config ``solvers:`` 段查表组装私有成员（``_fkine_solver`` 等），算法互调走公开门面：

- fkine：正运动学（Ch2，pin ✅ / mdh 🟡）
- ikine：逆运动学（Ch3，pin / analytic6r 🟡）
- jacobian：速度运动学与静力学（Ch4，pin / geometric 🟡）
- trajectory：轨迹规划（Ch5，default / toppra 🟡）
- dynamics：逆动力学与 M/C/G（Ch8，pin / lagrangian 🟡）
- control：控制律与执行器（Ch6/8/9，position 默认 🟡）
"""
from .fkine import fkine
from .ikine import ikine, ikine_constrained
from .jacobian import jac, manipulability, cond_number, statics
from .trajectory import (
    Trajectory,
    joint_cubic,
    joint_quintic,
    joint_lspb,
    joint_waypoints,
    cart_line,
    cart_arc,
    cart_waypoints,
    cart_to_joint,
    constant_velocity_retime,
    validate,
)
from .dynamics import idyn, mass_matrix, coriolis, gravity, cartesian_inertia
from .control import (
    ControlLoop,
    play_joint,
    play_cart,
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

__all__ = [
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
    "cart_waypoints",
    "cart_to_joint",
    "constant_velocity_retime",
    "validate",
    "idyn",
    "mass_matrix",
    "coriolis",
    "gravity",
    "cartesian_inertia",
    "ControlLoop",
    "play_joint",
    "play_cart",
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
]
