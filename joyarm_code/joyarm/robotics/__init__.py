"""``joyarm.robotics`` —— 算法层子包。

收口全部机器人算法实现（仅依赖 :mod:`joyarm.utils` 基础层与 numpy）：

- :mod:`joyarm.robotics.fkine`：正运动学（Ch2，已实现）。
- :mod:`joyarm.robotics.ikine`：逆运动学（Ch3）。
- :mod:`joyarm.robotics.jacobian`：速度运动学与静力学（Ch4）。
- :mod:`joyarm.robotics.trajectory`：轨迹生成（Ch5）。
- :mod:`joyarm.robotics.dynamics`：正/逆动力学（Ch8）。
- :mod:`joyarm.robotics.control`：运动学/动力学/力控（Ch6/Ch8/Ch9）。
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
    cart_to_joint,
    constant_velocity_retime,
    validate,
)
from .dynamics import fdyn, idyn, mass_matrix, coriolis, gravity, cartesian_inertia
from .control import (
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

__all__ = [
    # fkine / ikine / jacobian
    "fkine",
    "ikine",
    "ikine_constrained",
    "jac",
    "manipulability",
    "cond_number",
    "statics",
    # trajectory
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
    # dynamics
    "fdyn",
    "idyn",
    "mass_matrix",
    "coriolis",
    "gravity",
    "cartesian_inertia",
    # control
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
]
