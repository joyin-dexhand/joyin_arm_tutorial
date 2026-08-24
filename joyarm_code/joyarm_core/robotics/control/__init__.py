"""控制域（策略族 + 执行器 + 三类控制）。

Controller(ABC) · PositionController(``"position"``，默认，Ch6 占位)；
ControlLoop 为控制**执行器**（tick 循环驱动控制律）——不是策略；三类控制函数：
kinematic(Part1，Ch6) / dynamics_based(Part2，Ch8) / force(Part3，Ch9)。
``REGISTRY`` 供 config ``solvers.control`` 选型；运行期经 ``arm.set_controller`` 切换。
"""
from ...utils.types import ControlMode
from .controller import Controller
from .position_controller import PositionController
from .kinematic import (
    joint_position_control,
    joint_velocity_control,
    torque_control,
    arm_position_control,
    arm_velocity_control,
    ControlLoop,
    play_joint_trajectory,
    play_cart_trajectory,
)
from .dynamics_based import computed_torque_control, inverse_dynamics_control
from .force import (
    ForceTorqueSensor,
    JointTorqueSensor,
    pure_force_control,
    HybridForcePosition,
    ImpedanceControl,
    AdmittanceControl,
    compute_cartesian_impedance,
)

REGISTRY = {
    "position": PositionController,
}

__all__ = [
    "Controller", "PositionController", "REGISTRY",
    # Part 1：运动学控制（Ch6）
    "joint_position_control", "joint_velocity_control", "torque_control",
    "arm_position_control", "arm_velocity_control", "ControlLoop",
    "play_joint_trajectory", "play_cart_trajectory", "play_joint", "play_cart",
    # Part 2：动力学控制（Ch8）
    "computed_torque_control", "inverse_dynamics_control",
    # Part 3：力控制（Ch9）
    "ForceTorqueSensor", "JointTorqueSensor", "pure_force_control",
    "HybridForcePosition", "ImpedanceControl", "AdmittanceControl",
    "compute_cartesian_impedance",
]


def play_joint(arm, traj, mode=ControlMode.POSITION, hz: int = 200):
    """函数式入口（教学用）：委托 ``arm.play_joint``（按时间序列回放关节轨迹）。"""
    return arm.play_joint(traj, mode=mode, hz=hz)


def play_cart(arm, traj, hz: int = 200, **kw):
    """函数式入口（教学用）：委托 ``arm.play_cart``（OSC 回放笛卡尔轨迹）。"""
    return arm.play_cart(traj, hz=hz, **kw)
