"""第四章交互演示：雅可比速度映射、奇异性与静力学。

运行：
    cd ~/joyin_arm_tutorial/joyarm_code
    uv sync && source .venv/bin/activate
    python chapt/chapt4/chapt4_jacobian_demo.py

默认打开 GUI；自动验证可用 ``--headless --case all``。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from chapt.common.exercise_metrics import frame_end_name, jacobian_metrics
from chapt.common.model_visualization import MODEL_NAME
from joyarm_core import joyArm

REGULAR_Q = np.deg2rad([-25.0, -50.0, -55.0, -35.0, -30.0, 25.0])
SINGULAR_Q = np.deg2rad([0.0, -35.0, -70.0, 0.0, 0.0, 0.0])
JOINT_VELOCITY = np.deg2rad([8.0, -5.0, 6.0, 4.0, -3.0, 7.0])
EXTERNAL_WRENCH = np.array([0.0, 0.0, -19.62, 0.0, 0.0, 0.0])


def velocity_values(arm, q=REGULAR_Q, dq=JOINT_VELOCITY):
    """返回雅可比、各关节速度贡献和合成末端速度。"""
    jacobian = arm.get_jac(q, frame_end_name(arm))
    contributions = (jacobian * np.asarray(dq)[None, :]).T
    return jacobian, contributions, contributions.sum(axis=0)


def singularity_values(arm, q=SINGULAR_Q):
    """返回奇异指标及同一目标速度下两种逆解结果。"""
    jacobian = arm.get_jac(q, frame_end_name(arm))
    left_vectors, _, _ = np.linalg.svd(jacobian)
    target_twist = 0.03 * left_vectors[:, -1]
    dq_pinv = np.linalg.pinv(jacobian) @ target_twist
    dq_damped = arm.get_jac_pinv_damped(q, frame_end_name(arm), damping=0.03) @ target_twist
    sigma_min, condition = jacobian_metrics(jacobian)
    return sigma_min, condition, target_twist, dq_pinv, dq_damped


def statics_values(arm, q=REGULAR_Q, wrench=EXTERNAL_WRENCH):
    """返回外载荷、电机平衡力矩和硬限位占用率。"""
    jacobian = arm.get_jac(q, frame_end_name(arm))
    tau_external = jacobian.T @ np.asarray(wrench)
    tau_motor = -tau_external
    utilization = np.abs(tau_motor) / arm.joint_limits_hard_arm.tau_max
    return tau_external, tau_motor, utilization


def print_headless(arm, case: str) -> None:
    """打印精简数值，供无图形环境验证。"""
    if case in ("all", "velocity"):
        _, contributions, twist = velocity_values(arm)
        print("\n第四章 demo｜雅可比")
        print("  各关节贡献 [vx, vy, vz, wx, wy, wz]：")
        print(np.round(contributions, 6))
        print("  合成末端速度：", np.round(twist, 6).tolist())
    if case in ("all", "singularity"):
        sigma, condition, _, dq_pinv, dq_damped = singularity_values(arm)
        print("\n第四章 demo｜奇异性")
        print(f"  最小奇异值={sigma:.6g}，条件数={condition:.6g}")
        print(f"  ||dq伪逆||={np.linalg.norm(dq_pinv):.6g}，||dq阻尼||={np.linalg.norm(dq_damped):.6g}")
    if case in ("all", "statics"):
        tau_external, tau_motor, utilization = statics_values(arm)
        print("\n第四章 demo｜静力学")
        print("  外载荷产生的关节力矩：", np.round(tau_external, 4).tolist())
        print("  保持当前姿态所需电机力矩：", np.round(tau_motor, 4).tolist())
        print("  工程检查——最大力矩占用率/%：", np.round(100 * utilization, 2).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="第四章 JoyArm 雅可比交互演示")
    parser.add_argument("--headless", action="store_true", help="不打开 GUI，仅打印数值")
    parser.add_argument("--case", choices=("all", "velocity", "singularity", "statics"), default="all", help="无界面验证案例")
    args = parser.parse_args()
    arm = joyArm(MODEL_NAME)
    if args.headless:
        print_headless(arm, args.case)
        return
    from chapt.chapt4._jacobian_demo_gui import run_gui
    run_gui(arm, REGULAR_Q, JOINT_VELOCITY, EXTERNAL_WRENCH)


if __name__ == "__main__":
    main()
