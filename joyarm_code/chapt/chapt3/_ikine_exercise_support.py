"""第三章 Pinocchio 逆运动学实践的固定实验框架。"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from joyarm_core import PinIkineSolver, Pose, joyArm
from chapt.common.exercise_metrics import (
    frame_end_name,
    pose_error,
    within_joint_limits,
)
from chapt.common.model_visualization import (
    MODEL_NAME,
    add_pose_marker,
    create_viewer,
    set_arm_color,
)

START_Q = np.deg2rad([-30.0, -40.0, -40.0, -50.0, -20.0, 30.0])
TARGETS = (
    Pose(
        position=np.array([0.206600738, -0.044959354, 0.574167726]),
        orientation=np.array([0.900844828, 0.050466402, -0.423137965, 0.082981931]),
    ),
    Pose(
        position=np.array([0.237403736, -0.047753705, 0.574983933]),
        orientation=np.array([0.914503803, 0.060942316, -0.394184680, 0.067729355]),
    ),
    Pose(
        position=np.array([0.282852548, -0.047243452, 0.562353637]),
        orientation=np.array([0.936435230, 0.074557575, -0.338974806, 0.051247518]),
    ),
)


@dataclass
class CheckResult:
    """一组逆解的正运动学回代结果。"""

    position_error_mm: float
    orientation_error_deg: float
    in_limits: bool


def _solve_path(solver, arm, targets, start_q):
    """以上一目标的解为下一目标初值，连续求解全部目标。"""
    path = []
    q0 = np.asarray(start_q, dtype=float).copy()
    for index, target in enumerate(targets, start=1):
        result = solver.solve(arm, target, frame_end_name(arm), q0)
        if not result.success:
            raise RuntimeError(
                f"第 {index} 个目标求解失败：残差={result.err:.6g}，"
                f"迭代次数={result.n_iter}"
            )
        q0 = np.asarray(result.q, dtype=float).copy()
        path.append(q0)
    return np.asarray(path, dtype=float)


def _check_solution(arm, target, q):
    """使用正运动学回代验证一组逆解。"""
    checked = arm.solve_fkine(q, frame_end_name(arm))
    position_error_mm, orientation_error_deg = pose_error(target, checked)
    in_limits = within_joint_limits(q, arm.joint_limits_hard_arm)
    return CheckResult(position_error_mm, orientation_error_deg, in_limits)


def _print_results(title, arm, targets, path):
    """输出关节角及正运动学回代结果。"""
    print(f"\n{title}")
    print(f"{'目标':^6} {'限位内':^8} {'位置误差/mm':^16} {'姿态误差/°':^16} 关节角/°")
    print("-" * 102)
    for index, (target, q) in enumerate(zip(targets, path), start=1):
        result = _check_solution(arm, target, q)
        limit_text = "是" if result.in_limits else "否"
        angles = ", ".join(f"{value:.1f}" for value in np.rad2deg(q))
        print(
            f"{index:^6} {limit_text:^8} {result.position_error_mm:^16.6f} "
            f"{result.orientation_error_deg:^16.6f} [{angles}]"
        )


def _setup_viewer(targets, start_q):
    viewer = create_viewer(MODEL_NAME)
    colors = [0xFFD400, 0xFF9F1C, 0xFF5A36]
    for index, (target, color) in enumerate(zip(targets, colors), start=1):
        add_pose_marker(viewer, f"exercise/target_{index}", target, color)
    viewer.display(start_q)
    return viewer


def _play_path(viewer, start_q, path, title, color):
    print(f"\n播放：{title}")
    set_arm_color(viewer, color)
    last_q = start_q.copy()
    for target_index, target_q in enumerate(path, start=1):
        print(f"  移动到目标 {target_index}")
        for frame_index in range(101):
            value = frame_index / 100.0
            ratio = value * value * (3.0 - 2.0 * value)
            viewer.display(last_q + ratio * (target_q - last_q))
            time.sleep(0.02)
        time.sleep(0.5)
        last_q = target_q


def _show_paths(targets, start_q, practice_path, reference_path):
    viewer = _setup_viewer(targets, start_q)
    _play_path(
        viewer,
        start_q,
        practice_path,
        "实践求解器路径",
        [0.20, 0.55, 1.00, 1.0],
    )
    _play_path(
        viewer,
        start_q,
        reference_path,
        "库内参考求解器路径",
        [0.20, 0.80, 0.25, 1.0],
    )
    try:
        input("\n播放完成，按 Enter 退出，或在终端按 Ctrl+C。")
    except (EOFError, KeyboardInterrupt):
        pass


def run_exercise(practice_solver):
    """运行固定实验；传入实践中实现的求解器实例。"""
    parser = argparse.ArgumentParser(description="第三章 JoyArm Pinocchio 逆解实践")
    parser.add_argument("--headless", action="store_true", help="不启动 MeshCat")
    args = parser.parse_args()

    arm = joyArm(MODEL_NAME)
    try:
        practice_path = _solve_path(practice_solver, arm, TARGETS, START_Q)
    except NotImplementedError as exc:
        raise SystemExit(
            "第三章实践尚未完成：请补全 "
            "PracticePinIkineSolver.solve() 的实践实现区。"
        ) from exc
    reference_path = _solve_path(PinIkineSolver(), arm, TARGETS, START_Q)
    _print_results("实践求解器", arm, TARGETS, practice_path)
    _print_results("库内参考求解器", arm, TARGETS, reference_path)

    if not args.headless:
        _show_paths(TARGETS, START_Q, practice_path, reference_path)
