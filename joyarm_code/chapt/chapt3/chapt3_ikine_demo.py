"""
==============================================================================
JoyArm 位置逆运动学三维演示  (第三章 §3)
==============================================================================
【功能概要】
    离线演示同一目标位姿的 8 组解析逆解（无需连接硬件）：
      1. 用限位内的参考关节角构造可达目标位姿；
      2. 枚举 8 组解析候选解，逐组进行正运动学回代；
      3. 在 MeshCat 中自动播放全部构型，并同时显示目标 TCP 与当前 TCP；
      4. 机械臂用绿色表示限位内解，用红色表示超出硬限位的几何解。

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code && uv sync

    # 2. 运行本脚本
    uv run python chapt/chapt3/chapt3_ikine_demo.py

【命令行参数】
    --speed 0.5   以 0.5 倍速度播放
    --headless    只求解并打印结果，不启动 MeshCat
    --once        播放完成后直接退出，不等待输入
==============================================================================
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import meshcat.geometry as meshcat_geometry
import numpy as np

# 允许从 joyarm_code/ 目录直接运行本文件。
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from joyarm_core import joyArm
from chapt.common.exercise_metrics import (
    frame_end_name,
    pose_error,
    within_joint_limits,
)
from chapt.common.model_visualization import (
    MODEL_NAME,
    create_viewer,
    set_arm_color,
)

REFERENCE_DEG = np.array([-20, -46, -50, -50, -25, 30])
#REFERENCE_DEG = np.array([118,-76,-160,-65,17,46,])
WITHIN_LIMIT_COLOR = np.array([0.20, 0.80, 0.25, 1.0])
OUTSIDE_LIMIT_COLOR = np.array([0.90, 0.15, 0.12, 1.0])


@dataclass
class DemoSolution:
    """一组待显示的逆解及其回代检查结果。"""

    method: str
    source: str
    start_q: np.ndarray
    solution_q: np.ndarray | None
    success: bool
    within_limits: bool
    position_error_mm: float | None
    orientation_error_deg: float | None


def parse_args() -> argparse.Namespace:
    """读取播放速度与无界面运行选项。"""
    parser = argparse.ArgumentParser(description="JoyArm 位置逆运动学三维演示")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率")
    parser.add_argument("--headless", action="store_true", help="只计算，不启动 MeshCat")
    parser.add_argument("--once", action="store_true", help="播放完成后直接退出")
    args = parser.parse_args()
    if args.speed <= 0.0:
        parser.error("--speed 必须大于 0")
    return args


def solve_analytic(arm, target, frame: str, playback_start: np.ndarray) -> list[DemoSolution]:
    """枚举解析候选解并逐组回代检查。"""
    arm.set_ikine_solver("analytic_ikine_solver")
    result = arm.solve_ikine_all(target, frame)
    if not result.success:
        return []

    solutions: list[DemoSolution] = []
    previous_q = playback_start.copy()
    for index, solution_q in enumerate(np.atleast_2d(result.q), start=1):
        checked = arm.solve_fkine(solution_q, frame)
        position_error, orientation_error = pose_error(target, checked)
        is_within_limits = within_joint_limits(
            solution_q,
            arm.joint_limits_hard_arm,
            tolerance=1e-9,
        )
        solutions.append(DemoSolution(
            method="解析解",
            source=f"候选 {index}",
            start_q=previous_q,
            solution_q=np.asarray(solution_q, dtype=float),
            success=True,
            within_limits=is_within_limits,
            position_error_mm=position_error,
            orientation_error_deg=orientation_error,
        ))
        previous_q = np.asarray(solution_q, dtype=float)
    return solutions


def compact_angles(joint_angles: np.ndarray | None) -> str:
    """将六关节角压缩为适合表格显示的角度字符串。"""
    if joint_angles is None:
        return "—"
    values = np.rad2deg(joint_angles)
    return "[" + ", ".join(f"{value:.1f}" for value in values) + "]"


def print_results(solutions: list[DemoSolution]) -> None:
    """以紧凑表格打印全部求解与回代结果。"""
    header = (
        f"{'编号':^4}  {'方法':^6}  {'来源':^8}  {'限位内':^6}  "
        f"{'位置误差/mm':>11}  {'姿态误差/°':>11}  关节角/°"
    )
    print("\n" + header)
    print("-" * 108)
    for index, solution in enumerate(solutions, start=1):
        limit_text = "是" if solution.within_limits else "否"
        position_text = (
            "—" if solution.position_error_mm is None
            else f"{solution.position_error_mm:.6f}"
        )
        orientation_text = (
            "—" if solution.orientation_error_deg is None
            else f"{solution.orientation_error_deg:.6f}"
        )
        print(
            f"{index:^4}  {solution.method:^6}  {solution.source:^8}  "
            f"{limit_text:^6}  {position_text:>11}  {orientation_text:>11}  "
            f"{compact_angles(solution.solution_q)}"
        )


def setup_visualizer(arm, target):
    """启动 MeshCat，加载模型并创建目标/当前 TCP 标记。"""
    visualizer = create_viewer(MODEL_NAME)

    viewer = visualizer.viewer
    viewer["ik_demo/target_tcp/frame"].set_object(meshcat_geometry.triad(0.10))
    viewer["ik_demo/target_tcp/frame"].set_transform(target.T)
    viewer["ik_demo/target_tcp/point"].set_object(
        meshcat_geometry.Sphere(0.012),
        meshcat_geometry.MeshLambertMaterial(
            color=0xFFD400,
            opacity=0.65,
            transparent=True,
        ),
    )
    viewer["ik_demo/target_tcp/point"].set_transform(target.T)
    viewer["ik_demo/current_tcp/frame"].set_object(meshcat_geometry.triad(0.065))
    viewer["ik_demo/current_tcp/point"].set_object(
        meshcat_geometry.Sphere(0.008),
        meshcat_geometry.MeshLambertMaterial(color=0xFFFFFF),
    )
    update_visualizer(visualizer, arm, np.deg2rad(REFERENCE_DEG))
    return visualizer


def update_visualizer(visualizer, arm, joint_angles: np.ndarray) -> None:
    """更新机械臂构型和当前 TCP 坐标系。"""
    visualizer.display(joint_angles)
    current_pose = arm.solve_fkine(joint_angles, frame_end_name(arm))
    visualizer.viewer["ik_demo/current_tcp/frame"].set_transform(current_pose.T)
    visualizer.viewer["ik_demo/current_tcp/point"].set_transform(current_pose.T)


def set_robot_color(visualizer, within_limit: bool) -> None:
    """按候选解的限位状态重新设置全部机械臂网格颜色。"""
    color = WITHIN_LIMIT_COLOR if within_limit else OUTSIDE_LIMIT_COLOR
    set_arm_color(visualizer, color)


def smooth_step(ratio: float) -> float:
    """返回端点速度为零的三次插值比例。"""
    clipped = min(1.0, max(0.0, ratio))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def pause(seconds: float, speed: float) -> None:
    """按播放倍率停留指定演示时长。"""
    time.sleep(seconds / speed)


def animate_solution(
    visualizer,
    arm,
    solution: DemoSolution,
    speed: float,
) -> None:
    """从初值平滑播放到一组逆解。"""
    if not solution.success or solution.solution_q is None:
        return

    limit_note = "限位内" if solution.within_limits else "超出硬限位，仅演示几何解"
    print(f"播放：{solution.source}（{limit_note}）")
    set_robot_color(visualizer, solution.within_limits)
    update_visualizer(visualizer, arm, solution.start_q)
    pause(1.0, speed)

    duration = 3.0 / speed
    frame_rate = 50.0
    frame_count = max(2, int(duration * frame_rate))
    for frame_index in range(frame_count + 1):
        ratio = smooth_step(frame_index / frame_count)
        joint_angles = (
            solution.start_q
            + ratio * (solution.solution_q - solution.start_q)
        )
        update_visualizer(visualizer, arm, joint_angles)
        time.sleep(1.0 / frame_rate)
    pause(2.0, speed)


def main() -> None:
    """求解、打印并按需播放 8 组解析解。"""
    args = parse_args()
    arm = joyArm(MODEL_NAME)
    if arm is None:
        raise SystemExit("JoyArm 模型创建失败")

    frame = frame_end_name(arm)
    reference_q = np.deg2rad(REFERENCE_DEG)
    target = arm.solve_fkine(reference_q, frame)
    print(f"目标帧：{frame}")
    print(f"目标位置（mm）：{np.round(target.position * 1000.0, 3)}")
    print(f"目标姿态四元数（wxyz）：{np.round(target.orientation, 6)}")

    analytic_solutions = solve_analytic(arm, target, frame, reference_q)
    print_results(analytic_solutions)

    if args.headless:
        return

    visualizer = setup_visualizer(arm, target)
    for solution in analytic_solutions:
        animate_solution(visualizer, arm, solution, args.speed)

    print(
        "演示播放完成。绿色机械臂表示限位内解，红色机械臂表示超限几何解；"
        "目标 TCP 为大坐标轴和黄色点，当前 TCP 为小坐标轴和白色点。"
    )
    if not args.once:
        try:
            input("按 Enter 退出，或在终端按 Ctrl+C。")
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
