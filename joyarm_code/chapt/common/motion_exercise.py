"""
第三至第六章公共实践流程：目标位姿 → 逆解 → 雅可比 → 轨迹 → 控制

功能概要：
    使用同一个 JoyArm 离线任务串联第三至第六章的四个算法模块。公共流程负责
    准备模型与目标、连接模块、检查结果、模拟控制执行并输出报告；各章只替换
    对应的算法子类。

    默认全部使用 joyarm_core 官方实现。文件中的“章节模块替换区”为每一章
    保留了官方实现和读者实现两种选择：启用其中一行，并注释另一行。选用
    读者实现时，流程还会启用该章专属的日志或 MeshCat 可视化反馈。

环境与运行：
    cd ~/joyin_arm_tutorial/joyarm_code
    uv sync
    source .venv/bin/activate
    python chapt/common/motion_exercise.py

    显示关节轨迹与控制跟踪曲线：
    python chapt/common/motion_exercise.py --plot

    使用 MeshCat 播放离线控制得到的实际关节轨迹：
    python chapt/common/motion_exercise.py --meshcat

说明：
    本流程默认离线运行，不连接机械臂，也不向真实电机下发指令。
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 允许从 joyarm_code/ 目录直接运行本文件，并导入各章实践模块。
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from joyarm_core import (  # noqa: E402
    JointPositionController,
    JointState,
    PinIkineSolver,
    PinJacobianSolver,
    Pose,
    SingleJointCubicTrajPlanner,
    TrajFrame,
    joyArm,
)
from chapt.common.exercise_metrics import frame_end_name, jacobian_metrics, pose_error
from chapt.common.model_visualization import (
    MODEL_NAME,
    add_path,
    add_pose_marker,
    create_viewer,
)


# =============================================================================
# 章节模块替换区
#
# 每一组只启用一种实现：
#   1. 使用官方实现时，保留“官方”行，注释“读者实现”行；
#   2. 完成本章接口后，注释“官方”行，取消对应“读者实现”两行的注释。
# =============================================================================

# 第三章：逆运动学 -----------------------------------------------------------
IKINE_SOLVER, IKINE_IS_PRACTICE = PinIkineSolver(), False  # 官方实现
# from chapt.chapt3.chapt3_ikine_exercise import PracticePinIkineSolver
# IKINE_SOLVER, IKINE_IS_PRACTICE = PracticePinIkineSolver(), True  # 读者实现

# 第四章：雅可比 -------------------------------------------------------------
JACOBIAN_SOLVER, JACOBIAN_IS_PRACTICE = PinJacobianSolver(), False  # 官方实现
# from chapt.chapt4.chapt4_jacobian_exercise import PracticeJacobianSolver
# JACOBIAN_SOLVER, JACOBIAN_IS_PRACTICE = PracticeJacobianSolver(), True

# 第五章：轨迹规划 -----------------------------------------------------------
TRAJ_PLANNER, TRAJ_IS_PRACTICE = SingleJointCubicTrajPlanner(), False  # 官方实现
# from chapt.chapt5.chapt5_trajectory_exercise import PracticeTrajPlanner
# TRAJ_PLANNER, TRAJ_IS_PRACTICE = PracticeTrajPlanner(), True

# 第六章：运动控制 -----------------------------------------------------------
CONTROLLER, CONTROL_IS_PRACTICE = JointPositionController(), False  # 官方实现
# from chapt.chapt6.chapt6_control_exercise import PracticeController
# CONTROLLER, CONTROL_IS_PRACTICE = PracticeController(), True


@dataclass(frozen=True)
class MotionTask:
    """公共流程的固定输入。"""

    q_start: np.ndarray
    target_pose: Pose
    duration: float = 4.0
    sample_hz: float = 100.0
    settle_time: float = 0.5


@dataclass(frozen=True)
class IkineStageResult:
    """第三章阶段的求解结果与 FK 回代数据。"""

    q_goal: np.ndarray
    checked_pose: Pose
    residual: float
    iterations: int
    position_error_mm: float
    orientation_error_deg: float


@dataclass(frozen=True)
class MotionResult:
    """四个阶段的主要结果，供输出和绘图使用。"""

    q_goal: np.ndarray
    target_pose: Pose
    ik_checked_pose: Pose
    sample_hz: float
    ik_residual: float
    ik_iterations: int
    ik_position_error_mm: float
    ik_orientation_error_deg: float
    sigma_min_goal: float
    condition_goal: float
    sample_times: np.ndarray
    q_reference: np.ndarray
    dq_reference: np.ndarray
    sigma_min_path: np.ndarray
    q_actual: np.ndarray
    tcp_reference: np.ndarray
    tcp_actual: np.ndarray
    tracking_error_deg: np.ndarray
    final_tracking_error_deg: float


class OfflineExerciseArm:
    """只为核心规划器和控制器提供最小离线鸭子接口。"""

    def __init__(self, model_arm, q_start: np.ndarray):
        self._model_arm = model_arm
        self.home_arm = np.asarray(model_arm.home_arm, dtype=float).copy()
        self.joint_limits_hard_arm = model_arm.joint_limits_hard_arm
        self._target_q: np.ndarray | None = None
        self._state = JointState(
            q=np.asarray(q_start, dtype=float).copy(),
            dq=np.zeros_like(q_start, dtype=float),
            tau=np.zeros_like(q_start, dtype=float),
        )

    def __getattr__(self, name):
        """运动学与模型能力直接复用离线 JoyArm。"""
        return getattr(self._model_arm, name)

    def reset_state(self, q: np.ndarray) -> None:
        """把离线机械臂恢复到指定关节状态。"""
        q = np.asarray(q, dtype=float).copy()
        self._state = JointState(
            q=q,
            dq=np.zeros_like(q),
            tau=np.zeros_like(q),
        )
        self._target_q = None

    def get_state_arm(self) -> JointState:
        """返回当前离线关节状态。"""
        return self._state

    def send_position_arm(self, q, vlim=None, flim=None) -> None:
        """接收核心位置控制器下发的本拍目标。"""
        del vlim, flim
        self._target_q = np.asarray(q, dtype=float).reshape(-1).copy()

    def advance(self, dt: float) -> None:
        """用最小一阶对象响应位置指令，不模拟 Backend。"""
        if self._target_q is None:
            raise RuntimeError("控制器本拍未下发位置指令")
        q = self._state.q.copy()
        dq_limit = np.asarray(self.joint_limits_hard_arm.dq_max, dtype=float)
        dq = np.clip((self._target_q - q) / 0.08, -dq_limit, dq_limit)
        q_next = np.clip(
            q + dq * dt,
            self.joint_limits_hard_arm.q_min,
            self.joint_limits_hard_arm.q_max,
        )
        self._state = JointState(
            t=self._state.t + dt,
            q=q_next,
            dq=dq,
            tau=np.zeros_like(q_next),
        )


def default_task() -> MotionTask:
    """返回四章共用的 JoyArm 末端位姿任务。"""
    return MotionTask(
        q_start=np.deg2rad([-30.0, -40.0, -40.0, -50.0, -20.0, 30.0]),
        target_pose=Pose(
            position=np.array([0.282852548, -0.047243452, 0.562353637]),
            orientation=np.array(
                [0.936435230, 0.074557575, -0.338974806, 0.051247518]
            ),
        ),
    )


def solve_inverse_kinematics(arm, task: MotionTask) -> IkineStageResult:
    """第三章阶段：目标末端位姿转换为目标关节角，并做 FK 回代。"""

    # 逆运动学接口调用
    result = IKINE_SOLVER.solve(
        arm,
        task.target_pose,
        frame_end_name(arm),
        task.q_start,
    )
    if not result.success or result.q is None:
        raise RuntimeError(f"逆运动学失败：残差={result.err:.6g}")

    q_goal = np.asarray(result.q, dtype=float).reshape(-1)
    checked = arm.solve_fkine(q_goal, frame_end_name(arm))
    position_error_mm, orientation_error_deg = pose_error(
        task.target_pose,
        checked,
    )
    return IkineStageResult(
        q_goal=q_goal,
        checked_pose=checked,
        residual=float(result.err),
        iterations=int(result.n_iter),
        position_error_mm=position_error_mm,
        orientation_error_deg=orientation_error_deg,
    )


def analyze_jacobian(arm, q: np.ndarray) -> tuple[float, float]:
    """第四章阶段：计算最小奇异值和条件数。"""

    #雅可比接口调用
    jacobian = np.asarray(
        JACOBIAN_SOLVER.jac(arm, q, frame_end_name(arm)),
        dtype=float,
    )
    expected_shape = (6, q.size)
    if jacobian.shape != expected_shape:
        raise ValueError(
            f"雅可比形状错误：得到 {jacobian.shape}，期望 {expected_shape}"
        )
    return jacobian_metrics(jacobian)


def plan_trajectory(
    arm: OfflineExerciseArm,
    task: MotionTask,
    q_goal: np.ndarray,
) -> tuple[np.ndarray, list[TrajFrame]]:
    """第五章阶段：规划并按固定频率采样关节轨迹。"""
    arm.reset_state(task.q_start)
    start_time = time.time()
    target_time = start_time + task.duration

    #轨迹接口调用
    target = TrajFrame(
        time=target_time,
        q=q_goal.copy(),
        dq=np.zeros_like(q_goal),
    )
    if not TRAJ_PLANNER.plan(arm, target):
        raise RuntimeError("轨迹规划器返回失败")

    sample_count = int(round(task.duration * task.sample_hz)) + 1
    sample_times = np.linspace(start_time, target_time, sample_count)
    frames = [TRAJ_PLANNER.sample(float(t)) for t in sample_times]
    _validate_trajectory(arm, frames, q_goal)
    return sample_times, frames


def _validate_trajectory(arm, frames: list[TrajFrame], q_goal: np.ndarray) -> None:
    """检查轨迹维度、有限性、关节限位和终点。"""
    q_reference = np.asarray([frame.q for frame in frames], dtype=float)
    dq_reference = np.asarray([frame.dq for frame in frames], dtype=float)
    if q_reference.shape[1] != q_goal.size:
        raise ValueError("轨迹关节维度与逆解维度不一致")
    if not np.all(np.isfinite(q_reference)) or not np.all(np.isfinite(dq_reference)):
        raise ValueError("轨迹包含 NaN 或无穷值")
    if np.any(q_reference < arm.joint_limits_hard_arm.q_min) or np.any(
        q_reference > arm.joint_limits_hard_arm.q_max
    ):
        raise ValueError("轨迹超出关节位置硬限位")
    if not np.allclose(q_reference[-1], q_goal, atol=1e-6):
        raise ValueError("轨迹终点与逆运动学目标不一致")


def scan_trajectory_jacobian(arm, frames: list[TrajFrame]) -> np.ndarray:
    """使用第四章模块扫描整条轨迹的最小奇异值。"""
    return np.asarray(
        [analyze_jacobian(arm, np.asarray(frame.q))[0] for frame in frames]
    )


def run_controller(
    arm: OfflineExerciseArm,
    task: MotionTask,
    frames: list[TrajFrame],
) -> tuple[np.ndarray, float]:
    """第六章阶段：逐拍调用控制器，并在离线对象中观察跟踪结果。"""
    arm.reset_state(task.q_start)
    dt = 1.0 / task.sample_hz
    actual = [arm.get_state_arm().q.copy()]

    for frame in frames[1:]:
        CONTROLLER.step(arm, frame, arm.get_state_arm())
        arm.advance(dt)
        actual.append(arm.get_state_arm().q.copy())

    # 轨迹结束后继续保持终点，观察控制误差是否收敛。
    settle_steps = int(round(task.settle_time * task.sample_hz))
    final_frame = frames[-1]
    for _ in range(settle_steps):
        CONTROLLER.step(arm, final_frame, arm.get_state_arm())
        arm.advance(dt)

    final_error_deg = float(
        np.rad2deg(np.linalg.norm(final_frame.q - arm.get_state_arm().q))
    )
    return np.asarray(actual), final_error_deg


def tcp_positions(arm, joint_path: np.ndarray) -> np.ndarray:
    """对一组关节角做 FK，返回对应的 TCP 三维位置。"""
    return np.asarray(
        [arm.solve_fkine(q, frame_end_name(arm)).position for q in joint_path],
        dtype=float,
    )


def run_motion_exercise(task: MotionTask | None = None) -> MotionResult:
    """执行第三至第六章公共流程并返回结构化结果。"""
    task = default_task() if task is None else task
    model_arm = joyArm(MODEL_NAME)
    arm = OfflineExerciseArm(model_arm, task.q_start)

    ikine = solve_inverse_kinematics(arm, task)
    sigma_min_goal, condition_goal = analyze_jacobian(arm, ikine.q_goal)
    sample_times, frames = plan_trajectory(arm, task, ikine.q_goal)
    sigma_min_path = scan_trajectory_jacobian(arm, frames)
    q_actual, final_tracking_error = run_controller(arm, task, frames)
    q_reference = np.asarray([frame.q for frame in frames])
    dq_reference = np.asarray([frame.dq for frame in frames])

    return MotionResult(
        q_goal=ikine.q_goal,
        target_pose=task.target_pose,
        ik_checked_pose=ikine.checked_pose,
        sample_hz=task.sample_hz,
        ik_residual=ikine.residual,
        ik_iterations=ikine.iterations,
        ik_position_error_mm=ikine.position_error_mm,
        ik_orientation_error_deg=ikine.orientation_error_deg,
        sigma_min_goal=sigma_min_goal,
        condition_goal=condition_goal,
        sample_times=sample_times,
        q_reference=q_reference,
        dq_reference=dq_reference,
        sigma_min_path=sigma_min_path,
        q_actual=q_actual,
        tcp_reference=tcp_positions(arm, q_reference),
        tcp_actual=tcp_positions(arm, q_actual),
        tracking_error_deg=np.rad2deg(
            np.linalg.norm(q_reference - q_actual, axis=1)
        ),
        final_tracking_error_deg=final_tracking_error,
    )


def print_report(result: MotionResult) -> None:
    """只打印已启用读者实现的章节结果。"""
    has_practice_module = any(
        (
            IKINE_IS_PRACTICE,
            JACOBIAN_IS_PRACTICE,
            TRAJ_IS_PRACTICE,
            CONTROL_IS_PRACTICE,
        )
    )
    if not has_practice_module:
        print("\n公共流程执行完成：当前全部使用官方实现。")
        return

    if IKINE_IS_PRACTICE:
        print("\n第三章实践｜逆运动学")
        print(f"  目标关节角/°：{np.round(np.rad2deg(result.q_goal), 3).tolist()}")
        print(f"  FK 回代位置误差：{result.ik_position_error_mm:.6f} mm")
        print(f"  FK 回代姿态误差：{result.ik_orientation_error_deg:.6f}°")
        print(f"  迭代次数：{result.ik_iterations}")
        print(f"  求解残差：{result.ik_residual:.6g}")

    if JACOBIAN_IS_PRACTICE:
        print("\n第四章实践｜雅可比与奇异性")
        print(f"  目标构型最小奇异值：{result.sigma_min_goal:.6g}")
        print(f"  目标构型条件数：{result.condition_goal:.6g}")
        print(f"  整条轨迹最小奇异值：{np.min(result.sigma_min_path):.6g}")
        print("  MeshCat 播放时将实时输出当前最小奇异值")

    if TRAJ_IS_PRACTICE:
        print("\n第五章实践｜轨迹规划")
        print(f"  采样帧数：{len(result.sample_times)}")
        print(f"  峰值关节速度：{np.max(np.abs(result.dq_reference)):.6f} rad/s")
        print("  MeshCat 将显示本次规划得到的 TCP 路径")

    if CONTROL_IS_PRACTICE:
        print("\n第六章实践｜运动控制")
        print(f"  最终关节跟踪误差：{result.final_tracking_error_deg:.6f}°")
        print("  MeshCat 将对比参考/实际 TCP 路径并实时输出跟踪误差")


def plot_result(result: MotionResult) -> None:
    """绘制参考轨迹、离线跟踪轨迹和最小奇异值。"""
    relative_time = result.sample_times - result.sample_times[0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for joint_index in range(result.q_reference.shape[1]):
        axes[0].plot(
            relative_time,
            np.rad2deg(result.q_reference[:, joint_index]),
            label=f"q{joint_index + 1} 参考",
        )
        axes[0].plot(
            relative_time,
            np.rad2deg(result.q_actual[:, joint_index]),
            linestyle="--",
            alpha=0.75,
            label=f"q{joint_index + 1} 实际",
        )
    axes[0].set_ylabel("关节角/°")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)

    axes[1].plot(relative_time, result.sigma_min_path, color="tab:red")
    axes[1].set_xlabel("时间/s")
    axes[1].set_ylabel("最小奇异值")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("第三至第六章公共运动流程")
    fig.tight_layout()
    plt.show()


def show_meshcat(result: MotionResult) -> None:
    """在 MeshCat 中播放轨迹，并按实践模块显示对应章节反馈。"""
    viewer = create_viewer(MODEL_NAME)
    add_pose_marker(
        viewer,
        "motion_exercise/target",
        result.target_pose,
        0xFF5A36,
        size=0.08,
    )

    if IKINE_IS_PRACTICE:
        solved_node = viewer.viewer["motion_exercise/ik_solution/frame"]
        solved_node.set_object(mg.triad(0.06))
        solved_node.set_transform(result.ik_checked_pose.T)
        print("MeshCat 实践反馈｜第三章：显示逆解 FK 回代后的 TCP 坐标系")

    if TRAJ_IS_PRACTICE:
        add_path(
            viewer,
            "motion_exercise/trajectory/planned_tcp",
            result.tcp_reference,
            0x276EF1,
        )
        print("MeshCat 实践反馈｜第五章：蓝线为规划得到的 TCP 路径")

    if CONTROL_IS_PRACTICE:
        add_path(
            viewer,
            "motion_exercise/control/reference_tcp",
            result.tcp_reference,
            0x276EF1,
        )
        add_path(
            viewer,
            "motion_exercise/control/actual_tcp",
            result.tcp_actual,
            0x25A55F,
        )
        print("MeshCat 实践反馈｜第六章：蓝线为参考路径，绿线为实际路径")

    print("\nMeshCat｜播放离线控制轨迹")
    frame_interval = 1.0 / result.sample_hz
    log_stride = max(1, int(round(result.sample_hz / 10.0)))
    has_realtime_feedback = JACOBIAN_IS_PRACTICE or CONTROL_IS_PRACTICE
    for index, q in enumerate(result.q_actual):
        viewer.display(q)
        feedback = []
        if JACOBIAN_IS_PRACTICE:
            feedback.append(f"最小奇异值 {result.sigma_min_path[index]:.6g}")
        if CONTROL_IS_PRACTICE:
            feedback.append(f"关节跟踪误差 {result.tracking_error_deg[index]:.4f}°")
        if feedback and (index % log_stride == 0 or index == len(result.q_actual) - 1):
            print("\r实践反馈｜" + " ｜ ".join(feedback), end="", flush=True)
        time.sleep(frame_interval)
    if has_realtime_feedback:
        print()

    try:
        input("播放完成，按 Enter 退出，或在终端按 Ctrl+C。")
    except (EOFError, KeyboardInterrupt):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="第三至第六章公共运动实践流程")
    parser.add_argument("--plot", action="store_true", help="显示轨迹和控制结果")
    parser.add_argument(
        "--meshcat",
        action="store_true",
        help="使用 MeshCat 播放离线控制轨迹",
    )
    args = parser.parse_args()

    result = run_motion_exercise()
    print_report(result)
    if args.plot:
        plot_result(result)
    if args.meshcat:
        show_meshcat(result)


if __name__ == "__main__":
    main()
