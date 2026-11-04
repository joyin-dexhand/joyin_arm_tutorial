"""第四章雅可比实践的章内辅助框架。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from joyarm_core import JacobianSolver, Wrench, joyArm
from chapt.common.exercise_metrics import frame_end_name, jacobian_metrics
from chapt.common.model_visualization import MODEL_NAME, create_viewer

START_Q = np.deg2rad([-30.0, -40.0, -40.0, -50.0, -20.0, 30.0])
END_Q = np.deg2rad([0.0, -35.0, -70.0, 0.0, 0.0, 0.0])
LOAD_Q = np.deg2rad([-18.0, -55.0, -55.0, -42.0, -18.0, 22.0])
EXTERNAL_LOAD = Wrench(
    force=np.array([0.0, 0.0, -2.0 * 9.81]),
    torque=np.zeros(3),
)


@dataclass(frozen=True)
class JacobianExerciseResult:
    """第四章实践的计算结果。"""

    q_path: np.ndarray
    sigma_min_path: np.ndarray
    condition_path: np.ndarray
    load_q: np.ndarray
    external_load: Wrench
    tau_motor: np.ndarray
    utilization: np.ndarray
    sample_hz: float


def _joint_path(sample_hz: float, duration: float) -> np.ndarray:
    """生成无需轨迹规划器的固定关节路径。"""
    sample_count = int(round(sample_hz * duration)) + 1
    value = np.linspace(0.0, 1.0, sample_count)
    ratio = value * value * (3.0 - 2.0 * value)
    return START_Q + ratio[:, None] * (END_Q - START_Q)


def _show_meshcat(result: JacobianExerciseResult) -> None:
    """播放固定关节路径，并实时输出最小奇异值。"""
    viewer = create_viewer(MODEL_NAME)

    print("\nMeshCat｜第四章雅可比实践")
    log_stride = max(1, int(round(result.sample_hz / 10.0)))
    for index, q in enumerate(result.q_path):
        viewer.display(q)
        if index % log_stride == 0 or index == len(result.q_path) - 1:
            print(
                f"\r当前最小奇异值：{result.sigma_min_path[index]:.6f}",
                end="",
                flush=True,
            )
        time.sleep(1.0 / result.sample_hz)
    print()
    input("播放完成，按 Enter 退出。")


def run_jacobian_exercise(
    solver: JacobianSolver,
    load_mapper: Callable[[np.ndarray, Wrench], np.ndarray],
    *,
    meshcat: bool = False,
) -> JacobianExerciseResult:
    """运行第四章两个任务，并输出本章专属反馈。"""
    arm = joyArm(MODEL_NAME)
    sample_hz = 50.0
    q_path = _joint_path(sample_hz, duration=4.0)

    try:
        jacobians = np.asarray(
            [solver.jac(arm, q, frame_end_name(arm)) for q in q_path],
            dtype=float,
        )
    except NotImplementedError as exc:
        raise SystemExit(
            "第四章实践尚未完成：请补全 "
            "PracticeJacobianSolver.jac() 的实践实现区。"
        ) from exc
    if jacobians.shape != (len(q_path), 6, arm.pin_model.nv):
        raise ValueError(
            "jacobian_exercise.py - 检查雅可比："
            f"得到 {jacobians.shape}，期望 {(len(q_path), 6, arm.pin_model.nv)}"
        )

    metrics = [jacobian_metrics(jacobian) for jacobian in jacobians]
    sigma_min_path = np.asarray([item[0] for item in metrics])
    condition_path = np.asarray([item[1] for item in metrics])

    load_jacobian = solver.jac(arm, LOAD_Q, frame_end_name(arm))
    try:
        tau_motor = np.asarray(
            load_mapper(load_jacobian, EXTERNAL_LOAD),
            dtype=float,
        )
    except NotImplementedError as exc:
        raise SystemExit(
            "第四章实践尚未完成：请补全 "
            "motor_torque_from_external_load() 的实践实现区。"
        ) from exc
    utilization = np.abs(tau_motor) / arm.joint_limits_hard_arm.tau_max

    result = JacobianExerciseResult(
        q_path=q_path,
        sigma_min_path=sigma_min_path,
        condition_path=condition_path,
        load_q=LOAD_Q,
        external_load=EXTERNAL_LOAD,
        tau_motor=tau_motor,
        utilization=utilization,
        sample_hz=sample_hz,
    )

    print("\n第四章实践｜任务 1：雅可比与奇异性")
    print(f"  起点最小奇异值：{sigma_min_path[0]:.6f}")
    print(f"  终点最小奇异值：{sigma_min_path[-1]:.6f}")
    print(f"  路径最小奇异值：{np.min(sigma_min_path):.6f}")
    print(f"  路径最大条件数：{np.max(condition_path):.4f}")

    print("\n第四章实践｜任务 2：末端负载映射为关节负载")
    print(f"  计算构型/°：{np.round(np.rad2deg(LOAD_Q), 3).tolist()}")
    print(f"  外力/N：{np.round(EXTERNAL_LOAD.force, 3).tolist()}")
    print(f"  外力矩/(N·m)：{np.round(EXTERNAL_LOAD.torque, 3).tolist()}")
    print(f"  电机平衡力矩/(N·m)：{np.round(tau_motor, 4).tolist()}")
    print(f"  力矩限位占用率/%：{np.round(utilization * 100.0, 2).tolist()}")

    if meshcat:
        _show_meshcat(result)
    return result
