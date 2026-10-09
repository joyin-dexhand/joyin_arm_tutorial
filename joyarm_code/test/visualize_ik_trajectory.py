#末端目标
#→ 解析 IK 得到 q_goal
#→ 五次关节插补
#→ FK 计算实际 TCP 轨迹
#→ 画图观察轨迹和终点

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from joyarm_core import AnalyticIkineSolver, FakeArm
from joyarm_core import PinIkineSolver
from joyarm_core.robotics import PinFkineSolver

from unittest.mock import patch

from joyarm_core.robotics.trajectory import ToJointTrajPlanner
from joyarm_core.utils.types import TrajFrame


def sample_planned_trajectory(arm, q1, duration, samples):
    """调用核心五次规划器一次，再离线采样；不启动周期线程。

    固定测试时钟，保证规划起点与采样起点完全一致，无须访问私有系数。
    提前量取 0，允许测试任意正时长（核心规划器最小时长为 1 ms）。
    """
    t0 = 1000.0  # 离线测试的绝对时间基准，不代表实际执行时间
    arm.set_target_traj([TrajFrame(time=t0 + duration, q=q1)])
    planner = ToJointTrajPlanner(dt_min_required=0.0)
    with patch('time.time', return_value=t0):
        if not planner.plan_once(arm):
            raise RuntimeError('visualize_fk_trajectory.py - 轨迹规划：未生成轨迹')
    t = np.linspace(0.0, duration, samples)
    frames = [planner.sample_frame(t0 + float(dt)) for dt in t]
    q = np.stack([frame.q for frame in frames])
    dq = np.stack([frame.dq for frame in frames])
    ddq = np.stack([frame.ddq for frame in frames])
    return t, q, dq, ddq

def main():
    frame = "frame_end_tcp"
    

    q_start_deg = np.array([-90, -30, -30, -50, -60, 60],dtype=float,)#起点

    q_known_deg = np.array([-30, -40, -20, 40, 60, 120],dtype=float,)#可达末端目标

    q_start = np.deg2rad(q_start_deg)
    q_known = np.deg2rad(q_known_deg)

    arm = FakeArm("joyarm_dm",
                      q0=q_start,)

    pin_fk = PinFkineSolver()
    ik_solver = PinIkineSolver()
    #ik_solver = AnalyticIkineSolver()

    # 用已知可行关节角经独立的 Pin FK 生成完整目标位姿，避免手写位置和姿态不匹配。
    target = pin_fk.solve(arm, q_known, frame, rep="pose")

    result = ik_solver.solve(arm, target, frame, q_start)


    if not result.success:
        raise RuntimeError("visualize_ik_trajrctory.py - main：解析 IK 求解失败")

    q_goal = result.q


    duration = 5.0
    samples = 501

    t, q, dq, ddq = sample_planned_trajectory(
        arm,
        q_goal,
        duration,
        samples,
    )
    print("轨迹起点（deg）：")
    print(q_start_deg)

    print("用于生成目标的已知角度（deg）：")
    print(q_known_deg)

    print("解析 IK 输出角度（deg）：")
    print(np.rad2deg(q_goal))


##位置速度加速度
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
    )

    for joint in range(6):
        axes[0].plot(
            t,
            np.rad2deg(q[:, joint]),
            label=f"J{joint + 1}",
        )

    axes[0].set_ylabel("Position (deg)")
    axes[0].set_title("Joint Position")
    axes[0].grid(True)
    axes[0].legend(ncol=6)

    for joint in range(6):
        axes[1].plot(
            t,
            np.rad2deg(dq[:, joint]),
            label=f"J{joint + 1}",
        )

    axes[1].set_ylabel("Velocity (deg/s)")
    axes[1].set_title("Joint Velocity")
    axes[1].grid(True)

    for joint in range(6):
        axes[2].plot(
            t,
            np.rad2deg(ddq[:, joint]),
            label=f"J{joint + 1}",
        )

    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Acceleration (deg/s²)")
    axes[2].set_title("Joint Acceleration")
    axes[2].grid(True)

    fig.tight_layout()

    # 批量 FK 使用 rep="T"，直接得到形状为 (N, 4, 4) 的齐次矩阵。
    actual_T = pin_fk.solve(arm, q, frame, rep="T")
    actual_position = actual_T[:, :3, 3]
    target_position = target.position

    final_position_error = np.linalg.norm(actual_position[-1] - target_position)
    print("目标TCP位置（m）：")
    print(target_position)
    print("实际TCP位置（m）：")
    print(actual_position[-1])
    print(f"最终位置误差：{final_position_error:.3e} m")

    fig_tcp = plt.figure(figsize=(9, 7))
    ax_tcp = fig_tcp.add_subplot(111, projection='3d')
    ax_tcp.plot(actual_position[:, 0], actual_position[:, 1], actual_position[:, 2],color="tab:blue",linewidth=2,label="Actual TCP path",)

    ax_tcp.scatter(
        actual_position[0, 0],
        actual_position[0, 1],
        actual_position[0, 2],
        color="green",
        s=80,
        label="Start",
    )
    ax_tcp.scatter(
        target_position[0],
        target_position[1],
        target_position[2],
        color="red",
        marker="*",
        s=180,
        label="Target",
    )
    ax_tcp.scatter(
        actual_position[-1, 0],
        actual_position[-1, 1],
        actual_position[-1, 2],
        facecolors="none",
        edgecolors="black",
        marker="o",
        s=120,
        linewidths=2,
        label="Actual end",
    )

    ax_tcp.set_xlabel("X (m)")
    ax_tcp.set_ylabel("Y (m)")
    ax_tcp.set_zlabel("Z (m)")
    ax_tcp.set_title("TCP Path from Joint Trajectory")
    ax_tcp.legend()
    ax_tcp.grid(True)
    center = (
        actual_position.max(axis=0)
        + actual_position.min(axis=0)
    ) / 2.0

    radius = max(
        np.ptp(actual_position, axis=0).max() * 0.55,
        0.01,
    )

    ax_tcp.set_xlim(
        center[0] - radius,
        center[0] + radius,
    )
    ax_tcp.set_ylim(
        center[1] - radius,
        center[1] + radius,
    )
    ax_tcp.set_zlim(
        center[2] - radius,
        center[2] + radius,
    )
    ax_tcp.set_box_aspect((1, 1, 1))
###误差error
    position_error_mm = np.linalg.norm(
        actual_position - target_position,
        axis=1,
    ) * 1000.0
    target_rotation = target.T[:3, :3]
    actual_rotation = actual_T[:, :3, :3]

    relative_rotation = np.einsum(
        "ij,njk->nik",
        target_rotation.T,
        actual_rotation,
    )

    skew_vector = np.column_stack([
        relative_rotation[:, 2, 1] - relative_rotation[:, 1, 2],
        relative_rotation[:, 0, 2] - relative_rotation[:, 2, 0],
        relative_rotation[:, 1, 0] - relative_rotation[:, 0, 1],
    ])

    sin_angle = np.linalg.norm(
        skew_vector,
        axis=1,
    ) / 2.0

    cos_angle = (
        np.trace(
            relative_rotation,
            axis1=1,
            axis2=2,
        )
        - 1.0
    ) / 2.0

    orientation_error_deg = np.rad2deg(
        np.arctan2(
            sin_angle,
            cos_angle,
        )
    )
    print(
        "最终 TCP 位置误差（mm）：",
        position_error_mm[-1],
    )

    print(
        "最终 TCP 姿态误差（deg）：",
        orientation_error_deg[-1],
    )
    fig_error, axes_error = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
    )
    coordinate_names = ("X", "Y", "Z")
    coordinate_colors = (
        "tab:red",
        "tab:green",
        "tab:blue",
    )

    for index, (name, color) in enumerate(
        zip(coordinate_names, coordinate_colors)
    ):
        axes_error[0].plot(
            t,
            actual_position[:, index],
            color=color,
            label=f"Actual {name}",
        )

        axes_error[0].axhline(
            target_position[index],
            color=color,
            linestyle="--",
            alpha=0.6,
            label=f"Target {name}",
        )

    axes_error[0].set_ylabel("Position (m)")
    axes_error[0].set_title("TCP Position and Final Target")
    axes_error[0].grid(True)
    axes_error[0].legend(ncol=3)
    axes_error[1].plot(
        t,
        position_error_mm,
        color="tab:purple",
    )

    axes_error[1].set_ylabel("Distance (mm)")
    axes_error[1].set_title("TCP Distance to Final Target")
    axes_error[1].grid(True)

    axes_error[2].plot(
        t,
        orientation_error_deg,
        color="tab:orange",
    )

    axes_error[2].set_xlabel("Time (s)")
    axes_error[2].set_ylabel("Angle Error (deg)")
    axes_error[2].set_title(
        "TCP Orientation Error to Final Target"
    )
    fig_error.tight_layout()
    axes_error[2].grid(True)


    plt.show()



if __name__ == "__main__":
    main()
