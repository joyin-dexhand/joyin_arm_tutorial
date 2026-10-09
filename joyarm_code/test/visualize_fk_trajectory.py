"""核心 ToJointTrajPlanner 五次关节插补 + 手写 MDH / Pin FK 对比（离线，不连接硬件）。

使用：在 joyarm_code 目录执行 uv sync 创建环境并安装依赖，随后运行
    .venv/bin/python test/visualize_fk_trajectory.py
无图形界面时：
    .venv/bin/python test/visualize_fk_trajectory.py --no-show
可用 --start / --end 指定六个角度（度），--duration 指定时长（秒）。
输出四张 PNG；默认写入仓库 output/fk_trajectory，图中文字采用英文。
"""
from __future__ import annotations

import argparse
from email import parser
from pathlib import Path
from unittest.mock import patch

import numpy as np
import matplotlib.pyplot as plt

from joyarm_core import FakeArm
from joyarm_core.robotics import MdhFkineSolver, PinFkineSolver
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
    parser = argparse.ArgumentParser(description=__doc__)
# 第 6 组
    parser.add_argument('--start', nargs=6, type=float,
                    default=[-90, -30, -30, -50, -60, 60])
    parser.add_argument('--end', nargs=6, type=float,
                    default=[90, -120, -120, 70, 70, -120])
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--samples', type=int, default=501)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'output/fk_trajectory')
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 1e-3 or args.samples < 3:
        parser.error('duration 必须为至少 0.001 秒的有限数，samples 必须至少为 3')
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    endpoints = np.deg2rad(np.array([args.start, args.end]))
    if not np.all(np.isfinite(endpoints)):
        parser.error('起终点角度必须为有限数')
    arm = FakeArm('joyarm_dm', q0=endpoints[0])
    if np.any(endpoints < arm.arm_limits.q_min) or np.any(endpoints > arm.arm_limits.q_max):
        parser.error('起终点关节角超出硬限位，请调整 --start / --end（单位：度）')
    t, q, dq, ddq = sample_planned_trajectory(arm, endpoints[1], args.duration, args.samples)
    pin_T = PinFkineSolver().solve(arm, q, arm.ee_frame_name, rep='T')
    mdh_T = MdhFkineSolver().solve(arm, q, arm.ee_frame_name, rep='T')
    pin_p, mdh_p = pin_T[:, :3, 3], mdh_T[:, :3, 3]
    position_error = np.linalg.norm(pin_p - mdh_p, axis=1)
    # atan2 提取相对旋转角，避免近零角时 acos 的精度损失。
    R = pin_T[:, :3, :3] @ mdh_T[:, :3, :3].transpose(0, 2, 1)
    skew = np.column_stack([R[:, 2, 1] - R[:, 1, 2],
                            R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]])
    angle_error = np.arctan2(np.linalg.norm(skew, axis=1) / 2,
                             (np.trace(R, axis1=1, axis2=2) - 1) / 2)

    fig1, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout='constrained')
    for ax, values, label in zip(axes, (q, dq, ddq),
                                ('Position (deg)', 'Velocity (deg/s)', 'Acceleration (deg/s²)')):
        for j in range(q.shape[1]):
            ax.plot(t, np.rad2deg(values[:, j]), label=f'J{j + 1}')
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=6)
    axes[-1].set_xlabel('Time (s)')
    fig1.suptitle('Quintic joint trajectory — zero endpoint velocity and acceleration\n'
                  'Position, velocity and acceleration join continuously to rest')

    fig2 = plt.figure(figsize=(12, 8), layout='constrained')
    grid = fig2.add_gridspec(3, 2)
    ax3 = fig2.add_subplot(grid[:, 0], projection='3d')
    ax3.plot(*pin_p.T, label='Pin FK', linewidth=3)
    ax3.plot(*mdh_p.T, '--', label='MDH FK', linewidth=1.5)
    ax3.scatter(*pin_p[0], color='green', label='Start')
    ax3.scatter(*pin_p[-1], color='red', label='End')
    # 各轴采用相同米制范围，避免视觉上的路径形状失真。
    center = (pin_p.max(axis=0) + pin_p.min(axis=0)) / 2
    radius = max(float(np.ptp(pin_p, axis=0).max()) * 0.55, 0.01)
    for setter, c in zip((ax3.set_xlim, ax3.set_ylim, ax3.set_zlim), center):
        setter(c - radius, c + radius)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='TCP path')
    ax3.legend()
    for i, label in enumerate('XYZ'):
        ax = fig2.add_subplot(grid[i, 1])
        ax.plot(t, pin_p[:, i], label='Pin FK', linewidth=3)
        ax.plot(t, mdh_p[:, i], '--', label='MDH FK')
        ax.set(xlabel='Time (s)', ylabel=f'{label} (m)')
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend()
    fig2.suptitle('Same joint samples → two forward kinematics implementations')

    fig3, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
    for p, label, style in ((pin_p, 'Pin FK', '-'), (mdh_p, 'MDH FK', '--')):
        speed = np.linalg.norm(np.gradient(p, t, axis=0, edge_order=2), axis=1)
        axes[0].plot(t, speed, style, label=label)
    axes[0].set_ylabel('TCP linear speed (m/s)')
    axes[0].legend()
    axes[0].set_title('Speed estimated by finite differences (endpoint approximation)')
    axes[1].plot(t, position_error)
    axes[1].set_ylabel('Position error (m)')
    axes[2].plot(t, angle_error)
    axes[2].set_ylabel('Orientation error (rad)')
    axes[2].set_xlabel('Time (s)')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig3.suptitle('TCP speed and FK agreement')
    # 相同起终位置和时长；五次加速度来自核心规划器，三次仅作解析对照。
    delta = endpoints[1] - endpoints[0]
    joint = int(np.argmax(np.abs(delta)))
    cubic_acc = np.rad2deg((6 - 12 * t / args.duration) * delta[joint] / args.duration**2)
    quintic_acc = np.rad2deg(ddq[:, joint])
    # 重复端点时刻，明确绘出三次与静止段连接时的竖直跳变。
    display_t = np.concatenate(([-1.0, 0.0], t, [args.duration, args.duration + 1.0]))
    cubic_display = np.concatenate(([0.0, 0.0], cubic_acc, [0.0, 0.0]))
    quintic_display = np.concatenate(([0.0, 0.0], quintic_acc, [0.0, 0.0]))
    fig4, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    ax.axvspan(-1, 0, color='gray', alpha=0.12)
    ax.axvspan(args.duration, args.duration + 1, color='gray', alpha=0.12)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    ax.axvline(args.duration, color='gray', linestyle=':', linewidth=1)
    ax.plot(display_t, cubic_display, label='Cubic (analytic reference)', color='tab:orange', linewidth=2)
    ax.plot(display_t, quintic_display, '--', label='Quintic (ToJointTrajPlanner)', color='tab:blue', linewidth=2)
    ax.scatter([0, args.duration], [0, 0], color='tab:blue', zorder=5)
    if abs(delta[joint]) > 1e-12:
        for moment, acceleration, offset in ((0, cubic_acc[0], 35),
                                             (args.duration, cubic_acc[-1], -115)):
            ax.annotate('Cubic acceleration jump', xy=(moment, acceleration / 2),
                        xytext=(offset, 35), textcoords='offset points',
                        arrowprops={'arrowstyle': '->', 'color': 'tab:orange'}, fontsize=9)
    ax.set(xlabel='Time (s)', ylabel='Joint acceleration (deg/s²)',
           xlim=(-1, args.duration + 1),
           title=f'J{joint + 1}: cubic vs quintic start/stop acceleration\n'
                 'Same endpoints and duration; shaded regions = 1 s rest')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper center')
    print(f'加速度对比选择 J{joint + 1}（运动幅度最大）：'
          f'{np.rad2deg(delta[joint]):.2f} deg；三次起终加速度 '
          f'{cubic_acc[0]:.3f} / {cubic_acc[-1]:.3f} deg/s²；'
          f'五次 {quintic_acc[0]:.3e} / {quintic_acc[-1]:.3e} deg/s²')
    if np.all(np.abs(delta) <= 1e-12):
        print('起终角度相同，没有运动，两种加速度均为零。')
    args.output.mkdir(parents=True, exist_ok=True)
    for fig, name in ((fig1, 'joint_curves'), (fig2, 'tcp_path'), (fig3, 'fk_errors'),
                      (fig4, 'acceleration_comparison')):
        path = args.output / f'{name}.png'
        fig.savefig(path, dpi=160)
        print(f'图片：{path}')
    print(f'最大位置误差：{position_error.max():.3e} m')
    print(f'最大姿态误差：{angle_error.max():.3e} rad')
    print(f'最大变换矩阵元素差：{np.max(np.abs(pin_T - mdh_T)):.3e}')
    print('五次插补起终速度和加速度均为零，与静止段连接时位置、速度和加速度连续。')
    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
