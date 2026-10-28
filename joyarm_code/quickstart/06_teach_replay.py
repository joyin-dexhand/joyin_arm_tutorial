"""
==============================================================================
拖动示教与回放  (快速上手 例程 06)
==============================================================================
【功能概要】
    本例程是教程 §7.3「示教和轨迹复现」的代码级实现（末端按钮触发版属后续章节）：
      1. 连接真机并启动运动状态机，切入 DAMP 阻尼态（可手动拖动、带重力补偿）；
      2. 手动拖动机械臂示教一段轨迹，以 50 Hz 录制 4 秒的关节角序列；
      3. 切回 MOTION 正常运动态，以 10 Hz 逐点下发录制路点，自动复现该轨迹。
    覆盖接口：connect_arm / start_motion_fsm / set_event(SafetyEvent) /
    wait_for_motion_state(MotionState) / joint_state_arm / home_arm /
    set_target_joint / stop_motion_fsm。

【安全须知】
    - 清空机械臂工作半径内的人员与障碍物
    - 全程值守并备好急停手段（急停按钮或直接断电）
    - 拖动示教与回放均保持缓慢小幅：建议各关节相对 home 位形偏移不超过 0.2 rad

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 2. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（在 joyarm_code/ 目录下，需已连接真机）
    python quickstart/06_teach_replay.py

    预期：进入阻尼态后手动拖动示教 4 秒 → 回车确认 → 机械臂平滑复现刚才的轨迹，
    最后回零失能、安全退出。
==============================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装时也可直接运行
from joyarm_core import MotionState, SafetyEvent, joyArm  # noqa: E402

RECORD_HZ = 50        # 录制采样频率（Hz）
RECORD_SEC = 4.0      # 录制时长（秒）
REPLAY_HZ = 10        # 回放路点下发频率（Hz）
REPLAY_AHEAD = 0.15   # 路点目标到达时刻提前量（秒；仅 cubic 规划器生效，见 ④ 注释）


def main() -> None:
    """拖动示教与回放主流程（真机，三处回车确认门）。"""
    with joyArm("rebot_arm_b601_dm") as arm:  # with 退出时自动安全收尾（close）
        arm.connect_arm()  # 连接 arm 电机总线（成功后自动设 cfg 默认模式）
        input("确认机械臂工作半径内无人员与障碍物后，回车开始首次运动（先自动回 home）：")
        arm.start_motion_fsm()  # 启动运动状态机：DISABLE → HOME → MOTION 自动迁移
        if arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0) != 1:
            print("10 秒内未进入 MOTION 正常运动态，安全收尾退出。")
            return
        print(f"已进入 MOTION 态，home 位形 = {np.round(arm.home_arm, 3)} rad")

        # ① 切入阻尼态：DAMP=阻尼态可手动拖动，重力补偿程度由 cfg 的 damping_k_gravity 决定
        arm.set_event(SafetyEvent.DAMPING)  # 阻尼防护事件 → FSM 目标态 DAMP
        if arm.wait_for_motion_state(MotionState.DAMP) != 1:
            print("未进入 DAMP 阻尼态，安全收尾退出。")
            return
        print("已进入 DAMP 阻尼态：现可用手握臂杆缓慢拖动。")

        # ② 录制：以 50 Hz 采样 4 秒的 arm 关节角序列
        input("缓慢拖动机械臂示范一段轨迹，就位后回车开始录制 4 秒...")
        frames, reported = [], 0  # 关节角帧列表 / 已打印的整秒数
        t0 = time.perf_counter()
        tick = t0
        while time.perf_counter() - t0 < RECORD_SEC:
            frames.append(list(arm.joint_state_arm.q))  # 关节位置快照 (n_arm,)
            sec = int(time.perf_counter() - t0)
            if sec > reported:  # 每秒打印一次录制进度
                reported = sec
                print(f"录制中 {sec}/{RECORD_SEC:.0f} 秒")
            tick += 1.0 / RECORD_HZ  # 按绝对时刻对齐节拍，保证约 50 Hz
            time.sleep(max(0.0, tick - time.perf_counter()))
        record = np.array(frames)  # 形状约 (200, n_arm)
        print(f"录制完成 {record.shape[0]} 帧，起点 q = {np.round(record[0], 3)} rad")

        # ③ 回放准备：切回正常事件（DAMP → HOME → MOTION，途中先自动回到 home 位形）
        arm.set_event(SafetyEvent.NORMAL)  # 正常事件 → FSM 目标态 MOTION
        if arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0) != 1:
            print("10 秒内未回到 MOTION 正常运动态，安全收尾退出。")
            return
        input("回放将复现刚才的轨迹，确认工作空间清空后回车...")

        # ④ 回放：录制序列每 0.1 秒取一路点（约 40 个），逐点以 10 Hz 下发
        waypoints = record[::RECORD_HZ // REPLAY_HZ]  # 每 5 帧取 1 帧作路点
        print(f"开始回放：共 {len(waypoints)} 个路点，{REPLAY_HZ} Hz 下发...")
        for q_wp in waypoints:
            # 默认激活 single_joint_hold_traj_planner（配置首位）：plan 仅取 q 直传、忽略 t，
            # 回放即 10 Hz 位置直传流，速度由电机 vlim 兜底；t = 当前时刻 + REPLAY_AHEAD 的
            # Unix 绝对时刻仅当 set_traj_solver 切到 single_joint_cubic_traj_planner 后生效
            # （其 plan 以 T = t - now 做三次多项式按到达时刻平滑跟帧）
            arm.set_target_joint(t=time.time() + REPLAY_AHEAD, q=q_wp)
            time.sleep(1.0 / REPLAY_HZ)
        time.sleep(REPLAY_AHEAD)  # 末路点后的收尾余量（cubic 规划器下恰为到达时刻）
        print(f"回放完成，当前 q = {np.round(arm.joint_state_arm.q, 3)} rad")

        # ⑤ 收尾：阻塞安全停机（经 HOME 回零失能），随后 with 退出自动 close
        arm.stop_motion_fsm()
        print("已安全停机，例程结束。")


if __name__ == "__main__":
    main()
