"""
==============================================================================
综合抓取  (快速上手 例程 07)
==============================================================================
【功能概要】
    综合例程——串联逆解、FSM 关节运动与夹爪动作完成「抓取—放置」闭环，
    一次运行即可体验 JoyArm 真机编程的典型代码形态：
      ① 连接本体/末端并启动运动状态机（FSM），等待 MOTION 后张开夹爪
      ② 正运动学取当前位姿快照，构造「上方点 + 抓取点」，逆解并做小幅安全校验
      ③ set_target_joint 关节运动：到上方点 → 降到抓取点 → 闭爪夹取 → 抬回上方点
      ④ 回 home 释放物体，stop_motion_fsm 安全停机，with 退出自动断连
    覆盖接口：connect_arm / connect_end / set_traj_solver / start_motion_fsm / wait_for_motion_state /
    solve_fkine / solve_ikine / set_target_joint / is_in_q_arm / set_action_end / stop_motion_fsm。

【安全须知】
    - 清空机械臂工作半径内的人员与障碍物，确认急停按钮在手边随时可按
    - 全程小幅运动：所有关节目标均校验在 home 位形 ±0.2 rad 邻域内；
      经三次多项式规划器按 2 s 行程插值，关节峰值速度 ≤0.15 rad/s（安全规范 ≤0.3 rad/s）
    - 首次运动前有回车确认门；运行中任何异常请立即急停或 Ctrl+C（退出自动安全停机）

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 2. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（在 joyarm_code/ 目录下，需连接真机）
    python quickstart/07_pick_demo.py

    预期输出：依次打印 ① 准备 → ② 规划（抓取偏移与关节偏差）→ ③ 抓取 → ④ 收尾
    四段日志，机械臂完成一次「到点 → 闭爪 → 抬起 → 回 home 释放」的小幅抓取动作。
==============================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装 joyarm_core 时也可直接运行
from joyarm_core import MotionState, Pose, joyArm  # noqa: E402


def move_to(arm, q_target, horizon_s=2.0, timeout_s=6.0):
    """下发单帧关节目标并轮询到位。

    目标时刻取 time.time()+horizon_s（Unix 秒）：main 已切换三次多项式规划器
    single_joint_cubic_traj_planner，按该时刻插值——小幅运动（≤0.2 rad）配 2 s
    行程，关节峰值速度约 ≤0.15 rad/s，满足安全限速规范（≤0.3 rad/s）。
    """
    arm.set_target_joint(t=time.time() + horizon_s, q=q_target)  # 单帧目标（FSM 规划/控制线程接管）
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not arm.is_in_q_arm(q_target):  # 轮询到位（逐关节容差判断）
        time.sleep(0.05)
    if not arm.is_in_q_arm(q_target):
        print("    警告：限时内未到位，继续下一步（请检查是否被障碍阻挡）")


def main():
    print("=" * 62)
    print("JoyArm 快速上手 07 —— 综合抓取（真机例程）")
    print("=" * 62)

    with joyArm("rebot_arm_b601_dm") as arm:           # with 退出自动安全收尾（停机/失能/断连）
        # ① 准备：连接本体与末端 → 切限速规划器 → 确认门 → 启动 FSM → 等待 MOTION → 张开夹爪
        if arm.connect_arm() is not True:      # 连接本体电机总线（backend 失败仅 warn 不抛错，须检查返回值）
            raise RuntimeError("arm 总线连接失败，请检查电源与总线后重试")
        if arm.connect_end() is not True:      # 连接末端（夹爪）电机总线（未连接时夹爪动作会被静默跳过）
            raise RuntimeError("end 总线连接失败：夹爪动作将被静默跳过，请检查末端电机与总线")
        arm.set_traj_solver("single_joint_cubic_traj_planner")  # 切三次多项式规划器：按 t 插值限速（默认直传规划器忽略 t）
        input("\n确认机械臂工作半径内无人员与障碍物后，回车开始首次运动：")
        arm.start_motion_fsm()                 # 启动运动状态机：DISABLE → HOME → MOTION 自动迁移
        if arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0) != 1:
            raise RuntimeError("FSM 未在 10 s 内进入 MOTION 态，请检查电机状态")
        arm.set_action_end("open")             # 张开夹爪，准备抓取
        print("① 准备完成：FSM 已进入 MOTION 态，夹爪已张开")

        # ② 规划：FK 取当前位姿快照 → 构造抓取点/上方点 → IK 求解并做小幅校验
        base = arm.solve_fkine()               # 当前（home 位形）末端位姿快照
        offset = np.array([0.05, 0.0, -0.02])  # 抓取点偏移：前方 5 cm、下移 2 cm（姿态不变）
        for _ in range(5):                     # 校验不过则等比例缩小偏移重解，最多 5 次
            pose_grab = Pose(position=base.position + offset,
                             orientation=base.orientation)      # 抓取点位姿（姿态与当前一致）
            pose_above = Pose(position=base.position + offset + np.array([0.0, 0.0, 0.05]),
                              orientation=base.orientation)     # 上方点 = 抓取点再抬高 5 cm
            res_grab = arm.solve_ikine(pose_grab, q0=arm.home_arm)   # 逆解（home 初值 → 解靠近 home）
            res_above = arm.solve_ikine(pose_above, q0=arm.home_arm)
            ok = res_grab.success and res_above.success
            # 安全校验：关节解偏离 home 须 <0.2 rad（小幅运动规范，严于 0.6 rad 设计上限）
            dev = max(np.max(np.abs(res_grab.q - arm.home_arm)),
                      np.max(np.abs(res_above.q - arm.home_arm))) if ok else float("inf")
            if ok and dev < 0.2:
                break                          # 校验通过：两点均可行且幅度小幅
            offset = offset * 0.5              # 逆解失败或幅度超限 → 偏移减半重解
        else:
            raise RuntimeError("② 规划失败：小幅邻域内无可行抓取点，请调整初始偏移")
        print(f"② 规划完成：抓取偏移 {np.round(offset, 4)} m，关节解偏离 home 最大 {dev:.3f} rad")

        # ③ 抓取：到上方点 → 降到抓取点 → 闭爪夹取 → 抬回上方点
        print("③ 开始抓取：先运动到上方点")
        move_to(arm, res_above.q)              # 上方点关节解（接近抓取点，便于竖直下探）
        print("    下降到抓取点")
        move_to(arm, res_grab.q)               # 抓取点关节解
        arm.set_action_end("close")            # 闭合夹爪，夹取物体
        time.sleep(2.5)                        # 等闭合到位再抬臂：行程 5.0 rad / vlim 3.0 rad/s ≈ 1.67 s，取 2.5 s 留余量
        print("    夹取完成，抬回上方点")
        move_to(arm, res_above.q)

        # ④ 放置与收尾：回 home → 张爪释放 → 安全停机 → with 退出自动断连
        print("④ 回 home 位形释放物体")
        move_to(arm, arm.home_arm)             # 回 home 位形（放回起始位置）
        arm.set_action_end("open")             # 张开夹爪，释放物体
        time.sleep(0.5)                        # 等待物体脱离
        arm.stop_motion_fsm()                  # 阻塞安全停机：经 HOME 回零后失能
        print("④ 收尾完成：FSM 已停机，with 退出时自动断开连接")

    print("\n例程运行完成：已走通「规划 → 抓取 → 放置」完整闭环。")


if __name__ == "__main__":
    main()
