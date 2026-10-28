"""
==============================================================================
直接指令与夹爪  (快速上手 例程 03)
==============================================================================
【功能概要】
    真机演示不经轨迹规划、直接向 JoyArm 下发底层指令的代码形态：
      1. 连接并使能本体（arm）与夹爪（end）两条总线
      2. 位置模式：send_position_arm 下发小幅目标 + is_in_q_arm 轮询到位
      3. 速度模式：send_vel_arm 单关节点动 1 秒后发零速停止
      4. MIT 模式：send_mit_arm 五元组指令保持当前位形
      5. 夹爪：set_action_end 离散动作（open/close/home/position）
         与 send_position_end 连续位置指令
    本例程在 FSM 未运行时直接下发指令；FSM 运行期间 arm 侧 send_*/set_mode/enable
    会被互斥拦截（warn + 空操作），end 侧不受限。

【安全须知】
    - 运行前清空机械臂工作半径内的人员与障碍物，确认本体与夹爪运动路径无遮挡
    - 手边准备急停：随时 Ctrl+C 中断，with 退出会自动安全收尾
    - 全程小幅缓慢运动：位置目标以当前位形为基准 ±0.2 rad 内，速度不超过 0.3 rad/s

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 2. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（在 joyarm_code/ 目录下，需已连接 rebot_arm_b601_dm 真机并上电）
    python quickstart/03_direct_command.py

    预期输出：各阶段中文提示；本体第 2、3 关节小幅抬起后点动，夹爪开合、
    回中并运动到指定位置，最后失能退出。
==============================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装时也可直接运行
from joyarm_core import joyArm, ControlMode  # noqa: E402


def main() -> None:
    with joyArm("rebot_arm_b601_dm") as arm:                        # with 退出自动安全收尾
        # ① 连接并使能：本体（arm）与夹爪（end）两条总线
        arm.connect_arm()                                   # 连接本体总线（通道取配置缺省值）
        arm.connect_end()                                   # 连接夹爪总线
        arm.enable_arm()                                    # 使能本体全部关节电机
        arm.enable_end()                                    # 使能夹爪电机

        input("\n>>> 即将开始小幅运动，请确认工作区安全后按回车继续……")

        # ② 位置模式：以当前位形为基准，仅第 2、3 关节各 +0.15 rad
        arm.set_mode_arm(ControlMode.POSITION)              # 切位置模式（自动随写位置环增益）
        q_goal = arm.joint_state_arm.q.copy()               # 基准取当前位形快照（copy 再修改），
                                                            # 其余关节目标=当前值保持不动，避免整机向 home 归位
        q_goal[1] += 0.15                                   # 第 2 关节 +0.15 rad
        q_goal[2] += 0.15                                   # 第 3 关节 +0.15 rad
        print(f"位置模式：目标关节角 q_goal = {np.round(q_goal, 3)} rad")
        arm.send_position_arm(q_goal)                       # 整组下发目标关节角（越限自动裁剪）
        t0, ok = time.time(), False
        while time.time() - t0 <= 3.0:                      # 最多轮询 3 秒
            if arm.is_in_q_arm(q_goal, tol_q=0.05):         # 到位判断（全关节容差 0.05 rad）
                ok = True
                break
            time.sleep(0.05)                                # 0.05 s 轮询间隔
        print("位置模式：已到位" if ok else "位置模式：3 秒未到位（负载/阻尼偏大），继续下一步")

        # ③ 速度模式：仅第 2 关节以 0.3 rad/s 点动 1 秒
        arm.set_mode_arm(ControlMode.VELOCITY)              # 切速度模式
        dq = np.zeros(arm.n_arm)                            # n_arm = 本体关节数
        dq[1] = 0.3                                         # 第 2 关节 +0.3 rad/s，其余为零
        print("速度模式：第 2 关节 0.3 rad/s 运行 1 秒")
        arm.send_vel_arm(dq)                                # 整组下发目标速度
        time.sleep(1.0)
        arm.send_vel_arm(np.zeros(arm.n_arm))               # 先发零速停止，再切其他模式

        # ④ MIT 模式：五元组 (q, dq, tau, kp, kd) = 目标位置/速度前馈/力矩前馈/位置增益/速度增益
        arm.set_mode_arm(ControlMode.MIT)                   # 切 MIT 模式
        arm.send_mit_arm(q=arm.joint_state_arm.q)           # 目标=当前位形原地保持（kp/kd 缺省取配置）
        time.sleep(0.5)

        # ⑤ 夹爪（end 侧不受 FSM 互斥）：离散动作 + 连续位置（行程硬限位 [-5.49, 0.0] rad）
        arm.set_action_end("open")                          # 张开到固定行程位（-5.0 rad）
        time.sleep(1.5)
        arm.set_action_end("close")                         # 闭合到固定行程位（0.0 rad）
        time.sleep(1.5)
        arm.set_action_end("home")                          # 回夹爪中位（-3.0 rad，即 home_end）
        time.sleep(1.0)
        arm.set_action_end("position", position=-3.5)       # 带参动作：夹爪运动到指定位置 -3.5 rad
        time.sleep(1.0)
        arm.send_position_end(np.array([-4.0]))             # 连续位置指令（越硬限位就近裁剪）
        time.sleep(1.0)
        print(f"夹爪当前位置 = {np.round(arm.joint_state_end.q, 3)} rad")

        # ⑥ 收尾：显式失能，随后 with 退出自动断开并安全收尾
        arm.disable_arm()                                   # 失能本体全部关节电机
        arm.disable_end()                                   # 失能夹爪电机
        print("例程完成：机械臂已失能。")


if __name__ == "__main__":
    main()
