"""
==============================================================================
例程 04 —— 直通硬件电机指令（需真机）
==============================================================================
【功能概要】
    1. 连接并使能 arm 与 end，打印当前模式与关节位置；
    2. MIT 模式：set_mode_arm/end(MIT) → send_mit_arm/end；
    3. 位置模式：set_mode_arm/end（POSITION） → send_position_arm/end  → is_in_q_arm/end 到位判断；
    4. 速度模式：set_mode_arm/end(VELOCITY) → send_velocity_arm/end；
    5. 末端离散动作：set_action_end("action") ，如 "open" / "close" / "home"；
    6. FSM 互斥说明：FSM 运行时 arm 侧下发类指令会被拦下（warn警告，不执行）。

【原理简介】
    底层电机通常有三类控制模式：MIT 模式、位置模式和速度模式；闭环增益（cfg.backend 配置）通常在切模式时（位置/速度模式）可由 backend 自动写入电机；
    send_mit/position/velocity_arm 指令可绕过轨迹规划与 FSM，由用户直接发送指令至电机，响应即时但无安全过渡；
    FSM 运行时 arm 电机被独占，此时在外部 send 指令会被拦下（warn 提示 + 不执行），end 不受限制。

【安全须知】
    - 运行前清空机械臂工作半径内的人员与障碍物，确认本体与夹爪运动路径无遮挡
    - 手边准备随时急停或断电；随时 Ctrl+C 中断，with 退出会自动失能并断开总线
    - 全程小幅缓慢运动，运动时切勿强行扳动关节

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code && uv sync

    # 2. 运行本脚本（需连接对应的机械臂）
    uv run python quick_start/04_direct_motor_command.py <型号>  # 如：rebot_arm_b601_dm

【预期输出】
    连接 → 使能 → 位置模式 → 速度模式 → MIT 模式 → 末端动作 → 失能退出

==============================================================================
"""

from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from joyarm_core import joyArm, ControlMode, listArm


# ============================================================
# 一、命令行型号参数：解析必填的 <型号> 命令行参数，缺失时打印用法并以退出码 1 结束
# ============================================================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并列出可用型号（listArm），以退出码 1 结束。"""
    if len(sys.argv) != 2:                                # 必须恰好携带 1 个型号参数
        print(f"用法：uv run python quick_start/04_direct_motor_command.py <型号>")  # 缺参：先打印用法
        listArm()                                         # 顺带列出全部可用型号
        sys.exit(1)                                       # 退出码 1，不进入演示
    return sys.argv[1].strip()                            # 型号名


# ============================================================
# 二、主流程：连接使能 → 三模式直通指令 → 夹爪 → FSM 互斥说明 → 失能收尾
# ============================================================
def main() -> None:
    """真机流程：连接使能 → 位置/速度/MIT 直通指令 → 夹爪 → FSM 互斥说明 → 失能收尾。"""
    model = parse_model()                                 # 必填型号参数

    with joyArm(model) as arm:                            # with 退出（含异常）自动 close：停线程 → 失能 → 断连

        print("=============== ① 连接与使能 ======================================= ")
        ok = arm.connect_arm()                            # 连接 arm
        print(f"connect_arm 返回：{ok}；is_connected_arm：{arm.is_connected_arm}")
        ok = arm.connect_end()                            # 连接 end
        print(f"connect_end 返回：{ok}；is_connected_end：{arm.is_connected_end}")
        ret_arm = arm.enable_arm()                            # 使能 arm
        print(f"enable_arm 返回：{ret_arm}；使能后 is_enabled_arm：{arm.is_enabled_arm}")
        ret_arm = arm.enable_end()                            # 使能 end
        print(f"enable_end 返回：{ret_arm}；使能后 is_enabled_end：{arm.is_enabled_end}")
        time.sleep(0.01)                                  # 等待电机使能并状态更新完毕
        print(f"arm 模式：{arm.get_mode_arm()}，q = {np.round(arm.joint_state_arm.q, 3).tolist()} rad")
        print(f"end 模式：{arm.get_mode_end()}，q = {np.round(arm.joint_state_end.q, 3).tolist()} rad")
        print("—— 三类控制模式（ControlMode 枚举）与对应整组下发指令——")
        print("模式        枚举值        整组下发指令              下发内容（须先 set_mode）")
        print("MIT        mit          send_mit_*              五个参数 (q, dq, tau, kp, kd)")
        print("POSITION   position     send_position_*         目标关节角 q + 限速 vlim / 限流 flim（缺省取 cfg 配置）")
        print("VELOCITY   velocity     send_velocity_*         目标关节速度 dq")

        print("=============== ② MIT 模式：send_mit_arm / end ======================== ")
        yes = input("【确认】arm 与 end 将切 MIT 模式低增益原地保持，输入 yes 执行，其他输入跳过：").strip() == "yes"
        if yes:
            ret_arm = arm.set_mode_arm(ControlMode.MIT)
            ret_end = arm.set_mode_end(ControlMode.MIT)
            print(f"set_mode_arm / end(MIT) 返回：{ret_arm} / {ret_end}")
            q_cur = np.asarray(arm.joint_state_arm.q, dtype=float)  # MIT 目标 = 当前 q
            arm.send_mit_arm(q=q_cur, dq=np.zeros(arm.n_arm), tau=np.zeros(arm.n_arm), kp=np.full(arm.n_arm, 20.0), kd=np.full(arm.n_arm, 0.5))
            arm.send_mit_end(q=np.asarray(arm.joint_state_end.q, dtype=float), dq=np.zeros(arm.n_end), tau=np.zeros(arm.n_end), kp=np.full(arm.n_end, 20.0), kd=np.full(arm.n_end, 0.5))
            input("按任意键结束 MIT 模式...")
            print(f"arm 保持后 q = {np.round(arm.joint_state_arm.q, 3).tolist()} rad")
            print(f"end 保持后 q = {np.round(arm.joint_state_end.q, 3).tolist()} rad")
        else:
            print("未确认 yes：已跳过 MIT 模式演示。")

        print("=============== ③ 位置模式 POSITION：send_position_arm / end =========== ")
        yes = input("【确认】arm 将回零位形、end 将运动到 -3.0 rad，输入 yes 执行，其他输入跳过：").strip() == "yes"
        if yes:                                           # 确认 yes 才执行
            ret_arm = arm.set_mode_arm(ControlMode.POSITION)  # 切位置模式
            ret_end = arm.set_mode_end(ControlMode.POSITION)
            print(f"set_mode_arm / set_mode_end(POSITION) 返回：{ret_arm} / {ret_end}")
            arm.send_position_arm(arm.zero_arm)           # arm：回零位形
            arm.send_position_end(np.array([-3.0]))       # end：位置指令
            time.sleep(3.0)
            print(f"arm is_in_q_arm(zero_arm)：{arm.is_in_q_arm(arm.zero_arm, tol_q=0.05)}；q = {np.round(arm.joint_state_arm.q, 3).tolist()} rad")
            print(f"end is_in_q_end([-3.0])：{arm.is_in_q_end(np.array([-3.0]), tol_q=0.05)}；q = {np.round(arm.joint_state_end.q, 3).tolist()} rad")
        else:
            print("未确认 yes：已跳过位置模式演示。")

        print("=============== ④ 速度模式 VELOCITY：send_velocity_arm / end ============ ")
        yes = input("【确认】arm 关节 1 将以 0.3 rad/s 点动 1 秒、end 将以 -1.5 rad/s 点动 1 秒，输入 yes 执行，其他输入跳过：").strip() == "yes"
        if yes:
            ret_arm = arm.set_mode_arm(ControlMode.VELOCITY)
            ret_end = arm.set_mode_end(ControlMode.VELOCITY)
            print(f"set_mode_arm / set_mode_end(VELOCITY) 返回：{ret_arm} / {ret_end}")
            dq = np.zeros(arm.n_arm)                      # arm：仅关节 1 +0.3 rad/s 点动 1 秒
            dq[0] = 0.3
            arm.send_velocity_arm(dq)
            arm.send_velocity_end(np.full(arm.n_end, -1.5))    # end：全部关节 -1.5 rad/s 点动 1 秒
            time.sleep(1.0)
            arm.send_velocity_arm(np.zeros(arm.n_arm))         # 先发零速停止，再做任何其他操作
            arm.send_velocity_end(np.zeros(arm.n_end))         # 先发零速停止
            time.sleep(0.1)
            print(f"arm 点动后 q = {np.round(arm.joint_state_arm.q, 3).tolist()} rad（arm 关节 1 应正方向运动 0.3 rad）")
            print(f"end 点动后 q = {np.round(arm.joint_state_end.q, 3).tolist()} rad（end 全部关节应负方向运动 1.5 rad）")
        else:
            print("未确认 yes：已跳过速度模式演示。")

        print("=============== ⑤ end 离散动作 set_action_end：如 open/close/home ====== ")
        yes = input("【确认】end 将依次 open / close / home，输入 yes 执行，其他输入跳过：").strip() == "yes"
        if yes:
            arm.set_action_end("open") 
            time.sleep(3.0) 
            print(f"open 后 q = {np.round(arm.joint_state_end.q, 3).tolist()} rad")
            arm.set_action_end("close")
            time.sleep(3.0)
            print(f"close 后 q = {np.round(arm.joint_state_end.q, 3).tolist()} rad")
            arm.set_action_end("home")
            time.sleep(2.0)
            print(f"home 后 q = {np.round(arm.joint_state_end.q, 3).tolist()} rad")
        else:
            print("未确认 yes：已跳过夹爪离散动作演示。")

        print("=============== ⑥ FSM 互斥说明（本例程未启动运动状态机） ================ ")
        print("运动状态机（FSM）运行期间，电机被独占：arm 侧 enable/disable、set_mode、send_mit/position/velocity、clear_error 等指令类方法会被拦下（warn，不执行）。")
        print("本例程全程未调用 start_motion_fsm，上述指令均能够直通硬件。")

        print("=============== ⑦ 失能收尾 ======================================== ")
        print(f"disable_arm 返回：{arm.disable_arm()}；disable_end 返回：{arm.disable_end()}") 
        print(f"失能后 is_enabled_arm：{arm.is_enabled_arm}，is_enabled_end：{arm.is_enabled_end}")
    print("结束收尾：机械臂已失能，with 退出自动断开总线。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:                # Ctrl+C
        print("\n检测到 Ctrl+C：安全退出。")
