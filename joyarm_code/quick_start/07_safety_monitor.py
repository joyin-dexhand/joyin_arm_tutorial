"""
==============================================================================
例程 07 —— 安全监视器（需真机）
==============================================================================
【功能概要】
    1. 按顺序「先 start_motion_fsm 再 start_monitor」，并说明原因；
    2. 定义自定义超速规则 overspeed 并经 set_monitor_rules 挂接（可传单条或列表）；
    3. 输入 yes 确认后以一次快速小幅运动触发规则，观察事件升为 DAMPING、状态机转入 DAMP 阻尼；
    4. 输入 yes 确认后 set_event(NORMAL) 手动复位：FSM 经 HOME 自动回 MOTION；
    5. 按 stop_monitor() → stop_motion_fsm() 顺序退出收尾，with 退出再兜底。

【原理简介】
    监视器是独立后台线程，按 monitor_hz（由 cfg.joyarm 配置）周期检查：只读关节状态缓存、FSM 状态与事件，不下发指令；
    内置规则（故障码 ≥ 2 → DAMPING；断连 / 连续 0.1 s 无新鲜数据 → DISABLE）与自定义规则的安全状态处置意见合并取最严重者，只升不降；
    命中后经 set_event 改写 FSM 目标状态；
    监视器只判定、FSM 才执行（故先启 FSM 再启 Monitor）。

【安全须知】
    - 运行前清空机械臂工作半径内的人员与障碍物，备好急停 / 断电
    - 本例让关节 1 快速小幅摆动约 0.25 rad 刻意触发超速防护、恢复时自动运动回 home，两处均须输入 yes 确认
    - Ctrl+C 随时中断，with 退出自动安全收尾

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code && uv sync

    # 2. 运行本脚本（需连接对应的机械臂）
    uv run python quick_start/07_safety_monitor.py <型号>  # 如：rebot_arm_b601_dm

【预期输出】
    FSM 启动与规则挂接结果 → yes 确认后快速运动、状态表出现 DAMPING / DAMP 行 → yes 确认后复位回 MOTION → 按序安全收尾退出。
==============================================================================
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from joyarm_core import MotionState, SafetyEvent, joyArm, listArm

DQ_LIMIT = 0.5  # 超速阈值（rad/s）：任一本体关节 |dq| 超过即判定超速
TRIG_Q = 0.25  # 触发运动的关节 1 位置偏移（rad，约 14°，小幅摆动）
TRIG_T = 0.2    # 触发运动的目标时长（s）：名义峰值速度 1.5×Δq/T ≈ 1.9 rad/s，越过阈值


# ============================================================
# 一、命令行参数解析：必填型号，缺失打印用法并以退出码 1 结束
# ============================================================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并列出可用型号（listArm），以退出码 1 结束。"""
    if len(sys.argv) != 2:                                # 必须恰好携带 1 个型号参数
        print(f"用法：uv run python quick_start/07_safety_monitor.py <型号>")  # 缺参：先打印用法
        listArm()                                         # 顺带列出全部可用型号
        sys.exit(1)                                       # 退出码 1，不进入演示
    return sys.argv[1].strip()                            # 型号名

# ============================================================
# 二、打印格式化助手：表格列按终端显示宽度对齐
# ============================================================
def pad(s: str, width: int, align: str = "left") -> str:
    """按终端显示宽度补空格使表格列对齐（全角字符计 2 列，超宽不截断）。"""
    w = sum(2 if ord(c) > 0x2E7F else 1 for c in str(s))   # 显示宽：全角 2 列、半角 1 列
    gap = " " * max(width - w, 0)                          # 缺口 = 目标宽 − 显示宽（不为负）
    return gap + s if align == "right" else s + gap

# ============================================================
# 三、自定义监视规则：任一本体关节速度越限则提出阻尼防护意见
# ============================================================
def overspeed(state_arm, state_end, motion, event):
    """自定义超速规则：任一本体关节 |dq| > DQ_LIMIT → 返回 DAMPING，否则 None。

    参数由监视线程每周期只读传入：state_arm / state_end 为状态缓存，motion / event 为 FSM 当前状态与安全事件；规则内抛异常会被吞掉并告警。
    """
    if (state_arm is not None                              # 判空：数据不新鲜时监视线程传 None
            and np.all(np.isfinite(state_arm.dq))          # 防脏数据：dq 含 NaN/inf 时本周期放弃判定
            and np.any(np.abs(state_arm.dq) > DQ_LIMIT)):  # 任一关节速度越限 → 超速
        return SafetyEvent.DAMPING                         # 超速意见：阻尼防护（FSM 将转入 DAMP 软停）
    return None                                            # 无意见：本周期不触发

# ============================================================
# 四、演示主流程：启动 FSM 与监视器 → 挂规则 → yes 确认触发 → yes 确认恢复 → 安全收尾
# ============================================================
def main() -> None:
    """监视器演示主流程：连接 → 先启 FSM → 挂规则并启监视器 → yes 确认触发观察 → yes 确认恢复 → 收尾。"""
    model = parse_model()                                       # 必填型号参数

    with joyArm(model) as arm:                                  # with 退出自动安全收尾（失能 + 断连）
        # logging.basicConfig(level=logging.INFO, format="%(message)s")  # 运行期 info（FSM 状态迁移等）单行显示

        if arm.connect_arm() is not True:                       # 连接本体总线（监视仅涉 arm 侧）
            print("arm 总线连接失败：请检查电源与总线后重试")
            return                                              # with 退出仍会执行安全收尾

        print("=============== ① 启动 FSM：先 start_motion_fsm 再 start_monitor ============ ")
        print("· 规则的第 3 参 motion 直接消费 FSM 当前状态；规则命中后经 set_event 改写 FSM 目标状态，")
        print("  防护动作由 FSM 线程执行——监视器只判定、不直接驱动电机，故需要先启动 FSM 再启动监视器。")
        input("\n>>> 即将启动 FSM：自动回 home；确认工作半径内无人无障碍后 Enter 继续……")
        arm.start_motion_fsm()                                  # 启动状态机：DISABLE → HOME → MOTION 自动迁移
        if arm.wait_for_motion_state(MotionState.MOTION, timeout=5.0) != 1:  # 1=到达 / 0=超时
            print("FSM 5 秒内未进入 MOTION 态，无法继续演示，直接收尾退出。")
            return
        print(f"FSM 已到达 MOTION 态（当前安全事件 {arm.get_event().name}）\n")

        print("=============== ② 自定义规则：四参数签名与 set_monitor_rules ================ ")
        print("规则签名 rule(state_arm, state_end, motion, event) → SafetyEvent | None：")
        rows = [                                                # (参数名, 含义)
            ("state_arm", "本体关节状态缓存（不新鲜或未读到时为 None，规则内须判空）"),
            ("state_end", "末端状态缓存（同上；本例未连接 end ，恒为 None）"),
            ("motion", "FSM 当前运动状态（MotionState，如 MOTION / DAMP）"),
            ("event", "当前安全事件（SafetyEvent，如 NORMAL / DAMPING）"),
        ]
        for name, desc in rows:
            print(f"  {pad(name, 12)}{desc}")
        print("· 返回 None = 无意见；返回 SafetyEvent = 命中，与内置规则（故障码 ≥ 2 → DAMPING；断连 / 连续 0.1 s 无新鲜数据 → DISABLE）合并取最严重者，只升不降。")
        arm.set_monitor_rules(overspeed)                        # 规则挂接：传单条或规则列表（整表替换）
        arm.start_monitor()                                     # 启动监视线程
        monitor_hz = arm.get_config()["joyarm"]["MotionFSM"]["monitor_hz"]
        print(f"· 规则已挂接、监视器已启动：monitor_hz = {monitor_hz:g} Hz\n")

        print("=============== ③ 触发演示：yes 确认后关节 1 快速摆动 → 观察超速防护 ========= ")
        if not arm.set_traj_solver("single_joint_cubic_traj_planner"):  # 热切换 traj 域规划器
            print("· 切换三次多项式规划器失败（保持直传），仍继续尝试触发。")
        q_trig = np.array(arm.home_arm, dtype=float)            # 触发目标：home 基准仅动关节 1
        q_trig[0] += TRIG_Q
        yes = input(f"【确认】关节 1 将快速摆动 {TRIG_Q} rad 刻意触发超速防护（触发后进入 DAMP 阻尼状态），输入 yes 执行，其他输入跳过：").strip() == "yes"
        if yes:                                                 # 确认 yes 才执行危险触发
            arm.set_target_joint(time.time() + TRIG_T, q_trig)  # 单帧关节目标：0.2 s 内到达
            print(pad("耗时s", 8) + pad("安全事件", 12) + pad("运动状态", 12) + "max|dq| (rad/s)")
            t0 = time.time()
            triggered = False                                   # 是否观察到事件升为 DAMPING
            while time.time() - t0 <= 5.0:                      # 最多观察 5 s（监视器 50 Hz，远快于本表）
                snap = arm.joint_state_arm                      # 后台只读缓存（不发总线读帧）
                dq_max = "—" if snap is None else f"{np.max(np.abs(snap.dq)):.2f}"
                print(pad(f"{time.time() - t0:5.1f}", 8) + pad(arm.get_event().name, 12) + pad(arm.get_current_motion_state().name, 12) + dq_max)
                if arm.get_event() is SafetyEvent.DAMPING:      # 事件只升不降，触发后保持
                    triggered = True
                    break
                time.sleep(0.2)                                 # 0.2 s 刷新一次表格
            if triggered:
                print("· 已触发：事件升为 DAMPING、状态机转入 DAMP。")
            else:
                print("· 5 秒内未触发：反馈速度未越限或跟踪滞后，可增大 TRIG_Q、缩短 TRIG_T 后重试。")
        else:
            print("未确认 yes：跳过触发演示。")

        print("=============== ④ 恢复：set_event(NORMAL) 手动复位 ========================== ")
        if arm.get_event() is SafetyEvent.NORMAL:               # 未触发防护（跳过或未越限）：无需恢复
            print("当前安全事件 NORMAL：未触发防护，无需恢复。")
        else:
            print(f"当前安全事件 {arm.get_event().name}、运动状态 {arm.get_current_motion_state().name}：")
            print("· 勿强行拖动机械臂，先排查并消除触发原因，再手动复位。")
            yes = input("【确认】复位后 FSM 经 HOME 自动回 MOTION（机械臂将运动回 home），输入 yes 执行，其他输入跳过：").strip() == "yes"
            if yes:                                             # 确认 yes 才执行恢复
                arm.set_monitor_rules([])                       # 撤下超速规则
                arm.set_event(SafetyEvent.NORMAL)               # 手动复位，继续执行原任务（q_trig[0] += TRIG_Q）
                ok = arm.wait_for_motion_state(MotionState.MOTION, timeout=5.0)  # DAMP → HOME → MOTION：1=到达 / 0=超时
                print(f"恢复结果：{ok}（1=已回 MOTION、0=超时）；当前状态 {arm.get_current_motion_state().name}，安全事件 {arm.get_event().name}")
            else:
                print("未确认 yes：跳过恢复")

        print("=============== ⑤ 收尾：stop_monitor() → stop_motion_fsm() ================== ")
        print("· 收尾规则：当前安全事件不高于 ZEROING → 先回零位再失能；若更严重（如 DAMPING）→ 直接失能停机。")
        input(f"【确认】输入任意内容执行结束收尾：").strip()
        arm.stop_monitor()                                      # 停监视线程
        arm.stop_motion_fsm()                                   # 阻塞安全停机
        print("运行结束完成，with 退出将再次执行一次兜底收尾（失能 + 断连）。")


if __name__ == "__main__":
    try:
        main()                  # 主流程
    except KeyboardInterrupt:   # Ctrl+C
        print("\n检测到 Ctrl+C：已安全退出。")
