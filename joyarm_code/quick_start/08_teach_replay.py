# """
# ==============================================================================
# 例程 08 —— 拖动示教与轨迹复现（需真机）（尚未完成，注释掉！）
# ==============================================================================
# 【功能概要】
#     1. connect_arm 连接真机，输入 yes 确认后 start_motion_fsm 启动状态机，wait_for_motion_state 等待自动迁移至 MOTION 正常运动态；
#     2. 输入 yes 确认后 set_event(DAMPING) 切入 DAMP 阻尼态，用手缓慢拖动机械臂示教，以 50 Hz 只读 joint_state_arm 缓存录制 4 秒（约 200 帧）；
#     3. set_event(NORMAL) 切回 MOTION 态，输入 yes 确认后录制序列每 5 帧取 1 帧作路点，set_target_joint 以 10 Hz 逐点下发复现轨迹；
#     4. stop_motion_fsm 阻塞安全停机（回零失能），with 退出自动断连；未确认 yes 的危险环节跳过并安全收尾。

# 【原理简介】
#     拖动示教的核心是 DAMP 阻尼态：FSM 每周期下发 MIT 帧（kp=0、kd=阻尼系数 damping_kd、tau=重力补偿率 damping_k_gravity × 重力矩），电机如带阻尼的铰链般顺从外力可徒手拖动；
#     录制端只读 joint_state_arm 后台缓存（后台线程持续维护、不额外占用总线），按绝对时刻卡准采样节奏，采样率稳定在 50 Hz；
#     回放端逐路点 set_target_joint 写入目标帧：默认激活的 single_joint_direct_traj_planner 把帧内 q/dq/ddq/tau 直传、忽略 t，10 Hz 连续刷新目标即成轨迹流；
#     若用 set_traj_solver 切到三次多项式规划器 single_joint_cubic_traj_planner，则 t 生效、按到达时刻平滑插值跟帧。

# 【安全须知】
#     - 清空机械臂工作半径内的人员与障碍物，机械臂稳固放置后再上电
#     - 全程值守，急停按钮或电源开关随手可及，异常立即断电
#     - 拖动与回放均保持缓慢小幅：建议各关节相对 home 位形偏移不超过 0.2 rad
#     - 阻尼态下勿快速甩动关节，也勿松手任其自由摆动

# 【环境与运行】
#     # 0. 安装 uv（仅首次）
#     curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
#     irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

#     # 1. 创建虚拟环境并同步依赖
#     cd <仓库路径>/joyarm_code
#     uv sync
#
#     # 2. 运行本脚本（需连接对应的机械臂）
#     uv run python quick_start/08_teach_replay.py <型号>  # 如：rebot_arm_b601_dm

# 【预期输出】
#     连接结果与 FSM 状态迁移 → 录制进度与帧数 → 回放路点进度与终点关节角；
#     未确认 yes 的环节跳过，结束前机械臂回零失能，with 退出自动断连。
# ==============================================================================
# """

# from pathlib import Path
# import sys
# import time

# import numpy as np

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# from joyarm_core import MotionState, SafetyEvent, joyArm, listArm


# # ============================================================
# # 一、运行参数：录制与回放的频率 / 时长常量
# # ============================================================
# RECORD_HZ = 50        # 录制采样频率（Hz）：每秒记录 50 帧关节角
# RECORD_SEC = 4.0      # 录制时长（秒）：总帧数 ≈ 50 × 4 = 200 帧
# REPLAY_HZ = 10        # 回放路点下发频率（Hz）：每 0.1 s 下发一路点
# REPLAY_AHEAD = 0.15   # 路点目标到达时刻提前量（秒；仅 cubic 规划器生效，见 ③ 回放注释）


# # ============================================================
# # 二、命令行参数解析：必填型号，缺失打印用法、列出可用型号并以退出码 1 结束
# # ============================================================
# def parse_model() -> str:
#     """解析必填的命令行型号参数；缺失时打印用法并列出可用型号（listArm），以退出码 1 结束。"""
#     if len(sys.argv) != 2:                              # 必须恰好携带 1 个型号参数
#         print(f"用法：uv run python quick_start/08_teach_replay.py <型号>")  # 缺参：先打印用法
#         listArm()                                       # 顺带列出全部可用型号
#         sys.exit(1)                                     # 退出码 1，不进入演示
#     return sys.argv[1].strip()                          # 型号名


# # ============================================================
# # 三、主流程：连接启动 → 阻尼示教与录制 → 轨迹回放 → 安全收尾
# # ============================================================
# def main() -> None:
#     """拖动示教与轨迹复现主流程（真机，三处 yes 确认）。"""
#     model = parse_model()                                 # 必填型号参数

#     with joyArm(model) as arm:                            # with 退出（含异常）自动安全收尾 close()

#         print("=============== ① 连接真机并启动运动状态机 ================================ ")
#         ret = arm.connect_arm()                           # 连接本体 arm 电机总线（成功后自动设 cfg 默认模式）
#         print(f"connect_arm 返回：{ret}；is_connected_arm：{arm.is_connected_arm}")
#         if ret is not True:                               # 未连接成功（False 失败 / None 意外）则放弃后续动作
#             print("arm 总线连接失败，请检查电源与总线后重试。")
#             return                                        # with 退出自动安全收尾
#         print(f"型号 {model}：{arm.n_arm} 个关节，home = {np.round(arm.home_arm, 3).tolist()} rad")
#         yes = input("【确认】启动 FSM 将自动回 home 位形并使能运动，确认工作半径内无人无障碍，输入 yes 继续，其他输入退出：").strip() == "yes"
#         if not yes:                                       # 未确认 yes：不启动 FSM，直接收尾退出
#             print("未确认 yes：不启动 FSM，安全收尾退出。")
#             return
#         arm.start_motion_fsm()                            # 启动 FSM：复位缓存后自动 DISABLE → HOME → MOTION
#         if arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0) != 1:  # 阻塞等待：1=到达 / 0=超时
#             print("10 秒内未进入 MOTION 正常运动态，执行收尾退出。")
#             arm.stop_motion_fsm()                         # 阻塞回零失能
#             return
#         print(f"已进入 MOTION 正常运动态（当前态 = {arm.get_current_motion_state().name}）")

#         print("=============== ② 阻尼态拖动示教与录制 ================================== ")
#         yes = input("【确认】切入 DAMP 阻尼态并立即开始 4 秒录制（确认后马上用手缓慢拖动机械臂），输入 yes 执行，其他输入跳过：").strip() == "yes"
#         if not yes:                                       # 未确认 yes：示教与回放一并跳过，收尾退出
#             print("未确认 yes：已跳过示教录制与轨迹回放，执行收尾。")
#             arm.stop_motion_fsm()                         # 阻塞回零失能
#             return
#         arm.set_event(SafetyEvent.DAMPING)                # 阻尼防护事件 → FSM 目标态 DAMP（STATE_FROM 映射）
#         if arm.wait_for_motion_state(MotionState.DAMP, timeout=5.0) != 1:  # 阻塞等待：1=到达 / 0=超时
#             print("未进入 DAMP 阻尼态，执行收尾退出。")
#             arm.stop_motion_fsm()                         # 阻塞回零失能
#             return
#         print("已进入 DAMP 阻尼态：立即用手握住臂杆缓慢拖动（重力补偿程度由 cfg 的 damping_k_gravity 决定）")
#         frames, reported = [], 0                          # 录制帧列表 / 已打印的整秒进度
#         t0 = tick = time.perf_counter()                   # 录制起点 = 首个采样点（perf_counter 单调递增，适合测时长）
#         while time.perf_counter() - t0 < RECORD_SEC:      # 按绝对时刻卡准 50 Hz 采样节奏，只读后台缓存不发总线帧
#             frames.append(list(arm.joint_state_arm.q))    # 采样一帧关节角（后台缓存为活引用 → 拷贝成 list）
#             if int(time.perf_counter() - t0) > reported:  # 每进入新的一整秒打印一次进度
#                 reported = int(time.perf_counter() - t0)
#                 print(f"录制中 {reported}/{RECORD_SEC:.0f} 秒")
#             tick += 1.0 / RECORD_HZ                       # 下一采样点的绝对时刻推进一个周期
#             time.sleep(max(0.0, tick - time.perf_counter()))  # 睡到下一采样点（已滞后则不睡，追回节奏）
#         record = np.array(frames)                         # (帧数, n_arm)：50 Hz × 4 s ≈ 200 帧
#         print(f"录制完成 {record.shape[0]} 帧，起点 q = {np.round(record[0], 3).tolist()} rad")

#         print("=============== ③ 轨迹回放 ======================================== ")
#         arm.set_event(SafetyEvent.NORMAL)                 # 正常事件 → 目标态 MOTION（经 HOME 中转，途中先回 home 位形）
#         if arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0) != 1:  # 阻塞等待：1=到达 / 0=超时
#             print("10 秒内未回到 MOTION 正常运动态，执行收尾退出。")
#             arm.stop_motion_fsm()                         # 阻塞回零失能
#             return
#         yes = input("【确认】回放将主动复现示教轨迹（首个路点相对 home 可有小幅跳变），输入 yes 执行，其他输入跳过：").strip() == "yes"
#         if not yes:                                       # 未确认 yes：跳过回放，直接收尾
#             print("未确认 yes：已跳过轨迹回放。")
#         else:
#             waypoints = record[::RECORD_HZ // REPLAY_HZ]  # 每 5 帧取 1 帧作路点（50 Hz → 10 Hz 抽样，约 40 个）
#             print(f"开始回放：共 {len(waypoints)} 个路点，{REPLAY_HZ} Hz 下发……")
#             # 默认 direct 规划器直传帧内关节域（回放路点仅填 q）、忽略 t；t（当前时刻 + REPLAY_AHEAD）仅切到 cubic 后生效
#             for i, q_wp in enumerate(waypoints, start=1):
#                 arm.set_target_joint(t=time.time() + REPLAY_AHEAD, q=q_wp)  # 写入单帧关节目标（原子通道）
#                 if i % 10 == 0 or i == len(waypoints):    # 每 10 个路点（及最后一个）打印进度
#                     print(f"回放进度 {i}/{len(waypoints)} 路点")
#                 time.sleep(1.0 / REPLAY_HZ)
#             time.sleep(REPLAY_AHEAD)                      # 末路点后的收尾余量（cubic 规划器下恰为到达时刻）
#             print(f"回放完成，当前 q = {np.round(arm.joint_state_arm.q, 3).tolist()} rad")

#         print("=============== ④ 安全收尾 ======================================== ")
#         arm.stop_motion_fsm()                             # 阻塞等待回零到位后失能（事件更严重时直接失能）
#         print("已安全停机（回零失能完成），with 退出后自动断开总线。例程结束。")


# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:                             # Ctrl+C
#         print("\n检测到 Ctrl+C：已安全退出（停 FSM → 失能 → 断连）。")
