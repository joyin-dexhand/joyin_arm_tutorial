# """
# ==============================================================================
# 例程 09 —— 综合抓取演示（需真机）（尚未完成，注释掉！）
# ==============================================================================
# 【功能概要】
#     1. 连接并使能本体与末端夹爪，yes 确认后启动运动状态机（FSM）并等待 MOTION 态后张开夹爪；
#     2. solve_fkine 取当前位姿快照，构造上方点/抓取点，solve_ikine 求关节解并校验小幅邻域；
#     3. yes 确认后依次运动：到上方点 → 降到抓取点 → 闭爪夹取 → 抬回上方点；
#     4. 回 home 位形释放物体，stop_motion_fsm 阻塞安全停机，with 退出自动断连。

# 【原理简介】
#     抓取演示 = 「离线求目标关节角」+「FSM 在线托管运动」两步；
#     solve_fkine 由当前关节角算出末端位姿，在其快照上平移出「上方点 / 抓取点」，solve_ikine 再把目标位姿反解为关节角（以 home 为初值，解自然贴近 home）；
#     FSM 启动后，规划线程按目标帧时刻 t 做三次多项式插值限速、控制线程整组下发，脚本只需 set_target_joint 写入目标并轮询 is_in_q_arm 到位；
#     夹爪是末端独立电机：set_action_end 直接下发固定行程位（open=-5.0 / close=0.0 rad），不经 FSM，与本体运动互不影响。

# 【安全须知】
#     - 清空机械臂工作半径内的人员与障碍，确认急停按钮在手边随时可按
#     - 全程小幅运动：关节目标校验在 home ±0.2 rad 邻域内，配 2 s 三次多项式行程，关节峰值速度约 ≤0.15 rad/s（安全规范 ≤0.3 rad/s）
#     - 启动 FSM（自动回 home）与开始抓取运动两处危险操作均需输入 yes 确认，其他输入跳过
#     - 运行中任何异常请立即急停或 Ctrl+C（自动安全停机并断连）

# 【环境与运行】
#     # 0. 安装 uv（仅首次）
#     curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
#     irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

#     # 1. 创建虚拟环境并同步依赖
#     cd <仓库路径>/joyarm_code
#     uv sync
#     source .venv/bin/activate

#     # 2. 运行本脚本（需连接对应的机械臂）
#     python quick_start/09_pick_and_place.py <型号>  # 如：rebot_arm_b601_dm

# 【预期输出】
#     ① 准备 → ② 规划（含上方点 / 抓取点的偏移与偏离 home 幅值）→ ③ 抓取 → ④ 收尾 四段日志；
#     机械臂完成一次「到上方点 → 下降闭爪 → 抬起 → 回 home 释放」的小幅抓取动作。
# ==============================================================================
# """

# import sys
# import time
# from pathlib import Path

# import numpy as np

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# from joyarm_core import MotionState, Pose, joyArm, listArm


# # ============================================================
# # 一、演示参数：小幅安全运动的偏移量与幅度上限
# # ============================================================
# GRAB_OFFSET = np.array([0.05, 0.0, -0.02])                       # 抓取点偏移 (m)：前方 5 cm、下移 2 cm（姿态不变）
# LIFT_OFFSET = np.array([0.0, 0.0, 0.05])                         # 上方点相对抓取点的抬升 (m)：+5 cm（先到上方点再竖直下探）
# DEV_LIMIT = 0.2                                                  # 关节解偏离 home 的安全上限 (rad)：小幅运动规范


# # ============================================================
# # 二、命令行型号参数：解析必填的 <型号>，缺失时打印用法并以退出码 1 结束
# # ============================================================
# def parse_model() -> str:
#     """解析必填的命令行型号参数；缺失时打印用法并列出可用型号（listArm），以退出码 1 结束。"""
#     if len(sys.argv) != 2:                                       # 必须恰好携带 1 个型号参数
#         print(f"用法：python quick_start/09_pick_and_place.py <型号>")  # 缺参：先打印用法
#         listArm()                                                # 顺带列出全部可用型号
#         sys.exit(1)                                              # 退出码 1，不进入演示
#     return sys.argv[1].strip()                                   # 型号名


# # ============================================================
# # 三、单点运动：下发关节目标并限时轮询到位（主流程复用 4 次，保留为助手）
# # ============================================================
# def move_to(arm, q_target, horizon_s=2.0, timeout_s=6.0):
#     """下发单帧关节目标（到达时刻 = 当前 + horizon_s）并限时轮询到位；超时只告警不中断。"""
#     arm.set_target_joint(t=time.time() + horizon_s, q=q_target)  # 单帧关节目标（FSM 规划/控制线程接管）
#     deadline = time.monotonic() + timeout_s                      # 单调时钟，不受对时影响
#     while time.monotonic() < deadline and not arm.is_in_q_arm(q_target):  # 限时轮询到位（逐关节容差判断）
#         time.sleep(0.05)
#     if not arm.is_in_q_arm(q_target):
#         print("警告：限时内未到位，继续下一步（请检查是否被障碍阻挡）")  # 告警不中断：留操作员现场处置


# # ============================================================
# # 四、主流程：准备 → 规划 → 抓取 → 收尾
# # ============================================================
# def main() -> None:
#     """综合抓取演示主流程：准备 → 规划 → 抓取 → 收尾（两处 yes 确认门，Ctrl+C 可随时安全退出）。"""
#     model = parse_model()                                        # 必填型号参数
    
#     with joyArm(model) as arm:                                   # with 退出自动安全收尾（停机/失能/断连）

#         print("=============== ① 准备：连接与使能 → 切限速规划器 → 启动 FSM → 张开夹爪 =========== ")
#         conn_arm = arm.connect_arm()                             # 连接本体电机总线：True=已连接、False=未连接、None=未知
#         conn_end = arm.connect_end()                             # 连接末端（夹爪）电机总线
#         print(f"总线连接结果：arm={conn_arm}，end={conn_end}")
#         if conn_arm is not True:                                 # backend 连接失败仅 warn 不抛错，须显式检查
#             print("arm 总线连接失败：请检查电源与总线后重试。")
#             return                                               # with 退出自动安全收尾
#         if conn_end is not True:                                 # end 未连接则夹爪动作无法执行
#             print("end 总线连接失败：请检查末端电机与总线后重试。")
#             return
#         ret = arm.enable_end()                                   # 使能末端电机：FSM 只使能本体关节，夹爪须单独显式使能
#         print(f"enable_end 返回：{ret}；is_enabled_end：{arm.is_enabled_end}")
#         time.sleep(0.01)                                         # 等待电机使能并状态更新完毕
#         ok = arm.set_traj_solver("single_joint_cubic_traj_planner")  # 切三次多项式规划器：按到达时刻 t 插值限速（默认直传规划器忽略 t）
#         print(f"set_traj_solver 返回：{ok}")
#         yes = input("【确认】启动 FSM 后机械臂将自动回 home（关节 2/3 各移动约 1 rad），"
#                     "确认工作半径内无人员与障碍物后输入 yes 执行，其他输入跳过：").strip() == "yes"
#         if not yes:                                              # 危险操作确认门①：FSM 启动即自动回 home
#             print("未确认 yes：已跳过演示。")
#             return
#         arm.start_motion_fsm()                                   # 启动运动状态机：DISABLE → HOME → MOTION 自动迁移
#         if arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0) != 1:  # 阻塞等待进入 MOTION（1=到达、0=超时）
#             print("FSM 未在 10 s 内进入 MOTION 态，请检查电机状态。")
#             return
#         arm.set_action_end("open")                               # 张开夹爪（固定行程位 -5.0 rad），准备抓取
#         print("准备完成：FSM 已进入 MOTION 态，夹爪已张开")

#         print("=============== ② 规划：FK 快照 → 上方点/抓取点 → IK 求解与校验 =============== ")
#         base = arm.solve_fkine()                                 # 当前（home 位形）末端位姿快照（缺省末端帧）
#         print(f"当前末端位置快照：{np.round(base.position, 3).tolist()} m（姿态不变，仅位置平移）")
#         pos_grab = base.position + GRAB_OFFSET                   # 抓取点位置 = 当前位置 + 偏移
#         pose_grab = Pose(position=pos_grab, orientation=base.orientation)  # 抓取点位姿（姿态保持不变）
#         pose_above = Pose(position=pos_grab + LIFT_OFFSET, orientation=base.orientation)  # 上方点 = 抓取点再抬 5 cm
#         res_grab = arm.solve_ikine(pose_grab, q0=arm.home_arm)   # 抓取点逆解（home 初值 → 解靠近 home）
#         res_above = arm.solve_ikine(pose_above, q0=arm.home_arm)  # 上方点逆解（同初值保证两点解连续）
#         if not (res_grab.success and res_above.success):         # 任一点未收敛即放弃抓取
#             print(f"IK 未收敛（grab={res_grab.success}，above={res_above.success}）：跳过演示，安全收尾退出。")
#             return
#         dev_grab = float(np.max(np.abs(res_grab.q - arm.home_arm)))  # 抓取点解偏离 home 的最大关节角
#         dev_above = float(np.max(np.abs(res_above.q - arm.home_arm)))
#         if max(dev_grab, dev_above) >= DEV_LIMIT:                # 安全校验：两点均须落在 home ±0.2 rad 邻域内
#             print(f"关节解偏离 home 最大 {max(dev_grab, dev_above):.3f} rad，超出 ±{DEV_LIMIT} 安全邻域：跳过演示，安全收尾退出。")
#             return
#         print(f"上方点：位置偏移 {np.round(GRAB_OFFSET + LIFT_OFFSET, 3).tolist()} m，偏离 home 最大 {dev_above:.3f} rad，IK 收敛")
#         print(f"抓取点：位置偏移 {np.round(GRAB_OFFSET, 3).tolist()} m，偏离 home 最大 {dev_grab:.3f} rad，IK 收敛")

#         print("=============== ③ 抓取：上方点 → 下降闭爪 → 抬回上方点 ======================= ")
#         yes = input("【确认】开始小幅抓取运动（到上方点 → 下降闭爪 → 抬回上方点），输入 yes 执行，其他输入跳过：").strip() == "yes"
#         if yes:                                                  # 危险操作确认门②：未确认跳过抓取、直接收尾
#             move_to(arm, res_above.q)                            # 先到上方点（接近抓取点，便于竖直下探）
#             print("已到上方点，竖直下降到抓取点")
#             move_to(arm, res_grab.q)                             # 降到抓取点
#             arm.set_action_end("close")                          # 闭合夹爪（固定行程位 0.0 rad），夹取物体
#             time.sleep(2.5)                                      # 等闭合到位再抬臂：行程 5.0 rad、限速 3.0 rad/s ≈ 1.7 s，取 2.5 s 留余量
#             print("夹取完成，抬回上方点")
#             move_to(arm, res_above.q)                            # 抬升回上方点（带物体离开抓取位）
#         else:
#             print("未确认 yes：已跳过抓取，直接收尾。")

#         print("=============== ④ 收尾：回 home 释放物体 → 安全停机 ======================= ")
#         move_to(arm, arm.home_arm)                               # 回 home 位形（把物体放回起始位置）
#         arm.set_action_end("open")                               # 张开夹爪，释放物体
#         time.sleep(0.5)                                          # 等待物体脱离夹爪
#         arm.stop_motion_fsm()                                    # 阻塞安全停机：经 HOME 回零后失能
#         print("收尾完成：FSM 已停机，with 退出时自动断开连接")

#     print("=============== 完成：已走通「准备 → 规划 → 抓取 → 收尾」完整闭环。 ================ ")  # 仅在未中断时打印


# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:                                    # Ctrl+C：with 已先走安全收尾（停机/失能/断连）
#         print("\n检测到 Ctrl+C：已中断例程，机械臂经 with 退出路径安全停机并断连。")
