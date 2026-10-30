"""
==============================================================================
例程 09 —— 综合抓取演示（需真机）
==============================================================================
【功能概要】
    1. 连接本体与末端夹爪，启动运动状态机（FSM）并等待 MOTION 态后张开夹爪；
    2. solve_fkine 取当前位姿快照，构造上方点/抓取点，solve_ikine 求关节解并做小幅校验；
    3. set_target_joint 依次运动：到上方点 → 降到抓取点 → 闭爪夹取 → 抬回上方点；
    4. 回 home 位形释放物体，stop_motion_fsm 阻塞安全停机，with 退出自动断连。

【原理简介】
    抓取演示 = 「离线求目标关节角」+「FSM 在线托管运动」两步：
      1. solve_fkine 由当前关节角算出末端位姿，在其快照上平移出「上方点/抓取点」；
         solve_ikine 再把目标位姿反解为关节角（以 home 为初值，解自然贴近 home）；
      2. FSM 启动后，规划线程按目标帧时刻 t 做三次多项式插值限速，控制线程整组下发，
         脚本只需 set_target_joint 写入目标并轮询 is_in_q_arm 到位；
      3. 夹爪是末端独立电机：set_action_end 直接下发固定行程位（open=-5.0 / close=0.0 rad），
         不经 FSM，与本体运动互不影响。

【安全须知】
    - 清空机械臂工作半径内的人员与障碍物，确认急停按钮在手边随时可按；
    - 全程小幅运动：关节目标校验在 home ±0.2 rad 邻域内，配 2 s 三次多项式行程，
      关节峰值速度约 ≤0.15 rad/s（安全规范 ≤0.3 rad/s）；
    - 首次运动前需回车确认；运行中任何异常请立即急停或 Ctrl+C（自动安全停机并断连）。

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（需已连接真机并上电）
    python quick_start/09_pick_and_place.py <型号>

【预期输出】
    依次打印 ① 准备 → ② 规划（含点位表格）→ ③ 抓取 → ④ 收尾 四段日志，
    机械臂完成一次「到上方点 → 下降闭爪 → 抬起 → 回 home 释放」的小幅抓取动作。
==============================================================================
"""

import sys                                                       # 标准库：命令行参数与脚本退出
import time                                                      # 标准库：目标时刻（Unix 秒）与轮询计时
from pathlib import Path                                         # 标准库：跨平台定位脚本上级目录

import numpy as np                                               # 数组运算：位姿偏移与关节偏差幅值

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 把 joyarm_code/ 加入搜索路径
from joyarm_core import MotionState, Pose, joyArm                # 核心库：FSM 状态枚举 / 位姿类型 / 对象入口


# ============================================================
# 一、演示参数：小幅安全运动的偏移量与幅度上限
# ============================================================
GRAB_OFFSET0 = np.array([0.05, 0.0, -0.02])                      # 抓取点初始偏移 (m)：前方 5 cm、下移 2 cm（姿态不变）
LIFT_OFFSET = np.array([0.0, 0.0, 0.05])                         # 上方点相对抓取点的抬升 (m)：+5 cm（先到上方点再竖直下探）
DEV_LIMIT = 0.2                                                  # 关节解偏离 home 的安全上限 (rad)：小幅运动规范


# ============================================================
# 二、命令行与格式化助手：型号解析与表格显示
# ============================================================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并以退出码 1 结束。"""
    if len(sys.argv) != 2:                                       # 必须恰好携带 1 个型号参数
        sys.exit(f"用法：python quick_start/09_pick_and_place.py <型号>")  # 缺参退出，不进入演示
    return sys.argv[1].strip()                                   # 型号名


def fmt_flag(flag):
    """三值标志转可读文本：True→成功、False→失败、None/其他→「—」。"""
    return {True: "成功", False: "失败"}.get(flag, "—")              # None（意外未知）按规范显示「—」


def fmt_vec(vec):
    """一维数组转定宽文本（每元素带符号三位小数），用于表格列对齐。"""
    return "[" + ", ".join(f"{x:+.3f}" for x in np.asarray(vec, dtype=float)) + "]"  # 拼接为 [x, y, z] 形式


# ============================================================
# 三、单点运动：下发关节目标并限时轮询到位
# ============================================================
def move_to(arm, q_target, horizon_s=2.0, timeout_s=6.0):
    """下发单帧关节目标并轮询到位。

    目标时刻取 time.time() + horizon_s（Unix 秒）：main 已切换三次多项式规划器
    single_joint_cubic_traj_planner，规划线程按该时刻插值——小幅运动（≤0.2 rad）
    配 2 s 行程，关节峰值速度约 ≤0.15 rad/s，满足安全限速规范（≤0.3 rad/s）。
    """
    arm.set_target_joint(t=time.time() + horizon_s, q=q_target)  # 单帧关节目标（FSM 规划/控制线程接管）
    deadline = time.monotonic() + timeout_s                      # 到位轮询截止时刻（单调时钟，不受对时影响）
    while time.monotonic() < deadline and not arm.is_in_q_arm(q_target):  # 限时轮询到位（逐关节容差判断）
        time.sleep(0.05)                                         # 50 ms 轮询步进（对齐状态刷新周期量级）
    if not arm.is_in_q_arm(q_target):                            # 超时仍未到位：告警不中断，继续下一步
        print("    警告：限时内未到位，继续下一步（请检查是否被障碍阻挡）")                 # 告警不中断：留操作员现场处置


# ============================================================
# 四、主流程：准备 → 规划 → 抓取 → 收尾
# ============================================================
def main() -> None:
    """综合抓取演示主流程：准备 → 规划 → 抓取 → 收尾（Ctrl+C 可随时安全退出）。"""
    print("=" * 78)                                              # 大节：标题上边框
    print("例程 09 —— 综合抓取演示（需真机）")                                # 大节标题
    print("=" * 78)                                              # 大节：标题下边框
    model = parse_model()                                        # 必填型号参数
    try:
        with joyArm(model) as arm:                               # with 退出自动安全收尾（停机/失能/断连）
            # ---- ① 准备：连接本体/末端 → 切限速规划器 → 启动 FSM → 张开夹爪 ----
            print("-" * 78)                                      # 小节：① 上边框
            print("① 准备：连接本体/末端 → 切限速规划器 → 启动 FSM → 张开夹爪")       # 小节标题
            print("-" * 78)                                      # 小节：① 下边框
            conn_arm = arm.connect_arm()                         # 连接本体电机总线（返回 True/False/None）
            conn_end = arm.connect_end()                         # 连接末端（夹爪）电机总线
            print(f"    总线连接结果：arm={fmt_flag(conn_arm)}，end={fmt_flag(conn_end)}")  # 连接结果（None 显「—」）
            if conn_arm is not True:                             # backend 连接失败仅 warn 不抛错，须显式检查
                raise RuntimeError("arm 总线连接失败，请检查电源与总线后重试")
            if conn_end is not True:                             # end 未连接时夹爪动作会被静默跳过
                raise RuntimeError("end 总线连接失败：夹爪动作将被静默跳过，请检查末端电机与总线")
            arm.set_traj_solver("single_joint_cubic_traj_planner")  # 切三次多项式规划器：按 t 插值限速（默认直传规划器忽略 t）
            input("\n确认机械臂工作半径内无人员与障碍物后，回车开始首次运动：")              # 危险动作前人工确认
            arm.start_motion_fsm()                               # 启动运动状态机：DISABLE → HOME → MOTION 自动迁移
            if arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0) != 1:  # 阻塞等待进入 MOTION（返回 1=到达）
                raise RuntimeError("FSM 未在 10 s 内进入 MOTION 态，请检查电机状态")
            arm.set_action_end("open")                           # 张开夹爪（固定行程位 -5.0 rad），准备抓取
            print("    准备完成：FSM 已进入 MOTION 态，夹爪已张开")             # ① 阶段收尾提示

            # ---- ② 规划：FK 位姿快照 → 构造上方点/抓取点 → IK 求解与小幅校验 ----
            print("-" * 78)                                      # 小节：② 上边框
            print("② 规划：FK 位姿快照 → 构造上方点/抓取点 → IK 求解与小幅校验")       # 小节标题
            print("-" * 78)                                      # 小节：② 下边框
            base = arm.solve_fkine()                             # 当前（home 位形）末端位姿快照（缺省末端帧）
            print(f"    当前末端位置快照：{fmt_vec(base.position)} m")    # 姿态保持不变，仅用位置做平移
            offset = GRAB_OFFSET0.copy()                         # 抓取偏移工作副本（失败时减半，不改动常量）
            for _ in range(5):                                   # 校验不过则等比缩小偏移重解，最多 5 次
                pose_grab = Pose(position=base.position + offset,  # 抓取点位姿 = 快照位置 + 偏移
                                 orientation=base.orientation)     # 姿态与当前一致（纯平移可达）
                pose_above = Pose(position=base.position + offset + LIFT_OFFSET,  # 上方点 = 抓取点再抬 5 cm
                                  orientation=base.orientation)                   # 姿态同上，便于竖直下探
                res_grab = arm.solve_ikine(pose_grab, q0=arm.home_arm)    # 逆解（home 初值 → 解靠近 home）
                res_above = arm.solve_ikine(pose_above, q0=arm.home_arm)  # 上方点逆解（同初值保证两点解连续）
                dev_grab = (float(np.max(np.abs(res_grab.q - arm.home_arm)))    # 抓取点解偏离 home 的最大关节角
                            if res_grab.success else float("inf"))              # 未收敛按无穷大（必减半重试）
                dev_above = (float(np.max(np.abs(res_above.q - arm.home_arm)))  # 上方点解偏离 home 的最大关节角
                             if res_above.success else float("inf"))            # 同上：未收敛按无穷大
                if max(dev_grab, dev_above) < DEV_LIMIT:         # 安全校验：两点均收敛且幅度 <0.2 rad（严于 0.6 rad 设计上限）
                    break                                        # 校验通过：两点均可行且小幅
                offset = offset * 0.5                            # 逆解失败或幅度超限 → 偏移减半重解
            else:
                raise RuntimeError("② 规划失败：小幅邻域内无可行抓取点，请调整初始偏移")
            print(f"    {'点位':<4}{'位置偏移 [x, y, z] m':<30}{'偏离 home 最大 (rad)':<20}IK 收敛")    # 表头
            print(f"    {'上方点':<4}{fmt_vec(offset + LIFT_OFFSET):<30}{dev_above:<20.3f}是")  # 上方点行
            print(f"    {'抓取点':<4}{fmt_vec(offset):<30}{dev_grab:<20.3f}是")                 # 抓取点行

            # ---- ③ 抓取：到上方点 → 降到抓取点 → 闭爪夹取 → 抬回上方点 ----
            print("-" * 78)                                      # 小节：③ 上边框
            print("③ 抓取：上方点 → 下降 → 闭爪夹取 → 抬回上方点")                # 小节标题
            print("-" * 78)                                      # 小节：③ 下边框
            move_to(arm, res_above.q)                            # 先到上方点关节解（接近抓取点，便于竖直下探）
            print("    已到上方点，开始竖直下降到抓取点")                        # 阶段进度提示
            move_to(arm, res_grab.q)                             # 降到抓取点关节解
            arm.set_action_end("close")                          # 闭合夹爪（固定行程位 0.0 rad），夹取物体
            time.sleep(2.5)                                      # 等闭合到位再抬臂：行程 5.0 rad、限速 3.0 rad/s ≈ 1.7 s，取 2.5 s 留余量
            print("    夹取完成，抬回上方点")                              # 阶段进度提示
            move_to(arm, res_above.q)                            # 抬升回上方点（带物体离开抓取位）

            # ---- ④ 收尾：回 home 释放物体 → 安全停机 ----
            print("-" * 78)                                      # 小节：④ 上边框
            print("④ 收尾：回 home 释放物体 → 安全停机")                     # 小节标题
            print("-" * 78)                                      # 小节：④ 下边框
            move_to(arm, arm.home_arm)                           # 回 home 位形（把物体放回起始位置）
            arm.set_action_end("open")                           # 张开夹爪，释放物体
            time.sleep(0.5)                                      # 等待物体脱离夹爪
            arm.stop_motion_fsm()                                # 阻塞安全停机：经 HOME 回零后失能
            print("    收尾完成：FSM 已停机，with 退出时自动断开连接")             # ④ 阶段收尾提示

        # 全部完成横幅（仅在未中断时打印）
        print("=" * 78)                                          # 大节：完成上边框
        print("例程运行完成：已走通「准备 → 规划 → 抓取 → 收尾」完整闭环。")              # 完成标语
        print("=" * 78)                                          # 大节：完成下边框
    except KeyboardInterrupt:                                    # Ctrl+C：with 已先走安全收尾（停机/失能/断连）
        print("\n检测到 Ctrl+C：已中断例程，机械臂经 with 退出路径安全停机并断连。")


if __name__ == "__main__":                                       # 直接运行入口（被导入时不执行）
    main()
