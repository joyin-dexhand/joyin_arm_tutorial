"""
==============================================================================
例程 07 —— 安全监视器（需真机）
==============================================================================
【功能概要】
    1. 聚焦安全监视器核心用法：按推荐顺序「先 start_motion_fsm 再 start_monitor」启动，并说明这样排序的原因；
    2. 定义自定义超速规则 overspeed 并用 set_monitor_rules 挂接（单条 / 列表两种传法）；
    3. 用一次快速小幅运动触发规则，观察事件升为 DAMPING、状态机转入 DAMP 阻尼防护；
    4. 打印恢复流程说明：处理原因后 set_event(NORMAL)，FSM 经 HOME 回 MOTION；
    5. 按 stop_monitor() → stop_motion_fsm() 顺序收尾，with 退出再兜底一次。

【原理简介】
    安全监视器是独立后台线程，按配置 joyarm.MotionFSM.monitor_hz（本型号 50 Hz）的频率周期性巡查，每个周期只读地采集关节状态缓存（不新鲜时为 None）、FSM 当前状态与当前事件，绝不下发指令；
    内置规则（故障码 ≥ 2 → DAMPING；断连 / 连续 0.1 s 无新鲜数据 → DISABLE）与自定义规则的意见合并时取最严重者，且「只升不降」；
    命中后经 set_event 改写 FSM 目标状态（DAMPING → DAMP），由 FSM 执行防护（DAMP 态每个周期下发 MIT 纯阻尼 + 重力补偿软停）；
    监视器只判定、FSM 才执行——FSM 未运行时防护不落地，故先启 FSM。

【安全须知】
    - 运行前清空机械臂工作半径内的人员与障碍物，手边备好急停（急停按钮 / 断电）
    - 本例将让关节 1 快速小幅摆动约 0.25 rad（瞬时速度 > 1.5 rad/s）以刻意触发超速防护，触发后机械臂自动进入 DAMP 阻尼软停；首次运动前会暂停等待人工确认
    - 触发防护后勿强行拖动机械臂：先排查原因，再按脚本打印的恢复流程操作；全程可 Ctrl+C 中断，with 退出会自动安全收尾（不回零、直接失能）

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（需已连接真机并上电）
    python quick_start/07_safety_monitor.py <型号>

【预期输出】
    各阶段中文提示与对齐表格；快速运动后表格出现 DAMPING / DAMP 行（同时打印 FSM 状态迁移 info 日志），随后打印恢复流程说明并按序安全收尾退出。
==============================================================================
"""

import logging      # 日志配置：显示 FSM 状态迁移、监视器启停等 info 日志
import sys          # sys.path 注入：未安装 joyarm_core 时也能直接运行本脚本
import time         # 计时：触发运动的目标时刻与轮询周期
import unicodedata  # 计算字符串的终端显示宽度：表格中文列对齐
from pathlib import Path  # 路径定位：找到本脚本上一级的 joyarm_code/ 目录

import numpy as np  # 数组运算：超速判定 |dq| 与目标位形构造

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 把 joyarm_code/ 加入模块搜索路径
from joyarm_core import MotionState, SafetyEvent, joyArm  # # JoyArm 入口 + 状态机 / 安全事件枚举

DQ_LIMIT = 1.5      # 超速阈值（rad/s）：任一本体关节 |dq| 超过即判定超速
TRIG_DQ = 0.25      # 触发运动的关节 1 位置偏移（rad，约 14°，属小幅摆动）
TRIG_T = 0.2        # 触发运动的目标时长（s）：0.25 rad / 0.2 s 名义峰值速度约 1.9 rad/s


# ============================================================
# 一、命令行解析：必填型号参数，缺参打印用法并以退出码 1 结束
# ============================================================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并以退出码 1 结束。"""
    if len(sys.argv) != 2:                                # 必须恰好携带 1 个型号参数
        sys.exit(f"用法：python quick_start/07_safety_monitor.py <型号>")  # 缺参退出，不进入演示
    return sys.argv[1].strip()                            # 型号名


# ============================================================
# 二、自定义监视规则：任一本体关节速度越限即提出阻尼防护意见
# ============================================================
def overspeed(state_arm, state_end, motion, event):
    """自定义监视规则：任一本体关节 |dq| > 1.5 rad/s → 阻尼防护 DAMPING，否则 None。

    规则签名固定为四参数（均只读，由监视线程每个周期传入；规则内抛异常会被吞掉并告警）：
      state_arm / state_end —— 本体 / 末端关节状态缓存（JointState：q/dq/tau/温度/
                               错误码）；缓存先经新鲜度过滤，过期或未读到时为 None，
                               规则内必须判空后再取字段；
      motion                —— FSM 当前运动状态（MotionState，如 MOTION / DAMP）；
      event                 —— 当前安全事件（SafetyEvent，如 NORMAL / DAMPING）。
    返回 None 表示「本规则无意见」；返回 SafetyEvent 表示命中意见，与内置规则及
    其他自定义规则的意见合并后取最严重者生效。
    """
    if (state_arm is not None                             # 判空：数据不新鲜时监视线程传 None
            and np.all(np.isfinite(state_arm.dq))         # 防脏数据：dq 含 NaN/inf 时本周期放弃判定
            and np.any(np.abs(state_arm.dq) > DQ_LIMIT)):  # 任一关节速度越限 → 超速
        return SafetyEvent.DAMPING                        # 超速意见：阻尼防护（FSM 将转入 DAMP 软停）
    return None                                           # 无意见：本周期不触发


# ============================================================
# 三、格式化助手：表格列对齐与数值占位显示
# ============================================================
def pad(s: str, width: int, align: str = "left") -> str:
    """按终端显示宽度补空格使表格列对齐（全角字符计 2 列，超宽不截断）。"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))  # 逐字符累加显示列宽
    gap = " " * max(width - w, 0)                         # 缺口 = 目标宽 − 显示宽（不为负）
    return gap + s if align == "right" else s + gap       # left 右补空格 / right 左补空格


def fmt(value, spec=".2f"):
    """表格数值格式化：None（未读到数据）显「—」，数值按 spec 格式化。"""
    return "—" if value is None else format(value, spec)  # 「—」占位避免打印 None 造成歧义


# ============================================================
# 四、演示主流程：启动 FSM 与监视器 → 挂规则 → 触发观察 → 安全收尾
# ============================================================
def main() -> None:
    """监视器演示主流程：连接 → 先启 FSM → 挂规则并启监视器 → 触发观察 → 收尾。"""
    model = parse_model()                                 # 必填型号参数
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # info 日志单行显示（含状态迁移轨迹）

    try:
        with joyArm(model) as arm:                        # 创建整机对象；with 退出自动安全收尾（可重复调用）
            print("=" * 78)                               # 大节分隔线
            print("例程 07：安全监视器 safe_monitor（挂规则 → 触发 → 恢复说明 → 收尾）")  # 大节标题
            print("=" * 78)                               # 大节分隔线
            print(f"『{model}』创建成功：本体 {arm.n_arm} 关节、末端 {arm.n_end} 电机")  # 创建结果
            arm.connect_arm()                             # 连接本体总线（channel/protocol 缺省取配置值）

            # ---- ① 启动顺序：先启 FSM、再启监视器 ----
            print("=" * 78)                               # 大节分隔线
            print("一、启动依赖与顺序：先 start_motion_fsm 再 start_monitor")  # 大节标题
            print("-" * 78)                               # 小节分隔线
            print("· 监视规则的第 3 参 motion 直接消费 FSM 当前运动状态（每个周期传入）；")  # 依赖一
            print("· 规则命中后经 set_event 改写 FSM 目标状态，防护动作（如 DAMP 阻尼软停）")  # 依赖二
            print("  由 FSM 线程执行——监视器只判定，不直接驱动电机；")  # 依赖二的续句
            print("· start_monitor 本身不校验 FSM 是否运行，但 FSM 未启动时事件无人消费、")  # 依赖三
            print("  防护不落地，因此建议先启 FSM 再启监视器（本例按此顺序演示）。")  # 依赖三的结论
            input("\n>>> 确认机械臂工作半径内无人员与障碍物；回车后启动 FSM（自动回 home）……")  # 启动前人工确认
            arm.start_motion_fsm()                        # 启动运动状态机：DISABLE → HOME → MOTION 自动迁移
            if arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0) != 1:  # 1=到达 / 0=超时
                print("FSM 10 秒内未进入 MOTION 态，本例无法继续演示，直接收尾退出。")  # 超时分支
                return                                    # 提前返回：with 退出仍会执行安全收尾
            print(f"FSM 已到达 MOTION 态（当前安全事件 {arm.get_event().name}）")  # 到达结果

            # ---- ② 自定义规则：四参数签名与两种传法 ----
            print("=" * 78)                               # 大节分隔线
            print("二、自定义规则：四参数签名与 set_monitor_rules 两种传法")  # 大节标题
            print("-" * 78)                               # 小节分隔线
            print("  规则签名 rule(state_arm, state_end, motion, event) → SafetyEvent | None：")  # 签名行
            rows = [                                      # 四参数含义速查表（左列对齐）
                ("state_arm", "本体关节状态缓存（q/dq/tau…；不新鲜或未读到时为 None，须判空）"),  # 参数 1
                ("state_end", "末端关节状态缓存（同上；本例未连接 end 总线，恒为 None）"),  # 参数 2
                ("motion", "FSM 当前运动状态（MotionState，如 MOTION / DAMP）"),  # 参数 3
                ("event", "当前安全事件（SafetyEvent，如 NORMAL / DAMPING）"),  # 参数 4
            ]
            for name, desc in rows:                       # 逐行打印参数说明
                print(f"  {pad(name, 12)}{desc}")         # 参数名对齐 + 含义说明
            print("· 返回 None = 无意见；返回 SafetyEvent = 命中，与内置规则（故障码 ≥ 2 →")  # 返回值语义
            print("  DAMPING；断连 / 连续 0.1 s 无新鲜数据 → DISABLE）合并取最严重者。")  # 合并判定规则
            arm.set_monitor_rules(overspeed)              # 传法 1：直接传单条可调用对象
            print("· 传法 1：set_monitor_rules(overspeed)——单条挂接")
            arm.set_monitor_rules([overspeed])            # 传法 2：传规则列表（整表替换，非追加）
            print("· 传法 2：set_monitor_rules([overspeed])——列表可一次挂多条规则，重复")
            print("  绑定为整表替换语义；传入非可调用项会抛 ValueError。")
            monitor_hz = arm.get_config()["joyarm"]["MotionFSM"]["monitor_hz"]  # 从配置数据读监视频率
            arm.start_monitor()                           # 启动监视线程（daemon；重复调用安全忽略）
            print(f"· 监视器已启动：轮询频率 monitor_hz = {monitor_hz:g} Hz")

            # ---- ③ 触发演示：快速小幅运动 → 超速防护 DAMP ----
            print("=" * 78)                               # 大节分隔线
            print(f"三、触发演示：关节 1 快速摆动 {TRIG_DQ} rad → 观察超速防护")  # 大节标题
            print("-" * 78)                               # 小节分隔线
            # 默认规划器为「直传」single_joint_hold_traj_planner：忽略目标时刻 t、按电机内部
            # 能力直达目标，快慢不可控；切换三次多项式规划器后 t 决定运动时长——
            # 0.25 rad / 0.2 s 的 rest-to-rest 三次轨迹名义峰值约 1.5×Δq/T ≈ 1.9 rad/s，
            # 越过 1.5 rad/s 阈值（实际速度以电机反馈为准；plan 线程周期重规划会更陡）
            if not arm.set_traj_solver("single_joint_cubic_traj_planner"):  # 热切换 traj 域规划器
                print("· 切换三次多项式规划器失败（保持直传），仍继续尝试触发。")
            home = arm.home_arm.copy()                    # home 位形基准（property 只读引用，先复制再改）
            q_trig = home.copy()                          # 构造触发目标：以 home 为基准仅动关节 1
            q_trig[0] += TRIG_DQ                          # 关节 1 +0.25 rad（home 附近小幅摆动）
            input("\n>>> 即将快速摆动触发超速防护（触发后自动进入 DAMP 阻尼软停）；"
                  "\n    确认机械臂工作区安全后回车开始……")
            arm.set_target_joint(time.time() + TRIG_T, q_trig)  # 单帧关节目标：到达时刻 = 当前 + 0.2 s
            print("  " + pad("耗时s", 8) + pad("安全事件", 12) + pad("运动状态", 12) + "max|dq| (rad/s)")
            print("  " + "-" * 46)                        # 表头下的分隔线
            t0 = time.time()                              # 触发运动的起始时刻（表格计时基准）
            triggered = False                             # 是否观察到事件升为 DAMPING
            while time.time() - t0 <= 5.0:                # 最多观察 5 s（监视器 50 Hz，远快于本表）
                snap = arm.joint_state_arm                # 后台只读缓存（不发总线读帧）
                dq_max = float(np.max(np.abs(snap.dq))) if snap is not None else None  # 速度峰值（None 显「—」）
                print(f"  {pad(f'{time.time() - t0:5.1f}', 8)}"
                      f"{pad(arm.get_event().name, 12)}"                 # 当前安全事件列
                      f"{pad(arm.get_current_motion_state().name, 12)}"  # FSM 当前状态列
                      f"{fmt(dq_max)}")                                  # 速度峰值列
                if arm.get_event() is SafetyEvent.DAMPING:  # 超速规则已命中并升级（事件只升不降、会保持）
                    triggered = True                      # 记录：已观察到目标现象
                    break                                 # 结束观察，转入恢复说明
                time.sleep(0.2)                           # 0.2 s 刷新一次表格
            print("-" * 78)
            if triggered:                                 # 命中分支：说明当前防护状态
                print("· 已触发：事件升为 DAMPING（危险级 4/5）、状态机转入 DAMP——逐拍下发")
                print("  MIT 纯阻尼 + 重力补偿指令（q=dq=kp=0），机械臂软停并由重力补偿托住。")
            else:                                         # 未命中分支：如实报告并继续收尾
                print("· 5 秒内未触发：实际反馈速度未越过阈值或跟踪滞后；可增大 TRIG_DQ、")  # 未触发原因
                print("  缩短 TRIG_T 后重试，本例继续走恢复说明与收尾流程。")  # 重试建议
            print("· 恢复流程（本例仅打印说明、不自动执行，避免未排查先恢复）：")  # 恢复流程标题
            print("  ① 阻尼防护中勿强行拖动机械臂，先排查并消除触发原因（本例为演示性超速；")  # 步骤①排查
            print("     实际使用应调慢规划速度或按需调整规则阈值）；")  # 步骤①的实际使用建议
            print("  ② 原因消除后手动复位：arm.set_event(SafetyEvent.NORMAL) → 目标状态回")  # 步骤②复位
            print("     MOTION，FSM 按路线表 DAMP → HOME → MOTION 自动恢复运动；")  # 步骤②的恢复路径
            print("  ③ 本例不复位，直接收尾（事件为 DAMPING 时收尾走「直接失能」路径，见下）。")  # 步骤③本例不执行

            # ---- ④ 收尾：先停监视器、再停状态机 ----
            print("=" * 78)                               # 大节分隔线
            print("四、收尾：stop_monitor() → stop_motion_fsm()")  # 大节标题
            print("-" * 78)                               # 小节分隔线
            was_damping = arm.get_event() is SafetyEvent.DAMPING  # 收尾前记录事件（决定停机路径）
            arm.stop_monitor()                            # 停监视线程：join 带超时（超时仅告警、不强杀）
            print("· stop_monitor()：句柄保留，待线程退出后再次调用本方法完成收尾。")
            arm.stop_motion_fsm()                         # 阻塞安全停机（含等待回零 / 失能），最终失能 arm
            if was_damping:                               # 事件危险级高于 ZEROING 时的停机路径
                print("· stop_motion_fsm()：事件为 DAMPING（危险级 > ZEROING），直接失能停机（不回零）。")
            else:                                         # 事件未超 ZEROING 时的停机路径
                print("· stop_motion_fsm()：事件未超 ZEROING，按置 ZEROING 回零失能路径停机。")
            print("例程完成，with 退出将再执行一次兜底安全收尾（disable + disconnect）。")
    except KeyboardInterrupt:                             # Ctrl+C：with 退出已先置 DISABLE 再安全收尾
        print("\n检测到 Ctrl+C：已中断演示（对象销毁路径已完成安全收尾），程序安全退出。")


if __name__ == "__main__":
    main()
