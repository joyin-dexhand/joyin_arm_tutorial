"""
==============================================================================
例程 06 —— 基于运动状态机的运动（需真机）
==============================================================================
【功能概要】
    1. 确认 FSM 运动流程依赖的 traj / ctrl 两域求解器，并把默认直传规划器切换为三次多项式规划器（按目标时刻 t 平滑插值到达）；
    2. start_motion_fsm 启动状态机：先核对 4 项启动条件，再以 wait_for_motion_state 等待自动迁移 DISABLE→HOME→MOTION（返回 1=到达、0=超时）；
    3. set_target_joint 下发 home 基准小偏移的关节目标，get_target_trajframe 读回并逐字段打印 TrajFrame；
    4. solve_fkine / solve_ikine 构造小位移笛卡尔目标并实际运动，set_target_cart 写入笛卡尔目标帧并读回；
    5. set_event(HOLDING) 触发 HOLD 原位锁定、set_event(NORMAL) 恢复 MOTION，get_event 读回当前事件；
    6. stop_motion_fsm 回零收尾：当前事件不重于 ZEROING 就先回零位再失能，更严重的事件直接失能。

【原理简介】
    运动状态机（MotionFSM）共六状态：DISABLE / ZERO / HOME / MOTION / HOLD / DAMP，其中 HOME 是中转站——在 ZERO 与 MOTION 之间切换都必须先经过 HOME；
    本例依次途经 DISABLE→HOME→MOTION→HOLD→MOTION→HOME→ZERO 五种状态（不含 DAMP）。
    安全事件 SafetyEvent 按严重程度分六级，从轻到重为 NORMAL < HOLDING < HOMING < ZEROING < DAMPING < DISABLE，FSM 按 STATE_FROM 对照表迁往对应状态；
    安全监视器的判定只升不降（同一周期多条规则命中时按最严重的处理、绝不自动降级），恢复正常必须手动 set_event(NORMAL)。
    运行期三线程分工：FSM 线程（fsm_hz）管状态迁移与维持；进入 MOTION 后另起 plan 线程（plan_hz，消费目标帧做规划）与 ctrl 线程（ctrl_hz，采样轨迹并下发控制指令）；
    目标帧通道由 set_target_* 一次性完整写入（不会写一半被别的线程读到）、plan 消费、ctrl 执行，三者互不阻塞。

【安全须知】
    - 运行前清空机械臂工作半径内的人员与障碍物，全程手持急停手段待命；
    - 启动 FSM 将自动回 home（关节 2/3 各移动约 1 rad），收尾回零同理；两段演示运动均为 home 基准 ±0.15 rad 内、2 s 到达的小幅慢速运动；
    - 首次运动前脚本暂停等待确认；Ctrl+C 可随时中断，with 退出自动安全收尾。

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（需已连接真机并上电）
    python quick_start/06_fsm_motion.py <型号>

【预期输出】
    依次打印求解器、FSM 启动条件、目标帧字段三张表，以及状态迁移、到位判断、安全事件切换与回零收尾结果；
    连接失败或启动条件不满足时提示原因并安全退出。
==============================================================================
"""

import sys                                                    # 注入仓库路径用（未安装 joyarm_core 也可运行）
import time                                                   # 计算目标时刻与轮询等待用
import unicodedata                                            # 计算中文串的终端显示宽度（表格对齐用）
from pathlib import Path                                      # 定位本文件上一级目录用

import numpy as np                                            # 关节角 / 位姿数组的构造与运算

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 把 joyarm_code/ 加入模块搜索路径
from joyarm_core import (                                     # 上两行注入路径后再导入（避免 E402 误报）
    joyArm,                                                   # JoyArm 对象创建入口
    listArm,                                                  # 可用型号列表（listArm）
    MotionState,                                              # FSM 六状态枚举
    SafetyEvent,                                              # 安全事件六级枚举
    STATE_FROM,                                               # SafetyEvent → MotionState 映射表
    Pose,                                                     # 统一位姿表示（position + orientation）
)

TRAJ_CUBIC = "single_joint_cubic_traj_planner"                # 三次多项式规划器注册名（本例切换目标）


# ============================================================
# 一、命令行参数解析：必填型号，缺失打印用法并以退出码 1 结束
# ============================================================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并列出可用型号（listArm），以退出码 1 结束。"""
    if len(sys.argv) != 2:                                    # 必须恰好携带 1 个型号参数
        print(f"用法：python quick_start/06_fsm_motion.py <型号>")  # 缺参：先打印用法
        listArm()                                             # 顺带列出全部可用型号
        sys.exit(1)                                           # 退出码 1，不进入演示
    return sys.argv[1].strip()                                # 型号名


# ============================================================
# 二、打印格式化助手：显示宽度、列补空格与数组定宽显示
# ============================================================
def disp_width(s: str) -> int:
    """字符串在终端的显示宽度（中日韩宽字符按 2 列计，其余按 1 列）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)  # 逐字符累加显示列宽


def pad(s: str, width: int, align: str = "left") -> str:
    """按显示宽度补空格使表格列对齐（align="right" 为右对齐；超宽原样返回）。"""
    gap = " " * max(width - disp_width(s), 0)                 # 缺口 = 目标列宽 − 显示宽（不为负）
    return gap + s if align == "right" else s + gap           # 左对齐在右侧补空格、右对齐在左侧补空格


def fmt_arr(arr, width: int = 6) -> str:
    """一维数组格式化为逐元素定宽串；None 显示「—」（表格空值规范）。"""
    if arr is None:                                           # 字段未填写（如手写帧缺省项）
        return "—"
    return " ".join(f"{x:+{width}.3f}" for x in np.asarray(arr, dtype=float))  # 带符号 + 3 位小数，逐列对齐


# ============================================================
# 三、过程助手：到位轮询与目标帧逐字段打印
# ============================================================
def wait_q_arrival(arm, q_goal, tol_q: float = 0.05, timeout: float = 3.0) -> bool:
    """以 0.05 s 步进轮询 is_in_q_arm：到位返回 True，超时返回 False。"""
    deadline = time.monotonic() + timeout                     # 超时截止时刻（monotonic 不受系统改时影响）
    while time.monotonic() < deadline:                        # 截止前持续轮询
        if arm.is_in_q_arm(q_goal, tol_q=tol_q):              # 全关节偏差 ≤ tol_q 即到位
            return True                                       # 到位：提前返回
        time.sleep(0.05)                                      # 0.05 s 轮询间隔
    return False                                              # 超时：未到位


def print_frame_table(frame, title: str) -> None:
    """逐字段打印 TrajFrame 读回值（未填写字段显示「—」）。"""
    print(f"· {title}")                                       # 小节标题行
    rows = [                                                  # (字段名, 读回值字符串) 行集：全部 9 个字段
        ("time", "—" if frame.time is None else f"{frame.time:.3f} s（目标到达时刻，Unix 秒）"),  # 到达时刻
        ("pose", "—" if frame.pose is None else               # 末端位姿（位置 + 四元数）
                 f"pos = {fmt_arr(frame.pose.position)} m；quat(w,x,y,z) = {fmt_arr(frame.pose.orientation)}"),
        ("twist", "—" if frame.twist is None else             # 末端速度（线 / 角）
                  f"linear = {fmt_arr(frame.twist.linear)}；angular = {fmt_arr(frame.twist.angular)}"),
        ("accel", "—" if frame.accel is None else             # 末端加速度（线 / 角）
                  f"linear = {fmt_arr(frame.accel.linear)}；angular = {fmt_arr(frame.accel.angular)}"),
        ("wrench", "—" if frame.wrench is None else           # 末端力 / 力矩参考
                   f"force = {fmt_arr(frame.wrench.force)} N；torque = {fmt_arr(frame.wrench.torque)} N·m"),
        ("q", "—" if frame.q is None else f"{fmt_arr(frame.q)} rad"),  # 关节位置
        ("dq", "—" if frame.dq is None else f"{fmt_arr(frame.dq)} rad/s"),  # 关节速度
        ("ddq", "—" if frame.ddq is None else f"{fmt_arr(frame.ddq)} rad/s²"),  # 关节加速度
        ("tau", "—" if frame.tau is None else f"{fmt_arr(frame.tau)} N·m"),  # 前馈力矩
    ]
    print(pad("字段", 8) + "读回值")                            # 表头（字段列宽 8 显示列）
    for name, value in rows:                                  # 逐行打印，字段列对齐
        print(pad(name, 8) + value)                           # 名称列补齐后拼接读回值


# ============================================================
# 四、真机演示全流程：连接 → 求解器 → FSM → 目标帧 → 安全事件 → 回零收尾
# ============================================================
def run_demo() -> None:
    """真机演示全流程：连接 → 求解器 → FSM → 关节/笛卡尔目标 → 安全事件 → 回零收尾。"""
    model = parse_model()                                     # 必填型号参数
    with joyArm(model) as arm:                                # with 退出自动安全收尾（停 FSM→失能→断连）
        print("================== 例程 06 —— 基于运动状态机（MotionFSM）的运动 ================ ")

        # ---- 连接：FSM 仅涉 arm 侧，无需连接 end ----
        if arm.connect_arm() is not True:                     # 连接 arm 总线（通道取配置缺省值）
            print("arm 总线连接失败：请检查电源与总线后重试")                   # 失败提示（with 退出自动断连收尾）
            return                                            # 终止演示
        q_home = np.array(arm.home_arm, dtype=float)          # home 位形（只读引用 → 拷贝为可写运动基准）
        print(f"已连接 {model}（{arm.n_arm} 关节），"
              f"home = {fmt_arr(q_home)} rad\n")              # 打印连接信息与运动基准位形

        print("================== ① 求解器确认：traj（轨迹规划）与 ctrl（关节控制）两域 ================ ")
        traj_names = arm.get_traj_solvers()                   # traj 域已加载规划器注册名（排序深拷贝）
        ctrl_names = arm.get_ctrl_solvers()                   # ctrl 域已加载控制器注册名
        traj_act = arm.get_activated_traj_solver()            # traj 域当前激活规划器
        ctrl_act = arm.get_activated_ctrl_solver()            # ctrl 域当前激活控制器
        print(pad("域", 6) + pad("求解器注册名", 34) + "状态")         # 表头（列宽按显示列对齐）
        for name in traj_names:                               # traj 域逐行打印
            mark = "默认激活" if name == traj_act else "本例待切换"    # 标注当前激活项
            print(pad("traj", 6) + pad(name, 34) + mark)      # 一行一个规划器
        for name in ctrl_names:                               # ctrl 域逐行打印
            mark = "默认激活" if name == ctrl_act else "备选"   # 标注当前激活项
            print(pad("ctrl", 6) + pad(name, 34) + mark)      # 一行一个控制器
        print(f"默认激活的 {traj_act} 为直传规划器：忽略目标帧的 t 等其余字段，")     # 说明默认规划器行为
        print("把目标 q 当作恒定目标直发（不按到达时刻插值、无法约束运动时长）。")           # 为什么要切换
        ok_switch = arm.set_traj_solver(TRAJ_CUBIC)           # 切三次多项式规划器（先试规划成功才替换）
        print(f"演示运动前切换为 {TRAJ_CUBIC}："
              f"{'成功' if ok_switch else '失败（保持原激活）'}")        # 切换结果
        print(f"当前激活 = {arm.get_activated_traj_solver()}，"
              f"plan_hz = {arm.plan_hz} Hz\n")                # 切换后激活名与规划线程频率

        print("================== ② 启动运动状态机：start_motion_fsm（启动条件 4 项） ================ ")
        state = arm.joint_state_arm                           # arm 被动状态快照（刷新线程持续供给）
        if state.t > 0:                                       # t=0 表示从未收到状态帧
            lag_ms = (time.time() - state.t) * 1000.0         # 反馈时延（毫秒）
            basis_fresh = f"now − t = {lag_ms:.1f} ms ≤ 50 ms"  # 判定依据：时延数值
        else:                                                 # 从未收到反馈
            lag_ms, basis_fresh = float("inf"), "从未收到状态帧（t = 0）"  # 时延视为无穷大
        fresh = 0.0 <= lag_ms / 1000.0 <= 0.05                # 新鲜判据与库一致：≤ 1/状态刷新率（20 Hz → 0.05 s）
        faultless = bool(np.all(np.isin(state.error, (0, 1))))  # 故障码 0=失能、1=使能，≥2 为故障
        configured = bool(traj_names) and bool(ctrl_names)    # traj/ctrl 两域均已加载求解器成员
        checks = [                                            # (条件名, 判定依据, 是否通过) 行集
            ("总线已连接", "is_connected_arm is True", arm.is_connected_arm is True),  # 条件 1：连接
            ("反馈新鲜", basis_fresh, fresh),                 # 条件 2：状态帧时延 ≤ 50 ms
            ("无电机故障", "joint_state_arm.error 全部 ∈ {0, 1}", faultless),  # 条件 3：故障码
            ("求解器已配置", "traj/ctrl 两域已加载成员非空", configured),    # 条件 4：两域求解器
        ]
        print(pad("启动条件", 14) + pad("判定依据", 38) + "结果")       # 表头（列宽按显示列对齐）
        for name, basis, ok in checks:                        # 逐行打印 4 项启动条件
            print(pad(name, 14) + pad(basis, 38) + ("通过" if ok else "不通过"))  # 结果列：通过/不通过
        input("\n>>> 即将启动 FSM：自动回 home 后开始小幅运动，"
              "确认工作半径内无人无障碍后回车继续……")         # 危险动作前的人工确认
        try:
            arm.start_motion_fsm()                            # 非阻塞启动（条件不满足会抛 RuntimeError）
        except RuntimeError as err:                           # 启动检查不通过：打印原因退出
            print(f"start_motion_fsm 拒绝启动：{err}")        # 拒绝原因（连接/反馈/故障/域未配置）
            return                                            # with 退出自动安全收尾
        ok = arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0)  # 返回 1=已到达 / 0=超时
        print(f"wait_for_motion_state(MOTION) = {ok}（{'已到达' if ok else '超时'}），"
              f"当前态 = {arm.get_current_motion_state().name}，"
              f"目标态 = {arm.get_target_motion_state().name}")  # 自动迁移 DISABLE→HOME→MOTION 结果
        if not ok:                                            # 未进入 MOTION：收尾退出
            print("FSM 未进入 MOTION：执行 stop_motion_fsm 收尾退出")   # 失败路径提示
            arm.stop_motion_fsm()                             # 阻塞回零失能
            return                                            # 终止演示
        print()                                               # 节间空行

        print("================== ③ 关节目标帧：set_target_joint（home 基准小偏移） ================ ")
        q_goal = q_home.copy()                                # 目标 = home 拷贝（不污染只读引用）
        q_goal[0] += 0.15                                     # 关节 1 +0.15 rad（home±0.2 rad 安全包络内）
        q_goal[1] -= 0.15                                     # 关节 2 −0.15 rad
        t_goal = time.time() + 2.0                            # 目标到达时刻：2 s 后（cubic 规划器按 t 定时长）
        arm.set_target_joint(t=t_goal, q=q_goal)              # 单帧关节目标 → 一次性完整写入目标通道
        frames = arm.get_target_trajframe()                   # 深拷贝读回（单帧目标 → 单元素列表）
        print_frame_table(frames[0], "set_target_joint 写入后读回 TrajFrame")  # 逐字段打印读回帧
        print(f"到位判断 is_in_q_arm：{'是' if wait_q_arrival(arm, q_goal) else '否'}\n")  # 轮询到位（容差 0.05 rad）

        print("================== ④ 笛卡尔目标：solve_fkine / solve_ikine 构造，set_target_cart 写入 ================ ")
        pose_cur = arm.solve_fkine()                          # 正运动学（q 缺省取当前快照）→ Pose
        pose_goal = Pose(position=pose_cur.position + np.array([0.04, 0.0, 0.02]),
                         orientation=pose_cur.orientation)    # 沿基座系 x/z 平移 4/2 cm，姿态保持不变
        res = arm.solve_ikine(pose_goal)                      # 数值法自当前 q 迭代 → IKResult 单解
        print(f"逆解：success = {res.success}，err = {res.err:.2e}，n_iter = {res.n_iter}")  # 逆解质量
        if not res.success or res.q is None:                  # 未收敛（q 可能为 None）：跳过本节
            print("逆解未收敛，跳过笛卡尔运动")               # 失败路径提示
        else:
            dev = float(np.abs(res.q - q_home).max())         # 逆解相对 home 的最大关节偏差
            if dev > 0.2:                                     # 超出 home±0.2 rad 安全包络：不下发
                print(f"逆解超出 home±0.2 rad 安全包络（最大偏差 {dev:.3f} rad），跳过笛卡尔运动")
            else:                                             # 逆解可用：执行笛卡尔小位移运动
                print(f"逆解 q = {fmt_arr(res.q)} rad（相对 home 最大偏差 {dev:.3f} rad）")  # 逆解结果
                arm.set_target_joint(t=time.time() + 2.0, q=res.q)  # 实际执行走关节通道（激活控制器为关节位置控制）
                arrived = wait_q_arrival(arm, res.q)          # 轮询到位
                print(f"笛卡尔运动到位：{'是' if arrived else '否'}，"
                      f"位姿验证 is_in_pose_arm = {'是' if arm.is_in_pose_arm(pose_goal) else '否'}")  # 到位与位姿复核
                arm.set_target_joint(t=time.time() + 2.0, q=q_home)  # 先回 home（下一步通道演示不引发运动）
                wait_q_arrival(arm, q_home)                   # 等回 home 到位
                arm.set_target_cart(t=time.time() + 2.0, pose=pose_goal)  # 笛卡尔单帧目标写入同一通道
                frames = arm.get_target_trajframe()           # 读回：帧内 pose 有值、q 字段为 None
                print_frame_table(frames[0], "set_target_cart 写入后读回 TrajFrame")  # 逐字段打印读回帧
                arm.set_target_joint(t=time.time() + 2.0, q=q_home)  # 恢复关节目标：当前两规划器仅消费 q，
                # ↑ pose-only 帧会被规划器回退为恒定 q_home（已在 home 位，不产生意外运动）
        print()                                               # 节间空行

        print("================== ⑤ 安全事件：set_event / get_event ================ ")
        print("严重程度从轻到重：NORMAL < HOLDING < HOMING < ZEROING < DAMPING < DISABLE")  # 六级链速查
        print(f"当前事件 = {arm.get_event().name}，"
              f"目标态 = {STATE_FROM[arm.get_event()].name}")    # 事件 → 目标态映射（STATE_FROM）
        arm.set_event(SafetyEvent.HOLDING)                    # 触发原位锁定：目标态同步为映射的 HOLD
        ok_hold = arm.wait_for_motion_state(MotionState.HOLD, timeout=5.0)  # 返回 1=已到达 / 0=超时
        print(f"set_event(HOLDING) → {STATE_FROM[SafetyEvent.HOLDING].name}："  # HOLDING 的映射目标态
              f"等待 = {ok_hold}（{'已到达' if ok_hold else '超时'}），"
              f"当前态 = {arm.get_current_motion_state().name}，"
              f"事件 = {arm.get_event().name}")               # HOLD 到达情况与当前事件
        arm.set_event(SafetyEvent.NORMAL)                     # 恢复正常（监视器只升不降，恢复须手动下发）
        ok_back = arm.wait_for_motion_state(MotionState.MOTION, timeout=5.0)  # 返回 1=已到达 / 0=超时
        print(f"set_event(NORMAL) → {STATE_FROM[SafetyEvent.NORMAL].name}："  # NORMAL 的映射目标态
              f"等待 = {ok_back}（{'已到达' if ok_back else '超时'}），"
              f"当前态 = {arm.get_current_motion_state().name}，"
              f"事件 = {arm.get_event().name}\n")             # MOTION 恢复情况与当前事件

        print("================== ⑥ 收尾：stop_motion_fsm（阻塞式） ================ ")
        print("收尾规则：当前事件不重于 ZEROING → 先回零位（经 HOME）再失能；")  # 收尾规则一
        print("            更严重（DAMPING/DISABLE）→ 直接失能停机。")  # 收尾规则二
        arm.stop_motion_fsm()                                 # 当前事件 NORMAL（≤ZEROING）→ 走回零失能路径
        print(f"FSM 已停止：当前态 = {arm.get_current_motion_state().name}，"
              f"事件 = {arm.get_event().name}（收尾统一复位为 DISABLE）")  # 最终状态复核
        print("================== 完成：机械臂已回零失能，with 退出将自动断开连接。 ================= ")


# ============================================================
# 五、脚本入口：运行演示，Ctrl+C 时安全退出
# ============================================================
def main() -> None:
    """入口：运行演示；Ctrl+C 中断被吞掉（with 退出路径已完成安全收尾）。"""
    try:
        run_demo()                                            # 全部真机演示逻辑
    except KeyboardInterrupt:                                 # Ctrl+C：不上抛堆栈，仅提示
        print("\n检测到 Ctrl+C：已安全退出（with 已停 FSM、失能并断连）。")       # 安全退出提示


if __name__ == "__main__":                                    # 脚本直跑入口（被导入时不执行）
    main()
