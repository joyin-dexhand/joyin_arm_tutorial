"""
==============================================================================
例程 04 —— 直通硬件电机指令（需真机）
==============================================================================
【功能概要】
    真机演示在运动状态机（FSM）未运行时，直通硬件电机的底层指令代码形态：
      1. 连接并使能本体（arm）与夹爪（end），每步打印使能 / 模式 / 位置变化
      2. 故障码检查与清错（先排查故障原因，再执行清除）：arm / end 双侧对称演示
         get_error / check_error，任一侧有故障码时经确认后 clear_error
      3. 位置模式：send_position_arm 回零位形 + is_in_q_arm 轮询到位
      4. 速度模式：send_vel_arm / send_vel_end 单自由度点动后发零速停止
      5. MIT 模式：send_mit_arm / send_mit_end 五个参数 (q, dq, tau, kp, kd) 低增益安全示例
      6. 夹爪：get_mode_end / set_mode_end 模式切换、set_action_end 离散动作、
         send_position_end 连续位置指令 + is_in_q_end 到位判断
      7. 说明 FSM 互斥：FSM 运行时 arm 侧直通指令会被拦下（打一条 warn 警告，不真的执行）

【原理简介】
    JoyArm 底层电机有三类控制模式（ControlMode 枚举）：MIT 模式一次下发五个参数
    (q, dq, tau, kp, kd)，电机端执行 tau + kp*(q - q_cur) + kd*(dq - dq_cur)；
    位置 / 速度模式只发单一目标，闭环增益参数在切模式时由 backend 自动写入电机。
    直通指令绕过轨迹规划与 FSM、由用户线程直接写电机：响应即时但无安全过渡，
    只适合小幅、低速、空载的点动验证；FSM 运行时电机由其控制线程独占，arm 侧
    指令会被 _fsm_guard 拦下（warn 提示 + 不执行），end 侧无控制线程、不受限。

【安全须知】
    - 运行前清空机械臂工作半径内的人员与障碍物，确认本体与夹爪运动路径无遮挡
    - 手边准备急停：随时 Ctrl+C 中断，with 退出会自动失能并断开总线
    - 全程小幅缓慢运动：目标为零位形、点动速度 0.3 rad/s、MIT 低增益短时保持
    - 使能后电机进入锁定状态，切勿强行扳动关节；清错后关节自动失能属正常现象

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（需已连接真机并上电）
    python quick_start/04_direct_motor_command.py <型号>

【预期输出】
    依次打印连接 / 使能 / 双侧清错检查、三模式点动、夹爪 MIT 保持与速度点动、
    离散动作与到位判断的前后状态对比；机械臂回零位形并点动第 1 关节，夹爪开合
    回中后低速点动张开，最后失能退出。
==============================================================================
"""

from pathlib import Path                                   # 路径工具：定位仓库根目录
import sys                                                 # 系统接口：修改模块搜索路径
import time                                                # 时间接口：轮询间隔与点动计时

import numpy as np                                         # 数值计算：构造目标向量与增益向量

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 把 joyarm_code/ 加入搜索路径，未安装库也可运行
# # 机械臂对象创建入口 / 可用型号列表与控制模式枚举
from joyarm_core import joyArm, ControlMode, listArm


# ============================================================
# 一、命令行型号参数：解析必填的 <型号> 命令行参数，缺失时打印用法并以退出码 1 结束
# ============================================================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并列出可用型号（listArm），以退出码 1 结束。"""
    if len(sys.argv) != 2:                                # 必须恰好携带 1 个型号参数
        print(f"用法：python quick_start/04_direct_motor_command.py <型号>")  # 缺参：先打印用法
        listArm()                                         # 顺带列出全部可用型号
        sys.exit(1)                                       # 退出码 1，不进入演示
    return sys.argv[1].strip()                            # 型号名


# ============================================================
# 二、输出格式化工具：状态 / 向量 / 模式转可读字符串，大小节标题打印与到位轮询
# ============================================================
def vec_str(v) -> str:
    """数值向量转字符串：None 显「—」，其余逐元素保留 3 位小数。"""
    return "—" if v is None else np.round(v, 3).tolist()  # None=读取失败/未知，显破折号


def mode_str(mode) -> str:
    """控制模式转字符串：None 显「—」，否则显示「枚举名(枚举值)」。"""
    return "—" if mode is None else f"{mode.name}({mode.value})"  # None=全组不一致/读取失败


def tri(v) -> str:
    """三值布尔（True / False / None）转字符串：None 显「—」。"""
    return "—" if v is None else str(v)                   # 连接/使能标志均为三值


def sec(title: str) -> None:
    """打印单行大节横幅：35 个 "=" + 标题 + 右侧 "=" 补齐至 84 字符（末尾带一个空格）。"""
    print("\n" + "=" * 35 + f" {title} " + "=" * max(46 - len(title), 2) + " ")  # 前空一行更醒目


def sub(title: str) -> None:
    """打印小节标题（上下各一条 78 字符 "-" 分隔线）。"""
    print("-" * 78)                                       # 小节分隔线（上）
    print(title)                                          # 小节标题本体
    print("-" * 78)                                       # 小节分隔线（下）


def status_arm(arm) -> str:
    """格式化 arm 侧关键状态：使能 + 模式（主动读）+ 关节位置（读后台缓存）。"""
    return (f"使能={tri(arm.is_enabled_arm)}，模式={mode_str(arm.get_mode_arm())}，"
            f"q={vec_str(arm.joint_state_arm.q)} rad")     # 三项关键状态拼一行


def status_end(arm) -> str:
    """格式化 end 侧关键状态：使能 + 模式（主动读）+ 电机位置（读后台缓存）。"""
    return (f"使能={tri(arm.is_enabled_end)}，模式={mode_str(arm.get_mode_end())}，"
            f"q={vec_str(arm.joint_state_end.q)} rad")     # 与 status_arm 同构，对应夹爪


def wait_arrival(judge, q, tol_q: float, timeout: float = 3.0, hz: float = 20.0):
    """按固定频率轮询到位判断，直至到位或超时。

    :param judge: 到位判断函数（arm.is_in_q_arm / arm.is_in_q_end，均读后台缓存）。
    :param q: 目标关节角 / 电机角（rad）。
    :param tol_q: 到位容差（rad）。
    :param timeout: 最长轮询时长（秒）。
    :param hz: 轮询频率（Hz）。
    :return: (是否到位, 实际耗时秒)。
    """
    t0 = time.time()                                      # 记录轮询起始时刻
    while time.time() - t0 <= timeout:                    # 超时前持续轮询
        if judge(q, tol_q=tol_q):                         # 全关节同时满足容差即到位
            return True, time.time() - t0                 # 提前返回到位标志与耗时
        time.sleep(1.0 / hz)                              # 按轮询频率休眠，避免刷总线
    return False, time.time() - t0                        # 超时仍未到位


# ============================================================
# 三、演示主流程：按「连接使能 → 清错检查 → 三模式直通指令 → 夹爪 → 失能」顺序完整演示
# ============================================================
def run_demo(arm) -> None:
    """按「连接使能 → 清错检查 → 三模式直通指令 → 夹爪 → 失能」顺序完整演示。"""

    # ---- ① 连接与使能 ----
    sec("① 连接与使能")
    ok = arm.connect_arm()                                # 连接本体总线（通道取配置缺省值，连接后自动设默认模式）
    print(f"connect_arm 返回：{tri(ok)}；is_connected_arm：{tri(arm.is_connected_arm)}")  # True=成功 / False=失败 / None=意外
    ok = arm.connect_end()                                # 连接夹爪总线（与本体可共用通道）
    print(f"connect_end 返回：{tri(ok)}；is_connected_end：{tri(arm.is_connected_end)}")  # 同上三值语义

    sub("使能前后状态对比")                               # 小节：使能前后对比
    print(f"使能前 arm：{status_arm(arm)}")               # 使能前三项关键状态基线
    print(f"使能前 end：{status_end(arm)}")               # 夹爪侧基线
    print(f"enable_arm 返回：{arm.enable_arm()}；enable_end 返回：{arm.enable_end()}")  # 1=全部成功、0=有失败
    print(f"使能后 arm：{status_arm(arm)}")               # 使能后状态（应为 True / position 模式）
    print(f"使能后 end：{status_end(arm)}")               # 夹爪侧使能后状态

    sub("三类控制模式（ControlMode 枚举）与对应整组下发指令")  # 小节：模式速查表
    print("模式       枚举值       整组下发指令           下发内容")  # 表头（前三列纯 ASCII 保证对齐）
    print("MIT        mit          send_mit_arm           五个参数 (q, dq, tau, kp, kd)")  # MIT 行
    print("POSITION   position     send_position_arm      目标关节角 q")  # 位置模式行
    print("VELOCITY   velocity     send_vel_arm           目标关节速度 dq")  # 速度模式行
    print("切 POSITION / VELOCITY 时 backend 自动随写 POS_VEL 四增益寄存器；MIT 增益随指令帧下发。")  # 增益来源说明

    # ---- ② 故障码检查与清错（arm / end 双侧对称） ----
    sec("② 故障码检查与清错（先排查原因，再清除；arm / end 双侧）")
    errs = arm.get_error_arm()                            # 读本体各关节状态码缓存
    print(f"arm 各关节状态码 get_error_arm：{vec_str(errs)}（0=失能、1=使能、≥2=故障）")  # 状态码含义
    print(f"arm 健康检查 check_error_arm：{arm.check_error_arm()}（全 0/1 判正常 True）")  # 一键健康判定
    print(f"end 各电机状态码 get_error_end：{vec_str(arm.get_error_end())}（状态码含义同上）")  # end 侧状态码
    print(f"end 健康检查 check_error_end：{arm.check_error_end()}（与 check_error_arm 同判据）")  # end 一键健康判定
    if arm.check_error_arm() and arm.check_error_end():   # 双侧均无故障码：不需要清错
        print("双侧状态码全 0/1，无故障码：无需 clear_error，直接进入下一步。")  # 无故障分支说明
    else:                                                 # 任一侧存在 ≥2 故障码：先排查再清除
        print("检测到故障码（≥2）：请先排查并消除原因（过压 / 过流 / 过温 / 通信等），")  # 提示排查故障根因
        print("切勿盲目清错后继续运行——故障未消除会立即再次触发。")  # 强调清错前提
        input(">>> 确认故障原因已排除、工作区安全后，按回车执行 clear_error……")  # 清错前人工确认
        if not arm.check_error_arm():                     # arm 侧有故障码：clear_error_arm（FSM 运行时会被拦下，本例程未启 FSM）
            ret = arm.clear_error_arm()                   # 逐关节发清错帧并核对；清错后关节自动失能属正常
            print(f"clear_error_arm 返回：{ret}（True=全部清除成功）")  # 清错结果
        if not arm.check_error_end():                     # end 侧有故障码：clear_error_end 对称清除（end 不受此限制）
            ret_end = arm.clear_error_end()               # 逐电机发清错帧并核对；清错后电机自动失能属正常
            print(f"clear_error_end 返回：{ret_end}（True=全部清除成功）")  # end 清错结果
        print(f"清错后需重新使能：enable_arm 返回 {arm.enable_arm()}，enable_end 返回 {arm.enable_end()}")  # 清错失能后重新使能
        print(f"重新使能后 arm：{status_arm(arm)}")       # 复查使能状态
        print(f"重新使能后 end：{status_end(arm)}")       # 复查夹爪使能状态

    input("\n>>> 即将开始小幅运动（回零 / 点动 / MIT 保持 / 夹爪开合），请确认工作区安全后按回车……")  # 运动前人工确认

    # ---- ③ 位置模式 ----
    sec("③ 位置模式 POSITION：send_position_arm + is_in_q_arm 轮询到位")
    print(f"切模式前 arm：{status_arm(arm)}")             # 切模式前基线
    ret = arm.set_mode_arm(ControlMode.POSITION)          # 切位置模式（成功返回 1 并乐观更新模式缓存）
    print(f"set_mode_arm(POSITION) 返回：{ret}（1=成功、0=失败）")  # 切模式结果
    print(f"切模式后 arm：{status_arm(arm)}（get_mode_arm 读回验证）")  # 读回验证乐观缓存
    q_goal = arm.zero_arm                                 # 目标 = 零位形（全 0 rad，库内只读引用）
    print(f"下发 send_position_arm：q_goal = {vec_str(q_goal)} rad（越硬限位由库自动就近裁剪）")  # 目标与保护
    arm.send_position_arm(q_goal)                         # 整组下发目标关节角（限速 / 限流取配置 POS_VEL）
    ok, dt = wait_arrival(arm.is_in_q_arm, q_goal, tol_q=0.05, timeout=3.0)  # 轮询到位（容差与配置缺省 0.05 rad 一致）
    print(f"is_in_q_arm 轮询：{'已到位' if ok else '3 秒未到位（负载 / 阻尼偏大），继续下一步'}（耗时 {dt:.2f} s）")  # 到位结果
    print(f"到位后 arm：{status_arm(arm)}")               # 到位后位置应为全 0 附近

    # ---- ④ 速度模式 ----
    sec("④ 速度模式 VELOCITY：send_vel_arm 单关节点动")
    ret = arm.set_mode_arm(ControlMode.VELOCITY)          # 切速度模式
    print(f"set_mode_arm(VELOCITY) 返回：{ret}")          # 切模式结果
    print(f"点动前 arm：{status_arm(arm)}")               # 点动前基线
    dq = np.zeros(arm.n_arm)                              # 目标速度向量（n_arm = 本体关节数）
    dq[0] = 0.3                                           # 仅第 1 关节 +0.3 rad/s（幅值越 dq_max 由库裁剪）
    print(f"下发 send_vel_arm：dq = {vec_str(dq)} rad/s，运行 1 秒")  # 点动参数
    arm.send_vel_arm(dq)                                  # 整组下发目标速度
    time.sleep(1.0)                                       # 点动持续 1 秒（约转 0.3 rad）
    arm.send_vel_arm(np.zeros(arm.n_arm))                 # 先发零速停止，再做任何其他操作
    time.sleep(0.2)                                       # 等零速帧生效
    print(f"点动后 arm：{status_arm(arm)}（第 1 关节应比点动前多约 0.3 rad）")  # 点动效果核对

    # ---- ⑤ MIT 模式 ----
    sec("⑤ MIT 模式：send_mit_arm 五个参数 (q, dq, tau, kp, kd) 低增益示例")
    ret = arm.set_mode_arm(ControlMode.MIT)               # 切 MIT 模式
    print(f"set_mode_arm(MIT) 返回：{ret}")               # 切模式结果
    print(f"保持前 arm：{status_arm(arm)}")               # 保持前基线
    n = arm.n_arm                                         # 关节数（构造逐关节向量用）
    q_cur = np.asarray(arm.joint_state_arm.q, dtype=float)  # 当前位形（MIT 目标 = 原地保持）
    dq_ff = np.zeros(n)                                   # 速度前馈 0：不主动给速度
    tau_ff = np.zeros(n)                                  # 力矩前馈 0：不主动给力矩
    kp = np.full(n, 20.0)                                 # 位置增益 20（配置缺省 120，取低值降低刚度）
    kd = np.full(n, 0.5)                                  # 速度增益 0.5（配置缺省 5.0，低增益软保持）
    print(f"五个参数：q=当前位形，dq={vec_str(dq_ff)}，tau={vec_str(tau_ff)}，kp=20.0，kd=0.5")  # 五个参数内容
    arm.send_mit_arm(q=q_cur, dq=dq_ff, tau=tau_ff, kp=kp, kd=kd)  # 整组下发 MIT 指令
    time.sleep(1.0)                                       # 低增益保持 1 秒（受重力轻微下垂属正常）
    print(f"保持后 arm：{status_arm(arm)}")               # 保持后位置对比
    ret = arm.set_mode_arm(ControlMode.POSITION)          # 演示完切回位置模式（恢复刚度，防低增益下垂）
    arm.send_position_arm(arm.zero_arm)                   # 回零位形，后续夹爪演示期间稳定保持
    ok, _ = wait_arrival(arm.is_in_q_arm, arm.zero_arm, tol_q=0.05, timeout=3.0)  # 等回零到位
    print(f"切回 POSITION 并回零：set_mode_arm 返回 {ret}，{'已到位' if ok else '未完全到位'}")  # 回零结果

    # ---- ⑥ 末端夹爪 ----
    sec("⑥ 末端夹爪 end：get/set_mode_end + set_action_end + send_position_end")
    print(f"当前 end：{status_end(arm)}")                 # 夹爪侧基线

    sub("模式读取与切换 + MIT 五个参数 send_mit_end 低增益保持")  # 小节：end 模式切换与 MIT 保持
    print(f"get_mode_end：{mode_str(arm.get_mode_end())}（连接后默认 position 模式）")  # 读当前模式
    ret = arm.set_mode_end(ControlMode.MIT)               # 切 MIT：演示 end 侧模式切换（end 侧不受 FSM 限制）
    print(f"set_mode_end(MIT) 返回：{ret}；读回：{mode_str(arm.get_mode_end())}")  # 切换结果与读回
    n_end = arm.n_end                                     # 末端电机数（本型号夹爪单自由度 = 1）
    q_end_cur = np.asarray(arm.joint_state_end.q, dtype=float)  # 当前电机位置（MIT 目标 = 原地保持）
    print(f"保持前 end：{status_end(arm)}")               # 保持前基线
    arm.send_mit_end(q=q_end_cur, dq=np.zeros(n_end), tau=np.zeros(n_end),  # 五个参数：q=当前位形、dq/tau 前馈 0
                     kp=np.full(n_end, 20.0), kd=np.full(n_end, 0.5))  # kp/kd 低增益软保持（与 ⑤ arm 侧同参）
    time.sleep(1.0)                                       # 低增益保持 1 秒（夹爪原地不动）
    print(f"保持后 end：{status_end(arm)}（send_mit_end 五个参数保持，位置应基本不变）")  # 保持后对比
    ret = arm.set_mode_end(ControlMode.POSITION)          # 切回 POSITION（后续动作 / 位置指令所需模式）
    print(f"set_mode_end(POSITION) 返回：{ret}；读回：{mode_str(arm.get_mode_end())}")  # 切换结果与读回

    sub("离散动作 set_action_end + is_in_q_end 到位判断")  # 小节：离散动作与到位
    print(f"动作前 end：{status_end(arm)}")               # 动作前夹爪位置
    arm.set_action_end("open")                            # 张开到固定行程位 -5.0 rad（非 POSITION 模式时自动切换）
    ok, dt = wait_arrival(arm.is_in_q_end, np.array([-5.0]), tol_q=0.1, timeout=3.0)  # 轮询到位（配置缺省容差仅 0.005，故显式放宽）
    print(f"动作 open（目标 -5.0 rad）：{'已到位' if ok else '3 秒未到位'}（耗时 {dt:.2f} s）")  # open 结果
    arm.set_action_end("close")                           # 闭合到固定行程位 0.0 rad
    time.sleep(1.5)                                       # 等夹爪走完行程
    arm.set_action_end("home")                            # 回夹爪中位 -3.0 rad（即 home_end）
    ok, dt = wait_arrival(arm.is_in_q_end, arm.home_end, tol_q=0.1, timeout=3.0)  # 用 home_end 只读引用作到位目标
    print(f"动作 home（目标 home_end={vec_str(arm.home_end)} rad）：{'已到位' if ok else '3 秒未到位'}（耗时 {dt:.2f} s）")  # home 结果

    sub("连续位置指令 send_position_end")                 # 小节：连续位置指令
    q_goal_end = np.array([-4.0])                         # 目标电机位置（行程硬限位 [-5.49, 0.0] rad 内）
    print(f"下发 send_position_end：q_goal_end = {vec_str(q_goal_end)} rad（越硬限位就近裁剪）")  # 目标与保护
    arm.send_position_end(q_goal_end)                     # 连续位置指令整组下发（限速 / 限流取配置 POS_VEL）
    ok, dt = wait_arrival(arm.is_in_q_end, q_goal_end, tol_q=0.1, timeout=3.0)  # is_in_q_end 到位判断
    print(f"is_in_q_end 轮询：{'已到位' if ok else '3 秒未到位'}（耗时 {dt:.2f} s）")  # 到位结果
    print(f"动作后 end：{status_end(arm)}")               # 动作后夹爪位置（应约 -4.0 rad）

    sub("速度模式点动 send_vel_end（张开方向低速 0.5 rad/s × 0.5 s）")  # 小节：end 速度指令
    ret = arm.set_mode_end(ControlMode.VELOCITY)          # 切速度模式（end 侧同样三模式可选）
    print(f"set_mode_end(VELOCITY) 返回：{ret}")          # 切模式结果
    print(f"点动前 end：{status_end(arm)}")               # 点动前基线
    dq_end = np.full(n_end, -0.5)                         # 负方向 = 张开（行程 [-5.49, 0]，远离闭合端更安全）
    arm.send_vel_end(dq_end)                              # 整组下发目标速度（幅值越 dq_max 由库裁剪）
    time.sleep(0.5)                                       # 点动持续 0.5 秒（约张开 0.25 rad）
    arm.send_vel_end(np.zeros(n_end))                     # 先发零速停止，再做任何其他操作
    time.sleep(0.2)                                       # 等零速帧生效
    print(f"点动后 end：{status_end(arm)}（应比点动前再张开约 0.25 rad）")  # 点动效果核对
    ret = arm.set_mode_end(ControlMode.POSITION)          # 切回 POSITION（收尾前的稳定模式）
    print(f"set_mode_end(POSITION) 返回：{ret}；读回：{mode_str(arm.get_mode_end())}")  # 切换结果与读回

    # ---- ⑦ FSM 互斥说明 ----
    sec("⑦ FSM 互斥说明（本例程未启动运动状态机）")
    print("运动状态机（FSM）运行期间，电机由其内部控制线程独占：")  # 互斥背景
    print("  - arm 侧 enable/disable、set_mode、send_mit/position/vel、clear_error")  # 被拦截方法清单
    print("    等指令类方法会被 _fsm_guard 拦下：打一条 warn 提示，本次不执行")  # 拦截行为
    print("  - end 侧（夹爪）无控制线程：send_mit/position/vel_end、set_action_end 不受限")  # end 侧不受限
    print("本例程全程未调用 start_motion_fsm，上述指令均直通硬件；")  # 本例程状态
    print("FSM 的启停与状态切换请见例程 06（基于运动状态机的运动）。")  # 指向后续例程

    # ---- ⑧ 失能收尾 ----
    sec("⑧ 失能收尾")
    print(f"失能前 arm：{status_arm(arm)}")               # 失能前状态
    print(f"失能前 end：{status_end(arm)}")               # 失能前夹爪状态
    print(f"disable_arm 返回：{arm.disable_arm()}；disable_end 返回：{arm.disable_end()}")  # 1=全部成功
    print(f"失能后 arm：{status_arm(arm)}")               # 失能后应为 False / 模式缓存已清
    print(f"失能后 end：{status_end(arm)}")               # 失能后夹爪状态
    print("例程完成：机械臂已失能；with 退出将自动断开总线并完成安全收尾。")  # 收尾说明


def main() -> None:
    """真机入口：创建 JoyArm 对象并执行演示，Ctrl+C 时经 with 收尾后安全退出。"""
    model = parse_model()                                 # 必填型号参数
    try:                                                  # 捕获 Ctrl+C，经 with 收尾后友好退出
        with joyArm(model) as arm:                        # with 退出（含异常）自动 close：停线程 → 失能 → 断连
            run_demo(arm)                                 # 执行全部演示步骤
    except KeyboardInterrupt:                             # Ctrl+C：with 已完成安全收尾，此处仅提示
        print("\n检测到 Ctrl+C：例程已中断，机械臂已失能并断开总线，安全退出。")  # 安全退出提示


if __name__ == "__main__":
    main()                                                # 脚本直接运行时执行入口
