"""
==============================================================================
例程 05 —— 机器人学离线计算与求解器设置（全程离线）
==============================================================================
【功能概要】
    无需连接硬件，纯控制台体验 JoyArm 的机器人学离线计算与六域求解器管理：
      1. 六域 18 法总览与热切换：get_<域>_solvers 查名单、set_<域>_solver 切换、get_activated_<域>_solver 查激活（含非法名拒绝、traj 域先试规划成功才替换）；
      2. fkine 域 pin ↔ mdh 两种正运动学算法热切换，同一 q 对比结果差异；
      3. 离线数学门面：solve_fkine（末端帧 vs 基座系 frame=0）、solve_ikine 数值法往返、solve_ikine_all 解析法全解与数值法激活时的 RuntimeError 拦截；
      4. 雅可比 get_jac 与阻尼伪逆 get_jac_pinv_damped；
      5. 动力学四函数 get_mass_matrix / get_coriolis / get_gravity / solve_idyn（RNEA 一致性验证）；
      6. 到位判断 is_in_q_arm / is_in_pose_arm 的离线行为（读硬件快照恒 False）；
      7. rand_q_arm 缺省软限位采样与自定义限位采样（JointLimits 构造）；rand_q_end 末端电机侧同名采样（含无 end 段型号的行为说明）。

【原理简介】
    JoyArm 把机器人学六类算法各划为一「域」：正运动学 fkine、逆运动学 ikine、雅可比 jacobian、动力学 dynamics、轨迹规划 trajectory、控制 control。
    每域可加载多个求解器（config robotics 段的注册名，段内首个成员即默认激活），门面方法（solve_fkine / get_jac 等）只把计算转发给该域当前激活的求解器对象。
    set_<域>_solver 可运行期热切换——同一接口背后换算法，结果随之而变。
    数学门面的 q/dq 参数缺省取硬件被动快照 joint_state_arm：离线未连接时快照恒无效（t=0、q=NaN），缺省调用会直接报错。
    故本脚本全部显式传 q——只需 URDF 数字模型即可完成全部计算。
    到位判断 is_in_q_arm / is_in_pose_arm 同样读该快照：离线无新鲜状态帧恒返回 False（设计内行为，真机在线后才以当前关节状态参与判定，真机路径见例程 04）。

【安全须知】
    - 本例程全程离线：不连接硬件、不上电、不下发任何控制指令，无运动风险
    - set_target_joint / set_target_trajframe 仅写入对象内的目标数据，同样不触硬件

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（全程离线，无需连接硬件）
    python quick_start/05_offline_math_and_solvers.py <型号>    # 如 rebot_arm_b601_dm

【预期输出】
    ①~③ 三大节依次打印：六域求解器总览与切换结果表、FK/IK/雅可比/动力学数值对比、到位判断离线行为说明，末尾打印「离线例程运行完毕」。
    期间伴随的库 warn 日志（到位判断无快照、traj 试规划失败、非法求解器名、close 豁免）均为预期现象而非报错；随机采样的数值每次运行略有不同。
==============================================================================
"""

from pathlib import Path                                   # 路径工具：定位仓库根目录
import sys                                                 # 系统接口：修改模块搜索路径
import time                                                # 时间接口：构造未来时刻的目标帧
import unicodedata                                         # 字符宽度判定：表格按终端显示宽对齐

import numpy as np                                         # 数值计算：向量构造与误差度量

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 把 joyarm_code/ 加入搜索路径，未安装库也可运行
# JoyArm 创建入口与关节限位类型（JointLimits 定义在 joyarm_core/utils/types.py，顶层 re-export）
from joyarm_core import JointLimits, joyArm  # noqa: E402


# ============================================================
# 一、命令行型号参数：解析必填的型号入参，缺失即打印用法退出
# ============================================================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并以退出码 1 结束。"""
    if len(sys.argv) != 2:                                # 必须恰好携带 1 个型号参数
        sys.exit(f"用法：python quick_start/05_offline_math_and_solvers.py <型号>")     # 缺参退出，不进入演示
    return sys.argv[1].strip()                            # 型号名


# ============================================================
# 二、输出格式化工具：节标题、表格列对齐与数值显示
# ============================================================
def sec(title: str) -> None:
    """打印大节标题（上下各一条 78 字符 "=" 分隔线）。"""
    print("\n" + "=" * 78)                                 # 大节分隔线（前空一行更醒目）
    print(title)                                           # 大节标题本体
    print("=" * 78)                                        # 大节分隔线（下）


def sub(title: str) -> None:
    """打印小节标题（上下各一条 78 字符 "-" 分隔线）。"""
    print("-" * 78)                                        # 小节分隔线（上）
    print(title)                                           # 小节标题本体
    print("-" * 78)                                        # 小节分隔线（下）


def fmt(v, nd: int = 3) -> str:
    """数值向量转定宽字符串：None 显「—」，其余逐元素保留 nd 位小数对齐。"""
    if v is None:                                          # None = 缺席/未知，统一显破折号
        return "—"                                         # 缺省显示占位符
    return "[" + " ".join(f"{x: .{nd}f}" for x in np.asarray(v, float)) + "]"  # 每元素定宽 7 字符对齐


def disp_width(s: str) -> int:
    """字符串的终端显示宽度：全角/宽字符（中文等）计 2、半角计 1。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "FW" else 1 for ch in s)  # 仅 F/W 两类算宽字符


def pad(s: str, width: int, align: str = "left") -> str:
    """按终端显示宽度补空格保证列对齐：缺省左对齐，align="right" 时右对齐。"""
    gap = " " * max(0, width - disp_width(s))              # 缺口 = 目标宽 − 显示宽（不为负）
    return gap + s if align == "right" else s + gap        # 左对齐右补 / 右对齐左补；超宽不截断


def flag(ok: bool) -> str:
    """set_<域>_solver 返回值转可读标记：True → 「成功」，False → 「拒绝」。"""
    return "成功" if ok else "拒绝"                            # 拒绝 = 返回 False，保持原激活不变


# ============================================================
# 三、求解器六域管理演示：查询、热切换与 pin↔mdh 算法对比（小节 1.1~1.3）
# ============================================================
def demo_solver_overview(arm) -> None:
    """1.1 查询 12 法：六域各调 get_<域>_solvers() 与 get_activated_<域>_solver()。"""
    sub("1.1 查询 12 法：get_<域>_solvers()（已加载名单）+ get_activated_<域>_solver()（当前激活）")
    domains = [                                            # (域显示名, 已加载查询, 激活查询) 三元组列表
        ("fkine", arm.get_fkine_solvers, arm.get_activated_fkine_solver),      # 正运动学域
        ("ikine", arm.get_ikine_solvers, arm.get_activated_ikine_solver),      # 逆运动学域
        ("jacobian", arm.get_jac_solvers, arm.get_activated_jac_solver),       # 雅可比域（方法名用缩写 jac）
        ("dynamics", arm.get_dyn_solvers, arm.get_activated_dyn_solver),       # 动力学域（方法名用缩写 dyn）
        ("trajectory", arm.get_traj_solvers, arm.get_activated_traj_solver),   # 轨迹规划域
        ("control", arm.get_ctrl_solvers, arm.get_activated_ctrl_solver),      # 控制域
    ]
    for name, get_all, get_act in domains:                 # 六域逐一查询，每域两行对齐输出
        print(f"   {pad(name, 12)}已加载：{'、'.join(get_all())}")   # 第一行：域内全部注册名（顿号分隔）
        print(f"   {pad('', 12)}当前激活：{get_act()}")      # 第二行：当前参与计算的求解器
    print("   说明：已加载名单来自 config robotics 各段；段内首个成员即默认激活；")
    print("         方法名规律——雅可比/动力学域用缩写 jac/dyn，其余用全称。")


def demo_solver_set(arm) -> None:
    """1.2 设置 6 法：六域各调 set_<域>_solver(name)，traj 域演示三条替换路径。"""
    sub("1.2 设置 6 法：set_<域>_solver(name) 热切换（traj 域先试规划成功才替换）")
    print(f"   {pad('域', 12)}{pad('set 动作', 30)}{pad('返回', 8)}切换后激活")       # 表头四列
    rows = []                                              # (域, 动作, 返回值, 切后激活) 待打印行

    # ---- ① 六域常规热切换动作：重选自身 / 切换算法 / 非法名拒绝 ----
    rows.append(("jacobian", "重选自身（域内仅一个）", arm.set_jac_solver(arm.get_activated_jac_solver()),
                 arm.get_activated_jac_solver()))          # jacobian 域：重选自身（域内仅 1 个成员）
    rows.append(("dynamics", "重选自身（域内仅一个）", arm.set_dyn_solver(arm.get_activated_dyn_solver()),
                 arm.get_activated_dyn_solver()))          # dynamics 域：重选自身
    rows.append(("control", "重选自身（域内仅一个）", arm.set_ctrl_solver(arm.get_activated_ctrl_solver()),
                 arm.get_activated_ctrl_solver()))         # control 域：重选自身（离线可选，不触硬件）
    rows.append(("ikine", "切解析法", arm.set_ikine_solver("analytic_ikine_solver"),
                 arm.get_activated_ikine_solver()))        # ikine 域：数值法 → 解析法（可求全解）
    rows.append(("ikine", "切回数值法", arm.set_ikine_solver("pin_ikine_solver"),
                 arm.get_activated_ikine_solver()))        # ikine 域：解析法 → 数值法（切回默认）
    rows.append(("fkine", "切 mdh 解析式", arm.set_fkine_solver("mdh_fkine_solver"),
                 arm.get_activated_fkine_solver()))        # fkine 域：URDF pinocchio → cfg MDH 解析式
    rows.append(("fkine", "非法名 no_such_solver", arm.set_fkine_solver("no_such_solver"),
                 arm.get_activated_fkine_solver()))        # fkine 域：非法名——warn 并保持原激活
    rows.append(("fkine", "切回 pin", arm.set_fkine_solver("pin_fkine_solver"),
                 arm.get_activated_fkine_solver()))        # fkine 域：切回默认 pin（对比见 1.3）

    # ---- ② trajectory 域三路径：新求解器先对当前目标帧试规划一次，成功才替换激活（防半成品接管在途轨迹）----
    rows.append(("trajectory", "a) 空目标帧切 cubic", arm.set_traj_solver("single_joint_cubic_traj_planner"),
                 arm.get_activated_traj_solver()))         # 路径 a：目标帧为空 → cubic 回退恒定 home 段，仍算规划成功
    arm.set_target_joint(time.time() + 5.0, arm.home_arm.copy())  # 写一个 5 秒后的单帧关节目标（纯数据，不触硬件）
    rows.append(("trajectory", "b) 未来目标帧切 hold", arm.set_traj_solver("single_joint_hold_traj_planner"),
                 arm.get_activated_traj_solver()))         # 路径 b：hold 直传不读硬件状态 → 规划成功
    rows.append(("trajectory", "c) 未来目标帧切 cubic", arm.set_traj_solver("single_joint_cubic_traj_planner"),
                 arm.get_activated_traj_solver()))         # 路径 c：cubic 需 get_state_arm 取规划起点，离线无状态 → 试规划异常
    arm.set_target_trajframe(None)                         # 清空目标帧（收尾，恢复初始空目标状态）

    for name, action, ok, act in rows:                     # 逐行按显示宽度对齐打印结果表
        print(f"   {pad(name, 12)}{pad(action, 30)}{pad(flag(ok), 8)}{act}")    # 域 | 动作 | 返回 | 切后激活
    print("   说明：「拒绝」= 返回 False 并保持原激活（库同时打 warn 日志）：非法名不在已加载")  # 拒绝语义
    print("         名单；路径 c 因离线取不到规划起点试规划失败。真机上 get_state_arm 可读到状态，")  # 路径 c 离线失败原因
    print("         路径 c 与 a/b 同样切换成功——traj 域这条「先试规划才替换」规则保证不接管失败。")  # traj 域规则小结


def demo_fkine_compare(arm) -> None:
    """1.3 fkine 域 pin ↔ mdh 切换对比：同一 q 的两次 solve_fkine 结果差异。"""
    sub("1.3 fkine 域 pin ↔ mdh 切换：同一 q 的 solve_fkine 结果对比")
    q = arm.home_arm                                        # 取 home 位形作为对比输入
    pose_pin = arm.solve_fkine(q)                           # pin 激活：URDF 模型 + pinocchio 计算
    arm.set_fkine_solver("mdh_fkine_solver")                # 热切换到 mdh 解析式实现
    pose_mdh = arm.solve_fkine(q)                           # mdh 激活：cfg 名义 MDH 参数连乘齐次变换
    arm.set_fkine_solver("pin_fkine_solver")                # 切回默认 pin（后续小节继续用它）
    d_pos = float(np.linalg.norm(np.asarray(pose_pin.position) - np.asarray(pose_mdh.position)))  # 位置差范数
    print(f"   pin 末端位置 (m)：{fmt(pose_pin.position, 6)}")    # pin 结果
    print(f"   mdh 末端位置 (m)：{fmt(pose_mdh.position, 6)}")    # mdh 结果
    print(f"   四元数 pin / mdh：{fmt(pose_pin.orientation, 4)} / {fmt(pose_mdh.orientation, 4)}")  # 姿态对比
    print(f"   位置差范数 ‖Δp‖ = {d_pos:.2e} m —— cfg 的 MDH 参数由理想化 URDF 生成，")  # 差异来源
    print("   两套算法同源，差异仅为浮点精度量级；mdh 实现仅支持末端帧（frame 传其他名会报错）。")  # 结论与 mdh 限制


# ============================================================
# 四、离线数学门面演示：正/逆运动学、雅可比与动力学（小节 2.1~2.4）
# ============================================================
def demo_fkine(arm) -> None:
    """2.1 正运动学 solve_fkine：home 与随机采样输入；末端帧（缺省）与基座系 frame=0。"""
    sub("2.1 solve_fkine：home + 随机采样；末端帧（缺省）与基座系（frame=0）")
    samples = [                                            # 对比输入组：home 位形 + 两组随机采样
        ("home", arm.home_arm),                            # 归位位形（确定值，每次运行相同）
        ("随机 1", arm.rand_q_arm()),                      # 缺省在软限位内均匀采样
        ("随机 2", arm.rand_q_arm()),                      # 再采一组
    ]
    for label, q in samples:                               # 逐组求解并打印
        pose = arm.solve_fkine(q)                          # q → 末端位姿（frame 缺省 = cfg joyarm.frame_end）
        print(f"   [{label}] q (rad)      = {fmt(q)}")     # 输入关节角
        print(f"          末端位置 (m)   = {fmt(pose.position, 4)}")   # 位置 xyz
        print(f"          末端四元数     = {fmt(pose.orientation, 4)}")  # 姿态 (w, x, y, z)
    pose_base = arm.solve_fkine(arm.home_arm, frame=0)     # frame=0：基座（universe）帧在基座系下的位姿
    print(f"   frame=0（基座系）位姿：位置 {fmt(pose_base.position, 4)}，"
          f"四元数 {fmt(pose_base.orientation, 4)}")       # 基座帧对自身恒为恒等变换（位置 0、四元数 [1,0,0,0]）
    print("   说明：frame 可传帧名（默认末端帧）或帧索引（0 = 基座系），返回该帧在基座系下的位姿。")


def demo_ikine(arm) -> None:
    """2.2 逆运动学：数值法 solve_ikine 往返 + 解析法 solve_ikine_all 全解与拦截。"""
    sub("2.2 solve_ikine 往返与 solve_ikine_all 全解（数值法激活时被 RuntimeError 拦截）")
    q_tgt = arm.home_arm + np.array([0.3, -0.3, 0.3, 0.3, 0.3, 0.3])  # 固定扰动目标（软限位内，可复现）
    pose_tgt = arm.solve_fkine(q_tgt)                      # 先用 FK 得到目标位姿（IK 的输入是位姿不是关节角）
    res = arm.solve_ikine(pose_tgt, q0=arm.home_arm)       # 数值法：自 q0=home 迭代收敛到单解
    d_q = float(np.max(np.abs(res.q - q_tgt)))             # 解与原目标关节角的最大偏差（往返误差）
    print(f"   数值法：success={res.success}，err={res.err:.2e}，n_iter={res.n_iter}，"
          f"max|q_解 − q_目标|={d_q:.2e} rad")             # 收敛信息 + 往返一致性
    q_rand = arm.rand_q_arm()                              # 随机目标再来一次往返（验证一般位姿）
    res_rand = arm.solve_ikine(arm.solve_fkine(q_rand), q0=arm.home_arm)  # FK(rand) 作目标、home 作初值
    print(f"   随机往返：success={res_rand.success}，err={res_rand.err:.2e}，n_iter={res_rand.n_iter}")

    arm.set_ikine_solver("analytic_ikine_solver")          # 切解析法：枚举肩/肘/腕分支，可一次求出全部解
    res_all = arm.solve_ikine_all(pose_tgt)                # 全解：q 形状 (K, n_arm)，每行一组解析解
    print(f"   解析法 solve_ikine_all：success={res_all.success}，共 K = {res_all.q.shape[0]} 组解")
    for k, q_k in enumerate(res_all.q, start=1):          # 逐行打印各组解（同一末端位姿的多个关节构型）
        print(f"      解 {k}：{fmt(q_k)}")                 # 第 k 组解
    arm.set_ikine_solver("pin_ikine_solver")               # 切回数值法
    try:                                                   # 数值法激活时演示门面拦截
        arm.solve_ikine_all(pose_tgt)                      # 数值法只能给单解、无 solve_all 实现
        print("   ✘ 未拦截（不应发生）")                    # 到这里说明库行为变了（与本脚本预期不符）
    except RuntimeError as e:                              # 门面把 NotImplementedError 转成 RuntimeError
        print(f"   数值法激活时 solve_ikine_all 被 RuntimeError 拦截 ✓（提示切换解析法）")
    soft = arm.joint_limits_soft_arm                       # 软限位（用于核对全解是否越限）
    n_over = int(np.sum(np.any((res_all.q < soft.q_min) | (res_all.q > soft.q_max), axis=1)))  # 统计任一关节超软限位的解数
    print("   说明：solve_ikine 单解（数值法迭代 / 解析法选离 q0 最近解）；全解仅解析法实现，")
    print(f"         且不剔除越限解——本例 {n_over}/{res_all.q.shape[0]} 组超出软限位，选用前需自行核对关节限位。")


def demo_jac(arm) -> None:
    """2.3 雅可比：get_jac 与阻尼伪逆 get_jac_pinv_damped（damping=1e-3）。"""
    sub("2.3 get_jac 与 get_jac_pinv_damped（damping=1e-3）")
    q = arm.home_arm                                        # 以 home 位形计算
    J = arm.get_jac(q)                                      # 基座系雅可比 J(q)：形状 (6, n_arm)
    Jp = arm.get_jac_pinv_damped(q, damping=1e-3)           # 阻尼最小二乘广义逆 J* = Jᵀ(JJᵀ + λ²I)⁻¹
    print(f"   J 形状 {J.shape}，J* 形状 {Jp.shape}")       # (6,6) 与 (6,6)（该型号 n_arm = 6）
    print(f"   J (rad→m/rad 缩放矩阵，逐元素 3 位小数)：\n{np.round(J, 3)}")  # 雅可比本体
    resid = float(np.max(np.abs(J @ Jp - np.eye(6))))       # J·J* 与单位阵的最大偏差（阻尼引入的收缩）
    print(f"   max|J·J* − I₆| = {resid:.2e} —— 阻尼项 λ² 使奇异方向收缩、伪逆恒有界，")
    print("   远离奇异位形时偏差极小；越接近奇异偏差越大（这正是阻尼伪逆的设计目的）。")


def demo_dynamics(arm) -> None:
    """2.4 动力学四函数：质量矩阵 / 科氏 / 重力 / 逆动力学（RNEA 一致性验证）。"""
    sub("2.4 动力学：get_mass_matrix / get_coriolis / get_gravity / solve_idyn")
    q, dq, ddq = arm.home_arm, np.full(arm.n_arm, 0.5), np.ones(arm.n_arm)  # 位置 home、速度 0.5、加速度 1.0
    M = arm.get_mass_matrix(q)                              # 关节空间惯量阵 M(q)：(n_arm, n_arm)
    C0 = arm.get_coriolis(q, np.zeros(arm.n_arm))           # 科氏项 C(q,dq)·dq：dq=0 时恒为零向量
    C = arm.get_coriolis(q, dq)                             # 科氏项：dq 非零时出现耦合力矩
    G = arm.get_gravity(q)                                  # 重力项 G(q)：各关节需克服的重力力矩
    tau = arm.solve_idyn(q, dq, ddq)                        # 逆动力学 RNEA：(q,dq,ddq) → 总关节力矩
    resid = float(np.max(np.abs(tau - (M @ ddq + C + G))))  # RNEA 与 M·ddq + C + G 的最大偏差
    print(f"   M 对角元（各关节等效转动惯量 kg·m²）：{fmt(np.diag(M), 4)}")  # 惯量阵对角
    print(f"   C(q, 0)（零速度无科氏）  = {fmt(C0, 4)}")    # 验证 dq=0 → 科氏为零
    print(f"   C(q, 0.5)（科氏力矩）   = {fmt(C, 4)} N·m")  # 非零速度的科氏/向心力矩
    print(f"   G(q)（重力力矩）        = {fmt(G, 4)} N·m")  # 重力项
    print(f"   solve_idyn(q, dq, ddq)  = {fmt(tau, 4)} N·m")  # RNEA 总力矩
    print(f"   一致性 max|tau − (M·ddq + C + G)| = {resid:.2e} —— 两种写法同一物理，偏差为浮点精度")


# ============================================================
# 五、到位判断与随机采样演示：离线恒 False 机制与双限位采样（小节 3.1~3.2）
# ============================================================
def demo_arrival(arm) -> None:
    """3.1 到位判断离线行为：is_in_q_arm / is_in_pose_arm 读硬件被动快照恒 False。"""
    sub("3.1 is_in_q_arm / is_in_pose_arm：读硬件被动快照，离线未连接恒返回 False")
    q_perturbed = arm.home_arm.copy()                      # 构造加扰动的目标：第 1 关节 +0.1 rad
    q_perturbed[0] += 0.1                                  # 与 home 明显不同（超出到位容差）
    print(f"   is_in_q_arm(home_arm)         = {arm.is_in_q_arm(arm.home_arm)}")     # 目标=home：离线仍 False
    print(f"   is_in_q_arm(home + 0.1 rad)   = {arm.is_in_q_arm(q_perturbed)}")      # 加扰动：同样 False
    print(f"   is_in_pose_arm(fk(home))      = {arm.is_in_pose_arm(arm.solve_fkine(arm.home_arm))}")  # 位姿自判
    print("   机制：这三族函数把「当前实际关节状态」与目标比对，而当前状态取自硬件被动")  # 判定机制第一句
    print("   快照 joint_state_arm（新鲜度门 0.1 s）；离线未连接时快照恒无效（t=0、q=NaN），")  # 快照新鲜度门
    print("   故无论目标是什么一律判「未到位」False——这是设计内行为，不是计算错误；伴随的")  # 离线恒 False 的结论
    print("   「无可用关节状态快照」warn 日志属预期。真机在线后快照新鲜，目标进入容差即 True")  # warn 属预期 + 真机行为
    print("   （is_in_q_arm 的真机轮询到位路径见例程 04；is_in_pose_arm 同理，真机快照有效后参与判定）。")  # 指向真机例程


def demo_rand_q(arm) -> None:
    """3.2 rand_q_arm / rand_q_end：缺省软限位采样与自定义 JointLimits 限位采样。"""
    sub("3.2 rand_q_arm / rand_q_end：缺省软限位采样 vs 自定义限位采样（JointLimits 构造）")

    # ---- ① rand_q_arm：缺省软限位采样与自定义限位采样（JointLimits 构造）----
    soft = arm.joint_limits_soft_arm                       # 缺省采样限位：软限位（只读 property）
    q_soft = arm.rand_q_arm()                              # 缺省调用：在软限位 [q_min, q_max] 内均匀采样
    over_soft = float(np.max(np.maximum(soft.q_min - q_soft, q_soft - soft.q_max)))  # 相对软限位的越界量（≤0 即在内）
    center = (soft.q_min + soft.q_max) / 2.0               # 以软限位中点为中心构造自定义窄带限位
    custom = JointLimits(q_min=center - 0.2, q_max=center + 0.2)  # 自定义限位：中点 ±0.2 rad（仅需 q_min/q_max 两字段）
    q_custom = arm.rand_q_arm(custom)                      # 传入自定义限位：在该窄带内均匀采样
    over_custom = float(np.max(np.maximum(custom.q_min - q_custom, q_custom - custom.q_max)))  # 相对自定义带的越界量
    print(f"   {pad('采样方式', 12)}限位来源")              # 表头：两行式表格，第一行为限位来源
    print(f"   {pad('缺省调用', 12)}软限位 joint_limits_soft_arm（property）")  # 缺省行的限位来源
    print(f"   {pad('', 12)}q (rad) = {fmt(q_soft)}   越界量 {over_soft:.2e}")   # 缺省采样的结果与越界量
    print(f"   {pad('自定义限位', 12)}JointLimits(q_min=软限位中点−0.2, q_max=中点+0.2)")  # 自定义行的限位来源
    print(f"   {pad('', 12)}q (rad) = {fmt(q_custom)}   越界量 {over_custom:.2e}")  # 自定义采样的结果与越界量
    print("   说明：越界量 ≤ 0 表示采样落在对应限位内；JointLimits 其余字段（dq_max/tau_max）")  # 越界量判读
    print("         缺省 ±inf，对位置采样无影响；采样均匀且独立（每次运行结果不同）。")  # 其余字段行为

    # ---- ② rand_q_end：末端电机侧的同名采样接口（与 rand_q_arm 完全对称）----
    soft_end = arm.joint_limits_soft_end                   # end 侧软限位（无 end 段型号为 None）
    if soft_end is None:                                   # 无 end 段：采样返回全零向量（无实际意义）
        print("   rand_q_end：本型号无 end 软限位段，采样返回全零向量。")  # 无 end 段行为说明
    else:
        q_end = arm.rand_q_end()                           # 缺省调用：在 end 软限位内均匀采样
        over_end = float(np.max(np.maximum(soft_end.q_min - q_end, q_end - soft_end.q_max)))  # 相对 end 软限位的越界量
        print(f"   rand_q_end() = {fmt(q_end)} rad（缺省 end 软限位内采样，越界量 {over_end:.2e}）")  # 采样结果
        print("   说明：末端夹爪为单自由度固定行程（open/close/home 三档已覆盖常用动作），")  # 使用场景说明
        print("         rand_q_end 主要用于算法测试等一般化采样需求，日常操作用 set_action_end。")  # 指向离散动作


# ============================================================
# 六、主流程：创建对象 → 三大节演示 → 安全收尾
# ============================================================
def main() -> None:
    """离线主流程：创建对象 → 六域求解器管理 → 数学门面 → 到位判断与采样 → 收尾。"""
    model = parse_model()                                  # 必填型号参数
    sec("创建 JoyArm（离线加载 URDF 数字模型与六域求解器，不连接任何硬件）")
    arm = joyArm(model)                                    # 离线创建：装配模型与算法，全程不触硬件
    print(arm)                                             # __str__：多行表格摘要（与例程 01 相同）

    sec("① 求解器六域管理：get / set / get_activated 三类共 18 法")
    demo_solver_overview(arm)                              # 1.1 查询 12 法（六域 × 两个查询入口）
    demo_solver_set(arm)                                   # 1.2 设置 6 法（含 traj 三路径与非法名拒绝）
    demo_fkine_compare(arm)                                # 1.3 fkine 域 pin ↔ mdh 切换对比

    sec("② 离线数学门面：solve_fkine / solve_ikine / get_jac / 动力学（全部显式传 q）")
    print("   门面 q/dq 缺省取硬件被动快照，离线无效会报错——本节全部显式传参，纯模型即可计算。")
    demo_fkine(arm)                                        # 2.1 正运动学：采样输入 + 末端帧/基座系
    demo_ikine(arm)                                        # 2.2 逆运动学：往返 + 全解 + RuntimeError 拦截
    demo_jac(arm)                                          # 2.3 雅可比与阻尼伪逆
    demo_dynamics(arm)                                     # 2.4 动力学四函数与 RNEA 一致性

    sec("③ 到位判断离线行为与 rand_q_arm / rand_q_end 双限位采样")
    demo_arrival(arm)                                      # 3.1 is_in_* 离线恒 False 的机制演示
    demo_rand_q(arm)                                       # 3.2 缺省软限位 / 自定义限位采样

    print("=" * 78)                                        # 收尾分隔线
    arm.close()                                            # 显式安全收尾（幂等；离线时打印豁免日志属正常）
    print("离线例程运行完毕（close 时「未连接……豁免跳过」类日志为正常提示）")


if __name__ == "__main__":
    main()                                                 # 作为脚本直接运行时执行主流程
