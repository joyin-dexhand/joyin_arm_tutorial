"""
==============================================================================
例程 05 —— 机器人学在线计算与求解器设置（需真机）
==============================================================================
【功能概要】
    1. 求解器六域查询：get_<域>_solvers 已加载 与 get_activated_<域>_solver 当前激活；
    2. 求解器热切换：set_<域>_solver 切换
    3. 数学计算（solve_fkine / solve_ikine / get_jac / 动力学）：q/dq 缺省取真机状态缓存
    4. 重力补偿演示：MIT 模式 send_mit_arm
    5. 到位判断 is_in_q_* / is_in_pose_arm；rand_q_arm / rand_q_end 关节随机采样。

【原理简介】
    机器人学六类算法各划为一「域」：fkine / ikine / jacobian / dynamics / trajectory / control；
    每域可加载多个求解器（cfg.robotics 段注册，首个为默认激活）；
    统一入口（solve_fkine、get_jac 等）把计算转发给当前激活的求解器，set_<域>_solver 负责切换求解器；
    机器人学求解方法的 q/dq 参数缺省取硬件真实状态缓存；显式传参则可离线计算；
    到位判断同样读硬件实时状态缓存，目标值位于容差内则判为 True。

【安全须知】
    - 运行前清空机械臂工作半径内的人员与障碍物，机械臂稳固固定后再上电
    - 机械臂运行异常立即急停或断电，随时可 Ctrl+C 打断并进入安全收尾

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（需连接对应的机械臂）
    python quick_start/05_robotics_solvers_and_math.py <型号>  # 如：rebot_arm_b601_dm

【预期输出】
    连接使能 / 状态码 → 六域名单 / 激活与热切换结果 → FK / IK / 雅可比 / 动力学数值 → 到位判断 → 重力补偿 → 失能收尾；

==============================================================================
"""

from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from joyarm_core import ControlMode, JointLimits, joyArm, listArm 


# ================= 命令行参数解析：型号必填，缺失即打印用法退出 ======================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并列出可用型号（listArm），以退出码 1 结束。"""
    if len(sys.argv) != 2:                                # 必须恰好携带 1 个型号参数
        print(f"用法：python quick_start/05_robotics_solvers_and_math.py <型号>")  # 缺参：先打印用法
        listArm()                                         # 顺带列出全部可用型号
        sys.exit(1)                                       # 退出码 1，不进入演示
    return sys.argv[1].strip()                            # 型号名


# ======= main() 流程：连接使能 → 求解器六域 → 数学计算 → 到位判断与采样 → 重力补偿 → 收尾 ====
def main() -> None:
    """真机流程：连接使能 → 求解器六域查询与热切换 → 数学计算（缺省读当前状态）→ 到位判断与采样 → 重力补偿（默认跳过）→ 失能收尾。"""
    model = parse_model()                                 # 必填型号参数

    with joyArm(model) as arm:                            # with 退出自动 close：停线程 → 失能 → 断连

        print("=============== ① 连接与使能 ===================================== ")
        ok = arm.connect_arm()                            # 连接本体电机总线
        print(f"connect_arm 返回：{ok}；is_connected_arm：{arm.is_connected_arm}")
        ok = arm.connect_end()                            # 连接末端（夹爪）电机总线
        print(f"connect_end 返回：{ok}；is_connected_end：{arm.is_connected_end}")
        ret = arm.enable_arm()                            # 使能本体全部关节电机：1=全部成功、0=有失败
        print(f"enable_arm 返回：{ret}；使能后 is_enabled_arm：{arm.is_enabled_arm}")
        ret = arm.enable_end()                            # 使能末端电机（返回 1/0，含义同上）
        print(f"enable_end 返回：{ret}；使能后 is_enabled_end：{arm.is_enabled_end}")
        time.sleep(0.01)                                  # 等待电机使能并状态更新完毕
        print(f"arm 状态码：{arm.get_error_arm()}；check_error_arm：{arm.check_error_arm()}")
        print(f"end 状态码：{arm.get_error_end()}；check_error_end：{arm.check_error_end()}")

        print("=============== ② 求解器六域方法：查询与热切换 ================= ")
        print("—— 查询：get_<域>_solvers 已加载名单 + get_activated_<域>_solver 当前激活 ——")
        domains = [                                       # (域名, 全部名单, 当前激活)
            ("fkine", arm.get_fkine_solvers, arm.get_activated_fkine_solver),
            ("ikine", arm.get_ikine_solvers, arm.get_activated_ikine_solver),
            ("jacobian", arm.get_jac_solvers, arm.get_activated_jac_solver),
            ("dynamics", arm.get_dyn_solvers, arm.get_activated_dyn_solver),
            ("trajectory", arm.get_traj_solvers, arm.get_activated_traj_solver),
            ("control", arm.get_ctrl_solvers, arm.get_activated_ctrl_solver),
        ]
        for name, get_all, get_act in domains:            # 每域两行：已加载名单 + 当前激活
            print(f"   {name:<12}已加载：{'、'.join(get_all())}")
            print(f"   {'':<12}当前激活：{get_act()}")
        print("   说明：已加载名单来自 cfg.robotics 的配置，首个默认激活。")

        print("—— 热切换：set_<域>_solver(name)，返回 True=成功、False=拒绝 ——")
        ret = arm.set_fkine_solver("pin_fkine_solver")
        print(f"   set_fkine_solver('pin_fkine_solver') 返回：{ret}；当前激活：{arm.get_activated_fkine_solver()}")
        ret = arm.set_ikine_solver("pin_ikine_solver")
        print(f"   set_ikine_solver('pin_ikine_solver') 返回：{ret}；当前激活：{arm.get_activated_ikine_solver()}")
        ret = arm.set_jac_solver("pin_jacobian_solver")
        print(f"   set_jac_solver('pin_jacobian_solver') 返回：{ret}；当前激活：{arm.get_activated_jac_solver()}")
        ret = arm.set_dyn_solver("pin_dynamics_solver")
        print(f"   set_dyn_solver('pin_dynamics_solver') 返回：{ret}；当前激活：{arm.get_activated_dyn_solver()}")
        ret = arm.set_traj_solver("single_joint_cubic_traj_planner") 
        print(f"   set_traj_solver('single_joint_cubic_traj_planner') 返回：{ret}；当前激活：{arm.get_activated_traj_solver()}")
        ret = arm.set_ctrl_solver("joint_position_controller")
        print(f"   set_ctrl_solver('joint_position_controller') 返回：{ret}；当前激活：{arm.get_activated_ctrl_solver()}")

        print("—— pin ↔ mdh 对比：同一当前位形的 solve_fkine 结果 ——")
        pose_pin = arm.solve_fkine()                      # pin 激活：URDF 模型 + pinocchio 计算
        arm.set_fkine_solver("mdh_fkine_solver")          # 切换到 MDH 解析式方式求解
        pose_mdh = arm.solve_fkine()                      # mdh 激活：cfg MDH 位姿矩阵连乘变换
        arm.set_fkine_solver("pin_fkine_solver")          # 切回默认 pin
        d_pos = float(np.linalg.norm(np.asarray(pose_pin.position) - np.asarray(pose_mdh.position)))
        print(f"   pin 末端位置 (m)：{np.round(pose_pin.position, 6).tolist()}")
        print(f"   mdh 末端位置 (m)：{np.round(pose_mdh.position, 6).tolist()}")
        print(f"   位置差范数 {d_pos:.2e} m")

        print("=============== ③ 数学计算：FK / IK / 雅可比 / 动力学 ============ ")
        print("—— solve_fkine：参数 q 缺省取真机状态缓存）————————————————")
        q_cur = np.asarray(arm.joint_state_arm.q, dtype=float)  # 当前 q 
        print(f"   [当前] q (rad)          = {np.round(q_cur, 3).tolist()}")
        pose_cur = arm.solve_fkine()                      # 当前 pose（q 缺省取真机状态缓存）
        print(f"          末端位置 (m)     = {np.round(pose_cur.position, 4).tolist()}")
        print(f"          末端四元数 (wxyz) = {np.round(pose_cur.orientation, 4).tolist()}")
        # 显式传 q ，支持离线计算：
        # pose = arm.solve_fkine(arm.home_arm)          # 指定位形 home
        # pose = arm.solve_fkine(arm.rand_q_arm())      # 软限位内随机位形

        print("—— solve_ikine 数值法往返 与 solve_ikine_all 解析法全解 ——")
        pose_tgt = arm.solve_fkine()                      # 目标位姿 = FK 当前状态
        res = arm.solve_ikine(pose_tgt)                   # q0 缺省取当前真机状态缓存为迭代起点
        d_q = float(np.max(np.abs(res.q - q_cur)))        # 解与当前 q 的最大偏差（往返误差）
        print(f"   数值法：success={res.success}，err={res.err:.2e}，n_iter={res.n_iter}，max|q解−q当前|={d_q:.2e} rad")
        arm.set_ikine_solver("analytic_ikine_solver")     # 切解析法
        res_all = arm.solve_ikine_all(pose_tgt)           # 全解：形状 (K, n_arm)，每行一组解
        print(f"   解析法 solve_ikine_all：success={res_all.success}，共 {res_all.q.shape[0]} 组解")
        for k, q_k in enumerate(res_all.q, start=1):
            print(f"      解 {k}：{np.round(q_k, 3).tolist()}")
        arm.set_ikine_solver("pin_ikine_solver")          # 切回数值法
        # 显式传 pose 和 q0，支持离线计算：
        # q_tgt = arm.home_arm + np.array([0.3, -0.3, 0.3, 0.3, 0.3, 0.3])    
        # res = arm.solve_ikine(arm.solve_fkine(q_tgt), q0=arm.home_arm) 

        print("—— get_jac 与 get_jac_pinv_damped（damping=1e-3）——")
        J = arm.get_jac()                                 # 基座系下的雅可比 (6, n_arm)，q 缺省取当前真机状态缓存
        Jp = arm.get_jac_pinv_damped(damping=1e-3)        # 阻尼伪逆 J* = Jᵀ(JJᵀ + λ²I)⁻¹，q 缺省取当前真机状态缓存
        print(f"   J 形状 {J.shape}，J* 形状 {Jp.shape}；J：\n{np.round(J, 3)}")
        resid = float(np.max(np.abs(J @ Jp - np.eye(6))))     # J·J* 与单位阵的最大偏差（阻尼引入的收缩）
        print(f"   max|J·J* − I| = {resid:.2e} —— 阻尼项使奇异方向收缩、伪逆恒有界，远离奇异位形时偏差极小")
        # 显式传 q，支持离线计算：
        # J = arm.get_jac(arm.home_arm)
        # Jp = arm.get_jac_pinv_damped(arm.home_arm, damping=1e-3)

        print("—— 动力学：get_mass_matrix / get_coriolis / get_gravity / solve_idyn ——")
        M = arm.get_mass_matrix()                         # 关节空间惯量阵 (n_arm, n_arm)，q 缺省取当前真机状态缓存
        C = arm.get_coriolis()                            # 科氏项（静止时 ≈ 0），q/dq 缺省取当前真机状态缓存
        G = arm.get_gravity()                             # 重力力矩项，q 缺省取当前真机状态缓存
        ddq = np.ones(arm.n_arm)           
        tau = arm.solve_idyn(ddq=ddq)                     # 逆动力学：当前状态（缺省取真机状态缓存）+ 指定 ddq 下的总关节力矩
        print(f"   M 对角元（惯量 kg·m²）= {np.round(np.diag(M), 4).tolist()}")
        print(f"   C（科氏力矩）  = {np.round(C, 4).tolist()} N·m")
        print(f"   G（重力力矩）  = {np.round(G, 4).tolist()} N·m")
        print(f"   solve_idyn(ddq=1)    = {np.round(tau, 4).tolist()} N·m")
        resid = float(np.max(np.abs(tau - (M @ ddq + C + G))))  # 直接逆运动学 与 M·ddq + C + G 的最大偏差
        print(f"   一致性 max|tau − (M·ddq + C + G)| = {resid:.2e}")
        # 显式传 q/dq，支持离线计算：
        # q, dq = arm.home_arm, np.full(arm.n_arm, 0.5)
        # M = arm.get_mass_matrix(q)
        # C = arm.get_coriolis(q, dq)
        # G = arm.get_gravity(q)

        print("=============== ④ 到位判断与随机采样 ============================= ")
        print("—— is_in_q_arm / is_in_q_end / is_in_pose_arm：读硬件实时状态判定 ——")
        print(f"   当前快照 q (rad) = {np.round(q_cur, 3).tolist()}")
        print(f"   is_in_q_arm(home_arm)      = {arm.is_in_q_arm(arm.home_arm)}（机械臂当前是否在 home）")
        print(f"   is_in_q_arm(当前快照 q)    = {arm.is_in_q_arm(q_cur)}（目标即为当前状态 → 必 True）")
        print(f"   is_in_q_end(home_end)      = {arm.is_in_q_end(arm.home_end)}")
        print(f"   is_in_pose_arm(fk(当前 q)) = {arm.is_in_pose_arm(arm.solve_fkine(q_cur))}")
        print("   机制：全部关节均在容差内则为 True；容差可经 cfg 文件中的 tol_q / tol_pos / tol_rot 调整")

        print("—— rand_q_arm / rand_q_end：限位内随机采样——")
        # q = arm.rand_q_arm()                              # joint_limits 限位内随机采样，参数缺省取软限位
        # custom = JointLimits()                            # 自定义限位
        # q = arm.rand_q_arm(custom)                        # 自定义限位内均匀采样
        # q = arm.rand_q_end()                              # end 相同调用用法
        print("   说明：采样均匀且独立、只依赖关节限位；参数缺省取 cfg 配置的机械臂关节软限位。")

        print("============= ⑤ 重力补偿：MIT tau=G(q)、kp=0、kd 轻阻尼，可轻缓拖动 === ")
        yes = input("【确认】arm 即将进入 MIT 重力补偿拖动模式，输入 yes 执行，其他输入跳过：").strip() == "yes"
        if yes:                                 
            ret = arm.set_mode_arm(ControlMode.MIT)       # 切 MIT 模式
            print(f"set_mode_arm(MIT) 返回：{ret}")
            kd = np.array([2, 2, 2, 1.5, 1, 1])           # 逐关节轻阻尼
            print(">>> 重力补偿运行中，按 Ctrl+C 退出……")
            try:
                while True:                               # 死循环
                    arm.send_mit_arm(q=np.zeros(arm.n_arm), dq=np.zeros(arm.n_arm),tau=arm.get_gravity(), kp=np.zeros(arm.n_arm), kd=kd)
                    time.sleep(0.02)                      # 50 Hz 节拍
            except KeyboardInterrupt:                     # Ctrl+C
                pass
        else:
            print("未确认 yes：已跳过重力补偿演示。")

        print("=============== ⑥ 收尾 =========================================== ")
        print(f"disable_arm 返回：{arm.disable_arm()}；disable_end 返回：{arm.disable_end()}")  # 主动失能
    print("结束收尾：机械臂已失能，with 退出自动断开总线。")


if __name__ == "__main__":
    try:
        main()                                             # 正常入口
    except KeyboardInterrupt:                              # Ctrl+C
        print("\n检测到 Ctrl+C：已安全退出。")
