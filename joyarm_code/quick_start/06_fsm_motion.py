"""
==============================================================================
例程 06 —— 基于运动状态机（MotionFSM）的运动（需真机）
==============================================================================
【功能概要】
    1. 连接 arm 总线，打印 traj / ctrl 求解器并选定规划器与控制器；
    2. start_motion_fsm 启动状态机，wait_for_motion_state 等待状态迁移完成；
    3. set_target_joint 下发小幅目标（set_target_trajframe 通道），get_target_trajframe 读回；
    4. solve_fkine / solve_ikine 构造小位移笛卡尔目标，set_target_cart 写入并读回；
    5. set_event() 触发安全事件和期望控制状态，get_event() 读回当前事件；
    6. set_ctrl_solver 切换到拖动示教控制器（MIT 重力补偿 + 阻尼/刚度、平衡点缓释跟随），MOTION 态徒手拖动、松手即停；
    7. stop_motion_fsm 收尾：危险性不高于 ZEROING 则先回零再失能，更危险的状况（阻尼和失能）则直接失能。

【原理简介】
    运动状态机六状态：DISABLE / ZERO / HOME / MOTION / HOLD / DAMP。HOME 为中转站——ZERO 与 MOTION 之间切换须先经过 HOME；
    安全事件 SafetyEvent 危险性从轻到重：NORMAL < HOLDING < HOMING < ZEROING < DAMPING < DISABLE，FSM 按 STATE_FROM 映射表迁往安全事件对应的状态；
    进入 MOTION 后另起 plan 线程（消费目标帧做规划）与 ctrl 线程（采样轨迹并下发指令），目标帧由 set_target_* 写入。三者互不阻塞；
    拖动示教：set_ctrl_solver 切换控制器后，FSM 自动把电机模式设置为控制器 MODE，控制律 tau = G(q) + kp·(q_eq − q) + kd·(0 − dq)。

【安全须知】
    - 运行前清空机械臂工作半径内的人员与障碍物，全程准备好急停或断电手段
    - Ctrl+C 随时中断，with 退出自动安全收尾
    - 拖动示教环节徒手轻缓拖动机械臂，禁止猛烈拉扯或强行超出关节限位

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（需连接对应的机械臂）
    python quick_start/06_fsm_motion.py <型号>  # 如：rebot_arm_b601_dm

【预期输出】
    启动条件与求解器确认 → 关节/笛卡尔两段小幅运动与目标帧读回 → 安全事件切换 → 拖动示教（热切换控制器，MIT）→ 回零收尾；异常时提示原因并安全退出。
==============================================================================
"""

from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from joyarm_core import joyArm, listArm, MotionState, SafetyEvent, STATE_FROM, Pose, TrajFrame


# ============================================================
# 一、命令行解析：必填型号参数，缺参打印用法、列出可用型号并以退出码 1 结束
# ============================================================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并列出可用型号（listArm），以退出码 1 结束。"""
    if len(sys.argv) != 2:                                # 必须恰好携带 1 个型号参数
        print(f"用法：python quick_start/06_fsm_motion.py <型号>")  # 缺参：先打印用法
        listArm()                                         # 顺带列出全部可用型号
        sys.exit(1)                                       # 退出码 1，不进入演示
    return sys.argv[1].strip()                            # 型号名


# ============================================================
# 二、打印格式化助手：数组定宽显示
# ============================================================
def fmt_arr(arr) -> str:
    """一维数值数组转逐元素带符号 3 位小数串；None 显示「—」。"""
    return "—" if arr is None else " ".join(f"{x:+.3f}" for x in np.asarray(arr, dtype=float))


# ============================================================
# 三、真机演示全流程：连接与求解器 → FSM 启动 → 关节/笛卡尔目标 → 安全事件 → 回零收尾
# ============================================================
def main() -> None:
    """真机演示全流程：连接与求解器 → FSM 启动 → 关节/笛卡尔目标 → 安全事件 → 回零收尾。"""
    model = parse_model()                                 # 必填型号参数

    with joyArm(model) as arm:                            # with 退出自动安全收尾（停 FSM→失能→断连）

        print("=============== ① 连接与求解器确认 ==================================== ")
        if arm.connect_arm() is not True:                 # 连接 arm 总线
            print("arm 连接失败：请检查电源与总线后重试")
            return
        q_home = np.array(arm.home_arm, dtype=float)      # home 位形
        print(f"已连接 {model}（{arm.n_arm} 关节），home = {fmt_arr(q_home)} rad")

        traj_names = arm.get_traj_solvers()               # 两域求解器注册名列表
        ctrl_names = arm.get_ctrl_solvers()
        print(f"traj 求解器：{traj_names}（默认激活 {arm.get_activated_traj_solver()}）")
        print(f"ctrl 求解器：{ctrl_names}（默认激活 {arm.get_activated_ctrl_solver()}）")
        ret = arm.set_traj_solver("single_joint_direct_traj_planner")   # 直传规划器
        print(f"set_traj_solver('single_joint_direct_traj_planner') 返回：{ret}")
        ret = arm.set_ctrl_solver("joint_position_direct_controller")    # 直传控制器
        print(f"set_ctrl_solver('joint_position_direct_controller') 返回：{ret}")

        print("=============== ② 启动运动状态机 start_motion_fsm ==================== ")
        state = arm.joint_state_arm                       # 后台状态快照
        lag_ms = (time.time() - state.t) * 1000.0 if state.t > 0 else float("inf")  # 快照陈旧度（ms）
        # 启动条件 4 项（start_motion_fsm 内部同样校验，不满足拒绝启动）
        print(f"连接 {arm.is_connected_arm is True}；反馈新鲜 {lag_ms <= 50.0}（now − t = {lag_ms:.1f} ms ≤ 50 ms）；"
              f"无故障码 {bool(np.all(np.isin(state.error, (0, 1))))}；求解器已配置 {bool(traj_names) and bool(ctrl_names)}")

        yes = input("【确认】启动 FSM 将自动回 home 并开始小幅运动，输入 yes 执行，其他输入退出：").strip() == "yes"
        if not yes:
            print("未确认 yes：不启动 FSM，安全退出。")
            return
        try:
            arm.start_motion_fsm()                        # 非阻塞启动（条件不满足抛 RuntimeError）
        except RuntimeError as err:
            print(f"start_motion_fsm 拒绝启动：{err}")
            return
        ok = arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0)  # 等待自动迁移 DISABLE→HOME→MOTION
        print(f"wait_for_motion_state(MOTION) = {ok}（{'已到达' if ok else '超时'}），"
              f"当前态 = {arm.get_current_motion_state().name}，目标态 = {arm.get_target_motion_state().name}")
        if not ok:
            print("FSM 未进入 MOTION：执行 stop_motion_fsm 收尾退出")
            arm.stop_motion_fsm()                         # 阻塞回零失能
            return

        print("=============== ③ 关节目标运动 set_target_joint ===================== ")
        q_goal = q_home.copy()
        q_goal[1] -= 0.3
        q_goal[2] -= 0.3
        yes = input("【确认】arm 关节 1/2/3 将小幅运动，输入 yes 执行，其他输入跳过：").strip() == "yes"
        if yes:
            arm.set_target_joint(t=time.time(), q=q_goal)  # 单帧目标一次性完整写入
            print(f"set_target_joint 写入后读回 TrajFrame：{arm.get_target_trajframe()}")
            time.sleep(2.0)  # 等待规划器消费目标帧并下发控制器
            print(f"到位判断 is_in_q_arm：{'是' if arm.is_in_q_arm(q_goal) else '否'}")
            frame = TrajFrame(time=time.time(), q=q_goal) 
            arm.set_target_trajframe(frame)                # 底层通道，set_target_joint/cart 的内部实现方式，deepcopy 后原子写入
            print(f"set_target_trajframe 写入后读回 TrajFrame：{arm.get_target_trajframe()}")
            time.sleep(2.0)  # 等待规划器消费目标帧并下发控制器
        else:
            print("未确认 yes：已跳过关节目标运动。")

        print("=============== ④ 笛卡尔目标运动 ===================================== ")
        pose_cur = arm.solve_fkine()                      # 正运动学（q 缺省取当前快照）→ Pose
        pose_goal = Pose(position=pose_cur.position + np.array([-0.04, 0.0, -0.05]),
                         orientation=pose_cur.orientation)  # 姿态保持不变
        res = arm.solve_ikine(pose_goal)                  # 数值法逆解（自当前 q 迭代）
        print(f"逆解：success = {res.success}，err = {res.err:.2e}，n_iter = {res.n_iter}")
        dev = float("inf") if res.q is None else float(np.abs(res.q - q_home).max())  # 逆解相对 home 最大偏差
        if not res.success or dev > 0.6:                  # 未收敛或超安全包络：不下发
            print("逆解未收敛或超出安全包络：跳过笛卡尔运动")
        else:
            print(f"逆解 q = {fmt_arr(res.q)} rad（相对 home 偏差 {dev:.3f} rad）")
            yes = input("【确认】机械臂将运动到逆解位形并途经 home，输入 yes 执行，其他输入跳过：").strip() == "yes"
            if yes:
                arm.set_target_joint(t=time.time()+2.0, q=res.q)        # 设置关节类目标，set_target_trajframe 的便利封装
                print(f"set_target_joint 写入后读回 TrajFrame：{arm.get_target_trajframe()}")
                time.sleep(2.0)  # 等待规划器消费目标帧并下发控制器
                print(f"位姿验证 is_in_pose_arm = {'是' if arm.is_in_pose_arm(pose_goal) else '否'}")
                arm.set_target_cart(t=time.time()+2.0, pose=pose_goal)  # 设置笛卡尔目标，set_target_trajframe 的便利封装
                time.sleep(2.0)  # 等待规划器消费目标帧并下发控制器
                print(f"set_target_cart 写入后读回 TrajFrame：{arm.get_target_trajframe()}")
            else:
                print("未确认 yes：已跳过笛卡尔运动。")

        print("=============== ⑤ 安全事件 set_event / get_event ================== ")
        print(f"事件严重程度从轻到重：NORMAL < HOLDING < HOMING < ZEROING < DAMPING < DISABLE；")
        print(f"当前事件 = {arm.get_event().name}，目标态 = {STATE_FROM[arm.get_event()].name}")
        input("【确认】将依次触发 HOLDING、NORMAL，观察 FSM 状态迁移，输入任意键执行：")
        arm.set_target_joint(t=time.time(), q=q_home)  # 设定目标帧，使 motion 状态维持 home 位形不动
        for event, target in ((SafetyEvent.HOLDING, MotionState.HOLD), (SafetyEvent.NORMAL, MotionState.MOTION)):
            arm.set_event(event)                          # HOLDING 锁定原位；NORMAL 恢复
            ok = arm.wait_for_motion_state(target, timeout=5.0)
            print(f"set_event({event.name}) → {STATE_FROM[event].name}："
                  f"等待 = {ok}（{'已到达' if ok else '超时'}），"
                  f"当前态 = {arm.get_current_motion_state().name}，事件 = {arm.get_event().name}")
            time.sleep(2.0)

        print("=============== ⑥ 拖动示教：set_ctrl_solver 热切换控制器 ================= ")
        print("拖动示教控制器：MIT 模式，tau = G(q) 重力补偿 + kp 刚度维持平衡点 + kd 阻尼，平衡点按阈值/步进率缓释跟随（拖得动、松手停）")
        yes = input("【确认】将热切换到拖动示教控制器，电机模式自动切 MIT，可徒手拖动机械臂，输入 yes 执行，其他输入跳过：").strip() == "yes"
        if yes:
            ret = arm.set_ctrl_solver("joint_drag_teach_controller")  # 运行中热切换：等在途 step 后换指针，模式自愈交 FSM 维持拍
            print(f"set_ctrl_solver('joint_drag_teach_controller') 返回：{ret}；当前激活：{arm.get_activated_ctrl_solver()}")
            if ret:
                time.sleep(0.1)                     # 等 FSM 维持拍（50 Hz）把电机模式自愈为 MIT
                print(f"电机模式 get_mode_arm：{arm.get_mode_arm()}；当前态 = {arm.get_current_motion_state().name}")
                print(">>> 拖动运行中：徒手轻缓拖动机械臂，松手即在当前位置停稳；按 Ctrl+C 结束拖动……")
                try:
                    while True:
                        time.sleep(0.2)             # 控制环在 FSM ctrl 线程（200 Hz）自主运行，主线程仅等待
                except KeyboardInterrupt:            # 与 05 ⑤ 同款：本地吞 Ctrl+C，继续 ⑦ 收尾
                    pass
                print(f"拖动后当前关节角：{fmt_arr(np.asarray(arm.joint_state_arm.q, dtype=float))} rad")
            else:
                print("热切换失败：cfg robotics.control 未配置 joint_drag_teach_controller，跳过拖动演示。")
        else:
            print("未确认 yes：已跳过拖动示教演示。")

        print("=============== ⑦ 状态机收尾 stop_motion_fsm ======================= ")
        print("规则：当前事件不重于 ZEROING → 先回零位（经 HOME）再失能；更严重（DAMPING/DISABLE）→ 直接失能停机")
        arm.stop_motion_fsm()
        print(f"FSM 已停止：当前态 = {arm.get_current_motion_state().name}，事件 = {arm.get_event().name}（收尾统一复位为 DISABLE）")
        print("=============== 完成：机械臂已回零失能，with 退出将自动断开连接。 =================== ")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:                             # Ctrl+C
        print("\n检测到 Ctrl+C：已安全退出。")
