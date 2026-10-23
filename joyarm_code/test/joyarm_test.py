"""JoyArm 手动冒烟脚本：默认仅执行导入 + 对象创建初始化，其余操作全量注释备用。

【功能概要】
    默认只做两步：① 导入 joyarm_core 对象创建入口与共享类型；② 创建 joyarm_dm
    型机械臂对象（离线构造：加载 config / URDF / 求解器装配并自检，不连接硬件、
    不启动任何线程）。joyarm.py 公开面 7.1~7.11 的其余操作在下方全量列出并注释，
    按需取消注释使用；连接 / 使能 / 指令 / 设零等真机操作启用前请确认现场安全。
    本文件为手动脚本（非 pytest 用例）：主动代码置于 __main__ 守卫内，被 pytest
    收集时仅空导入、不执行任何动作，与同目录套件（test_*.py）互不干扰。

【环境与使用】
    # 首次环境安装（Python >=3.10,<3.12；uv未安装时先安装uv）
    cd <仓库>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 运行本脚本（离线构造，不连接硬件）
    python test/joyarm_test.py
"""
from __future__ import annotations

import time

import numpy as np

from joyarm_core import (
    ControlMode,
    MotionState,
    SafetyEvent,
    TrajFrame,
    joyArm,
    listArm,
)

if __name__ == "__main__":
    # ============================================================
    # ① 导入验证 + ② 对象创建（本脚本默认仅执行到这两步）
    # ============================================================
    arm = joyArm("joyarm_dm")                            # 创建对象（离线构造；任一自检失败抛携带原因的异常）
    
    print(f"『joyarm_dm』创建成功：arm {arm.n_arm} 关节 / end {arm.n_end} 电机")

    # ---- 7.3 硬件后端委托面（真机操作居多，启用前确认安全） ----
    # ---- 连接 / 断开 ----
    arm.connect_arm()                                  # 连接本体（channel/protocol 缺省取 cfg）
    arm.connect_end()                                  # 连接末端
    # ---- 使能 / 失能 ----
    # arm.enable_arm()                                   # 使能（电机上力，注意安全）
    # arm.enable_end()
    time.sleep(0.01)                                   # 等待使能完成
    # print(arm.is_enabled_arm, arm.is_enabled_end)      # 使能状态
    # ---- 状态读取 ----
    # state = arm.joint_state_arm                        # 被动快照（零总线帧）
    # state = arm.joint_state_end
    # state = arm.get_state_arm()                        # 读时刷新
    # state = arm.get_state_end()
    # ---- 电机参数与模式 ----
    # arm.set_mode_arm(ControlMode.VELOCITY)             # 切模式（MIT/POSITION/VELOCITY）
    # arm.set_mode_end(ControlMode.VELOCITY)
    # print(arm.get_mode_arm(), arm.get_mode_end())      # 模式缓存
    # ---- 指令下发（直达后端，不经 FSM） ----
    # arm.send_position_arm(arm.home_arm)                  # 关节/电机零位
    # arm.send_position_end(arm.home_end)
    # arm.send_vel_arm(np.zeros(arm.n_arm))              # 速度指令
    # arm.send_vel_end(np.zeros(arm.n_end))
    # arm.send_vel_arm([0.0, -0.0, -0.0, -0.03, 0.0, 0.0])              # 速度指令
    # arm.send_vel_end([0.0])
    # arm.send_mit_arm(arm.home_arm, kd=2.0)             # MIT 指令（q 必填，dq/tau/kp/kd 可选）
    # arm.send_mit_end(arm.home_end, kd=2.0)
    # arm.set_action_end("open")                         # 末端离散动作（open/home/close）
    # arm.set_action_end("position", position=-2.0)      # 指定位置动作

    # ---- 7.4 求解器六域管理：查询 / 切换（六域同构三件套） ----
    # print(arm.get_fkine_solvers())                     # fkine 域已加载注册名
    # arm.set_fkine_solver("pin_fkine_solver")           # 切换激活
    # print(arm.get_activated_fkine_solver())            # 当前激活注册名
    # print(arm.get_ikine_solvers())                     # ikine 域
    # arm.set_ikine_solver("pin_ikine_solver")
    # print(arm.get_activated_ikine_solver())
    # print(arm.get_jac_solvers())                       # jacobian 域
    # arm.set_jac_solver("pin_jacobian_solver")
    # print(arm.get_activated_jac_solver())
    # print(arm.get_dyn_solvers())                       # dynamics 域
    # arm.set_dyn_solver("pin_dyn_solver")
    # print(arm.get_activated_dyn_solver())
    # print(arm.get_traj_solvers())                      # trajectory 域
    # arm.set_traj_solver("single_joint_cubic_traj_planner")
    # print(arm.get_activated_traj_solver())
    # print(arm.get_ctrl_solvers())                      # control 域
    # arm.set_ctrl_solver("joint_position_controller")
    # print(arm.get_activated_ctrl_solver())

    # ---- 7.5 目标帧通道（FSM MOTION 态的运动目标来源） ----
    # t_reach = time.time() + 2.0                        # 目标到达时刻（Unix 秒标量）
    # arm.set_target_joint(t_reach, arm.home_arm)        # 单帧关节目标（dq/ddq/tau 可选）
    # pose_now = arm.solve_fkine(arm.home_arm)           # 现算末端位姿（作笛卡尔目标示例）
    # arm.set_target_cart(t_reach, pose_now)             # 单帧笛卡尔目标（pose 容 Pose 或 (7,)）
    # frame = TrajFrame(time=t_reach, q=arm.home_arm)    # 直接组帧
    # arm.set_target_trajframe(frame)                    # 写入（单帧/帧列表，None 清空）
    # print(arm.get_target_trajframe())                  # 读当前目标（深拷贝）

    # ---- 7.6 数学门面（显式传参离线可用，缺省取硬件状态快照） ----
    # pose = arm.solve_fkine(arm.home_arm)               # 正运动学（frame 可选）
    # ik = arm.solve_ikine(pose)                         # 逆运动学（返回 IKResult）
    # sols = arm.solve_ikine_all(pose)                   # 全部逆解
    # J = arm.get_jac()                                  # 雅可比
    # J_pinv = arm.get_jac_pinv_damped(damping=1e-3)     # 阻尼伪逆
    # tau_ff = arm.solve_idyn(ddq=np.zeros(arm.n_arm))   # 逆动力学（前馈力矩）
    # M = arm.get_mass_matrix()                          # 质量矩阵
    # C = arm.get_coriolis()                             # 科氏项
    # g = arm.get_gravity()                              # 重力项

    # ---- 7.7 到位判断 ----
    # arm.is_in_pose_arm(pose, tol_pos=1e-3, tol_rot=1e-2) # 末端位姿到位
    # arm.is_in_q_arm(arm.home_arm)                      # 关节到位（容差缺省取 cfg arrival_tol）
    # arm.is_in_q_end(arm.home_end)                      # 电机到位

    # ---- 7.8 MotionFSM 启停与安全事件 ----
    # arm.start_motion_fsm()                             # 启动状态机（plan/ctrl 线程随之起搏）
    # print(arm.get_current_motion_state())              # 当前运动状态
    # print(arm.get_target_motion_state())               # 目标运动状态
    # arm.set_event(SafetyEvent.HOMING)                  # 发安全事件（六档，DISABLE 最高）
    # print(arm.get_event())                             # 当前事件
    # ok = arm.wait_for_motion_state(MotionState.HOME)   # 阻塞等待状态（1=到位 / 0=超时）
    # arm.stop_motion_fsm()                              # 停止状态机

    # ---- 7.9 safe_monitor 监视器 ----
    # arm.start_monitor()                                # 启动安全监视（内置硬规则）
    # 自定义监视规则示例：任一关节速度超 2.0 rad/s → 阻尼；返回 None = 无意见
    # def over_speed(state_arm, state_end, motion, event):
    #     if state_arm is not None and np.all(np.isfinite(state_arm.dq)) and np.any(np.abs(state_arm.dq) > 2.0):
    #         return SafetyEvent.DAMPING
    #     return None
    # arm.set_monitor_rules(over_speed)                  # 绑定规则（单条或列表）
    # arm.stop_monitor()                                 # 停止监视

    # ---- 7.10 运行状态可视化（HTTP 轮询 + 自动弹窗） ----
    # arm.start_state_viz()                              # 启动（traj_points 缺省 200）
    # arm.stop_state_viz()                               # 停止（JOYARM_VIZ_NOWINDOW=1 禁弹窗）

    # ---- 7.11 销毁安全退出 ----
    # arm.close()                                        # 停 FSM/监视/可视化、断连并回收资源
    # with joyArm("joyarm_dm") as arm_ctx:               # 上下文管理（__exit__ 自动 close）
    #     arm_ctx.start_motion_fsm()


    time.sleep(3) 