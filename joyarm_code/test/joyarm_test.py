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
    arm.enable_arm()                                   # 使能（电机上力，注意安全）
    arm.enable_end()                               # 等待使能完成
    # print(arm.is_enabled_arm, arm.is_enabled_end)      # 使能状态
    # ---- 电机参数与模式 ----
    arm.set_mode_arm(ControlMode.MIT)             # 切模式（MIT/POSITION/VELOCITY）
    arm.set_mode_end(ControlMode.POSITION)                                    # 等待模式切换完成
    time.sleep(0.1)                       # MIT 指令（关节电机）

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
    # # 自定义监视规则示例：任一关节速度超 2.0 rad/s → 阻尼；返回 None = 无意见
    # def over_speed(state_arm, state_end, motion, event):
    #     if state_arm is not None and np.all(np.isfinite(state_arm.dq)) and np.any(np.abs(state_arm.dq) > 2.0):
    #         return SafetyEvent.DAMPING
    #     return None
    # arm.set_monitor_rules(over_speed)                  # 绑定规则（单条或列表）
    # time.sleep(1000)
    # arm.stop_monitor()                                 # 停止监视

    # ---- 7.10 运行状态可视化（HTTP 轮询 + 自动弹窗） ----
    # arm.start_state_viz()                              # 启动（traj_points 缺省 200）
    # time.sleep(1000)                                   
    # arm.stop_state_viz()                               # 停止（JOYARM_VIZ_NOWINDOW=1 禁弹窗）



    time.sleep(1000)  # 保持进程存活，便于手动测试 / 观察日志 / 连接调试器