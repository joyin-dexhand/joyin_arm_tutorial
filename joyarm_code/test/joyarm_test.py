"""JoyArm 手动冒烟脚本：默认仅执行导入 + 对象创建初始化，其余操作全量注释备用。

【功能概要】
    默认只做两步：① 导入 joyarm_core 对象创建入口与共享类型；② 创建 joyarm_dm
    型机械臂对象（离线构造：加载 config / URDF / 求解器装配并自检，不连接硬件、
    不启动任何线程）。joyarm.py 公开面 7.1~7.11 的其余操作在下方全量列出并注释，
    按需取消注释使用；连接 / 使能 / 指令 / 设零等真机操作启用前请确认现场安全。
    主动段整体置于 with 块内：块退出（含末段 Ctrl+C）自动 close() 安全收尾。
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

    with joyArm("joyarm_dm") as arm:                     # 创建对象（离线构造；任一自检失败抛携带原因的异常）；with 退出自动收尾

        print(f"『joyarm_dm』创建成功：arm {arm.n_arm} 关节 / end {arm.n_end} 电机")

        # ---- 硬件后端委托面（真机操作居多，启用前确认安全） ----
        arm.connect_arm()                                # 连接本体（channel/protocol 缺省取 cfg）
        arm.connect_end()                                # 连接末端
        arm.enable_arm()                                 # 使能（电机上力，注意安全）
        arm.enable_end()                                 # 等待使能完成
        print(arm.is_enabled_arm, arm.is_enabled_end)    # 使能状态

        # ---- 7.8 MotionFSM 真机测试（按 ①→⑥ 逐段放开；FSM 自动切模式+使能，运动注意安全） ----
        # ① 启动状态机：DISABLE → HOME（机械臂运动到 home 位形；HOME 态进入即返 1，不含到位）
        arm.start_motion_fsm()                           # 启动状态机（DISABLE -> HOME -> MOTION）
        print(arm.wait_for_motion_state(MotionState.MOTION, timeout=3.0))   # 约 1.93 秒启动完毕并进入 MOTION（若电脑配置较差，可能需要更长时间）
        print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event()) # 打印当前的状态
        # ② 小幅单关节目标：第 6 关节 +2°（从当前实测位置出发，三次规划 3 s 到达）
        q_goal = arm.joint_state_arm.q.copy()            # 当前快照
        q_goal[5] += np.deg2rad(20.0)                    # 第 6 关节（1 基）+20°，须在软限位内
        arm.set_target_joint(time.time() + 3.0, q_goal)  # 目标到达时刻 = now + 3 s
        time.sleep(3.1)                                  # 等运动走完
        print(arm.is_in_q_arm(q_goal, 0.01), "\n", arm.joint_state_arm.q)   # True=1 运动到位
        # ③ 保护事件演练：原位锁定 → 恢复
        # arm.set_event(SafetyEvent.HOLDING)             # HOLD：锁存当前 q 逐拍重发
        # print(arm.wait_for_motion_state(MotionState.HOLD, timeout=5.0))
        # arm.set_event(SafetyEvent.NORMAL)              # 恢复正常运动（须显式发 NORMAL）
        # print(arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0))
        # ④ 阻尼演练：MIT 纯阻尼 kp=0，臂顺重力缓落——确认支撑后再放开！
        # arm.set_event(SafetyEvent.DAMPING)             # 发安全事件（六档，DISABLE 最高）
        # print(arm.wait_for_motion_state(MotionState.DAMPING, timeout=5.0))
        # time.sleep(2.0)                                # 观察阻尼滞留
        # arm.set_event(SafetyEvent.NORMAL)              # 恢复（DAMPING 经 HOME 回 MOTION）
        # print(arm.wait_for_motion_state(MotionState.MOTION, timeout=30.0))
        # ⑤ 停机：经 HOME 回全零位并失能（显式回零路径）
        # arm.stop_motion_fsm()                          # 停止状态机（阻塞，含回零等待）
        # print(arm.get_current_motion_state(), arm.is_enabled_arm)   # 期望 DISABLE / False
        # ---- 急停（随时放开：不回零直接失能；恢复须先 set_event(NORMAL)） ----
        # arm.set_event(SafetyEvent.DISABLE)
        # ---- 收尾 ----（with 退出时再调 close() 为幂等跳过）
        arm.close()                                      # 停 FSM/监视/可视化、断连并回收

        # ----safe_monitor 监视器 ----
        # arm.start_monitor()                            # 启动安全监视（内置硬规则）
        # # 自定义监视规则示例：任一关节速度超 2.0 rad/s → 阻尼；返回 None = 无意见
        # def over_speed(state_arm, state_end, motion, event):
        #     if state_arm is not None and np.all(np.isfinite(state_arm.dq)) and np.any(np.abs(state_arm.dq) > 2.0):
        #         return SafetyEvent.DAMPING
        #     return None
        # arm.set_monitor_rules(over_speed)              # 绑定规则（单条或列表）
        # time.sleep(1000)
        # arm.stop_monitor()                             # 停止监视

        # ---- 运行状态可视化（HTTP 轮询 + 自动弹窗） ----
        # arm.start_state_viz()                          # 启动（traj_points 缺省 200）
        # time.sleep(1000)
        # arm.stop_state_viz()                           # 停止（JOYARM_VIZ_NOWINDOW=1 禁弹窗）



        try:
            time.sleep(1000)                             # 保持进程存活，便于手动测试 / 观察日志 / 连接调试器
        except KeyboardInterrupt:
            pass                                         # 吞掉中断：以正常退出路径走 with 收尾（默认回零停机）
