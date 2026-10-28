"""JoyArm 手动冒烟脚本：默认执行 真机连接 + 状态可视化冒烟（MotionFSM 转移序列按需启用）。

【功能概要】
    主动段默认依次执行：① 导入 joyarm_core 并创建 rebot_arm_b601_dm 型机械臂对象
    （离线构造：加载 config / URDF / 求解器装配并自检）；② 连接本体与末端；
    ③ 启动状态可视化（start_state_viz，默认自动弹监控窗；JOYARM_VIZ_NOWINDOW=1
    禁弹窗）。④（按需启用，当前默认注释）启动 MotionFSM，按 event0 选定起始事件
    逐一验证状态转移（set_event → wait_for_motion_state → 打印 当前态/目标态/事件
    三元组；FSM 自动切模式+使能，运动会真实发生）。joyarm.py 公开面 7.1~7.11 的
    其余操作在下方按需摘录并注释；真机操作启用前请确认现场安全。
    主动段整体置于 with 块内：块退出（含末段 Ctrl+C）自动 close() 安全收尾。
    本文件为手动脚本（非 pytest 用例）：主动代码置于 __main__ 守卫内，被 pytest
    收集时仅空导入、不执行任何动作，与同目录套件（test_*.py）互不干扰。

【环境与使用】
    # 首次环境安装（Python >=3.10,<3.12；uv未安装时先安装uv）
    cd <仓库>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 运行本脚本（真机测试：连接本体并弹监控窗；FSM 段启用后驱动真实运动，注意现场安全）
    python test/joyarm_test.py
"""
from __future__ import annotations

import time

import numpy as np

from joyarm_core import (
    ControlMode,
    MotionState,
    SafetyEvent,
    STATE_FROM,
    TrajFrame,
    joyArm,
    listArm,
)

if __name__ == "__main__":

    with joyArm("rebot_arm_b601_dm") as arm:                     # 创建对象（离线构造；任一自检失败抛携带原因的异常）；with 退出自动收尾

        print(f"『rebot_arm_b601_dm』创建成功：arm {arm.n_arm} 关节 / end {arm.n_end} 电机")

        # ---- 硬件后端委托面（真机操作居多，启用前确认安全） ----
        arm.connect_arm()                                # 连接本体（channel/protocol 缺省取 cfg）
        arm.connect_end()                                # 连接末端
        # arm.enable_arm()                                 # 使能（电机上力，注意安全）
        # arm.enable_end()                                 # 等待使能完成
        # print(arm.is_enabled_arm, arm.is_enabled_end)    # 使能状态

        # ---- MotionFSM 真机测试：状态转移路线表验证（FSM 自动切模式+使能，运动注意安全） ----

        # event0=SafetyEvent.DISABLE
        # event0=SafetyEvent.HOLDING
        # event0=SafetyEvent.HOMING
        event0=SafetyEvent.ZEROING
        # event0=SafetyEvent.DAMPING
        # event0=SafetyEvent.NORMAL

        # print("============================= 启动运动状态机 ============================")
        # time.sleep(10)
        # arm.start_motion_fsm()
        # arm.wait_for_motion_state(MotionState.MOTION, timeout=5.0)
        # print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())


        # print("========================== To NORMAL(MOTION) ============================")
        # time.sleep(2)
        # arm.set_event(event0)
        # arm.wait_for_motion_state(STATE_FROM[event0], timeout=5.0)
        # print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())
        # time.sleep(2)
        # arm.set_event(SafetyEvent.NORMAL)
        # arm.wait_for_motion_state(MotionState.MOTION, timeout=5.0)
        # print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())


        # print("========================== To HOMING(HOME) ============================")
        # time.sleep(2)
        # arm.set_event(event0)
        # arm.wait_for_motion_state(STATE_FROM[event0], timeout=5.0)
        # print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())
        # time.sleep(2)
        # arm.set_event(SafetyEvent.HOMING)
        # arm.wait_for_motion_state(MotionState.HOME, timeout=5.0)
        # print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())

        # print("========================== To HOLDING(HOLD) ============================")
        # time.sleep(2)
        # arm.set_event(event0)
        # arm.wait_for_motion_state(STATE_FROM[event0], timeout=5.0)
        # print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())
        # time.sleep(2)
        # arm.set_event(SafetyEvent.HOLDING)
        # arm.wait_for_motion_state(MotionState.HOLD, timeout=5.0)
        # print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())

        # print("========================== To ZEROING(ZERO) ============================")
        # time.sleep(2)
        # arm.set_event(event0)
        # arm.wait_for_motion_state(STATE_FROM[event0], timeout=5.0)
        # print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())
        # time.sleep(2)
        # arm.set_event(SafetyEvent.ZEROING)
        # arm.wait_for_motion_state(MotionState.ZERO, timeout=5.0)
        # print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())

        # print("========================== To DAMPING(DAMP) ============================")
        # time.sleep(2)
        # arm.set_event(event0)
        # arm.wait_for_motion_state(STATE_FROM[event0], timeout=5.0)
        # print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())
        # time.sleep(2)
        # arm.set_event(SafetyEvent.DAMPING)
        # arm.wait_for_motion_state(MotionState.DAMP, timeout=5.0)
        # print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())

        # print("========================== To DISABLE(DISABLE) ============================")
        # time.sleep(2)
        # arm.set_event(event0)
        # arm.wait_for_motion_state(STATE_FROM[event0], timeout=5.0)
        # print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())
        # time.sleep(2)
        # arm.set_event(SafetyEvent.DISABLE)
        # arm.wait_for_motion_state(MotionState.DISABLE, timeout=5.0)
        # print(arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())
        # print("=================================================================")
        # time.sleep(2)

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
        arm.start_state_viz()                          # 启动（traj_points 缺省 200）
        time.sleep(100)
        arm.stop_state_viz()                           # 停止（JOYARM_VIZ_NOWINDOW=1 禁弹窗）