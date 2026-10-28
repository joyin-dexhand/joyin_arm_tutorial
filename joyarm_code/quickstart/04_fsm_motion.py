"""
==============================================================================
FSM关节与笛卡尔运动  (快速上手 例程 04)
==============================================================================
【功能概要】
    真机演示运动状态机（MotionFSM）驱动下的关节空间与笛卡尔空间小幅运动：
      ① 启动 FSM，等待自动迁移 DISABLE→HOME→MOTION，查询当前态/目标态；
      ② set_target_joint 下发单帧关节目标（home 基准 ±0.15 rad 小偏移）；
      ③ TrajFrame 目标帧通道写入/读回（set_target_joint 即该通道的单帧便捷封装）；
      ④ solve_fkine / solve_ikine 笛卡尔小位移运动 + set_target_cart 通道演示；
      ⑤ SafetyEvent 安全事件触发 HOLD 原位锁定与 NORMAL 恢复；
      ⑥ stop_motion_fsm 阻塞式回零失能收尾。

【安全须知】
    - 运行前清空机械臂工作半径内的人员与障碍物，全程手持急停手段（急停/断电）待命
    - 三段演示运动均为小幅（home 基准 ±0.2 rad 内，2 s 到达、峰值速度约 0.14 rad/s）；
      但启动 DISABLE→HOME 自动回 home、收尾 HOME→ZERO 回零失能均可移动约 1 rad
      （关节 2/3，速度上限为位置模式 vlim 0.65~0.8 rad/s），异常立即断电
    - 首次运动前脚本会暂停等待确认，回车前请再次确认现场安全

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 2. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（在 joyarm_code/ 目录下，需连接 joyarm_dm 真机）
    python quickstart/04_fsm_motion.py

    预期输出：依次打印 FSM 状态迁移结果、三段运动的到位判断与安全事件切换；
    连接失败或 FSM 未就绪时会提示原因并安全退出。
==============================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装时也可直接运行
from joyarm_core import (  # noqa: E402
    joyArm, MotionState, SafetyEvent, STATE_FROM, Pose, TrajFrame,
)


def wait_q_arrival(arm, q_goal, tol_q: float = 0.05, timeout: float = 3.0) -> bool:
    """轮询关节到位判断（is_in_q_arm），到位返回 True，超时返回 False。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if arm.is_in_q_arm(q_goal, tol_q=tol_q):
            return True
        time.sleep(0.05)
    return False


def main() -> None:
    with joyArm("joyarm_dm") as arm:                       # with 退出自动安全收尾
        # ---- ① 启动：连接 → 确认门 → FSM 启动并等待 MOTION ----
        if arm.connect_arm() is not True:                  # FSM 仅涉 arm 侧，可不连 end
            print("arm 总线连接失败，请检查电源与总线后重试")
            return
        q_home = arm.home_arm                              # home 位形：本例小幅运动的基准
        print(f"已连接 joyarm_dm（{arm.n_arm} 关节），home = {np.round(q_home, 3)} rad")

        # 切换三次多项式规划器：按到达时刻 t 插值（0.15 rad / 2 s，峰值速度约 0.14 rad/s）
        arm.set_traj_solver("single_joint_cubic_traj_planner")

        input(">>> 已确认机械臂工作半径内无人员与障碍物？回车启动 FSM（自动回 home 后小幅运动）：")

        arm.start_motion_fsm()                             # 前置：已连接/反馈新鲜/无故障码/traj+ctrl 已配置
        ok = arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0)
        print(f"① FSM 就绪（自动 DISABLE→HOME→MOTION）：当前态={arm.get_current_motion_state().name} "
              f"目标态={arm.get_target_motion_state().name}（{'OK' if ok else '超时'}）")
        if not ok:
            print("FSM 未进入 MOTION，安全收尾退出")
            arm.stop_motion_fsm()
            return

        # ---- ② 关节空间：set_target_joint 单帧关节目标 ----
        q_goal1 = q_home.copy()
        q_goal1[0] += 0.15                                 # 关节 1/2 各偏 0.15 rad（±0.2 rad 内）
        q_goal1[1] -= 0.15
        arm.set_target_joint(t=time.time() + 2.0, q=q_goal1)   # t 为目标到达时刻（Unix 秒）
        print(f"② 关节运动到位：{wait_q_arrival(arm, q_goal1)}")

        # ---- ③ TrajFrame 通道：手写目标帧写入与读回 ----
        q_goal2 = q_home.copy()
        q_goal2[2] += 0.15                                 # 关节 3 偏 0.15 rad（另一组小偏移）
        frame = TrajFrame(time=time.time() + 2.0, q=q_goal2)   # 手写帧 time 必填，缺省会被规划器拒绝
        arm.set_target_trajframe([frame])                  # set_target_joint 即该通道的单帧便捷封装
        back = arm.get_target_trajframe()                  # 深拷贝读回（单元素列表）
        print(f"③ 读回目标帧 q={np.round(back[0].q, 3)}，到位：{wait_q_arrival(arm, q_goal2)}")

        # ---- ④ 笛卡尔空间：fkine → ikine → 关节通道执行 ----
        pose_cur = arm.solve_fkine()                       # q 缺省取当前关节快照
        pose_goal = Pose(position=pose_cur.position + np.array([0.04, 0.0, 0.02]),
                         orientation=pose_cur.orientation)  # 沿 x/z 平移 4/2 cm，姿态不变
        res = arm.solve_ikine(pose_goal)                   # 数值法自当前 q 迭代求单解
        dev_home = np.abs(res.q - q_home)                  # 逆解相对 home 的逐关节偏差
        if not res.success:
            print(f"④ 逆解失败（err={res.err:.4f}），跳过笛卡尔运动")
        elif dev_home.max() > 0.2:                         # 安全包络校验：超出 home±0.2 rad 的逆解不下发
            print(f"④ 逆解超出 home±0.2 rad 安全包络（最大偏差 {dev_home.max():.3f} rad），跳过笛卡尔运动")
        else:
            arm.set_target_joint(t=time.time() + 2.0, q=res.q)
            arrived = wait_q_arrival(arm, res.q)
            print(f"④ 笛卡尔运动到位：{arrived}，位姿验证 is_in_pose_arm={arm.is_in_pose_arm(pose_goal)}")
            arm.set_target_joint(t=time.time() + 2.0, q=q_home)  # 先回 home（下步通道演示不引发运动）
            wait_q_arrival(arm, q_home)
            arm.set_target_cart(t=time.time() + 2.0, pose=pose_goal)  # 笛卡尔目标帧写入通道（演示）
            cart = arm.get_target_trajframe()              # 读回：帧内为 pose、q 字段为空
            print(f"④ set_target_cart 读回目标位置：{np.round(cart.pose.position, 4)} m")
            arm.set_target_joint(t=time.time() + 2.0, q=q_home)  # 本例驱动运动走 ikine→set_target_joint 路径

        # ---- ⑤ 安全事件：HOLDING 锁定 → NORMAL 恢复 ----
        arm.set_event(SafetyEvent.HOLDING)                 # 触发 FSM 迁移至 HOLD 原位锁定
        arm.wait_for_motion_state(MotionState.HOLD, timeout=5.0)
        print(f"⑤ 事件 HOLDING → {STATE_FROM[SafetyEvent.HOLDING].name}，"   # STATE_FROM：事件→状态映射
              f"当前态={arm.get_current_motion_state().name}")
        arm.set_event(SafetyEvent.NORMAL)                  # 恢复正常（危险度递增 NORMAL<HOLDING<HOMING<ZEROING<DAMPING<DISABLE）
        arm.wait_for_motion_state(MotionState.MOTION, timeout=5.0)
        print(f"⑤ 事件 NORMAL → 当前态={arm.get_current_motion_state().name}")

        # ---- ⑥ 收尾：阻塞式回零失能，with 退出自动断连 ----
        print("⑥ 收尾：回零并失能……")
        arm.stop_motion_fsm()                              # 阻塞等待回零到位后失能


if __name__ == "__main__":
    main()
