"""
==============================================================================
监视器与可视化  (快速上手 例程 05)
==============================================================================
【功能概要】
    真机演示 JoyArm 的两大运行期观测能力与标准收尾顺序：
      1. 安全监视器 safe_monitor：挂接一条自定义超速规则
         （任一关节 |dq| > 1.5 rad/s → 阻尼防护 DAMPING，否则无意见返回 None）；
         监视线程按 cfg 的 monitor_hz 节拍轮询，多规则命中取最危险仲裁；
      2. 运行状态可视化：三栏监控面板（静态参考 / meshcat 3D / 动态状态面板），
         默认自动弹 PySide6 窗口，浏览器地址见控制台 meshcat 日志；
      3. 启动运动状态机做两组小偏移往复运动（home ±0.2 rad 内、dq 目标为零即到位停），
         期间用 arm.joint_state_arm 被动快照每 0.5 s 打印关节位置摘要；
      4. 收尾：停状态机 → 停监视器 → 停可视化（各自可安全重复调用），with 退出兜底。

【安全须知】
    - 运行前清空机械臂工作半径内的人员与障碍物
    - 全程值守并备好急停（急停按钮 / 断电）；本例单次偏移仅 0.15 rad 的小幅运动，
      首次运动前会暂停等待确认
    - 若监视器触发阻尼防护（DAMPING），机械臂会自动进入软防护态，排查后再重新上电运行

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 2. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（在 joyarm_code/ 目录下）
    python quickstart/05_monitor_viz.py

    预期输出：控制台依次打印创建/连接与监控页地址（info 日志），自动弹出监控窗口；
    运动期间每 0.5 s 打印一行关节位置快照；结束自动回零失能并安全退出。
    无显示器或不想弹窗时，可设环境变量 JOYARM_VIZ_NOWINDOW=1（仅控制台 + 浏览器方式）。
==============================================================================
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装时也可直接运行
from joyarm_core import MotionState, SafetyEvent, joyArm  # noqa: E402


def overspeed_rule(state_arm, state_end, motion, event):
    """自定义监视规则：任一关节 |dq| > 1.5 rad/s → 阻尼防护；否则 None（无意见）。

    规则签名固定为 (state_arm, state_end, motion, event)——前两参为 arm/end 关节
    状态快照、motion 为当前运动状态、event 为当前安全事件，均只读；返回 None 表示
    本规则不触发意见，与内置硬规则合并后取危险程度最高者。
    """
    if (state_arm is not None
            and np.all(np.isfinite(state_arm.dq))
            and np.any(np.abs(state_arm.dq) > 1.5)):
        return SafetyEvent.DAMPING          # 超速 → 阻尼防护（FSM 自动转 DAMP 态）
    return None


def move_and_report(arm, q_goal, tag, duration=2.0):
    """下发一组关节目标，行进期间每 0.5 s 打印 arm 侧被动快照 q 摘要。"""
    arm.set_target_joint(time.time() + duration, q_goal, dq=np.zeros_like(q_goal))
    # ↑ t 为目标到达时刻（Unix 秒，= 当前时刻 + 时长）、dq 目标为零即到位停；
    # 本型号默认激活直传规划器（目标位置直达、t/dq 仅按 API 语义填写），小幅目标即低速
    t_end = time.time() + duration + 0.5    # 观察窗口：多看 0.5 s 确认到位
    while time.time() < t_end:
        # joint_state_arm 为被动只读快照（不触发读帧），由后台刷新线程持续维护
        print(f"[{tag}] q(rad) = {np.round(arm.joint_state_arm.q, 3)}")
        time.sleep(0.5)


def main() -> None:
    # 显示 joyarm_core 的 info 日志（含监控页与 meshcat 浏览器地址）
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with joyArm("joyarm_dm") as arm:        # 创建整机对象；with 退出自动安全收尾
        print(f"『joyarm_dm』创建成功：本体 {arm.n_arm} 关节")

        arm.connect_arm()                   # 连接本体总线（channel/protocol 缺省取 cfg）

        # ---- ① 安全监视器：先绑规则、再启动 ----
        arm.set_monitor_rules([overspeed_rule])   # 挂接自定义规则列表（内置硬规则不受影响）
        # 启动监视线程：按 cfg 的 monitor_hz 节拍轮询，硬规则与自定义规则命中合并、
        # 多规则命中取最危险仲裁；本例小幅运动不会触发，仅演示挂接方式
        arm.start_monitor()

        # ---- ② 运行状态可视化 ----
        # 三栏监控面板 = 静态参考 / meshcat 3D / 状态面板；默认自动弹 PySide6 窗口
        # （环境变量 JOYARM_VIZ_NOWINDOW=1 可禁用弹窗）；浏览器地址看控制台 meshcat
        # 日志；traj_points=200 为轨迹显示采样点数（0 = 关闭，需 FSM 启动且处于 MOTION 态）
        arm.start_state_viz()

        # ---- ③ 小幅往复运动（首次运动前的人工确认门） ----
        input("\n>>> 确认机械臂工作半径内无人员与障碍物；回车后开始运动（先自动回 home）……")

        arm.start_motion_fsm()              # 启动运动状态机（自动 DISABLE → HOME → MOTION）
        if arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0) != 1:  # 1=到达 / 0=超时
            print("10 秒内未进入 MOTION 态，跳过运动段，直接收尾")
        else:
            home = arm.home_arm.copy()      # 以 home 位形为基准（property 只读，复制后修改）
            q_off = home.copy()
            q_off[0] += 0.15                # 关节 1 小偏移 +0.15 rad（home ±0.2 rad 内）
            move_and_report(arm, q_off, "偏移")   # 第 1 组：离开 home 一个小偏移
            move_and_report(arm, home, "返回")    # 第 2 组：返回 home（往复完成）

        # ---- ④ 收尾（顺序：停运动 → 停监视 → 停可视化；各自可安全重复调用） ----
        arm.stop_motion_fsm()               # 阻塞安全停机：经 HOME 回零失能（含等待）
        arm.stop_monitor()                  # 停安全监视器线程
        arm.stop_state_viz()                # 停可视化（关 HTTP 服务、监控窗口与 meshcat）
        print("例程完成，with 退出将再执行一次兜底安全收尾（disable + disconnect）。")


if __name__ == "__main__":
    main()
