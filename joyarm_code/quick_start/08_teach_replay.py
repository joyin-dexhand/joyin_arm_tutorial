"""
==============================================================================
例程 08 —— 拖动示教与轨迹复现（需真机）
==============================================================================
【功能概要】
    真机例程：完整走一遍「拖动示教 → 录制 → 回放」的轨迹复现流程：
    1. 连接真机并启动运动状态机（FSM），等待自动迁移至 MOTION 正常运动态；
    2. 发送 DAMPING 安全事件切入 DAMP 阻尼态，用手缓慢拖动机械臂示教轨迹；
    3. 以 50 Hz 采样录制 4 秒关节角序列（约 200 帧）；
    4. 切回 MOTION 态，录制序列每 5 帧取 1 帧作路点，以 10 Hz 逐点下发复现轨迹；
    5. 阻塞安全停机（回零失能），with 退出自动断连；
    6. 覆盖接口：connect_arm / start_motion_fsm / set_event / wait_for_motion_state / joint_state_arm /
       home_arm / set_target_joint / stop_motion_fsm。

【原理简介】
    - 拖动示教的核心是 DAMP 阻尼态：FSM 每拍向电机下发 MIT 帧（kp=0、kd=阻尼系数 damping_kd、
      tau=重力补偿率 damping_k_gravity × 重力矩），电机像「带阻尼的铰链」般顺从外力，可徒手缓慢拖动；
    - 录制端只读 joint_state_arm 被动快照（后台线程持续维护、不额外占用总线），按绝对时刻对齐节拍，
      采样率稳定在 50 Hz；
    - 回放端逐路点调用 set_target_joint 写入目标帧：默认激活的 single_joint_hold_traj_planner
      仅取帧内 q 直传控制器（忽略到达时刻 t），10 Hz 连续刷新目标即形成轨迹流；若用 set_traj_solver
      切到三次多项式规划器 single_joint_cubic_traj_planner，则 t 生效、按到达时刻平滑插值跟帧。

【安全须知】
    - 清空机械臂工作半径内的人员与障碍物，机械臂稳固放置后再上电；
    - 全程值守，急停按钮或电源开关随手可及，异常立即断电；
    - 拖动与回放均保持缓慢小幅：建议各关节相对 home 位形偏移不超过 0.2 rad；
    - 阻尼态下勿快速甩动关节，也勿松手任其自由摆动。

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（需已连接真机并上电）
    python quick_start/08_teach_replay.py <型号>

【预期输出】
    依次打印参数表、FSM 状态迁移结果、录制进度与帧数、回放路点进度与终点关节角；
    结束后机械臂回零失能，提示已安全退出。
==============================================================================
"""

from pathlib import Path             # Path：定位本脚本上级目录（仓库代码根）
import sys                           # sys：把代码根加入模块搜索路径
import time                          # time：节拍对齐、目标到达时刻与超时判断

import numpy as np                   # numpy：关节角序列的数组存储与抽样

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装 joyarm_core 时也可直接运行本脚本
# 机械臂主类 joyArm + 运动状态 / 安全事件两个枚举（须在 sys.path 注入之后导入）
from joyarm_core import MotionState, SafetyEvent, joyArm  # noqa: E402


# ============================================================
# 一、运行参数：录制与回放的频率 / 时长常量
# ============================================================
RECORD_HZ = 50        # 录制采样频率（Hz）：每秒记录 50 帧关节角
RECORD_SEC = 4.0      # 录制时长（秒）：总帧数 ≈ 50 × 4 = 200 帧
REPLAY_HZ = 10        # 回放路点下发频率（Hz）：每 0.1 s 下发一路点
REPLAY_AHEAD = 0.15   # 路点目标到达时刻提前量（秒；仅 cubic 规划器生效，见 replay_teach 注释）


# ============================================================
# 二、命令行参数与打印助手：解析必填型号参数，向量转可读文本
# ============================================================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并以退出码 1 结束。"""
    if len(sys.argv) != 2:                              # 必须恰好携带 1 个型号参数
        sys.exit(f"用法：python quick_start/08_teach_replay.py <型号>")  # 缺参退出，不进入演示
    return sys.argv[1].strip()                          # 型号名


def vec_str(v) -> str:
    """向量转字符串：None 显「—」，其余保留 3 位小数后转 list。"""
    return "—" if v is None else np.round(v, 3).tolist()  # None（未知/无效）与数值统一为可读文本


# ============================================================
# 三、示教录制：DAMP 阻尼态下按固定频率只读快照关节角
# ============================================================
def record_teach(arm, hz: float, sec: float) -> np.ndarray:
    """以 hz 频率录制 sec 秒关节角序列（DAMP 阻尼态下调用）。

    只读被动快照 joint_state_arm（不触发总线读帧），按绝对时刻对齐节拍。

    :param arm: 已进入 DAMP 阻尼态的 joyArm 对象。
    :param hz: 采样频率（Hz）。
    :param sec: 录制时长（秒）。
    :return: 形状约 (hz × sec, n_arm) 的关节角数组（rad）。
    """
    frames, reported = [], 0        # 帧列表 / 已打印的整秒进度（每进入新一整秒报一次）
    t0 = time.perf_counter()        # 录制起点（perf_counter 单调递增，适合测时长）
    tick = t0                       # 下一拍的理论绝对时刻
    while time.perf_counter() - t0 < sec:             # 未满 sec 秒持续采样
        frames.append(list(arm.joint_state_arm.q))    # 追加一帧关节位置快照 (n_arm,)
        sec_now = int(time.perf_counter() - t0)       # 已录制完成的整秒数
        if sec_now > reported:                        # 每进入新的一整秒打印一次进度
            reported = sec_now                        # 更新已打印的整秒数
            print(f"    录制中 {sec_now}/{sec:.0f} 秒")
        tick += 1.0 / hz                              # 下一拍绝对时刻推进一个采样周期
        time.sleep(max(0.0, tick - time.perf_counter()))  # 睡到下一拍（已滞后则不睡，追赶节拍）
    return np.array(frames)                           # 汇总为 (帧数, n_arm) 数组


# ============================================================
# 四、轨迹回放：录制序列抽样为路点，按 10 Hz 逐点下发
# ============================================================
def replay_teach(arm, record: np.ndarray, ahead: float) -> None:
    """按 10 Hz 逐路点下发录制序列（每 5 帧取 1 帧），复现示教轨迹。

    :param arm: 已回到 MOTION 态的 joyArm 对象。
    :param record: 录制的关节角数组 (帧数, n_arm)。
    :param ahead: 路点目标到达时刻提前量（秒；仅 cubic 规划器生效）。
    """
    waypoints = record[::RECORD_HZ // REPLAY_HZ]      # 每 5 帧取 1 帧作路点（50 Hz → 10 Hz 抽样）
    total = len(waypoints)                            # 路点总数（4 秒录制约 40 个）
    print(f"    开始回放：共 {total} 个路点，{REPLAY_HZ} Hz 下发……")
    for i, q_wp in enumerate(waypoints, start=1):     # 逐路点下发（i 从 1 计数便于显示进度）
        # 默认规划器 single_joint_hold_traj_planner（配置首位）只取 q 直传、忽略 t：
        # 回放即 10 Hz 位置直传流，速度由电机 vlim 兜底；t（= 当前时刻 + ahead 的 Unix
        # 绝对时刻）仅在 set_traj_solver 切到 single_joint_cubic_traj_planner 后生效
        # （按 T = t - now 构造三次多项式、按到达时刻平滑跟帧）
        arm.set_target_joint(t=time.time() + ahead, q=q_wp)  # 写入单帧关节目标（原子通道）
        if i % 10 == 0 or i == total:                 # 每 10 个路点（及最后一个）打印进度
            print(f"    回放进度 {i}/{total} 路点")
        time.sleep(1.0 / REPLAY_HZ)                   # 按 10 Hz 节拍下发下一路点
    time.sleep(ahead)                                 # 末路点后的收尾余量（cubic 规划器下恰为到达时刻）


# ============================================================
# 五、主流程：连接启动 → 阻尼示教与录制 → 轨迹回放 → 安全收尾
# ============================================================
def main() -> None:
    """拖动示教与轨迹复现主流程（真机，三处回车确认门）。"""
    print("=" * 78)                                   # 大节横幅：开
    print("例程 08 —— 拖动示教与轨迹复现（需真机）")     # 例程标题
    print("=" * 78)                                   # 大节横幅：闭

    # ---- ⓪ 参数表：先看清本例程的全部可调常量 ----
    rows = [                                          # (参数名, 取值, 说明) 三列数据
        ("RECORD_HZ", f"{RECORD_HZ} Hz", "录制采样频率：每秒记录的关节角帧数"),
        ("RECORD_SEC", f"{RECORD_SEC} s", "录制时长：总帧数 ≈ 频率 × 时长 = 200 帧"),
        ("REPLAY_HZ", f"{REPLAY_HZ} Hz", "回放路点下发频率：每 0.1 s 下发一路点"),
        ("REPLAY_AHEAD", f"{REPLAY_AHEAD} s", "路点到达时刻提前量（仅 cubic 规划器生效）"),
    ]
    print(f"{'参数':<16}{'取值':>10}   说明")          # 表头（列宽与下方数据行对齐）
    print("-" * 78)                                   # 表头与数据行的分隔线
    for name, val, desc in rows:                      # 逐行打印参数表
        print(f"{name:<16}{val:>10}   {desc}")

    model = parse_model()                             # 必填型号参数
    with joyArm(model) as arm:                        # with 退出（含异常）自动安全收尾 close()

        # ---- ① 连接并启动 FSM：自动 DISABLE → HOME → MOTION ----
        print("=" * 78)                               # 大节横幅：开
        print("① 连接真机并启动运动状态机")               # 大节标题
        print("=" * 78)                               # 大节横幅：闭
        ret = arm.connect_arm()                       # 连接本体 arm 电机总线（成功后自动设 cfg 默认模式）
        print(f"    connect_arm 返回：{'—' if ret is None else ret}；"
              f"is_connected_arm：{arm.is_connected_arm}")   # None（意外）显「—」
        if ret is not True:                           # 未连接成功（False 失败 / None 意外）则放弃后续动作
            print("    arm 总线连接失败，请检查电源与总线后重试。")
            return                                    # 提前返回：with 退出自动安全收尾
        print(f"    型号 {model}：{arm.n_arm} 个关节，"
              f"home = {vec_str(arm.home_arm)} rad")  # home 位形：安全包络的基准

        input(">>> 确认机械臂工作半径内无人员与障碍物？"
              "回车启动 FSM（自动回 home 后进入可拖动示教的流程）：")
        arm.start_motion_fsm()                        # 启动 FSM：复位缓存后自动 DISABLE → HOME → MOTION
        if arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0) != 1:  # 阻塞等待进入 MOTION
            print("    10 秒内未进入 MOTION 正常运动态，安全收尾退出。")
            return                                    # 提前返回：with 退出自动安全收尾
        print(f"    已进入 MOTION 正常运动态"
              f"（当前态={arm.get_current_motion_state().name}），"   # 当前 FSM 状态名
              f"home = {vec_str(arm.home_arm)} rad")  # home 位形（运动基准）

        # ---- ② DAMP 阻尼态拖动示教 + 50 Hz 录制 4 秒 ----
        print("=" * 78)                               # 大节横幅：开
        print("② 阻尼态拖动示教与录制")                 # 大节标题
        print("=" * 78)                               # 大节横幅：闭
        arm.set_event(SafetyEvent.DAMPING)            # 阻尼防护事件 → FSM 目标态 DAMP（STATE_FROM 映射）
        if arm.wait_for_motion_state(MotionState.DAMP, timeout=5.0) != 1:    # 阻塞等待实际进入 DAMP
            print("    未进入 DAMP 阻尼态，安全收尾退出。")
            return                                    # 提前返回：with 退出自动安全收尾
        print("    已进入 DAMP 阻尼态：现可用手握臂杆缓慢拖动"
              "（重力补偿程度由 cfg 的 damping_k_gravity 决定）。")   # DAMP 态操作提示
        input(">>> 缓慢拖动机械臂示范一段轨迹，就位后回车开始录制 4 秒……")     # 录制前确认门
        record = record_teach(arm, RECORD_HZ, RECORD_SEC)  # 50 Hz × 4 s → 约 200 帧
        print(f"    录制完成 {record.shape[0]} 帧，起点 q = {vec_str(record[0])} rad")   # 录制结果

        # ---- ③ 切回 MOTION 并以 10 Hz 逐路点回放 ----
        print("=" * 78)                               # 大节横幅：开
        print("③ 轨迹回放")                             # 大节标题
        print("=" * 78)                               # 大节横幅：闭
        arm.set_event(SafetyEvent.NORMAL)             # 正常事件 → FSM 目标态 MOTION（经 HOME 中转、途中先回 home 位形）
        if arm.wait_for_motion_state(MotionState.MOTION, timeout=10.0) != 1:  # 阻塞等待回到 MOTION
            print("    10 秒内未回到 MOTION 正常运动态，安全收尾退出。")
            return                                    # 提前返回：with 退出自动安全收尾
        input(">>> 回放将复现刚才的轨迹（首个路点相对 home 可有小幅跳变），"
              "确认工作空间已清空后回车……")
        replay_teach(arm, record, REPLAY_AHEAD)       # 10 Hz 逐路点下发复现示教轨迹
        print(f"    回放完成，当前 q = {vec_str(arm.joint_state_arm.q)} rad")

        # ---- ④ 收尾：阻塞安全停机（经 HOME 回零失能），with 退出自动断连 ----
        print("=" * 78)                               # 大节横幅：开
        print("④ 安全收尾")                             # 大节标题
        print("=" * 78)                               # 大节横幅：闭
        arm.stop_motion_fsm()                         # 阻塞等待回零到位后失能（危险度更高时直接失能）
        print("    已安全停机（回零失能完成），with 退出后自动断开总线。例程结束。")


if __name__ == "__main__":       # 直接运行本文件时执行 main()（被 import 时不执行）
    try:                         # 中断兜底：Ctrl+C 传播出 main 后在此吞掉、只留提示
        main()                   # 运行主流程（with 内任一时刻 Ctrl+C 均先触发 with 退出收尾再传播至此）
    except KeyboardInterrupt:    # 捕获并吞掉 Ctrl+C：with 退出路径已置 DISABLE 并完成 close() 安全收尾
        print("\n检测到 Ctrl+C，例程已中止——已随 with 退出自动安全收尾（停 FSM → 失能 → 断连）。")
