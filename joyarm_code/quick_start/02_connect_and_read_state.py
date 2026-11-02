"""
==============================================================================
例程 02 —— 硬件连接与状态读取（需真机）
==============================================================================
【功能概要】
    1. 连接 / 断开 arm 与 end 的电机总线，查询连接状态（True=已连接、False=未连接、None=未知）；
    2. 启动 state_viz 状态可视化（默认自动弹出原生窗口）；
    3. 使能 / 失能电机，观察使能状态（同样 True / False / None 三种状态）；
    4. 读取控制模式、逐关节状态码、健康检查结果与电机内部参数；
    5. 读关节状态：读状态缓存 joint_state_arm（后台刷新，频率配置于 cfg.backend） + send 指令应答式刷新后 get_state_arm。

【原理简介】
    JoyArm 经 backend 后端与电机总线通信：连接后后台刷新线程持续读帧(若状态过旧则主动读)，为每个关节维护一份 JointState 快照（q / dq / tau / 温度 / 状态码）。
    读状态两种方式：get_state_arm 主动读取拿最新值（需实时数据时用）；joint_state_arm 引用最近一次帧缓存（用作可视化数据）。
    state_viz 是独立的只读可视化线程 + 本地 HTTP 监控页面（中间 meshcat 3D 模型 + 两侧状态面板）。

【安全须知】
    - 清空机械臂工作半径内的人员与障碍，机械臂稳固固定后再上电
    - 机械臂运行异常立即拍下急停或断电，随时可 Ctrl+C 提前进入安全收尾

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（需连接对应的机械臂）
    python quick_start/02_connect_and_read_state.py <型号>  # 如：rebot_arm_b601_dm

【预期输出】
    依次打印连接 / 可视化 / 使能 / 控制模式 / 状态码 / 参数读取结果，随后引用缓存与主动读关节状态；
    机械臂全程保持不动。
==============================================================================
"""

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from joyarm_core import joyArm, listArm, ControlMode


# ================= 命令行参数解析：型号必填，缺失即打印用法退出 ======================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并列出可用型号（listArm），以退出码 1 结束。"""
    if len(sys.argv) != 2:                                # 必须恰好携带 1 个型号参数
        print(f"用法：python quick_start/02_connect_and_read_state.py <型号>")  # 缺参：先打印用法
        listArm()                                         # 顺带列出全部可用型号
        sys.exit(1)                                       # 退出码 1，不进入演示
    return sys.argv[1].strip()                            # 型号名


# ======= main() 流程：连接 → 可视化 → 使能 → 模式/错误码/参数 → 两种读法节拍 → 安全收尾 ===========
def main() -> None:
    """真机流程：连接 → 可视化 → 使能 → 模式/状态码/参数 → 引用缓存与主动读各自统计 → 安全收尾。"""
    model = parse_model()                                 # 必填型号参数

    with joyArm(model) as arm:                            # with 创建机械臂对象；退出自动 close() 安全收尾

        print("=============== ① 连接：arm/end ================================== ")
        ok = arm.connect_arm()                            # 连接 arm
        print(f"connect_arm 返回：{ok}；is_connected_arm：{arm.is_connected_arm}")
        ok = arm.connect_end()                            # 连接 end
        print(f"connect_end 返回：{ok}；is_connected_end：{arm.is_connected_end}")
        print("（连接的三种结果：True=已连接、False=未连接、None=未知）")  # 三种结果说明

        print("=============== ② 可视化 ========================================= ")
        print("默认弹出原生监控窗口；无显示器 / 远程环境设 JOYARM_VIZ_NOWINDOW=1 可禁用")  # 弹窗与禁用法
        arm.start_state_viz(traj_points=200)              # 启动可视化
        time.sleep(2)                                     # 等待启动完毕

        print("=============== ③ 使能 ========================================== ")
        print(f"使能前 is_enabled_arm：{arm.is_enabled_arm}（None=未知）")
        ret = arm.enable_arm()                            # 使能 arm ：1=全部成功、0=有失败
        print(f"enable_arm 返回：{ret}；使能后 is_enabled_arm：{arm.is_enabled_arm}")
        ret = arm.enable_end()                            # 使能 end ：1=全部成功、0=有失败
        print(f"enable_end 返回：{ret}；使能后 is_enabled_end：{arm.is_enabled_end}")

        print("=============== ④ 模式与错误码 ===================================== ")
        mode = arm.get_mode_arm()                              # 读模式（主动读；全组一致返回该模式）
        print(f"arm 控制模式：{mode}（连接时已自动设置）")
        errs = arm.get_error_arm()                             # 各关节状态码缓存（0=失能、1=使能、≥2=故障）
        print(f"arm 状态码缓存 get_error_arm：{errs}")           # 逐关节状态码
        print(f"状态检查 check_error_arm：{arm.check_error_arm()}（全 0/1 判 True）")

        print("=============== ⑤ 参数读取 ======================================== ")
        vals = arm.read_param_arm("pos_kp")                    # 参数读取
        print(f"read_param_arm('pos_kp') 返回：{vals}")         # 逐关节参数值
        print("（可读键名依型号而定；由实际硬件的 backend 定义）")

        print("=============== ⑥ 引用状态缓存 joint_state_arm ==================== ")
        hz = arm.get_config().get("backend", {}).get("state_refresh_hz", 20.0)  # 后台刷新频率
        print(f"（后台刷新，频率 {hz:g} Hz 配置于 cfg.backend；不主动读取、只取缓存，陈旧应<=刷新周期）")
        s = arm.joint_state_arm                            # 引用后台缓存（不发帧）
        now = time.time()                                  # 当前时刻
        print(f"状态快照时间戳 {s.t:.3f}｜当前时间戳 {now:.3f}｜陈旧度 {now - s.t:.3f} s" if s.t else "快照 —（从未收到帧）")

        print("=============== ⑦ 主动读关节状态：MIT 空指令应答式刷新 ======================= ")
        print("（读旧快照 t → 发 MIT 指令 → 电机应答式刷新状态 → 组装新状态 → 统计陈旧）")
        ret = arm.set_mode_arm(ControlMode.MIT)            # 切 MIT 模式：1=成功（乐观更新模式缓存）、0=失败
        print(f"set_mode_arm(MIT) 返回：{ret}；当前模式：{arm.get_mode_arm()}")
        if ret == 1:                                       # 切模式成功
            t_send = time.time()                           # 发出时刻
            arm.send_mit_arm(arm.zero_arm, arm.zero_arm, arm.zero_arm, arm.zero_arm, arm.zero_arm)
            s = arm.get_state_arm()                        # 组装状态快照
            while (s is None or s.t - t_send <= 0.000001): # 等新状态更新完毕
                time.sleep(0.0000001)
                s = arm.get_state_arm()
            now = time.time()                              # 当前时刻
            print(f"状态快照时间戳 {s.t:.3f}｜当前时间戳 {now:.3f} ｜陈旧度 {now - s.t:.3f} s")

        # print("=============== ⑧ 收尾 ========================================== ")
        # ret_a = arm.disable_arm()                         # 失能 arm
        # ret_e = arm.disable_end()                         # 失能 end
        # print(f"disable_arm 返回：{ret_a}；disable_end 返回：{ret_e}")
        # arm.stop_state_viz()                              # 停可视化
        # arm.close
    print("=============== 结束自动收尾：机械臂已失能并断开 ==================== ")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:                             # 任意时刻 Ctrl+C 均安全退出，with 内中断时 close() 已自动失能并断连
        print("\n检测到 Ctrl+C：已安全退出（若尚在连接，with 收尾已自动失能并断开总线）。")
