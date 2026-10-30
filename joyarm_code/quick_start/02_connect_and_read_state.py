"""
==============================================================================
例程 02 —— 硬件连接与状态读取（需真机）
==============================================================================
【功能概要】
    1. 连接 / 断开本体 arm 与末端 end 电机总线，查询连接状态三态取值；
    2. 启动 state_viz 状态可视化（默认自动弹原生窗口，可用环境变量禁用）；
    3. 使能 / 失能两端电机，观察使能状态三态取值（True / False / None）；
    4. 双口径读关节状态：get_state_arm（读时刷新）与 joint_state_arm（被动快照）；
    5. 以 0.2 Hz（5 s 一拍、共 6 拍）刷新结构化状态表，并列对照两口径 q 差异；
    6. 读取控制模式、逐关节状态码、健康检查结果与电机参数 pmax。
    全程不主动运动机械臂；另附三个高危接口的注释示例（默认不执行）。

【原理简介】
    JoyArm 经 backend 后端与电机总线通信：连接后后台低频刷新线程持续读帧，为每个关节维护一份 JointState 快照（q / dq / tau / 温度 / 状态码）。
    读状态有两种口径——get_state_arm 在调用瞬间主动发读帧、返回此刻最新值（要实时数据时用）；joint_state_arm 只读后台线程最近一次成功读帧的缓存、不触发读帧（高频轮询时用，不占总线）。
    两口径存在时间差，本例程并排打印以观察差异。
    state_viz 是独立的只读可视化线程 + 本地 HTTP 监控页面（meshcat 3D 模型 + 状态面板），不要求已连接，连接后快照自动出数。

【安全须知】
    - 清空机械臂工作半径内的人员与障碍物，机械臂稳固放置后再上电
    - 熟悉急停开关位置，运行中保持随手可及，异常立即拍下急停
    - 使能后电机进入锁定状态，切勿强行扳动关节
    - 周期刷新阶段保持机械臂静止；随时可 Ctrl+C 提前进入安全收尾

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（需连接对应的机械臂）
    python quick_start/02_connect_and_read_state.py <型号>

【预期输出】
    依次打印连接 / 可视化 / 使能 / 两口径状态 / 模式 / 状态码 / 参数读取结果，随后每 5 s 刷新一拍结构化状态表（共 6 拍）；
    机械臂全程保持不动。
==============================================================================
"""

from pathlib import Path                        # 组装脚本上级目录，保证未安装 joyarm_core 也能运行
import sys                                      # 向模块搜索路径注入上级目录

import numpy as np                              # 数值向量四舍五入显示
import time                                     # 周期刷新的拍间等待

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装库时直接运行本脚本
from joyarm_core import joyArm                  # noqa: E402  机械臂统一门面类

BEATS = 6                                       # 周期刷新总拍数
PERIOD_S = 5.0                                  # 拍周期（秒）：1 / 5 s = 0.2 Hz


# ============================================================
# 一、命令行参数解析：型号必填，缺失即打印用法退出
# ============================================================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并以退出码 1 结束。"""
    if len(sys.argv) != 2:                                  # 必须恰好携带 1 个型号参数
        sys.exit(f"用法：python quick_start/02_connect_and_read_state.py <型号>")  # 缺参退出，不进入演示
    return sys.argv[1].strip()                              # 型号名


# ============================================================
# 二、打印格式化助手：表格对齐与数值显示
# ============================================================
def vec_str(v) -> str:
    """数值向量转字符串：None 显「—」，其余逐元素保留 3 位小数。"""
    return "—" if v is None else np.round(v, 3).tolist()    # round + tolist 便于终端阅读


def disp_width(s: str) -> int:
    """粗估字符串的终端显示宽度：CJK 全角字符按 2 列、其余按 1 列计。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)    # 0x2E80 起为中日韩区块


def pad(s: str, width: int, align: str = "left") -> str:
    """按显示宽度补空格对齐表格列（中文表头与 ASCII 数据混排时仍列对齐）。"""
    gap = max(width - disp_width(s), 0)                     # 需补的空格数（不足则不补）
    return (s + " " * gap) if align == "left" else (" " * gap + s)  # 左 / 右对齐二选一


def num(x, width: int = 10, nd: int = 3) -> str:
    """单数值右对齐定宽显示（nd 位小数）；None 显「—」。"""
    return pad("—", width, "right") if x is None else pad(f"{x:.{nd}f}", width, "right")


def cell(arr, i: int, width: int = 10, nd: int = 3) -> str:
    """取数组第 i 个元素定宽显示；数组为 None 或过短时显「—」占位。"""
    ok = arr is not None and i < len(arr)                   # 有数据才取元素，否则占位
    return num(float(arr[i]) if ok else None, width, nd)    # 统一走 num 格式化


# ============================================================
# 三、状态打印：单次快照 / 周期主表 / 双口径对照
# ============================================================
def print_snapshot(tag: str, s) -> None:
    """打印一份关节状态快照（q / dq / tau / 双温度 / 状态码）；s 为 None 显读取失败。"""
    if s is None:                                           # 读取失败：显式提示而非崩溃
        print(f"{tag}：读取失败（返回 None）")
        return
    print(f"{tag}：")                                        # 以下逐物理量键值行输出
    print(f"    位置 q (rad)     ：{vec_str(s.q)}")            # 关节位置向量
    print(f"    速度 dq (rad/s)  ：{vec_str(s.dq)}")           # 关节速度向量
    print(f"    力矩 tau (N·m)   ：{vec_str(s.tau)}")          # 关节力矩向量
    print(f"    驱动板温度 (℃)  ：{vec_str(s.temp_mos)}")         # MOS 温度（固件不反馈则恒 0）
    print(f"    转子温度 (℃)    ：{vec_str(s.temp_rotor)}")      # 转子线圈温度（同上）
    print(f"    状态码 error     ：{'—' if s.error is None else s.error.tolist()}"
          "（0=失能、1=使能、≥2=故障）")                              # 状态码逐关节列出 + 口径说明


def print_beat_table(k: int, beats: int, s_get, names) -> None:
    """打印一拍主表：关节名 + q / dq / tau / error（get_state_arm 读时刷新口径）。"""
    print(f"第 {k}/{beats} 拍｜get_state_arm 主表（读时刷新：本拍主动发读帧取最新值）")
    header = (pad("关节名", 10) + pad("q (rad)", 11, "right") + pad("dq (rad/s)", 12, "right")
              + pad("tau (N·m)", 11, "right") + pad("error", 7, "right"))  # 表头与数据列同宽
    print(header)                                           # 输出表头行
    if s_get is None:                                       # 本拍读时刷新失败：以提示行占位
        print("（本拍 get_state_arm 返回 None：本轮读帧失败，下一拍自动重试）")
    for i, name in enumerate(names):                        # 逐关节一行，列宽与表头一致
        row = pad(name, 10)                                 # 关节名列（cfg 顺序 = q 向量顺序）
        row += cell(s_get.q if s_get is not None else None, i, 11)  # 位置 q
        row += cell(s_get.dq if s_get is not None else None, i, 12)  # 速度 dq
        row += cell(s_get.tau if s_get is not None else None, i, 11)  # 力矩 tau
        row += cell(s_get.error if s_get is not None else None, i, 7, nd=0)  # 状态码（整数）
        print(row)                                          # 输出一行数据


def print_compare_table(s_get, s_snap, names) -> None:
    """并列对照两口径 q：读时刷新 vs 被动快照，并给出逐关节差值。"""
    print("双口径对照｜q (rad)：get_state_arm（读时刷新） vs joint_state_arm（被动快照）")
    header = (pad("关节名", 10) + pad("q 读时刷新", 12, "right")
              + pad("q 被动快照", 12, "right") + pad("Δq 差值", 10, "right"))  # Δq = 前者 − 后者
    print(header)                                           # 输出表头行
    for i, name in enumerate(names):                        # 逐关节并排两口径与差值
        q_get = (float(s_get.q[i])                          # 口径一：本拍主动读
                 if s_get is not None and i < len(s_get.q) else None)
        q_snap = (float(s_snap.q[i])                        # 口径二：后台线程最近缓存
                  if s_snap is not None and i < len(s_snap.q) else None)
        d_q = None if (q_get is None or q_snap is None) else q_get - q_snap  # 任一口径缺数则差值缺省
        row = pad(name, 10)                                 # 关节名列
        row += num(q_get, 12)                               # 读时刷新 q
        row += num(q_snap, 12)                              # 被动快照 q
        row += num(d_q, 10)                                 # 差值（毫秒级时间差所致，通常很小）
        print(row)                                          # 输出一行数据


# ============================================================
# 四、真机主流程：连接 → 可视化 → 使能 → 读取 → 周期刷新 → 安全收尾
# ============================================================
def main() -> None:
    """真机流程：连接 → 可视化 → 使能 → 双口径读取 → 模式/故障/参数 → 周期刷新 → 安全收尾。"""
    model = parse_model()                                   # 必填型号参数
    # with 创建机械臂对象（型号 = 命令行参数）；退出自动 close() 安全收尾
    with joyArm(model) as arm:

        # ---- ① 连接：本体与末端电机总线 ----
        print("=" * 78)                                     # 大节分隔线
        print("① 连接本体 arm 与末端 end 电机总线")                    # 大节标题：连接
        ok = arm.connect_arm()                              # 连接本体总线；成功后自动设 cfg 默认控制模式
        print(f"connect_arm 返回：{ok}；is_connected_arm：{arm.is_connected_arm}")  # 回显连接结果与标志
        ok = arm.connect_end()                              # 连接末端（夹爪）电机总线
        print(f"connect_end 返回：{ok}；is_connected_end：{arm.is_connected_end}")  # 回显连接结果与标志
        print("（连接标志三态：True=已连接、False=连接失败、None=未知）")       # 三态口径说明

        # ---- ② state_viz 状态可视化（在读接口演示之前） ----
        print("=" * 78)                                     # 大节分隔线
        print("② 启动 state_viz 状态可视化（单页面监控网页 + 只读刷新线程）")     # 大节标题：可视化
        print("    - 默认自动弹出原生监控窗口（左静态参考 | 中 meshcat 3D | 右状态面板）")  # 提示：默认弹窗
        print("    - 无显示器 / 远程环境：设环境变量 JOYARM_VIZ_NOWINDOW=1 可禁用弹窗，")  # 提示：禁用弹窗方法
        print("      仅保留监控网页 URL（启动后见终端日志）")                # 提示：禁用后仍可网页监控
        print("    - 不要求已连接：未连接也能启动，连接后 backend 快照自动出数")    # 提示：启动前置条件
        arm.start_state_viz(traj_points=200)                # 启动可视化（轨迹采样点 200；0=关闭轨迹）

        # ---- ③ 使能：三态取值 True / False / None ----
        print("=" * 78)                                     # 大节分隔线
        print("③ 使能两端电机（使能后关节锁定，切勿强行扳动）")                   # 大节标题：使能
        print(f"使能前 is_enabled_arm：{arm.is_enabled_arm}"
              "（None=未知：未读到有效状态、各关节不一致或含故障码）")              # 三态口径说明
        input("现场确认安全后按回车使能（Ctrl+C 可随时中止）…")                # 危险动作前人工确认
        ret = arm.enable_arm()                              # 使能本体全部电机：1=全部成功、0=有失败
        print(f"enable_arm 返回：{ret}；使能后 is_enabled_arm：{arm.is_enabled_arm}")  # 回显使能结果与三态
        ret = arm.enable_end()                              # 使能末端电机（返回 1/0，含义同上）
        print(f"enable_end 返回：{ret}；使能后 is_enabled_end：{arm.is_enabled_end}")  # 回显使能结果与三态

        # ---- ④ 双口径单次读取：读时刷新 vs 被动快照 ----
        print("=" * 78)                                     # 大节分隔线
        print("④ 关节状态的两种读取口径（单次读取演示）")                      # 大节标题：双口径单次读取
        print("    - get_state_arm：读时刷新——调用瞬间主动发读帧，返回此刻最新值")  # 口径一说明
        print("      （要实时数据时用，占用一次总线读）")                    # 口径一说明（续）
        print("    - joint_state_arm/end：被动快照——只读后台低频刷新线程的最近缓存，")  # 口径二说明
        print("      不触发读帧（高频轮询 / 监视时用，不占总线）")              # 口径二说明（续）
        s = arm.get_state_arm()                             # 口径一：主动读（读时刷新）
        print_snapshot("get_state_arm（读时刷新）", s)            # 打印口径一快照
        print_snapshot("joint_state_arm（被动快照）", arm.joint_state_arm)  # 口径二：本体被动快照
        print_snapshot("joint_state_end（夹爪被动快照）", arm.joint_state_end)  # 末端同理（口径二）

        # ---- ⑤ 控制模式与故障检查 ----
        print("=" * 78)                                     # 大节分隔线
        print("⑤ 控制模式与逐关节状态码")                              # 大节标题：模式与故障
        mode = arm.get_mode_arm()                           # 读模式（读时刷新；全组一致才返回该模式）
        print(f"arm 控制模式：{mode}（连接时已自动设默认模式；全组一致才返回，否则 None）")  # 回显模式（枚举或 None）
        errs = arm.get_error_arm()                          # 各关节状态码缓存（0=失能、1=使能、≥2=故障）
        print(f"arm 状态码缓存 get_error_arm：{errs}")            # 回显逐关节状态码
        print(f"健康检查 check_error_arm：{arm.check_error_arm()}（全 0/1 判正常 True）")  # 回显健康检查结果

        # ---- ⑥ 电机参数读取 ----
        print("=" * 78)                                     # 大节分隔线
        print("⑥ 电机参数读取")                                   # 大节标题：参数读取
        vals = arm.read_param_arm("pmax")                   # pmax = MIT 位域位置量程 (rad)，逐关节返回
        print(f"read_param_arm('pmax') 返回：{vals}")          # 回显逐关节参数值
        print("（可读键还有 vmax / tmax / pos_kp 等；写入接口见下方高危注释块）")  # 提示：其他可读键

        # ---- ⑦ 高危接口（默认注释不执行） ----
        print("=" * 78)                                     # 大节分隔线
        print("⑦ 高危接口注释块（理解风险并确认需要时，再手动放开执行）")              # 大节标题：高危接口
        # arm.write_param_arm("pmax", [12.5] * arm.n_arm)  # 写电机参数寄存器（RAM 态、断电丢失；写错可致电机行为异常）
        # arm.set_zero_arm()                               # 把当前位置记为零点（零位标定；执行后零点即被改写）
        # arm.clear_error_arm()                            # 清除各关节硬件故障码（先排查并消除故障原因，再清除）
        print("（write_param_arm / set_zero_arm / clear_error_arm 三项见上方注释，默认不执行）")  # 确认未执行高危接口

        # ---- ⑧ 0.2 Hz 周期刷新：结构化状态表 × 6 拍 ----
        print("=" * 78)                                     # 大节分隔线
        print(f"⑧ 周期刷新：{BEATS} 拍 × {PERIOD_S:.0f} s（0.2 Hz）；"
              "拍间用 = 分隔、拍内用 - 分隔；Ctrl+C 可提前结束")             # 本节规则说明
        names = arm.get_joint_names_arm()                   # 关节名 tuple（cfg 顺序 = q 向量顺序）
        if names is None:                                   # 兜底：查询异常时按关节数生成占位名
            names = tuple(f"joint{i + 1}" for i in range(arm.n_arm))
        try:                                                # 循环内捕获 Ctrl+C：提前结束进入安全收尾
            for k in range(1, BEATS + 1):                   # 拍号从 1 计，逐拍刷新
                if k > 1:                                   # 首拍紧邻本节标题，无需再分隔
                    print("=" * 78)                         # 拍与拍之间用 = 分隔
                s_get = arm.get_state_arm()                 # 口径一：本拍主动读（读时刷新）
                s_snap = arm.joint_state_arm                # 口径二：紧随其后取被动快照引用（近似同时刻）
                print_beat_table(k, BEATS, s_get, names)    # 主表：关节名 + q / dq / tau / error
                print("-" * 78)                             # 拍内小节分隔线
                print_compare_table(s_get, s_snap, names)   # 对照表：两口径 q 并排与差值
                if k < BEATS:                               # 末拍后不再等待，直接进入收尾
                    print(f"（{PERIOD_S:.0f} s 后刷新下一拍…）")    # 提示等待，避免误以为卡死
                    time.sleep(PERIOD_S)                    # 拍间等待（此刻 Ctrl+C 即提前结束）
        except KeyboardInterrupt:                           # 捕获拍内任意时刻的 Ctrl+C
            print("\n检测到 Ctrl+C：提前结束状态刷新，进入安全收尾。")

        # ---- ⑨ 安全收尾：失能 → 停可视化 → 断开 ----
        print("=" * 78)                                     # 大节分隔线
        print("⑨ 安全收尾：失能两端 → 停止可视化 → 断开总线（顺序不可颠倒）")         # 大节标题：安全收尾
        ret_a = arm.disable_arm()                           # 主动失能本体电机（1=全部成功、0=有失败）
        ret_e = arm.disable_end()                           # 主动失能末端电机（含义同上）
        print(f"disable_arm 返回：{ret_a}；disable_end 返回：{ret_e}")  # 回显失能结果
        arm.stop_state_viz()                                # 停可视化（收刷新线程、关 HTTP 服务与监控窗口）
        print("stop_state_viz 已调用（在断开总线之前完成可视化收尾）")         # 确认可视化已收尾
        arm.disconnect_end()                                # 断开末端总线（先尽力失能再断、清模式缓存）
        arm.disconnect_arm()                                # 断开本体总线（同上）
        print(f"disconnect_end / disconnect_arm 已调用；"
              f"is_connected_arm：{arm.is_connected_arm}，"
              f"is_connected_end：{arm.is_connected_end}")   # 复查连接标志（应为 False）
    # with 退出（含异常）：__exit__ 调 close() 幂等兜底——arm 已主动断连则豁免失能/断连，仅停线程清缓存
    print("=" * 78)                                         # 收尾分隔线
    print("例程结束：机械臂已失能并断开，全程未主动运动。")                        # 结束语


if __name__ == "__main__":                                  # 直接运行入口（被导入时不执行）
    try:                                                    # 末段兜底：任意时刻 Ctrl+C 均安全退出
        main()                                              # 执行真机演示主流程
    except KeyboardInterrupt:                               # with 内中断时 close() 已自动失能并断连
        print("\n检测到 Ctrl+C：已安全退出（若尚在连接，with 收尾已自动失能并断开总线）。")
