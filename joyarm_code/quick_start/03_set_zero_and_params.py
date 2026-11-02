"""
==============================================================================
例程 03 —— 标零与参数设置（需真机）
==============================================================================
【功能概要】
    1. 连接并使能本体（arm）与末端（end）电机总线（型号由命令行参数指定）
    2. 标零演示：set_zero_arm / set_zero_end 打印说明与当前位置；执行需连续两次输入
       大写 YES 确认（默认跳过、零点不动）
    3. 参数读取：read_param_arm / read_param_end 读 pmax / vmax / tmax，逐关节对齐表格
       输出（读取失败显「—」并说明）
    4. 参数写入安全演示：读当前值 → write_param_arm / write_param_end 把刚读到的值原样写回 →
       读回核对一致（不改变电机行为；真正改值的写法以注释示例给出，默认不执行）

【原理简介】
    每个关节电机内部有一组 RAM 参数寄存器，主控按「参数键 → 寄存器号」经总线逐关节读写；
    本型号可读写 MIT 模式的三个量程 pmax/vmax/tmax 与四个闭环增益（pos_kp/pos_ki/vel_kp/vel_ki），
    另有只读的硬件/软件版本与序列号（hw_ver/sw_ver/sn，写侧直接拒绝）。
    这些参数存于 RAM：断电即丢、上电恢复固件默认；写错量程会使指令按错误量程换算，
    电机行为立即异常，故写参数务必「先读 → 写 → 读回核对」。
    零点标定：编码器只能测相对转角，电机把「标零时刻的读数」记为零点，此后关节角均相对
    零点计算；set_zero_* 即把当前位置记为零点——零点一旦改写，本次上电内后续所有位置
    指令的基准随之改变（对应教程「整机初始化标定与调试」章）。

【安全须知】
    - 标零高危：执行前必须把机械臂摆到该型号定义的零位构型，并清空工作半径内的人员与障碍物
    - 熟悉急停开关位置，运行中保持随手可及，异常立即拍下急停
    - 使能后电机进入锁定状态，切勿强行扳动关节；Ctrl+C 随时中断，with 退出自动安全收尾
    - 本例程的参数写演示只做「原值写回」（读出多少写回多少），不改变任何电机行为

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（需已连接真机并上电）
    python quick_start/03_set_zero_and_params.py <型号>

【预期输出】
    依次打印连接/使能结果、标零说明与当前位置（默认跳过执行）、pmax/vmax/tmax 逐关节
    表格、arm 与 end 两侧原值写回核对结果（全部一致），最后失能退出。
==============================================================================
"""

from pathlib import Path  # 路径模块：定位本文件上一层目录（joyarm_code/）
import sys                # 系统模块：解析命令行参数、注入模块搜索路径
import time               # 时间模块：标零后等待一个刷新周期

import numpy as np  # 数值计算：状态向量取整与最大绝对值计算

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装 joyarm_core 时也可直接运行
from joyarm_core import joyArm   # JoyArm 整机类（连接/标零/参数读写统一入口）

ZERO_TOL = 0.05  # 标零复核容差（rad）：与 cfg arrival_tol 的关节到位容差同量级


# ============================================================
# 一、命令行解析：必填型号参数，缺参打印用法并以退出码 1 结束
# ============================================================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并以退出码 1 结束。"""
    if len(sys.argv) != 2:                       # 必须恰好携带 1 个型号参数
        sys.exit(f"用法：python quick_start/03_set_zero_and_params.py <型号>")  # 缺参退出，不进入演示
    return sys.argv[1].strip()                   # 型号名


# ============================================================
# 二、打印格式化助手：显示宽度计算与表格列对齐补空格
# ============================================================
def disp_width(s: str) -> int:
    """计算字符串的终端显示宽度（中文字符按 2 列计），用于表格列对齐。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)  # 逐字符累加：CJK 记 2、其余记 1


def pad(s: str, width: int, align: str = "left") -> str:
    """按显示宽度补空格对齐，保证含中文的表格列对齐。

    :param s: 待对齐字符串。
    :param width: 目标显示宽度（列宽）。
    :param align: "right" 右对齐（数值列），"left" 左对齐（名称列）。
    :return: 补齐空格后的字符串。
    """
    gap = max(width - disp_width(s), 0)                          # 需补的空格数（不足 0 则不补）
    return " " * gap + s if align == "right" else s + " " * gap  # 按对齐方向补空格


# ============================================================
# 三、标零演示：两次 YES 确认与执行后复核
# ============================================================
def confirm_two_level(label: str) -> bool:
    """标零的两次确认：两次均须输入大写 YES 才放行；任何一级输入其他内容即跳过。

    :param label: 总线名称（如「本体 arm」），用于提示语。
    :return: True = 两次均确认 YES；False = 任一级未确认（跳过执行）。
    """
    print(f"  【{label} 标零确认 1/2】即将把该总线的电机当前位置记为零点（零点即被改写）。")
    if input("    输入 YES 继续，其他任意输入跳过：").strip() != "YES":  # 第一级：输入非 YES 即放弃
        return False
    print("  【标零确认 2/2】请确认已理解：当前位置将被记为零点、零点即被改写，")
    return input("    输入 YES 执行，其他任意输入跳过：").strip() == "YES"  # 第二级：再输一次 YES 才执行


def demo_set_zero(arm: "joyArm", family: str) -> None:
    """单条总线的标零演示：打印说明与当前位置 → 两次 YES 确认 → 执行 → 复核 q 接近 0。

    :param arm: 已连接的 JoyArm 实例。
    :param family: `"arm"`（本体六关节）或 `"end"`（末端夹爪电机）。
    """
    label = "本体 arm" if family == "arm" else "末端 end"                     # 提示语用的总线名称
    get_state = arm.get_state_arm if family == "arm" else arm.get_state_end  # 主动读的状态接口
    set_zero = arm.set_zero_arm if family == "arm" else arm.set_zero_end     # 设零接口（arm 侧受 FSM 互斥，本例程未启 FSM）
    print("-" * 78)                                                          # 小节分隔线
    print(f"○ {label} 标零（set_zero_{family}）：默认只演示不执行。")        # 小节标题（说明默认路径不执行）
    print("  含义：把该总线电机「当前时刻的位置」记为零点；执行后零点即被改写，")                             # 标零语义
    print("  本次上电内后续一切关节角都以新零点为基准（高危、影响持久）。")  # 影响范围提示
    s = get_state()                                                          # 读当前关节位置（主动读）
    if s is None:                                                            # 读取失败：无法展示当前位置，直接跳过本段
        print(f"  当前位置读取失败（返回 None），跳过 {label} 标零演示。")   # 失败分支提示
        return                                                               # 提前返回，不进入确认流程
    print(f"  当前位置 q（rad）：{np.round(s.q, 3).tolist()}")               # 当前位置基线
    print("  执行前请确认：机械臂已摆到该型号定义的零位构型（见教程「整机初始化标定与调试」章）。")                    # 执行前提
    if not confirm_two_level(label):                                         # 两次确认：任一级非 YES 即跳过
        print(f"  未两次确认 YES：已跳过 {label} 标零（零点未改动）。")
        return
    ret = set_zero()  # 执行标零：1 = 全部成功、0 = 任一失败
    print(f"  set_zero_{family} 返回：{ret}（1=全部成功、0=任一失败）")
    time.sleep(0.3)   # 等 0.3 s：让后台刷新线程收到标零后的新状态帧
    s2 = get_state()  # 复核：标零后当前位置应接近全零
    if s2 is None:    # 复核读取失败：提示后结束本段
        print("  复核读取失败（返回 None）。")
        return
    q_max = float(np.max(np.abs(s2.q)))                               # 各关节绝对值的最大者
    ok = q_max <= ZERO_TOL                                            # 容差内即认为「当前位置已成新零点」
    print(f"  复核 q（rad）：{np.round(s2.q, 3).tolist()}")           # 标零后的位置读数
    print(f"  最大绝对关节角 |q|max = {q_max:.4f} rad（容差 {ZERO_TOL} rad）→ "  # 复核结论：容差内即标零生效
          f"{'接近零点，标零生效' if ok else '未接近零点（状态帧可能尚未刷新，可重跑复核）'}")


# ============================================================
# 四、参数读取：pmax / vmax / tmax 逐关节对齐表格
# ============================================================
def print_param_table(names, key_vals: dict, units: dict) -> None:
    """打印逐关节参数表：首列关节名，其后每键一列；None 单元格显「—」。

    :param names: 关节名序列（cfg 顺序 = q 向量顺序）。
    :param key_vals: {参数键: 逐关节值列表 或 None}。
    :param units: {参数键: 单位字符串}。
    """
    header = pad("关节", 12, "left")                                             # 表头首列：关节名（左对齐）
    header += "".join(pad(f"{k} ({u})", 17, "right") for k, u in units.items())  # 表头数值列：键名 + 单位（右对齐）
    print(header)
    for i, name in enumerate(names):                                            # 逐关节一行
        row = pad(str(name), 12, "left")                                        # 首列：关节名
        for vals in key_vals.values():                                          # 该关节在各参数键下的取值
            v = "—" if (vals is None or vals[i] is None) else f"{vals[i]:.3f}"  # 读不到显「—」
            row += pad(v, 17, "right")                                          # 数值列右对齐
        print(row)


def demo_read_params(arm: "joyArm") -> None:
    """读取 arm 与 end 的 pmax / vmax / tmax，逐关节表格输出。

    可用参数键由 backend 键表决定（本型号：pos_kp / pos_ki / vel_kp / vel_ki /
    pmax / vmax / tmax，另有只读的 hw_ver / sw_ver / sn）；末端以实际返回为准，
    读不到（None）在表格中显「—」。
    """
    units = {"pmax": "rad", "vmax": "rad/s", "tmax": "N·m"}  # 三键的含义与单位（MIT 模式量程）
    for family in ("arm", "end"):                            # 先本体后末端，各打一张表
        print("-" * 78)
        read = arm.read_param_arm if family == "arm" else arm.read_param_end                 # 参数整组读接口
        names = arm.get_joint_names_arm() if family == "arm" else arm.get_joint_names_end()  # 关节名（cfg 顺序）
        key_vals = {k: read(k) for k in units}                                               # 每键一次整组读：逐关节值列表或 None
        print(f"○ {'本体 arm' if family == 'arm' else '末端 end'} 参数读取 read_param_{family}（pmax / vmax / tmax）")
        if any(v is None for v in key_vals.values()):  # 有键读取失败：显式说明，不中断后续演示
            print("  注：值为「—」的键读取失败（未连接 / 总线异常 / 电机不支持该键）。")
        print_param_table(names, key_vals, units)  # 逐关节对齐表格


# ============================================================
# 五、参数写入安全演示：原值写回并读回核对
# ============================================================
def demo_write_same_value(arm: "joyArm", family: str) -> None:
    """单条总线的参数写入安全演示：读 pmax 当前值 → write_param 写同值 → 读回核对一致。

    原值写回不改变寄存器内容，电机行为不变；用于演示写参数的标准安全流程
    （arm 侧 write_param_arm 与 end 侧 write_param_end 接口对称，各演示一遍）。

    :param arm: 已连接的 JoyArm 实例。
    :param family: `"arm"`（本体六关节）或 `"end"`（末端夹爪电机）。
    """
    label = "本体 arm" if family == "arm" else "末端 end"                                # 提示语用的总线名称
    read = arm.read_param_arm if family == "arm" else arm.read_param_end                 # 参数整组读接口
    write = arm.write_param_arm if family == "arm" else arm.write_param_end              # 参数整组写接口
    names = arm.get_joint_names_arm() if family == "arm" else arm.get_joint_names_end()  # 关节名（cfg 顺序）
    print("-" * 78)                                                                      # 小节分隔线
    print(f"○ {label} 安全写法演示：read 当前值 → write_param_{family} 写同值 → read 回核对")            # 小节标题
    vals = read("pmax")                                                                  # 第一步：写前整组读（逐关节当前量程）
    if vals is None:                                                                     # 写前读失败：无可靠基准，跳过写演示
        print("  写前读取失败（返回 None），跳过写演示。")
        return
    ret = write("pmax", vals)  # 第二步：把读到的原值原样写回（RAM 寄存器值不变）
    print(f"  write_param_{family}('pmax', 写前值) 返回：{ret}（1=成功、0=失败）")
    back = read("pmax")  # 第三步：写后再读一次，用于核对
    if back is None:     # 写后读失败：无法核对，提示后结束
        print("  写后读回失败（返回 None），无法核对。")
        return
    header = (pad("关节", 12, "left") + pad("写前 pmax (rad)", 17, "right")
              + pad("写后 pmax (rad)", 17, "right") + pad("偏差", 12, "right"))        # 核对表表头
    print(header)                                                                     # 打印表头
    all_ok = True                                                                     # 全关节核对结果汇总标志
    for i, name in enumerate(names):                                                  # 逐关节比对写前/写后值
        diff = abs(vals[i] - back[i])                                                 # 写前写后绝对偏差
        ok = diff <= 1e-3                                                             # 一致判据：偏差 ≤ 1e-3（float32 精度）
        all_ok = all_ok and ok                                                        # 任一关节不一致则整体判不一致
        print(pad(str(name), 12, "left") + pad(f"{vals[i]:.3f}", 17, "right")         # 关节名 + 写前值
              + pad(f"{back[i]:.3f}", 17, "right") + pad(f"{diff:.4f}", 12, "right")  # 写后值 + 偏差
              + (" 一致" if ok else " 不一致!"))                                      # 单关节结论（前置空格与偏差列隔开）
    print(f"  核对结果：{'全部一致——原值写回未改变电机行为' if all_ok else '存在不一致（见上表），请排查总线与电机状态'}")
    # ---- 风险与「真正改值」的写法（默认不执行，理解风险并确认需要时再手动放开） ----
    # write_param 写的是电机 RAM 寄存器：断电即丢、上电恢复固件默认；
    # 写错量程/增益会直接改变电机行为（指令按错误量程换算），可能引发失控，务必先读后写、写回核对。
    # arm.write_param_arm("vmax", [30.0] * arm.n_arm)  # 例：本体全部关节速度量程改为 30.0 rad/s
    # arm.write_param_end("pmax", [12.5] * arm.n_end)  # 例：末端电机位置量程改为 12.5 rad
    # 注意：pos_kp 等增益键在每次切 POSITION/VELOCITY 模式时会被 cfg 增益自动随写覆盖；
    #       只读键（hw_ver / sw_ver / sn）写侧直接拒绝。


# ============================================================
# 六、主流程：连接使能 → 标零与参数演示 → 失能收尾
# ============================================================
def main() -> None:
    """真机流程：连接使能 → 标零演示（默认跳过）→ 参数读取 → 原值写回演示 → 失能收尾。"""
    model = parse_model()  # 必填型号参数
    # ---- ① 连接与使能 ----
    print("=" * 78)
    print("① 连接与使能")
    with joyArm(model) as arm:  # with 退出（含异常）自动安全收尾：停线程→失能→断连
        ok = arm.connect_arm()  # 连接本体电机总线（通道取配置缺省值）
        print(f"connect_arm 返回：{ok}；is_connected_arm：{arm.is_connected_arm}")
        ok = arm.connect_end()  # 连接末端（夹爪）电机总线
        print(f"connect_end 返回：{ok}；is_connected_end：{arm.is_connected_end}")
        ret = arm.enable_arm()  # 使能本体全部关节电机：1=全部成功、0=有失败
        print(f"enable_arm 返回：{ret}；使能后 is_enabled_arm：{arm.is_enabled_arm}")
        ret = arm.enable_end()  # 使能末端电机（返回 1/0，含义同上）
        print(f"enable_end 返回：{ret}；使能后 is_enabled_end：{arm.is_enabled_end}")

        # ---- ② 标零：默认演示路径只打印说明，不执行 ----
        print("=" * 78)
        print("② 标零（set_zero_arm / set_zero_end）：默认演示路径只打印说明，不执行")
        demo_set_zero(arm, "arm")  # 本体标零演示：两次 YES 确认后才真正执行
        demo_set_zero(arm, "end")  # 末端标零演示：流程同上（零点改写影响开合行程基准）

        # ---- ③ 参数读取 ----
        print("=" * 78)
        print("③ 参数读取（read_param_arm / read_param_end）")
        demo_read_params(arm)  # arm 与 end 的 pmax/vmax/tmax 逐关节表格

        # ---- ④ 参数写入安全演示：原值写回，不改变电机行为 ----
        print("=" * 78)
        print("④ 参数写入安全演示（write_param_arm / write_param_end 原值写回，不改变电机行为）")
        demo_write_same_value(arm, "arm")  # 本体侧：读 → 写同值 → 读回核对的完整安全流程
        demo_write_same_value(arm, "end")  # 末端侧：同一流程对称演示（write_param_end）

        # ---- ⑤ 收尾 ----
        print("=" * 78)
        print("⑤ 收尾")
        print(f"disable_arm 返回：{arm.disable_arm()}；disable_end 返回：{arm.disable_end()}")  # 主动失能两端电机
    print("例程完成：机械臂已失能，with 退出自动断开总线。")


if __name__ == "__main__":  # 直接运行本文件时执行 main（被 import 时不执行）
    try:
        main()                 # 主流程：连接 → 标零/参数演示 → 失能收尾
    except KeyboardInterrupt:  # Ctrl+C：with 已在异常退出时自动安全收尾（失能+断连）
        print("\n检测到 Ctrl+C：已安全退出（电机已失能、总线已断开）。")
