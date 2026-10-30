"""
==============================================================================
例程 01 —— 对象创建与初始化(全程离线)
==============================================================================
【功能概要】
    1. 调用函数 listArm()（joyarm_core.joyarm 提供）列出全部可用型号（满足“型号有效”判断条件）；
    2. 终端交互式输入型号创建对象：无效则再次请求输入（输入 q 退出程序）；也可直接：python quick_start/01_model_create.py [型号名]；
    3. 成功创建机械臂对象后，格式化并打印该型号机械臂对象的核心信息。

【原理简介】
    有效型号需同时具有三类文件：config/<型号>.yaml（整机参数）、robot_model/<型号>/（URDF+网格模型）和 backend/<注册名>.py（硬件后端）；
    型号名 name 贯穿前两类文件：name = config 文件名 = 模型目录名 = URDF 文件名；backend 与型号一一对应，注册命名因来源而异（第三方取 backend_<型号>，官方取 backend_<版本>，如 joyarm_dm ↔ backend_dm）。
    listArm() 内部对 config/*.yaml 逐型号做三条件过滤（config 非空 dict 且三顶层段齐全、backend.name 已注册、robot_model 路径齐全）。
    joyArm(name) 工厂做四步校验（前三步同listArm，另有④构造自检）后进入初始化。

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 2. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（在 joyarm_code/ 目录下；离线脚本，无需硬件）
    python quick_start/01_model_create.py                      # 交互输入型号
    python quick_start/01_model_create.py rebot_arm_b601_dm    # 命令行参数直连

【预期输出】listArm() 列出的可用型号与判断规则 → 选定型号 → 创建并打印核心信息
    （关节、限位、特征位形、末端限位、频率与求解器）；收尾时「arm 未连接……豁免跳过」
    类日志属正常提示，非报错。
==============================================================================
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装时也可直接运行
from joyarm_core import joyArm, listArm  # noqa: E402 —— 型号创建工厂 + 可用型号列表（joyarm/__init__.py 提供）


def print_rules() -> None:
    """打印"型号有效"的判断规则（listArm 过滤与 joyArm 校验）。"""
    print("判断规则（listArm 内部对 config/*.yaml 逐型号过滤；工厂 joyArm 校验同一口径）：")
    print("  ① config/<型号>.yaml 存在、解析为非空 dict，且 joyarm/robotics/backend 三顶层段齐全（段须为 dict）")
    print("  ② yaml 的 backend.name 已在 backend REGISTRY 注册（一型号一后端，注册名 = backend 文件名去 .py）")
    print("  ③ robot_model/<型号>/ 四路径齐全（目录、meshes/、urdf/、URDF 文件）")
    print("  （工厂创建另做④构造自检 JoyArm(name)；listArm 只按①~③过滤、不构造）")


# ============================================================
# 型号选择：命令行参数优先，否则交互输入循环
# ============================================================
def ask_model(models: list):
    """交互输入循环请求型号名；返回有效型号名，输入 q 或输入流结束时返回 None。"""
    while True:                                       # 循环直至 有效 / q 退出 / 流结束
        try:
            answer = input("请输入要创建的型号名（q 退出）: ").strip()  # 读一行并去首尾空白
        except EOFError:                              # 管道喂完输入仍未得到有效型号
            print("\n输入流已结束：未创建对象，安全退出。")
            return None
        if answer.lower() == "q":                     # q → 用户选择退出
            print("已选择退出：未创建对象，再见。")
            return None
        if answer in models:                          # 有效型号 → 返回并结束循环
            return answer
        shown = answer if answer else "（空输入）"     # 空输入显「（空输入）」，避免打印空引号
        print(f"『{shown}』不是可用型号，重新列出可用型号：")
        models = listArm()                            # 库函数重列（打印最新列表并返回）


def choose_model(models: list):
    """选定型号：有命令行参数则校验参数直连，否则走交互输入；返回型号名或 None。"""
    if len(sys.argv) > 1:                             # 命令行参数直连（无交互环境用）
        name = sys.argv[1].strip()                    # 只取第 1 个参数作型号名
        if name in models:                            # 参数同样须在可用列表内
            return name
        print(f"命令行参数『{name}』不是可用型号（打印可用列表后以退出码 1 结束）。")
        listArm()                                     # 库函数列出可用型号
        sys.exit(1)                                   # 退出码 1 区别于正常退出
    return ask_model(models)                          # 无参数 → 交互输入循环


# ============================================================
# 打印格式化助手（表格列对齐 / None 显「—」）
# ============================================================
def disp_width(s: str) -> int:
    """字符串的终端显示宽度（中文等全角字符计 2，ASCII 计 1）。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)  # 0x2E80 以上大致为 CJK 区，按双宽计


def pad(s: str, width: int, align: str = "left") -> str:
    """按显示宽度补空格，保证中文表头与数值混排时表格列对齐。"""
    gap = " " * max(width - disp_width(s), 0)         # 缺口 = 目标宽 - 显示宽（不为负）
    return gap + s if align == "right" else s + gap   # 数值列 right 右对齐，文本列 left 左对齐


def fmt_num(v, width: int = 9) -> str:
    """数值格式化：None 显「—」、非有限显 ±inf、其余保留 3 位小数并右对齐。"""
    if v is None:                                     # 未配置的限位段 → 显「—」
        return pad("—", width, "right")
    v = float(v)                                      # 统一转 float 再判有限性
    if not np.isfinite(v):                            # ±inf = 该方向不限位
        return pad("inf" if v > 0 else "-inf", width, "right")
    return pad(f"{v:.3f}", width, "right")            # 常规数值：3 位小数右对齐


def vec_str(v) -> str:
    """向量转字符串：None 显「—」，ndarray 逐元素保留 3 位小数转 list。"""
    return "—" if v is None else np.round(np.asarray(v, dtype=float), 3).tolist()


# ============================================================
# 创建对象并分节打印最关键的核心信息
# ============================================================
def _lim_cells(lim, i: int) -> list:
    """取限位对象第 i 关节的 (q_min, q_max, dq_max, tau_max)；lim 为 None 时全取 None。"""
    if lim is None:                                   # 该段未配置限位 → 四列全显「—」
        return [None, None, None, None]
    return [float(a[i]) for a in (lim.q_min, lim.q_max, lim.dq_max, lim.tau_max)]


def print_limits_table(title: str, names, soft) -> None:
    """打印一段（arm 或 end）逐关节软限位一行表（硬限位拦指令，软限位供上层判断）。"""
    print(f"—— {title} 逐关节软限位（硬限位见 config/URDF，此处列上层用的软限位）——")
    heads = ["关节", "q_min", "q_max", "dq_max", "tau_max"]   # 五列：名称 + 四项软限位
    widths = [11, 9, 9, 9, 10]
    print("".join(pad(h, w) for h, w in zip(heads, widths)))  # 表头行（按显示宽度对齐）
    for i in range(len(names) if names else 0):       # 每个关节一行
        cells = _lim_cells(soft, i)                   # 该关节四项软限位
        print(pad(str(names[i]), widths[0]) + "".join(  # 关节名左对齐 + 数值列右对齐
            fmt_num(c, w) for c, w in zip(cells, widths[1:])))


def print_solver_domains(arm) -> None:
    """打印求解器六域的当前激活与注册数（完整注册清单与切换演示见例程 05）。"""
    domains = [                                       # (域全名, 激活查询, 注册清单查询)
        ("fkine", arm.get_activated_fkine_solver, arm.get_fkine_solvers),     # 正运动学域
        ("ikine", arm.get_activated_ikine_solver, arm.get_ikine_solvers),     # 逆运动学域
        ("jacobian", arm.get_activated_jac_solver, arm.get_jac_solvers),      # 雅可比域
        ("dynamics", arm.get_activated_dyn_solver, arm.get_dyn_solvers),      # 动力学域
        ("trajectory", arm.get_activated_traj_solver, arm.get_traj_solvers),  # 轨迹规划域
        ("control", arm.get_activated_ctrl_solver, arm.get_ctrl_solvers),     # 控制域
    ]
    for name, get_act, get_all in domains:            # 每域一行：当前激活 + 注册数
        print(f"   {pad(name, 11)}激活：{get_act()}（已注册 {len(get_all())} 个）")


def create_and_show(name: str) -> None:
    """创建 joyArm(name) 并分节打印最关键的核心信息（全程离线，不触硬件）。"""
    print("=" * 78)
    print(f"② 创建对象 joyArm(\"{name}\")：四步工厂校验 + 两阶段初始化（离线装配）")
    arm = joyArm(name)                                # config→backend→robot_model→构造，任一步失败抛携带原因的异常
    print(f"   print(arm) 展示 __repr__：{arm}")      # 一行摘要：型号/关节数/末端帧/激活求解器/FSM 状态

    print("=" * 78)                                   # 大节分隔线
    print("③ 关节数与关节名（cfg 顺序 = q 向量顺序）")
    print(f"   本体关节数 n_arm = {arm.n_arm}，末端电机数 n_end = {arm.n_end}")
    print(f"   本体关节名：{arm.get_joint_names_arm()}")
    print(f"   末端电机名：{arm.get_joint_names_end()}")

    print("=" * 78)
    print("④ 逐关节软限位（joint_limits_soft；越软限位不拦指令、供上层状态判断）")
    print_limits_table("本体 arm", arm.get_joint_names_arm(), arm.joint_limits_soft_arm)
    print_limits_table("末端 end", arm.get_joint_names_end(), arm.joint_limits_soft_end)

    print("=" * 78)
    print("⑤ 特征位形与末端空间限位")
    print(f"   zero_arm     = {vec_str(arm.zero_arm)}")       # 全零位形（上电标零参考）
    print(f"   neutral_arm  = {vec_str(arm.neutral_arm)}")    # 软限位中点（数值求解默认初值）
    print(f"   home_arm     = {vec_str(arm.home_arm)}")       # cfg 配置的归位位形
    print(f"   zero_end = {vec_str(arm.zero_end)}，home_end = {vec_str(arm.home_end)}")  # 末端两特征位形
    tcp = arm.tcp_limits                              # TcpLimits：工作空间包围盒 + 速度/力上限
    if tcp.workspace_box is None:                     # 未配置包围盒 → 显「—」
        print("   tcp 工作空间包围盒 (m)：—（未配置）")
    else:                                             # 已配置 → 三轴区间一行
        b = tcp.workspace_box                         # (3,2) 数组：每行 x/y/z 的 [min, max]
        print(f"   tcp 工作空间包围盒 (m)：x {vec_str(b[0])}，y {vec_str(b[1])}，z {vec_str(b[2])}")
    print(f"   tcp 速度/力上限：v_lin {tcp.v_lin_max} m/s，v_ang {tcp.v_ang_max} rad/s，"
          f"f {tcp.f_max} N，t {tcp.t_max} N·m")

    print("=" * 78)
    print("⑥ 运行频率与求解器六域（get_activated_<域>_solver / get_<域>_solvers）")
    print(f"   规划线程频率 plan_hz = {arm.plan_hz}，控制线程频率 ctrl_hz = {arm.ctrl_hz}")
    print_solver_domains(arm)

    print("=" * 78)
    print("⑦ config 配置快照（get_config() 返回深拷贝，顶层三段）")
    print(f"   顶层段名：{list(arm.get_config().keys())}")

    arm.close()                                       # 显式安全收尾（幂等；离线仅停线程并清缓存）
    print("=" * 78)
    print("例程运行完毕（收尾时「arm 未连接……豁免跳过」类日志为正常提示）。")


def main() -> None:
    """主流程：listArm() 列出可用型号 → 选定型号（参数优先/交互）→ 创建并打印核心信息。"""
    print("=" * 78)
    print("① 可用机械臂型号（调用库函数 listArm()：内部扫描 config/*.yaml 三条件过滤）")
    models = listArm()                                # 库函数：打印可用型号列表并返回
    print_rules()                                     # 补充说明 listArm 的三条件判断规则
    name = choose_model(models)                       # 命令行参数优先，否则交互输入
    if name is None:                                  # q 退出 / 输入流结束：不创建对象
        return
    create_and_show(name)                             # 创建 joyArm 并分节打印核心信息


if __name__ == "__main__":
    try:
        main()                                        # 正常入口
    except KeyboardInterrupt:                         # Ctrl+C：离线脚本未连硬件，直接安全退出
        print("\n检测到 Ctrl+C：未连接任何硬件，已安全退出。")
