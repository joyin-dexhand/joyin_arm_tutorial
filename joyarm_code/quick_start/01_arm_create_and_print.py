"""
==============================================================================
例程 01 —— 对象创建与信息打印（全程离线）
==============================================================================
【功能概要】
    1. 调用 listArm() 列出全部可用型号（满足「型号有效」判别条件）；
    2. 选定型号并创建对象：命令行参数直接指定（无效则列出可用型号后退出），或不带参数进入交互输入（q 退出）；
    3. 创建成功后分节打印核心信息：对象摘要、关节、硬/软限位、特征位形、运行频率与求解器六域。

【原理简介】
    有效型号判断需同时具有三类文件：config/<型号>.yaml（整机参数）、robot_model/<型号>/（URDF+网格模型）和 backend/<注册名>.py（硬件后端）；
    型号名 name 与前两类文件需一致（name = config 文件名 = 模型目录名 = URDF 文件名）；
    backend 与型号一一对应，注册命名因来源而异（第三方取 backend_<型号>，官方取 backend_<版本>，其中 <官方型号> = joyarm_<版本>）。

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（全程离线，无需硬件）
    python quick_start/01_arm_create_and_print.py <型号>    # 如 rebot_arm_b601_dm
    python quick_start/01_arm_create_and_print.py          # 不带参数则交互输入型号

【预期输出】
    可用型号列表与判断规则 → 选定型号 → 分节打印：对象摘要表、关节数与关节名、硬/软限位四表、特征位形与 tcp 限位、运行频率与六域激活求解器。
    收尾时「arm 未连接……豁免跳过」类日志属正常提示，非报错。
==============================================================================
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 注入上级目录到搜索路径
from joyarm_core import joyArm, listArm         # 对象创建入口 + 可用型号列表


# ============================================================
# 一、型号选择：命令行参数优先，否则交互输入
# ============================================================
def ask_model(models: list):
    """循环请求输入型号名，输错重列可用型号，q 退出；返回型号名或 None。"""
    while True:                                   # 直至：有效型号 / q 退出 / 输入流结束
        try:
            answer = input("\n 请输入要创建的型号名（q 退出）: ").strip()  # 读一行，去首尾空白
        except EOFError:                          # 输入流结束仍未得到有效型号
            print("\n输入流已结束：未创建对象，安全退出。")
            return None
        if answer.lower() == "q":                 # q → 主动退出
            print("已选择退出：未创建对象，再见。")
            return None
        if answer in models:                      # 有效型号 → 返回并结束循环
            return answer
        shown = answer if answer else "（空输入）"  # 空输入显占位符，避免打印空引号
        print(f"『{shown}』不是可用型号，重新列出可用型号：")
        models = listArm()                        # 重列可用型号


def choose_model(models: list):
    """选定型号：命令行参数有效则直用，否则交互输入；返回型号名或 None。"""
    if len(sys.argv) > 1:                         # 有命令行参数：校验后直接使用
        name = sys.argv[1].strip()                # 只取第 1 个参数作型号名
        if name in models:                        # 参数同样必须在可用列表内
            return name
        print(f"命令行参数『{name}』不是可用型号，可用列表如下：")
        listArm()                                 # 列出可用型号供对照
        sys.exit(1)                               # 退出码 1，区别于正常退出
    return ask_model(models)                      # 无参数 → 交互输入循环


# ============================================================
# 二、打印格式化助手：表格对齐与数值显示
# ============================================================
def disp_width(s: str) -> int:
    """字符串的终端显示宽度：中文等全角字符计 2，ASCII 计 1。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s: str, width: int, align: str = "left") -> str:
    """按显示宽度补空格，使中文表头与数值混排的表格列对齐。"""
    gap = " " * max(width - disp_width(s), 0)     # 缺口 = 目标宽 − 显示宽（不为负）
    return gap + s if align == "right" else s + gap


def fmt_num(v, width: int = 9) -> str:
    """数值定宽右对齐：None 显「—」，非有限显 inf/-inf，其余保留 3 位小数。"""
    if v is None:                                 # 该段未配置限位 → 占位符
        return pad("—", width, "right")
    v = float(v)                                  # 统一转 float 再判有限性
    if not np.isfinite(v):                        # ±inf = 该方向不限位
        return pad("inf" if v > 0 else "-inf", width, "right")
    return pad(f"{v:.3f}", width, "right")


def vec_str(v) -> str:
    """向量转可读字符串：None 显「—」，其余逐元素保留 3 位小数。"""
    return "—" if v is None else np.round(np.asarray(v, dtype=float), 3).tolist()


# ============================================================
# 三、核心信息分节打印
# ============================================================
def _lim_cells(lim, i: int) -> list:
    """取限位对象第 i 关节的 (q_min, q_max, dq_max, tau_max)；未配置时全为 None。"""
    if lim is None:
        return [None, None, None, None]
    return [float(a[i]) for a in (lim.q_min, lim.q_max, lim.dq_max, lim.tau_max)]


def print_limits_table(title: str, names, limits, kind: str) -> None:
    """打印一段（arm 或 end）逐关节限位表，kind 传「硬」或「软」。"""
    note = ("取自 cfg backend 段各关节条目，指令下发前据此裁剪"
            if kind == "硬" else
            "取自 cfg joyarm.joint_limits_soft 段，越限不拦指令、供上层判断")
    print(f"—— {title} 逐关节{kind}限位（{note}）——")
    heads = ["关节", "q_min", "q_max", "dq_max", "tau_max"]   # 五列：名称 + 四项限位
    widths = [11, 11, 12, 12, 12]
    print(pad(heads[0], widths[0]) + "".join(pad(h, w, "right") for h, w in zip(heads[1:], widths[1:])))  # 表头行
    for i in range(len(names) if names else 0):   # 每个关节一行
        cells = _lim_cells(limits, i)
        print(pad(str(names[i]), widths[0]) + "".join(fmt_num(c, w) for c, w in zip(cells, widths[1:])))  # 名称左对齐，数值右对齐


def print_solver_domains(arm) -> None:
    """打印求解器六域各自的激活求解器与注册数（切换演示见例程 05）。"""
    domains = [                                   # (域名, 激活查询, 注册清单查询)
        ("fkine", arm.get_activated_fkine_solver, arm.get_fkine_solvers),
        ("ikine", arm.get_activated_ikine_solver, arm.get_ikine_solvers),
        ("jacobian", arm.get_activated_jac_solver, arm.get_jac_solvers),
        ("dynamics", arm.get_activated_dyn_solver, arm.get_dyn_solvers),
        ("trajectory", arm.get_activated_traj_solver, arm.get_traj_solvers),
        ("control", arm.get_activated_ctrl_solver, arm.get_ctrl_solvers),
    ]
    for name, get_act, get_all in domains:        # 每域一行：当前激活 + 注册数
        print(f"   {pad(name, 13)}激活：{get_act()}（已注册 {len(get_all())} 个）")


def create_and_show(name: str) -> None:
    """创建 joyArm(name)，按 ②~⑥ 分节打印核心配置信息。"""
    print("=" * 78)
    print(f"② 创建对象 joyArm(\"{name}\")：自检校验")
    arm = joyArm(name)                            # 任一步校验失败抛带原因的异常
    print(arm)                                    # __str__：多行表格摘要

    print("=" * 78)
    print("③ 关节数与关节名")
    print(f"   本体关节数 n_arm = {arm.n_arm}，末端电机数 n_end = {arm.n_end}")
    print(f"   本体关节名：{arm.get_joint_names_arm()}")
    print(f"   末端电机名：{arm.get_joint_names_end()}")

    print("=" * 78)
    print("④ 逐关节硬限位与软限位（joint_limits_hard / joint_limits_soft）")
    print_limits_table("本体 arm", arm.get_joint_names_arm(), arm.joint_limits_hard_arm, "硬")
    print_limits_table("本体 arm", arm.get_joint_names_arm(), arm.joint_limits_soft_arm, "软")
    print_limits_table("末端 end", arm.get_joint_names_end(), arm.joint_limits_hard_end, "硬")
    print_limits_table("末端 end", arm.get_joint_names_end(), arm.joint_limits_soft_end, "软")

    print("=" * 78)
    print("⑤ 特征位形与 tcp 限位")
    print(f"   neutral_arm  = {vec_str(arm.neutral_arm)}")   # 软限位中点（数值ik默认初值）
    print(f"   home_arm     = {vec_str(arm.home_arm)}")      # cfg 配置的归位位形
    print(f"   neutral_end  = {vec_str(arm.neutral_end)}")   # 末端软限位中点
    print(f"   home_end     = {vec_str(arm.home_end)}")      # 末端 cfg 配置的归位位形
    tcp = arm.tcp_limits                          # 工作空间包围盒 + 速度/力上限
    if tcp.workspace_box is None:                 # 未配置包围盒 → 显示「—」
        print("   tcp 工作空间包围盒 (m)：—（未配置）")
    else:                                         # 已配置 → 三轴区间一行
        b = tcp.workspace_box                     # (3,2)：每行 x/y/z 的 [min, max]
        print(f"   tcp 工作空间包围盒 (m)：x {vec_str(b[0])}，y {vec_str(b[1])}，z {vec_str(b[2])}")
    print(f"   tcp 速度/力上限：v_lin {tcp.v_lin_max} m/s，v_ang {tcp.v_ang_max} rad/s，f {tcp.f_max} N，t {tcp.t_max} N·m")

    print("=" * 78)
    print(f"⑥ 运行频率与求解器六域：plan_hz = {arm.plan_hz}，ctrl_hz = {arm.ctrl_hz}")
    print_solver_domains(arm)

    arm.close()                                   # 显式安全收尾（离线仅停线程并清缓存）
    print("=" * 78)
    print("例程运行完毕（收尾时「arm 未连接……豁免跳过」类日志为正常提示）。")


def main() -> None:
    """主流程：列出可用型号 → 选定型号 → 创建对象并分节打印核心信息。"""
    print("=" * 78)
    print("① 可用机械臂型号（库函数 listArm() 扫描 config/ 并按三条件过滤）")
    models = listArm()                            # 打印可用型号列表与判断规则并返回
    name = choose_model(models)                   # 命令行参数优先，否则交互输入
    if name is None:                              # q 退出 / 输入流结束：不创建对象
        return
    create_and_show(name)


if __name__ == "__main__":
    try:
        main()                                    # 正常入口
    except KeyboardInterrupt:                     # Ctrl+C：离线脚本未连硬件，直接退出
        print("\n检测到 Ctrl+C：已安全退出。")
