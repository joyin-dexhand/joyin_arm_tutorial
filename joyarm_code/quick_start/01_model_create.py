"""
==============================================================================
对象创建与初始化  (快速上手 例程 01)
==============================================================================
【功能概要】
    1. 扫描 joyarm_core/config/*.yaml 列出全部可用型号，
       并打印「可用型号」的三条判断规则；
    2. 交互输入型号创建对象：无效输入重列可用型号与规则后再问、输入 q 退出；也支持
       命令行参数直连：python quick_start/01_model_create.py [型号名]（无效参数打印
       提示与可用列表后以退出码 1 结束）；
    3. 创建 joyArm(型号) 后格式化打印核心信息：__repr__、关节数与关节名、
       pin_model 缩减模型、逐关节硬/软限位表、zero/neutral/home 特征位形、
       tcp_limits 末端限位、plan/ctrl 频率、求解器六域注册与激活、
       get_config() 顶层三段键名。

【原理简介】
    一个型号 = 三件套：config/<型号>.yaml（整机参数）、robot_model/<型号>/（URDF+网格
    模型）、backend_<型号>.py（硬件通信后端）；型号名 name 贯穿三件套：= config 文件名
    = 模型目录名 = URDF 文件名。joyArm(name) 工厂先做四步外部校验（config→backend→
    robot_model→构造），再进入 JoyArm 两阶段初始化：先声明全部成员并赋安全默认值，再按
    依赖顺序（cfg→pin 模型→限位→backend→求解器）赋真值，任一步失败即抛携带原因的异常、
    对象不构造。本例程全程离线：只装配数字模型与算法，不触碰任何硬件。

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

【预期输出】可用型号与三条判断规则 → 选定型号 → 创建并分节打印核心信息；
    收尾时「arm 未连接……豁免跳过」类日志属正常提示，非报错。
==============================================================================
"""

from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装时也可直接运行
import joyarm_core  # noqa: E402 —— 取包安装路径以定位 config/ 与 robot_model/ 两目录
from joyarm_core import joyArm  # noqa: E402 —— 型号创建工厂（四步校验后实例化 JoyArm）
from joyarm_core.backend import REGISTRY  # noqa: E402 —— backend 注册表（判断规则③查它）

# 三件套中两件的磁盘位置（第三件 backend/ 已随包导入，经 REGISTRY 引用）
_CONFIG_DIR = Path(joyarm_core.__file__).resolve().parent / "config"            # 规则① config/<型号>.yaml
_ROBOT_MODEL_DIR = Path(joyarm_core.__file__).resolve().parent / "robot_model"  # 规则② robot_model/<型号>/


# ============================================================
# 型号扫描与三条判断规则
# ============================================================
def model_is_available(name: str) -> bool:
    """按三条规则判断型号是否可用（须同时满足）。

    ① `config/<name>.yaml` 存在且能解析为字典；② `robot_model/<name>/` 目录存在；
    ③ yaml 的 `backend.name` 已在 backend `REGISTRY` 注册。
    """
    yaml_path = _CONFIG_DIR / f"{name}.yaml"          # 规则①对象：型号配置文件
    if not yaml_path.is_file():                       # 文件不存在 → 规则①不过
        return False
    try:
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))  # 解析 yaml 供规则③取键
    except yaml.YAMLError:
        return False                                  # 语法错误 → 无法取 backend.name，判不可用
    if not isinstance(cfg, dict):                     # 顶层须为字典（空文件/列表均不可用）
        return False
    if not (_ROBOT_MODEL_DIR / name).is_dir():        # 规则②：URDF+网格 模型目录
        return False
    backend_cfg = cfg.get("backend")                  # 规则③对象：backend 段
    backend_name = backend_cfg.get("name") if isinstance(backend_cfg, dict) else None
    return backend_name in REGISTRY                   # 注册名须是 REGISTRY 的键


def scan_available_models() -> list:
    """扫描 `config/*.yaml`，返回可用型号名列表。"""
    models = []                                       # 收集通过三条规则的型号名
    for f in sorted(_CONFIG_DIR.glob("*.yaml")):      # 按文件名排序，输出顺序稳定
        if model_is_available(f.stem):                # 三条规则逐条检查
            models.append(f.stem)
    return models


def print_models_and_rules(models: list) -> None:
    """打印可用型号列表与三条判断规则（首次展示与无效输入后重印共用）。"""
    print("-" * 78)                                   # 小节分隔线
    if models:                                        # 有可用型号 → 一行列出
        print(f"可用机械臂型号（共 {len(models)} 个）：{'、'.join(models)}")  # 一行列出全部可用型号
    else:                                             # 无可用型号 → 提示排查方向
        print("可用机械臂型号：无（请检查 joyarm_core/config/ 与 joyarm_core/robot_model/ 是否齐全）")  # 排查提示
    print("可用判断规则（三条同时满足）：")                   # 规则清单标题
    print("  ① config/<型号>.yaml 配置文件存在且能解析为字典")  # 规则①
    print("  ② robot_model/<型号>/ 模型目录（URDF+网格）存在")  # 规则②
    print(f"  ③ yaml 的 backend.name 已在 backend REGISTRY 注册（当前已注册：{sorted(REGISTRY)}）")  # 规则③


# ============================================================
# 型号选择：命令行参数优先，否则交互输入循环
# ============================================================
def ask_model(models: list):
    """交互输入循环请求型号名；返回有效型号名，输入 q 或输入流结束时返回 None。"""
    while True:                                       # 循环直至 有效 / q 退出 / 流结束
        try:
            answer = input("请输入要创建的型号名（q 退出）: ").strip()  # 读一行并去首尾空白
        except EOFError:                              # 管道喂完输入仍未得到有效型号（无 argv 场景兜底）
            print("\n输入流已结束：未创建对象，安全退出。")
            return None
        if answer.lower() == "q":                     # q → 用户选择退出
            print("已选择退出：未创建对象，再见。")
            return None
        if answer in models:                          # 有效型号 → 返回并结束循环
            return answer
        shown = answer if answer else "（空输入）"     # 空输入显「（空输入）」，避免打印空引号
        print(f"『{shown}』不是可用型号，请重新输入。")
        print_models_and_rules(models)                # 无效 → 重印可用型号与规则后再问


def choose_model(models: list):
    """选定型号：有命令行参数则校验参数直连，否则走交互输入；返回型号名或 None。"""
    if len(sys.argv) > 1:                             # 命令行参数直连（无交互环境用）
        name = sys.argv[1].strip()                    # 只取第 1 个参数作型号名
        if name in models:                            # 参数同样走三条规则校验
            return name
        print(f"命令行参数『{name}』不是可用型号（打印可用列表后以退出码 1 结束）。")
        print_models_and_rules(models)                # 无效参数 → 提示 + 可用列表
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


def fmt_num(v, width: int = 8) -> str:
    """数值格式化：None 显「—」、非有限显 ±inf、其余保留 3 位小数并右对齐。"""
    if v is None:                                     # 未配置的限位段 → 显「—」
        return pad("—", width, "right")
    v = float(v)                                      # 统一转 float 再判有限性
    if not np.isfinite(v):                            # ±inf = 该方向不限位，原样显示
        return pad("inf" if v > 0 else "-inf", width, "right")
    return pad(f"{v:.3f}", width, "right")            # 常规数值：3 位小数右对齐


def vec_str(v) -> str:
    """向量转字符串：None 显「—」，ndarray 逐元素保留 3 位小数转 list。"""
    return "—" if v is None else np.round(np.asarray(v, dtype=float), 3).tolist()


# ============================================================
# 创建对象并分节打印核心信息
# ============================================================
def _print_limits_subtable(sub_title: str, heads: list, widths: list, names, spec: tuple) -> None:
    """打印一张限位子表（heads 为列名，spec 为每数值列的 (限位对象, 字段) 取数规则）。"""
    print(f"· {sub_title}")                           # 子表小标题
    print("".join(pad(h, w) for h, w in zip(heads, widths)))   # 表头行（按显示宽度对齐）
    n = len(names) if names else 0                    # 关节数（names 为 None/空则只打印表头）
    for i in range(n):                                # 每个关节一行
        cells = []                                    # 依列序取各限位字段的第 i 个元素
        for lim, key in spec:
            arr = None if lim is None else getattr(lim, key)   # 未配置段（None）取 None
            cells.append(None if arr is None or i >= len(arr) else float(arr[i]))
        row = pad(str(names[i]), widths[0]) + "".join(         # 关节名左对齐 + 数值列右对齐
            fmt_num(c, w) for c, w in zip(cells, widths[1:]))
        print(row)


def print_limits_table(title: str, names, hard, soft) -> None:
    """打印一段（arm 或 end）逐关节硬/软限位：位置限位表 + 速度/力矩限位表。

    hard/soft 为 JointLimits（字段 q_min/q_max/dq_max/tau_max 各 (n,) 数组）或
    None（未配置段对应列显「—」）。
    """
    print(f"—— {title} 逐关节限位（硬=物理极限，软=内缩保护，均只读）——")
    _print_limits_subtable(                           # 子表一：位置限位 q（rad）
        "位置限位 q（rad）",
        ["关节", "硬q_min", "硬q_max", "软q_min", "软q_max"], [11, 9, 9, 9, 9],  # 五列：名称 + 硬/软上下限
        names, ((hard, "q_min"), (hard, "q_max"), (soft, "q_min"), (soft, "q_max")))  # 取数规则：列→限位字段
    _print_limits_subtable(                           # 子表二：速度与力矩限位
        "速度与力矩限位（dq：rad/s，tau：N·m）",
        ["关节", "硬dq_max", "软dq_max", "硬tau_max", "软tau_max"], [11, 9, 9, 10, 10],  # 五列：名称 + 硬/软速度力矩上限
        names, ((hard, "dq_max"), (soft, "dq_max"), (hard, "tau_max"), (soft, "tau_max")))  # 取数规则同上


def print_solver_domains(arm) -> None:
    """打印求解器六域（fkine/ikine/jacobian/dynamics/trajectory/control）的注册与激活。"""
    domains = [                                       # (域全名, 已注册查询, 激活查询)
        ("fkine", arm.get_fkine_solvers, arm.get_activated_fkine_solver),       # 正运动学域
        ("ikine", arm.get_ikine_solvers, arm.get_activated_ikine_solver),       # 逆运动学域
        ("jacobian", arm.get_jac_solvers, arm.get_activated_jac_solver),        # 雅可比域
        ("dynamics", arm.get_dyn_solvers, arm.get_activated_dyn_solver),        # 动力学域
        ("trajectory", arm.get_traj_solvers, arm.get_activated_traj_solver),    # 轨迹规划域
        ("control", arm.get_ctrl_solvers, arm.get_activated_ctrl_solver),       # 控制域
    ]
    for name, get_all, get_act in domains:            # 每域两行：已注册成员 + 当前激活
        print(f"   {pad(name, 11)}已注册：{'、'.join(get_all())}")
        print(f"   {pad('', 11)}当前激活：{get_act()}")


def create_and_show(name: str) -> None:
    """创建 joyArm(name) 并分节打印核心信息（全程离线，不触硬件）。"""
    print("=" * 78)
    print(f"② 创建对象 joyArm(\"{name}\")：四步工厂校验 + 两阶段初始化（离线装配，不触硬件）")
    arm = joyArm(name)                                # config→backend→robot_model→构造，任一步失败抛携带原因的异常
    print(f"print(arm) 展示 __repr__：\n{arm}")       # 一行摘要：型号/关节数/末端帧/激活求解器/FSM 状态

    print("=" * 78)                                   # 大节分隔线
    print("③ 关节数与关节名（cfg 顺序 = q 向量顺序）")  # 大节标题
    print(f"   本体关节数 n_arm = {arm.n_arm}，末端电机数 n_end = {arm.n_end}")  # 两组关节数
    print(f"   本体关节名 get_joint_names_arm()：{arm.get_joint_names_arm()}")   # 本体关节名
    print(f"   末端电机名 get_joint_names_end()：{arm.get_joint_names_end()}")   # 末端电机名

    # pin_model：URDF 解析出的 pinocchio 缩减构型模型（只读共享，nq = n_arm）
    pm = arm.pin_model                                # 缩减模型本体（锁定末端电机关节后的 nq 维构型）
    print(f"   pin_model：nq = {pm.nq}，njoints = {pm.njoints}（含 1 个 universe 根），nframes = {pm.nframes}")  # 模型规模三指标
    print(f"   pin_model 关节名（1 基序号）：{[pm.names[i] for i in range(1, pm.njoints)]}")  # 缩减模型活动关节名
    print("   说明：pin_model 是 URDF 解析的内部上游对象——n_arm/n_end、六域求解器、限位与")  # 上游对象说明一
    print("   特征位形均由它派生；例程通常经派生属性间接消费它，无单独操作场景。")  # 上游对象说明二

    print("=" * 78)                                   # 大节分隔线
    print("④ 逐关节限位（joint_limits_hard/soft：硬限位拦指令，软限位供上层状态判断）")  # 大节标题
    print_limits_table("本体 arm", arm.get_joint_names_arm(),  # 本体限位表（6 关节）
                       arm.joint_limits_hard_arm, arm.joint_limits_soft_arm)     # 本体：6 关节
    print_limits_table("末端 end", arm.get_joint_names_end(),
                       arm.joint_limits_hard_end, arm.joint_limits_soft_end)     # 末端：电机空间

    print("=" * 78)
    print("⑤ 特征位形与末端空间限位")
    print(f"   zero_arm     = {vec_str(arm.zero_arm)}")         # 全零位形（上电标零参考）
    print(f"   neutral_arm  = {vec_str(arm.neutral_arm)}")      # 软限位中点（数值求解默认初值）
    print(f"   home_arm     = {vec_str(arm.home_arm)}")         # cfg 配置的归位位形
    print(f"   zero_end     = {vec_str(arm.zero_end)}")         # 末端全零位形
    print(f"   neutral_end  = {vec_str(arm.neutral_end)}")      # 末端软限位中点
    print(f"   home_end     = {vec_str(arm.home_end)}")         # 末端归位位形
    tcp = arm.tcp_limits                              # TcpLimits：工作空间包围盒 + 速度/力上限
    box = tcp.workspace_box                           # (3,2) 数组：每行 x/y/z 的 [min, max]
    if box is None:                                   # 未配置包围盒 → 显「—」
        print("   tcp_limits.workspace_box (m)：—（未配置）")  # 缺省占位输出
    else:
        print(f"   tcp_limits.workspace_box (m)："     # 包围盒三轴区间（每行 x/y/z 的 [min, max]）
              f"x {vec_str(box[0])}，y {vec_str(box[1])}，z {vec_str(box[2])}")
    print(f"   v_lin_max = {tcp.v_lin_max} m/s，v_ang_max = {tcp.v_ang_max} rad/s，"
          f"f_max = {tcp.f_max} N，t_max = {tcp.t_max} N·m")    # 末端速度/力四上限

    print("=" * 78)                                   # 大节分隔线
    print("⑥ 运行频率与求解器六域（get_<域>_solvers / get_activated_<域>_solver）")  # 大节标题
    print(f"   规划线程频率 plan_hz = {arm.plan_hz}，控制线程频率 ctrl_hz = {arm.ctrl_hz}")  # 起搏频率
    print_solver_domains(arm)                         # 六域逐域打印已注册与当前激活

    print("=" * 78)
    print("⑦ config 配置快照（get_config() 返回深拷贝，顶层三段）")
    cfg = arm.get_config()                            # 深拷贝快照：修改它不影响 arm 内部配置
    print(f"   顶层段名：{list(cfg.keys())}")
    for key, sec in cfg.items():                      # 逐段列一级子键，了解配置全貌
        subs = list(sec.keys()) if isinstance(sec, dict) else "—"
        print(f"   {pad(key + ' 段', 12)}一级子键：{subs}")

    arm.close()                                       # 显式安全收尾（幂等；离线仅停线程并清缓存）
    print("=" * 78)
    print("例程运行完毕（收尾时「arm 未连接……豁免跳过」类日志为正常提示）。")


def main() -> None:
    """主流程：扫描可用型号 → 选定型号（参数优先/交互）→ 创建并打印核心信息。"""
    print("=" * 78)
    print("① 可用机械臂型号（扫描 joyarm_core/config/*.yaml，三条规则过滤）")
    models = scan_available_models()                  # 扫描 + 三条规则过滤
    print_models_and_rules(models)                    # 列出可用型号与判断规则
    name = choose_model(models)                       # 命令行参数优先，否则交互输入
    if name is None:                                  # q 退出 / 输入流结束：不创建对象
        return
    create_and_show(name)                             # 创建 joyArm 并分节打印核心信息


if __name__ == "__main__":
    try:
        main()                                        # 正常入口
    except KeyboardInterrupt:                         # Ctrl+C：离线脚本未连硬件，直接安全退出
        print("\n检测到 Ctrl+C：未连接任何硬件，已安全退出。")
