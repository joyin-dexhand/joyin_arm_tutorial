"""
==============================================================================
硬件后端接入自检  (joyarm-backend-creator skill)
==============================================================================
【功能概要】
    1. 第一级：核对 --backend 指定的注册名可从 joyarm_core.backend 的 REGISTRY
       取到、对应类是 Backend 子类，并做空 cfg 与型号 yaml 两种离线构造自检；
    2. 第二级：用 importlib 按路径加载本 skill assets/ 下的 backend_template.py，核对
       29 个抽象方法桩齐全、逐桩调用按约定抛 NotImplementedError、
       核对表与 Backend.__abstractmethods__ 集合完全对齐；
    3. 全程离线（构造不做任何总线 I/O），两级各输出 PASS/FAIL 表并给出总结。

【原理简介】
    JoyArm 上层只面向 Backend 抽象基类编程，每个机械臂型号对应一个子类文件
    backend_<型号>.py，在 joyarm_core/backend/__init__.py 的 REGISTRY 注册表
    登记后，由型号 yaml 的 backend.name 键选型。Python 的 abc 机制会强制子类
    把基类 29 个 @abstractmethod 内核全部「定义」后才允许实例化（缺一个即
    TypeError）；backend_template.py 把 29 个方法全部定义成 NotImplementedError
    桩——既满足抽象方法强制，又让未实现的能力经基类转 warn 优雅降级。本脚本
    不接硬件：第一级验证「注册 + 继承 + 离线构造」链路，第二级验证模板桩的
    完备性，是 config → urdf → backend 三步创建流程（joyarm-config-creator →
    joyarm-urdf-creator → joyarm-backend-creator）的离线自检工具。

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code && uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（在 joyarm_code/ 目录下；离线脚本，无需硬件）
    python .agents/skills/joyarm-backend-creator/scripts/backend_check.py
    python .agents/skills/joyarm-backend-creator/scripts/backend_check.py --backend backend_<型号>  # 检查其他注册名

【预期输出】
    第一级 / 第二级各一张「检查项 | 结果 | 证据」表（29 桩按五个分组展开），
    末尾总结统计并给出结论；全 PASS/SKIP 退出码 0，任一 FAIL 退出码 1。
==============================================================================
"""
from __future__ import annotations

import argparse                        # 命令行参数解析（--backend）
import importlib.util                  # 按文件路径加载 backend_template.py
import sys                             # sys.path 注入与进程退出码
import unicodedata                     # 判定全角字符宽度（表格列对齐用）
from pathlib import Path               # 路径定位（模板文件 / 型号 yaml）

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # joyarm_code 入 path（scripts→backend-creator→skills→.agents→joyarm_code 共 4 级），未安装 joyarm_core 也可运行

import yaml                            # 解析型号 yaml 的 backend 段（path 注入后导入）

from joyarm_core.backend import REGISTRY, Backend  # # 后端注册表与抽象基类
from joyarm_core.utils.types import ControlMode    # # 控制模式枚举（设模式桩的示例入参用）

# ---- 路径常量（第二级检查对象与型号配置目录） ----
TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "assets" / "backend_template.py"   # Backend 子类实现模板（本 skill 的 assets/ 下）
JOYARM_CODE = Path(__file__).resolve().parents[4]                 # 代码库根 joyarm_code（scripts→backend-creator→skills→.agents→joyarm_code 共 4 级）
CONFIG_DIR = JOYARM_CODE / "joyarm_core" / "config"               # 型号 yaml 配置目录

# ---- 结果三态标记（与 backend_verify.py 的能力三分法一致） ----
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

# 29 个抽象内核的示例入参（与 backend.py 抽象方法签名一一对应）
KERNEL_ARGS = {
    "_connect_arm": ("/dev/null", "demo"), "_connect_end": ("/dev/null", "demo"),          # 连接：通道 + 协议
    "_disconnect_arm": (), "_disconnect_end": (),                                          # 断连：无参
    "_enable_joint_arm": (0,), "_enable_joint_end": (0,),                                  # 使能：关节下标
    "_disable_joint_arm": (0,), "_disable_joint_end": (0,),                                # 失能：关节下标
    "_read_joint_param_arm": (0, "pos_kp"), "_read_joint_param_end": (0, "pos_kp"),        # 读参数：下标 + 参数键
    "_read_joint_mode_arm": (0,), "_read_joint_mode_end": (0,),                            # 读模式：关节下标
    "_read_joint_state_arm": (0,), "_read_joint_state_end": (0,),                          # 读状态：关节下标
    "_write_joint_param_arm": (0, "pos_kp", 1.0), "_write_joint_param_end": (0, "pos_kp", 1.0),   # 写参数：下标 + 键 + 值
    "_set_joint_mode_arm": (0, ControlMode.MIT), "_set_joint_mode_end": (0, ControlMode.MIT),   # 设模式：下标 + 枚举
    "_set_joint_zero_arm": (0,), "_set_joint_zero_end": (0,),                              # 设零：关节下标
    "_send_joint_mit_arm": (0, 0.0, 0.0, 0.0, 0.0, 0.0), "_send_joint_mit_end": (0, 0.0, 0.0, 0.0, 0.0, 0.0),   # MIT 五元组
    "_send_joint_position_arm": (0, 0.0, 0.0, 0.0), "_send_joint_position_end": (0, 0.0, 0.0, 0.0),   # 位置：q + vlim/flim
    "_send_joint_velocity_arm": (0, 0.0), "_send_joint_velocity_end": (0, 0.0),                      # 速度：下标 + dq
    "_send_action_end": ("open",),                                                          # 离散动作：动作名
    "_clear_joint_error_arm": (0,), "_clear_joint_error_end": (0,),                         # 清错：关节下标
}

# 29 个内核按模板五部分分组（展示顺序 = 模板文件的分区顺序）
KERNEL_GROUPS = (
    ("第一部分：生命周期——连接与使能（8 个）",                     # 分组一标题
     ("_connect_arm", "_connect_end", "_disconnect_arm", "_disconnect_end",   # 连接与断连 4 桩
      "_enable_joint_arm", "_enable_joint_end", "_disable_joint_arm", "_disable_joint_end")),   # 使能与失能 4 桩
    ("第二部分：读取（6 个）",                                     # 分组二标题
     ("_read_joint_param_arm", "_read_joint_param_end", "_read_joint_mode_arm", "_read_joint_mode_end",   # 参数与模式读 4 桩
      "_read_joint_state_arm", "_read_joint_state_end")),         # 状态读 2 桩
    ("第三部分：写入（6 个）",                                     # 分组三标题
     ("_write_joint_param_arm", "_write_joint_param_end", "_set_joint_mode_arm", "_set_joint_mode_end",   # 写参数与设模式 4 桩
      "_set_joint_zero_arm", "_set_joint_zero_end")),             # 设零 2 桩
    ("第四部分：发送（7 个）",                                     # 分组四标题
     ("_send_joint_mit_arm", "_send_joint_mit_end", "_send_joint_position_arm", "_send_joint_position_end",   # MIT 与位置 4 桩
      "_send_joint_velocity_arm", "_send_joint_velocity_end", "_send_action_end")),   # 速度与离散动作 3 桩
    ("第五部分：清错（2 个）",                                     # 分组五标题
     ("_clear_joint_error_arm", "_clear_joint_error_end")),       # 清错 2 桩
)

# 空 cfg 构造时桩消息的回退前缀（backend_template.py 桩约定：『backend_<型号>.py - 方法名：功能缺失』）
STUB_PREFIX_FILE = "backend_<型号>.py"


def disp_w(text: str) -> int:
    """计算字符串的终端显示宽度（全角/宽字符计 2 列，其余 1 列）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)  # 逐字符累计列宽


def pad(text: str, width: int) -> str:
    """按显示宽度右补空格到 width 列（中英文混排列对齐的关键）。"""
    return text + " " * max(0, width - disp_w(text))  # 负余量时补零格


def fmt(value) -> str:
    """表格证据列格式化：None 显「—」，其余转字符串。"""
    return "—" if value is None else str(value)  # 任务书约定：None 一律显「—」


def print_rows(rows: list) -> None:
    """打印一张「检查项 | 结果 | 证据」三列对齐表（列宽自适应最长检查项）。"""
    if not rows:                                             # 空表（前序检查失败被跳过）
        print("（本表无检查项——前序失败已跳过）")
        return                                               # 直接返回不再打印表头
    w_item = max(disp_w(item) for item, _, _ in rows)        # 检查项列宽 = 最长项显示宽度
    print(f"{pad('检查项', w_item)}  结果  证据")             # 表头（结果列固定 4 字符宽）
    for item, result, evidence in rows:                      # 逐行输出，证据列 None 显「—」
        print(f"{pad(item, w_item)}  {result:<4}  {fmt(evidence)}")


def check_registered(name: str) -> list:
    """第一级检查：--backend 指定的注册名——可取到、是 Backend 子类、两种离线构造自检。

    构造自检：空 cfg `{}` 与型号 yaml 的 `backend:` 段均可离线构造（基类构造只解析配置、不做任何总线 I/O）。

    :param name: backend 注册名（REGISTRY 键，如 "backend_rebot_arm_b601_dm"）。
    :return: (检查项, 结果, 证据) 三元组列表。
    """
    rows = []                                                # 本级结果行累积器
    # ---- ① 注册名核验（导入即完成：backend/__init__.py 顶部已 import 全部注册类） ----
    if name not in REGISTRY:                                 # 注册名不在表内 = 不可用
        rows.append((f"注册名『{name}』在 REGISTRY 中", FAIL,
                     f"未注册；可用注册名：{', '.join(sorted(REGISTRY))}"))
        return rows                                          # 后续检查失去前提，提前返回
    rows.append((f"注册名『{name}』在 REGISTRY 中", PASS,
                 f"可用注册名：{', '.join(sorted(REGISTRY))}"))
    # ---- ② 继承关系核验（上层只面向 Backend 基类编程，非子类无法被框架使用） ----
    cls = REGISTRY[name]                                     # 取出已导入的后端类
    ok_sub = issubclass(cls, Backend)                        # 继承关系判定
    rows.append(("该类是 Backend 的子类", PASS if ok_sub else FAIL,
                 f"{cls.__name__} → Backend" if ok_sub else f"{cls.__name__} 未继承 Backend"))
    if not ok_sub:                                           # 非子类时构造自检无意义
        return rows                                          # 提前返回
    # ---- ③ 离线构造自检一：空 cfg（基类对缺段/缺键全部容错，构造不做任何总线 I/O） ----
    try:
        inst = cls({})                                       # 空 cfg 构造（无硬件依赖）
        try:
            rows.append(("空 cfg {} 构造（无总线 I/O）", PASS,   # 空 cfg 构造通过行
                         f"n_joints: arm={inst.n_joints_arm}, end={inst.n_joints_end}"))   # 证据：两侧关节数
        finally:
            inst.close()                                     # 停低频刷新线程收尾（不可重启，本实例即弃）
    except Exception as exc:                                 # 任何异常都记 FAIL 并保留原因
        rows.append(("空 cfg {} 构造（无总线 I/O）", FAIL, repr(exc)))   # 空 cfg 构造失败行
    # ---- ④ 离线构造自检二：型号 yaml 的 backend 段（存在才检查；未建 yaml 只 SKIP 不判负） ----
    model = name.removeprefix("backend_")                    # 注册名去前缀 = 型号名（config/<型号>.yaml）
    cfg_path = CONFIG_DIR / f"{model}.yaml"                  # 型号 yaml 路径
    if not cfg_path.is_file():                               # yaml 尚未创建（joyarm-config-creator 步未完成）
        rows.append((f"型号 yaml『{cfg_path.name}』backend 段构造", SKIP,   # SKIP 行：前置未就绪
                     "未找到型号 yaml（先完成 joyarm-config-creator）"))
        return rows                                          # 无 yaml 可测，提前返回
    try:
        with open(cfg_path, encoding="utf-8") as f:          # 读取型号 yaml
            section = yaml.safe_load(f)["backend"]           # 只取 backend 段（缺段由 KeyError 记 FAIL）
        inst = cls(section)                                  # 用真实 backend 段离线构造（仅解析配置）
        try:
            ok_name = inst.name == name                      # yaml backend.name 应与注册名一致（选型链路闭环）
            rows.append((f"型号 yaml backend 段构造（name 应=『{name}』）",   # yaml 段构造行
                         PASS if ok_name else FAIL,
                         f"name={inst.name or '—'}, n_joints: arm={inst.n_joints_arm}, end={inst.n_joints_end}"))   # 证据
        finally:
            inst.close()                                     # 停刷新线程收尾
    except Exception as exc:                                 # yaml 缺段 / 构造异常均记 FAIL
        rows.append((f"型号 yaml『{cfg_path.name}』backend 段构造", FAIL, repr(exc)))   # yaml 段构造失败行
    return rows                                              # 返回本级全部结果行


def check_template() -> tuple:
    """第二级检查：backend_template.py 按路径加载——29 桩齐全、桩消息格式、集合对齐。

    先加载模板并实例化（abc 强制 29 个抽象方法全部被定义），再逐桩调用核对 NotImplementedError 消息格式，
    最后与本脚本的核对表做集合对齐（防基类内核增删/改名后模板与核对表漂移）。

    :return: (概要检查行列表, [(分组标题, 分组行列表), ...]) 二元组。
    """
    rows = []                                                # 概要检查行累积器
    # ---- ① 模板文件存在性 ----
    if not TEMPLATE_PATH.is_file():                          # 模板被移动/删除
        return [("模板文件 backend_template.py 存在", FAIL, f"未找到：{TEMPLATE_PATH}")], []  # 概要 FAIL，分组表空
    rows.append(("模板文件 backend_template.py 存在", PASS,
                 str(TEMPLATE_PATH.relative_to(JOYARM_CODE))))  # 显示相对 joyarm_code 的路径
    # ---- ② importlib 按路径加载（模板已迁出 joyarm_core 包，不能按包名导入） ----
    try:
        spec = importlib.util.spec_from_file_location("backend_template_under_check", TEMPLATE_PATH)  # 按路径建模块规格
        mod = importlib.util.module_from_spec(spec)          # 建模块对象（不注册 sys.modules，避免污染）
        spec.loader.exec_module(mod)                         # 执行模块体（触发模板内的 joyarm_core 绝对导入）
        tpl = mod.BackendTemplate                            # 取模板类
    except Exception as exc:                                 # 导入失败（如依赖缺失/语法错误）
        rows.append(("importlib 按路径加载模板", FAIL, repr(exc)))
        return rows, []                                      # 加载失败则后续检查全部跳过
    rows.append(("importlib 按路径加载模板", PASS, f"取得模板类 {tpl.__name__}"))
    # ---- ③ 继承与「29 桩齐全」核验（abc：抽象方法未被全部定义则不可实例化） ----
    ok_sub = issubclass(tpl, Backend)                        # 继承关系判定
    rows.append(("模板类是 Backend 的子类", PASS if ok_sub else FAIL,
                 f"{tpl.__name__} → Backend" if ok_sub else f"{tpl.__name__} 未继承 Backend"))
    missing = sorted(tpl.__abstractmethods__)                # 模板仍未覆盖的抽象方法（空列表 = 桩齐全）
    rows.append((f"29 个抽象方法桩齐全（可实例化）", PASS if not missing else FAIL,
                 f"__abstractmethods__ 剩余 {len(missing)} 个" if missing else "__abstractmethods__ 已清空"))
    # ---- ④ 集合对齐核验（基类增删/改名内核即红，提醒同步模板与本脚本核对表） ----
    base_set = set(Backend.__abstractmethods__)              # 基类抽象方法全集
    ok_align = set(KERNEL_ARGS) == base_set                  # 与本脚本核对表逐名对齐
    diff = set(KERNEL_ARGS) ^ base_set                       # 对称差 = 双方不一致的方法名
    rows.append(("核对表与 Backend.__abstractmethods__ 集合对齐", PASS if ok_align else FAIL,   # 集合对齐行
                 f"双方各 {len(base_set)} / {len(KERNEL_ARGS)} 个，完全一致" if ok_align   # 通过说明
                 else f"不一致项：{sorted(diff)}（须同步模板与本脚本核对表）"))   # 失败说明与指引
    if not (ok_sub and not missing):                         # 继承失败或桩不齐 → 无法实例化，逐桩检查失去前提
        return rows, []                                      # 提前返回（概要已含 FAIL，总结退出码非 0）
    # ---- ⑤ 空 cfg 实例化 + 属性初值核验（构造不做任何总线 I/O） ----
    try:
        t = tpl({})                                          # 空 cfg 构造模板实例
        try:
            ok_init = (t.n_joints_arm == 0 and t.n_joints_end == 0     # 空 cfg 两族关节数为 0
                       and t.is_connected_arm is None and t.is_enabled_end is None)  # 生命周期标志初值未知（None）
            rows.append(("空 cfg 实例化与属性初值", PASS if ok_init else FAIL,   # 实例化与初值行
                         f"n_joints: arm={t.n_joints_arm}, end={t.n_joints_end}, is_connected_arm={fmt(t.is_connected_arm)}"))   # 证据
        finally:
            t.close()                                        # 停刷新线程收尾（逐桩调用不依赖刷新线程）
    except Exception as exc:                                 # 实例化异常（概要记 FAIL）
        rows.append(("空 cfg 实例化与属性初值", FAIL, repr(exc)))
        return rows, []                                      # 无法实例化则逐桩检查失去前提
    # ---- ⑥ 29 桩逐个调用：按约定格式抛 NotImplementedError ----
    stub_groups = []                                         # 分组行累积器（五个分组）
    for title, names in KERNEL_GROUPS:                       # 逐分组展开
        grows = []                                           # 本分组结果行
        for nm in names:                                     # 逐方法核对
            try:
                getattr(t, nm)(*KERNEL_ARGS[nm])             # 直调内核桩（绕过基类公开方法门禁）
                result, ev = FAIL, "未抛 NotImplementedError（桩被误实现？）"  # 桩不应有真实行为
            except NotImplementedError as exc:               # 预期：桩按约定格式上抛
                prefix = f"{STUB_PREFIX_FILE} - {nm}：功能缺失"      # 空 cfg 时桩消息回退前缀
                result, ev = (PASS, "桩消息格式正确") if str(exc).startswith(prefix) \
                    else (FAIL, f"消息前缀异常：{exc}")       # 前缀不符 = 桩约定被破坏
            except Exception as exc:                         # 其他异常 = 桩行为偏离约定
                result, ev = FAIL, repr(exc)
            grows.append((nm, result, ev))                   # 记入本分组行
        stub_groups.append((title, grows))                   # 记入分组表
    # ---- ⑦ 带注册名构造：桩消息前缀应切换为 _name（test 同款核对） ----
    try:
        t2 = tpl({"name": "backend_demo"})                   # 带注册名构造
        try:
            try:
                t2._connect_arm("/dev/null", "demo")         # 调连接桩触发消息
                result, ev = FAIL, "未抛 NotImplementedError"   # 未抛错分支
            except NotImplementedError as exc:               # 预期：前缀切换为注册名
                hit_prefix = str(exc).startswith("backend_demo - _connect_arm：功能缺失")   # 桩消息前缀比对
                # 前缀正确 → PASS；前缀未切换 = 桩约定被破坏 → FAIL
                result = PASS if hit_prefix else FAIL         # 本项判定结果
                ev = "前缀切换为注册名 backend_demo" if hit_prefix else f"消息前缀异常：{exc}"   # 证据文本
            rows.append(("带注册名构造时桩消息前缀切换", result, ev))   # 前缀切换行
        finally:
            t2.close()                                       # 停刷新线程收尾
    except Exception as exc:                                 # 带名构造异常记 FAIL
        rows.append(("带注册名构造时桩消息前缀切换", FAIL, repr(exc)))   # 前缀切换失败行
    return rows, stub_groups                                 # 返回概要行与分组表


def main() -> None:
    """脚本入口：解析参数 → 两级检查 → 表格化输出 → 总结与退出码。"""
    parser = argparse.ArgumentParser(                        # 命令行参数定义（中文帮助）
        description="硬件后端接入离线自检（joyarm-backend-creator）：第一级核对 REGISTRY 注册名可导入/是 Backend 子类/离线构造；"   # 描述一
                    "第二级核对 backend_template.py 的 29 个抽象方法桩")   # 描述二
    parser.add_argument("--backend", default="backend_rebot_arm_b601_dm", metavar="注册名",   # --backend 参数定义
                        help="backend 注册名（REGISTRY 键 = backend_<型号>.py 去扩展名）；缺省 backend_rebot_arm_b601_dm")   # 参数帮助
    args = parser.parse_args()                               # 解析命令行参数
    print("=" * 78)                                          # 大节：脚本标题
    print(f"硬件后端接入自检（joyarm-backend-creator）    目标注册名：{args.backend}")   # 标题行 1：目标注册名
    print(f"模板文件：{TEMPLATE_PATH.relative_to(JOYARM_CODE)}    （离线脚本，无需硬件）")   # 标题行 2：模板路径
    print("=" * 78)                                          # 大节：标题下边框
    # ---- 第一级：已注册后端 ----
    print(f"第一级检查：已注册后端『{args.backend}』——可导入 / 继承 / 离线构造")   # 第一级标题
    rows1 = check_registered(args.backend)                   # 执行第一级检查
    print_rows(rows1)                                        # 打印本级结果表
    # ---- 第二级：模板桩冒烟 ----
    print()                                                  # 空行分隔
    print("第二级检查：backend_template.py 离线桩冒烟（importlib 按路径加载）")   # 第二级标题
    rows2, stub_groups = check_template()                    # 执行第二级检查（概要行 + 分组表）
    print_rows(rows2)                                        # 打印概要结果表
    for title, grows in stub_groups:                         # 逐分组打印 29 桩核对表
        print("-" * 78)                                      # 小节分隔线
        print(f"◇ {title}")
        print_rows(grows)                                    # 打印本分组结果表
    # ---- 总结 ----
    all_rows = rows1 + rows2 + [r for _, g in stub_groups for r in g]  # 汇总两级全部结果行
    n1 = {k: sum(1 for _, r, _ in rows1 if r == k) for k in (PASS, FAIL, SKIP)}     # 第一级三态计数
    n2 = {k: sum(1 for _, r, _ in rows2 + [r for _, g in stub_groups for r in g] if r == k)
          for k in (PASS, FAIL, SKIP)}                       # 第二级三态计数（含逐桩行）
    print("=" * 78)                                          # 大节：总结
    print("总结")                                            # 总结标题
    print("-" * 78)                                          # 小节分隔线
    print(f"第一级（已注册后端）：{n1[PASS]} PASS / {n1[FAIL]} FAIL / {n1[SKIP]} SKIP")   # 第一级统计
    print(f"第二级（模板桩冒烟）：{n2[PASS]} PASS / {n2[FAIL]} FAIL / {n2[SKIP]} SKIP")   # 第二级统计
    if n1[FAIL] + n2[FAIL] == 0:                             # 全 PASS/SKIP 即通过
        print("结论：全部通过——注册链路与模板 29 桩自检均正常（真机验收请运行 backend_verify.py）")
        sys.exit(0)                                          # 退出码 0（供脚本/门控判定）
    print("结论：FAIL——存在未通过项，请按上表证据列逐项排查")
    sys.exit(1)                                              # 任一 FAIL 退出码 1


if __name__ == "__main__":                                   # 直接运行入口（被导入时不执行）
    main()
