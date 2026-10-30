"""
==============================================================================
型号配置文件完整性检查  (joyarm-config-creator skill 例程)
==============================================================================
【功能概要】
    1. 基础校验 config/<型号>.yaml：文件存在、YAML 可解析为非空字典、顶层
       joyarm / robotics / backend 三段齐全且为字典；
    2. 逐条校验硬性约束：joyarm.frame_end 已配置、软限位在硬限位内、zero/home
       位形不越限、robotics 六域注册名在册、backend.name 在册、关节名与 URDF
       驱动关节一致、backend 四键限位与 URDF <limit> 一致；
    3. 终极校验：joyArm(<型号>) 离线构造（触发库内全套自检，全程不连接硬件）；
    4. 输出 PASS/FAIL 对齐表格与总结：全过打印 SUCCESS（退出码 0），任一 FAIL
       以退出码 1 结束。

【原理简介】
    型号配置文件是 JoyArm 的"装配图纸"：构造 JoyArm(<型号>) 时，库读取
    config/<型号>.yaml，用 joyarm 段装配整机参数、robotics 段按注册名实例化
    六域求解器、backend 段实例化硬件通信后端，并与 robot_model/<型号>/urdf/
    下的 URDF 做关节名与限位的交叉核对，任一项不符即抛异常
    （joyarm_core/joyarm/joyarm.py 的 _check_cfg 等初始化自检）。
    本脚本把这套校验拆成逐项 PASS/FAIL 表：先做单项检查（问题定位更精准），
    最后再做一次 joyArm 离线构造作为终极校验——构造通过即代表该配置能真正
    装配成整机对象。全程离线，不需要连接机械臂。

【环境与运行】
    ⓪ 装 uv：curl -LsSf https://astral.sh/uv/install.sh | sh
      （Windows 用 irm https://astral.sh/uv/install.ps1 | iex）
    ① cd <仓库路径>/joyarm_code && uv sync && source .venv/bin/activate
    ② python .agents/skills/joyarm-config-creator/scripts/config_check.py --model rebot_arm_b601_dm
      （离线脚本，无需硬件；--model 缺省 rebot_arm_b601_dm）

【预期输出】
    ①~⑤ 五节信息：目标定位 → 基础校验表 → 硬性约束表 → 终极构造 → 总结；
    全过时末行打印「SUCCESS：…」，任一 FAIL 时打印「FAILED」并以退出码 1 结束。
==============================================================================
"""

import argparse                                                  # 命令行参数解析（--model）
import re                                                        # URDF 驱动关节命名约定的正则匹配
import sys                                                       # sys.path 注入与退出码返回
import unicodedata                                               # 全宽字符宽度判定（中文表格对齐）
import xml.etree.ElementTree as ET                               # 解析 URDF 的 <joint>/<limit> 元素
from pathlib import Path                                         # 配置与 URDF 文件路径定位

import numpy as np                                               # 限位数组的维度与区间比较
import yaml                                                      # 型号配置文件解析

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))    # 未安装 joyarm_core 时也可直接运行（指向 joyarm_code/：scripts→joyarm-config-creator→skills→.agents 共 4 级）
from joyarm_core import (                                        # 核心库公开接口
    joyArm,                                                      # 工厂入口：终极离线构造校验
    listArm,                                                     # 列出当前可用型号（① 节展示）
    limits_from_joint_cfgs,                                      # backend joints 四键 → 硬限位
    soft_limits_from_cfg,                                        # joyarm 软限位四键 → JointLimits
)
from joyarm_core.backend import REGISTRY as _BACKEND_REGISTRY    # backend 注册表
from joyarm_core.robotics.fkine import REGISTRY as _FKINE_REGISTRY       # fkine 域注册表
from joyarm_core.robotics.ikine import REGISTRY as _IKINE_REGISTRY       # ikine 域注册表
from joyarm_core.robotics.jacobian import REGISTRY as _JAC_REGISTRY      # jacobian 域注册表
from joyarm_core.robotics.dynamics import REGISTRY as _DYN_REGISTRY      # dynamics 域注册表
from joyarm_core.robotics.trajectory import REGISTRY as _TRAJ_REGISTRY   # trajectory 域注册表
from joyarm_core.robotics.control import REGISTRY as _CTRL_REGISTRY      # control 域注册表

_CODE_ROOT = Path(__file__).resolve().parents[4]                 # joyarm_code/ 根（config、robot_model 的上级；本脚本再下四级）
_URDF_LIMIT_TOL = 1e-4                                           # cfg↔URDF 限位核对容差（与库 joyarm.py 同值）
_ACTIVE_JOINT_TYPES = ("revolute", "continuous", "prismatic")    # URDF 活动关节类型（fixed 不可登记，库同口径）
_DOMAINS = [                                                     # robotics 六域：cfg 段键 → 对应 REGISTRY
    ("fkine", _FKINE_REGISTRY),                                  # 正运动学域
    ("ikine", _IKINE_REGISTRY),                                  # 逆运动学域
    ("jacobian", _JAC_REGISTRY),                                 # 雅可比域
    ("dynamics", _DYN_REGISTRY),                                 # 动力学域
    ("trajectory", _TRAJ_REGISTRY),                              # 轨迹规划域
    ("control", _CTRL_REGISTRY),                                 # 控制域
]
_COL_NO, _COL_ITEM, _COL_RES = 5, 34, 6                          # 表格前三列显示宽度（编号 / 校验项 / 结果）


# ============================================================
# 终端输出辅助（中文全宽字符按 2 列计，保证表格列对齐）
# ============================================================
def _disp_width(text: str) -> int:
    """计算字符串的终端显示宽度（东亚全宽字符记 2，其余记 1）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "FW" else 1 for ch in text)  # 逐字符累加显示宽度


def _pad(text: str, width: int) -> str:
    """按显示宽度在右侧补空格至 width（表格列对齐用）。"""
    return text + " " * max(0, width - _disp_width(text))        # 不足宽度补空格，超宽原样返回


def _wrap(text: str, width: int, indent: str = "") -> str:
    """按显示宽度折行：空格分隔的词尽量不拆行、超宽词按字符硬折；首行顶格、续行前缀 indent。"""
    def _hard_break(seg: str) -> list:                           # 超宽词按字符硬折成多段
        parts, cur = [], ""                                      # 硬折结果与当前段
        for ch in seg:                                           # 逐字符装箱（全宽字符不拆半）
            if cur and _disp_width(cur) + _disp_width(ch) > width:   # 装不下且当前段非空 → 收段
                parts.append(cur)                                # 收尾当前段
                cur = ""                                         # 新段起步
            cur += ch                                            # 装入本字符
        parts.append(cur)                                        # 末段收尾
        return parts                                             # 返回硬折段列表

    out: list = []                                               # 已完成的行（仅内容，不含缩进）
    line = ""                                                    # 当前行内容
    for word in text.split(" "):                                 # 按空格分词（优先在空格处折行）
        cand = word if not line else line + " " + word           # 试装入当前行
        if _disp_width(cand) <= width:                           # 整词装得下 → 收入当前行
            line = cand
        else:                                                    # 装不下 → 收当前行、词另行处理
            if line:                                             # 当前行有内容才需要收行
                out.append(line)                                 # 收尾当前行
            pieces = _hard_break(word) if _disp_width(word) > width else [word]   # 超宽词硬折
            out.extend(pieces[:-1])                              # 硬折出的完整段直接成行
            line = pieces[-1]                                    # 末段留作当前行继续装
    if line or not out:                                          # 末行收尾（全空文本 → 空行占位）
        out.append(line)
    if not text:                                                 # 空文本 → 返回空串
        return ""
    first, *rest = out                                           # 首行与续行分组
    return "\n".join([first] + [indent + r for r in rest])       # 首行顶格、续行统一加缩进前缀


def _print_table(rows: list) -> None:
    """打印校验结果表：表头 + 78 宽分隔线 + 逐行（编号 / 校验项 / 结果 / 说明，说明自动折行）。"""
    print(_pad("编号", _COL_NO) + _pad("校验项", _COL_ITEM) + _pad("结果", _COL_RES) + "说明")  # 表头行
    print("-" * 78)                                              # 小节分隔线（统一样式）
    indent = " " * (_COL_NO + _COL_ITEM + _COL_RES)              # 说明列续行缩进（对齐说明列起始位置）
    for no, item, ok, detail in rows:                            # 逐行打印（行 = 四元组）
        result = "PASS" if ok else "FAIL"                        # 布尔结果 → PASS/FAIL 文本
        body = _wrap(str(detail), 78 - len(indent), indent)      # 说明列折行（列宽 78 减去前三列）
        print(_pad(no, _COL_NO) + _pad(item, _COL_ITEM) + _pad(result, _COL_RES) + body)  # 对齐输出一行


# ============================================================
# ② 基础校验：文件存在 / YAML 解析 / 顶层三段
# ============================================================
def _basic_rows(cfg_path: Path) -> tuple:
    """基础校验三行（B1~B3）。返回 (行列表, 解析成功的 cfg 字典或 None)。"""
    rows: list = []                                              # 结果行累积器
    if cfg_path.is_file():                                       # B1：配置文件存在性
        rows.append(("B1", "配置文件存在", True,
                     str(cfg_path.relative_to(_CODE_ROOT))))     # 说明显示仓库内相对路径（简短易读）
    else:
        rows.append(("B1", "配置文件存在", False,                # B1 失败行：附起步指引
                     f"不存在：{cfg_path}（新型号先复制 assets/template.yaml 起步）"))
        return rows, None                                        # 文件不存在 → 后续无从校验
    try:
        with open(cfg_path, encoding="utf-8") as fh:             # 按 utf-8 读取（与库 load_config 一致）
            cfg = yaml.safe_load(fh)                             # B2：yaml 解析（语法错误抛 YAMLError）
    except Exception as e:                                       # 捕获 YAMLError / OSError 等读取解析异常
        rows.append(("B2", "YAML 解析为非空字典", False, f"解析失败：{e}"))   # B2 失败行：附异常消息
        return rows, None                                        # 解析失败 → 后续无从校验
    if isinstance(cfg, dict) and cfg:                            # 顶层须为非空映射（空文件/纯注释均不合规）
        rows.append(("B2", "YAML 解析为非空字典", True, f"顶层共 {len(cfg)} 个键"))   # B2 通过行
        cfg_ok = cfg                                             # cfg 可供后续硬性约束校验使用
    else:
        rows.append(("B2", "YAML 解析为非空字典", False,         # B2 失败行：附实际类型
                     f"顶层须为非空字典，实际解析为 {type(cfg).__name__}"))
        return rows, None                                        # 顶层结构错误 → 后续无从校验
    bad = [k for k in ("joyarm", "robotics", "backend")          # B3：三顶层段均须存在且为字典
           if not isinstance(cfg_ok.get(k), dict)]
    if not bad:                                                  # 三段齐全 → 通过
        rows.append(("B3", "顶层三段齐全且为字典", True, "joyarm / robotics / backend"))   # B3 通过行
    else:                                                        # 缺段或写成列表等 → 不通过（仍返回 cfg 供逐项定位）
        rows.append(("B3", "顶层三段齐全且为字典", False, f"缺失或非字典：{bad}"))   # B3 失败行
    return rows, cfg_ok                                          # B3 失败时也返回 cfg（硬性约束区逐项报缺）


# ============================================================
# ③ 硬性约束校验的各单项助手（口径均与库 joyarm.py 初始化自检一致）
# ============================================================
def _soft_within_hard(soft, hard, tag: str) -> list:
    """软限位越出硬限位的问题清单（有限软限位逐维比较，±1e-9 容差，与库 _soft_beyond_hard 同口径）。

    :param soft: soft_limits_from_cfg 解析出的软限位（None 表示无可比项）。
    :param hard: limits_from_joint_cfgs 解析出的硬限位（None 表示无可比项）。
    :param tag: 问题定位前缀（如 "arm 关节"）。
    :return: 问题字符串列表（空列表 = 未越出）。
    """
    if soft is None or hard is None:                             # 任一侧缺失 → 无可比项
        return []                                                # 缺失本身由调用方的行另行报告
    problems: list = []                                          # 问题累积器
    for key, is_upper in (("q_min", False), ("q_max", True),     # 四键逐键比较（q_min 比下限，其余比上限）
                          ("dq_max", True), ("tau_max", True)):
        s = np.asarray(getattr(soft, key), dtype=float)          # 软限位该键数组
        h = np.asarray(getattr(hard, key), dtype=float)          # 硬限位该键数组
        for i in range(min(s.shape[0], h.shape[0])):             # 逐关节比较（维度不一致时按短侧，长度问题另有行报告）
            if not np.isfinite(s[i]):                            # ±∞ 软限维 = 不设软限（库口径）
                continue                                         # 跳过该维
            beyond = s[i] > h[i] + 1e-9 if is_upper else s[i] < h[i] - 1e-9   # 越出方向按上/下限判定
            if beyond:                                           # 越出 → 记录问题（数值 4 位有效数字易读）
                problems.append(f"{tag}[{i + 1}] 软{key}={s[i]:.4g} 越出硬{key}={h[i]:.4g}")
    return problems                                              # 空列表 = 软限位均在硬限位内


def _zero_in_hard(hard, tag: str) -> list:
    """全零位形（ZERO 态回零目标）逐关节须落在硬限位区间内（1e-12 容差，与库同口径）。"""
    if hard is None:                                             # 无该组限位（如无 end 关节）
        return []                                                # 不适用
    q_min = np.asarray(hard.q_min, dtype=float)                  # 硬限位下限数组
    q_max = np.asarray(hard.q_max, dtype=float)                  # 硬限位上限数组
    bad = np.where((q_min > 1e-12) | (q_max < -1e-12))[0]        # 区间不含零位的关节下标（库同判据）
    return [f"{tag}[{i + 1}] 硬限位 [{q_min[i]:.4g}, {q_max[i]:.4g}] 不含零位" for i in bad]  # 逐关节报告


def _home_in_soft(home_raw, soft, n: int, key: str) -> list:
    """home 位形维度须等于关节数且逐关节在软限位内（1e-12 容差，与库 _parse_home 同口径）。"""
    if n == 0:                                                   # 无该组关节 → 不适用
        return []                                                # 空清单
    if home_raw is None:                                         # 有关节却缺 home → 违规（库必配）
        return [f"joyarm.{key} 缺失（须 {n} 元列表）"]
    try:
        home = np.asarray(home_raw, dtype=float).reshape(-1)     # 展平为一位数组（非数值序列会抛错）
    except (TypeError, ValueError) as e:                         # 捕获非数值内容
        return [f"joyarm.{key} 非数值序列：{e}"]
    if home.shape[0] != n:                                       # 维度须等于关节数
        return [f"joyarm.{key} 维度 {home.shape[0]} ≠ 关节数 {n}"]
    if soft is None:                                             # 软限位缺失/解析失败已由 H3/H4 报告
        return []                                                # 此处不重复报告
    q_min = np.asarray(soft.q_min, dtype=float)                  # 软限位下限数组
    q_max = np.asarray(soft.q_max, dtype=float)                  # 软限位上限数组
    bad = np.where((home < q_min - 1e-12) | (home > q_max + 1e-12))[0]        # 越出软限位的关节下标
    return [f"joyarm.{key}[{i + 1}]（1 基）值 {home[i]:.4g} 越出软限位 [{q_min[i]:.4g}, {q_max[i]:.4g}]"
            for i in bad]                                        # 逐关节报告（关节序号 + 值 + 软限区间）


def _parse_specs(spec) -> list:
    """把 robotics.<域> 规格解析成 [(注册名, 构造参数), ...]；结构非法项记 (None, 原始值)。"""
    if spec is None:                                             # 段未配置 → 空列表
        return []
    specs = spec if isinstance(spec, (list, tuple)) else [spec]  # 允许单条不包列表（库同口径）
    parsed: list = []                                            # 解析结果累积器
    for s in specs:                                              # 逐条规格解析
        if isinstance(s, str):                                   # 形式一：注册名字符串
            parsed.append((s, {}))                               # 无构造参数
        elif isinstance(s, dict):                                # 形式二：{name: 注册名, **构造参数}
            params = dict(s)                                     # 拷贝字典（不改动原 cfg）
            name = params.pop("name", None)                      # 取出注册名
            parsed.append((name, params if name is not None else s))  # 缺 name 时保留原值供报告
        else:                                                    # 其他类型（数字等）→ 非法
            parsed.append((None, s))                             # None 名占位（调用方报告非法类型）
    return parsed


def _domain_problems(cfg_key: str, spec, registry: dict) -> list:
    """robotics.<cfg_key> 段问题清单：未配置 / 结构非法 / 注册名不在册 / traj·ctrl 起搏键非法。"""
    if spec is None:                                             # 段缺失（本检查按 template 六域全配口径判 FAIL）
        return [f"robotics.{cfg_key} 未配置（template.yaml 六域全配，缺域将禁用对应功能）"]
    problems: list = []                                          # 问题累积器
    pacing = "plan_hz" if cfg_key == "trajectory" else (         # 起搏键：traj 域必配 plan_hz、
        "ctrl_hz" if cfg_key == "control" else None)             # ctrl 域必配 ctrl_hz，其余域无
    for name, params in _parse_specs(spec):                      # 逐条规格检查
        if name is None:                                         # 结构非法（缺 name 键或类型错误）
            problems.append(f"规格缺 name 键或类型非法：{params!r}")
            continue                                             # 本条无法继续查
        if name not in registry:                                 # 注册名不在对应域 REGISTRY
            problems.append(f"注册名『{name}』不在 {cfg_key} 域 REGISTRY（可用：{sorted(registry)}）")
        if pacing:                                               # traj/ctrl 域另查起搏键（库 _check_pacing_hz 同口径）
            v = params.get(pacing) if isinstance(params, dict) else None        # 取起搏键值
            try:                                                 # 数值合法性：非空、≥1、整值
                ok = v is not None and float(v) >= 1.0 and float(v).is_integer()
            except (TypeError, ValueError):                      # 非数值（字符串等）→ 非法
                ok = False
            if not ok:                                           # 起搏键非法 → 报告
                problems.append(f"『{name}』{pacing}={v!r} 非法（须 int 或整值浮点且 ≥1）")
    return problems


def _urdf_table(urdf_path: Path) -> tuple:
    """解析 URDF 全部 <joint> 元素。返回 ({关节名: 信息}, None) 或 (None, 错误消息)。

    信息键：type（关节类型）、mimic（是否从动）、q_min/q_max/dq_max/tau_max（<limit> 四键）。
    """
    def _num(lim, key):                                          # <limit> 属性值转 float
        v = None if lim is None else lim.get(key)                # 无 <limit> 元素或无该属性 → None
        return None if v is None else float(v)                   # 缺值不参与比对（库同口径）

    if not urdf_path.is_file():                                  # URDF 文件缺失（joyarm-urdf-creator 的 robot_model 未就绪）
        return None, f"URDF 文件不存在：{urdf_path}"
    try:
        root = ET.parse(str(urdf_path)).getroot()                # 解析 XML 取 <robot> 根元素
    except Exception as e:                                       # XML 语法错误等
        return None, f"URDF 解析失败：{e}"
    table: dict = {}                                             # {关节名: 信息} 累积器
    for j in root.iterfind("joint"):                             # 遍历全部 <joint> 元素
        lim = j.find("limit")                                    # <limit lower/upper/effort/velocity>
        table[j.get("name")] = {                                 # 四键命名与 cfg 对齐便于比对
            "type": j.get("type"),                               # 关节类型（revolute/prismatic/fixed…）
            "mimic": j.find("mimic") is not None,                # 是否 mimic 从动关节
            "q_min": _num(lim, "lower"),                         # lower → q_min
            "q_max": _num(lim, "upper"),                         # upper → q_max
            "dq_max": _num(lim, "velocity"),                     # velocity → dq_max
            "tau_max": _num(lim, "effort"),                      # effort → tau_max
        }
    return table, None                                           # 解析成功


def _joint_name_problems(arm_cfgs: list, end_cfgs: list, table, err) -> list:
    """cfg 关节登记与 URDF 驱动关节的一致性问题清单（同名同序 / 全登记 / 命名约定 / 禁登记项）。"""
    if not arm_cfgs:                                             # 本体电机列表必配（库同口径）
        return ["backend.arm.joints 为空（本体电机列表必配）"]
    if table is None:                                            # URDF 缺失或解析失败 → 直接报告原因
        return [err]
    problems: list = []                                          # 问题累积器
    driven = [n for n, u in table.items()                        # URDF 驱动关节 = 活动类型且非 mimic
              if u["type"] in _ACTIVE_JOINT_TYPES and not u["mimic"]]
    arm_driven = [n for n in driven if re.fullmatch(r"joint\d+", n)]         # arm 命名约定 joint1~jointN
    end_driven = [n for n in driven if re.fullmatch(r"joint_end\d+", n)]     # end 命名约定 joint_end1~joint_endM
    other = [n for n in driven if n not in arm_driven and n not in end_driven]   # 不符约定的驱动关节
    if other:                                                    # 命名不符约定 → 报告（库同口径）
        problems.append(f"URDF 驱动关节 {other} 命名不符约定（arm 须 jointN、end 须 joint_endN）")
    if arm_driven != [f"joint{i}" for i in range(1, len(arm_driven) + 1)]:    # 编号须从 1 连续且按序
        problems.append(f"URDF arm 驱动关节编号/顺序异常：{arm_driven}")
    if end_driven != [f"joint_end{i}" for i in range(1, len(end_driven) + 1)]:    # end 同理
        problems.append(f"URDF end 驱动关节编号/顺序异常：{end_driven}")
    for part, joint_cfgs, ref in (("arm", arm_cfgs, arm_driven), ("end", end_cfgs, end_driven)):   # 分组比对
        names = [str(j.get("name", "")) for j in joint_cfgs]     # cfg 侧登记名（按序）
        dup = sorted({n for n in names if names.count(n) > 1})   # cfg 侧重名检测
        if dup:                                                  # 重名 → 报告
            problems.append(f"backend.{part}.joints 关节名重复：{dup}")
        for j in joint_cfgs:                                     # 登记的每个关节须为 URDF 驱动关节
            u = table.get(str(j.get("name", "")))                # URDF 侧同名关节信息
            if u is None:                                        # URDF 中不存在 → 报告
                problems.append(f"backend.{part}.joints『{j.get('name')}』在 URDF 中不存在")
            elif u["type"] not in _ACTIVE_JOINT_TYPES:           # fixed 等非活动关节 → 不允许登记
                problems.append(f"backend.{part}.joints『{j.get('name')}』为 {u['type']}（fixed 不允许登记）")
            elif u["mimic"]:                                     # mimic 从动关节 → 不允许登记
                problems.append(f"backend.{part}.joints『{j.get('name')}』为 mimic 从动关节（不允许登记）")
        if names != ref:                                         # cfg 与 URDF 须同名同序 → 报告两侧清单
            problems.append(f"backend.{part}.joints 与 URDF 不同名同序：cfg={names}，URDF={ref}")
    cfg_names = ([str(j.get("name", "")) for j in arm_cfgs]      # cfg 两侧全部登记名
                 + [str(j.get("name", "")) for j in end_cfgs])
    unregistered = [n for n in driven if n not in cfg_names]     # URDF 驱动关节中未被 cfg 登记者
    if unregistered:                                             # 驱动关节须全部登记 → 报告
        problems.append(f"URDF 驱动关节 {unregistered} 未在 cfg.backend.joints 登记（须全部登记）")
    return problems


def _limit_match_problems(arm_cfgs: list, end_cfgs: list, table) -> list:
    """cfg 四键限位与 URDF <limit> 逐键比对的问题清单（容差 _URDF_LIMIT_TOL，与库同口径）。"""
    if table is None:                                            # URDF 不可用（原因已在 H14 报告）
        return []                                                # 无可比项
    problems: list = []                                          # 问题累积器
    for part, joint_cfgs in (("arm", arm_cfgs), ("end", end_cfgs)):      # arm / end 两组分别比对
        for j in joint_cfgs:                                     # 逐关节条目比对
            u = table.get(str(j.get("name", "")))                # URDF 侧同名关节信息
            if u is None:                                        # 不存在的问题已在 H14 报告
                continue                                         # 跳过
            for key in ("q_min", "q_max", "dq_max", "tau_max"):  # 四键逐一比对
                cfg_v, urdf_v = j.get(key), u[key]               # cfg 侧值与 URDF 侧值
                if cfg_v is None or urdf_v is None:              # 任一侧缺值不比对（库同口径）
                    continue
                if abs(float(cfg_v) - urdf_v) > _URDF_LIMIT_TOL:  # 超容差 → 不一致
                    problems.append(f"{part}『{j.get('name')}』{key} cfg={cfg_v} ≠ URDF={urdf_v}")
    return problems


# ============================================================
# ③ 硬性约束逐项校验（H1~H15）
# ============================================================
def _hard_rows(cfg: dict, urdf_path: Path) -> list:
    """硬性约束逐项校验，返回 H1~H15 行列表（行 = (编号, 校验项, 是否通过, 说明)）。"""
    rows: list = []                                              # 结果行累积器
    jcfg = cfg.get("joyarm") if isinstance(cfg.get("joyarm"), dict) else {}        # joyarm 段（缺失按空字典）
    bcfg = cfg.get("backend") if isinstance(cfg.get("backend"), dict) else {}      # backend 段
    rcfg = cfg.get("robotics") if isinstance(cfg.get("robotics"), dict) else {}    # robotics 段

    def joints_of(part: str) -> list:                            # backend.<part>.joints 安全提取（仅字典条目）
        seg = bcfg.get(part)                                     # arm / end 子段
        js = seg.get("joints") if isinstance(seg, dict) else None                 # joints 键
        return [j for j in js if isinstance(j, dict)] if isinstance(js, list) else []   # 仅保留字典条目

    arm_cfgs, end_cfgs = joints_of("arm"), joints_of("end")      # 两侧关节条目
    n_arm, n_end = len(arm_cfgs), len(end_cfgs)                  # 两组关节数（后续维度基准）

    # ---- H1：末端帧名 ----
    frame_end = jcfg.get("frame_end")                            # 末端坐标系名（缺失时为 None）
    rows.append(("H1", "joyarm.frame_end 已配置", bool(frame_end),
                 str(frame_end) if frame_end else "缺失（末端帧名，须为 URDF 中存在的 pin 模型帧）"))

    # ---- H2：本体电机列表 ----
    unnamed = sum(1 for j in arm_cfgs if not j.get("name"))      # 缺 name 的条目数
    rows.append(("H2", "backend.arm.joints 非空且含 name", n_arm > 0 and unnamed == 0,   # H2 行：非空且全含 name
                 f"共 {n_arm} 个关节：{', '.join(str(j.get('name')) for j in arm_cfgs)}" if n_arm and not unnamed
                 else "为空或存在缺 name 的条目（本体电机列表必配）"))

    # ---- 硬限位与软限位解析（公共接口，与库同源）----
    hard_arm = limits_from_joint_cfgs(arm_cfgs)                  # 硬限位 = backend.arm.joints 四键
    hard_end = limits_from_joint_cfgs(end_cfgs)                  # n_end=0 时返回 None

    def _soft(section_key: str, n: int) -> tuple:                # 软限位段解析（结构问题转文字）
        raw = jcfg.get(section_key)                              # 四键字典
        if n == 0:                                               # 无该组关节 → 软限位不适用
            return None, None                                    # （关节列表为空的问题由 H2 报告）
        if raw is None:                                          # 有关节却缺软限位段 → 违规
            return None, f"joyarm.{section_key} 缺失（{n} 个关节必配）"
        try:
            return soft_limits_from_cfg(raw, n), None            # 公共接口解析（结构/长度/负值自检）
        except Exception as e:                                   # 解析失败 → 返回问题文字
            return None, f"joyarm.{section_key} 解析失败：{e}"

    soft_arm, err_arm = _soft("joint_limits_soft_arm", n_arm)    # arm 软限位
    soft_end, err_end = _soft("joint_limits_soft_end", n_end)    # end 软限位

    # ---- H3/H4：软限位须位于硬限位内 ----
    problems = _soft_within_hard(soft_arm, hard_arm, "arm 关节") if n_arm else ["backend.arm.joints 为空（见 H2）"]
    problems = ([err_arm] if err_arm else []) + problems         # 合并软限位段自身的问题
    rows.append(("H3", "软限位在硬限位内（arm）", not problems,
                 "；".join(problems) if problems else "q/dq/tau 逐维均未越出"))
    if n_end:                                                    # end 组有关节才需校验
        problems = _soft_within_hard(soft_end, hard_end, "end 关节")
        problems = ([err_end] if err_end else []) + problems     # 同上合并
        rows.append(("H4", "软限位在硬限位内（end）", not problems,
                     "；".join(problems) if problems else "q/dq/tau 逐维均未越出"))
    else:                                                        # 无 end 关节 → 不适用（按通过计）
        rows.append(("H4", "软限位在硬限位内（end）", True, "无 end 关节（不适用）"))

    # ---- H5：zero（全零位形）须在硬限位内 ----
    problems = (_zero_in_hard(hard_arm, "arm") if n_arm else ["backend.arm.joints 为空（见 H2）"])
    problems += _zero_in_hard(hard_end, "end") if n_end else []  # end 组无关节则不比对
    rows.append(("H5", "zero 全零位形在硬限位内", not problems,
                 "；".join(problems) if problems else "zero=ZERO 态回零目标，逐关节均在硬限位内"))

    # ---- H6：home 位形须在软限位内（软⊆硬，故亦在硬限位内）----
    problems = (_home_in_soft(jcfg.get("home_arm"), soft_arm, n_arm, "home_arm")   # arm 侧 home 校验
                if n_arm else ["backend.arm.joints 为空（见 H2）"])
    problems += _home_in_soft(jcfg.get("home_end"), soft_end, n_end, "home_end") if n_end else []   # end 侧（有关节才查）
    rows.append(("H6", "home 位形在软限位内", not problems,      # H6 行：两侧 home 均在软限位内
                 "；".join(problems) if problems else "home_arm / home_end 逐关节均在软限位内"))

    # ---- H7~H12：robotics 六域注册名逐域在册 ----
    for i, (cfg_key, registry) in enumerate(_DOMAINS, start=7):  # 六域逐域一行
        problems = _domain_problems(cfg_key, rcfg.get(cfg_key), registry)        # 该域问题清单
        names = [n if n else "<非法>" for n, _ in _parse_specs(rcfg.get(cfg_key))]   # 已配置注册名（展示用）
        rows.append((f"H{i}", f"robotics.{cfg_key} 注册名在册", not problems,   # H7~H12 行：注册名在册
                     f"已配置 {', '.join(names)}" if not problems else "；".join(problems)))

    # ---- H13：backend.name 在 backend REGISTRY ----
    bname = bcfg.get("name")                                     # backend 注册名（缺失为 None，显示「—」）
    in_reg = isinstance(bname, str) and bname in _BACKEND_REGISTRY       # 在册判定
    rows.append(("H13", "backend.name 在 REGISTRY", in_reg,      # H13 行：注册名在册
                 f"{bname}（可用：{sorted(_BACKEND_REGISTRY)}）" if in_reg   # 通过说明：注册名 + 可用清单
                 else f"{bname if bname is not None else '—'} 不在 REGISTRY"   # 失败说明：实际名
                      f"（可用：{sorted(_BACKEND_REGISTRY)}；须先用 joyarm-backend-creator 完成 backend 注册）"))   # 修复指引

    # ---- H14：关节名与 URDF 驱动关节一致 ----
    table, urdf_err = _urdf_table(urdf_path)                     # URDF 关节表（或错误消息）
    problems = _joint_name_problems(arm_cfgs, end_cfgs, table, urdf_err)         # 一致性问题清单
    rows.append(("H14", "关节名与 URDF 驱动关节一致", not problems,   # H14 行：同名同序全登记
                 "同名同序且 URDF 驱动关节全部登记" if not problems else "；".join(problems)))

    # ---- H15：backend 四键限位与 URDF <limit> 一致 ----
    problems = _limit_match_problems(arm_cfgs, end_cfgs, table)  # 逐键比对问题清单
    ok15 = table is not None and not problems                    # URDF 可用且无比对问题才算通过
    detail = (f"逐关节 q/dq/tau 四键均在容差 {_URDF_LIMIT_TOL} 内" if ok15   # 通过说明
              else "URDF 不可用，无法比对（原因见 H14）" if table is None else "；".join(problems))   # 失败说明
    rows.append(("H15", "硬限位四键与 URDF 一致", ok15, detail))   # H15 行
    return rows                                                  # 返回全部硬性约束行


# ============================================================
# 主流程：目标定位 → 基础校验 → 硬性约束 → 终极构造 → 总结
# ============================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数：--model 待校验型号名（缺省 rebot_arm_b601_dm）。"""
    parser = argparse.ArgumentParser(                            # 参数解析器
        description="JoyArm 型号配置文件完整性检查（离线，无需硬件）")
    parser.add_argument("--model", default="rebot_arm_b601_dm",  # 型号名参数
                        help="待校验型号名（= config/<型号>.yaml 文件名；缺省 rebot_arm_b601_dm）")
    return parser.parse_args()                                   # 返回解析后的参数


def main() -> int:
    """主流程入口。返回进程退出码：0 = 全部通过（SUCCESS），1 = 任一 FAIL。"""
    args = parse_args()                                          # 命令行参数
    model = args.model                                           # 待校验型号名
    cfg_path = _CODE_ROOT / "joyarm_core" / "config" / f"{model}.yaml"       # 型号配置文件路径
    urdf_path = (_CODE_ROOT / "joyarm_core" / "robot_model" / model          # 型号 URDF 路径（命名链同型号名）
                 / "urdf" / f"{model}.urdf")

    print("=" * 78)                                              # ① 大节：目标定位
    print("① 目标定位（型号名 = 配置文件名 = URDF 名）")
    print("-" * 78)                                              # 小节分隔线
    print(f"待校验型号：{model}")                                 # 型号名
    print(f"配置文件　：{cfg_path}")                              # 配置文件绝对路径
    print(f"URDF 文件　：{urdf_path}")                            # URDF 绝对路径（H14/H15 与终极校验使用）
    models = listArm()                                           # 打印并返回当前可直接创建的型号清单
    print(f"待检型号是否已在可用清单：{'是' if model in models else '否（新型号开发中属预期）'}")

    print("=" * 78)                                              # ② 大节：基础校验
    print("② 基础校验：文件 / 解析 / 顶层三段")
    basic, cfg = _basic_rows(cfg_path)                           # 基础三行 + 解析出的 cfg（失败为 None）
    _print_table(basic)                                          # 打印基础校验表

    print("=" * 78)                                              # ③ 大节：硬性约束
    print("③ 硬性约束逐项校验（口径与库 JoyArm 初始化自检一致）")
    if cfg is None:                                              # 基础校验未过 → 无 cfg 可查
        hard: list = []                                          # 硬性约束行置空
        print("  基础校验未通过，硬性约束逐项校验跳过（先修复 B1/B2 后再查）")   # 跳过说明
    else:                                                        # cfg 可用 → 逐项校验
        hard = _hard_rows(cfg, urdf_path)                        # H1~H15 行
        _print_table(hard)                                       # 打印硬性约束表

    print("=" * 78)                                              # ④ 大节：终极校验
    print("④ 终极校验：joyArm 离线构造（库内全套自检 + URDF 装配 + 求解器实例化，不连接硬件）")
    print("-" * 78)                                              # 小节分隔线
    ultimate_ok = False                                          # 终极校验结果（默认未通过）
    if cfg is None:                                              # 基础未过 → 构造必然失败，跳过免噪声
        print("FAIL：基础校验未通过，joyArm 构造必然失败，跳过（先修复 B1/B2）")
    else:                                                        # 基础可过 → 真正做一次离线构造
        try:
            arm = joyArm(model)                                  # 工厂四步校验 + 类内自检（离线不触硬件）
            arm.close()                                          # 显式安全收尾（幂等；离线仅打印豁免日志）
            ultimate_ok = True                                   # 构造成功
            print(f"PASS：joyArm({model!r}) 离线构造成功（已 close 收尾；"
                  "期间「未连接……豁免跳过」类日志为正常提示）")
        except Exception as e:                                   # 构造失败（ValueError 等携具体原因）
            print(f"FAIL：joyArm({model!r}) 构造失败，原因：")
            for line in str(e).splitlines() or [repr(e)]:        # 异常消息逐行缩进展示（多行消息可读）
                print(f"    {line}")

    all_rows = basic + hard                                      # 全部表行（汇总统计用）
    n_pass = sum(1 for _, _, ok, _ in all_rows if ok)            # 通过项计数
    print("=" * 78)                                              # ⑤ 大节：总结
    print("⑤ 总结")
    print("-" * 78)                                              # 小节分隔线
    n_basic, n_hard = len(basic), len(hard)                      # 两表项数
    p_basic = sum(1 for _, _, ok, _ in basic if ok)              # 基础通过数
    p_hard = sum(1 for _, _, ok, _ in hard if ok)                # 硬性约束通过数
    if n_pass == len(all_rows) and ultimate_ok:                  # 全部通过（含终极校验）
        print(f"  基础校验 {p_basic}/{n_basic}；硬性约束 {p_hard}/{n_hard}；终极校验 通过")
        print(f"SUCCESS：型号『{model}』配置文件完整可用")
        return 0                                                 # 退出码 0
    failed = [no for no, _, ok, _ in all_rows if not ok]         # 未通过项编号清单
    print(f"  基础校验 {p_basic}/{n_basic}；硬性约束 {p_hard}/{n_hard}；"   # 两表通过统计
          f"终极校验 {'通过' if ultimate_ok else '未通过/跳过'}")
    print(f"  未通过项：{failed if failed else '无（仅终极构造失败，见 ④ 节原因）'}")   # 未通过项定位
    print(f"FAILED：型号『{model}』配置存在上述问题，请按各行说明修复（退出码 1）")   # 失败总结语
    return 1                                                     # 退出码 1


if __name__ == "__main__":                                       # 直接运行入口（被导入时不执行）
    sys.exit(main())                                             # main 返回退出码（0 全过 / 1 任一 FAIL）
