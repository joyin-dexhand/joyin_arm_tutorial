"""
==============================================================================
URDF 模型资产检查  (joyarm-urdf-creator skill)
==============================================================================
【功能概要】
    离线检查某型号机械臂的 robot_model 模型资产是否满足 joyarm_core 接入要求：
      1. URDF 文件存在（固定路径 robot_model/<型号>/urdf/<型号>.urdf）
      2. pinocchio 可加载（pin.buildModelFromUrdf 解析成功）
      3. <robot name> 与型号名一致（型号名 = 目录名 = URDF 文件名）
      4. 逐关节四限位完整（非 fixed 关节 lower/upper/effort/velocity 四键齐全）
      5. mesh 相对路径引用且文件存在（../meshes/*.STL 逐个核对）
      6. config 的 joyarm.frame_end 末端帧存在于 pin 模型 frames
      7. 活动关节名与 config backend.arm 段同名同序（对齐 JoyArm._build_pin_model）
    七项全过 → 总结 SUCCESS、退出码 0；任一失败 → 退出码 1。

【原理简介】
    URDF 是机械臂的结构说明文件：joyarm_core 创建 JoyArm 对象时按
    `robot_model/<型号>/urdf/<型号>.urdf` 固定路径加载它，交给 pinocchio 构建
    运动学/动力学模型；URDF 以相对路径（../meshes/*.STL）引用外观网格，因此
    urdf/ 与 meshes/ 必须整体存放、一并迁移。本脚本不构造 JoyArm 整机（那是
    joyarm-arm-creator 的事），而是在"资产层"按接入方 joyarm.py 的同一口径逐项检查：
    XML 层用 ElementTree 读取 <robot>/<joint>/<mesh> 标签；模型层先建全量
    pin 模型，再锁定非 config.arm 名单的活动关节构建缩减模型，比对关节名与
    自由度——与 JoyArm._build_pin_model 的做法完全一致。

【环境与运行】
    ⓪ 安装 uv（仅首次）：curl -LsSf https://astral.sh/uv/install.sh | sh
       （Windows PowerShell：irm https://astral.sh/uv/install.ps1 | iex）
    ① cd <仓库路径>/joyarm_code && uv sync && source .venv/bin/activate
    ② python .agents/skills/joyarm-urdf-creator/scripts/urdf_check.py --model <型号>
       （离线脚本，无需硬件；--model 缺省 rebot_arm_b601_dm）

【预期输出】
    七项检查逐项 [PASS]/[FAIL] 及明细表；全部通过时末尾打印 SUCCESS 并以
    退出码 0 结束，任一失败则退出码 1。
==============================================================================
"""

import argparse                                                    # 命令行参数解析（--model 型号名）
import sys                                                         # sys.path 注入与进程退出码
import xml.etree.ElementTree as ET                                 # 解析 URDF 的 XML 标签
from pathlib import Path                                           # 路径拼接与文件存在性检查

import pinocchio as pin                                            # URDF → 运动学/动力学模型（与 joyarm_core 同一入口）

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))       # 仓库未安装时也可直接运行（parents[4] = joyarm_code/）
from joyarm_core.joyarm import load_config                         # 读取 config/<型号>.yaml（joyarm_core 公开入口）

# ---------------- 常量（校验口径与 joyarm.py 保持一致） ----------------
_JOYARM_CODE_DIR = Path(__file__).resolve().parents[4]             # joyarm_code/ 根目录（scripts→skill→skills→.agents→joyarm_code）
_ROBOT_MODEL_DIR = _JOYARM_CODE_DIR / "joyarm_core" / "robot_model"  # 全部型号模型资产的根目录
_ACTIVE_JOINT_TYPES = ("revolute", "continuous", "prismatic")      # 活动关节类型（同 joyarm.py _ACTIVE_JOINT_TYPES）
_LIMIT_KEYS = ("lower", "upper", "effort", "velocity")             # <limit> 四限位键（同 URDF 规范第 4 节）
_DEFAULT_MODEL = "rebot_arm_b601_dm"                               # --model 缺省值（当前唯一可用型号）


# ============================================================
# 一、终端格式化助手（None 显「—」，表格列按显示宽度对齐）
# ============================================================
def _disp_w(text: str) -> int:
    """计算字符串的终端显示宽度（中日韩全角字符按 2 列、其余按 1 列）。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)        # 按码点区间粗略区分全角/半角


def _fmt(v) -> str:
    """数值格式化：None →「—」；可转数值则去尾零输出；否则原样返回。"""
    if v is None:                                                  # 缺失值统一显示「—」（规范要求）
        return "—"
    try:
        return f"{float(v):g}"                                     # 数值 → 最短有效表示（如 -2.8、178.15）
    except ValueError:
        return str(v)                                              # 非数值（写坏的属性）原样展示，便于排查


def _table(rows: list, right_cols: frozenset = frozenset()) -> list:
    """把二维表格（首行为表头）渲染为终端等宽对齐的文本行列表。

    :param rows: 表格行列表（每行为单元格列表；第一行视为表头）。
    :param right_cols: 需右对齐的列下标集合（数值列右对齐更易比大小）。
    :return: 对齐后的文本行列表（可直接逐行 print）。
    """
    ncol = len(rows[0])                                            # 列数（以表头行为准）
    widths = [0] * ncol                                            # 逐列最大显示宽度
    for row in rows:                                               # 第一遍扫描：求每列最宽单元格
        for c in range(ncol):
            widths[c] = max(widths[c], _disp_w(str(row[c])))       # 取该列显示宽度的最大值
    lines = []                                                     # 渲染结果行
    for row in rows:                                               # 第二遍渲染：逐行按列宽补空格
        cells = []                                                 # 本行各列的渲染结果
        for c in range(ncol):
            gap = " " * (widths[c] - _disp_w(str(row[c])))         # 对齐需补的空格数
            cells.append(gap + str(row[c]) if c in right_cols      # 右对齐列补左空格
                         else str(row[c]) + gap)                   # 左对齐列补右空格
        lines.append("     " + "  ".join(cells))                   # 行首缩进 5 空格，列间 2 空格
    return lines


# ============================================================
# 二、七项检查（每项返回 (是否通过, 明细行列表)）
# ============================================================
def check_file(urdf_path: Path) -> tuple:
    """检查①：URDF 文件存在（固定路径 = robot_model/<型号>/urdf/<型号>.urdf）。

    :param urdf_path: 待检查的 URDF 文件路径。
    :return: (是否通过, 明细行列表)。
    """
    lines = [f"路径：{urdf_path}"]                                  # 无论成败先打印路径，便于定位
    if not urdf_path.is_file():                                    # 文件不存在 → 附列 robot_model/ 现有目录辅助排查
        lines.append("结果：文件不存在")                            # 不存在结论
        dirs = sorted(p.name for p in _ROBOT_MODEL_DIR.iterdir()   # 扫描 robot_model/ 下一级子目录
                      if p.is_dir()) if _ROBOT_MODEL_DIR.is_dir() else []   # 目录缺失时回退空列表
        lines.append(f"参考：robot_model/ 下现有目录 {dirs if dirs else '—'}")   # 现有目录参考
        return False, lines                                        # FAIL + 明细
    size_kb = urdf_path.stat().st_size / 1024                      # 文件字节数换算 KB（stat().st_size 为字节）
    lines.append(f"结果：文件存在（{size_kb:.1f} KB）")             # 存在结论（附大小）
    return True, lines                                             # PASS + 明细


def check_pin_load(urdf_path: Path, pin_model) -> tuple:
    """检查②：pinocchio 可加载 URDF（pin.buildModelFromUrdf 不抛异常）。

    :param urdf_path: URDF 文件路径（提示用）。
    :param pin_model: 预先构建的全量 pin 模型（None 表示未构建成功）。
    :return: (是否通过, 明细行列表)。
    """
    if pin_model is None:                                          # 前置失败（文件缺失或 pinocchio 解析报错）
        return False, [f"结果：buildModelFromUrdf 未成功（文件：{urdf_path}）"]
    names = [pin_model.names[i] for i in range(1, pin_model.njoints)]  # 活动关节名（下标 0 是 universe 根关节）
    lines = [f"结果：解析成功——活动关节 {len(names)} 个、帧 {pin_model.nframes} 个、nq={pin_model.nq}",   # 解析成功摘要
             f"关节：{names}"]                                     # 活动关节名清单
    return True, lines                                             # PASS + 明细


def check_robot_name(model_name: str, root) -> tuple:
    """检查③：<robot name> 与型号名一致（型号名 = 目录名 = URDF 文件名）。

    :param model_name: 期望的型号名（--model 入参）。
    :param root: URDF 的 XML 根元素（None 表示前置失败）。
    :return: (是否通过, 明细行列表)。
    """
    if root is None:                                               # XML 未解析成功 → 无法读取该属性
        return False, ["结果：URDF XML 未解析成功，无法读取 <robot name>"]
    name = root.get("name") or "—"                                 # <robot> 根元素的 name 属性（缺失显「—」）
    ok = name == model_name                                        # 规范总原则：robot name = 型号名 = 目录名
    verdict = "一致" if ok else "不一致"                            # 判定文案
    return ok, [f"结果：<robot name> = {name!r}，型号名 = {model_name!r}，{verdict}"]


def check_limits(root) -> tuple:
    """检查④：逐关节四限位完整（非 fixed 关节的 lower/upper/effort/velocity 齐全）。

    :param root: URDF 的 XML 根元素（None 表示前置失败）。
    :return: (是否通过, 明细行列表；明细含逐关节四限位表)。
    """
    if root is None:
        return False, ["结果：URDF XML 未解析成功，无法读取 <limit>"]
    rows = [["关节名", "类型", "mimic", "lower", "upper", "effort", "velocity", "判定"]]  # 表头行
    n_fixed = 0                                                    # fixed 关节计数（不设 limit，仅汇总提示）
    incomplete = []                                                # 四限位不完整的关节清单
    for j in root.iterfind("joint"):                               # 按文档顺序遍历全部 <joint> 元素
        jtype = j.get("type") or "—"                               # 关节类型属性（revolute/prismatic/fixed/…）
        if jtype not in _ACTIVE_JOINT_TYPES:                       # fixed 等非活动关节无运动，规范不要求 <limit>
            n_fixed += 1
            continue
        lim = j.find("limit")                                      # <limit> 子元素（可能整段缺失）
        vals = {k: (None if lim is None else lim.get(k))           # 四键取值：元素或属性缺失 → None
                for k in _LIMIT_KEYS}
        miss = [k for k, v in vals.items() if v is None]           # 本关节缺失的键
        jname = j.get("name") or "—"                               # 关节名（缺失显「—」）
        if miss:                                                   # 记入不完整清单（缺键名拼进结论）
            incomplete.append(f"{jname}（缺 {'/'.join(miss)}）")    # 缺键关节入清单
        rows.append([jname, jtype,                                 # 本行前两列：关节名 + 类型
                     "是" if j.find("mimic") is not None else "否",  # mimic 从动关节标记（规范：从动须带 <mimic>）
                     _fmt(vals["lower"]), _fmt(vals["upper"]),      # 位置上下限（revolute 单位 rad，prismatic 单位 m）
                     _fmt(vals["effort"]), _fmt(vals["velocity"]),  # 力/力矩上限与速度上限
                     "OK" if not miss else "缺 " + "/".join(miss)])  # 行判定
    lines = _table(rows, right_cols=frozenset({3, 4, 5, 6}))       # 数值列（lower/upper/effort/velocity）右对齐
    lines.append(f"另：非活动关节 {n_fixed} 个（fixed 无 <limit>，不检查）")
    if incomplete:                                                 # 任一活动关节缺键 → FAIL 并列出明细
        lines.append(f"结果：{len(incomplete)} 个活动关节四限位不完整——" + "；".join(incomplete))   # 不完整清单
        return False, lines                                        # FAIL + 明细
    lines.append("结果：全部活动关节四限位齐全")                   # 全齐结论
    return True, lines                                             # PASS + 明细


def check_meshes(urdf_path: Path, root) -> tuple:
    """检查⑤：mesh 相对路径引用且文件存在（../meshes/*.STL 相对 urdf/ 目录解析）。

    :param urdf_path: URDF 文件路径（相对引用的解析基准 = 其所在目录 urdf/）。
    :param root: URDF 的 XML 根元素（None 表示前置失败）。
    :return: (是否通过, 明细行列表)。
    """
    if root is None:
        return False, ["结果：URDF XML 未解析成功，无法读取 <mesh>"]
    refs = []                                                      # 去重后的 mesh 引用清单（保持文档顺序）
    for m in root.iter("mesh"):                                    # 深度遍历 visual/collision 下所有 <mesh>
        fn = m.get("filename") or "—"                              # mesh 文件名属性
        if fn not in refs:                                         # 同一 STL 被 visual 与 collision 各引一次，去重
            refs.append(fn)
    rows = [["mesh 引用（filename）", "相对路径", "文件存在", "解析路径"]]  # 表头行
    bad = []                                                       # 不合格引用（非相对路径或文件缺失）
    used_names = set()                                             # 已被引用的 STL 文件名（供孤立 mesh 提示）
    for fn in refs:
        is_rel = not (fn.startswith("/") or "://" in fn)           # 规范要求相对路径（排除绝对路径与 URI）
        target = (urdf_path.parent / fn).resolve() if is_rel else None  # 相对 urdf/ 目录解析（../ 回到型号根）
        exists = target is not None and target.is_file()           # 解析结果是否为已存在的文件
        if is_rel and target is not None:
            used_names.add(target.name)                            # 记录引用到的文件名
        if not (is_rel and exists):                                # 非相对 或 文件缺失 → 不合格
            bad.append(fn)
        rows.append([fn, "是" if is_rel else "否", "是" if exists else "否",
                     str(target.relative_to(_JOYARM_CODE_DIR)) if target is not None else "—"])  # 显示仓库内相对路径
    lines = _table(rows, right_cols=frozenset())                   # 全左对齐（路径列长短不一，左对齐易读）
    mesh_dir = urdf_path.parent.parent / "meshes"                  # 型号根下的 meshes/ 目录（与 urdf/ 平级）
    if mesh_dir.is_dir():                                          # 附加提示：目录中未被引用的孤立 mesh（不影响判定）
        orphans = sorted(p.name for p in mesh_dir.iterdir()        # 收集 meshes/ 下的全部文件名
                         if p.is_file() and p.name not in used_names)   # 过滤：未被 URDF 引用者
        lines.append(f"提示：meshes/ 目录孤立文件 {orphans if orphans else '—'}（未被 URDF 引用，不影响判定）")   # 孤立文件提示
    if bad:                                                        # 有不合格引用 → FAIL
        lines.append(f"结果：{len(bad)} 个 mesh 引用不合格（非相对路径或文件缺失）——" + "；".join(bad))   # 不合格清单
        return False, lines                                        # FAIL + 明细
    lines.append(f"结果：{len(refs)} 个 mesh 引用全部为相对路径且文件存在")   # 全部合格结论
    return True, lines                                             # PASS + 明细


def check_frame_end(model_name: str, cfg, pin_model) -> tuple:
    """检查⑥：config 的 joyarm.frame_end 末端帧存在于 pin 模型 frames。

    口径对齐 joyarm.py：getFrameId 查不到时不报错而返回越界哨兵，须用
    0 <= fid < nframes 拦截（buildReducedModel 保留全部 link/帧，故在
    全量模型与缩减模型上查帧结果一致）。

    :param model_name: 型号名（提示用）。
    :param cfg: load_config 返回的配置 dict（None 表示加载失败）。
    :param pin_model: 全量 pin 模型（None 表示未构建成功）。
    :return: (是否通过, 明细行列表)。
    """
    if cfg is None:                                                # config 缺失/解析失败 → 无法读取 frame_end
        return False, [f"结果：config/{model_name}.yaml 加载失败，无法读取 joyarm.frame_end"]
    jcfg = cfg.get("joyarm") if isinstance(cfg.get("joyarm"), dict) else {}  # joyarm 段（缺段按空 dict 处理）
    frame_end = str(jcfg.get("frame_end") or "")                   # 末端帧名（如 frame_end_tcp）
    if not frame_end:                                              # 帧名缺失 → 直接 FAIL
        return False, ["结果：config joyarm.frame_end 缺失（末端帧名，pin 模型帧）"]
    if pin_model is None:                                          # 模型未构建 → 无法查帧
        return False, [f"frame_end = {frame_end!r}（来自 config joyarm.frame_end）",
                       "结果：pin 模型未构建，无法查找该帧"]
    fid = pin_model.getFrameId(frame_end)                          # 查帧索引（查不到时静默返回越界哨兵）
    ok = 0 <= fid < pin_model.nframes                              # 用区间判断拦截越界哨兵（同 _build_pin_model）
    if ok:                                                         # 帧索引在界内 = 帧存在
        return True, [f"frame_end = {frame_end!r}（来自 config joyarm.frame_end）",   # PASS 明细行 1
                      f"结果：帧存在（索引 {fid}/{pin_model.nframes}）"]   # PASS 明细行 2：索引位置
    frames = [f.name for f in pin_model.frames]                    # 失败时列出模型全部可用帧辅助排查
    return False, [f"frame_end = {frame_end!r}（来自 config joyarm.frame_end）",
                   f"结果：帧不存在；模型可用帧：{frames}"]


def check_arm_joints(cfg, pin_model) -> tuple:
    """检查⑦：URDF 活动关节与 config backend.arm 段同名同序（对齐 _build_pin_model）。

    做法：锁定非 cfg.arm 名单的活动关节（含 end 与 mimic 等，锁定位形取零位）
    构建缩减模型，比对缩减模型关节名列表与 cfg 名单，并核对自由度 nq——
    与 joyarm.py JoyArm._build_pin_model 的判据完全一致。

    :param cfg: 配置 dict（None 表示加载失败）。
    :param pin_model: 全量 pin 模型（None 表示未构建成功）。
    :return: (是否通过, 明细行列表)。
    """
    if cfg is None:                                                # config 缺失 → 无法读取名单
        return False, ["结果：config 加载失败，无法读取 backend.arm.joints"]
    backend_cfg = cfg.get("backend") if isinstance(cfg.get("backend"), dict) else {}  # backend 段
    arm_cfgs = list((backend_cfg.get("arm") or {}).get("joints") or [])  # cfg arm 段关节条目列表
    arm_names = [str(j.get("name")) for j in arm_cfgs]             # cfg 关节名顺序（= q 向量顺序）
    lines = [f"config backend.arm.joints：{arm_names if arm_names else '—'}"]
    if not arm_names:                                              # 本体电机名单为空 → FAIL（必配段）
        return False, lines + ["结果：backend.arm.joints 为空（本体电机列表必配）"]
    if pin_model is None:                                          # 模型未构建 → 无法比对
        return False, lines + ["结果：pin 模型未构建，无法比对"]
    arm_set = set(arm_names)                                       # cfg 名单集合（锁定判据用）
    lock_ids = [pin_model.getJointId(n) for i, n in enumerate(pin_model.names)
                if i > 0 and n not in arm_set]                     # 非 arm 名单的活动关节全锁定（i>0 跳过 universe）
    reduced = pin.buildReducedModel(pin_model, lock_ids,           # 锁定位形取零位 → 缩减模型（link/帧全保留）
                                    pin.neutral(pin_model)) if lock_ids else pin_model
    names = [reduced.names[i] for i in range(1, reduced.njoints)]  # 缩减模型活动关节名（文档顺序）
    lines.append(f"URDF 缩减模型关节：{names}（nq={reduced.nq}）")
    ok = names == arm_names and reduced.nq == len(arm_names)       # 同名同序 + 自由度一致（_build_pin_model 同款判据）
    lines.append("结果：" + ("关节同名同序且 nq 一致（与 JoyArm._build_pin_model 判据一致）" if ok   # 一致结论
                           else "缩减模型关节与 cfg 名单不一致（同名同序或 nq 不符）"))   # 不一致结论
    return ok, lines                                               # 判定 + 明细


# ============================================================
# 三、入口：装配七项检查 → 逐项打印 → 总结并设置退出码
# ============================================================
def _parse_args() -> argparse.Namespace:
    """解析命令行参数：--model 型号名（缺省 rebot_arm_b601_dm）。"""
    parser = argparse.ArgumentParser(description="JoyArm URDF 模型资产检查（离线，七项 PASS/FAIL）")  # 命令行说明
    parser.add_argument("--model", default=_DEFAULT_MODEL,         # 型号名 = robot_model/<型号>/ 目录名
                        help="型号名（= robot_model/<型号>/ 目录名 = urdf 文件名），缺省 %(default)s")
    return parser.parse_args()                                     # 解析并返回参数命名空间


def main() -> None:
    """入口：依序执行七项检查并逐项打印，末尾按通过数输出总结与退出码。"""
    args = _parse_args()                                           # 解析 --model 入参
    model_name = args.model                                        # 型号名
    urdf_path = _ROBOT_MODEL_DIR / model_name / "urdf" / f"{model_name}.urdf"  # 固定路径约定（同 JoyArmFactory）

    # ---- 前置加载三项公共数据（失败不抛异常，交由对应检查项报 FAIL） ----
    root = None                                                    # URDF 的 XML 根元素
    try:
        if urdf_path.is_file():                                    # 文件存在才解析（well-formed 校验随解析完成）
            root = ET.parse(urdf_path).getroot()
    except ET.ParseError:
        root = None                                                # XML 语法错误 → 置 None，由检查③④⑤报错
    pin_model = None                                               # pinocchio 全量模型
    try:
        pin_model = pin.buildModelFromUrdf(str(urdf_path))         # 与 joyarm_core 加载 URDF 的同一入口
    except Exception:
        pin_model = None                                           # 解析失败 → 置 None，由检查②⑥⑦报错
    cfg = None                                                     # 型号配置 dict
    try:
        cfg = load_config(model_name)                              # 读 config/<型号>.yaml（公开入口）
    except Exception:
        cfg = None                                                 # 缺失/解析失败 → 置 None，由检查⑥⑦报错

    # ---- 七项检查有序表（标题, 零参闭包）：逐项执行并打印 ----
    checks = [                                                     # (标题, 触发闭包) 依序七项
        ("① URDF 文件存在", lambda: check_file(urdf_path)),        # 检查①：文件存在
        ("② pinocchio 可加载", lambda: check_pin_load(urdf_path, pin_model)),   # 检查②：pin 加载
        ("③ <robot name> = 型号名", lambda: check_robot_name(model_name, root)),   # 检查③：命名链
        ("④ 逐关节四限位完整", lambda: check_limits(root)),        # 检查④：四限位
        ("⑤ mesh 相对路径文件存在", lambda: check_meshes(urdf_path, root)),   # 检查⑤：mesh 引用
        ("⑥ frame_end 帧存在", lambda: check_frame_end(model_name, cfg, pin_model)),   # 检查⑥：末端帧
        ("⑦ 活动关节与 config arm 段一致", lambda: check_arm_joints(cfg, pin_model)),   # 检查⑦：关节名单
    ]
    print("=" * 78)                                                # 大节：总标题区
    print(f"URDF 模型资产检查：型号 {model_name!r}")
    print(f"URDF 路径：{urdf_path}")
    n_pass = 0                                                     # 通过项计数
    for title, run in checks:                                      # 依序执行并打印每项
        ok, lines = run()                                          # 执行检查 → (是否通过, 明细行列表)
        n_pass += 1 if ok else 0                                   # 累计通过数
        print("-" * 78)                                            # 小节分隔线
        print(f"  [{'PASS' if ok else 'FAIL'}] {title}")           # 判定行（PASS/FAIL 定宽对齐）
        for ln in lines:                                           # 明细行（表格 / 结论逐行打印）
            print(f"     {ln}")
    print("=" * 78)                                                # 大节：总结区
    if n_pass == len(checks):                                      # 七项全过 → SUCCESS、退出码 0
        print(f"检查总结：{n_pass}/{len(checks)} 项通过 → SUCCESS")
        print(f"型号 {model_name!r} 的 URDF 资产满足 joyarm_core 接入要求")
        sys.exit(0)                                                # 成功退出码
    print(f"检查总结：{n_pass}/{len(checks)} 项通过 → 存在问题（见上方 [FAIL] 项）")  # 有失败 → 退出码 1
    sys.exit(1)                                                    # 失败退出码（供 CI / 脚本判断）


if __name__ == "__main__":
    main()                                                         # 直接运行时执行入口
