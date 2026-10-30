"""
==============================================================================
整机集成四步校验  (joyarm-arm-creator skill)
==============================================================================
【功能概要】
    1. 以 --model 指定型号（缺省 rebot_arm_b601_dm，当前唯一可用型号）做整机体检；
    2. 第①步 config：config/<型号>.yaml 存在且解析为非空 dict（复用库的 load_config）；
    3. 第②步 backend：cfg 的 backend.name 命中 backend/__init__.py 的 REGISTRY 注册表；
    4. 第③步 robot_model：四路径按序存在（<型号>/、meshes/、urdf/、<型号>.urdf）；
    5. 第④步构造：joyArm(<型号>) 离线构造自检（不触碰任何硬件）并打印整机核心信息；
    6. 末尾汇总四步 PASS/FAIL 表：任一 FAIL 给出回 config/urdf/backend 对应
       creator skill 的修正指引，全 PASS 则打印验收阶梯建议（离线 01、05 →
       真机 02→04→06 → 真机冒烟）。

【原理简介】
    新型号 = 四件套：config yaml（joyarm-config-creator）+ robot_model 模型资产
    （joyarm-urdf-creator）+ backend 硬件后端（joyarm-backend-creator）+ 注册
    与整机自检（本 skill）。joyArm 工厂创建对象前按固定顺序做
    四步校验（见 joyarm_core/joyarm/__init__.py 文件头）：① config 存在且可解析 →
    ② backend 注册名命中 REGISTRY → ③ robot_model 四路径齐全 → ④ JoyArm 构造自检
    （cfg → pin 模型 → 限位 → backend → 求解器依序装配，任一步失败抛携带原因的异常）。
    本脚本把这四步拆开逐项体检：哪一步 FAIL 就回哪个 skill 修正，不必对着工厂一整串
    报错逐行猜原因；第④步全离线，只装配数字模型与算法，不连接任何硬件。

【环境与运行】
    ① 安装 uv（仅首次）
       curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux / macOS
       irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell
    ② 创建虚拟环境并同步依赖
       cd <仓库路径>/joyarm_code && uv sync && source .venv/bin/activate
    ③ 运行本脚本（在 joyarm_code/ 目录下；离线脚本，无需硬件）
       python .agents/skills/joyarm-arm-creator/scripts/arm_check.py --model rebot_arm_b601_dm

【预期输出】四步校验逐项 PASS/FAIL 与汇总表；全 PASS 打印验收阶梯建议并以退出码 0
    结束；任一 FAIL 打印对应回退 skill 指引并以退出码 1 结束（收尾「未连接……豁免跳过」
    类日志属正常提示，非报错）。
==============================================================================
"""

from pathlib import Path                        # 组装库内资源路径与脚本上级目录
import sys                                      # 注入模块搜索路径 + 控制退出码
import argparse                                 # 解析 --model 命令行参数

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # joyarm_code 根入 path（scripts→skill→skills→.agents→joyarm_code 共 4 级），未安装 joyarm_core 时也可直接运行本脚本
import joyarm_core                              # noqa: E402  库包本体：以其所在位置定位 config / robot_model
from joyarm_core import joyArm, listArm         # noqa: E402  工厂创建入口（第④步）与可用型号列举
from joyarm_core.joyarm import load_config      # noqa: E402  第①步复用的配置读取（与工厂第①步同一实现）
from joyarm_core.backend import REGISTRY        # noqa: E402  第②步对照的后端注册表

CORE_DIR = Path(joyarm_core.__file__).resolve().parent    # joyarm_core 库根目录（config/ robot_model/ 均在其下）
DEFAULT_MODEL = "rebot_arm_b601_dm"                       # 缺省体检型号（当前唯一可用型号）

# 验收阶梯：四步全 PASS 后按序推进（命令均在 joyarm_code/ 目录下执行）
LADDER = [
    ("离线", "quick_start/01_model_create.py",    "对象创建与核心信息打印"),   # 阶梯 1：离线对象创建
    ("离线", "quick_start/05_offline_math.py",    "机器人学离线计算与求解器设置"),   # 阶梯 2：离线数学
    ("真机", "quick_start/02_connect_read.py",    "硬件连接与状态读取"),   # 阶梯 3：真机只读
    ("真机", "quick_start/04_direct_command.py",  "直通硬件电机指令"),   # 阶梯 4：真机直通指令
    ("真机", "quick_start/06_fsm_motion.py",      "基于运动状态机的运动"),   # 阶梯 5：真机 FSM 运动
    ("真机", "test/joyarm_test.py",               "真机冒烟（连接 + 状态可视化）"),   # 阶梯 6：真机冒烟
]


def disp_pad(text: str, width: int) -> str:
    """按终端显示宽度右补空格（中日韩字符按 2 列计），用于表格列对齐。"""
    cols = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)   # 逐字符累计显示列数
    return text + " " * max(0, width - cols)                  # 不足宽度补空格，超宽原样返回


def step1_config(model: str):
    """第①步 config：`config/<型号>.yaml` 存在且解析为非空 dict。

    复用 `joyarm_core.joyarm.load_config`（与工厂第①步同一实现，报错消息自带
    文件路径与原始解析错误）；成功返回解析后的 cfg dict，失败返回 None。
    """
    print("-" * 78)                                           # 小节分隔线
    print("第 ① 步 config —— config/<型号>.yaml 存在且解析为非空 dict")
    print("-" * 78)
    yaml_path = CORE_DIR / "config" / f"{model}.yaml"         # 目标配置文件路径（文件名 = 型号名）
    print(f"      检查文件：{yaml_path}")
    try:
        cfg = load_config(model)                              # 库函数：文件缺失/语法错/空文件均抛 ValueError
    except ValueError as e:
        print(f"[FAIL] {e}")                                  # 失败：直接打印库给出的具体原因
        print("      回退指引：回 ../joyarm-config-creator skill——拷贝 template.yaml 为")
        print("      config/<型号>.yaml 并逐段填写后重跑本脚本")
        return None                                           # 失败返回 None，第②步将跳过
    print(f"[PASS] 解析成功：顶层段键 {sorted(cfg)}")         # 成功：展示顶层段键（joyarm/robotics/backend）
    return cfg                                                # 返回 cfg 供第②步继续体检


def step2_backend(cfg) -> bool:
    """第②步 backend：cfg 的 `backend.name` 须命中 `backend/__init__.py` 的 REGISTRY。

    :param cfg: 第①步解析出的配置 dict；为 None（第①步失败）时本步跳过返回 False。
    """
    print("-" * 78)                                           # 小节分隔线
    print("第 ② 步 backend —— cfg 的 backend.name 命中 backend/__init__.py 的 REGISTRY")   # 小节标题
    print("-" * 78)                                           # 小节分隔线
    if cfg is None:                                           # 第①步未通过 → 无 cfg 可查
        print("[FAIL] 依赖第①步失败而跳过（无 config 可解析出 backend.name）")   # 依赖失败提示
        return False                                          # 记 FAIL，继续后面独立的③④步
    seg = cfg.get("backend")                                  # 取 backend 段（缺段时为 None）
    name = seg.get("name") if isinstance(seg, dict) else None  # 取注册名（段非 dict / 缺键时为 None）
    shown = name if name is not None else "—"                 # None 一律显「—」，避免打印空值歧义
    print(f"      backend.name = {shown}；REGISTRY 已注册 = {sorted(REGISTRY)}")
    hit = isinstance(name, str) and name in REGISTRY          # 注册名须为字符串且命中注册表键
    if hit:
        print(f"[PASS] 注册名命中 REGISTRY（对应后端类 {REGISTRY[name].__name__}）")   # 命中分支
        return True                                           # 命中即通过
    print("[FAIL] 注册名未命中 REGISTRY")                      # 未命中分支
    print("      回退指引：yaml 的 backend.name 填写回 ../joyarm-config-creator skill 核对；")   # 指引一：yaml 填写
    print("      backend_<型号>.py 的导入与 REGISTRY 注册回 ../joyarm-backend-creator skill 补齐")   # 指引二：子类注册
    return False


def step3_robot_model(model: str) -> bool:
    """第③步 robot_model：四路径按序存在（与工厂 `_model_paths_missing` 同一口径）。

    路径顺序：`robot_model/<型号>/` → `meshes/` → `urdf/` → `urdf/<型号>.urdf`。
    """
    print("-" * 78)                                           # 小节分隔线
    print("第 ③ 步 robot_model —— 四路径按序存在（<型号>/、meshes/、urdf/、<型号>.urdf）")   # 小节标题
    print("-" * 78)                                           # 小节分隔线
    base = CORE_DIR / "robot_model" / model                   # 型号模型目录（目录名 = 型号名）
    paths = [                                                 # 与工厂校验一致的四个路径，按序检查
        base,                                                 # ① 型号根目录
        base / "meshes",                                      # ② mesh 资产目录
        base / "urdf",                                        # ③ URDF 目录
        base / "urdf" / f"{model}.urdf",                      # ④ URDF 文件（文件名 = 型号名）
    ]
    ok = True                                                 # 全路径存在标志
    for p in paths:                                           # 逐路径打印存在性（✓/✗ 一目了然）
        mark = "✓" if p.exists() else "✗"                     # 存在 ✓ / 缺失 ✗
        ok = ok and p.exists()                                # 任一缺失即整步 FAIL
        print(f"      {mark} {p.relative_to(CORE_DIR)}")      # 相对库根显示，输出更短
    if ok:
        print("[PASS] 四路径齐全（URDF + meshes 模型资产就位）")   # 全存在分支
        return True
    print("[FAIL] 上表 ✗ 项缺失")                             # 有缺失分支
    print("      回退指引：回 ../joyarm-urdf-creator skill——按 URDF 格式规范把 urdf/<型号>.urdf")   # 指引一：放置 URDF
    print("      与 meshes/*.STL 放入 robot_model/<型号>/ 后重跑本脚本")   # 指引二：放置 mesh
    return False


def step4_construct(model: str) -> bool:
    """第④步构造：`joyArm(<型号>)` 离线构造自检（不连接任何硬件）并打印核心信息。

    工厂内部会重跑①~③外部校验再进入 JoyArm 构造自检（cfg → pin 模型 → 限位 →
    backend → 求解器依序装配）；with 退出自动 close() 安全收尾。
    """
    print("-" * 78)                                           # 小节分隔线
    print("第 ④ 步 构造自检 —— joyArm(<型号>) 离线构造（装配 cfg/pin 模型/限位/backend/求解器）")   # 小节标题
    print("-" * 78)                                           # 小节分隔线
    try:
        with joyArm(model) as arm:                            # 工厂入口：任一步失败抛携带原因的 ValueError
            print(f"[PASS] 构造自检通过：arm {arm.n_arm} 关节 / end {arm.n_end} 电机")   # 关节数即装配成功证据
            print(f"      本体关节名：{arm.get_joint_names_arm('all')}")               # cfg 顺序 = q 向量顺序
            print(f"      末端电机名：{arm.get_joint_names_end('all')}")               # 无 end 时为空 tuple
            plan_hz = arm.plan_hz                             # 激活规划器频率（可能为 None）
            ctrl_hz = arm.ctrl_hz                             # 激活控制器频率（可能为 None）
            print(f"      规划 / 控制频率："                  # 频率行（None 显「—」）
                  f"{plan_hz if plan_hz is not None else '—'} Hz / "
                  f"{ctrl_hz if ctrl_hz is not None else '—'} Hz")                     # None 显「—」
            print(f"      home 位形 home_arm (rad)：{arm.home_arm}")                    # 归位关节角（ndarray 直接打印）
    except Exception as e:                                    # 工厂四步任一失败均汇集到此处
        print(f"[FAIL] {e}")                                  # 打印携带原因的完整报错（含可用型号列表）
        print("      回退指引：按报错前缀定位——config 数值/结构问题回 ../joyarm-config-creator skill，")   # 指引一：config
        print("      URDF 内容不合规回 ../joyarm-urdf-creator skill，backend 实例化问题回 ../joyarm-backend-creator skill")   # 指引二/三
        return False
    return True                                               # with 正常退出 = 构造 + 收尾全部成功


def print_summary(results: dict) -> None:
    """打印四步校验汇总表（步骤 / 检查项 / 结果 / 不通过时回退 skill）。"""
    print("=" * 78)                                           # 大节分隔线
    print("四步校验汇总")
    print("=" * 78)
    rows = [                                                  # (步骤, 检查项, 结果键, 回退 skill)
        ("① config", "config/<型号>.yaml 存在且解析为非空 dict", "s1", "../joyarm-config-creator"),   # 第①步行
        ("② backend", "backend.name 命中 REGISTRY", "s2", "../joyarm-backend-creator"),   # 第②步行
        ("③ robot_model", "四路径按序存在", "s3", "../joyarm-urdf-creator"),   # 第③步行
        ("④ 构造自检", "joyArm(<型号>) 离线构造成功", "s4", "按报错定位回 config/urdf skill"),   # 第④步行
    ]
    print(f"  {disp_pad('步骤', 16)}{disp_pad('检查项', 42)}"
          f"{disp_pad('结果', 6)}不通过时回退")             # 表头（列宽与数据行一致）
    for step, item, key, back in rows:                        # 逐行按列宽对齐打印
        verdict = "PASS" if results[key] else "FAIL"          # 布尔结果转 PASS/FAIL 文本
        print(f"  {disp_pad(step, 16)}{disp_pad(item, 42)}{disp_pad(verdict, 6)}{back}")


def print_ladder(all_pass: bool, model: str) -> None:
    """打印验收阶梯的下一步建议：全 PASS 给阶梯命令，有 FAIL 给回退提醒。"""
    print("=" * 78)                                           # 大节分隔线
    print("验收阶梯 · 下一步建议")
    print("=" * 78)
    if not all_pass:                                          # 有 FAIL：先修正再验收，不走阶梯
        print("  ● 存在 FAIL 项：先按汇总表的回退 skill 指引修正对应件，再重跑本脚本确认全 PASS：")   # FAIL 提示
        print(f"      python .agents/skills/joyarm-arm-creator/scripts/arm_check.py --model {model}")   # 重跑命令
        return                                                # 提示完毕
    print("  ● 四步全部 PASS：整机装配成功，按以下阶梯继续验收（均在 joyarm_code/ 目录下执行）：")   # 阶梯标题
    for i, (hw, script, topic) in enumerate(LADDER, start=1):  # 逐级打印：离线在前、真机在后
        print(f"      {i}. [{hw}] python {disp_pad(script, 36)}← {topic}")   # 单级：硬件标记 + 命令 + 说明
    print("  ● 阶梯各级不通过时：离线两级回本脚本四步校验定位缺件；")   # 回退指引一
    print("      真机级多为 backend 联调或 config 参数问题，分别回 ../joyarm-backend-creator、")   # 回退指引二
    print("      ../joyarm-config-creator skill 修正后重试。")   # 回退指引三


def main() -> None:
    """主流程：解析型号 → 四步逐项体检 → 汇总表 + 阶梯建议 → 按结果设退出码。"""
    parser = argparse.ArgumentParser(description="JoyArm 整机集成四步校验（离线，无需硬件）")   # 参数解析器
    parser.add_argument("--model", default=DEFAULT_MODEL,     # 体检型号；缺省当前唯一可用型号
                        help=f"待体检的机械臂型号名（缺省 {DEFAULT_MODEL}）")
    args = parser.parse_args()                                # 解析命令行参数
    model = args.model                                        # 目标型号名（= config 文件名 = 模型目录名）

    print("=" * 78)                                           # 大节分隔线
    print(f"整机集成四步校验  ——  型号『{model}』")             # 大节标题（附目标型号）
    print("=" * 78)                                           # 大节分隔线
    print(f"  joyarm_core 库根：{CORE_DIR}")                  # 四件套中 config/robot_model/backend 的宿主目录
    avail = listArm()                                         # 列出可用型号（自打印一行；= 工厂失败消息同源）
    if model not in avail:                                    # 目标型号不在可用列表：先提示再继续逐项体检定位缺件
        print(f"  ⚠ 『{model}』不在可用型号列表——继续四步体检以定位缺失的件（可能仅个别件未就位）")   # 缺件提示
    results = {"s1": False, "s2": False, "s3": False, "s4": False}   # 四步结果登记表
    cfg = step1_config(model)                                 # 第①步：config 存在且可解析
    results["s1"] = cfg is not None                           # 返回非 None 即通过
    results["s2"] = step2_backend(cfg)                        # 第②步：backend 注册名命中
    results["s3"] = step3_robot_model(model)                  # 第③步：模型四路径齐全
    results["s4"] = step4_construct(model)                    # 第④步：joyArm 离线构造自检
    print_summary(results)                                    # 汇总表
    all_pass = all(results.values())                          # 四步全过才算整机体检通过
    print_ladder(all_pass, model)                             # 验收阶梯建议
    sys.exit(0 if all_pass else 1)                            # 退出码：全 PASS 为 0，任一 FAIL 为 1


if __name__ == "__main__":                                    # 直接运行入口（被导入时不执行）
    try:
        main()                                                # 执行主流程
    except KeyboardInterrupt:                                 # Ctrl+C 中断：离线脚本无硬件状态需清理
        print("\n检测到 Ctrl+C：四步校验已中断（离线脚本未触碰硬件），安全退出。")
        sys.exit(130)                                         # 惯例中断退出码
