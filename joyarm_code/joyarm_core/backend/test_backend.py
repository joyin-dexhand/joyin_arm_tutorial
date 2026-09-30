"""
==============================================================================
Backend 子类通用验收测试（上层视角）  joyarm_core/backend/test_backend.py
==============================================================================
【功能概要】
    从上层视角（只调 Backend 基类公开方法与属性，不触碰子类私有层）对**任意**
    backend 子类（关节电机 / 舵机 / 混合硬件 / 无 end 型号）做分类分项逐级验收：
      1. 离线段（init / gate 组）：构造、属性初值、限位解析、未连接门禁——无需硬件；
      2. 真机段（connect / read / mode / enable / write / motion / error /
         disconnect 组）：连接、读取、模式、使能、参数写入、运动发送、清错、断连。
    通用化设计（不内嵌任何子型专属知识）：
      - 能力三分法：PASS=行为符合基类契约；SKIP=能力缺失（如某型号不支持
        某模式、无参数读写——非缺陷）；FAIL=能力存在但违反契约；
      - 空族自动跳过：cfg 未配置 arm/end 关节的族，该族测试项自动 SKIP；
      - 参数键 / 末端离散动作名由子类定义，须经 --param / --action 显式告知。
    每项测试前打印测试项与目的，测试后打印 PASS/FAIL/SKIP 与证据；结尾输出
    机器可解析的 SUMMARY 行与人类可读结论（供开发者与 agent 共用）。
    安全机制：
      - 编号按风险递进（离线 → 连接 → 只读 → 模式 → 使能 → 写入 → 运动）；
      - 低危直接执行；中危交互确认（--yes 或非终端直接执行）；
      - 高危强制交互输入 yes（--yes 不放行；非终端自动 SKIP，agent 无法全自动执行高危项）；
      - set_zero（61/62）永久改写零位标定，排除出 all/online，仅显式编号执行。

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.sh | iex

    # 2. 创建虚拟环境并同步依赖
    cd ~/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（在 joyarm_code/ 目录下）
    python joyarm_code/joyarm_core/backend/test_backend.py --help                        # 完整用法与说明
    python joyarm_code/joyarm_core/backend/test_backend.py --list                        # 全部测试项清单
    python joyarm_code/joyarm_core/backend/test_backend.py --sub=dm --step=offline       # 离线段（缺省 --step 即 offline）
    python joyarm_code/joyarm_core/backend/test_backend.py --sub=dm --step=all           # 全部（不含 set_zero 61/62）
    python joyarm_code/joyarm_core/backend/test_backend.py --sub=dm --step=40            # 单项
    python joyarm_code/joyarm_core/backend/test_backend.py --sub=dm --step=init,20,30    # 逗号混选
    python joyarm_code/joyarm_core/backend/test_backend.py --sub=dm --step=online \
        --param=<参数键> --action=open,close                                 # 真机段（补齐参数/动作项）

【命令行参数】
    --sub     子型简称（如 dm、feetech、template）：自动导入 backend_<型号>.py 并
              寻找 Backend<型号> 类（大小写变体候选 + 唯一子类兜底），默认加载
              config/joyarm_<型号>.yaml 的 backend 段；缺省时仅 --help / --list 可用
    --step    数字=单项编号（如 40）；字符串=组名（init/gate/connect/read/mode/
              enable/write/motion/error/disconnect/offline/online/all）；
              逗号分隔可混选；缺省 offline（最安全一段）
    --config  指定 yaml 配置文件（默认 joyarm_core/config/joyarm_<sub>.yaml）
    --param   参数读写测试（34/60/63）的参数键（键表由子类定义；不给则相关项 SKIP）
    --action  send_action_end 测试（74）的离散动作名，逗号分隔（不给则该项 SKIP）
    --watch   item 25 实时状态刷新监视时长（秒，默认 10；每秒打印 arm/end 全部关节量表）
    --list    仅列出全部测试项（编号/组/风险/目的），不执行
    --yes     跳过低/中危交互确认（高危永不自动执行：仍需交互 yes，非终端自动 SKIP）
    退出码：全 PASS/SKIP=0；任一 FAIL=1；用法错误=2（便于 agent 判定）
==============================================================================
"""
from __future__ import annotations

import argparse
import copy
import enum
import importlib
import logging
import sys
import time
from pathlib import Path

# 未安装 joyarm_core 时兜底（本脚本可直接 python 运行）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import yaml

from joyarm_core.backend import Backend
from joyarm_core.utils.types import ControlMode

# ---------------- 安全常量（真机运动测试的小幅参数，子型中立） ----------------
MIT_AMP = 0.02        # MIT 测试目标偏移 rad
POS_AMP = 0.05        # 位置测试目标偏移 rad
END_AMP = 0.05        # 末端位置测试目标偏移 rad
SAFE_KP = 5.0         # MIT 测试用保守位置增益
SAFE_KD = 0.5         # MIT 测试用保守速度阻尼
SAFE_VLIM = 0.5       # 位置测试限速 rad/s
SAFE_FLIM = 0.5       # 位置测试归一化力矩电流上限（0~1）
VEL_TEST = 0.2        # 速度测试目标 rad/s
VEL_DUR = 0.5         # 速度测试持续时间 s
LIMIT_MARGIN = 0.01   # 目标距硬限位的安全余量 rad
SETTLE = 0.5          # 发送后观测等待 s

# 子类未实现内核的桩消息标记（backend_template.py 桩约定，基类转 warn 后保留原文）
CAP_MISSING = "功能缺失"
CAP_MISSING2 = "未实现"

# 基类 warn 消息文本标记（注意：限频通道名不出现在消息文本中，须按消息内容匹配；
# 且须选取在消息中唯一的子串——如"存在空值"同时出现在发送空值门禁与状态数据异常两种
# 消息里，须用发送门禁独有的尾部"，本次不发送"区分，防刷新线程的并发 warn 误判）
GATE_CONN = "未连接或状态未知"      # 连接门禁（_require_connected 唯一）
GATE_MODE = "≠ 所需"               # 模式门禁（_require_mode 唯一）
GATE_TYPE = "非 ControlMode 枚举"   # set_mode 类型门禁（唯一）
KERN_EXC = "内核异常"               # 子类内核异常（基类 _call 唯一）
EMPTY_VAL = "，本次不发送"          # 发送前空值检查（_require_present 唯一尾部；区别于状态数据异常的"本轮状态跳过更新"）
DIM_SEND = "指令维度"               # 发送维度校验（唯一）
CLIP_Q = "q 指令越限"               # 越限就近裁剪（send_mit/send_position 共用文本）

ALL_MODES = (ControlMode.MIT, ControlMode.POSITION, ControlMode.VELOCITY)


# ---------------- 日志捕获（joyarm_core 全包，含各子类 logger） ----------------
class _Cap(logging.Handler):
    """收集 joyarm_core 包内全部日志消息，供测试证据与能力缺失判定。"""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.msgs: list[str] = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


_cap = _Cap()
_pkg_logger = logging.getLogger("joyarm_core")
_pkg_logger.addHandler(_cap)
_pkg_logger.setLevel(logging.INFO)


def _mark() -> int:
    """记录当前日志水位（调用前打点，调用后取增量）。"""
    return len(_cap.msgs)


def _since(i0: int) -> list[str]:
    return _cap.msgs[i0:]


def _recent(msgs: list[str], k: int = 2) -> str:
    """最近 k 条日志作为证据摘要。"""
    return " | ".join(msgs[-k:]) if msgs else "无"


class _UsageError(Exception):
    """用法错误（退出码 2）。"""


# ---------------- 会话（进程内复用实例与状态，前置自动补齐） ----------------
class Session:
    """持有子类、cfg 与会话实例；gate 组等需要的未连接新实例经 fresh() 另建。"""

    def __init__(self, cls: type, cfg: dict, args: argparse.Namespace, cfg_path: Path):
        self.cls = cls
        self.cfg = cfg
        self.args = args
        self.cfg_path = cfg_path
        self.b: Backend | None = None

    def backend(self) -> Backend:
        if self.b is None:
            self.b = self.cls(copy.deepcopy(self.cfg))
        return self.b

    def fresh(self) -> Backend:
        """全新未连接实例（gate / 裁剪 / 维度探测用），用毕由调用方 close。"""
        return self.cls(copy.deepcopy(self.cfg))


# ---------------- 通用小工具 ----------------
def _n_family_cfg(cfg: dict, fam: str) -> int:
    return len(((cfg.get(fam) or {}).get("joints")) or [])


def _cfg_default_mode(b: Backend, fam: str) -> ControlMode:
    """按基类同规则解析 cfg 默认模式（缺省/非法回退 MIT）。"""
    raw = (b.cfg.get(fam) or {}).get("default_mode")
    if raw is None:
        return ControlMode.MIT
    try:
        return ControlMode(str(raw).lower())
    except ValueError:
        return ControlMode.MIT


def _safe_mid(lim, n: int) -> np.ndarray:
    """限位中点（限位未配置 ±inf 的关节取 0）。"""
    if lim is None or n == 0:
        return np.zeros(n)
    lo = np.asarray(lim.q_min, dtype=float)
    hi = np.asarray(lim.q_max, dtype=float)
    return np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2.0, 0.0)


def _clip_target(lim, q: np.ndarray, amp: float) -> np.ndarray:
    """自当前 q 偏移 amp 并裁剪进硬限位（留 LIMIT_MARGIN 余量；未配置限位只限偏移量）。"""
    lo = np.asarray(lim.q_min, dtype=float)
    hi = np.asarray(lim.q_max, dtype=float)
    lo = np.where(np.isfinite(lo), lo + LIMIT_MARGIN, q - amp)
    hi = np.where(np.isfinite(hi), hi - LIMIT_MARGIN, q + amp)
    return np.clip(q + amp, lo, hi)


def _cap_missing(msgs: list[str]) -> bool:
    return any((CAP_MISSING in m) or (CAP_MISSING2 in m) for m in msgs)


def _classify_send(msgs: list[str]) -> str:
    """发送类测试的三分判定：桩/未实现→SKIP；门禁/内核异常→FAIL；干净→PASS。

    只匹配与发送路径强相关的消息文本（见常量注释）——刷新线程并发的状态类
    warn（读状态失败/数据异常等）不在此列，不会被误判。
    """
    if _cap_missing(msgs):
        return "SKIP"
    bad = (GATE_CONN, GATE_MODE, KERN_EXC, EMPTY_VAL, DIM_SEND, "非数值", "一维序列")
    if any(any(mk in m for mk in bad) for m in msgs):
        return "FAIL"
    return "PASS"


def _current_q(b: Backend, fam: str):
    """当前实测位置，返回 (q, 来源说明)；状态不可读返回 (None, 原因)。

    运动测试须自实际位置小幅出发——状态不可读时盲发目标（如限位中点）可能
    与实际位置相差整段行程，小幅测试变大幅运动，故不可读即由调用方 SKIP。
    """
    st = getattr(b, f"get_state_{fam}")()
    if st is not None and st.q is not None:
        return np.asarray(st.q, dtype=float), "实测"
    return None, "状态不可读（get_state 返回 None 或缺 q）"


def _trunc(text: str, k: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= k else text[:k] + "…"


def _both_fams(s: Session) -> list[str]:
    """非空族列表（cfg 未配置关节的族不参与双族测试）。"""
    return [f for f in ("arm", "end") if _n_family_cfg(s.cfg, f) > 0]


def _aggregate(results: list[tuple[str, str, str]]) -> tuple[str, str]:
    """逐族结果聚合：任一 FAIL→FAIL；否则任一 PASS→PASS；全 SKIP/空→SKIP。

    results: [(族, 判定, 证据), ...]；证据按「arm: …；end: …」拼接。
    """
    if not results:
        return "SKIP", "arm/end 均为空族"
    if any(v == "FAIL" for _, v, _ in results):
        verdict = "FAIL"
    elif any(v == "PASS" for _, v, _ in results):
        verdict = "PASS"
    else:
        verdict = "SKIP"
    return verdict, "；".join(f"{fam}: {ev}" for fam, _, ev in results)


def _clear_throttle(b: Backend):
    """双族循环内的限频预清除：防第二族的同通道 warn 被 0.5s 限频吞掉而误判。"""
    b._warn_last.clear()


# ---------------- 前置自动补齐与风险确认 ----------------
def _prefill_fams(fam: str) -> tuple:
    return ("arm", "end") if fam == "both" else (fam,)


_RISK_ORDER = {"低": 0, "中": 1, "高": 2, "极高": 3}


def _plan_prefill(s: Session, needs: list) -> list[tuple[str, str]]:
    """干跑：计算将补齐哪些前置（不动状态），返回 [(动作描述, 风险)]。

    用于确认等级计算（前置风险并入该项确认）与确认文案。
    """
    plans: list[tuple[str, str]] = []
    b = s.b  # 可能尚未构造（视为全部前置缺失）
    for nd in needs:
        kind = nd[0]
        for fam in _prefill_fams(nd[1]):
            if _n_family_cfg(s.cfg, fam) == 0:
                continue
            conn = getattr(b, f"is_connected_{fam}", None) if b is not None else None
            if kind == "connected":
                if conn is not True:
                    plans.append((f"connect_{fam}", "中"))
            elif kind == "mode":
                mode = nd[2]
                mode_cur = getattr(b, f"mode_{fam}", None) if b is not None else None
                if mode_cur != mode:
                    if conn is not True:
                        plans.append((f"connect_{fam}", "中"))
                    plans.append((f"set_mode_{fam}({mode.name})", "中"))
            elif kind == "enabled":
                abled = getattr(b, f"is_abled_{fam}", None) if b is not None else None
                if abled is not True:
                    plans.append((f"enable_{fam}", "高"))
    return plans


def _prefill(s: Session, needs: list) -> tuple[str, list[str], str]:
    """按声明顺序补齐缺失前置（已就绪则跳过，不重复连接）。

    低/中危前置（connect/set_mode）静默补齐；高危前置（enable）由框架并入该项确认。
    无前置的项不构造会话实例（纯离线单项不应留下需收尾的硬件对象）。
    返回 (状态, 已执行的补齐动作描述, 原因)，状态：
      "ok"   全部就绪；
      "skip" 所需模式不可用——型号能力缺失（部分型号仅支持部分模式），该项应判 SKIP 非缺陷；
      "fail" 连接/使能等必实现核心前置未达成，该项判 FAIL。
    """
    if not needs:
        return "ok", [], ""
    b = s.backend()
    actions: list[str] = []
    for nd in needs:
        kind = nd[0]
        for fam in _prefill_fams(nd[1]):
            if _n_family_cfg(s.cfg, fam) == 0:
                continue
            if kind == "connected":
                if getattr(b, f"is_connected_{fam}") is True:
                    continue
                getattr(b, f"connect_{fam}")()
                actions.append(f"connect_{fam}")
                if getattr(b, f"is_connected_{fam}") is not True:
                    return "fail", actions, f"connect_{fam} 未成功（连接为必实现核心能力，详见上方 warn）"
            elif kind == "mode":
                mode = nd[2]
                if getattr(b, f"mode_{fam}") == mode:
                    continue
                if getattr(b, f"is_connected_{fam}") is not True:  # 防御：先保连接
                    getattr(b, f"connect_{fam}")()
                    actions.append(f"connect_{fam}")
                    if getattr(b, f"is_connected_{fam}") is not True:
                        return "fail", actions, f"connect_{fam} 未成功（连接为必实现核心能力，详见上方 warn）"
                ret = getattr(b, f"set_mode_{fam}")(mode)
                actions.append(f"set_mode_{fam}({mode.name})")
                if ret != 1:
                    return "skip", actions, (f"{fam} 族所需模式 {mode.name} 不可用"
                                             f"（set_mode 返回 0——型号能力缺失或硬件拒绝，非缺陷；详见上方 warn）")
            elif kind == "enabled":
                if getattr(b, f"is_abled_{fam}") is True:
                    continue
                if getattr(b, f"is_connected_{fam}") is not True:
                    return "fail", actions, "连接缺失"
                ret = getattr(b, f"enable_{fam}")()
                actions.append(f"enable_{fam}")
                if ret != 1:
                    return "fail", actions, f"enable_{fam} 未成功（使能为必实现核心能力，详见上方 warn）"
    return "ok", actions, ""


def _describe_needs(needs: list) -> str:
    parts = []
    for nd in needs:
        fams = "/".join(_prefill_fams(nd[1]))
        if nd[0] == "connected":
            parts.append(f"已连接（{fams}）")
        elif nd[0] == "enabled":
            parts.append(f"已使能（{fams}）")
        else:
            parts.append(f"{nd[2].name} 模式（{fams}）")
    return "、".join(parts)


def _confirm(level: str, args: argparse.Namespace) -> str:
    """风险确认：返回 run/skip/quit。

    低危直接执行；中危交互回车（--yes 或非终端直接执行）；
    高危/极高强制交互 yes（--yes 不放行；非终端自动 skip，不卡死）。
    """
    if level == "低":
        return "run"
    if not sys.stdin.isatty():
        return "skip" if level in ("高", "极高") else "run"
    if level == "中" and args.yes:
        return "run"
    if level in ("高", "极高"):
        tip = ("⚠⚠ 极高风险：将永久改写电机零位标定（不可逆）！" if level == "极高" else
               "⚠ 高风险：电机将上力/运动！确认工作空间已清空、急停可用、有人监护。")
        ans = input(f"{tip}输入 yes 执行（回车/s=跳过  q=退出）: ").strip().lower()
    else:
        ans = input("[回车]=执行  s=跳过  q=退出: ").strip().lower()
    if ans in ("q", "quit", "退出"):
        return "quit"
    if ans in ("s", "skip", "跳过"):
        return "skip"
    if level in ("高", "极高"):
        return "run" if ans in ("yes", "y") else "skip"
    return "run"


_hardware_banner_done = False


def _hardware_banner_once():
    global _hardware_banner_done
    if not _hardware_banner_done:
        _hardware_banner_done = True
        print("─" * 68)
        print("⚠ 进入真机测试段：请确认机械臂工作空间已清空、急停可用、有人监护")


# ============================================================
# 测试项实现（fn(s) -> (verdict, evidence)，verdict ∈ PASS/FAIL/SKIP）
# ============================================================

# ---------------- init 组（离线：构造与属性初值） ----------------
def _it1(s: Session):
    cls = s.cls
    ok = issubclass(cls, Backend)
    ev = f"模块 {cls.__module__}，类 {cls.__name__}，继承 Backend={ok}"
    return ("PASS" if ok else "FAIL"), ev


def _it2(s: Session):
    b = s.backend()  # 契约：构造不做任何总线 I/O
    ev = (f"完整 cfg 构造成功（无总线 I/O）：name={b.name!r}，"
          f"arm={b.n_joints_arm} 关节，end={b.n_joints_end} 关节")
    return "PASS", ev


def _it3(s: Session):
    b = s.cls({})  # 契约：空 cfg 可构造
    try:
        ok = (b.n_joints_arm == 0 and b.n_joints_end == 0
              and b.joint_limits_arm is None and b.joint_limits_end is None)
        ev = (f"空 cfg 构造成功：n_joints_arm={b.n_joints_arm}，n_joints_end={b.n_joints_end}，"
              f"joint_limits 均为 None={b.joint_limits_arm is None and b.joint_limits_end is None}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        b.close()


def _it4(s: Session):
    cfg = copy.deepcopy(s.cfg)
    cfg["state_refresh_hz"] = -1.0  # 非法：应回退默认并 warn
    i0 = _mark()
    b = s.cls(cfg)
    try:
        hits = [m for m in _since(i0) if "state_refresh_hz" in m and "沿用默认" in m]
        ev = (f"非正数 state_refresh_hz 构造未崩溃；回退 warn："
              f"{hits[0] if hits else '未捕获（契约要求 warn）'}")
        return ("PASS" if hits else "FAIL"), ev
    finally:
        b.close()


def _it5(s: Session):
    b = s.fresh()
    try:
        fails = []
        for fam in ("arm", "end"):
            if getattr(b, f"n_joints_{fam}") != _n_family_cfg(s.cfg, fam):
                fails.append(f"{fam} 关节数 {getattr(b, f'n_joints_{fam}')} ≠ cfg {_n_family_cfg(s.cfg, fam)}")
            for attr in (f"is_connected_{fam}", f"is_abled_{fam}", f"mode_{fam}"):
                if getattr(b, attr) is not None:
                    fails.append(f"{attr} 初值非 None")
        exp_name = str(s.cfg.get("name", ""))
        if b.name != exp_name:
            fails.append(f"name={b.name!r} ≠ cfg {exp_name!r}")
        ev = (f"name={b.name!r}，arm={b.n_joints_arm} 关节，end={b.n_joints_end} 关节，"
              f"is_connected*/is_abled*/mode* 初值均为 None")
        return ("FAIL" if fails else "PASS"), ev + (f"；异常：{fails}" if fails else "")
    finally:
        b.close()


def _it6(s: Session):
    b = s.fresh()
    try:
        fails = []
        for fam in ("arm", "end"):
            joints = (s.cfg.get(fam) or {}).get("joints") or []
            lim = getattr(b, f"joint_limits_{fam}")
            if not joints:
                if lim is not None:
                    fails.append(f"{fam} 空族 joint_limits 应为 None")
                continue
            if lim is None:
                fails.append(f"{fam} joint_limits 为 None（有关节却未解析）")
                continue
            for key, default in (("q_min", -np.inf), ("q_max", np.inf),
                                 ("dq_max", np.inf), ("tau_max", np.inf)):
                exp = np.array([float(j.get(key, default)) for j in joints])
                got = np.asarray(getattr(lim, key), dtype=float)
                if not np.allclose(got, exp):
                    fails.append(f"{fam}.{key}: {np.round(got, 3).tolist()} ≠ cfg {np.round(exp, 3).tolist()}")
        ev = (f"joint_limits_arm q_min={np.round(np.asarray(b.joint_limits_arm.q_min), 3).tolist() if b.joint_limits_arm else None}…"
              f"；end q_min={np.round(np.asarray(b.joint_limits_end.q_min), 3).tolist() if b.joint_limits_end else None}…（四键全量核对）")
        return ("FAIL" if fails else "PASS"), ev + (f"；异常：{fails}" if fails else "")
    finally:
        b.close()


def _it7(s: Session):
    b = s.fresh()
    try:
        fails = []
        for fam in ("arm", "end"):
            st = getattr(b, f"joint_state_{fam}")
            n = getattr(b, f"n_joints_{fam}")
            if st.t != 0.0:
                fails.append(f"{fam}.t={st.t} ≠ 0")
            if st.q.shape != (n,):
                fails.append(f"{fam}.q shape={st.q.shape}")
            elif n and not np.isnan(st.q).all():
                fails.append(f"{fam}.q 初值非 NaN")
            if n and not np.all(np.asarray(st.error) == -1):
                fails.append(f"{fam}.error 初值非 -1")
        ev = "joint_state 初值：t=0、q=NaN、error=-1（未读取约定）"
        return ("FAIL" if fails else "PASS"), ev + (f"；异常：{fails}" if fails else "")
    finally:
        b.close()


# ---------------- gate 组（离线：未连接门禁与安全降级） ----------------
def _gate_both(s: Session, checker) -> tuple[str, str]:
    """在 arm/end 每个非空族上以**各自独立的**全新未连接实例执行 checker(b, fam)。

    每族单独建实例：基类 warn 按通道限频（默认 0.5s），同实例上连续两族调用
    会被限频吞掉第二条 warn，导致检查误判。
    """
    lines, fails, tested = [], [], []
    for fam in ("arm", "end"):
        if _n_family_cfg(s.cfg, fam) == 0:
            continue
        b = s.fresh()
        try:
            tested.append(fam)
            ok, ev = checker(b, fam)
            lines.append(f"{fam}: {'OK' if ok else 'FAIL'}（{ev}）")
            if not ok:
                fails.append(fam)
        finally:
            b.close()
    if not tested:
        return "SKIP", "arm/end 均为空族"
    return ("FAIL" if fails else "PASS"), "；".join(lines)


def _it10(s: Session):
    def chk(b, fam):
        i0 = _mark()
        ret = getattr(b, f"get_state_{fam}")()
        gate = any(GATE_CONN in m for m in _since(i0))
        return ret is None and gate, f"返回 {ret}，门禁 warn={gate}"
    return _gate_both(s, chk)


def _it11(s: Session):
    def chk(b, fam):
        i0 = _mark()
        ret = getattr(b, f"get_mode_{fam}")()
        gate = any(GATE_CONN in m for m in _since(i0))
        return ret is None and gate, f"返回 {ret}，门禁 warn={gate}"
    return _gate_both(s, chk)


def _it12(s: Session):
    def chk(b, fam):
        key = s.args.param or "__probe__"  # 门禁与键名无关
        i0 = _mark()
        ret = getattr(b, f"read_param_{fam}")(key)
        gate = any(GATE_CONN in m for m in _since(i0))
        return ret is None and gate, f"read_param({key!r}) 返回 {ret}，门禁 warn={gate}"
    return _gate_both(s, chk)


def _it13(s: Session):
    def chk(b, fam):
        i0 = _mark()
        ret = getattr(b, f"set_mode_{fam}")(ControlMode.MIT)
        gate = any(GATE_CONN in m for m in _since(i0))
        return ret == 0 and gate, f"返回 {ret}，门禁 warn={gate}"
    return _gate_both(s, chk)


def _it14(s: Session):
    def chk(b, fam):
        n = getattr(b, f"n_joints_{fam}")
        q0 = _safe_mid(getattr(b, f"joint_limits_{fam}"), n)
        i0 = _mark()
        getattr(b, f"send_mit_{fam}")(np.zeros(n), q0, np.zeros(n), np.zeros(n), np.zeros(n))
        gate = any(GATE_CONN in m for m in _since(i0))
        return gate, f"合法值指令被连接门禁拦截={gate}（无帧发出）"
    return _gate_both(s, chk)


def _it15(s: Session):
    def chk(b, fam):
        r1 = getattr(b, f"enable_{fam}")()
        r2 = getattr(b, f"disable_{fam}")()
        return r1 == 0 and r2 == 0, f"enable→{r1}，disable→{r2}（均被门禁拦为 0，未崩溃）"
    return _gate_both(s, chk)


def _it16(s: Session):
    def chk(b, fam):
        e = getattr(b, f"get_error_{fam}")()
        c = getattr(b, f"check_error_{fam}")()
        return e is None and c is False, f"get_error→{e}，check_error→{c}"
    return _gate_both(s, chk)


def _it17(s: Session):
    def chk(b, fam):
        n = getattr(b, f"n_joints_{fam}")
        key = s.args.param or "__probe__"
        w = getattr(b, f"write_param_{fam}")(key, [0.0] * n)
        z = getattr(b, f"set_zero_{fam}")()
        cl = getattr(b, f"clear_error_{fam}")()
        return w == 0 and z == 0 and cl is False, \
            f"write_param→{w}，set_zero→{z}，clear_error→{cl}（安全失败不崩溃）"
    return _gate_both(s, chk)


def _it18(s: Session):
    b = s.fresh()
    try:
        b.close()
        b.close()
        return "PASS", "close() 连续两次均无异常（幂等；刷新定时器已停止）"
    except Exception as e:
        return "FAIL", f"close() 异常：{e!r}"


# ---------------- connect 组（真机：连接与断连） ----------------
def _connect_assert(b: Backend, fam: str, i0: int) -> tuple[str, str]:
    conn = getattr(b, f"is_connected_{fam}")
    if conn is not True:
        return "FAIL", (f"connect_{fam} 后 is_connected={conn}（连接失败——核心能力）；"
                        f"warn：{_recent(_since(i0))}")
    exp = _cfg_default_mode(b, fam)
    mode = getattr(b, f"mode_{fam}")
    if mode != exp:
        return "FAIL", (f"已连接，但默认模式未回填：mode_{fam}={mode} ≠ cfg default_mode={exp.name}"
                        f"（检查 cfg default_mode 与硬件能力是否匹配）")
    return "PASS", f"is_connected_{fam}=True；mode_{fam}={exp.name}（默认模式乐观回填）"


def _connect_item(fam: str):
    def fn(s: Session):
        b = s.backend()
        i0 = _mark()
        getattr(b, f"connect_{fam}")()
        return _connect_assert(b, fam, i0)
    return fn


def _it22(s: Session):
    b = s.backend()
    lines, fails = [], []
    for fam in ("arm", "end"):
        if getattr(b, f"n_joints_{fam}") == 0:
            continue
        i0 = _mark()
        getattr(b, f"connect_{fam}")()  # 已连接状态下重复连接
        conn = getattr(b, f"is_connected_{fam}")
        lines.append(f"{fam}: is_connected={conn}")
        if conn is not True:
            fails.append(f"{fam} 重复连接后 is_connected={conn}（{_recent(_since(i0), 1)}）")
    if not lines:
        return "SKIP", "arm/end 均为空族"
    return ("FAIL" if fails else "PASS"), "重复 connect 幂等：" + "；".join(lines)


def _it23(s: Session):
    b = s.backend()
    results = []
    for fam in _both_fams(s):  # 基类先尽力失能再断连（安全方向）
        _clear_throttle(b)
        i0 = _mark()
        getattr(b, f"disconnect_{fam}")()
        conn = getattr(b, f"is_connected_{fam}")
        mode = getattr(b, f"mode_{fam}")
        abled = getattr(b, f"is_abled_{fam}")
        ok = conn is False and mode is None and abled is None
        results.append((fam, "PASS" if ok else "FAIL",
                        f"is_connected={conn}，mode={mode}（缓存清空），is_abled={abled}"))
    return _aggregate(results)


def _it24(s: Session):
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        _clear_throttle(b)
        getattr(b, f"disconnect_{fam}")()
        if getattr(b, f"is_connected_{fam}") is not False:
            results.append((fam, "FAIL",
                            f"断连后 is_connected={getattr(b, f'is_connected_{fam}')}（应 False）"))
            continue
        i0 = _mark()
        getattr(b, f"connect_{fam}")()
        results.append((fam, *_connect_assert(b, fam, i0)))
    return _aggregate(results)


def _it25(s: Session):
    """实时状态刷新监视：连接后每秒读**被动快照**（不主动 get_state），
    打印 arm/end 全部关节量表格——测试点正是基类低频刷新线程在无主动读取时
    持续更新快照；Ctrl+C 可提前结束（按已采样判定）。"""
    b = s.backend()
    fams = _both_fams(s)
    secs = max(1, int(s.args.watch))
    names = {fam: [j.get("name", f"joint{i}")
                   for i, j in enumerate((s.cfg.get(fam) or {}).get("joints") or [])]
             for fam in fams}
    header = (f"{'族.关节':<14}{'q(rad)':>10}{'dq(rad/s)':>11}{'tau(N·m)':>10}"
              f"{'T_mos(℃)':>10}{'T_rot(℃)':>10}{'error':>7}{'快照龄(ms)':>10}")
    ages: dict[str, list[float]] = {fam: [] for fam in fams}
    print(f"── 实时状态刷新监视（{secs}s，每秒采样被动快照 joint_state_*；Ctrl+C 提前结束）")
    try:
        for sec in range(1, secs + 1):
            print(f"── {sec}/{secs}s " + "─" * 52)
            print(header)
            now = time.time()
            for fam in fams:
                st = getattr(b, f"joint_state_{fam}")  # 被动读快照（不触发主动读帧）
                if st.t > 0:
                    ages[fam].append(now - st.t)
                age_ms = f"{int(max(0.0, now - st.t) * 1000)}" if st.t > 0 else "—"
                for i, jn in enumerate(names[fam]):
                    cells = [f"{fam}.{jn}"]
                    for arr, nd in ((st.q, 4), (st.dq, 4), (st.tau, 4),
                                    (st.temp_mos, 1), (st.temp_rotor, 1)):
                        cells.append("—" if arr is None
                                     else f"{np.asarray(arr, dtype=float)[i]:.{nd}f}")
                    cells.append("—" if st.error is None
                                 else str(int(np.asarray(st.error)[i])))
                    cells.append(age_ms)
                    print(f"{cells[0]:<14}{cells[1]:>10}{cells[2]:>11}{cells[3]:>10}"
                          f"{cells[4]:>10}{cells[5]:>10}{cells[6]:>7}{cells[7]:>10}")
            if sec < secs:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("（用户中断监视，按已采样判定）")
    results = []
    for fam in fams:
        if not ages[fam]:
            results.append((fam, "FAIL",
                            f"监视期间快照 t 恒为 0——低频刷新线程未更新状态（或状态读取失败，见上方 warn）"))
        elif max(ages[fam]) < 0.5:
            results.append((fam, "PASS",
                            f"{len(ages[fam])} 次采样快照龄 {int(min(ages[fam]) * 1000)}"
                            f"~{int(max(ages[fam]) * 1000)}ms"))
        else:
            results.append((fam, "FAIL",
                            f"快照龄最大 {int(max(ages[fam]) * 1000)}ms ≥500ms（刷新存在停滞，见上方 warn）"))
    return _aggregate(results)


# ---------------- read 组（真机：只读） ----------------
def _state_item(fam: str):
    def fn(s: Session):
        b = s.backend()
        i0 = _mark()
        st1 = getattr(b, f"get_state_{fam}")()
        if st1 is None:
            return "SKIP", f"get_state_{fam} 返回 None（读失败/数据异常——见上方 warn）"
        time.sleep(0.05)
        st2 = getattr(b, f"get_state_{fam}")()
        n = getattr(b, f"n_joints_{fam}")
        fails = []
        for name in ("q", "dq", "tau", "temp_mos", "temp_rotor", "error"):
            v = getattr(st1, name)
            if v is None:  # 硬件不提供的量 → 合法缺席
                continue
            arr = np.asarray(v, dtype=float)
            if arr.shape != (n,):
                fails.append(f"{name} shape={arr.shape} ≠ ({n},)")
            elif name != "error" and not np.isfinite(arr).all():
                fails.append(f"{name} 含非有限值")
        if st1.t <= 0:
            fails.append(f"t={st1.t}")
        if st2 is not None and st2.t <= st1.t:
            fails.append("两次读取 t 未更新")
        notes = []
        lim = getattr(b, f"joint_limits_{fam}")
        if st1.q is not None and lim is not None:
            q = np.asarray(st1.q, dtype=float)
            if ((q < np.asarray(lim.q_min, dtype=float) - 1e-6).any()
                    or (q > np.asarray(lim.q_max, dtype=float) + 1e-6).any()):
                notes.append("部分关节位置超出硬限位（硬件状态，非 backend 缺陷）")
        present = [name for name in ("q", "dq", "tau", "temp_mos", "temp_rotor", "error")
                   if getattr(st1, name) is not None]
        t2 = f"{st2.t:.3f}" if st2 is not None else "None"
        ev = (f"JointState 在场字段 {present} 均为 (n,) 且有限；t={st1.t:.3f}→{t2}（更新）"
              + (f"；{notes[0]}" if notes else ""))
        return ("FAIL" if fails else "PASS"), ev + (f"；异常：{fails}" if fails else "")
    return fn


def _it30(s: Session):
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        i0 = _mark()
        ret = getattr(b, f"get_mode_{fam}")()
        if isinstance(ret, ControlMode):
            cache = getattr(b, f"mode_{fam}")
            results.append((fam, "PASS",
                            f"读回 {ret.name}；mode_{fam}={cache.name if cache else None}"))
        else:
            results.append((fam, "SKIP",
                            f"返回 None（读失败或无模式回读——部分型号经本地镜像实现回读；warn：{_recent(_since(i0), 1)}）"))
    return _aggregate(results)


def _it33(s: Session):
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        st1 = getattr(b, f"get_state_{fam}")()
        if st1 is None:
            results.append((fam, "SKIP", "状态不可读，快照语义无法验证"))
            continue
        old = getattr(b, f"joint_state_{fam}")
        time.sleep(0.05)
        st2 = getattr(b, f"get_state_{fam}")()
        new = getattr(b, f"joint_state_{fam}")
        if st2 is None:
            results.append((fam, "SKIP", "第二次读取失败，快照语义无法完整验证"))
            continue
        ok = (new is not old) and (old is st1) and (new is st2)
        results.append((fam, "PASS" if ok else "FAIL",
                        f"换入新对象={new is not old}；属性与上次返回同引用={old is st1}；"
                        f"与本次同引用={new is st2}（旧引用冻结为快照）"))
    return _aggregate(results)


def _it34(s: Session):
    key = s.args.param
    if not key:
        return "SKIP", "未提供 --param（参数键由子类定义，须显式指定）"
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        _clear_throttle(b)
        i0 = _mark()
        vals = getattr(b, f"read_param_{fam}")(key)
        if vals is None:
            results.append((fam, "SKIP",
                            f"read_param({key!r}) 返回 None（键对该族无效或未实现参数读取）"))
            continue
        n = getattr(b, f"n_joints_{fam}")
        if len(vals) != n or not all(np.isscalar(v) for v in vals):
            results.append((fam, "FAIL", f"返回非逐关节标量列表：{vals!r}（长度 {len(vals)} ≠ {n}）"))
            continue
        results.append((fam, "PASS", f"read_param({key!r}) = {vals}"))
    return _aggregate(results)


def _it35(s: Session):
    """错误码读取与检查：逐族完整执行 get_error（逐关节码）与 check_error（bool）并都输出结果，
    断言两者一致性（全 0/1→True；读到 None→check 应 False）。"""
    b = s.backend()
    time.sleep(0.3)  # 等低频刷新更新状态快照
    results = []
    for fam in _both_fams(s):
        _clear_throttle(b)
        i0 = _mark()
        errs = getattr(b, f"get_error_{fam}")()
        ret = getattr(b, f"check_error_{fam}")()
        if errs is None:
            # 契约：读不到状态码（快照过期或 error 段缺失）时 get 返回 None、check 应 False
            ok = (ret is False)
            ev = (f"get_error 返回 None（快照过期或 error 段缺失——部分硬件不提供状态码）；"
                  f"check_error={ret}（契约：读不到应 False，{'符合' if ok else '违反'}）")
            results.append((fam, "PASS" if ok else "FAIL", ev))
            continue
        fault = any(e >= 2 for e in errs)
        expect = not fault
        ok = isinstance(ret, bool) and (ret == expect)
        ev = f"get_error={errs}（0=失能、1=使能，均正常）；check_error={ret}"
        if fault:
            ev += "；⚠ 存在故障码（硬件状态，非 backend 缺陷）"
        ev += f"（一致性={'OK' if ok else '矛盾'}）"
        results.append((fam, "PASS" if ok else "FAIL", ev))
    return _aggregate(results)


# ---------------- mode 组（真机：模式设置） ----------------
def _mode_name(m):
    """模式显示名（None=无回读能力）。"""
    return m.name if isinstance(m, ControlMode) else "None"


def _modes_item(fam: str):
    """三模式循环设置：逐模式「先读（设前真值）→ 设 → 再读（设后真值）」，回读验证。"""
    def fn(s: Session):
        b = s.backend()
        lines, fails, any_ok = [], [], False
        for mode in ALL_MODES:
            before = getattr(b, f"get_mode_{fam}")()  # 先读
            _clear_throttle(b)
            i0 = _mark()
            ret = getattr(b, f"set_mode_{fam}")(mode)  # 再设
            cache = getattr(b, f"mode_{fam}")
            after = getattr(b, f"get_mode_{fam}")()  # 再读
            if ret == 1:
                any_ok = True
                bad = []
                if cache != mode:
                    bad.append(f"缓存={_mode_name(cache)}")
                if after is not None and after != mode:
                    bad.append(f"读回={_mode_name(after)}")
                if bad:
                    fails.append(f"set {mode.name}：" + "、".join(bad))
                lines.append(f"{mode.name}: 先读{_mode_name(before) if before is not None else 'None'}"
                             f"→设(1)→再读{_mode_name(after) if after is not None else 'None'}")
            else:
                lines.append(f"{mode.name}: 不可用（先读{_mode_name(before) if before is not None else 'None'}；"
                             f"{_recent(_since(i0), 1)}）")
        if not any_ok:
            return "SKIP", f"{fam} 族三种模式均设置失败（能力缺失或连接异常）；" + "；".join(lines)
        return ("FAIL" if fails else "PASS"), "；".join(lines) + (f"；异常：{fails}" if fails else "")
    return fn


class _UndefMode(enum.Enum):
    """未被 ControlMode 定义的模式（模拟新型号可能具备的额外模式，如力矩/步进）。"""
    UNDEFINED = "undefined"


def _it42(s: Session):
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        parts, ok_all = [], True
        for label, bad in (("字符串入参", "mit"), ("未定义模式", _UndefMode.UNDEFINED)):
            _clear_throttle(b)
            i0 = _mark()
            ret = getattr(b, f"set_mode_{fam}")(bad)
            hit = any(GATE_TYPE in m for m in _since(i0))
            ok = ret == 0 and hit
            ok_all = ok_all and ok
            parts.append(f"{label}→返回{ret}/门禁warn={hit}")
        results.append((fam, "PASS" if ok_all else "FAIL",
                        "、".join(parts) + "（均不达内核）"))
    return _aggregate(results)


def _it43(s: Session):
    b = s.backend()
    lines, fails = [], []
    for fam in ("arm", "end"):
        if getattr(b, f"n_joints_{fam}") == 0:
            continue
        exp = _cfg_default_mode(b, fam)
        ret = getattr(b, f"set_mode_{fam}")(exp)
        if ret != 1:
            fails.append(f"set_mode_{fam}({exp.name}) 返回 {ret}")
        else:
            lines.append(f"{fam}→{exp.name}")
    if not lines and not fails:
        return "SKIP", "arm/end 均为空族"
    ev = "恢复 cfg 默认模式：" + ("；".join(lines) or "无")
    return ("FAIL" if fails else "PASS"), ev + (f"；异常：{fails}" if fails else "")


# ---------------- enable 组（真机：使能，高风险） ----------------
def _power_item(fam: str, enable: bool):
    def fn(s: Session):
        b = s.backend()
        i0 = _mark()
        ret = getattr(b, f"{'enable' if enable else 'disable'}_{fam}")()
        abled = getattr(b, f"is_abled_{fam}")
        if ret != 1:
            return "FAIL", (f"{'enable' if enable else 'disable'}_{fam} 返回 0（核心能力失败）；"
                            f"warn：{_recent(_since(i0))}")
        if abled is not (True if enable else False):
            return "FAIL", f"返回 1 但 is_abled_{fam}={abled}（应 {True if enable else False}）"
        return "PASS", f"返回 1；is_abled_{fam}={abled}"
    return fn


def _it54(s: Session):
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        _clear_throttle(b)
        r1 = r2 = d1 = d2 = None
        try:
            r1, r2 = getattr(b, f"enable_{fam}")(), getattr(b, f"enable_{fam}")()
            d1, d2 = getattr(b, f"disable_{fam}")(), getattr(b, f"disable_{fam}")()
            abled = getattr(b, f"is_abled_{fam}")
            ok = (all(isinstance(r, int) for r in (r1, r2, d1, d2))
                  and r1 == 1 and d2 == 1 and abled is False)
            results.append((fam, "PASS" if ok else "FAIL",
                            f"enable×2→({r1},{r2})，disable×2→({d1},{d2})；最终 is_abled={abled}"))
        except Exception as e:
            results.append((fam, "FAIL",
                            f"重复调用异常（enable×2→({r1},{r2})，disable×2→({d1},{d2})）：{e!r}"))
    return _aggregate(results)


# ---------------- write 组（真机：参数写入与设零） ----------------
def _it60(s: Session):
    key = s.args.param
    if not key:
        return "SKIP", "未提供 --param，无法做写回核对"
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        _clear_throttle(b)
        orig = getattr(b, f"read_param_{fam}")(key)
        if orig is None:
            results.append((fam, "SKIP",
                            f"read_param({key!r}) 不可用（键对该族无效或未实现——无法获取原值）"))
            continue
        i0 = _mark()
        ret = getattr(b, f"write_param_{fam}")(key, orig)  # 写回原值：无实际副作用的写入验证
        if ret != 1:
            results.append((fam, "FAIL", f"写回原值失败（返回 {ret}）；warn：{_recent(_since(i0))}"))
            continue
        back = getattr(b, f"read_param_{fam}")(key)
        if back is None:
            results.append((fam, "FAIL", "写后读回失败（返回 None）"))
            continue
        same = all(np.allclose(float(a), float(v), rtol=1e-6, atol=1e-9)
                   for a, v in zip(orig, back))
        results.append((fam, "PASS" if same else "FAIL",
                        f"读原值 {orig} → 写回 → 读回 {back}（一致={same}）"))
    return _aggregate(results)


def _set_zero_item(fam: str):
    def fn(s: Session):
        b = s.backend()
        i0 = _mark()
        ret = getattr(b, f"set_zero_{fam}")()
        msgs = _since(i0)
        if ret == 1:
            return "PASS", f"set_zero_{fam} 全部关节成功"
        if _cap_missing(msgs):
            return "SKIP", "子类未实现设零内核（保留 NotImplementedError 桩）"
        return "FAIL", f"返回 0；warn：{_recent(msgs)}"
    return fn


def _it63(s: Session):
    key = s.args.param or "__probe__"
    results = []
    for fam in _both_fams(s):
        b = s.fresh()  # 未连接实例：维度门禁在连接门禁之前触发，无帧发出
        try:
            n = getattr(b, f"n_joints_{fam}")
            i0 = _mark()
            ret = getattr(b, f"write_param_{fam}")(key, [0.0] * (n + 1))
            hit = any("维度" in m for m in _since(i0))
            results.append((fam, "PASS" if (ret == 0 and hit) else "FAIL",
                            f"write_param({key!r}, n+1 值) 返回 {ret}；维度 warn={hit}"))
        finally:
            b.close()
    return _aggregate(results)


# ---------------- motion 组（真机：运动发送，高风险） ----------------
def _it70(s: Session):
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        n = getattr(b, f"n_joints_{fam}")
        q0, src = _current_q(b, fam)
        if q0 is None:
            results.append((fam, "SKIP",
                            f"{src}——无法自实际位置小幅出发，盲发目标有大幅运动风险"))
            continue
        _clear_throttle(b)
        target = _clip_target(getattr(b, f"joint_limits_{fam}"), q0, MIT_AMP)
        i0 = _mark()
        getattr(b, f"send_mit_{fam}")(np.zeros(n), target, np.zeros(n),
                                      np.full(n, SAFE_KP), np.full(n, SAFE_KD))
        verdict = _classify_send(_since(i0))
        time.sleep(SETTLE)
        st = getattr(b, f"get_state_{fam}")()
        ev = (f"目标 q={np.round(target, 4).tolist()}（自{src} +{MIT_AMP} rad，"
              f"kp={SAFE_KP}/kd={SAFE_KD}，tau=0）")
        if st is not None and st.q is not None:
            ev += f"；{SETTLE}s 后实测 q={np.round(np.asarray(st.q, dtype=float), 4).tolist()}"
        results.append((fam, verdict, ev))
    return _aggregate(results)


def _it71(s: Session):
    b = s.backend()
    n = b.n_joints_arm
    q0, src = _current_q(b, "arm")
    if q0 is None:
        return "SKIP", f"{src}——无法自实际位置小幅出发，盲发目标有大幅运动风险，跳过"
    target = _clip_target(b.joint_limits_arm, q0, POS_AMP)
    i0 = _mark()
    b.send_position_arm(target, np.full(n, SAFE_VLIM), np.full(n, SAFE_FLIM))
    verdict = _classify_send(_since(i0))
    time.sleep(SETTLE)
    st = b.get_state_arm()
    ev = f"目标 q={np.round(target, 4).tolist()}（自{src} +{POS_AMP} rad，vlim={SAFE_VLIM} rad/s，flim={SAFE_FLIM}）"
    if st is not None and st.q is not None:
        ev += f"；{SETTLE}s 后实测 q={np.round(np.asarray(st.q, dtype=float), 4).tolist()}"
    return verdict, ev


def _it72(s: Session):
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        n = getattr(b, f"n_joints_{fam}")
        q0, src = _current_q(b, fam)
        if q0 is None:
            results.append((fam, "SKIP", f"{src}——电机反馈异常时不宜做速度测试"))
            continue
        _clear_throttle(b)
        i0 = _mark()
        getattr(b, f"send_vel_{fam}")(np.full(n, VEL_TEST))
        m1 = _since(i0)
        time.sleep(VEL_DUR)
        st = getattr(b, f"get_state_{fam}")()
        i1 = _mark()
        getattr(b, f"send_vel_{fam}")(np.zeros(n))  # 停止
        m2 = _since(i1)
        verdict = _classify_send(m1 + m2)
        time.sleep(0.2)
        ev = f"dq={VEL_TEST} rad/s 持续 {VEL_DUR}s 后已发 dq=0 停止"
        if st is not None and st.dq is not None:
            ev += f"；运动中实测 dq={np.round(np.asarray(st.dq, dtype=float), 3).tolist()}"
        results.append((fam, verdict, ev))
    return _aggregate(results)


def _it73(s: Session):
    b = s.backend()
    n = b.n_joints_end
    q0, src = _current_q(b, "end")
    if q0 is None:
        return "SKIP", f"{src}——无法自实际位置小幅出发，盲发目标有大幅运动风险，跳过"
    target = _clip_target(b.joint_limits_end, q0, END_AMP)
    i0 = _mark()
    b.send_position_end(target, np.full(n, SAFE_VLIM), np.full(n, SAFE_FLIM))
    verdict = _classify_send(_since(i0))
    time.sleep(SETTLE)
    st = b.get_state_end()
    ev = f"末端目标 q={np.round(target, 4).tolist()}（自{src} +{END_AMP} rad，vlim={SAFE_VLIM} rad/s）"
    if st is not None and st.q is not None:
        ev += f"；{SETTLE}s 后实测 q={np.round(np.asarray(st.q, dtype=float), 4).tolist()}"
    return verdict, ev


def _it74(s: Session):
    acts = s.args.action or ["home"]  # 默认仅测 home（--action 可覆盖为子类定义的其他动作）
    b = s.backend()
    worst, lines = "PASS", []
    for a in acts:
        i0 = _mark()
        b.send_action_end(a)
        msgs = _since(i0)
        # 离散动作集由子类自定义：未定义该动作（内核异常，如 KeyError）与未实现内核均属能力缺失 → SKIP
        if any(GATE_CONN in m for m in msgs):
            v = "FAIL"
        elif _cap_missing(msgs) or any(KERN_EXC in m for m in msgs):
            v = "SKIP"
        else:
            v = "PASS"
        lines.append(f"{a}: {v}")
        if v == "FAIL":
            worst = "FAIL"
        elif v == "SKIP" and worst == "PASS":
            worst = "SKIP"
    return worst, "；".join(lines)


def _it75(s: Session):
    results = []
    for fam in _both_fams(s):
        b = s.fresh()  # 未连接实例：裁剪 warn 在连接门禁之前触发，无帧发出
        try:
            n = getattr(b, f"n_joints_{fam}")
            hi = np.asarray(getattr(b, f"joint_limits_{fam}").q_max, dtype=float)
            if not np.isfinite(hi).any():
                results.append((fam, "SKIP", "q_max 均未配置（±inf），裁剪行为不可观测"))
                continue
            q_bad = np.where(np.isfinite(hi), hi + 10.0, 0.0)
            i0 = _mark()
            getattr(b, f"send_mit_{fam}")(np.zeros(n), q_bad, np.zeros(n),
                                          np.zeros(n), np.zeros(n))
            msgs = _since(i0)
            clip = any(CLIP_Q in m for m in msgs)
            gate = any(GATE_CONN in m for m in msgs)
            results.append((fam, "PASS" if (clip and gate) else "FAIL",
                            f"越限 q（q_max+10）触发裁剪 warn={clip}；未连接被门禁拦截={gate}"
                            f"（无帧发出；tau=0/kp=0/kd=0 物理无输出）"))
        finally:
            b.close()
    return _aggregate(results)


def _it76(s: Session):
    results = []
    for fam in _both_fams(s):
        b = s.fresh()  # 维度门禁最先触发，无需连接、无帧发出
        try:
            n = getattr(b, f"n_joints_{fam}")
            i0 = _mark()
            getattr(b, f"send_mit_{fam}")(np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1),
                                          np.zeros(n + 1), np.zeros(n + 1))
            hit = any(DIM_SEND in m for m in _since(i0))
            results.append((fam, "PASS" if hit else "FAIL",
                            f"错误维度（n+1）被拒绝且 warn 指令维度={hit}（无帧发出）"))
        finally:
            b.close()
    return _aggregate(results)


# ---------------- error 组（真机：清错） ----------------
def _clear_error_item(fam: str):
    def fn(s: Session):
        b = s.backend()
        i0 = _mark()
        ret = getattr(b, f"clear_error_{fam}")()
        msgs = _since(i0)
        if ret is True:
            return "PASS", f"clear_error_{fam} 返回 True（正常状态下清错指令已发送）"
        if _cap_missing(msgs):
            return "SKIP", "子类未实现清错内核（该型号无清错命令属能力缺失）"
        return "FAIL", f"返回 False；warn：{_recent(msgs)}"
    return fn


# ---------------- disconnect 组（真机：断连收尾） ----------------
def _it90(s: Session):
    b = s.backend()
    lines, fails = [], []
    for fam in ("arm", "end"):
        if getattr(b, f"n_joints_{fam}") == 0:
            continue
        r = getattr(b, f"disable_{fam}")()
        getattr(b, f"disconnect_{fam}")()
        conn = getattr(b, f"is_connected_{fam}")
        mode = getattr(b, f"mode_{fam}")
        if conn is not False:
            fails.append(f"{fam}: is_connected={conn}（应 False）")
        if mode is not None:
            fails.append(f"{fam}: mode 缓存未清空（{mode}）")
        lines.append(f"{fam}（disable→{r}，is_connected={conn}，mode={mode}）")
    if not lines:
        return "SKIP", "arm/end 均为空族"
    return ("FAIL" if fails else "PASS"), "失能→断连编排：" + "；".join(lines)


def _it91(s: Session):
    b = s.backend()
    b.close()
    return "PASS", "close() 无异常（刷新定时器已停止；main 收尾将再次 close，幂等）"


# ============================================================
# 测试项注册表（编号=十位组+序号，风险递进；family 空/非空的族自动跳过）
# ============================================================
ITEMS: list[dict] = []


def _item(num, name, group, risk, family, purpose, needs, fn, requires_args=()):
    ITEMS.append(dict(num=num, name=name, group=group, risk=risk, family=family,
                      purpose=purpose, needs=needs, fn=fn, requires_args=tuple(requires_args)))


# —— init 组（离线）——
_item(1, "模块导入与子类发现", "init", "低", None,
      "验证 backend_<型号>.py 可导入，且其中的 Backend<型号> 类确为 Backend 子类", [], _it1)
_item(2, "完整 cfg 构造", "init", "低", None,
      "验证以 cfg backend 段构造成功且不做任何总线 I/O（离线可构造）", [], _it2)
_item(3, "空 cfg 构造容错", "init", "低", None,
      "验证空 cfg {} 可构造（基类对缺段/缺键全部容错的契约）", [], _it3)
_item(4, "运行参数非法回退", "init", "低", None,
      "验证 state_refresh_hz 等运行参数非法（非正数）时回退默认并 warn", [], _it4)
_item(5, "只读属性初值", "init", "低", None,
      "验证关节数/name 与 cfg 一致，is_connected*/is_abled*/mode* 初值均为 None（三值未知）", [], _it5)
_item(6, "硬限位解析核对", "init", "低", None,
      "验证 joint_limits_* 的 q_min/q_max/dq_max/tau_max 与 cfg 各关节四键一致（发送裁剪的唯一依据）", [], _it6)
_item(7, "joint_state 初值", "init", "低", None,
      "验证初始快照 t=0、q=NaN、error=-1（从未读取的约定标记）", [], _it7)

# —— gate 组（离线：未连接门禁）——
_item(10, "未连接 get_state 门禁", "gate", "低", None,
      "验证未连接时 get_state_* 返回 None 且触发连接门禁 warn（不崩溃）", [], _it10)
_item(11, "未连接 get_mode 门禁", "gate", "低", None,
      "验证未连接时 get_mode_* 返回 None 且触发连接门禁 warn", [], _it11)
_item(12, "未连接 read_param 门禁", "gate", "低", None,
      "验证未连接时 read_param_* 返回 None 且触发连接门禁 warn", [], _it12)
_item(13, "未连接 set_mode 门禁", "gate", "低", None,
      "验证未连接时 set_mode_* 返回 0 且触发连接门禁 warn（不达内核）", [], _it13)
_item(14, "未连接 send_mit 门禁", "gate", "低", None,
      "验证未连接时发送指令被连接门禁拦截（合法值、无帧发出）", [], _it14)
_item(15, "未连接 enable/disable 门禁", "gate", "低", None,
      "验证未连接时 enable_*/disable_* 均返回 0 且不崩溃", [], _it15)
_item(16, "未连接 get_error/check_error", "gate", "低", None,
      "验证未连接时 get_error_* 返回 None、check_error_* 返回 False", [], _it16)
_item(17, "未连接写入类安全失败", "gate", "低", None,
      "验证未连接时 write_param/set_zero 返回 0、clear_error 返回 False（安全失败）", [], _it17)
_item(18, "close() 幂等", "gate", "低", None,
      "验证 close() 可连续调用无异常（刷新定时器停止）", [], _it18)

# —— connect 组（真机）——
_item(20, "connect_arm 连接", "connect", "中", "arm",
      "验证连接成功后 is_connected_arm=True 且 cfg 默认模式乐观回填 mode_arm", [], _connect_item("arm"))
_item(21, "connect_end 连接", "connect", "中", "end",
      "验证末端族连接成功后 is_connected_end=True 且默认模式回填", [], _connect_item("end"))
_item(22, "重复 connect 幂等", "connect", "中", None,
      "验证已连接状态下重复 connect 不崩溃、连接标志保持 True", [("connected", "both")], _it22)
_item(23, "断连（arm/end 双族）", "connect", "中", None,
      "验证断连后 is_connected=False、模式缓存清空（防过期模式放行错误指令）、is_abled 置 None", [("connected", "both")], _it23)
_item(24, "断连后重连恢复（双族）", "connect", "中", None,
      "验证断连后重连可恢复（默认模式重新乐观回填）", [("connected", "both")], _it24)
_item(25, "实时状态刷新监视", "connect", "低", None,
      "验证 connect 后基类低频刷新线程持续更新快照：每秒打印 arm/end 全部关节量表（被动快照，"
      "--watch 秒，默认 10；全程快照龄 <500ms 判 PASS）", [("connected", "both")], _it25)

# —— read 组（真机：只读）——
_item(30, "get_mode 读回（双族）", "read", "低", None,
      "验证模式读回返回 ControlMode 枚举（无回读能力则 SKIP）", [("connected", "both")], _it30)
_item(31, "get_state_arm 状态契约", "read", "低", "arm",
      "验证 JointState 在场字段为 (n,) 且有限、t>0 且随调用更新（缺字段=None 合法）", [("connected", "arm")], _state_item("arm"))
_item(32, "get_state_end 状态契约", "read", "低", "end",
      "验证末端族 JointState 契约（与 31 对称）", [("connected", "end")], _state_item("end"))
_item(33, "状态快照引用语义（双族）", "read", "低", None,
      "验证 get_state 原子换入新 JointState 对象、joint_state 属性实时引用、旧引用冻结为快照", [("connected", "both")], _it33)
_item(34, "read_param 参数读取（双族）", "read", "低", None,
      "验证参数读取返回逐关节标量 list（须 --param 指定子类定义的键；不可用则 SKIP）",
      [("connected", "both")], _it34, requires_args=("param",))
_item(35, "错误码读取与检查（双族）", "read", "低", None,
      "验证 get_error 逐关节状态码（0/1 正常、≥2 故障）与 check_error 一致性（全 0/1→True；"
      "读不到→False）；两者结果均输出", [("connected", "both")], _it35)

# —— mode 组（真机；原 40/41/42 三模式拆分项已合并入 40）——
_item(40, "set_mode 三模式（arm）", "mode", "中", "arm",
      "循环设 MIT/POSITION/VELOCITY：逐模式先读→设→再读（回读验证；切换 POSITION/VELOCITY 随模式写增益寄存器）",
      [("connected", "arm")], _modes_item("arm"))
_item(41, "set_mode 三模式（end）", "mode", "中", "end",
      "末端族三模式循环设置（与 40 对称；至少一种可用即 PASS）", [("connected", "end")], _modes_item("end"))
_item(42, "set_mode 非法/未定义模式门禁（双族）", "mode", "中", None,
      "验证非法入参（字符串）与未被 ControlMode 定义的模式（伪枚举）均被类型门禁拒绝"
      "（返回 0 + warn，不达内核）", [("connected", "both")], _it42)
_item(43, "恢复默认模式", "mode", "中", None,
      "收尾：将 arm/end 恢复 cfg 默认模式（后续组的模式前置由此兜底）", [("connected", "both")], _it43)

# —— enable 组（真机：高风险）——
_item(50, "enable_arm 使能", "enable", "高", "arm",
      "验证使能全部关节返回 1 且 is_abled_arm=True（电机上力）", [("connected", "arm")], _power_item("arm", True))
_item(51, "enable_end 使能", "enable", "高", "end",
      "验证末端族使能契约（与 50 对称）", [("connected", "end")], _power_item("end", True))
_item(52, "disable_arm 失能", "enable", "高", "arm",
      "验证失能全部关节返回 1 且 is_abled_arm=False（安全方向）", [("connected", "arm")], _power_item("arm", False))
_item(53, "disable_end 失能", "enable", "高", "end",
      "验证末端族失能契约（与 52 对称）", [("connected", "end")], _power_item("end", False))
_item(54, "enable/disable 重复调用（双族）", "enable", "高", None,
      "验证使能/失能重复调用安全（返回 int、最终失能态）", [("connected", "both")], _it54)

# —— write 组（真机：高风险；61/62 极高危仅显式编号执行）——
_item(60, "write_param 写回核对（双族）", "write", "高", None,
      "验证参数写入：读原值→写回原值→读回核对（无副作用；须 --param）",
      [("connected", "both")], _it60, requires_args=("param",))
_item(61, "set_zero_arm 设零", "write", "极高", "arm",
      "【永久改写零位标定，不可逆】验证 arm 逐关节设零返回 1（仅显式 --step=61 执行）", [("connected", "arm")], _set_zero_item("arm"))
_item(62, "set_zero_end 设零", "write", "极高", "end",
      "【永久改写零位标定，不可逆】验证 end 逐电机设零返回 1（仅显式 --step=62 执行）", [("connected", "end")], _set_zero_item("end"))
_item(63, "write_param 维度拒绝（双族）", "write", "低", None,
      "验证维度不符的写入被拒绝（返回 0 + 维度 warn；维度门禁最先触发，无帧发出）", [], _it63)

# —— motion 组（真机：高风险；75/76 为基类管线探测，无帧发出）——
_item(70, "send_mit 小幅（双族）", "motion", "高", None,
      "验证 MIT 指令小幅下发（自当前位置 +0.02 rad、低增益、tau=0；arm/end 双族）",
      [("connected", "both"), ("mode", "both", ControlMode.MIT), ("enabled", "both")], _it70)
_item(71, "send_position_arm 小幅", "motion", "高", "arm",
      "验证位置指令小幅下发（+0.05 rad、限速 0.5 rad/s）", [("connected", "arm"), ("mode", "arm", ControlMode.POSITION), ("enabled", "arm")], _it71)
_item(72, "send_vel 低速短时（双族）", "motion", "高", None,
      "验证速度指令低速短时下发（0.2 rad/s × 0.5s 后发 dq=0 停止；arm/end 双族）",
      [("connected", "both"), ("mode", "both", ControlMode.VELOCITY), ("enabled", "both")], _it72)
_item(73, "send_position_end 小幅", "motion", "高", "end",
      "验证末端位置指令小幅下发（末端最常用能力；不支持则 SKIP）", [("connected", "end"), ("mode", "end", ControlMode.POSITION), ("enabled", "end")], _it73)
_item(74, "send_action_end 离散动作", "motion", "高", "end",
      "验证末端离散动作下发（默认仅测 home；子类未定义该动作则 SKIP 属能力缺失；"
      "--action 可覆盖；离散动作通常为位置语义，故前置 POSITION 模式）",
      [("connected", "end"), ("mode", "end", ControlMode.POSITION), ("enabled", "end")], _it74)
_item(75, "越限裁剪管线（双族）", "motion", "低", None,
      "验证越限指令被基类就近裁剪并 warn（未连接实例探测，tau=0/kp=0 无物理输出）", [], _it75)
_item(76, "发送维度拒绝（双族）", "motion", "低", None,
      "验证错误维度指令被拒绝（维度 warn，无帧发出）", [], _it76)

# —— error 组（真机）——
_item(80, "clear_error_arm 清错", "error", "中", "arm",
      "验证正常状态下清错指令发送返回 True（未实现则 SKIP，属能力缺失）", [("connected", "arm")], _clear_error_item("arm"))
_item(81, "clear_error_end 清错", "error", "中", "end",
      "验证末端族清错契约（与 80 对称）", [("connected", "end")], _clear_error_item("end"))

# —— disconnect 组（真机：收尾）——
_item(90, "失能→断连全族编排", "disconnect", "低", None,
      "验证先失能再断连的编排后两族 is_connected=False、模式缓存清空", [("connected", "both")], _it90)
_item(91, "close() 定时器销毁", "disconnect", "低", None,
      "验证 close() 停止低频刷新定时器且无异常（不可重启）", [], _it91)

ALL_NUMS = {it["num"] for it in ITEMS}
_BY_NUM = {it["num"]: it for it in ITEMS}

# 组名 → 编号集合（61/62 排除出 write/online/all，仅显式编号执行）
GROUPS: dict[str, list[int]] = {
    "init": [1, 2, 3, 4, 5, 6, 7],
    "gate": [10, 11, 12, 13, 14, 15, 16, 17, 18],
    "connect": [20, 21, 22, 23, 24, 25],
    "read": [30, 31, 32, 33, 34, 35],
    "mode": [40, 41, 42, 43],
    "enable": [50, 51, 52, 53, 54],
    "write": [60, 63],
    "motion": [70, 71, 72, 73, 74, 75, 76],
    "error": [80, 81],
    "disconnect": [90, 91],
}
GROUPS["offline"] = GROUPS["init"] + GROUPS["gate"]
GROUPS["online"] = (GROUPS["connect"] + GROUPS["read"] + GROUPS["mode"] + GROUPS["enable"]
                    + GROUPS["write"] + GROUPS["motion"] + GROUPS["error"] + GROUPS["disconnect"])
GROUPS["all"] = GROUPS["offline"] + GROUPS["online"]


# ============================================================
# 子类发现与配置加载
# ============================================================
def discover(sub: str) -> type:
    """解析子类：优先查官方注册表 REGISTRY，再按文件约定导入 backend_<sub> 寻找 Backend 子类。"""
    from joyarm_core.backend import REGISTRY  # 官方注册路径（六步接入第 4 步），键名 = 文件名如 "backend_dm"
    reg_name = f"backend_{sub}"
    if reg_name in REGISTRY:
        return REGISTRY[reg_name]
    mod_name = f"joyarm_core.backend.backend_{sub}"
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        avail = sorted(set(p.stem for p in Path(__file__).resolve().parent.glob("backend_*.py"))
                       | set(REGISTRY))
        raise _UsageError(f"找不到 {mod_name}（--sub={sub}）；REGISTRY 与 backend/ 目录现有：{avail or '无'}")
    cands = {o for o in vars(mod).values()
             if isinstance(o, type) and issubclass(o, Backend) and o is not Backend
             and o.__module__ == mod_name}
    for cand in (f"Backend{sub.upper()}", f"Backend{sub.capitalize()}", f"Backend{sub}"):
        cls = getattr(mod, cand, None)
        if cls in cands:
            return cls
    if len(cands) == 1:
        return next(iter(cands))
    raise _UsageError(
        f"{mod_name} 中 Backend 子类不唯一或未找到：{sorted(c.__name__ for c in cands) or '无'}")


def load_cfg(args: argparse.Namespace, sub: str) -> tuple[dict, Path]:
    if args.config:
        path = Path(args.config)
    else:
        path = Path(__file__).resolve().parents[1] / "config" / f"joyarm_{sub}.yaml"
    if not path.is_file():
        raise _UsageError(f"配置文件不存在：{path}（--config 可指定其他 yaml）")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bcfg = data.get("backend")
    if not isinstance(bcfg, dict):
        raise _UsageError(f"{path} 缺少有效的 backend: 段")
    return dict(bcfg), path


def resolve_steps(spec: str) -> list[int]:
    nums: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok.isdigit():
            n = int(tok)
            if n not in ALL_NUMS:
                raise _UsageError(f"未知步骤编号 {n}；有效编号：{sorted(ALL_NUMS)}")
            nums.append(n)
        elif tok in GROUPS:
            nums.extend(GROUPS[tok])
        else:
            raise _UsageError(f"未知步骤 {tok!r}；有效组名：{sorted(GROUPS)}，或编号 {sorted(ALL_NUMS)}")
    if not nums:
        raise _UsageError(f"--step={spec!r} 未解析出任何测试项")
    return sorted(set(nums))


# ============================================================
# 执行框架
# ============================================================
def run_item(it: dict, s: Session) -> tuple[str, str]:
    """单项执行：打印项与目的 → 风险确认 → 前置补齐 → 执行 → 打印结果。"""
    num = it["num"]
    print()
    print(f"[{num}] {it['name']} | 组={it['group']} | 风险={it['risk']}")
    print(f"目的：{it['purpose']}")
    if it["family"] and _n_family_cfg(s.cfg, it["family"]) == 0:
        ev = f"{it['family']} 族未配置任何关节（n=0），空族自动跳过"
        print(f"[{num}] SKIP —— {ev}")
        return "SKIP", ev
    missing = [k for k in it.get("requires_args", ()) if not getattr(s.args, k, None)]
    if missing:
        ev = (f"未提供 --{'/'.join(missing)}（该值由子类定义，须显式指定后方可测试）")
        print(f"[{num}] SKIP —— {ev}")
        return "SKIP", ev
    plans = _plan_prefill(s, it["needs"])
    level = it["risk"]
    for _, lv in plans:  # 将补齐的前置风险并入该项确认等级
        if _RISK_ORDER[lv] > _RISK_ORDER[level]:
            level = lv
    if level != "低":
        _hardware_banner_once()
    if it["needs"]:
        extra = f"；将自动补齐：{'、'.join(d for d, _ in plans)}" if plans else "（已就绪）"
        print(f"前置：{_describe_needs(it['needs'])}{extra}")
    ans = _confirm(level, s.args)
    if ans == "quit":
        ev = "用户退出（q）"
        print(f"[{num}] SKIP —— {ev}")
        return "QUIT", ev
    if ans == "skip":
        if not sys.stdin.isatty() and level in ("高", "极高"):
            ev = "非终端环境自动跳过（高危永不自动执行；交互终端下需输入 yes）"
        else:
            ev = f"用户跳过（风险={level}）"
        print(f"[{num}] SKIP —— {ev}")
        return "SKIP", ev
    status, actions, reason = _prefill(s, it["needs"])
    if status == "fail":
        ev = f"前置补齐失败：{'；'.join(actions) or '（无）'}——{reason}"
        print(f"[{num}] FAIL —— {ev}")
        return "FAIL", ev
    if status == "skip":
        ev = f"前置不可用：{reason}"
        print(f"[{num}] SKIP —— {ev}")
        return "SKIP", ev
    if actions:
        print(f"（已自动补齐前置：{'；'.join(actions)}）")
    if s.b is not None:
        s.b._warn_last.clear()  # 测试基建：重置 warn 限频时间戳，防上一项同通道 0.5s 限频吞掉本项 warn（判定依赖 warn 文本）
    try:
        verdict, evidence = it["fn"](s)
    except Exception as e:
        verdict, evidence = "FAIL", f"未预期异常 {type(e).__name__}: {e}"
    print(f"[{num}] {verdict} —— {evidence}")
    return verdict, evidence


def _cleanup(s: Session):
    """任何退出路径的兜底收尾：尽力失能 → 断连 → close（全程不抛）。"""
    if s.b is None:
        return
    for call in (s.b.disable_arm, s.b.disable_end, s.b.disconnect_arm, s.b.disconnect_end, s.b.close):
        try:
            call()
        except Exception:
            pass


def print_list():
    print("Backend 子类验收测试项清单（编号=十位组+序号，按风险递进；arm/end 对称覆盖）")
    print("=" * 68)
    for it in sorted(ITEMS, key=lambda x: x["num"]):
        tag = "　★极高危：仅显式 --step=编号 执行，不含于 write/online/all" if it["risk"] == "极高" else ""
        print(f"[{it['num']:>2}] {it['name']}｜组={it['group']}｜风险={it['risk']}{tag}")
        print(f"      目的：{it['purpose']}")
    print("=" * 68)
    print("基础组：init / gate / connect / read / mode / enable / write / motion / error / disconnect")
    print("复合组：offline=init+gate（--step 缺省）｜online=真机全组（不含 61/62）｜all=offline+online（不含 61/62）")
    print("通用机制：能力缺失=SKIP（非缺陷）；空族自动跳过；参数项须 --param、动作项须 --action")


EPILOG = """\
示例：
  python joyarm_core/backend/test_backend.py --help                     # 本说明
  python joyarm_core/backend/test_backend.py --list                     # 全部测试项
  python joyarm_core/backend/test_backend.py --sub=dm                   # 离线段（缺省 --step=offline）
  python joyarm_core/backend/test_backend.py --sub=dm --step=all        # 全部（不含 set_zero 61/62）
  python joyarm_core/backend/test_backend.py --sub=dm --step=40         # 单项
  python joyarm_core/backend/test_backend.py --sub=dm --step=init,20,30 # 混选
  python joyarm_core/backend/test_backend.py --sub=dm --step=25 --watch 30   # 刷新监视 30s
  python joyarm_core/backend/test_backend.py --sub=dm --step=online --param=<键> --action=open,close

风险与确认策略：
  低危：直接执行（离线构造 / 只读 / 无帧管线探测）。
  中危：交互回车确认（--yes 或非终端环境直接执行）。
  高危：强制交互输入 yes（--yes 不放行；非终端自动 SKIP——agent 无法全自动执行高危项）。
  极高危（61/62 set_zero）：永久改写零位标定，仅显式 --step=61/62 执行并双重警示。

判定（能力三分法）：
  PASS=符合基类契约；SKIP=能力缺失（该子类/硬件不提供，非缺陷）；FAIL=违反契约。
  前置自动补齐：单项可独立执行（如 --step=50 会自动 connect + enable，高危前置并入确认）。

退出码：全 PASS/SKIP=0；任一 FAIL=1；用法错误=2。结尾输出 SUMMARY: total=… pass=… fail=… skip=…（机器可解析）。
"""


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="test_backend.py",
        description=("Backend 子类通用验收测试（上层视角）：只调 Backend 基类公开 API，"
                     "对任意子类（关节电机/舵机/混合硬件）做分类分项逐级测试。"
                     "能力三分法（能力缺失=SKIP）+ 空族自动跳过 + 风险分级确认。"),
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sub", help="子型简称（如 dm、feetech、template）：自动导入 backend_<型号>.py 并寻找"
                                 " Backend<型号> 类，默认加载 config/joyarm_<型号>.yaml 的 backend 段")
    p.add_argument("--step", default="offline",
                   help="数字=单项编号（如 40）；字符串=组名（init/gate/connect/read/mode/enable/write/"
                        "motion/error/disconnect/offline/online/all）；逗号分隔混选；缺省 offline")
    p.add_argument("--config", help="yaml 配置文件路径（默认 joyarm_core/config/joyarm_<sub>.yaml）")
    p.add_argument("--param", help="参数读写测试（34/60/63）的参数键（键表由子类定义；缺省相关项 SKIP）")
    p.add_argument("--action", help="send_action_end 测试（74）的离散动作名（逗号分隔；默认仅测 home，"
                                    "子类动作名不同时可覆盖）")
    p.add_argument("--watch", type=float, default=10.0,
                   help="item 25 实时状态刷新监视时长（秒，默认 10；每秒打印 arm/end 全部关节量表）")
    p.add_argument("--list", action="store_true", help="仅列出全部测试项（编号/组/风险/目的），不执行")
    p.add_argument("--yes", action="store_true",
                   help="跳过低/中危交互确认（高危永不自动执行：仍需交互 yes，非终端自动 SKIP）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:  # stdout 行缓冲：与 stderr 日志正确交错（管道/重定向场景）
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.list:
        print_list()
        return 0
    if not args.sub:
        print("用法错误：执行测试须提供 --sub=<型号>（仅 --help/--list 可省略；--list 查看全部测试项）",
              file=sys.stderr)
        return 2
    args.action = [a.strip() for a in (args.action or "").split(",") if a.strip()]
    try:
        cls = discover(args.sub)
        cfg, cfg_path = load_cfg(args, args.sub)
        nums = resolve_steps(args.step)
    except _UsageError as e:
        print(f"用法错误：{e}", file=sys.stderr)
        return 2
    s = Session(cls, cfg, args, cfg_path)
    tty = sys.stdin.isatty()
    print("=" * 68)
    print(f"Backend 子类验收测试：{cls.__name__}（backend_{args.sub}.py）")
    print(f"配置：{cfg_path}（arm {_n_family_cfg(cfg, 'arm')} 关节 / end {_n_family_cfg(cfg, 'end')} 关节）")
    print(f"交互终端：{'是' if tty else '否'}；--yes={args.yes}；"
          f"高危策略：{'交互 yes（--yes 不放行）' if tty else '非终端自动 SKIP'}")
    print(f"待执行 {len(nums)} 项：{nums}")
    results: list[tuple[dict, str, str]] = []
    quit_flag = False
    try:
        for num in nums:
            it = _BY_NUM[num]
            if quit_flag:
                ev = "用户提前退出（q，未执行）"
                print(f"[{num}] SKIP —— {ev}")
                results.append((it, "SKIP", ev))
                continue
            v, ev = run_item(it, s)
            if v == "QUIT":
                quit_flag = True
                results.append((it, "SKIP", ev))
            else:
                results.append((it, v, ev))
    finally:
        _cleanup(s)
    n_pass = sum(1 for _, v, _ in results if v == "PASS")
    n_fail = sum(1 for _, v, _ in results if v == "FAIL")
    n_skip = sum(1 for _, v, _ in results if v == "SKIP")
    print()
    print("=" * 68)
    print(f"SUMMARY: total={len(results)} pass={n_pass} fail={n_fail} skip={n_skip}")
    fails = [(it, ev) for it, v, ev in results if v == "FAIL"]
    skips = [(it, ev) for it, v, ev in results if v == "SKIP"]
    if not fails:
        concl = "结论：无失败项，被测子类符合 Backend 基类全部已测契约。"
        if skips:
            concl += f"共 {n_skip} 项跳过（能力缺失/空族/用户跳过，详见各项证据，均非缺陷）。"
    else:
        concl = f"结论：{n_fail} 项失败——"
        print(concl)
        for it, ev in fails:
            print(f"  FAIL [{it['num']}] {it['name']}：{_trunc(ev)}")
        concl = f"（详见上方 {n_fail} 条 FAIL 证据；其余 pass={n_pass}、skip={n_skip}）"
    print(concl)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
