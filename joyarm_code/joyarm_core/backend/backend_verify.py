"""
==============================================================================
Backend 子类全方位测试验证（上层视角）  joyarm_core/backend/backend_verify.py
==============================================================================
【功能概要】
    从上层视角（只调 Backend 基类公开方法与属性，不触碰子类私有层、与型号实现
    细节无关）对**任意** backend 子类（关节电机 / 舵机 / 混合硬件 / 无 end 型号）
    做分组分项测试验证（verify_backend.py 的迭代版，编号改为两位数十进制分组）：
      00 组  初始化验证：构造、属性初值、限位解析（离线，全低危）；
      10 组  连接编排：arm/end 六种连断次序 + 重复连断 + 断开重联（全低危）；
      20 组  只读验证：单次状态获取、10s 实时刷新监视、模式获取、参数读取、
             错误码读取 / 检查（全低危）；
      30 组  控制验证：三模式设置（低危）、非法模式门禁（低危）、使能 / 失能
             与重复调用、写参数（高危）、设置零点（极高危，仅显式单项 + 每族
             二次确认）；
      40 组  运动发送：MIT 零力矩 / 零点保持、position / vel 自动小幅（原幅度
             减半 + 限位裕量自动选向）、末端离散动作（全高危，五项均发送前
             参数预览 + 确认，--yes 不放行）；
      50 组  清除错误（双族，高危：使能→读错误码→清错→再读核对）。
    通用化设计（不内嵌任何子型专属知识）：
      - 能力三分法：PASS=行为符合基类契约；SKIP=能力缺失（如某型号不支持
        某模式、无参数读写——非缺陷）；FAIL=能力存在但违反契约；
      - 空族自动跳过：cfg 未配置 arm/end 关节的族，该族测试项自动 SKIP；
      - 参数键 / 末端离散动作名由子类定义，须经 --param / --action 显式告知
        （--action 缺省测 home）。
    每项测试前打印测试项与目的，测试后打印 PASS/FAIL/SKIP 与证据；结尾输出
    机器可解析的 SUMMARY 行与人类可读结论（供开发者与 agent 共用）。
    安全机制：
      - 编号按风险递进（初始化 → 连接 → 只读 → 控制 → 运动 → 清错）；
      - 低危直接执行；中危交互确认（--yes 或非终端直接执行）；
      - 高危强制交互输入 yes（--yes 不放行；非终端自动 SKIP，agent 无法全自动
        执行高危项）；
      - 38 设置零点（极高危）永久改写零位标定：仅显式 --step=38 执行（任何
        组展开不含本项），且每族各自二次确认（任一次非 yes 即跳过该族及后续）。

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.sh | iex

    # 2. 创建虚拟环境并同步依赖
    cd <仓库>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（在 joyarm_code/ 目录下）
    python joyarm_core/backend/backend_verify.py --help                          # 完整用法
    python joyarm_core/backend/backend_verify.py --list                          # 全部测试项清单
    python joyarm_core/backend/backend_verify.py --sub=dm --step=offline         # 00 组（缺省）
    python joyarm_core/backend/backend_verify.py --sub=dm --step=all             # 全部（不含 38）
    python joyarm_core/backend/backend_verify.py --sub=dm --step=12              # 单项
    python joyarm_core/backend/backend_verify.py --sub=dm --step=00,20,read      # 编号/组名混选
    python joyarm_core/backend/backend_verify.py --sub=dm --step=connect,read    # 组合
    python joyarm_core/backend/backend_verify.py --sub=dm --step=online \
        --param=<参数键> --action=open,close                                     # 真机段

【命令行参数】
    --sub     子型简称（如 dm、feetech）：经 REGISTRY 或文件约定导入
              backend_<型号>.py 并寻找 Backend<型号> 类，默认加载
              config/joyarm_<型号>.yaml 的 backend 段；缺省时仅 --help/--list 可用
    --step    数字=单项编号（如 38）；字符串=组名（init/connect/read/ctrl/motion/
              error/offline/online/all）；逗号分隔可单项与组任意混选；缺省 offline
    --config  指定 yaml 配置文件（默认 joyarm_core/config/joyarm_<sub>.yaml）
    --param   参数读取 / 写入测试（23/37）的参数键（键表由子类定义；不给则 SKIP）
    --action  send_action_end 测试（44）的离散动作名，逗号分隔（默认仅测 home）
    --list    仅列出全部测试项（编号/组/风险/目的），不执行
    --yes     跳过中危交互确认（高危/极高危永不自动执行：仍需交互 yes，
              非终端自动 SKIP）
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

# ---------------- 安全常量（真机测试的小幅参数，子型中立；幅度=verify_backend 的一半） ----------------
POS_AMP = 0.025       # 位置测试目标偏移 rad（原 0.05 减半）
VEL_TEST = 0.1        # 速度测试目标 rad/s（原 0.2 减半）
VEL_DUR = 0.5         # 速度测试持续时间 s
SAFE_KP = 5.0         # MIT 零点保持用保守位置增益
SAFE_KD = 0.5         # MIT 零点保持用保守速度阻尼
SAFE_VLIM = 0.5       # 位置测试限速 rad/s
SAFE_FLIM = 0.5       # 位置测试归一化力矩电流上限（0~1）
LIMIT_MARGIN = 0.01   # 目标距硬限位的安全余量 rad
SETTLE = 0.5          # 发送后观测等待 s
WATCH_SECS = 10       # item 21 实时刷新监视时长（固定 10s，无 CLI 参数）
ZERO_TOL = 0.05       # item 38 设零后回读关节位置归零容差 rad

# 子类未实现内核的桩消息标记（backend_template.py 桩约定，基类转 warn 后保留原文）
CAP_MISSING = "功能缺失"
CAP_MISSING2 = "未实现"

# 基类 warn 消息文本标记（注意：须按消息内容匹配，且须选取在消息中唯一的子串——如"存在空值"
# 同时出现在发送空值门禁与状态数据异常两种消息里，须用发送门禁独有的尾部"，本次不发送"区分，
# 防刷新线程的并发 warn 误判）
GATE_CONN = "未连接或状态未知"      # 连接门禁（_require_connected 唯一）
GATE_MODE = "≠ 所需"               # 模式门禁（_require_mode 唯一）
GATE_TYPE = "非 ControlMode 枚举"   # set_mode 类型门禁（唯一）
KERN_EXC = "内核异常"               # 子类内核异常（基类 _call 唯一）
EMPTY_VAL = "，本次不发送"          # 发送前空值检查（_require_present 唯一尾部）
DIM_SEND = "指令维度"               # 发送维度校验（唯一）

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
    """持有子类、cfg 与会话实例；离线项经 fresh() 另建全新实例，用毕 close。"""

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
        """全新未连接实例（离线构造类测试用），用毕由调用方 close。"""
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


def _mode_name(m):
    """模式显示名（None=无回读能力）。"""
    return m.name if isinstance(m, ControlMode) else str(m)


def _safe_target(lim, q: np.ndarray, amp: float) -> np.ndarray:
    """位置测试自动目标：自当前 q 偏移 ±amp 并按限位裕量**自动选向**。

    机械臂零位可能位于极限位置——逐关节在「正向剩余行程 q_max−q」与「负向剩余
    行程 q−q_min」中选裕量大的一侧；裕量不足（amp+LIMIT_MARGIN 内）的方向不选，
    两端皆不足取 0（不动）；未配置限位（±inf）的侧不构成约束。最终再裁剪进
    硬限位（留 LIMIT_MARGIN 余量，浮点兜底）。
    """
    if lim is None:
        return q + amp
    lo = np.asarray(lim.q_min, dtype=float)
    hi = np.asarray(lim.q_max, dtype=float)
    room_hi = np.where(np.isfinite(hi), hi - q, np.inf)
    room_lo = np.where(np.isfinite(lo), q - lo, np.inf)
    margin = amp + LIMIT_MARGIN
    can_hi = room_hi > margin
    can_lo = room_lo > margin
    off = np.where(can_hi & (room_hi >= room_lo), amp, np.where(can_lo, -amp, 0.0))
    lo_c = np.where(np.isfinite(lo), lo + LIMIT_MARGIN, -np.inf)
    hi_c = np.where(np.isfinite(hi), hi - LIMIT_MARGIN, np.inf)
    return np.clip(q + off, lo_c, hi_c)


def _safe_vel_dirs(lim, q: np.ndarray, v: float) -> np.ndarray:
    """速度测试的逐关节限位保护选向。

    基类对速度指令只裁幅值（±dq_max），不做位置-限位联动——本测试自选方向：
    每关节在「正向剩余行程 q_max−q」与「负向剩余行程 q−q_min」中选裕量大的一侧，
    裕量不足（测试行程 v×VEL_DUR + 0.1 rad 余量内）的方向不选；两端皆不足取 0
    （不动）。已在限位外（如零位偏移）的关节会自动选往行程内的一侧。
    """
    lo = np.asarray(lim.q_min, dtype=float)
    hi = np.asarray(lim.q_max, dtype=float)
    room_hi = hi - q
    room_lo = q - lo
    margin = v * VEL_DUR + 0.1  # 测试行程 + 安全余量
    can_hi = room_hi > margin
    can_lo = room_lo > margin
    return np.where(can_hi & (room_hi >= room_lo), v, np.where(can_lo, -v, 0.0))


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


# ---------------- 前置自动补齐与风险确认 ----------------
def _prefill_fams(fam: str) -> tuple:
    return ("arm", "end") if fam == "both" else (fam,)


_RISK_ORDER = {"低": 0, "中": 1, "高": 2, "极高": 3}


def _plan_prefill(s: Session, needs: list) -> list[tuple[str, str]]:
    """干跑：计算将补齐哪些前置（不动状态），返回 [(动作描述, 风险)]。

    用于确认等级计算（前置风险并入该项确认）与确认文案。
    已计划的动作记入 planned_*（干跑不改真实状态，后续需求据此去重，
    防同一 connect 在文案中重复列出）。
    """
    plans: list[tuple[str, str]] = []
    planned_conn: set = set()
    planned_mode: dict = {}
    planned_abled: set = set()
    b = s.b  # 可能尚未构造（视为全部前置缺失）
    for nd in needs:
        kind = nd[0]
        for fam in _prefill_fams(nd[1]):
            if _n_family_cfg(s.cfg, fam) == 0:
                continue
            conn = ((getattr(b, f"is_connected_{fam}", None) if b is not None else None) is True
                    or fam in planned_conn)
            if kind == "connected":
                if not conn:
                    plans.append((f"connect_{fam}", "低"))
                    planned_conn.add(fam)
            elif kind == "mode":
                mode = nd[2]
                mode_cur = getattr(b, f"mode_{fam}", None) if b is not None else None
                if mode_cur != mode and planned_mode.get(fam) != mode:
                    if not conn:
                        plans.append((f"connect_{fam}", "低"))
                        planned_conn.add(fam)
                    plans.append((f"set_mode_{fam}({mode.name})", "低"))
                    planned_mode[fam] = mode
            elif kind == "enabled":
                abled = ((getattr(b, f"is_abled_{fam}", None) if b is not None else None) is True
                         or fam in planned_abled)
                if not abled:
                    plans.append((f"enable_{fam}", "高"))
                    planned_abled.add(fam)
    return plans


def _prefill(s: Session, needs: list) -> tuple[str, list[str], str]:
    """按声明顺序补齐缺失前置（已就绪则跳过，不重复连接）。

    低危前置（connect/set_mode）静默补齐；高危前置（enable）由框架并入该项确认。
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
        tip = ("⚠⚠ 极高风险：将永久改写 arm/end 关节零位标定（不可逆；每族还需二次确认）！"
               if level == "极高" else
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

# ---------------- 00 组（离线：构造与初值） ----------------
def _it00(s: Session):
    cls = s.cls
    ok = issubclass(cls, Backend)
    ev = f"模块 {cls.__module__}，类 {cls.__name__}，继承 Backend={ok}"
    return ("PASS" if ok else "FAIL"), ev


def _it01(s: Session):
    b = s.backend()  # 契约：构造不做任何总线 I/O
    ev = (f"完整 cfg 构造成功（无总线 I/O）：name={b.name!r}，"
          f"arm={b.n_joints_arm} 关节，end={b.n_joints_end} 关节")
    return "PASS", ev


def _it02(s: Session):
    b = s.cls({})  # 契约：空 cfg 可构造
    try:
        ok = (b.n_joints_arm == 0 and b.n_joints_end == 0
              and b.joint_limits_arm is None and b.joint_limits_end is None)
        ev = (f"空 cfg 构造成功：n_joints_arm={b.n_joints_arm}，n_joints_end={b.n_joints_end}，"
              f"joint_limits 均为 None={b.joint_limits_arm is None and b.joint_limits_end is None}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        b.close()


def _it03(s: Session):
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


def _it04(s: Session):
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


def _it05(s: Session):
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


def _it06(s: Session):
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


# ---------------- 10 组（真机：连接编排） ----------------
def _snap(b: Backend, fam: str) -> tuple:
    """族三值状态快照（连接/模式/使能），用于对侧族隔离断言对比。"""
    return (getattr(b, f"is_connected_{fam}"), getattr(b, f"mode_{fam}"),
            getattr(b, f"is_abled_{fam}"))


def _reset_disconnected(b: Backend) -> list[str]:
    """把会话实例复位到全断开态（仅在已使能/已连接时动作），返回已执行动作。

    连接编排项必须自全断开态出发；上电/上电可能残留使能，先失能再断连（安全方向）。
    """
    acts = []
    for fam in ("arm", "end"):
        if getattr(b, f"n_joints_{fam}") == 0:
            continue
        if getattr(b, f"is_abled_{fam}") is True:
            getattr(b, f"disable_{fam}")()
            acts.append(f"disable_{fam}")
        if getattr(b, f"is_connected_{fam}") is not False:
            getattr(b, f"disconnect_{fam}")()
            acts.append(f"disconnect_{fam}")
    return acts


def _check_connected(b: Backend, fam: str) -> tuple[str, bool]:
    conn = getattr(b, f"is_connected_{fam}")
    if conn is not True:
        return f"is_connected={conn}（连接失败——核心能力）", False
    exp = _cfg_default_mode(b, fam)
    mode = getattr(b, f"mode_{fam}")
    if mode != exp:
        return f"已连接但 mode={_mode_name(mode)} ≠ cfg 默认 {exp.name}（默认模式未回填）", False
    return f"is_connected=True，mode={exp.name}", True


def _check_disconnected(b: Backend, fam: str) -> tuple[str, bool]:
    conn = getattr(b, f"is_connected_{fam}")
    mode = getattr(b, f"mode_{fam}")
    abled = getattr(b, f"is_abled_{fam}")
    ok = conn is False and mode is None and abled is None
    return f"is_connected={conn}，mode={mode}，is_abled={abled}（缓存清空）", ok


def _seq_item(seq: list):
    """连接编排项：按 seq=[("connect"/"disconnect", 族), ...] 依序执行。

    每步断言本族三值状态（连=True+默认模式回填；断=False+缓存清空）与对侧族
    状态不受影响（共享单总线型号断一族不得影响另一族）；任一步失败即 FAIL 并继续。
    """
    def fn(s: Session):
        fams_needed = {fam for _, fam in seq}
        missing = sorted(f for f in fams_needed if _n_family_cfg(s.cfg, f) == 0)
        if missing:
            return "SKIP", f"{'、'.join(missing)} 族未配置关节（n=0），该连接编排无法执行"
        b = s.backend()
        acts = _reset_disconnected(b)
        if acts:
            print(f"（已复位到全断开态：{'、'.join(acts)}）")
        lines, fails = [], []
        for op, fam in seq:
            other = "end" if fam == "arm" else "arm"
            prev = _snap(b, other) if _n_family_cfg(s.cfg, other) > 0 else None
            i0 = _mark()
            if op == "connect":
                getattr(b, f"connect_{fam}")()
                ev, ok = _check_connected(b, fam)
            else:
                getattr(b, f"disconnect_{fam}")()
                ev, ok = _check_disconnected(b, fam)
            if not ok:
                ev += f"；warn：{_recent(_since(i0), 1)}"
            if prev is not None:
                iso = _snap(b, other) == prev
                ev += f"；对侧 {other} 不受影响={iso}"
                ok = ok and iso
            lines.append(f"{op}_{fam}: {'OK' if ok else 'FAIL'}（{ev}）")
            if not ok:
                fails.append(f"{op}_{fam}: {ev}")
        return ("FAIL" if fails else "PASS"), "；".join(lines)
    return fn


def _it16(s: Session):
    """重复连接重复断开：connect×2 / disconnect×2 均幂等——连接标志保持、断连缓存清空。"""
    b = s.backend()
    acts = _reset_disconnected(b)
    if acts:
        print(f"（已复位到全断开态：{'、'.join(acts)}）")
    results = []
    for fam in _both_fams(s):
        i0 = _mark()
        getattr(b, f"connect_{fam}")()
        getattr(b, f"connect_{fam}")()  # 重复连接
        conn2 = getattr(b, f"is_connected_{fam}")
        mode2 = getattr(b, f"mode_{fam}")
        getattr(b, f"disconnect_{fam}")()
        getattr(b, f"disconnect_{fam}")()  # 重复断开
        ev3, ok = _check_disconnected(b, fam)
        ok = ok and conn2 is True and mode2 == _cfg_default_mode(b, fam)
        ev = (f"connect×2 后 is_connected={conn2}，mode={_mode_name(mode2)}；"
              f"disconnect×2 后 {ev3}")
        if not ok:
            ev += f"；warn：{_recent(_since(i0), 1)}"
        results.append((fam, "PASS" if ok else "FAIL", ev))
    return _aggregate(results)


def _it17(s: Session):
    """断开重联：连→断→重连恢复（is_connected=True 且默认模式重新乐观回填），双族。"""
    b = s.backend()
    acts = _reset_disconnected(b)
    if acts:
        print(f"（已复位到全断开态：{'、'.join(acts)}）")
    results = []
    for fam in _both_fams(s):
        i0 = _mark()
        getattr(b, f"connect_{fam}")()
        c1 = getattr(b, f"is_connected_{fam}")
        getattr(b, f"disconnect_{fam}")()
        ev_d, ok_d = _check_disconnected(b, fam)
        i1 = _mark()
        getattr(b, f"connect_{fam}")()
        ev_c, ok_c = _check_connected(b, fam)
        ok = c1 is True and ok_d and ok_c
        ev = f"连（is_connected={c1}）→断（{ev_d}）→重连（{ev_c}）"
        if not ok:
            ev += f"；warn：{_recent(_since(i0) + _since(i1), 1)}"
        results.append((fam, "PASS" if ok else "FAIL", ev))
    return _aggregate(results)


# ---------------- 20 组（真机：只读） ----------------
def _fmt_state(st) -> str:
    """JointState 实测值单行摘要（None 字段以 — 表示；位数与 21 项表格一致）。"""
    parts = []
    for name, nd in (("q", 4), ("dq", 4), ("tau", 4), ("temp_mos", 1), ("temp_rotor", 1)):
        v = getattr(st, name)
        parts.append(f"{name}=" + ("—" if v is None
                                   else str(np.round(np.asarray(v, dtype=float), nd).tolist())))
    err = st.error
    parts.append("error=" + ("—" if err is None else str([int(e) for e in err])))
    return "；".join(parts)


def _it20(s: Session):
    """单次状态获取（双族）：输出实测值（q/dq/tau/温度/错误码），并校验返回值契约
    （在场字段 (n,) 且有限、t>0 且随调用更新）+ 成员变量更新（joint_state_* 与本次
    返回值同引用、旧引用冻结为快照）。"""
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        st1 = getattr(b, f"get_state_{fam}")()
        if st1 is None:
            results.append((fam, "SKIP", "get_state 返回 None（读失败/数据异常——见上方 warn）"))
            continue
        old = getattr(b, f"joint_state_{fam}")  # 成员变量（应与本次返回同引用）
        time.sleep(0.05)
        st2 = getattr(b, f"get_state_{fam}")()
        new = getattr(b, f"joint_state_{fam}")
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
        ref_ok = old is st1
        if st2 is not None:
            ref_ok = ref_ok and (new is st2) and (new is not old)
        if not ref_ok:
            fails.append("成员变量引用语义不符（joint_state_* 未与返回值同引用或旧引用未冻结）")
        t2 = f"{st2.t:.3f}" if st2 is not None else "None"
        ev = (f"实测：{_fmt_state(st1)}；t={st1.t:.3f}→{t2}（更新）；"
              f"成员变量与返回值同引用且旧引用冻结={'OK' if ref_ok else 'FAIL'}")
        results.append((fam, "FAIL" if fails else "PASS", ev + (f"；异常：{fails}" if fails else "")))
    return _aggregate(results)


def _it21(s: Session):
    """实时状态刷新监视（固定 10s，无参数）：连接后每秒读**被动快照**（不主动
    get_state），打印 arm/end 全部关节量表格——测试点正是基类低频刷新线程在无
    主动读取时持续更新快照；Ctrl+C 可提前结束（按已采样判定）。"""
    b = s.backend()
    fams = _both_fams(s)
    secs = WATCH_SECS
    names = {fam: [j.get("name", f"joint{i}")
                   for i, j in enumerate((s.cfg.get(fam) or {}).get("joints") or [])]
             for fam in fams}
    header = (f"{'族.关节':<14}{'q(rad)':>10}{'dq(rad/s)':>11}{'tau(N·m)':>10}"
              f"{'T_mos(℃)':>10}{'T_rot(℃)':>10}{'error':>7}{'快照龄(ms)':>10}")
    ages: dict[str, list[float]] = {fam: [] for fam in fams}
    print(f"── 实时状态刷新监视（固定 {secs}s，每秒采样被动快照 joint_state_*；Ctrl+C 提前结束）")
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


def _it22(s: Session):
    """控制模式获取（双族）：输出逐 joint 模式与族模式（get_mode 返回值），并校验
    成员变量 mode_* 及逐 joint 缓存与之一致。"""
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        i0 = _mark()
        ret = getattr(b, f"get_mode_{fam}")()
        # 测试基建：逐 joint 模式无公开访问器，读私有缓存 _joint_mode_* 仅作结果展示
        jm = list(getattr(b, f"_joint_mode_{fam}"))
        cache = getattr(b, f"mode_{fam}")
        jm_disp = "[" + "、".join(_mode_name(m) for m in jm) + "]"
        if isinstance(ret, ControlMode):
            ok = cache == ret and bool(jm) and all(m == ret for m in jm)
            ev = (f"单joint={jm_disp}；族模式 get_mode 返回={ret.name}；"
                  f"mode_{fam}={_mode_name(cache)}（一致={'OK' if ok else '矛盾'}）")
            results.append((fam, "PASS" if ok else "FAIL", ev))
        else:
            results.append((fam, "SKIP",
                            f"单joint={jm_disp}；族模式 get_mode 返回 None（读失败、各 joint 模式"
                            f"不一致或无回读能力——见上方 warn；mode_{fam}={_mode_name(cache)}）"))
    return _aggregate(results)


def _it23(s: Session):
    key = s.args.param
    if not key:
        return "SKIP", "未提供 --param（参数键由子类定义，须显式指定）"
    b = s.backend()
    results = []
    for fam in _both_fams(s):
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


def _it24(s: Session):
    """错误码读取（双族）：get_error 逐关节状态码（0=失能、1=使能、≥2 故障）。"""
    b = s.backend()
    time.sleep(0.3)  # 等低频刷新更新状态快照（get_error 读槽不发帧）
    results = []
    for fam in _both_fams(s):
        errs = getattr(b, f"get_error_{fam}")()
        if errs is None:
            results.append((fam, "SKIP",
                            "get_error 返回 None（快照过期或该硬件不提供状态码——非缺陷）"))
            continue
        n = getattr(b, f"n_joints_{fam}")
        codes = [int(e) for e in errs]
        fault = [c for c in codes if c >= 2]
        ok = len(codes) == n
        ev = f"get_error={codes}"
        ev += f"；⚠ 故障码 {fault}（硬件状态，非 backend 缺陷）" if fault else "（0=失能、1=使能，均正常）"
        if not ok:
            ev += f"；长度 {len(codes)} ≠ 关节数 {n}"
        results.append((fam, "PASS" if ok else "FAIL", ev))
    return _aggregate(results)


def _it25(s: Session):
    """错误码检查（双族）：check_error 返回 bool 且与 get_error 一致
    （全部 0/1→True；读到 None→False）。"""
    b = s.backend()
    time.sleep(0.3)  # 等低频刷新更新状态快照
    results = []
    for fam in _both_fams(s):
        errs = getattr(b, f"get_error_{fam}")()
        ret = getattr(b, f"check_error_{fam}")()
        if not isinstance(ret, bool):
            results.append((fam, "FAIL", f"check_error 返回非 bool：{ret!r}"))
            continue
        if errs is None:
            ok = ret is False
            results.append((fam, "PASS" if ok else "FAIL",
                            f"get_error=None（不可读），check_error={ret}"
                            f"（契约：读不到应 False，{'符合' if ok else '违反'}）"))
        else:
            fault = any(int(e) >= 2 for e in errs)
            ok = ret == (not fault)
            results.append((fam, "PASS" if ok else "FAIL",
                            f"check_error={ret}；get_error={[int(e) for e in errs]}"
                            f"（一致性={'OK' if ok else '矛盾'}）"))
    return _aggregate(results)


# ---------------- 30 组（真机：模式 / 使能 / 写参 / 设零） ----------------
def _modes_item(fam: str):
    """三模式循环设置：逐模式「先读（设前真值）→ 设 → 再读（设后真值）」，回读验证。"""
    def fn(s: Session):
        b = s.backend()
        lines, fails, any_ok = [], [], False
        for mode in ALL_MODES:
            before = getattr(b, f"get_mode_{fam}")()  # 先读
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


def _it32(s: Session):
    """set_mode 非法/未定义模式门禁（双族）：非法入参（字符串）与未被 ControlMode
    定义的模式（伪枚举）均被类型门禁拒绝（返回 0 + warn，不达内核）。"""
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        parts, ok_all = [], True
        for label, bad in (("字符串入参", "mit"), ("未定义模式", _UndefMode.UNDEFINED)):
            i0 = _mark()
            ret = getattr(b, f"set_mode_{fam}")(bad)
            hit = any(GATE_TYPE in m for m in _since(i0))
            ok = ret == 0 and hit
            ok_all = ok_all and ok
            parts.append(f"{label}→返回{ret}/门禁warn={hit}")
        results.append((fam, "PASS" if ok_all else "FAIL",
                        "、".join(parts) + "（均不达内核）"))
    return _aggregate(results)


def _power_item(fam: str):
    """使能和失能：读 is_abled → enable → 回读 → disable → 回读（返回值均应为 1）。"""
    def fn(s: Session):
        b = s.backend()
        i0 = _mark()
        pre = getattr(b, f"is_abled_{fam}")  # 读
        r_en = getattr(b, f"enable_{fam}")()  # 使能
        abled_en = getattr(b, f"is_abled_{fam}")  # 回读
        r_dis = getattr(b, f"disable_{fam}")()  # 失能
        abled_dis = getattr(b, f"is_abled_{fam}")  # 回读
        fails = []
        if r_en != 1:
            fails.append(f"enable 返回 {r_en}（核心能力失败；warn：{_recent(_since(i0))}）")
        if abled_en is not True:
            fails.append(f"使能后 is_abled={abled_en}（应 True）")
        if r_dis != 1:
            fails.append(f"disable 返回 {r_dis}")
        if abled_dis is not False:
            fails.append(f"失能后 is_abled={abled_dis}（应 False）")
        ev = (f"1.读 is_abled_{fam}={pre}（执行前）\n"
              f"2.使能 enable_{fam}() 返回 {r_en}\n"
              f"3.回读 is_abled_{fam}={abled_en}\n"
              f"4.失能 disable_{fam}() 返回 {r_dis}\n"
              f"5.回读 is_abled_{fam}={abled_dis}\n"
              f"——五步{'全部符合' if not fails else '存在不符'}")
        return ("FAIL" if fails else "PASS"), ev + (f"；异常：{fails}" if fails else "")
    return fn


def _it35(s: Session):
    """使能重复调用（双族）：enable×2 均返回 int 且首次为 1、终态 True；收尾 disable 复位安全态。"""
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        i0 = _mark()
        try:
            r1, r2 = getattr(b, f"enable_{fam}")(), getattr(b, f"enable_{fam}")()
            abled = getattr(b, f"is_abled_{fam}")
            rd = getattr(b, f"disable_{fam}")()  # 收尾复位安全态（不参与判定）
            ok = r1 == 1 and isinstance(r2, int) and abled is True
            ev = f"enable×2→({r1},{r2})，终态 is_abled={abled}；收尾 disable→{rd}"
            results.append((fam, "PASS" if ok else "FAIL",
                            ev + ("" if ok else f"；warn：{_recent(_since(i0), 1)}")))
        except Exception as e:
            results.append((fam, "FAIL", f"重复使能调用异常：{e!r}"))
    return _aggregate(results)


def _it36(s: Session):
    """失能重复调用（双族）：disable×2 均返回 1 且终态 False（幂等安全）。"""
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        i0 = _mark()
        try:
            r1, r2 = getattr(b, f"disable_{fam}")(), getattr(b, f"disable_{fam}")()
            abled = getattr(b, f"is_abled_{fam}")
            ok = r1 == 1 and r2 == 1 and abled is False
            ev = f"disable×2→({r1},{r2})，终态 is_abled={abled}"
            results.append((fam, "PASS" if ok else "FAIL",
                            ev + ("" if ok else f"；warn：{_recent(_since(i0), 1)}")))
        except Exception as e:
            results.append((fam, "FAIL", f"重复失能调用异常：{e!r}"))
    return _aggregate(results)


def _it37(s: Session):
    """写参数（双族）：读原值 → 写回原值 → 读回核对（无实际副作用的写入验证）。"""
    key = s.args.param
    if not key:
        return "SKIP", "未提供 --param，无法做写回核对"
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        orig = getattr(b, f"read_param_{fam}")(key)
        if orig is None:
            results.append((fam, "SKIP",
                            f"read_param({key!r}) 不可用（键对该族无效或未实现——无法获取原值）"))
            continue
        i0 = _mark()
        ret = getattr(b, f"write_param_{fam}")(key, orig)  # 写回原值
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


def _it38(s: Session):
    """设置零点（双族，极高风险）：每族「二次确认」后 set_zero 并回读关节位置≈0。

    仅显式 --step=38 执行（任何组展开不含本项）；任一确认非 yes 即跳过该族及
    后续（两族之间可中断）。
    """
    b = s.backend()
    fams = _both_fams(s)
    lines: list[str] = []
    fails: list[str] = []
    n_pass = 0
    aborted = False
    for fam in fams:
        if aborted:
            lines.append(f"{fam}: SKIP（前面族被跳过/中断，不再执行）")
            continue
        n = getattr(b, f"n_joints_{fam}")
        st = getattr(b, f"get_state_{fam}")()
        q_now = None if (st is None or st.q is None) else np.round(np.asarray(st.q, dtype=float), 4).tolist()
        print(f"  ⚠ 即将永久改写 {fam} 族 {n} 个关节的零位标定（当前 q={q_now}；"
              f"设零后当前位置被定义为 0，不可逆）")
        if not sys.stdin.isatty():
            lines.append(f"{fam}: SKIP（非终端自动跳过）")
            aborted = True
            continue
        ans = input("    第二次确认（yes=对该族执行设零，其他输入=跳过该族及后续）: ").strip().lower()
        if ans not in ("yes", "y"):
            lines.append(f"{fam}: SKIP（用户在二次确认跳过）")
            aborted = True
            continue
        i0 = _mark()
        ret = getattr(b, f"set_zero_{fam}")()
        msgs = _since(i0)
        if ret == 1:
            time.sleep(0.1)  # 等设零生效后状态可读
            st2 = getattr(b, f"get_state_{fam}")()
            q2 = None if (st2 is None or st2.q is None) else np.asarray(st2.q, dtype=float)
            if q2 is None:
                lines.append(f"{fam}: 设零返回 1，但回读位置不可读（无法核对归零）")
                fails.append(f"{fam} 设零后位置不可读")
            else:
                ok = bool(np.max(np.abs(q2)) <= ZERO_TOL)
                lines.append(f"{fam}: 设零成功，回读 q={np.round(q2, 4).tolist()}"
                             f"（|q|max={np.max(np.abs(q2)):.4f} ≤ {ZERO_TOL}={ok}）")
                if ok:
                    n_pass += 1
                else:
                    fails.append(f"{fam} 回读位置未归零")
        elif _cap_missing(msgs):
            lines.append(f"{fam}: SKIP（子类未实现设零内核——能力缺失）")
        else:
            lines.append(f"{fam}: 返回 0；warn：{_recent(msgs)}")
            fails.append(f"{fam} set_zero 返回 0")
    verdict = "FAIL" if fails else ("PASS" if n_pass else "SKIP")
    return verdict, "；".join(lines)


# ---------------- 40 组（真机：运动发送，高风险；均发送前预览 + 确认） ----------------
def _it40(s: Session):
    """send mit 零力矩（双族）：tau/kp/kd/q/dq 全 0 下发——电机不上力、自由悬垂。"""
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        n = getattr(b, f"n_joints_{fam}")
        i0 = _mark()
        getattr(b, f"send_mit_{fam}")(np.zeros(n), np.zeros(n), np.zeros(n),
                                      np.zeros(n), np.zeros(n))
        verdict = _classify_send(_since(i0))
        results.append((fam, verdict,
                        f"MIT 全零指令 ×{n} 关节（tau=0, q=0, dq=0, kp=0, kd=0）→ 自由悬垂"))
    return _aggregate(results)


def _it41(s: Session):
    """send mit 零点保持（双族）：以当前实测位置为 MIT 目标原地保持（保守增益）。"""
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        q0, src = _current_q(b, fam)
        if q0 is None:
            results.append((fam, "SKIP", f"{src}——无法取当前位置作保持目标"))
            continue
        n = getattr(b, f"n_joints_{fam}")
        i0 = _mark()
        getattr(b, f"send_mit_{fam}")(np.zeros(n), q0, np.zeros(n),
                                      np.full(n, SAFE_KP), np.full(n, SAFE_KD))
        verdict = _classify_send(_since(i0))
        time.sleep(SETTLE)
        st = getattr(b, f"get_state_{fam}")()
        ev = (f"目标 q=当前实测 {np.round(q0, 4).tolist()}（{src}；tau=0, dq=0, "
              f"kp={SAFE_KP}, kd={SAFE_KD}）原地保持")
        if st is not None and st.q is not None:
            drift = float(np.max(np.abs(np.asarray(st.q, dtype=float) - q0)))
            ev += f"；{SETTLE}s 后 |偏差|max={drift:.4f} rad"
        results.append((fam, verdict, ev))
    return _aggregate(results)


def _it42(s: Session):
    """send position（双族）：自动小幅目标（原幅度减半 + 限位裕量自动选向）。"""
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        n = getattr(b, f"n_joints_{fam}")
        q0, src = _current_q(b, fam)
        if q0 is None:
            results.append((fam, "SKIP", f"{src}——无法自实际位置小幅出发，盲发目标有大幅运动风险"))
            continue
        lim = getattr(b, f"joint_limits_{fam}")
        target = _safe_target(lim, q0, POS_AMP)
        i0 = _mark()
        getattr(b, f"send_position_{fam}")(target, np.full(n, SAFE_VLIM), np.full(n, SAFE_FLIM))
        verdict = _classify_send(_since(i0))
        time.sleep(SETTLE)
        st = getattr(b, f"get_state_{fam}")()
        ev = (f"目标 q={np.round(target, 4).tolist()}（自{src} 按限位裕量自动选向 ±{POS_AMP} rad，"
              f"vlim={SAFE_VLIM} rad/s，flim={SAFE_FLIM}）")
        if st is not None and st.q is not None:
            ev += f"；{SETTLE}s 后实测 q={np.round(np.asarray(st.q, dtype=float), 4).tolist()}"
        results.append((fam, verdict, ev))
    return _aggregate(results)


def _it43(s: Session):
    """send vel（双族）：低速短时下发（原幅度减半 + 按限位裕量逐关节选向 + 运动
    时间 VEL_DUR）后发 dq=0 停止。"""
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        n = getattr(b, f"n_joints_{fam}")
        q0, src = _current_q(b, fam)
        if q0 is None:
            results.append((fam, "SKIP", f"{src}——电机反馈异常时不宜做速度测试"))
            continue
        dq_cmd = _safe_vel_dirs(getattr(b, f"joint_limits_{fam}"), q0, VEL_TEST)
        i0 = _mark()
        getattr(b, f"send_vel_{fam}")(dq_cmd)
        m1 = _since(i0)
        time.sleep(VEL_DUR)
        st = getattr(b, f"get_state_{fam}")()
        i1 = _mark()
        getattr(b, f"send_vel_{fam}")(np.zeros(n))  # 停止
        m2 = _since(i1)
        verdict = _classify_send(m1 + m2)
        ev = (f"dq 指令={np.round(dq_cmd, 2).tolist()} rad/s（按限位裕量逐关节选向，"
              f"裕量不足侧不发、两端皆近为 0）持续 {VEL_DUR}s 后已发 dq=0 停止")
        if st is not None and st.dq is not None:
            ev += f"；运动中实测 dq={np.round(np.asarray(st.dq, dtype=float), 3).tolist()}"
        results.append((fam, verdict, ev))
    return _aggregate(results)


def _it44(s: Session):
    """send action end：离散动作下发（动作名经 --action 输入，默认 home）+ 到位延迟
    （0.2s 采样、连续 3 拍变化 <0.01 rad 提前结束，最长 4s）。"""
    acts = s.args.action or ["home"]
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
        ev = v
        if v == "PASS":
            # 动作目标由子类定义（上层不可知），以位置连续稳定判到位
            prev, trace, stable = None, [], 0
            for _ in range(20):
                time.sleep(0.2)
                st = b.get_state_end()
                q = None if (st is None or st.q is None) else np.asarray(st.q, dtype=float)
                if q is not None:
                    trace.append(np.round(q, 3).tolist())
                    moved = prev is not None and np.max(np.abs(q - prev)) >= 0.01
                    stable = 0 if moved else stable + 1
                    prev = q
                    if stable >= 3 and len(trace) >= 5:
                        break
            if trace:
                ev += f"；到位监测 {len(trace)}×0.2s：q 起 {trace[0]} → 末 {trace[-1]}"
            else:
                ev += "；执行期间状态不可读（未监测到位置）"
        lines.append(f"{a}: {ev}")
        if v == "FAIL":
            worst = "FAIL"
        elif v == "SKIP" and worst == "PASS":
            worst = "SKIP"
    return worst, "；".join(lines)


# ---------------- 50 组（真机：清错，高风险） ----------------
def _it50(s: Session):
    """清除错误（双族，四步序列）：1.使能 2.读错误码 3.清除错误 4.再读错误码核对。
    清错后全部关节码应 <2（0=失能、1=使能，≥2 故障；正常码下的清错链路验证）。"""
    b = s.backend()
    results = []
    for fam in _both_fams(s):
        i0 = _mark()
        lines: list[str] = []
        fails: list[str] = []
        # 1. 使能
        r_en = getattr(b, f"enable_{fam}")()
        abled = getattr(b, f"is_abled_{fam}")
        if r_en != 1 or abled is not True:
            results.append((fam, "FAIL",
                            f"1.使能 enable_{fam}() 返回 {r_en}，is_abled={abled}"
                            f"（核心能力失败）；warn：{_recent(_since(i0))}"))
            continue
        lines.append(f"1.使能 enable_{fam}() 返回 {r_en}，is_abled_{fam}={abled}")
        # 2. 读错误码
        time.sleep(0.2)  # 等使能生效与快照刷新（get_error 读槽不发帧）
        e1 = getattr(b, f"get_error_{fam}")()
        if e1 is None:
            lines.append("2.读错误码 get_error=None（不可读——快照过期或硬件不提供状态码）")
        else:
            codes1 = [int(e) for e in e1]
            fault1 = [c for c in codes1 if c >= 2]
            lines.append(f"2.读错误码 get_error={codes1}"
                         + (f"；⚠ 存在故障码 {fault1}（正待清除）" if fault1 else "（0=失能、1=使能，均正常）"))
        # 3. 清除错误
        i1 = _mark()
        r_cl = getattr(b, f"clear_error_{fam}")()
        msgs = _since(i1)
        if _cap_missing(msgs):
            lines.append("3.清除错误 子类未实现清错内核（能力缺失）")
            lines.append("4.（未执行）")
            results.append((fam, "SKIP", "\n".join(lines)))
            continue
        if r_cl is True:
            lines.append(f"3.清除错误 clear_error_{fam}() 返回 True（指令已发送）")
        else:
            lines.append(f"3.清除错误 clear_error_{fam}() 返回 {r_cl}（失败）；warn：{_recent(msgs)}")
            fails.append(f"clear_error_{fam} 返回 {r_cl}")
        # 4. 再读错误码核对
        time.sleep(0.2)
        e2 = getattr(b, f"get_error_{fam}")()
        if e2 is None:
            lines.append("4.读错误码 get_error=None（不可读，无法核对清除效果）")
            fails.append("清错后错误码不可读，无法核对")
        else:
            codes2 = [int(e) for e in e2]
            clean = all(c < 2 for c in codes2)
            lines.append(f"4.读错误码 get_error={codes2}（全部 <2={'OK' if clean else '仍存在故障码'}）")
            if not clean:
                fails.append(f"清错后仍存在故障码 {[c for c in codes2 if c >= 2]}（硬件故障持续或清除无效）")
        verdict = "FAIL" if fails else "PASS"
        results.append((fam, verdict, "\n".join(lines) + (f"\n——异常：{fails}" if fails else "")))
    return _aggregate(results)


# ---------------- 40 组发送前参数预览（预览 + 二次确认由框架统一执行） ----------------
def _preview40(s: Session):
    b = s.backend()
    for fam in _both_fams(s):
        n = getattr(b, f"n_joints_{fam}")
        print(f"  预览[{fam}]：MIT 全零指令（tau=0, q=0, dq=0, kp=0, kd=0）× {n} 关节 → 自由悬垂（电机不上力）")


def _preview41(s: Session):
    b = s.backend()
    for fam in _both_fams(s):
        q0, src = _current_q(b, fam)
        if q0 is None:
            print(f"  预览[{fam}]：{src}，将 SKIP")
            continue
        print(f"  预览[{fam}] 当前 q = {np.round(q0, 4).tolist()}（{src}）")
        print(f"           MIT 目标 q 同当前（tau=0, dq=0, kp={SAFE_KP}, kd={SAFE_KD}）→ 原地保持")


def _preview42(s: Session):
    b = s.backend()
    for fam in _both_fams(s):
        q0, src = _current_q(b, fam)
        if q0 is None:
            print(f"  预览[{fam}]：{src}，将 SKIP")
            continue
        target = _safe_target(getattr(b, f"joint_limits_{fam}"), q0, POS_AMP)
        print(f"  预览[{fam}] 当前 q = {np.round(q0, 4).tolist()}（{src}）")
        print(f"           目标 q = {np.round(target, 4).tolist()}（限位裕量自动选向 ±{POS_AMP} rad，"
              f"vlim={SAFE_VLIM} rad/s，flim={SAFE_FLIM}）")


def _preview43(s: Session):
    b = s.backend()
    for fam in _both_fams(s):
        q0, src = _current_q(b, fam)
        if q0 is None:
            print(f"  预览[{fam}]：{src}，将 SKIP")
            continue
        dq = _safe_vel_dirs(getattr(b, f"joint_limits_{fam}"), q0, VEL_TEST)
        q_pred = q0 + dq * VEL_DUR
        print(f"  预览[{fam}] 当前位置 q = {np.round(q0, 4).tolist()}（{src}）")
        print(f"           期望速度 dq = {np.round(dq, 2).tolist()} rad/s（按限位裕量逐关节选向）")
        print(f"           期望到达 q ≈ {np.round(q_pred, 4).tolist()}（{VEL_DUR}s 后，估算值），随后发 dq=0 停止")


def _preview44(s: Session):
    acts = s.args.action or ["home"]
    print(f"  预览[end]：离散动作 {acts}（动作目标由子类定义；执行后到位延迟=稳定判据等待，最长 4s）")


# ============================================================
# 测试项注册表（编号=两位数十位组+序号，风险递进；family 空/非空的族自动跳过）
# ============================================================
ITEMS: list[dict] = []


def _item(num, name, group, risk, family, purpose, needs, fn, requires_args=(), preview=None,
          hold=False):
    ITEMS.append(dict(num=num, name=name, group=group, risk=risk, family=family,
                      purpose=purpose, needs=needs, fn=fn, requires_args=tuple(requires_args),
                      preview=preview, hold=hold))


# —— 00 组（离线：backend 初始化验证）——
_item(0, "模块导入与子类发现", "init", "低", None,
      "验证 backend_<型号>.py 可发现，且其中的 Backend<型号> 类确为 Backend 子类", [], _it00)
_item(1, "完整 cfg 构造", "init", "低", None,
      "验证以 cfg backend 段构造成功且不做任何总线 I/O（离线可构造）", [], _it01)
_item(2, "空 cfg 构造容错", "init", "低", None,
      "验证空 cfg {} 可构造（基类对缺段/缺键全部容错的契约）", [], _it02)
_item(3, "运行参数非法回退", "init", "低", None,
      "验证 state_refresh_hz 等运行参数非法（非正数）时回退默认并 warn", [], _it03)
_item(4, "只读属性初值", "init", "低", None,
      "验证关节数/name 与 cfg 一致，is_connected*/is_abled*/mode* 初值均为 None（三值未知）", [], _it04)
_item(5, "硬限位解析核对", "init", "低", None,
      "验证 joint_limits_* 的 q_min/q_max/dq_max/tau_max 与 cfg 各关节四键一致（发送裁剪的唯一依据）", [], _it05)
_item(6, "joint_state 初值", "init", "低", None,
      "验证初始快照 t=0、q=NaN、error=-1（从未读取的约定标记）", [], _it06)

# —— 10 组（真机：连接编排）——
_item(10, "arm 连接→arm 断开", "connect", "低", "arm",
      "验证 connect_arm 后 is_connected_arm=True 且默认模式回填，disconnect_arm 后三值复位",
      [], _seq_item([("connect", "arm"), ("disconnect", "arm")]))
_item(11, "end 连接→end 断开", "connect", "低", "end",
      "验证末端族连接/断开契约（与 10 对称）",
      [], _seq_item([("connect", "end"), ("disconnect", "end")]))
_item(12, "arm 连→end 连→arm 断→end 断", "connect", "低", None,
      "双族先后连接、按先 arm 后 end 次序断开：逐步验证三值状态与对侧族隔离",
      [], _seq_item([("connect", "arm"), ("connect", "end"),
                     ("disconnect", "arm"), ("disconnect", "end")]))
_item(13, "arm 连→end 连→end 断→arm 断", "connect", "低", None,
      "双族先后连接、按先 end 后 arm 次序断开（断开次序相反）",
      [], _seq_item([("connect", "arm"), ("connect", "end"),
                     ("disconnect", "end"), ("disconnect", "arm")]))
_item(14, "end 连→arm 连→arm 断→end 断", "connect", "低", None,
      "连接次序相反（先 end 后 arm），断开先 arm 后 end",
      [], _seq_item([("connect", "end"), ("connect", "arm"),
                     ("disconnect", "arm"), ("disconnect", "end")]))
_item(15, "end 连→arm 连→end 断→arm 断", "connect", "低", None,
      "连接与断开次序均相反（先 end 后 arm）",
      [], _seq_item([("connect", "end"), ("connect", "arm"),
                     ("disconnect", "end"), ("disconnect", "arm")]))
_item(16, "重复连接重复断开", "connect", "低", None,
      "验证 connect×2 / disconnect×2 幂等：连接标志保持 True、断连后缓存清空（不崩溃）",
      [], _it16)
_item(17, "断开重联", "connect", "低", None,
      "验证断开后重连可恢复（is_connected=True 且默认模式重新乐观回填），双族",
      [], _it17)

# —— 20 组（真机：只读）——
_item(20, "单次状态获取", "read", "低", None,
      "双族验证 get_state：输出实测值（q/dq/tau/温度/错误码），并校验返回值契约"
      "（在场字段 (n,) 且有限、t>0 且更新）+ 成员变量 joint_state_* 更新"
      "（与返回值同引用、旧引用冻结）",
      [("connected", "both")], _it20)
_item(21, "实时状态刷新监视（10s）", "read", "低", None,
      "验证连接后基类低频刷新线程持续更新快照：固定 10s 每秒打印 arm/end 全部关节量表"
      "（被动快照，无 CLI 参数；全程快照龄 <500ms 判 PASS）",
      [("connected", "both")], _it21)
_item(22, "控制模式获取", "read", "低", None,
      "双族输出逐 joint 模式 + 族模式（get_mode 返回值），并校验成员变量 mode_* 及逐 joint "
      "缓存一致（读失败/各 joint 不一致/无回读能力 SKIP）",
      [("connected", "both")], _it22)
_item(23, "参数读取", "read", "低", None,
      "双族验证参数读取返回逐关节标量 list（须 --param 指定子类定义的键；不可用则 SKIP）",
      [("connected", "both")], _it23, requires_args=("param",))
_item(24, "错误码读取", "read", "低", None,
      "双族验证 get_error 逐关节状态码（0=失能、1=使能、≥2 故障；不可读 SKIP 非缺陷）",
      [("connected", "both")], _it24)
_item(25, "错误码检查", "read", "低", None,
      "双族验证 check_error 返回 bool 且与 get_error 一致（全 0/1→True；读不到→False）",
      [("connected", "both")], _it25)

# —— 30 组（真机：模式/使能/写参/设零）——
_item(30, "arm 三模式设置", "ctrl", "低", "arm",
      "循环设 MIT/POSITION/VELOCITY：逐模式先读→设→再读（回读验证；切换 POSITION/VELOCITY 随模式写增益）",
      [("connected", "arm")], _modes_item("arm"))
_item(31, "end 三模式设置", "ctrl", "低", "end",
      "末端族三模式循环设置（与 30 对称；至少一种可用即 PASS）",
      [("connected", "end")], _modes_item("end"))
_item(32, "set_mode 非法/未定义模式门禁", "ctrl", "低", None,
      "双族验证非法入参（字符串）与未被 ControlMode 定义的模式（伪枚举）均被类型门禁拒绝"
      "（返回 0 + warn，不达内核）",
      [("connected", "both")], _it32)
_item(33, "arm 使能和失能", "ctrl", "高", "arm",
      "验证读 is_abled→enable（返回 1、回读 True）→disable（返回 1、回读 False）（电机上力）",
      [("connected", "arm")], _power_item("arm"))
_item(34, "end 使能和失能", "ctrl", "高", "end",
      "末端族使能/失能契约（与 33 对称）",
      [("connected", "end")], _power_item("end"))
_item(35, "使能重复调用", "ctrl", "高", None,
      "双族验证 enable×2 安全（首次返回 1、重复返回 int 不崩溃、终态 True；收尾自动失能复位）",
      [("connected", "both")], _it35)
_item(36, "失能重复调用", "ctrl", "高", None,
      "双族验证 disable×2 幂等（均返回 1、终态 False）",
      [("connected", "both")], _it36)
_item(37, "写参数", "ctrl", "高", None,
      "双族验证参数写入：读原值→写回原值→读回核对（无副作用；须 --param）",
      [("connected", "both")], _it37, requires_args=("param",))
_item(38, "设置零点", "ctrl", "极高", None,
      "【永久改写零位标定，不可逆】双族逐族：二次确认后 set_zero 并回读关节位置≈0"
      "（容差 0.05 rad）；仅显式 --step=38 执行（任何组不含本项；每族二次确认，"
      "任一次非 yes 即跳过该族及后续）",
      [("connected", "both")], _it38)

# —— 40 组（真机：运动发送，高风险；五项均发送前预览+确认，--yes 不放行）——
_item(40, "send mit 零力矩", "motion", "高", None,
      "双族 MIT 全零指令（tau=0, q=0, dq=0, kp=0, kd=0）→ 自由悬垂（电机不上力）",
      [("connected", "both"), ("mode", "both", ControlMode.MIT), ("enabled", "both")],
      _it40, preview=_preview40, hold=True)
_item(41, "send mit 零点保持", "motion", "高", None,
      "双族以当前实测位置为 MIT 目标原地保持（tau=0, dq=0, kp=5, kd=0.5 保守增益）",
      [("connected", "both"), ("mode", "both", ControlMode.MIT), ("enabled", "both")],
      _it41, preview=_preview41, hold=True)
_item(42, "send position", "motion", "高", None,
      "双族位置指令：自当前位置自动小幅 ±0.025 rad（原幅度减半，按限位裕量自动选向——"
      "零位在极限位置时自动往行程内走），限速 0.5 rad/s",
      [("connected", "both"), ("mode", "both", ControlMode.POSITION), ("enabled", "both")],
      _it42, preview=_preview42, hold=True)
_item(43, "send vel", "motion", "高", None,
      "双族速度指令：0.1 rad/s（原幅度减半）按限位裕量逐关节选向 × 0.5s（运动时间）"
      "后发 dq=0 停止",
      [("connected", "both"), ("mode", "both", ControlMode.VELOCITY), ("enabled", "both")],
      _it43, preview=_preview43, hold=True)
_item(44, "send action end", "motion", "高", "end",
      "末端离散动作下发（--action 输入动作名，默认 home）+ 到位延迟（稳定判据等待，最长 4s）",
      [("connected", "end"), ("mode", "end", ControlMode.POSITION), ("enabled", "end")],
      _it44, preview=_preview44, hold=True)

# —— 50 组（真机：清错，高风险）——
_item(50, "清除错误", "error", "高", None,
      "双族四步序列：1.使能 2.读错误码 3.清除错误 4.再读错误码核对（清后全部 <2；"
      "未实现清错内核 SKIP 属能力缺失）",
      [("connected", "both")], _it50)

ALL_NUMS = {it["num"] for it in ITEMS}
_BY_NUM = {it["num"]: it for it in ITEMS}

# 组名 → 编号集合（38 极高危排除出 ctrl/online/all，仅显式编号执行）
GROUPS: dict[str, list[int]] = {
    "init": [0, 1, 2, 3, 4, 5, 6],
    "connect": [10, 11, 12, 13, 14, 15, 16, 17],
    "read": [20, 21, 22, 23, 24, 25],
    "ctrl": [30, 31, 32, 33, 34, 35, 36, 37],
    "motion": [40, 41, 42, 43, 44],
    "error": [50],
}
GROUPS["offline"] = GROUPS["init"]
GROUPS["online"] = (GROUPS["connect"] + GROUPS["read"] + GROUPS["ctrl"]
                    + GROUPS["motion"] + GROUPS["error"])
GROUPS["all"] = GROUPS["offline"] + GROUPS["online"]


# ============================================================
# 子类发现与配置加载
# ============================================================
def discover(sub: str) -> type:
    """解析子类：优先查官方注册表 REGISTRY，再按文件约定导入 backend_<sub> 寻找 Backend 子类。"""
    from joyarm_core.backend import REGISTRY  # 官方注册路径，键名 = 文件名如 "backend_dm"
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
    """解析 --step：编号（支持 00/0 两种写法）与组名逗号混选。"""
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
    """单项执行：打印项与目的 → 风险确认 → 前置补齐 → （运动项）预览+二次确认 → 执行 → 打印结果。"""
    num = it["num"]
    print()
    print(f"[{num:02d}] {it['name']} | 组={it['group']} | 风险={it['risk']}")
    print(f"目的：{it['purpose']}")
    if it["family"] and _n_family_cfg(s.cfg, it["family"]) == 0:
        ev = f"{it['family']} 族未配置任何关节（n=0），空族自动跳过"
        print(f"[{num:02d}] SKIP —— {ev}")
        return "SKIP", ev
    missing = [k for k in it.get("requires_args", ()) if not getattr(s.args, k, None)]
    if missing:
        ev = (f"未提供 --{'/'.join(missing)}（该值由子类定义，须显式指定后方可测试）")
        print(f"[{num:02d}] SKIP —— {ev}")
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
        print(f"[{num:02d}] SKIP —— {ev}")
        return "QUIT", ev
    if ans == "skip":
        if not sys.stdin.isatty() and level in ("高", "极高"):
            ev = "非终端环境自动跳过（高危永不自动执行；交互终端下需输入 yes）"
        else:
            ev = f"用户跳过（风险={level}）"
        print(f"[{num:02d}] SKIP —— {ev}")
        return "SKIP", ev
    status, actions, reason = _prefill(s, it["needs"])
    if status == "fail":
        ev = f"前置补齐失败：{'；'.join(actions) or '（无）'}——{reason}"
        print(f"[{num:02d}] FAIL —— {ev}")
        return "FAIL", ev
    if status == "skip":
        ev = f"前置不可用：{reason}"
        print(f"[{num:02d}] SKIP —— {ev}")
        return "SKIP", ev
    if actions:
        print(f"（已自动补齐前置：{'；'.join(actions)}）")
    if it.get("preview"):
        # 运动参数预览 + 二次确认：前置就绪后列出当前/期望值，用户确认才发送
        try:
            it["preview"](s)
        except Exception as e:
            print(f"（预览失败：{type(e).__name__}: {e}）")
        if not sys.stdin.isatty():
            ans2 = "skip"
        else:
            ans2 = input("⚠ 以上为即将下发的运动参数，输入 yes 执行发送（回车/s=跳过  q=退出）: ").strip().lower()
        if ans2 in ("q", "quit", "退出"):
            ev = "用户在预览后退出（q）"
            print(f"[{num:02d}] SKIP —— {ev}")
            return "QUIT", ev
        if ans2 not in ("yes", "y"):
            ev = "用户在预览后跳过发送"
            print(f"[{num:02d}] SKIP —— {ev}")
            return "SKIP", ev
    try:
        verdict, evidence = it["fn"](s)
    except Exception as e:
        verdict, evidence = "FAIL", f"未预期异常 {type(e).__name__}: {e}"
    print(f"[{num:02d}] {verdict} —— {evidence}")
    if it.get("hold") and sys.stdin.isatty():
        # 运动项销毁保持：指令维持中（零力矩悬垂/原地保持/位置/0 速/夹爪位），
        # 等用户观察完毕按回车，才进入失能→断连→close 收尾（非终端无此保持）
        input("（指令维持中——观察完毕后按回车销毁：失能→断连→close）")
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
    print("Backend 子类测试验证项清单（编号=两位数十位组+序号，按风险递进；arm/end 对称覆盖）")
    print("=" * 68)
    for it in sorted(ITEMS, key=lambda x: x["num"]):
        tag = "　★极高危：仅显式 --step=编号 执行，不含于 ctrl/online/all" if it["risk"] == "极高" else ""
        print(f"[{it['num']:02d}] {it['name']}｜组={it['group']}｜风险={it['risk']}{tag}")
        print(f"      目的：{it['purpose']}")
    print("=" * 68)
    print("基础组：init(00) / connect(10) / read(20) / ctrl(30) / motion(40) / error(50)")
    print("复合组：offline=init（--step 缺省）｜online=真机全组（不含 38）｜all=offline+online（不含 38）")
    print("通用机制：能力缺失=SKIP（非缺陷）；空族自动跳过；参数项须 --param、动作项须 --action（默认 home）")


EPILOG = """\
示例：
  python joyarm_core/backend/backend_verify.py --help                       # 本说明
  python joyarm_core/backend/backend_verify.py --list                       # 全部测试项
  python joyarm_core/backend/backend_verify.py --sub=dm                     # 00 组（缺省 offline）
  python joyarm_core/backend/backend_verify.py --sub=dm --step=all          # 全部（不含 38）
  python joyarm_core/backend/backend_verify.py --sub=dm --step=12           # 单项
  python joyarm_core/backend/backend_verify.py --sub=dm --step=00,20,read   # 编号/组名混选
  python joyarm_core/backend/backend_verify.py --sub=dm --step=connect,read # 组合
  python joyarm_core/backend/backend_verify.py --sub=dm --step=online --param=<键> --action=open,close

风险与确认策略（与 verify_backend.py 相同）：
  低危：直接执行（初始化 / 连接编排 / 只读）。
  中危：交互回车确认（--yes 或非终端环境直接执行；本清单暂无中危项，机制保留）。
  高危：强制交互输入 yes（--yes 不放行；非终端自动 SKIP——agent 无法全自动执行高危项）。
  极高危（38 设置零点）：永久改写零位标定，仅显式 --step=38 执行，且每族各自二次确认。

判定（能力三分法）：
  PASS=符合基类契约；SKIP=能力缺失（该子类/硬件不提供，非缺陷）；FAIL=违反契约。
  前置自动补齐：单项可独立执行（如 --step=33 会自动 connect + enable，高危前置并入确认）。
  运动组（40-44）：发送前参数预览 + 二次确认（--yes 不放行）。

退出码：全 PASS/SKIP=0；任一 FAIL=1；用法错误=2。结尾输出 SUMMARY: total=… pass=… fail=… skip=…（机器可解析）。
"""


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="backend_verify.py",
        description=("Backend 子类全方位测试验证（上层视角）：只调 Backend 基类公开 API，"
                     "对任意子类（关节电机/舵机/混合硬件）做分组分项测试（verify_backend.py 迭代版，"
                     "两位数十进制分组）。能力三分法（能力缺失=SKIP）+ 空族自动跳过 + 风险分级确认。"),
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sub", help="子型简称（如 dm、feetech）：经 REGISTRY 或文件约定导入"
                                 " backend_<型号>.py 并寻找 Backend<型号> 类，默认加载"
                                 " config/joyarm_<型号>.yaml 的 backend 段")
    p.add_argument("--step", default="offline",
                   help="数字=单项编号（如 38）；字符串=组名（init/connect/read/ctrl/motion/error/"
                        "offline/online/all）；逗号分隔混选；缺省 offline")
    p.add_argument("--config", help="yaml 配置文件路径（默认 joyarm_core/config/joyarm_<sub>.yaml）")
    p.add_argument("--param", help="参数读取/写入测试（23/37）的参数键（键表由子类定义；缺省相关项 SKIP）")
    p.add_argument("--action", help="send_action_end 测试（44）的离散动作名（逗号分隔；默认仅测 home）")
    p.add_argument("--list", action="store_true", help="仅列出全部测试项（编号/组/风险/目的），不执行")
    p.add_argument("--yes", action="store_true",
                   help="跳过中危交互确认（高危/极高危永不自动执行：仍需交互 yes，非终端自动 SKIP）")
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
    args.action = [a.strip() for a in (args.action or "").split(",") if a.strip()] or ["home"]
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
    print(f"Backend 子类测试验证：{cls.__name__}（backend_{args.sub}.py）")
    print(f"配置：{cfg_path}（arm {_n_family_cfg(cfg, 'arm')} 关节 / end {_n_family_cfg(cfg, 'end')} 关节）")
    print(f"交互终端：{'是' if tty else '否'}；--yes={args.yes}；"
          f"高危策略：{'交互 yes（--yes 不放行）' if tty else '非终端自动 SKIP'}")
    print(f"待执行 {len(nums)} 项：{[f'{n:02d}' for n in nums]}")
    results: list[tuple[dict, str, str]] = []
    quit_flag = False
    try:
        for num in nums:
            it = _BY_NUM[num]
            if quit_flag:
                ev = "用户提前退出（q，未执行）"
                print(f"[{num:02d}] SKIP —— {ev}")
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
            print(f"  FAIL [{it['num']:02d}] {it['name']}：{_trunc(ev)}")
        concl = f"（详见上方 {n_fail} 条 FAIL 证据；其余 pass={n_pass}、skip={n_skip}）"
    print(concl)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
