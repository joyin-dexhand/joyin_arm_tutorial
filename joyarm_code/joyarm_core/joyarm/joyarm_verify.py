"""JoyArm 分层逐项验收：默认离线 DummyBackend，显式 --real 才连接真机。

【执行顺序】
    L0 基础线程 / 工厂 / 配置自检 / 只读模型
    L1 硬件后端公开委托契约（Dummy）
    L2 求解器 / 数学门面 / 目标通道 / 验收判据自检
    L3 FSM 基础迁移 / 自愈 / 失败恢复 / 完整运动管线
    L4 安全监视 / 保护抢占 / 控制器切换与停止并发
    L5 可视化 HTTP 服务 / 降级 / 重复启停
    L6 close / with 退出 / 析构与资源回收

    编号用于稳定定位，功能组用于筛选，level 决定执行顺序。原有 43 项编号
    保留，新增 16 项近期契约，总计 59 项离线验收。FSM 互斥31/32移至L4，
    目标写入65/66移至L2，避免依赖上层功能的检查抢在基础验证前执行。
    新增项覆盖：周期停止不多跑一拍/超时防双启动，控制器选择不切模式，
    自愈一致零重复写/临时失败恢复/持续失败不改事件，首帧返回False与异常回收，
    ZERO到位失能/HOLD锁定位置，保护事件抢占，部分启动恢复，停止与自愈并发，
    目标经plan/ctrl实际到达后端，以及陈旧反馈/完全不动不被验收为PASS。

【逐项与逐级判定】
    每项打印层级/内容目的/结果/证据；各项自带 finally 收尾，故障注入局部恢复。
    当前层出现 FAIL 后继续测同层其他项，默认跳过更高层；已选择的收尾项仍执行。
    --keep-going 可继续高层诊断。每项结束显式关闭本项 Dummy 刷新线程，结束时
    恢复 REGISTRY/load_config 注入；不依赖上一项的硬件、缓存或线程残留。
    PASS=已测契约满足；FAIL=违反契约；SKIP=未验证（环境/能力/依赖/用户跳过），
    SKIP不能当作通过。退出码：任一FAIL=1，用法错误=2，其余=0。
    结尾 SUMMARY / LEVEL / GROUP 行提供机器可解析统计。

【真机】
    保留 r0连接只读→r1使能模式参数→r2小幅指令→r3状态机→r4监视可视化→r5清错。
    前置缺失自动补齐，前置风险并入项级确认；低危直执，中危回车确认（--yes可跳），
    高危/极高危仍须交互输入yes，非终端自动SKIP，--yes不放行。
    运动项发送前预览并再次确认；反馈须新鲜有效，目标通道131须实际位移并在
    0.01rad容差内到位。151设零永久改写标定，逐族二次确认，仅显式 --step=151
    选择；r5/rall/--level/--through均不包含151。真机未注入Dummy。

【环境与使用】
    # 首次环境安装（Python >=3.10,<3.12；uv未安装时先安装uv）
    cd <仓库>/joyarm_code
    uv sync
    source .venv/bin/activate

    python joyarm_core/joyarm/joyarm_verify.py --list
    python joyarm_core/joyarm/joyarm_verify.py                  # 全部，由底到高
    python joyarm_core/joyarm/joyarm_verify.py --level L3        # 仅L3
    python joyarm_core/joyarm/joyarm_verify.py --through L3      # L0→L3
    python joyarm_core/joyarm/joyarm_verify.py --step 68,69,74    # 单项定位
    python joyarm_core/joyarm/joyarm_verify.py --step fsm        # 功能组，可跨层
    python joyarm_core/joyarm/joyarm_verify.py --real --list     # 只列清单，不连接
    python joyarm_core/joyarm/joyarm_verify.py --real --through r2
    python joyarm_core/joyarm/joyarm_verify.py --real --step 151

    --step / --level / --through互斥；--list可与选择器组合。完整参数见 --help。
    源码直接可运行，不依赖pytest；L0～L4可脱离图形/HTTP服务完成基础验收。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import copy
import logging
import os
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

# 离线验收不弹监控窗口（start_state_viz 默认自动弹 PySide6 窗口，此处仅测服务面）
os.environ.setdefault("JOYARM_VIZ_NOWINDOW", "1")

# 未安装 joyarm_core 时兜底（本脚本可直接 python 运行；仓库源优先于任何安装副本）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pinocchio as pin
import yaml

from joyarm_core.backend import get_backend
from joyarm_core.backend.backend import Backend
from joyarm_core.joyarm import joyArm
from joyarm_core.joyarm import joyarm as _jm_mod   # joyarm.py 模块（JoyArm 类内 load_config 全局）
import joyarm_core.joyarm as _joyarm_pkg           # joyarm 包（JoyArmFactory/listArm/_available_models）
from joyarm_core.joyarm.joyarm import JoyArm
from joyarm_core.robotics.control import Controller
from joyarm_core.utils.transforms import quat_conj, quat_mul, quat_to_axis_angle
from joyarm_core.utils.types import ControlMode, JointState, MotionState, Pose, SafetyEvent

__all__ = ["main"]

MODEL = "joyarm_dm"                      # 被测型号（真实 config + robot_model 资产）
DUMMY_KEY = "backend_verify_dummy"       # DummyBackend 临时注册名（注入用，用完恢复）
VERIFY_CTRL = "verify_test_controller"   # ctrl 选择算法/频率（模式由 FSM 管理）验证用临时控制器注册名

# FSM / 事件类用例的状态等待上限（秒；fsm_hz=50，正常迁移 ≤ 数拍）
_WAIT_STATE = 3.0
# monitor 类用例的事件等待上限（秒；monitor_hz=50）
_WAIT_EVENT = 2.0

# 仅验收程序管理 Dummy 生命周期，逐项结束时显式停止其刷新线程。
_CASE_BACKENDS: list = []


# ---------------- 日志捕获（joyarm_core 全包，含 joyarm/backend/robotics logger） ----------------
class _Cap(logging.Handler):
    """收集 joyarm_core 包内全部日志消息，供测试证据与 warn 判定。"""

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


def _hit(msgs: list[str], *subs: str) -> bool:
    """增量日志中是否存在同时包含全部子串的消息。"""
    return any(all(t in m for t in subs) for m in msgs)


def _trunc(text: str, k: int = 130) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= k else text[:k] + "…"


def quat_err(q1, q2) -> float:
    """两单位四元数姿态差角（rad，双倍角公式）。"""
    return float(quat_to_axis_angle(quat_mul(quat_conj(np.asarray(q1, float)),
                                             np.asarray(q2, float)))[1])


# ============================================================
# 自包含 DummyBackend：最小 Backend 子类（29 个抽象内核全部脚本化）
# ============================================================
class DummyBackend(Backend):
    """验收用 Dummy 后端（自包含实现，不 import 测试文件）。

    行为脚本化：

    - `calls`：内核调用记录 `(内核名, 参数...)`（`recording=False` 时停记，
      供高频拍降噪声；判定一律用水位后增量，见 :meth:`calls_since`）；
    - `modes` / `jstates`：逐关节模式与状态字典可编程（读时注入
      `t=time.time()`，使基类状态装配拿到新鲜时间戳、不触发回退 warn）；
    - `registers`：参数读写寄存器模拟；
    - `fail_at`：`{内核名: None|{joint 下标}}` 命中抛异常（`None`=整族），
      `ret_false` 同型命中返回 False；
    - `follow`：True 时 arm 位置指令一步到位跟随（模拟理想电机到位，供 FSM
      到位门槛 / stop 回零推演；False 时 jstates 冻结在脚本注入值）。
    """

    def __init__(self, cfg):
        self.calls: list = []
        self.recording: bool = True
        self.fail_at: dict = {}
        self.ret_false: dict = {}
        self.registers: dict = {}
        self.follow: bool = False
        super().__init__(cfg)
        _CASE_BACKENDS.append(self)
        self.modes = {f: [ControlMode.POSITION] * self._n(f) for f in ("arm", "end")}
        # jstates 默认：q=0（零位）、error=1（使能码，供刷新线程推导 is_enabled=True）
        self.jstates = {f: [{"q": 0.0, "dq": 0.0, "tau": 0.0,
                             "temp_mos": 30.0, "temp_rotor": 30.0,
                             "error": 1} for _ in range(self._n(f))]
                        for f in ("arm", "end")}

    # ---- 测试助手 ----
    def _rec(self, *entry):
        if self.recording:
            self.calls.append(entry)

    def _hit(self, name, i=None):
        """脚本化失败注入：命中 fail_at 抛异常；返回是否命中 ret_false。"""
        if name in self.fail_at and (i is None or self.fail_at[name] is None
                                     or i in self.fail_at[name]):
            raise RuntimeError(f"{name} scripted fail")
        return name in self.ret_false and (i is None or self.ret_false[name] is None
                                           or i in self.ret_false[name])

    def set_q(self, fam: str, q) -> None:
        """脚本化整族关节位置（follow=False 时的状态注入）。"""
        for i, v in enumerate(np.asarray(q, dtype=float).reshape(-1)):
            self.jstates[fam][i]["q"] = float(v)
            self.jstates[fam][i]["dq"] = 0.0

    def calls_since(self, i0: int, name: str) -> list:
        """水位后某内核名的增量调用记录。"""
        return [c for c in self.calls[i0:] if c and c[0] == name]

    # ---- 生命周期内核（arm/end 对称） ----
    def _connect_arm(self, channel, protocol):
        self._rec("connect_arm", channel, protocol)
        return not self._hit("connect_arm")

    def _connect_end(self, channel, protocol):
        self._rec("connect_end", channel, protocol)
        return not self._hit("connect_end")

    def _disconnect_arm(self):
        self._rec("disconnect_arm")
        self._hit("disconnect_arm")

    def _disconnect_end(self):
        self._rec("disconnect_end")
        self._hit("disconnect_end")

    def _enable_joint_arm(self, i):
        self._rec("enable_joint_arm", i)
        return not self._hit("enable_joint_arm", i)

    def _enable_joint_end(self, i):
        self._rec("enable_joint_end", i)
        return not self._hit("enable_joint_end", i)

    def _disable_joint_arm(self, i):
        self._rec("disable_joint_arm", i)
        return not self._hit("disable_joint_arm", i)

    def _disable_joint_end(self, i):
        self._rec("disable_joint_end", i)
        return not self._hit("disable_joint_end", i)

    # ---- 读取内核 ----
    def _read_joint_param_arm(self, i, key):
        self._rec("read_joint_param_arm", i, key)
        self._hit("read_joint_param_arm", i)
        return self.registers.get(("arm", i, key), 0.0)

    def _read_joint_param_end(self, i, key):
        self._rec("read_joint_param_end", i, key)
        self._hit("read_joint_param_end", i)
        return self.registers.get(("end", i, key), 0.0)

    def _read_joint_mode_arm(self, i):
        self._rec("read_joint_mode_arm", i)
        self._hit("read_joint_mode_arm", i)
        return self.modes["arm"][i]

    def _read_joint_mode_end(self, i):
        self._rec("read_joint_mode_end", i)
        self._hit("read_joint_mode_end", i)
        return self.modes["end"][i]

    def _read_joint_state_arm(self, i):
        self._rec("read_joint_state_arm", i)
        self._hit("read_joint_state_arm", i)
        snap = dict(self.jstates["arm"][i])
        snap["t"] = time.time()          # 新鲜时间戳（防基类装配回退 warn 刷屏）
        return snap

    def _read_joint_state_end(self, i):
        self._rec("read_joint_state_end", i)
        self._hit("read_joint_state_end", i)
        snap = dict(self.jstates["end"][i])
        snap["t"] = time.time()
        return snap

    # ---- 写入内核 ----
    def _write_joint_param_arm(self, i, key, value):
        self._rec("write_joint_param_arm", i, key, value)
        self._hit("write_joint_param_arm", i)
        self.registers[("arm", i, key)] = value

    def _write_joint_param_end(self, i, key, value):
        self._rec("write_joint_param_end", i, key, value)
        self._hit("write_joint_param_end", i)
        self.registers[("end", i, key)] = value

    def _set_joint_mode_arm(self, i, mode):
        self._rec("set_joint_mode_arm", i, mode)
        self._hit("set_joint_mode_arm", i)
        self.modes["arm"][i] = mode

    def _set_joint_mode_end(self, i, mode):
        self._rec("set_joint_mode_end", i, mode)
        self._hit("set_joint_mode_end", i)
        self.modes["end"][i] = mode

    def _set_joint_zero_arm(self, i):
        self._rec("set_joint_zero_arm", i)
        return not self._hit("set_joint_zero_arm", i)

    def _set_joint_zero_end(self, i):
        self._rec("set_joint_zero_end", i)
        return not self._hit("set_joint_zero_end", i)

    # ---- 发送内核 ----
    def _send_joint_mit_arm(self, i, tau, q, dq, kp, kd):
        self._rec("send_joint_mit_arm", i, tau, q, dq, kp, kd)
        self._hit("send_joint_mit_arm", i)

    def _send_joint_mit_end(self, i, tau, q, dq, kp, kd):
        self._rec("send_joint_mit_end", i, tau, q, dq, kp, kd)
        self._hit("send_joint_mit_end", i)

    def _send_joint_position_arm(self, i, q, vlim, flim):
        self._rec("send_joint_position_arm", i, q, vlim, flim)
        self._hit("send_joint_position_arm", i)
        if self.follow:                   # 理想电机：一步到位
            self.jstates["arm"][i]["q"] = float(q)
            self.jstates["arm"][i]["dq"] = 0.0

    def _send_joint_position_end(self, i, q, vlim, flim):
        self._rec("send_joint_position_end", i, q, vlim, flim)
        self._hit("send_joint_position_end", i)

    def _send_joint_vel_arm(self, i, dq):
        self._rec("send_joint_vel_arm", i, dq)
        self._hit("send_joint_vel_arm", i)

    def _send_joint_vel_end(self, i, dq):
        self._rec("send_joint_vel_end", i, dq)
        self._hit("send_joint_vel_end", i)

    def _send_action_end(self, action, **kwargs):
        self._rec("send_action_end", action, tuple(sorted(kwargs.items())))
        self._hit("send_action_end")

    # ---- 清错内核 ----
    def _clear_joint_error_arm(self, i):
        self._rec("clear_joint_error_arm", i)
        return not self._hit("clear_joint_error_arm", i)

    def _clear_joint_error_end(self, i):
        self._rec("clear_joint_error_end", i)
        return not self._hit("clear_joint_error_end", i)


# ctrl 选择算法/频率（模式由 FSM 管理）验证用临时控制器（MODE=VELOCITY，与默认 POSITION 控制器可区分）
class _VerifyTestController(Controller):
    """仅验收用：MODE=VELOCITY 的哑控制器，模式由 MOTION 进入/维持核验。"""

    MODE = ControlMode.VELOCITY

    def step(self, arm, target, jointstate) -> bool:
        return True


# ============================================================
# Dummy 注入机制（临时改 cfg.backend.name + 替换 REGISTRY，用完恢复）
# ============================================================
class _CfgSource:
    """load_config 替身：next 非 None 时返回该 cfg（一次性），否则返回基础注入 cfg。"""

    def __init__(self):
        self.base: dict = {}
        self.next: dict | None = None

    def __call__(self, name: str) -> dict:
        cfg = self.next if self.next is not None else self.base
        self.next = None
        return copy.deepcopy(cfg)


CFG_SOURCE = _CfgSource()


class _AttrPatch:
    """模块属性替换记录（持对象引用，恢复可靠）。"""

    def __init__(self, obj, name, old):
        self.obj = obj
        self.name = name
        self.old = old

    def restore(self):
        setattr(self.obj, self.name, self.old)


class _DictAddPatch:
    """注册表 dict 原地加键记录。

    joyarm.py 在 import 时以**值引用**捕获各域 REGISTRY（`_DOMAIN_REGISTRIES`），
    替换模块属性不影响已捕获引用——必须向原 dict 原地加键才能同时被
    `_check_cfg`/`_build_domains` 与 `get_backend` 看到。
    """

    def __init__(self, dct, key, value):
        self.dct = dct
        self.key = key
        self.value = value
        self.existed = key in dct

    def apply(self):
        if not self.existed:
            self.dct[self.key] = self.value

    def restore(self):
        if not self.existed:
            self.dct.pop(self.key, None)


_PATCHES: list = []


def install_injection() -> None:
    """装载注入（用完经 :func:`restore_injection` 恢复，不触碰仓库文件）：

    - backend.REGISTRY / control.REGISTRY **原地**加 Dummy 与测试控制器；
    - 两处 load_config 引用换成 _CfgSource（joyarm.py 模块内 JoyArm 类用的
      全局 + joyarm 包内 JoyArmFactory.create 用的全局）。"""
    import joyarm_core.backend as backend_pkg
    import joyarm_core.robotics.control as control_pkg
    adds = [_DictAddPatch(backend_pkg.REGISTRY, DUMMY_KEY, DummyBackend),
            _DictAddPatch(control_pkg.REGISTRY, VERIFY_CTRL, _VerifyTestController)]
    for p in adds:
        p.apply()
        _PATCHES.append(p)
    for obj, name, val in ((_jm_mod, "load_config", CFG_SOURCE),
                           (_joyarm_pkg, "load_config", CFG_SOURCE)):
        _PATCHES.append(_AttrPatch(obj, name, getattr(obj, name)))
        setattr(obj, name, val)


def restore_injection() -> None:
    """恢复全部注入（与 install_injection 对偶；重复调用安全）。"""
    _cleanup_backends()
    while _PATCHES:
        _PATCHES.pop().restore()
    CFG_SOURCE.next = None


def _cleanup_backends(start: int = 0) -> None:
    """回收当前验收项创建的 Dummy 刷新线程；SDK 本身的生命周期语义不变。"""
    created = _CASE_BACKENDS[start:]
    del _CASE_BACKENDS[start:]
    for backend in created:
        backend.close()


def _tweak_cfg(cfg: dict) -> dict:
    """注入 cfg 的验收定制（不改变被测行为语义，只让判定可区分 / 时长可控）：

    - backend.name → DUMMY_KEY（Dummy 注入点）；
    - runtime.move_wait_timeout → 1.0 s（默认 10 s，stop 超时强推项时长可控）；
    - traj 两规划器 plan_hz 拉开（hold 10 / cubic 20，切换联动可判）；
    - control 追加 verify_test_controller（ctrl_hz=100，选择/切换与 FSM MODE 自愈可判）。"""
    cfg = copy.deepcopy(cfg)
    cfg["backend"]["name"] = DUMMY_KEY
    cfg["joyarm"].setdefault("runtime", {})["move_wait_timeout"] = 1.0
    for spec in cfg["robotics"]["trajectory"]:
        spec["plan_hz"] = 10.0 if spec["name"] == "single_joint_hold_traj_planner" else 20.0
    cfg["robotics"]["control"].append({"name": VERIFY_CTRL, "ctrl_hz": 100.0})
    return cfg


def load_base_cfg() -> dict:
    """读真实 joyarm_dm.yaml 全量 cfg（验收的基准配置）。"""
    path = Path(__file__).resolve().parents[1] / "config" / f"{MODEL}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_arm(mutate=None) -> tuple[JoyArm, DummyBackend]:
    """构造注入 DummyBackend 的 JoyArm（每次全新实例）。

    :param mutate: 对注入 cfg 的追加改写函数（坏 cfg 变体用）。
    :return: (JoyArm 实例, 其内部 DummyBackend 引用——本脚本注入的测试替身，
        取私有引用仅作证据源，不作为被测行为)。
    """
    cfg = _tweak_cfg(load_base_cfg())
    if mutate is not None:
        mutate(cfg)
    CFG_SOURCE.next = cfg
    arm = JoyArm(MODEL)
    return arm, arm._backend


# ============================================================
# 通用小工具
# ============================================================
def _wait_state(arm: JoyArm, want: MotionState, timeout: float = _WAIT_STATE) -> bool:
    """轮询等待 FSM 迁移到目标状态。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if arm.get_current_motion_state() is want:
            return True
        time.sleep(0.02)
    return arm.get_current_motion_state() is want


def _wait_event(arm: JoyArm, want: SafetyEvent, timeout: float = _WAIT_EVENT) -> bool:
    """轮询等待安全事件。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if arm.get_event() is want:
            return True
        time.sleep(0.02)
    return arm.get_event() is want


def _settle_fsm(arm: JoyArm, d: DummyBackend) -> None:
    """FSM/monitor 项统一收尾：q 置 home + follow 开（回零路径最短）→ stop（异常记录不吞）。"""
    try:
        d.follow = True
        d.set_q("arm", arm.home_arm)
        arm.stop_motion_fsm()
    except Exception as e:  # noqa: BLE001（验收脚本须捕获一切）
        logging.warning("joyarm_verify - _settle_fsm：stop_motion_fsm 收尾异常（记录不吞）：%r", e)
    try:
        arm.stop_monitor()
    except Exception as e:  # noqa: BLE001（验收脚本须捕获一切）
        logging.warning("joyarm_verify - _settle_fsm：stop_monitor 收尾异常（记录不吞）：%r", e)
    d.follow = False


def _joyarm_thread_names() -> list[str]:
    """清点当前存活的 joyarm 自有线程名（fsm/monitor/viz/plan/ctrl/refresh）。"""
    names = ("joyarm-fsm", "joyarm-monitor", "joyarm-viz", "joyarm-viz-http",
             "joyarm-plan", "joyarm-ctrl", "backend-state-refresh")
    return [t.name for t in threading.enumerate() if t.name in names]


def _exc(fn, *a, **kw):
    """执行并返回 (异常实例 or None)。"""
    try:
        fn(*a, **kw)
        return None
    except Exception as e:  # noqa: BLE001（验收脚本须捕获一切）
        return e


# ============================================================
# 测试项实现（fn() -> (verdict, evidence)，verdict ∈ PASS/FAIL/SKIP）
# ============================================================

# ---------------- 00 组（工厂与 listArm） ----------------
def _it00():
    """未知名：临时禁用注入直连真实路径——工厂 ①config/②backend 错误消息均含可用型号列表。"""
    from joyarm_core.joyarm import _available_models
    # 禁用注入：注入生效时 load_config 永不抛错、backend 名必在册，①/② 两条错误
    # 消息路径结构性不可达（失败总落在 ③ robot_model）；本项须真实走到前两步
    restore_injection()
    try:
        models = _available_models()
        e1 = _exc(joyArm, "no_such_model_xyz")    # ① config：真实仓库无此 yaml
        ok1 = (isinstance(e1, ValueError) and "① config" in str(e1)
               and "可用型号" in str(e1) and all(m in str(e1) for m in models))
        # ② backend：临时目录放 badbackend.yaml（三键齐全、backend.name 不在真实
        #    REGISTRY）+ good.yaml（四路径齐全）→ 可用列表非空、断言有内容可对
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            cfg_dir = tdir / "config"
            cfg_dir.mkdir()
            (cfg_dir / "good.yaml").write_text(
                "joyarm: {frame_end: f}\nrobotics: {}\nbackend: {name: backend_dm}\n",
                encoding="utf-8")
            (cfg_dir / "badbackend.yaml").write_text(
                "joyarm: {frame_end: f}\nrobotics: {}\nbackend: {name: nope}\n",
                encoding="utf-8")
            good = tdir / "robot_model" / "good"
            (good / "meshes").mkdir(parents=True)
            (good / "urdf").mkdir()
            (good / "urdf" / "good.urdf").write_text("<robot/>", encoding="utf-8")
            old = (_joyarm_pkg._CONFIG_DIR, _joyarm_pkg._ROBOT_MODEL_DIR,
                   _jm_mod._CONFIG_DIR)
            _joyarm_pkg._CONFIG_DIR = str(cfg_dir)   # _available_models / 路径条件
            _joyarm_pkg._ROBOT_MODEL_DIR = str(tdir / "robot_model")
            _jm_mod._CONFIG_DIR = str(cfg_dir)       # 真实 load_config 读临时目录
            try:
                avail = _available_models()
                e2 = _exc(joyArm, "badbackend")      # ② backend：nope 不在 REGISTRY
            finally:
                (_joyarm_pkg._CONFIG_DIR, _joyarm_pkg._ROBOT_MODEL_DIR,
                 _jm_mod._CONFIG_DIR) = old
        ok2 = (isinstance(e2, ValueError) and "② backend" in str(e2)
               and avail == ["good"] and "good" in str(e2))
        e3 = _exc(get_backend, "no_such_backend")
        ok3 = isinstance(e3, ValueError) and "可用" in str(e3)
        ok = ok1 and ok2 and ok3
        ev = (f"禁用注入直连真实路径：joyArm('no_such_model_xyz') → {type(e1).__name__}，"
              f"① config 校验失败且消息含可用型号 {models}={ok1}；临时目录 "
              f"joyArm('badbackend') → {type(e2).__name__}，② backend 校验失败且消息"
              f"附可用列表 {avail}={ok2}；get_backend('no_such_backend') → "
              f"{type(e3).__name__}，消息含可用注册名={ok3}；"
              f"详情：{_trunc(e1, 110)}｜{_trunc(e2, 110)}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        install_injection()                       # 恢复注入（后续项依赖）


def _it01():
    """listArm 三条件（临时目录变量控制：三键/注册名/四路径 + 语法错与空文件）。"""
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        cfg_dir = tdir / "config"
        cfg_dir.mkdir()
        (cfg_dir / "good.yaml").write_text(
            "joyarm: {frame_end: f}\nrobotics: {}\nbackend: {name: backend_dm}\n",
            encoding="utf-8")
        for nm, text in (("nokeys.yaml", "joyarm: {frame_end: f}\n"),
                         ("badbackend.yaml",
                          "joyarm: {frame_end: f}\nrobotics: {}\nbackend: {name: nope}\n"),
                         ("nopath.yaml",
                          "joyarm: {frame_end: f}\nrobotics: {}\nbackend: {name: backend_dm}\n"),
                         ("broken.yaml", "::: [\n"),
                         ("empty.yaml", "")):
            (cfg_dir / nm).write_text(text, encoding="utf-8")
        # good 型号的四路径齐全（模拟 robot_model 资产）
        good = tdir / "robot_model" / "good"
        (good / "meshes").mkdir(parents=True)
        (good / "urdf").mkdir()
        (good / "urdf" / "good.urdf").write_text("<robot/>", encoding="utf-8")
        old_cfg_dir, old_model_dir = _joyarm_pkg._CONFIG_DIR, _joyarm_pkg._ROBOT_MODEL_DIR
        _joyarm_pkg._CONFIG_DIR = str(cfg_dir)
        _joyarm_pkg._ROBOT_MODEL_DIR = str(tdir / "robot_model")
        try:
            got = _joyarm_pkg._available_models()
        finally:
            _joyarm_pkg._CONFIG_DIR, _joyarm_pkg._ROBOT_MODEL_DIR = old_cfg_dir, old_model_dir
    ok = got == ["good"]
    ev = (f"临时目录 6 个 yaml（good/nokeys/badbackend/nopath/broken/empty），"
          f"_available_models()={got}（期望 ['good']：条件1=三顶层键、"
          f"条件2=backend.name∈REGISTRY、条件3=四路径齐全，逐条件否决）")
    return ("PASS" if ok else "FAIL"), ev


def _it02():
    """模板排除（真实仓库）：listArm() 不含 joyarm_template、含 joyarm_dm。"""
    models = _joyarm_pkg.listArm()        # 公开入口（打印一行可用型号）
    tmpl_path = Path(__file__).resolve().parents[1] / "robot_model" / "joyarm_template"
    ok = "joyarm_dm" in models and "joyarm_template" not in models
    ev = (f"listArm()（真实仓库）={models}；joyarm_template 排除="
          f"{('joyarm_template' not in models)}（robot_model/joyarm_template 目录存在="
          f"{tmpl_path.exists()}——无目录由路径条件自然排除，不做文件名特判）")
    return ("PASS" if ok else "FAIL"), ev


def _it03():
    """工厂成功路径（Dummy 注入 + 真实 joyarm_dm 资产）：四步校验过 + 创建成功 info。"""
    i0 = _mark()
    cfg = _tweak_cfg(load_base_cfg())
    CFG_SOURCE.next = cfg
    arm = joyArm(MODEL)                       # 工厂入口（JoyArmFactory.create 四步校验）
    d = arm._backend
    try:
        ok = (isinstance(arm, JoyArm) and MODEL in repr(arm)
              and _hit(_since(i0), "创建成功")
              and _hit(_since(i0), "初始化和自检完毕")
              and isinstance(d, DummyBackend) and d.n_joints_arm == arm.n_arm)
        ev = (f"joyArm('{MODEL}')（Dummy 注入）→ {type(arm).__name__}，"
              f"repr={_trunc(repr(arm), 90)}；创建成功 info={_hit(_since(i0), '创建成功')}；"
              f"自检完成 info={_hit(_since(i0), '初始化和自检完毕')}；"
              f"内部 backend 为 DummyBackend={isinstance(d, DummyBackend)}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


# ---------------- 01 组（初始化自检：坏 cfg 逐条异常） ----------------
def _bad_cfg_item(label, mutate, *markers):
    """坏 cfg 变体通用项：构造须抛 ValueError 且消息含全部 marker。"""
    def fn():
        e = _exc(make_arm, mutate)
        ok = isinstance(e, ValueError) and all(m in str(e) for m in markers)
        ev = (f"{label} → {type(e).__name__}；消息含标记 "
              f"{list(markers)}={all(m in str(e) for m in markers)}；"
              f"详情：{_trunc(e, 170) if e else '（未抛异常——违反自检契约）'}")
        return ("PASS" if ok else "FAIL"), ev
    return fn


def _m_del_top(cfg):
    del cfg["robotics"]


def _m_bad_reg(cfg):
    cfg["robotics"]["fkine"][0]["name"] = "no_such_fkine"


def _m_no_plan_hz(cfg):
    cfg["robotics"]["trajectory"][0].pop("plan_hz", None)


def _m_frac_hz(cfg):
    cfg["robotics"]["trajectory"][0]["plan_hz"] = 10.5


def _m_home_out(cfg):
    cfg["joyarm"]["home_arm"][0] = 9.9


def _m_soft_over(cfg):
    cfg["joyarm"]["joint_limits_soft_arm"]["q_min"][0] = -9.9


def _m_urdf_lim(cfg):
    cfg["backend"]["arm"]["joints"][0]["q_min"] = 1.234


def _m_home_nan(cfg):
    cfg["joyarm"]["home_arm"][0] = float("nan")


def _m_soft_null(cfg):
    cfg["joyarm"]["joint_limits_soft_arm"]["q_max"][1] = None


def _m_ikine_nan(cfg):
    cfg["robotics"]["ikine"][0]["damping"] = float("nan")


def _it17():
    """数值叶 NaN / null 拒绝构造：不再静默冻进 home/软限位/求解器参数后『安静地坏掉』。"""
    e1 = _exc(make_arm, _m_home_nan)
    e2 = _exc(make_arm, _m_soft_null)
    e3 = _exc(make_arm, _m_ikine_nan)
    ok1 = (isinstance(e1, ValueError) and "非有限数值" in str(e1)
           and "home_arm[0]" in str(e1))
    ok2 = (isinstance(e2, ValueError) and "为 null" in str(e2)
           and "q_max[1]" in str(e2))
    ok3 = (isinstance(e3, ValueError) and "非有限数值" in str(e3)
           and "robotics.ikine[0].damping" in str(e3))
    ok = ok1 and ok2 and ok3
    ev = (f"home_arm[0]=NaN → {type(e1).__name__}，消息含『非有限数值』且带 cfg 路径"
          f"home_arm[0]={ok1}（无此拦截则 NaN 冻进 home：到位判定永假、FSM 永驻 HOME）；"
          f"软限位 q_max[1]=null → {type(e2).__name__}，消息含『为 null』且带路径"
          f"q_max[1]={ok2}；robotics.ikine[0].damping=NaN → {type(e3).__name__}，"
          f"消息带路径 robotics.ikine[0].damping={ok3}（求解器构造参数同为 cfg 数值叶）；"
          f"详情：{_trunc(e1, 120)}｜{_trunc(e2, 120)}｜{_trunc(e3, 120)}")
    return ("PASS" if ok else "FAIL"), ev


# ---------------- 02 组（只读属性与防篡改） ----------------
def _it20():
    """writeable=False：全部对外数组原地写均抛异常（防篡改）。"""
    arm, _ = make_arm()
    try:
        targets = [
            ("home_arm", arm.home_arm), ("zero_arm", arm.zero_arm),
            ("neutral_arm", arm.neutral_arm), ("home_end", arm.home_end),
            ("zero_end", arm.zero_end), ("neutral_end", arm.neutral_end),
            ("hard_arm.q_min", arm.joint_limits_hard_arm.q_min),
            ("hard_arm.q_max", arm.joint_limits_hard_arm.q_max),
            ("hard_arm.dq_max", arm.joint_limits_hard_arm.dq_max),
            ("hard_arm.tau_max", arm.joint_limits_hard_arm.tau_max),
            ("soft_arm.q_min", arm.joint_limits_soft_arm.q_min),
            ("soft_end.q_max", arm.joint_limits_soft_end.q_max),
        ]
        if arm.tcp_limits.workspace_box is not None:
            targets.append(("tcp_limits.workspace_box", arm.tcp_limits.workspace_box))
        results = []
        for name, arr in targets:
            if arr is None or arr.size == 0:
                continue
            e = _exc(lambda a=arr: a.__setitem__(0, 123.456))
            results.append((name, e is not None))
        ok = all(r for _, r in results)
        bad = [n for n, r in results if not r]
        ev = (f"{len(results)} 个数组全部只读（原地写抛异常）："
              f"{[n for n, _ in results]}；未拦截={bad or '无'}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


def _it21():
    """get_config 深拷贝：外部篡改返回值不反哺内部。"""
    arm, _ = make_arm()
    try:
        c1 = arm.get_config()
        home0 = float(arm.home_arm[0])
        qmin0 = float(c1["backend"]["arm"]["joints"][0]["q_min"])
        c1["joyarm"]["home_arm"][0] = 999.0
        c1["backend"]["arm"]["joints"][0]["q_min"] = 999.0
        c1["joyarm"]["frame_end"] = "hacked"
        c2 = arm.get_config()
        ok = (float(arm.home_arm[0]) == home0
              and float(c2["joyarm"]["home_arm"][0]) == home0
              and float(c2["backend"]["arm"]["joints"][0]["q_min"]) == qmin0
              and c2["joyarm"]["frame_end"] != "hacked"
              and c1 is not c2)
        ev = (f"篡改 get_config() 返回值后：arm.home_arm[0]={arm.home_arm[0]}（原 {home0}），"
              f"二次 get_config 的 home_arm[0]={c2['joyarm']['home_arm'][0]}、"
              f"q_min={c2['backend']['arm']['joints'][0]['q_min']}（原 {qmin0}）、"
              f"frame_end={c2['joyarm']['frame_end']!r}；两次返回独立对象={c1 is not c2}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


def _it22():
    """只读 @property 零拷贝返本体：多次访问同引用、与 get_config 值一致。"""
    arm, _ = make_arm()
    try:
        cfg = arm.get_config()
        checks = [
            ("home_arm", arm.home_arm is arm.home_arm,
             np.allclose(arm.home_arm, cfg["joyarm"]["home_arm"])),
            ("joint_limits_hard_arm", arm.joint_limits_hard_arm is arm.joint_limits_hard_arm,
             np.allclose(arm.joint_limits_hard_arm.q_min,
                         [j["q_min"] for j in cfg["backend"]["arm"]["joints"]])),
            ("pin_model", arm.pin_model is arm.pin_model,
             arm.pin_model.nq == arm.n_arm),
            ("tcp_limits", arm.tcp_limits is arm.tcp_limits,
             np.allclose(arm.tcp_limits.workspace_box,
                         cfg["joyarm"]["tcp_limits"]["workspace_box"])),
        ]
        ok = all(a and b for _, a, b in checks)
        ev = "；".join(f"{n}：同引用={a}，值一致={b}" for n, a, b in checks)
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


def _it23():
    """get_joint_names 1 基边界：'all'/1/n 正常取值；0/n+1/非法类型 warn 返回 None。"""
    arm, _ = make_arm()
    try:
        n = arm.n_arm
        names = arm.get_joint_names_arm()
        i0 = _mark()
        got1 = arm.get_joint_names_arm(1)
        gotn = arm.get_joint_names_arm(n)
        bad = (arm.get_joint_names_arm(0), arm.get_joint_names_arm(n + 1),
               arm.get_joint_names_arm("nope"), arm.get_joint_names_arm(True))
        n_warn = sum("非法" in m for m in _since(i0))
        names_end = arm.get_joint_names_end()
        ok_end = (isinstance(names_end, tuple) and len(names_end) == arm.n_end
                  and (arm.n_end == 0
                       or arm.get_joint_names_end(arm.n_end) == names_end[-1]))
        ok = (isinstance(names, tuple) and len(names) == n
              and got1 == names[0] and gotn == names[-1]
              and all(b is None for b in bad) and n_warn == len(bad) and ok_end)
        ev = (f"arm：'all'→{n} 元 tuple，joint=1→{got1!r}（=首元）、joint={n}→"
              f"{gotn!r}（=末元，1 基序号口径）；越界/非法（0、{n + 1}、'nope'、"
              f"True）均返回 None 且逐次 warn（{n_warn}/{len(bad)}）；end：'all'→"
              f"{len(names_end)} 元 tuple、1 基序号 {arm.n_end} 取末元一致={ok_end}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


# ---------------- 03 组（委托面与互斥） ----------------
def _it30():
    """委托面透传核对：JoyArm 方法 → DummyBackend 对应内核（调用名 + 关键参数）。"""
    arm, d = make_arm()
    try:
        arm.connect_arm()
        n = arm.n_arm
        checks: list[tuple[str, bool, str]] = []

        def chk(label, cond, detail=""):
            checks.append((label, bool(cond), detail))

        i0 = len(d.calls)
        r = arm.enable_arm()
        chk("enable_arm→enable_joint_arm×n",
            r == 1 and len(d.calls_since(i0, "enable_joint_arm")) == n, f"ret={r}")
        i0 = len(d.calls)
        r = arm.set_mode_arm(ControlMode.VELOCITY)
        chk("set_mode_arm→set_joint_mode_arm×n（Dummy modes 更新）",
            r == 1 and len(d.calls_since(i0, "set_joint_mode_arm")) == n
            and all(m is ControlMode.VELOCITY for m in d.modes["arm"]),
            f"ret={r}, modes[0]={d.modes['arm'][0].name}")
        i0 = len(d.calls)
        m = arm.get_mode_arm()
        chk("get_mode_arm→read_joint_mode_arm×n",
            m is ControlMode.VELOCITY and len(d.calls_since(i0, "read_joint_mode_arm")) == n)
        i0 = len(d.calls)
        arm.send_vel_arm(np.full(n, 0.05))
        got = d.calls_since(i0, "send_joint_vel_arm")
        chk("send_vel_arm→send_joint_vel_arm×n（参数一致）",
            len(got) == n and all(abs(c[2] - 0.05) < 1e-12 for c in got),
            f"{len(got)} 帧, dq[0]={got[0][2] if got else None}")
        i0 = len(d.calls)
        arm.set_mode_arm(ControlMode.POSITION)
        q_send = np.clip(arm.home_arm + 0.02, arm.joint_limits_hard_arm.q_min,
                         arm.joint_limits_hard_arm.q_max)
        arm.send_position_arm(q_send)
        got = d.calls_since(i0, "send_joint_position_arm")
        chk("send_position_arm→send_joint_position_arm×n（参数一致）",
            len(got) == n and all(np.allclose(c[2], q_send[c[1]]) for c in got), f"{len(got)} 帧")
        i0 = len(d.calls)
        arm.set_mode_arm(ControlMode.MIT)
        arm.send_mit_arm(q=np.zeros(n), kp=np.full(n, 5.0), kd=np.full(n, 0.5))
        got = d.calls_since(i0, "send_joint_mit_arm")
        chk("send_mit_arm→send_joint_mit_arm×n（kp/kd 透传）",
            len(got) == n and all(abs(c[5] - 5.0) < 1e-9 and abs(c[6] - 0.5) < 1e-9
                                  for c in got), f"{len(got)} 帧")
        i0 = len(d.calls)
        r = arm.set_zero_arm()
        chk("set_zero_arm→set_joint_zero_arm×n",
            r == 1 and len(d.calls_since(i0, "set_joint_zero_arm")) == n)
        i0 = len(d.calls)
        r = arm.clear_error_arm()
        chk("clear_error_arm→clear_joint_error_arm×n",
            r is True and len(d.calls_since(i0, "clear_joint_error_arm")) == n)
        i0 = len(d.calls)
        vals = arm.read_param_arm("kp")
        chk("read_param_arm→read_joint_param_arm×n",
            vals is not None and len(vals) == n
            and len(d.calls_since(i0, "read_joint_param_arm")) == n)
        i0 = len(d.calls)
        r = arm.write_param_arm("kp", [1.5] * n)
        chk("write_param_arm→write_joint_param_arm×n（寄存器落值）",
            r == 1 and len(d.calls_since(i0, "write_joint_param_arm")) == n
            and d.registers.get(("arm", 0, "kp")) == 1.5)
        i0 = len(d.calls)
        st = arm.get_state_arm()
        chk("get_state_arm→read_joint_state_arm×n",
            st is not None and st.q.shape == (n,)
            and len(d.calls_since(i0, "read_joint_state_arm")) >= n)
        # ---- end 侧对称（end 侧不受互斥） ----
        i0 = len(d.calls)
        arm.connect_end()
        arm.set_action_end("home")
        got = d.calls_since(i0, "send_action_end")
        chk("set_action_end→send_action_end('home')",
            any(c[1] == "home" for c in got), f"{[c[1] for c in got]}")
        ok = all(c for _, c, _ in checks)
        ev = "；".join(f"{lb}={'OK' if c else 'FAIL'}（{dt}）" if dt else f"{lb}={'OK' if c else 'FAIL'}"
                      for lb, c, dt in checks)
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


def _it31():
    """FSM 外部指令互斥：FSM 运行中外部线程 arm 侧指令空操作 + warn；只读/end 侧放行。"""
    arm, d = make_arm()
    try:
        d.follow = False
        d.set_q("arm", arm.neutral_arm)     # 非 home：HOME 门槛拦住 MOTION，驻留 HOME
        arm.connect_arm()
        arm.start_motion_fsm()
        if not _wait_state(arm, MotionState.HOME):
            return "FAIL", f"FSM 未启动（current={arm.get_current_motion_state().name}）"
        arm.set_event(SafetyEvent.DISABLE)  # DISABLE 维持态（不发位置帧，增量判定干净）
        if not _wait_state(arm, MotionState.DISABLE):
            return "FAIL", f"DISABLE 直达失败（current={arm.get_current_motion_state().name}）"
        time.sleep(0.1)
        checks = []
        i0, m0 = len(d.calls), _mark()
        arm.send_position_arm(np.full(arm.n_arm, 0.111))
        checks.append(("send_position_arm 拦截（无内核帧 + warn）",
                       len(d.calls_since(i0, "send_joint_position_arm")) == 0
                       and _hit(_since(m0), "FSM 运行中", "互斥拦截")))
        i0, m0 = len(d.calls), _mark()
        r = arm.set_mode_arm(ControlMode.MIT)
        checks.append(("set_mode_arm 拦截（ret=0 + 无内核帧）",
                       r == 0 and len(d.calls_since(i0, "set_joint_mode_arm")) == 0))
        i0 = len(d.calls)
        r = arm.enable_arm()
        checks.append(("enable_arm 拦截（ret=0）",
                       r == 0 and len(d.calls_since(i0, "enable_joint_arm")) == 0))
        i0 = len(d.calls)
        r = arm.clear_error_arm()
        checks.append(("clear_error_arm 拦截（ret=False）",
                       r is False and len(d.calls_since(i0, "clear_joint_error_arm")) == 0))
        i0 = len(d.calls)
        r = arm.set_zero_arm()
        checks.append(("set_zero_arm 拦截（ret=0）",
                       r == 0 and len(d.calls_since(i0, "set_joint_zero_arm")) == 0))
        # 互斥放行面：只读 get_* 与 end 侧方法
        arm.connect_end()
        i0 = len(d.calls)
        st = arm.get_state_arm()
        r_end = arm.set_mode_end(ControlMode.MIT)
        checks.append(("get_state_arm / set_mode_end 放行",
                       st is not None and r_end == 1
                       and len(d.calls_since(i0, "read_joint_state_arm")) >= arm.n_arm
                       and len(d.calls_since(i0, "set_joint_mode_end")) >= 1))
        ok = all(c for _, c in checks)
        ev = "；".join(f"{lb}={'OK' if c else 'FAIL'}" for lb, c in checks)
        return ("PASS" if ok else "FAIL"), ev
    finally:
        _settle_fsm(arm, d)
        arm.close()


def _it32():
    """ctrl 线程互斥豁免：MOTION 态 ctrl 线程自身的 send_position_arm 照常到达 backend。"""
    arm, d = make_arm()
    try:
        d.follow = True
        d.set_q("arm", arm.home_arm)
        arm.connect_arm()
        arm.start_motion_fsm()
        if not _wait_state(arm, MotionState.MOTION):
            return "FAIL", (f"未进入 MOTION（current={arm.get_current_motion_state().name}，"
                            f"home 门槛未过——见上方日志）")
        i0 = len(d.calls)
        time.sleep(0.25)                    # ctrl 线程（200 Hz）采样窗口
        got = d.calls_since(i0, "send_joint_position_arm")
        q_ok = bool(got) and all(np.allclose(c[2], arm.home_arm[c[1]], atol=1e-9) for c in got)
        ok = len(got) >= 6 and q_ok
        ev = (f"MOTION 态 0.25 s 内 ctrl 线程经 JoyArm.send_position_arm 下发 "
              f"{len(got)} 帧（≥6；无豁免则 FSM 互斥拦截为 0 帧）；"
              f"q≈home_arm={q_ok}（hold 规划器无目标回退 q_home）")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        _settle_fsm(arm, d)
        arm.close()


# ---------------- 04 组（求解器域） ----------------
_DOMAINS = {
    "fkine": ("get_fkine_solvers", "set_fkine_solver",
              "get_activated_fkine_solver", "fkine"),
    "ikine": ("get_ikine_solvers", "set_ikine_solver",
              "get_activated_ikine_solver", "ikine"),
    "jac": ("get_jac_solvers", "set_jac_solver",
            "get_activated_jac_solver", "jacobian"),
    "dyn": ("get_dyn_solvers", "set_dyn_solver",
            "get_activated_dyn_solver", "dynamics"),
    "traj": ("get_traj_solvers", "set_traj_solver",
             "get_activated_traj_solver", "trajectory"),
    "ctrl": ("get_ctrl_solvers", "set_ctrl_solver",
             "get_activated_ctrl_solver", "control"),
}


def _it40():
    """六域查询/切换：名单=cfg、默认激活=cfg 首位、切换后激活名与 hz 更新，选择控制器不触碰模式。"""
    arm, d = make_arm()
    try:
        cfg = _tweak_cfg(load_base_cfg())
        checks = []
        for dom, (g_list, s_act, g_act, key) in _DOMAINS.items():
            specs = cfg["robotics"][key]
            expect = sorted(spec["name"] for spec in specs)
            default = specs[0]["name"]        # 默认激活 = cfg 首位
            got = getattr(arm, g_list)()
            checks.append((f"{dom} 名单/默认",
                           got == expect and getattr(arm, g_act)() == default,
                           f"{got}（默认 {getattr(arm, g_act)()}，期望首位 {default}）"))
        # 可切换域（成员 ≥2）：fkine / ikine / traj
        for dom, other in (("fkine", "mdh_fkine_solver"),
                           ("ikine", "analytic_ikine_solver"),
                           ("traj", "single_joint_cubic_traj_planner")):
            _, s_act, g_act, _ = _DOMAINS[dom]
            r = getattr(arm, s_act)(other)
            checks.append((f"{dom} 切换→{other}",
                           r is True and getattr(arm, g_act)() == other,
                           f"ret={r}, 激活={getattr(arm, g_act)()}"))
        # traj 切换后 plan_hz 联动（tweak：hold=10 / cubic=20）
        hz_cubic = arm.plan_hz
        arm.set_traj_solver("single_joint_hold_traj_planner")
        hz_hold = arm.plan_hz
        checks.append(("traj plan_hz 联动", hz_cubic == 20.0 and hz_hold == 10.0,
                       f"cubic={hz_cubic}→hold={hz_hold}（期望 20→10）"))
        # 离线选择控制器不触碰模式；连接后选择也不改当前模式。
        r = arm.set_ctrl_solver(VERIFY_CTRL)
        checks.append(("ctrl 离线选择", r is True and arm.ctrl_hz == 100.0,
                       f"ret={r}, ctrl_hz={arm.ctrl_hz}"))
        arm.connect_arm()
        before_mode = arm.get_mode_arm()
        i0 = len(d.calls)
        r = arm.set_ctrl_solver("joint_position_controller")
        got = d.calls_since(i0, "set_joint_mode_arm")
        checks.append(("ctrl 只选择算法与频率",
                       r is True and arm.ctrl_hz == 200.0 and not got
                       and arm.get_mode_arm() is before_mode,
                       f"切模式内核帧={len(got)}, ctrl_hz={arm.ctrl_hz}"))
        ok = all(c for _, c, _ in checks)
        ev = "；".join(f"{lb}={'OK' if c else 'FAIL'}（{dt}）" for lb, c, dt in checks)
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


def _it41():
    """坏名 warn 保持原激活：六域 set_*('no_such_solver') 均 False + warn。"""
    arm, _ = make_arm()
    try:
        checks = []
        for dom, (g_list, s_act, g_act, _key) in _DOMAINS.items():
            before = getattr(arm, g_act)()
            i0 = _mark()
            r = getattr(arm, s_act)("no_such_solver")
            after = getattr(arm, g_act)()
            ok = r is False and before == after and _hit(_since(i0), "未在该域已加载成员中")
            checks.append((dom, ok, f"ret={r}, {before}→{after}"))
        ok = all(c for _, c, _ in checks)
        ev = "；".join(f"{dm}={'OK' if c else 'FAIL'}（{t}）" for dm, c, t in checks)
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


def _it42():
    """traj 先 plan 后换指针：切换成功即已 plan（sample 立即可用）；plan 失败保持原激活。"""
    arm, d = make_arm()
    try:
        arm.connect_arm()
        d.set_q("arm", arm.home_arm)
        q_t = np.clip(arm.home_arm + 0.05, arm.joint_limits_hard_arm.q_min,
                      arm.joint_limits_hard_arm.q_max)
        arm.set_target_joint(time.time() + 1.0, q_t)
        r = arm.set_traj_solver("single_joint_cubic_traj_planner")
        planner = arm._activated_traj_solver
        sample_err = _exc(planner.sample, time.time() + 0.5)   # 中段采样（起步太近难判移动）
        q_mid = None if sample_err else np.asarray(planner.sample(time.time() + 0.5).q)
        moving = q_mid is not None and (np.linalg.norm(q_mid - arm.home_arm) >=
                                        0.1 * np.linalg.norm(q_t - arm.home_arm))
        ok1 = (r is True and arm.get_activated_traj_solver() == "single_joint_cubic_traj_planner"
               and sample_err is None and arm.plan_hz == 20.0 and moving)
        # plan 失败路径：断连后 cubic.plan 取不到起点 → 试规划异常 → 保持原激活
        arm.disconnect_arm()
        i1 = _mark()
        r2 = arm.set_traj_solver("single_joint_cubic_traj_planner")
        ok2 = (r2 is False
               and arm.get_activated_traj_solver() == "single_joint_cubic_traj_planner"
               and _hit(_since(i1), "试规划"))
        ok = ok1 and ok2
        ev = (f"连接态切换 cubic：ret={r}，切换后 sample 立即可用（先 plan 后换指针，"
              f"否则抛『尚未规划』）={sample_err is None}，q 朝目标移动={moving}，"
              f"plan_hz={arm.plan_hz}；断连态再切同名：ret={r2}，激活保持 "
              f"{arm.get_activated_traj_solver()}，warn 含『试规划』={_hit(_since(i1), '试规划')}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


# ---------------- 05 组（数学门面 vs pinocchio 原生对照） ----------------
def _it50():
    """9 方法对照：JoyArm 数学门面（显式传参）vs pinocchio 原生 API 同输入逐项对照。"""
    arm, _ = make_arm()
    try:
        model = arm.pin_model
        fid = model.getFrameId(arm.get_config()["joyarm"]["frame_end"])
        rng = np.random.default_rng(20261016)
        q = 0.5 * (arm.neutral_arm + rng.uniform(
            np.asarray(arm.joint_limits_soft_arm.q_min),
            np.asarray(arm.joint_limits_soft_arm.q_max)))
        dq = rng.uniform(-0.5, 0.5, q.shape)
        ddq = rng.uniform(-1.0, 1.0, q.shape)
        checks = []

        # 1) fkine vs updateFramePlacement
        pose = arm.solve_fkine(q)
        data = pin.Data(model)
        pin.forwardKinematics(model, data, q)
        pose_ref = Pose.from_T(pin.updateFramePlacement(model, data, fid).homogeneous)
        d_pos = float(np.linalg.norm(pose.position - pose_ref.position))
        d_rot = min(float(np.linalg.norm(pose.orientation - pose_ref.orientation)),
                    float(np.linalg.norm(pose.orientation + pose_ref.orientation)))
        checks.append(("solve_fkine vs updateFramePlacement",
                       d_pos < 1e-9 and d_rot < 1e-9,
                       f"|Δp|={d_pos:.2e} m, |Δquat|={d_rot:.2e}"))

        # 2) ikine（数值 LM 往返：fkine∘ikine ≈ 目标位姿）
        res = arm.solve_ikine(pose, q0=arm.neutral_arm)
        if not res.success:
            res = arm.solve_ikine(pose, q0=q)      # 多起点本是数值法用法，二次尝试
        pose_back = arm.solve_fkine(res.q)
        e_pos = float(np.linalg.norm(pose_back.position - pose.position))
        e_rot = quat_err(pose_back.orientation, pose.orientation)
        checks.append(("solve_ikine 往返（LM 数值）",
                       res.success and e_pos < 1e-3 and e_rot < 1e-2,
                       f"success={res.success}, |Δp|={e_pos:.2e}, Δθ={e_rot:.2e}"))

        # 3) ikine_all：数值激活→RuntimeError；解析激活→全解往返
        e_num = _exc(arm.solve_ikine_all, pose)
        arm.set_ikine_solver("analytic_ikine_solver")
        r_all = arm.solve_ikine_all(pose)
        errs = []
        if r_all.q is not None and np.asarray(r_all.q).size:
            for q_k in np.atleast_2d(r_all.q):
                pk = arm.solve_fkine(q_k)
                errs.append(max(float(np.linalg.norm(pk.position - pose.position)),
                                quat_err(pk.orientation, pose.orientation)))
        arm.set_ikine_solver("pin_ikine_solver")
        checks.append(("solve_ikine_all（数值拒 / 解析全解）",
                       isinstance(e_num, RuntimeError) and r_all.success
                       and len(errs) >= 1 and max(errs) < 5e-3,
                       f"数值激活抛 RuntimeError={isinstance(e_num, RuntimeError)}；解析 "
                       f"K={0 if r_all.q is None else np.atleast_2d(r_all.q).shape[0]} 解，"
                       f"max 往返误差={max(errs) if errs else float('nan'):.2e}"))

        # 4) jac vs computeFrameJacobian
        J = np.asarray(arm.get_jac(q))
        J_ref = pin.computeFrameJacobian(model, pin.Data(model), q, fid,
                                         pin.LOCAL_WORLD_ALIGNED)
        checks.append(("get_jac vs computeFrameJacobian",
                       J.shape == (6, arm.n_arm)
                       and np.allclose(J, J_ref, atol=1e-9),
                       f"shape={J.shape}, max|Δ|={np.max(np.abs(J - J_ref)):.2e}"))

        # 5) jac_pinv_damped vs 公式 Jᵀ(JJᵀ+λ²I)⁻¹
        lam = 1e-3
        Jp = np.asarray(arm.get_jac_pinv_damped(q, damping=lam))
        Jp_ref = J_ref.T @ np.linalg.solve(J_ref @ J_ref.T + lam ** 2 * np.eye(6), np.eye(6))
        checks.append(("get_jac_pinv_damped vs 公式",
                       Jp.shape == (arm.n_arm, 6) and np.allclose(Jp, Jp_ref, atol=1e-9),
                       f"max|Δ|={np.max(np.abs(Jp - Jp_ref)):.2e}"))

        # 6) idyn vs rnea
        tau = np.asarray(arm.solve_idyn(q, dq, ddq))
        tau_ref = pin.rnea(model, pin.Data(model), q, dq, ddq)
        checks.append(("solve_idyn vs rnea",
                       tau.shape == (arm.n_arm,) and np.allclose(tau, tau_ref, atol=1e-9),
                       f"max|Δ|={np.max(np.abs(tau - tau_ref)):.2e}"))

        # 7) mass vs crba（原生只填上三角，同对称化后对照）
        M = np.asarray(arm.get_mass_matrix(q))
        M_raw = np.array(pin.crba(model, pin.Data(model), q), dtype=float)
        M_ref = np.triu(M_raw) + np.triu(M_raw, 1).T
        checks.append(("get_mass_matrix vs crba(对称化)",
                       M.shape == (arm.n_arm, arm.n_arm) and np.allclose(M, M_ref, atol=1e-9),
                       f"max|Δ|={np.max(np.abs(M - M_ref)):.2e}"))

        # 8) coriolis（独立 API 优先，缺则 nle−g 公式）
        C = np.asarray(arm.get_coriolis(q, dq))
        try:
            C_ref = pin.computeCoriolis(model, pin.Data(model), q, dq)
            c_src = "computeCoriolis"
        except AttributeError:
            dd = pin.Data(model)
            C_ref = (pin.nonLinearEffects(model, dd, q, dq)
                     - pin.rnea(model, dd, q, dq * 0.0, dq * 0.0))
            c_src = "nle−g"
        checks.append(("get_coriolis vs " + c_src,
                       C.shape == (arm.n_arm,) and np.allclose(C, C_ref, atol=1e-8),
                       f"max|Δ|={np.max(np.abs(C - C_ref)):.2e}"))

        # 9) gravity（独立 API 优先，缺则 rnea(q,0,0)）
        G = np.asarray(arm.get_gravity(q))
        try:
            G_ref = pin.computeGeneralizedGravity(model, pin.Data(model), q)
            g_src = "computeGeneralizedGravity"
        except AttributeError:
            G_ref = pin.rnea(model, pin.Data(model), q, q * 0.0, q * 0.0)
            g_src = "rnea(q,0,0)"
        checks.append(("get_gravity vs " + g_src,
                       G.shape == (arm.n_arm,) and np.allclose(G, G_ref, atol=1e-9),
                       f"max|Δ|={np.max(np.abs(G - G_ref)):.2e}"))

        # 10) 缺省（不传参）契约：无硬件状态时抛 RuntimeError（显式传参离线可用的对照面）
        e_def = _exc(arm.solve_fkine)
        checks.append(("缺省取硬件状态契约（未连接→RuntimeError）",
                       isinstance(e_def, RuntimeError) and "无可用 q" in str(e_def),
                       f"{type(e_def).__name__}: {_trunc(e_def, 60)}"))

        ok = all(c for _, c, _ in checks)
        ev = "；".join(f"{lb}={'OK' if c else 'FAIL'}（{dt}）" for lb, c, dt in checks)
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


def _it51():
    """到位判断三函数：is_in_q_arm / is_in_q_end / is_in_pose_arm（Dummy 快照注入 + 拒绝路径 + 无快照路径）。"""
    arm, d = make_arm()
    try:
        n = arm.n_arm
        checks = []
        # 未连接（无状态快照）：初始 NaN 快照 → 判未到位 + warn（不抛）
        i0 = _mark()
        no_snap = (arm.is_in_q_arm(arm.home_arm) is False
                   and arm.is_in_pose_arm(Pose(position=np.zeros(3),
                                               orientation=np.array([1.0, 0.0, 0.0, 0.0]))) is False
                   and _hit(_since(i0), "无可用关节状态快照"))
        checks.append(("未连接无快照 → False + warn（三函数不抛）", no_snap))
        arm.connect_arm()
        arm.connect_end()
        d.set_q("arm", arm.home_arm)
        d.set_q("end", arm.home_end)
        time.sleep(0.12)                    # 等 backend 刷新线程装配快照（50 Hz）
        tol = float(np.asarray(
            arm.get_config()["joyarm"]["arrival_tol"]["arm"]["q"]).reshape(-1)[0])
        checks.append(("is_in_q_arm：到位 True / 偏离超容差 False",
                       arm.is_in_q_arm(arm.home_arm) is True
                       and arm.is_in_q_arm(arm.home_arm + (tol + 0.05)) is False))
        checks.append(("is_in_q_arm 容差边界（偏离 tol−0.01 内 True）",
                       arm.is_in_q_arm(arm.home_arm + (tol - 0.01)) is True))
        checks.append(("is_in_q_end：到位 True / 偏离 0.5 rad False",
                       arm.is_in_q_end(arm.home_end) is True
                       and arm.is_in_q_end(arm.home_end + 0.5) is False))
        pose = arm.solve_fkine(arm.home_arm)
        moved = Pose(position=pose.position + np.array([0.01, 0.0, 0.0]),
                     orientation=pose.orientation)
        checks.append(("is_in_pose_arm：到位 True / 平移 1 cm False（超 pos 容差 0.001）",
                       arm.is_in_pose_arm(pose) is True
                       and arm.is_in_pose_arm(moved) is False))
        e1 = _exc(arm.is_in_q_arm, np.zeros(n + 1))
        e2 = _exc(arm.is_in_q_arm, arm.home_arm, [0.1] * (n + 1))
        e3 = _exc(arm.is_in_pose_arm, [1.0] * 7)
        checks.append(("拒绝路径：目标维度/容差长度 ValueError、pose 类型 TypeError",
                       isinstance(e1, ValueError) and "维度" in str(e1)
                       and isinstance(e2, ValueError) and "tol_q 长度" in str(e2)
                       and isinstance(e3, TypeError) and "Pose" in str(e3)))
        ok = all(c for _, c in checks)
        ev = "；".join(f"{lb}={'OK' if c else 'FAIL'}" for lb, c in checks)
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


def _it52():
    """rand_q_arm / rand_q_end 随机采样：批量样本落在限位区间内且全有限、非退化。"""
    arm, _ = make_arm()
    try:
        checks = []
        lo = np.asarray(arm.joint_limits_soft_arm.q_min, dtype=float)
        hi = np.asarray(arm.joint_limits_soft_arm.q_max, dtype=float)
        s = np.array([arm.rand_q_arm() for _ in range(60)])
        checks.append(("rand_q_arm 缺省软限位：60 样本 ∈ [q_min,q_max] 且全有限",
                       s.shape == (60, arm.n_arm) and bool(np.all(np.isfinite(s)))
                       and bool(np.all(s >= lo - 1e-12)) and bool(np.all(s <= hi + 1e-12))))
        lo2 = np.asarray(arm.joint_limits_hard_arm.q_min, dtype=float)
        hi2 = np.asarray(arm.joint_limits_hard_arm.q_max, dtype=float)
        s2 = np.array([arm.rand_q_arm(arm.joint_limits_hard_arm) for _ in range(30)])
        checks.append(("rand_q_arm 显式硬限位入参：30 样本 ∈ 硬限位区间",
                       s2.shape == (30, arm.n_arm)
                       and bool(np.all(s2 >= lo2 - 1e-12)) and bool(np.all(s2 <= hi2 + 1e-12))))
        varied = bool(len(np.unique(s[:, 0])) > 1 and len(np.unique(s[:, -1])) > 1)
        checks.append(("样本非退化（同列取值多样，非常数列 / 单点）", varied))
        se = np.array([arm.rand_q_end() for _ in range(30)])
        loe = np.asarray(arm.joint_limits_soft_end.q_min, dtype=float).reshape(-1)
        hie = np.asarray(arm.joint_limits_soft_end.q_max, dtype=float).reshape(-1)
        ok_end = (se.ndim == 2 and se.shape[1] == arm.n_end and bool(np.all(np.isfinite(se)))
                  and bool(np.all(se >= loe - 1e-12)) and bool(np.all(se <= hie + 1e-12)))
        checks.append((f"rand_q_end：30 样本 ∈ end 软限位区间（n_end={arm.n_end}）", ok_end))
        ok = all(c for _, c in checks)
        ev = "；".join(f"{lb}={'OK' if c else 'FAIL'}" for lb, c in checks)
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


def _it53():
    """frame=0 基座（universe）帧回归：显式传 0 按基座帧求解，不再被吞成末端帧。"""
    arm, _ = make_arm()
    try:
        model = arm.pin_model
        rng = np.random.default_rng(20261020)
        q = 0.5 * (arm.neutral_arm + rng.uniform(
            np.asarray(arm.joint_limits_soft_arm.q_min),
            np.asarray(arm.joint_limits_soft_arm.q_max)))
        p0 = arm.solve_fkine(q, frame=0)            # 显式 0 = 基座 universe 帧
        data = pin.Data(model)
        pin.forwardKinematics(model, data, q)
        ref_pose = Pose.from_T(pin.updateFramePlacement(model, data, 0).homogeneous)
        d_pos = float(np.linalg.norm(p0.position - ref_pose.position))
        d_rot = min(float(np.linalg.norm(p0.orientation - ref_pose.orientation)),
                    float(np.linalg.norm(p0.orientation + ref_pose.orientation)))
        p_end = arm.solve_fkine(q)                  # 缺省 = 末端帧（对照组）
        d_end = float(np.linalg.norm(p0.position - p_end.position))
        differs = d_end > 1e-3
        J0 = np.asarray(arm.get_jac(q, frame=0))
        J_ref = pin.computeFrameJacobian(model, pin.Data(model), q, 0,
                                         pin.LOCAL_WORLD_ALIGNED)
        jac_ok = J0.shape == (6, arm.n_arm) and bool(np.allclose(J0, J_ref, atol=1e-9))
        ok = d_pos < 1e-9 and d_rot < 1e-9 and differs and jac_ok
        ev = (f"solve_fkine(q, frame=0) vs updateFramePlacement(帧 0)：|Δp|={d_pos:.2e} m、"
              f"|Δquat|={d_rot:.2e}；与缺省末端帧结果可区分（|Δp|=|p₀−p_end|={d_end:.3f} m"
              f"＞1e-3，证明 0 未被当成假值吞掉换末端帧）；get_jac(q, frame=0) vs "
              f"computeFrameJacobian(帧 0) 一致={jac_ok}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


# ---------------- 06 组（MotionFSM：DummyBackend 推演） ----------------
def _it60():
    """start 前置：未连接 start_motion_fsm 抛 RuntimeError。"""
    arm, _ = make_arm()
    try:
        e = _exc(arm.start_motion_fsm)
        ok = isinstance(e, RuntimeError) and "arm 未连接" in str(e)
        ev = (f"未连接 start_motion_fsm → {type(e).__name__}，"
              f"消息含『arm 未连接』={('arm 未连接' in str(e))}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


def _it61():
    """自动迁移序列：DISABLE→HOME（发 home 指令驱动）→（到位）→HOME→MOTION。"""
    arm, d = make_arm()
    try:
        d.follow = False
        d.set_q("arm", arm.neutral_arm)      # 偏离 home：先停 HOME 驱动到位
        arm.connect_arm()
        i0 = _mark()
        arm.start_motion_fsm()
        if not _wait_state(arm, MotionState.HOME):
            return "FAIL", f"未迁至 HOME（current={arm.get_current_motion_state().name}）"
        i1 = len(d.calls)
        d.follow = True                        # 理想电机：home 指令一步到位
        if not _wait_state(arm, MotionState.MOTION):
            return "FAIL", f"home 到位后未迁 MOTION（current={arm.get_current_motion_state().name}）"
        msgs = _since(i0)
        n_pos = len(d.calls_since(i1, "send_joint_position_arm"))
        ok = (_hit(msgs, "状态迁移 DISABLE → HOME")
              and _hit(msgs, "状态迁移 HOME → MOTION")
              and _hit(msgs, "HOME 已到位")
              and n_pos > 0
              and arm.get_current_motion_state() is MotionState.MOTION)
        ev = (f"日志序列：DISABLE→HOME={_hit(msgs, '状态迁移 DISABLE → HOME')}，"
              f"HOME 已到位={_hit(msgs, 'HOME 已到位')}，"
              f"HOME→MOTION={_hit(msgs, '状态迁移 HOME → MOTION')}；"
              f"HOME 驱动期 send_position 帧 {n_pos} 个（>0）；"
              f"终态 current={arm.get_current_motion_state().name}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        _settle_fsm(arm, d)
        arm.close()


def _it62():
    """home 到位门槛：q 未到位停 HOME 不进 MOTION；到位后放行迁移。"""
    arm, d = make_arm()
    try:
        d.follow = False
        d.set_q("arm", arm.neutral_arm)
        arm.connect_arm()
        arm.start_motion_fsm()
        if not _wait_state(arm, MotionState.HOME):
            return "FAIL", f"未迁至 HOME（current={arm.get_current_motion_state().name}）"
        time.sleep(0.4)                        # q 冻结在 neutral：门槛应持续拦住
        stuck = arm.get_current_motion_state() is MotionState.HOME
        d.set_q("arm", arm.home_arm)           # 到位（follow 保持 False，直接注入）
        moved = _wait_state(arm, MotionState.MOTION)
        ok = stuck and moved
        ev = (f"q=neutral 期间 0.4 s 停留 HOME={stuck}（门槛生效，MOTION 进入被拦）；"
              f"q 置 home 后迁 MOTION={moved}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        _settle_fsm(arm, d)
        arm.close()


def _it63():
    """保护事件直达：MOTION 下 HOLDING/DAMPING/DISABLE 依次直达 + 维持指令核对。"""
    arm, d = make_arm()
    try:
        d.follow = True
        d.set_q("arm", arm.home_arm)
        arm.connect_arm()
        arm.start_motion_fsm()
        if not _wait_state(arm, MotionState.MOTION):
            return "FAIL", f"未进入 MOTION（current={arm.get_current_motion_state().name}）"
        checks = []
        i0 = len(d.calls)
        arm.set_event(SafetyEvent.HOLDING)
        ok_h = _wait_state(arm, MotionState.HOLD)
        time.sleep(0.15)                       # 等数拍维持指令（50 Hz）落地再计数
        got = d.calls_since(i0, "send_joint_position_arm")
        checks.append(("HOLDING→HOLD 直达 + 原位锁定重发",
                       ok_h and bool(got)
                       and all(np.allclose(c[2], arm.home_arm[c[1]], atol=1e-6) for c in got),
                       f"{len(got)} 帧 hold_q≈home"))
        i0 = len(d.calls)
        arm.set_event(SafetyEvent.DAMPING)
        ok_d = _wait_state(arm, MotionState.DAMP)
        time.sleep(0.15)
        got = d.calls_since(i0, "send_joint_mit_arm")
        kd_cfg = np.asarray(
            arm.get_config()["joyarm"]["MotionFSM"]["damping_kd"], dtype=float).reshape(-1)
        kg_cfg = np.asarray(
            arm.get_config()["joyarm"]["MotionFSM"]["damping_k_gravity"], dtype=float).reshape(-1)
        tau_exp = kg_cfg * arm.get_gravity()             # 被动快照当前 q 下的重力补偿期望
        checks.append(("DAMPING→直达 + MIT 阻尼（kp=0, kd=cfg, tau=k_gravity·G(q)）",
                       ok_d and bool(got)
                       and all(abs(c[5]) < 1e-12 and abs(c[6] - kd_cfg[c[1]]) < 1e-9
                               and abs(c[2] - tau_exp[c[1]]) < 1e-6 for c in got),
                       f"{len(got)} 帧 kp=0, kd=cfg_i, tau=k_i·G_i"))
        i0 = len(d.calls)
        arm.set_event(SafetyEvent.DISABLE)
        ok_x = _wait_state(arm, MotionState.DISABLE)
        time.sleep(0.15)
        got = d.calls_since(i0, "disable_joint_arm")
        checks.append(("DISABLE→直达 + 失能维持", ok_x and bool(got), f"{len(got)} 帧"))
        ok = all(c for _, c, _ in checks)
        ev = "；".join(f"{lb}={'OK' if c else 'FAIL'}（{t}）" for lb, c, t in checks)
        return ("PASS" if ok else "FAIL"), ev
    finally:
        _settle_fsm(arm, d)
        arm.close()


def _it64():
    """stop 超时强推：回零永不到位 → 等满 move_wait_timeout 后 warn 强推收尾。"""
    arm, d = make_arm()
    try:
        d.follow = False
        d.set_q("arm", arm.home_arm)
        arm.connect_arm()
        arm.start_motion_fsm()
        if not _wait_state(arm, MotionState.MOTION):
            return "FAIL", f"未进入 MOTION（current={arm.get_current_motion_state().name}）"
        i0 = _mark()
        t0 = time.monotonic()
        arm.stop_motion_fsm()                 # follow=False：ZERO 永不到位 → 超时强推
        dt = time.monotonic() - t0
        msgs = _since(i0)
        warned = _hit(msgs, "超时强制推进")
        stopped = arm.get_current_motion_state() is MotionState.DISABLE
        ok = warned and stopped and dt >= 0.9
        ev = (f"stop 耗时 {dt:.2f}s（≥0.9s：等满注入 move_wait_timeout=1.0s）；"
              f"warn 超时强推={warned}；收尾后 current="
              f"{arm.get_current_motion_state().name}；"
              f"{_trunc(next((m for m in msgs if '超时' in m), ''), 110)}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        _settle_fsm(arm, d)
        arm.close()


def _it65():
    """set_target_joint / set_target_cart 拒绝路径：缺必填/维度不符/坏 t/非 TrajFrame 序列均 ValueError。"""
    arm, _ = make_arm()
    try:
        n = arm.n_arm
        t = time.time() + 1.0
        checks = []
        e = _exc(arm.set_target_joint, t, None)
        checks.append(("set_target_joint q 缺失", isinstance(e, ValueError) and "q 必填" in str(e)))
        e = _exc(arm.set_target_joint, t, np.zeros(n + 1))
        checks.append(("set_target_joint q 维度 n+1", isinstance(e, ValueError) and "维度" in str(e)))
        e = _exc(arm.set_target_joint, "bad", np.zeros(n))
        checks.append(("set_target_joint t 非标量", isinstance(e, ValueError) and "Unix 秒标量" in str(e)))
        e = _exc(arm.set_target_cart, t, None)
        checks.append(("set_target_cart pose 缺失", isinstance(e, ValueError) and "pose 必填" in str(e)))
        e = _exc(arm.set_target_cart, t, np.zeros(6))
        checks.append(("set_target_cart pose 维度 6≠7", isinstance(e, ValueError) and "维度" in str(e)))
        e = _exc(arm.set_target_cart, t, np.zeros(7), np.zeros(5))
        checks.append(("set_target_cart twist 维度 5≠6", isinstance(e, ValueError) and "twist" in str(e)))
        e = _exc(arm.set_target_trajframe, [object()])
        checks.append(("set_target_trajframe 非 TrajFrame 序列",
                       isinstance(e, ValueError) and "非 TrajFrame" in str(e)))
        arm.set_target_joint(t, np.zeros(n))   # 合法目标可写入（拒绝路径不破坏通道）
        ok_write = arm.get_target_trajframe() is not None
        ok = all(c for _, c in checks) and ok_write
        ev = ("；".join(f"{lb}={'OK' if c else 'FAIL'}" for lb, c in checks)
              + f"；拒绝后合法目标写入通道不受影响={ok_write}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.set_target_trajframe(None)
        arm.close()


def _it66():
    """目标帧通道 deepcopy 隔离：get 返回值与 set 入参的外部改写均不反哺内部。"""
    arm, _ = make_arm()
    try:
        t = time.time() + 2.0
        q_expect = np.clip(arm.home_arm + 0.05, arm.joint_limits_hard_arm.q_min,
                           arm.joint_limits_hard_arm.q_max).astype(float)
        q_t = q_expect.copy()
        arm.set_target_joint(t, q_t)
        q_t[0] = 999.0                        # set 后改写入参 → 内部深拷贝不受影响
        fr1 = arm.get_target_trajframe()
        fr1.q[0] = -999.0                     # 改 get 返回值 → 不反哺内部
        fr2 = arm.get_target_trajframe()
        ok_q = (fr1 is not fr2
                and bool(np.allclose(fr2.q, q_expect))
                and not np.allclose(fr2.q[0:1], [-999.0])
                and abs(float(fr2.time) - t) < 1e-9)
        pose = arm.solve_fkine(arm.home_arm)
        arm.set_target_cart(t, pose)
        pose.position[0] = 999.0              # 笛卡尔目标对 Pose 入参深拷贝
        frc = arm.get_target_trajframe()
        pos0 = (float(frc.pose.position[0])
                if frc is not None and frc.pose is not None else None)
        ok_c = pos0 is not None and frc.pose is not pose and pos0 < 1.0
        arm.set_target_trajframe(None)        # 清空通道
        ok_none = arm.get_target_trajframe() is None
        ok = ok_q and ok_c and ok_none
        ev = (f"joint 目标：set 后改入参 / 改 get 返回值均不反哺（二次 get 的 q 仍="
              f"{np.round(fr2.q, 3).tolist()[:3]}…，time 保持原值）={ok_q}；cart 目标："
              f"set 后改入参 Pose 不反哺（内部 pose.position[0]={pos0}）={ok_c}；"
              f"None 清空后 get=None={ok_none}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.set_target_trajframe(None)
        arm.close()


def _it67():
    """wait_for_motion_state 四路径：到达返 1；超时/未运行/非枚举返 0；timeout 非法 ValueError。"""
    arm, d = make_arm()
    try:
        # 参数与未运行路径（FSM 未启动，互不干扰）
        i0 = _mark()
        r_enum = arm.wait_for_motion_state("MOTION", timeout=0.2)
        warn_enum = _hit(_since(i0), "MotionState 枚举")
        e = _exc(arm.wait_for_motion_state, MotionState.MOTION, 0)
        bad_timeout = isinstance(e, ValueError) and "正有限" in str(e)
        i0 = _mark()
        r_idle = arm.wait_for_motion_state(MotionState.MOTION, timeout=0.2)
        warn_idle = _hit(_since(i0), "FSM 未运行")
        # 成功路径：驱动 FSM 至 MOTION，等待即返 1
        d.follow = True
        d.set_q("arm", arm.home_arm)
        arm.connect_arm()
        arm.start_motion_fsm()
        r_ok = arm.wait_for_motion_state(MotionState.MOTION, timeout=5.0)
        # 超时路径：等待无事件触发的不可达态 DAMP
        i0 = _mark()
        t0 = time.monotonic()
        r_to = arm.wait_for_motion_state(MotionState.DAMP, timeout=0.3)
        dt = time.monotonic() - t0
        warn_to = _hit(_since(i0), "超时")
        ok = (r_enum == 0 and warn_enum and bad_timeout
              and r_idle == 0 and warn_idle and r_ok == 1
              and r_to == 0 and warn_to and 0.25 <= dt <= 2.0)
        ev = (f"非枚举→0+warn={r_enum == 0 and warn_enum}；timeout=0→ValueError"
              f"={bad_timeout}；未运行→0+warn={r_idle == 0 and warn_idle}；"
              f"运行中等到 MOTION→1={r_ok == 1}；不可达态 0.3s 超时→0+warn"
              f"={r_to == 0 and warn_to}（实测耗时 {dt:.2f}s）")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        _settle_fsm(arm, d)
        arm.close()


# ---------------- 07 组（safe_monitor） ----------------
def _it70():
    """硬规则：error≥2 → DAMPING（事件与 FSM 迁移双通路）。"""
    arm, d = make_arm()
    try:
        d.follow = False
        d.set_q("arm", arm.home_arm)
        arm.connect_arm()
        arm.start_motion_fsm()
        home = _wait_state(arm, MotionState.HOME)
        arm.start_monitor()
        time.sleep(0.1)
        i0 = len(d.calls)
        d.jstates["arm"][0]["error"] = 2      # 单关节故障码
        ok_ev = _wait_event(arm, SafetyEvent.DAMPING)
        ok_st = _wait_state(arm, MotionState.DAMP)
        time.sleep(0.15)                       # 等数拍维持指令（50 Hz）落地再计数
        got = d.calls_since(i0, "send_joint_mit_arm")
        d.jstates["arm"][0]["error"] = 1
        ok = home and ok_ev and ok_st and bool(got)
        ev = (f"监视器前置 HOME 就绪={home}；error[0]=2 后：get_event()={arm.get_event().name}，"
              f"current={arm.get_current_motion_state().name}，"
              f"DAMP 维持 MIT 帧 {len(got)} 个（>0）")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        d.jstates["arm"][0]["error"] = 1
        _settle_fsm(arm, d)
        arm.close()


def _it71():
    """断连静默挂起/重连恢复，以及持续无新鲜状态触发 DISABLE 的两条独立路径。"""
    arm, backend = make_arm()
    try:
        backend.follow = True
        backend.set_q('arm', arm.home_arm)
        arm.connect_arm()
        arm.start_motion_fsm()
        _require(_wait_state(arm, MotionState.MOTION), '断连测试前未进入 MOTION')
        arm.start_monitor()
        mark = _mark()
        backend.disconnect_arm()
        _require(_wait_event(arm, SafetyEvent.DISABLE), '断连监视规则未设置 DISABLE')
        arm.stop_monitor()
        deadline = time.monotonic() + 1.0
        while not arm._fsm_disc_warned and time.monotonic() < deadline:
            time.sleep(0.02)
        _require(arm._fsm_disc_warned, '断连未进入挂起分支')
        current = arm.get_current_motion_state()
        call_mark = len(backend.calls)
        time.sleep(0.15)
        messages = _since(mark)
        warns = sum('FSM 挂起' in message for message in messages)
        _require(warns == 1, f'一次断连应只告警一次，实际 {warns} 次')
        _require(arm.get_current_motion_state() is current,
                 '断连挂起未保持当前状态')
        _require(not _hit(messages, 'FSM 单步失败'), '断连仍逐拍产生单步失败')
        _require(all(not backend.calls_since(call_mark, name) for name in
                     ('send_joint_position_arm', 'send_joint_mit_arm', 'send_joint_vel_arm',
                      'enable_joint_arm', 'set_joint_mode_arm')), '断连挂起期间仍有控制/自愈下发')
        _require(arm.connect_arm() is True, 'Dummy 重连失败')
        _require(_wait_state(arm, MotionState.DISABLE), '重连后未继续处理挂起期间的 DISABLE 事件')
        _require(not arm._fsm_disc_warned and arm.get_event() is SafetyEvent.DISABLE,
                 '重连未复位告警标志，或自动清除了保护事件')
        arm.set_event(SafetyEvent.NORMAL)
        _require(_wait_state(arm, MotionState.MOTION), '显式恢复 NORMAL 后未恢复运动管线')
        arm.set_event(SafetyEvent.DISABLE)
        arm.close()
        leaked = [name for name in _joyarm_thread_names() if name != 'backend-state-refresh']
        _require(not leaked, f'断连/重连验收收尾残留线程：{leaked}')
    finally:
        arm.close()
    # 保持连接但反馈持续失败：监视器应过滤陈旧快照，设置 DISABLE。
    arm, backend = make_arm()
    try:
        backend.follow = True
        backend.set_q('arm', arm.home_arm)
        arm.connect_arm()
        arm.start_motion_fsm()
        _require(_wait_state(arm, MotionState.MOTION), '反馈失败测试前未进入 MOTION')
        seen_none = []

        def probe(state_arm, state_end, motion, event):
            seen_none.append(state_arm is None)
            return None

        arm.set_monitor_rules(probe)
        arm.start_monitor()
        mark = _mark()
        backend.fail_at['read_joint_state_arm'] = None
        _require(_wait_event(arm, SafetyEvent.DISABLE), '持续无新鲜状态未触发 DISABLE')
        _require(_wait_state(arm, MotionState.DISABLE), '反馈失败未进入失能状态')
        # ctrl 缺反馈只跳拍；监视器过滤旧快照并独立发布 DISABLE。
        deadline = time.monotonic() + 1.0
        while not any(seen_none) and time.monotonic() < deadline:
            time.sleep(0.02)
        _require(any(seen_none) and not arm._ctrl_ready(),
                 '自定义规则未收到 None，或 ctrl 未撤销运动许可')
    finally:
        backend.fail_at.pop('read_joint_state_arm', None)
        _settle_fsm(arm, backend)
        arm.close()
    return 'PASS', ('断连仅1次挂起告警，状态保持且无下发；重连先处理DISABLE，显式NORMAL恢复；'
                    '反馈失败→DISABLE，自定义规则收到None，ctrl撤销运动许可；收尾无线程残留')


def _it72():
    """自定义规则绑定 + 仲裁：全部命中合并取危险最高、异常/非法返回忽略。"""
    arm, d = make_arm()
    try:
        d.follow = False
        d.set_q("arm", arm.neutral_arm)     # 非 home：门槛拦住 MOTION，驻留 HOME
        arm.connect_arm()
        arm.start_motion_fsm()
        home = _wait_state(arm, MotionState.HOME)
        arm.start_monitor()
        time.sleep(0.1)
        checks = []

        # 1) 同组多命中取危险最高：HOLDING(1) + DAMPING(4) → DAMPING
        arm.set_event(SafetyEvent.NORMAL)
        arm.set_monitor_rules([lambda sa, se, mo, ev: SafetyEvent.HOLDING,
                               lambda sa, se, mo, ev: SafetyEvent.DAMPING])
        ok1 = _wait_event(arm, SafetyEvent.DAMPING)
        checks.append(("同组多命中取危险最高（HOLDING+DAMPING→DAMPING）", ok1))

        # 2) 合并取最危险：硬 DAMPING(4) + 自定义 DISABLE(5) 同时命中 → DISABLE
        arm.set_monitor_rules([lambda sa, se, mo, ev: SafetyEvent.DISABLE])
        arm.set_event(SafetyEvent.NORMAL)
        d.jstates["arm"][0]["error"] = 2
        ok2 = _wait_event(arm, SafetyEvent.DISABLE)
        stay_dis = True
        t_end = time.monotonic() + 0.2
        while time.monotonic() < t_end:      # 窗口内两规则持续命中，事件须恒为最危险的 DISABLE
            if arm.get_event() is not SafetyEvent.DISABLE:
                stay_dis = False
            time.sleep(0.02)
        d.jstates["arm"][0]["error"] = 1
        arm.set_monitor_rules([])
        checks.append(("合并取最危险（硬 DAMPING+自定义 DISABLE→DISABLE）", ok2 and stay_dis))

        # 3) 异常规则与非 SafetyEvent 返回忽略（warn），不影响其他规则
        def r_bad(sa, se, mo, ev):
            raise RuntimeError("rule boom")

        def r_weird(sa, se, mo, ev):
            return 42

        arm.set_event(SafetyEvent.NORMAL)
        i0 = _mark()
        arm.set_monitor_rules([r_bad, r_weird,
                               lambda sa, se, mo, ev: SafetyEvent.HOLDING])
        ok3 = (_wait_event(arm, SafetyEvent.HOLDING)
               and _hit(_since(i0), "自定义规则", "抛异常")
               and _hit(_since(i0), "非 SafetyEvent/None"))
        checks.append(("异常/非法返回忽略 + 正常规则仍生效", ok3))
        arm.stop_monitor()
        ok = home and all(c for _, c in checks)
        ev = (f"前置 HOME 就绪={home}；"
              + "；".join(f"{lb}={'OK' if c else 'FAIL'}" for lb, c in checks))
        return ("PASS" if ok else "FAIL"), ev
    finally:
        d.jstates["arm"][0]["error"] = 1
        arm.stop_monitor()
        _settle_fsm(arm, d)
        arm.close()


# ---------------- 08 组（可视化） ----------------
def _viz_urls() -> list[str]:
    """从已捕获日志解析监控页 URL（只认『监控页面已启动』日志行；去重保序）。

    只捕获监控页日志行：meshcat 初始化成功时其 URL（『3D 视图已就绪』行）会先
    入日志，若按首个匹配 URL 取页面/端点，断言会打到 meshcat 端口（无监控页
    内容 / 404）而误判 FAIL——监控页正常也过不了。
    """
    import re as _re
    urls: list[str] = []
    for m in _cap.msgs:
        if "监控页面已启动" not in m:
            continue
        for u in _re.findall(r"http://127\.0\.0\.1:\d+/", m):
            if u not in urls:
                urls.append(u)
    return urls


def _fetch(url: str, timeout: float = 2.0) -> tuple[bool, bytes]:
    """只读 GET（失败返回 (False, b"")）。"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return True, r.read()
    except Exception:
        return False, b""


def _it80():
    """start/stop_state_viz 离线不炸：页面 + /state.json 轮询端点可用，stop 后关闭。"""
    arm, _ = make_arm()
    try:
        i0 = _mark()
        e = _exc(arm.start_state_viz)
        degraded = _hit(_since(i0), "降级为仅 HTML 状态面板")
        time.sleep(0.35)                       # worker 10 Hz：至少 2 拍填充面板数据
        urls = _viz_urls()
        ok_page, page = _fetch(urls[0]) if urls else (False, b"")
        ok_json, payload = _fetch(urls[0] + "state.json") if urls else (False, b"")
        has_keys = ok_json and all(k in payload for k in (b"model", b"fsm", b"event"))
        e2 = _exc(arm.stop_state_viz)
        gone = not _fetch(urls[0] + "state.json", timeout=0.8)[0] if urls else True
        ok = (e is None and e2 is None and bool(urls) and ok_page
              and b"JoyArm" in page and ok_json and has_keys and gone)
        ev = (f"start 异常={e is None}；meshcat 降级 warn={degraded}"
              f"（False=3D 视图真实拉起、True=初始化失败走降级）；"
              f"监控页 URL={urls[:1]}；GET /（页面含标题）={ok_page and b'JoyArm' in page}；"
              f"GET /state.json（含 model/fsm/event 键）={has_keys}；"
              f"stop 后端口关闭={gone}；stop 异常={e2 is None}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.stop_state_viz()
        arm.close()


def _it81():
    """meshcat 初始化失败降级：注入故障必现降级 warn，页面 HAS_MC=false。"""
    import joyarm_core.joyarm.joyarm as _jm

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("注入故障：初始化失败")

    arm = None
    try:
        # 保存/替换/构造三行均在 try 内（首行保存必成功 → finally 恢复时 orig 恒
        # 有定义；make_arm 中途抛异常也能恢复替换，不把 _Boom 残留到进程结束）
        orig = _jm.MeshcatVisualizer
        _jm.MeshcatVisualizer = _Boom
        arm, _ = make_arm()
        i0 = _mark()
        e = _exc(arm.start_state_viz)
        time.sleep(0.15)
        urls = _viz_urls()
        ok_page, page = _fetch(urls[-1]) if urls else (False, b"")  # 取末位=本项端口（前面项的 URL 已停服）
        degraded = (_hit(_since(i0), "meshcat 初始化失败")
                    and _hit(_since(i0), "降级为仅 HTML 状态面板"))
        has_false = ok_page and b"HAS_MC = false" in page
        e2 = _exc(arm.stop_state_viz)
        ok = e is None and e2 is None and degraded and has_false
        ev = (f"注入 MeshcatVisualizer 故障 → start 不炸={e is None}；"
              f"降级 warn={degraded}；页面 HAS_MC=false={has_false}；"
              f"stop 不炸={e2 is None}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        _jm.MeshcatVisualizer = orig
        if arm is not None:
            arm.stop_state_viz()
            arm.close()


def _it82():
    """重复启停幂等 + stop→start 循环 + 线程退出：start×2 忽略、stop×2 无异常、循环重启可用、无 viz 线程残留。"""
    arm, _ = make_arm()
    try:
        i0 = _mark()
        arm.start_state_viz()
        time.sleep(0.1)
        arm.start_state_viz()                  # 重复启动：忽略 + info
        ign = _hit(_since(i0), "忽略重复启动")
        e1 = _exc(arm.stop_state_viz)
        e2 = _exc(arm.stop_state_viz)          # 重复停止：无异常
        # stop→start 循环：stop 须完成收尾（含 meshcat-server 终止与句柄清空），
        # 残留时 start 的重入守卫会拒绝（RuntimeError）——循环重启即收尾生效的回归
        e3 = _exc(arm.start_state_viz)
        time.sleep(0.25)                       # worker 10 Hz：至少 2 拍填充面板数据
        urls = _viz_urls()
        page_ok = bool(urls) and _fetch(urls[-1])[0]   # 本轮监控页可 GET
        e4 = _exc(arm.stop_state_viz)
        leftovers = [t for t in _joyarm_thread_names() if "viz" in t]
        ok = (ign and e1 is None and e2 is None and e3 is None and page_ok
              and e4 is None and not leftovers)
        ev = (f"重复 start 忽略 info={ign}；stop×2 异常={e1 is None and e2 is None}；"
              f"stop→start 循环重启不炸={e3 is None}（stop 收尾残留句柄会触发重入守卫"
              f"拒绝）、重启后监控页可 GET={page_ok}；二次 stop 异常={e4 is None}；"
              f"残留 viz 线程={leftovers or '无'}（daemon 线程均已退出，不阻脚本退出码）")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.stop_state_viz()
        arm.close()


# ---------------- 09 组（销毁路径） ----------------
def _it90():
    """close 幂等：close×2 无未捕获异常，且断连序列到达 backend 内核。"""
    arm, d = make_arm()
    try:
        arm.connect_arm()
        i0 = len(d.calls)
        e1, e2 = _exc(arm.close), _exc(arm.close)
        disc = d.calls_since(i0, "disconnect_arm")
        ok = e1 is None and e2 is None and bool(disc)
        ev = (f"close×1 异常={e1 is None}，×2 异常={e2 is None}；"
              f"disconnect_arm 内核帧={len(disc)}（断连序列已到达 backend）")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


def _it91():
    """with __exit__ 序列：退出触发 close（disable 先于 disconnect、arm 先于 end 断连）。"""
    arm, d = make_arm()
    e_conn = _exc(arm.connect_arm) or _exc(arm.connect_end)
    if e_conn is not None:
        arm.close()
        return "FAIL", f"前置连接失败：{e_conn!r}"
    i0 = len(d.calls)
    with arm:
        pass                                   # __exit__ → close()
    names = [c[0] for c in d.calls[i0:]]
    try:
        i_da = max(i for i, nm in enumerate(names) if nm == "disable_joint_arm")
        i_de = max(i for i, nm in enumerate(names) if nm == "disable_joint_end")
        i_xa = min(i for i, nm in enumerate(names) if nm == "disconnect_arm")
        i_xe = min(i for i, nm in enumerate(names) if nm == "disconnect_end")
        ordered = i_da < i_xa and i_de < i_xe and i_da < i_xe
        detail = (f"disable_arm@{i_da} < disable_end@{i_de} < "
                  f"disconnect_arm@{i_xa} < disconnect_end@{i_xe}")
    except ValueError:
        ordered, detail = False, f"序列不完整：{sorted(set(names))}"
    ev = f"with 退出后内核相对序：{detail}；涉及内核：{sorted(set(names))}"
    return ("PASS" if ordered else "FAIL"), ev


def _it92():
    """__del__ 冒烟：存在且 best-effort（del + gc 后对象回收、进程不崩）。"""
    import gc
    import weakref
    arm, _ = make_arm()
    ref = weakref.ref(arm)
    has_del = hasattr(type(arm), "__del__")
    del arm
    for _ in range(3):
        gc.collect()
    collected = ref() is None
    ok = has_del and collected
    ev = (f"JoyArm.__del__ 存在={has_del}（注明：__del__ 不做执行时序承诺，仅 "
          f"best-effort 调 close；正式收尾应走 with / close()）；del+gc 后对象"
          f"回收={collected}，进程无崩溃（本项运行至此即证据）")
    return ("PASS" if ok else "FAIL"), ev


# ============================================================
# 分层补充项：基础线程 → 目标管线 → 自愈 → 保护与并发
# ============================================================
def _require(condition, message: str) -> None:
    """验收断言不受 python -O 影响；失败由逐项执行器转为 FAIL。"""
    if not condition:
        raise AssertionError(message)


@contextmanager
def _case_arm():
    """确定性故障注入用例：真配置 + Dummy，冻结后台刷新，由用例控制快照。"""
    arm, backend = make_arm(lambda cfg: cfg['backend'].update(write_settle=0.001))
    try:
        arm.connect_arm()
        backend.close()
        backend.set_q('arm', arm.home_arm)
        arm.get_state_arm()
        yield arm, backend
    finally:
        arm.set_event(SafetyEvent.DISABLE)
        arm.close()
        backend.close()


def _probe_thread(fn):
    """并发检查保留子线程异常，主线程必须核验，避免异常仅打印而测试误通过。"""
    errors = []

    def run():
        try:
            fn()
        except Exception as e:
            errors.append(e)

    thread = threading.Thread(target=run, daemon=True)
    return thread, errors


def _it18():
    """prepare 仅建句柄；停止唤醒周期等待后不得额外执行一步。"""
    stop = threading.Event()
    calls = []

    class StopOnWait:
        def is_set(self):
            return stop.is_set()

        def wait(self, timeout):
            stop.set()
            return True

    _jm_mod._run_periodic(StopOnWait(), 1.0, lambda: calls.append(True))
    _require(not calls, '停止唤醒等待后仍执行了额外一拍')
    worker = _jm_mod._PeriodicThread(lambda: None, 10.0, 'verify-periodic')
    try:
        thread = worker.prepare()
        _require(worker.thread is thread and thread.ident is None, 'prepare 应登记未启动线程')
        worker.stop()
        _require(worker.thread is None, '未起跑句柄 stop 后没有回收')
        worker.set_hz(20.0)
        worker.start()
        _require(worker.is_running() and worker.hz == 20.0, '周期线程未正确启动/改频')
    finally:
        worker.stop()
    return 'PASS', 'prepare/未启动回收/重新 start/20Hz/停止不多执行一拍均通过'


def _it19():
    """stop 超时保留句柄并拒绝双启动，原线程退出后可回收重启。"""
    entered, release = threading.Event(), threading.Event()
    worker = _jm_mod._PeriodicThread(lambda: (entered.set(), release.wait(2.0)), 100.0,
                                     'verify-periodic')
    try:
        worker.start()
        _require(entered.wait(1.0), '阻塞周期未启动')
        with patch.object(_jm_mod, '_JOIN_TIMEOUT', 0.01):
            error = _exc(worker.stop)
        _require(isinstance(error, RuntimeError) and worker.is_alive(), 'stop 超时未保留存活句柄')
        _require(not worker.is_running(), '收到 stop 的线程仍被判定可运行')
        _require(isinstance(_exc(worker.start), RuntimeError), '旧线程存活时允许了双启动')
    finally:
        release.set()
        worker.stop()
    worker.start()
    worker.stop()
    return 'PASS', '阻塞 stop 超时→句柄保留→拒绝双启动→退出后回收/重启通过'


def _it43():
    """保护态选择控制器不改硬件模式，MOTION 检查成功后才恢复下发。"""
    with _case_arm() as (arm, backend):
        arm._current_motion_state = MotionState.DAMP
        arm.set_event(SafetyEvent.DAMPING)
        backend.set_mode_arm(ControlMode.MIT)
        mark = len(backend.calls)
        _require(arm.set_ctrl_solver(VERIFY_CTRL), '控制器选择失败')
        _require(not backend.calls_since(mark, 'set_joint_mode_arm'), '选择控制器触碰了硬件模式')
        _require(arm._maintain_motion_state(MotionState.DAMP, SafetyEvent.DAMPING), '阻尼维持失败')
        _require(backend.get_mode_arm() is ControlMode.MIT, '选择控制器破坏了 MIT 阻尼模式')
        calls = []
        arm._current_motion_state = MotionState.MOTION
        arm.set_event(SafetyEvent.NORMAL)
        with patch.object(arm._activated_ctrl_solver, 'step', lambda *args: calls.append(True)):
            arm._tick_ctrl()
            _require(not calls, 'FSM 确认模式前新控制器已执行')
            _require(arm._maintain_motion_state(MotionState.MOTION, SafetyEvent.NORMAL), 'MOTION 自愈失败')
            arm._activated_traj_solver.plan(arm, None)
            arm._tick_ctrl()
            _require(calls == [True] and backend.get_mode_arm() is ControlMode.VELOCITY,
                     'MOTION 未按当前控制器 MODE 恢复下发')
    return 'PASS', 'DAMP 保持 MIT；ctrl 选择后禁发；MOTION 核验 VELOCITY 后恢复 step'


def _it44():
    """在途 step 与控制器选择互斥，选择操作不下发切模式命令。"""
    with _case_arm() as (arm, backend):
        entered, release, switched = threading.Event(), threading.Event(), threading.Event()

        class BlockingController:
            MODE = ControlMode.POSITION

            def step(self, owner, frame, state):
                entered.set()
                _require(release.wait(2.0), '在途 step 未被释放')
                owner.send_position_arm(frame.q)

        arm._activated_ctrl_solver = BlockingController()
        arm._motion_ready = arm._activated_ctrl_solver
        arm._current_motion_state = MotionState.MOTION
        arm.set_event(SafetyEvent.NORMAL)
        backend.set_mode_arm(ControlMode.POSITION)
        arm._activated_traj_solver.plan(arm, None)
        worker, errors = _probe_thread(arm._tick_ctrl)
        arm._ctrl_thread = worker  # 仅测试夹具登记当前 probe；不是第二个控制线程。
        setter, setter_errors = _probe_thread(lambda: (arm.set_ctrl_solver(VERIFY_CTRL), switched.set()))
        mark = len(backend.calls)
        worker.start()
        try:
            _require(entered.wait(1.0), '在途 step 未进入')
            setter.start()
            _require(not switched.wait(0.05), '在途 step 未结束就切换了控制器')
        finally:
            release.set()
            worker.join(2.0)
            if setter.ident is not None:
                setter.join(2.0)
        _require(not worker.is_alive() and not setter.is_alive() and not errors and not setter_errors,
                 f'并发线程残留/异常：{errors + setter_errors}')
        _require(switched.is_set() and arm._motion_ready is None, '选择完成后未撤销下发许可')
        _require(not backend.calls_since(mark, 'set_joint_mode_arm'), '选择控制器发送了切模式命令')
    return 'PASS', '在途 step 阻止选择抢入；step 结束后选择完成且不切模式，未留线程'


def _it54():
    """验收反馈校验器本身拒绝陈旧/未来/坏维度/NaN/发送前快照。"""
    cfg = {'backend': {'state_refresh_hz': 20.0}}
    source = SimpleNamespace(n_arm=6, get_config=lambda: cfg)
    checks = []
    for kind in ('stale', 'future', 'shape', 'nan', 'before_send'):
        q, timestamp, after = np.zeros(6), time.time(), 0.0
        if kind == 'stale':
            timestamp -= 1.0
        elif kind == 'future':
            timestamp += 1.0
        elif kind == 'shape':
            q = np.zeros((6, 1))
        elif kind == 'nan':
            q[0] = np.nan
        else:
            after = timestamp + 0.001
        state = JointState(q=q, t=timestamp)
        source.get_state_arm = lambda: state
        source.joint_state_arm = state
        _require(_real_current_q(source, after=after)[0] is None, f'{kind} 反馈被误接纳')
        checks.append(kind)
    source.get_state_arm = lambda: None
    source.joint_state_arm = JointState(q=np.zeros(6), t=time.time())
    q, origin = _real_current_q(source)
    _require(q is not None and '快照' in origin, '新鲜缓存兜底失效')
    return 'PASS', f'拒绝 {checks}；新鲜缓存兜底可用'


def _it55():
    """真机运动判据离线自检：完全不动 FAIL，实际到位 PASS，裕量不足 SKIP。"""
    class Clock:
        now = 100.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

        def sleep(self, duration):
            self.now += duration

    verdicts = []
    for follows, margin, expected in ((False, 1.0, 'FAIL'), (True, 1.0, 'PASS'),
                                      (False, 0.01, 'SKIP')):
        clock = Clock()
        q = np.zeros(6)
        arm = SimpleNamespace(n_arm=6, get_config=lambda: {'backend': {'state_refresh_hz': 20.0}},
                              joint_limits_soft_arm=SimpleNamespace(q_min=np.full(6, -margin),
                                                                    q_max=np.full(6, margin)),
                              start_motion_fsm=lambda: None)

        def read():
            arm.joint_state_arm = JointState(q=q.copy(), t=clock.time())
            return arm.joint_state_arm

        def target(t, values):
            if follows:
                q[:] = values

        arm.get_state_arm, arm.set_target_joint = read, target
        read()
        with patch.object(sys.modules[__name__], 'time', clock), \
                patch.object(sys.modules[__name__], '_wait_state', lambda *args, **kw: True):
            verdict, evidence = _rt131(SimpleNamespace(arm=lambda: arm))
        _require(verdict == expected, f'运动判据应为 {expected}，实际 {verdict}：{evidence}')
        verdicts.append(verdict)
    return 'PASS', f'完全不动/实际到位/裕量不足分别为 {verdicts}（均为离线探针）'


def _it56():
    """公开目标通道贯通 plan→ctrl→后端，核对目标实际出现在发送参数与状态中。"""
    arm, backend = make_arm()
    try:
        backend.follow = True
        backend.set_q('arm', arm.home_arm)
        arm.connect_arm()
        arm.start_motion_fsm()
        _require(_wait_state(arm, MotionState.MOTION), '未进入 MOTION')
        target = arm.home_arm.copy()
        target[0] += 0.04
        mark = len(backend.calls)
        arm.set_target_joint(time.time() + 0.5, target)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not arm.is_in_q_arm(target, tol_q=0.01):
            time.sleep(0.02)
        sent = backend.calls_since(mark, 'send_joint_position_arm')
        _require(arm.is_in_q_arm(target, tol_q=0.01), '目标通道未驱动 Dummy 实际到位')
        _require(any(c[1] == 0 and abs(c[2] - target[0]) < 1e-9 for c in sent),
                 '后端未收到新的关节目标')
        return 'PASS', f'目标关节1 +0.04rad，经 plan/ctrl 下发并到位；记录位置帧 {len(sent)} 个'
    finally:
        _settle_fsm(arm, backend)
        arm.close()


def _it68():
    """逐态核对模式/使能恢复与一致时不重复写。"""
    checked = []
    for state, event, mode in ((MotionState.HOME, SafetyEvent.HOMING, ControlMode.POSITION),
                               (MotionState.HOLD, SafetyEvent.HOLDING, ControlMode.POSITION),
                               (MotionState.MOTION, SafetyEvent.NORMAL, ControlMode.POSITION),
                               (MotionState.DAMP, SafetyEvent.DAMPING, ControlMode.MIT),
                               (MotionState.ZERO, SafetyEvent.ZEROING, ControlMode.POSITION)):
        with _case_arm() as (arm, backend):
            arm._current_motion_state = state
            arm._hold_q, arm._zero_moving = arm.home_arm.copy(), True
            arm.set_event(event)
            backend.set_mode_arm(ControlMode.VELOCITY)
            backend.disable_arm()
            mark = len(backend.calls)
            _require(arm._maintain_motion_state(state, event), f'{state.name} 自愈失败')
            _require(backend.get_mode_arm() is mode and backend.is_enabled_arm is True,
                     f'{state.name} 模式/使能未恢复')
            _require(len(backend.calls_since(mark, 'set_joint_mode_arm')) == arm.n_arm
                     and len(backend.calls_since(mark, 'enable_joint_arm')) == arm.n_arm,
                     f'{state.name} 未完成整组恢复')
            mark = len(backend.calls)
            _require(arm._maintain_motion_state(state, event), f'{state.name} 一致态维持失败')
            _require(not backend.calls_since(mark, 'set_joint_mode_arm')
                     and not backend.calls_since(mark, 'enable_joint_arm'), f'{state.name} 一致仍重复写')
            checked.append(state.name)
    return 'PASS', f'{checked} 均恢复所需模式/使能，一致时零重复写'


def _it69():
    """一次切模式失败后跳拍；下拍恢复，事件始终由外部或 monitor 决定。"""
    with _case_arm() as (arm, backend):
        arm._current_motion_state = MotionState.DAMP
        arm.set_event(SafetyEvent.DAMPING)
        backend.set_mode_arm(ControlMode.POSITION)
        backend.fail_at['set_joint_mode_arm'] = None
        mark = len(backend.calls)
        arm._fsm_tick()
        _require(not backend.calls_since(mark, 'send_joint_mit_arm'), '自愈失败仍下发阻尼指令')
        backend.fail_at.clear()
        arm._fsm_tick()
        _require(backend.calls_since(mark, 'send_joint_mit_arm'), '下一拍未恢复')
        _require(arm.get_event() is SafetyEvent.DAMPING, '自愈改写了安全事件')
    return 'PASS', '一次失败→跳拍；故障解除→MIT 阻尼恢复，事件保持'


def _it73():
    """使能/模式持续失败只跳拍，不自动改写安全事件。"""
    for failed in ('set_joint_mode_arm', 'enable_joint_arm'):
        with _case_arm() as (arm, backend):
            arm._current_motion_state = MotionState.DAMP
            arm.set_event(SafetyEvent.DAMPING)
            backend.set_mode_arm(ControlMode.POSITION)
            backend.disable_arm()
            backend.fail_at[failed] = None
            mark = len(backend.calls)
            for _ in range(55):
                arm._fsm_tick()
                _require(arm.get_event() is SafetyEvent.DAMPING, f'{failed} 失败改写了事件')
                _require(not backend.calls_since(mark, 'send_joint_mit_arm'), f'{failed} 失败仍在下发')
            arm.set_event(SafetyEvent.DISABLE)
            arm._fsm_tick()
            _require(arm.get_current_motion_state() is MotionState.DISABLE
                     and backend.is_enabled_arm is False, f'{failed} 显式失能未完成')
    return 'PASS', '模式/使能各连续55拍失败不改事件、不发 MIT；显式 DISABLE 后失能'


def _it74():
    """首帧 plan 失败不提交 MOTION；异常由周期线程统一记录，不修改事件。"""
    for throws in (False, True):
        with _case_arm() as (arm, backend):
            arm._current_motion_state, arm._home_arrived = MotionState.HOME, True
            arm.set_event(SafetyEvent.NORMAL)

            def fail(*args):
                if throws:
                    raise RuntimeError('scripted first-plan failure')
                return False

            with patch.object(arm._activated_traj_solver, 'plan', fail):
                mark = len(backend.calls)
                for _ in range(3):
                    try:
                        arm._fsm_tick()
                    except RuntimeError as exc:
                        _require(throws and str(exc) == 'scripted first-plan failure', str(exc))
                    else:
                        _require(not throws, 'FSM 意外吞掉算法异常')
                    _require(arm.get_current_motion_state() is MotionState.HOME, '规划失败错误提交 MOTION')
                    _require(arm._ctrl_periodic is None and arm._plan_periodic is None,
                             '首帧失败后残留运动线程')
                _require(not backend.calls_since(mark, 'enable_joint_arm'), '首帧失败前提前使能')
                _require(arm.get_event() is SafetyEvent.NORMAL, '首帧持续失败改写事件')
    return 'PASS', 'plan=False / RuntimeError 均不改事件、不提交 MOTION，无半启动线程'


def _it75():
    """进入失败不得提交新状态或清除旧 HOLD 位置。"""
    with _case_arm() as (arm, backend):
        arm._current_motion_state, arm._hold_q = MotionState.HOLD, arm.home_arm.copy()
        arm.set_event(SafetyEvent.HOMING)
        backend.set_mode_arm(ControlMode.MIT)
        backend.fail_at['set_joint_mode_arm'] = None
        arm._fsm_tick()
        _require(arm.get_current_motion_state() is MotionState.HOLD,
                 '进入 HOME 失败却提交了状态')
        _require(np.array_equal(arm._hold_q, arm.home_arm), '进入失败清掉了原 HOLD 位置')
    return 'PASS', 'HOLD→HOME 模式失败：状态与 hold_q 保留'


def _it76():
    """ZERO 到位后保持失能，HOLD 自愈保持原锁定位形。"""
    with _case_arm() as (arm, backend):
        arm._current_motion_state, arm._zero_moving = MotionState.ZERO, True
        arm.set_event(SafetyEvent.ZEROING)
        backend.set_q('arm', arm.zero_arm)
        arm.get_state_arm()
        _require(arm._maintain_motion_state(MotionState.ZERO, SafetyEvent.ZEROING), '回零到位处理失败')
        _require(not arm._zero_moving and backend.is_enabled_arm is False, 'ZERO 到位未失能')
        mark = len(backend.calls)
        for _ in range(3):
            _require(arm._maintain_motion_state(MotionState.ZERO, SafetyEvent.ZEROING), 'ZERO 失能保持失败')
        _require(not backend.calls_since(mark, 'enable_joint_arm'), 'ZERO 到位后被自愈重新使能')
        arm._current_motion_state = MotionState.HOLD
        arm.set_event(SafetyEvent.HOLDING)
        _require(arm._enter_motion_state(MotionState.HOLD, SafetyEvent.HOLDING), 'HOLD 进入失败')
        saved = arm._hold_q.copy()
        backend.set_q('arm', arm.home_arm)
        backend.disable_arm()
        mark = len(backend.calls)
        _require(arm._maintain_motion_state(MotionState.HOLD, SafetyEvent.HOLDING), 'HOLD 自愈失败')
        sent = backend.calls_since(mark, 'send_joint_position_arm')
        _require(bool(sent) and all(c[2] == saved[c[1]] for c in sent), 'HOLD 自愈重采了漂移位置')
    return 'PASS', 'ZERO 到位后连续维持失能；HOLD 漂移后仍下发最初锁定位置'


def _it77():
    """阻塞模式读取期间新保护事件抢占旧目标。"""
    with _case_arm() as (arm, backend):
        arm.set_event(SafetyEvent.HOMING)
        backend.disable_arm()

        def read_mode():
            arm.set_event(SafetyEvent.DISABLE)
            return ControlMode.MIT

        mark = len(backend.calls)
        with patch.object(backend, 'get_mode_arm', read_mode):
            _require(not arm._maintain_motion_state(MotionState.HOME, SafetyEvent.HOMING),
                     '事件已改变仍按旧目标自愈')
        _require(all(not backend.calls_since(mark, name) for name in
                     ('enable_joint_arm', 'set_joint_mode_arm', 'send_joint_position_arm')),
                 'DISABLE 事件后仍进行了旧目标的模式/使能/位置下发')
    return 'PASS', '模式读取期间收到 DISABLE，旧 HOME 目标的切模式/使能/下发全部取消'


def _it78():
    """部分启动失败可回收，取消失败的保护迁移后可重建管线。"""
    with _case_arm() as (arm, backend):
        arm._current_motion_state, arm._home_arrived = MotionState.HOME, True
        arm.set_event(SafetyEvent.NORMAL)
        original = _jm_mod._PeriodicThread.start

        def start(worker):
            if worker.name == 'joyarm-ctrl':
                raise RuntimeError('scripted ctrl start failure')
            return original(worker)

        with patch.object(_jm_mod._PeriodicThread, 'start', start):
            try:
                arm._fsm_tick()
            except RuntimeError as exc:
                _require(str(exc) == 'scripted ctrl start failure', str(exc))
            else:
                raise AssertionError('未观察到注入的线程启动异常')
        _require(arm._ctrl_periodic is None and arm._plan_periodic is None,
                 '部分启动失败未回收')
        arm._fsm_tick()
        _require(arm.get_current_motion_state() is MotionState.MOTION, '部分启动失败后未恢复')
        backend.fail_at['read_joint_state_arm'] = None
        arm.set_event(SafetyEvent.HOLDING)
        arm._fsm_tick()
        _require(arm.get_current_motion_state() is MotionState.MOTION
                 and arm._ctrl_periodic is None, '失败保护迁移的旧线程未停止')
        backend.fail_at.clear()
        arm.set_event(SafetyEvent.NORMAL)
        arm._fsm_tick()
        _require(arm._ctrl_ready() and arm._ctrl_periodic.is_running() and arm._plan_periodic.is_running(),
                 '保护迁移取消后运动管线未重建')
    return 'PASS', 'ctrl 部分启动失败可回收恢复；HOLD 采样失败后恢复 NORMAL 可重建管线'


def _it79():
    """停止抢占在途自愈，最终失能后不得重新使能。"""
    with _case_arm() as (arm, backend):
        entered, release = threading.Event(), threading.Event()
        original = backend.get_mode_arm

        def read():
            entered.set()
            _require(release.wait(2.0), '自愈阻塞未释放')
            return original()

        arm._current_motion_state = MotionState.DAMP
        arm.set_event(SafetyEvent.DAMPING)
        backend.disable_arm()
        arm._move_wait_timeout = 0.01
        arm._fsm_periodic = _jm_mod._PeriodicThread(arm._fsm_worker, 100.0, 'verify-fsm')
        stopper, errors = _probe_thread(arm.stop_motion_fsm)
        mark = len(backend.calls)
        with patch.object(backend, 'get_mode_arm', read):
            arm._fsm_periodic.start()
            try:
                _require(entered.wait(1.0), '自愈拍未进入')
                stopper.start()
                _require(arm._fsm_shutdown.wait(1.0), 'stop 未进入最终收尾')
            finally:
                release.set()
                if stopper.ident is not None:
                    stopper.join(2.0)
        _require(not stopper.is_alive() and not errors and not arm._fsm_running(),
                 f'停机线程残留或异常：{errors}')
        _require(not backend.calls_since(mark, 'enable_joint_arm') and backend.is_enabled_arm is False,
                 '最终收尾后被在途自愈重新使能')
    return 'PASS', '自愈读模式与 stop 并发：收尾禁止使能，FSM/stop 退出，最终失能'


# ============================================================
# 测试项注册表（稳定编号独立于依赖层级；全离线低危）
# ============================================================
# 级别是执行依赖顺序；编号保留兼容，功能组用于定向筛选，不决定顺序。
LEVELS = {
    'L0': '基础线程、工厂、配置自检与只读模型',
    'L1': '硬件后端公开委托契约（Dummy）',
    'L2': '求解器、数学门面、目标通道与验收判据',
    'L3': 'FSM 基础迁移、自愈、失败恢复与运动管线',
    'L4': '安全监视、保护抢占与并发互斥',
    'L5': '可视化服务与重复启停',
    'L6': '销毁、上下文退出与资源回收',
}
_GROUP_LEVEL = {
    'factory': 'L0', 'init': 'L0', 'readonly': 'L0', 'delegate': 'L1',
    'solvers': 'L2', 'math': 'L2', 'feedback': 'L2', 'fsm': 'L3',
    'monitor': 'L4', 'viz': 'L5', 'close': 'L6',
}
_ITEM_LEVEL = {31: 'L4', 32: 'L4', 43: 'L4', 44: 'L4', 56: 'L3',
               65: 'L2', 66: 'L2', 77: 'L4', 79: 'L4'}
ITEMS: list[dict] = []


def _item(num, name, group, risk, purpose, fn):
    ITEMS.append(dict(num=num, name=name, group=group, risk=risk,
                      level=_ITEM_LEVEL.get(num, _GROUP_LEVEL[group]), purpose=purpose, fn=fn))


# —— 00 组（工厂与 listArm）——
_item(0, "未知名异常含可用列表", "factory", "低",
      "工厂（JoyArmFactory）与 get_backend 对未知名抛 ValueError 且消息附可用列表（共用同一实现）", _it00)
_item(1, "listArm 三条件", "factory", "低",
      "临时目录变量控制：三顶层键 / backend.name∈REGISTRY / 四路径齐全逐条件否决（另含语法错与空文件）", _it01)
_item(2, "模板排除（真实仓库）", "factory", "低",
      "listArm() 含 joyarm_dm、不含 joyarm_template（模板无 robot_model 目录，由路径条件自然排除）", _it02)
_item(3, "工厂成功路径（Dummy 注入）", "factory", "低",
      "四步校验全过：Dummy 注入下 joyArm('joyarm_dm') 返回 JoyArm 实例并输出创建成功/自检完成 info", _it03)

# —— 01 组（初始化自检：坏 cfg 逐条异常）——
_item(10, "缺顶层键", "init", "低",
      "删 robotics 顶层段 → ValueError『缺少顶层配置段』（cfg 自检）",
      _bad_cfg_item("缺顶层键 robotics", _m_del_top, "缺少顶层配置段"))
_item(11, "坏注册名", "init", "低",
      "robotics.fkine 注册名不存在 → ValueError『未在子包 REGISTRY』（cfg 自检）",
      _bad_cfg_item("坏注册名", _m_bad_reg, "未在子包 REGISTRY"))
_item(12, "缺 plan_hz", "init", "低",
      "traj 规划器缺 plan_hz → ValueError『缺 plan_hz（必配、无缺省回退）』（cfg 自检）",
      _bad_cfg_item("缺 plan_hz", _m_no_plan_hz, "缺 plan_hz"))
_item(13, "非整值 plan_hz", "init", "低",
      "plan_hz=10.5 → ValueError『须为 int 或整值浮点且 ≥1』（cfg 自检）",
      _bad_cfg_item("非整值 plan_hz", _m_frac_hz, "非法", "整值"))
_item(14, "home 越软限位", "init", "低",
      "home_arm[0]=9.9 越出软限位 → ValueError『越出软限位区间』（特征位形校验）",
      _bad_cfg_item("home 越软限位", _m_home_out, "越出软限位区间"))
_item(15, "软限位越硬限位", "init", "低",
      "软限位 q_min=-9.9 越出硬限位 → ValueError『越出对应硬限位』（cfg 自检）",
      _bad_cfg_item("软限位越硬限位", _m_soft_over, "越出", "硬限位"))
_item(16, "限位与 URDF 不符", "init", "低",
      "cfg 关节 q_min 与 URDF <limit> 偏差超容差 → ValueError『与 URDF <limit> 不一致』（cfg 自检）",
      _bad_cfg_item("限位与 URDF 不符", _m_urdf_lim, "与 URDF <limit> 不一致"))
_item(17, "数值叶 NaN / null 拒绝构造", "init", "低",
      "cfg 数值叶 NaN（home_arm[0] / robotics.ikine[0].damping）/ null（软限位 q_max[1]）→ ValueError『非有限数值』/『为 null』且消息带 cfg 路径（不再静默冻进数组或求解器参数）", _it17)

# —— 02 组（只读属性与防篡改）——
_item(20, "writeable=False 防篡改", "readonly", "低",
      "全部对外数组（特征位形/硬软限位四键/tcp workspace_box）原地写均抛异常", _it20)
_item(21, "get_config 深拷贝", "readonly", "低",
      "篡改 get_config() 返回值（home/限位/帧名）不反哺内部状态与后续读取", _it21)
_item(22, "只读引用零拷贝", "readonly", "低",
      "只读 @property 多次访问返回同一本体（零拷贝）、与 get_config 值一致、pin_model.nq=n_arm", _it22)
_item(23, "get_joint_names 1 基边界", "readonly", "低",
      "get_joint_names_arm/end：'all' 返回全名 tuple、1 基序号 1/n 取首末名；越界（0、n+1）与非法类型（str/bool）warn 返回 None", _it23)

# —— 03 组（委托面与互斥）——
_item(30, "委托面透传核对", "delegate", "低",
      "JoyArm 公开方法逐项核对 DummyBackend 对应内核（调用名 + 关键参数 + 返回值透传；arm/end 对称）", _it30)
_item(31, "FSM 运行中外部指令互斥", "delegate", "低",
      "FSM 运行中外部线程的 arm 侧指令（send_position/set_mode/enable/clear_error/set_zero）空操作 + warn；只读与 end 侧放行", _it31)
_item(32, "ctrl 线程豁免", "delegate", "低",
      "MOTION 态 ctrl 线程自身的 send_position_arm 照常到达 backend（_is_internal_thread 豁免）", _it32)

# —— 04 组（求解器域）——
_item(40, "六域查询与切换联动", "solvers", "低",
      "get_*_solvers=cfg 名单、默认激活=cfg 首位、切换后激活名/plan_hz/ctrl_hz 更新；控制器可离线选择，不触碰硬件模式", _it40)
_item(41, "坏名 warn 保持原激活", "solvers", "低",
      "六域 set_*('no_such_solver') 均返回 False + warn『未在该域已加载成员中』且激活不变", _it41)
_item(42, "traj 先 plan 后换指针", "solvers", "低",
      "切换成功即新规划器已完成 plan（sample 立即可用）；断连 plan 失败保持原激活 + warn", _it42)

# —— 05 组（数学门面）——
_item(50, "9 方法 vs pinocchio 原生对照", "math", "低",
      "显式传参离线计算与 pinocchio 原生 API 同输入对照：fkine/ikine/ikine_all/jac/jac_pinv/idyn/mass/coriolis/gravity + 缺省硬件态契约", _it50)
_item(51, "到位判断三函数", "math", "低",
      "is_in_q_arm / is_in_q_end / is_in_pose_arm：Dummy 快照注入核对到位/偏离/容差边界；未连接无快照 False+warn；维度/容差长度 ValueError、pose 类型 TypeError", _it51)
_item(52, "rand_q 限位内采样", "math", "低",
      "rand_q_arm / rand_q_end 批量样本 ∈ 软/硬限位区间且全有限、非退化（缺省软限位 + 显式限位入参两口径）", _it52)
_item(53, "frame=0 基座帧回归", "math", "低",
      "solve_fkine(q, frame=0) / get_jac(q, frame=0) 按 universe 帧求解且与缺省末端帧结果可区分（显式 0 不被当假值吞掉）", _it53)

# —— 06 组（MotionFSM：DummyBackend 推演）——
_item(60, "start 前置 RuntimeError", "fsm", "低",
      "未连接 start_motion_fsm 抛 RuntimeError『arm 未连接』（拒绝启动）", _it60)
_item(61, "自动迁移序列", "fsm", "低",
      "start 后 DISABLE→HOME→MOTION 两跳迁移（路线表经 home 中转）+『HOME 已到位』日志 + home 指令驱动", _it61)
_item(62, "home 到位门槛", "fsm", "低",
      "q 未到位时停留 HOME 不进 MOTION（home 到位门槛）；到位后放行迁移", _it62)
_item(63, "保护事件直达", "fsm", "低",
      "MOTION 下 HOLDING/DAMPING/DISABLE 依次直达 + 各态维持指令（hold_q 重发/MIT 纯阻尼/失能轮发）核对", _it63)
_item(64, "stop 超时强推", "fsm", "低",
      "回零永不到位 → 等满 move_wait_timeout 后 warn『超时强制推进』并完成失能与线程收尾", _it64)
_item(65, "set_target 拒绝路径", "fsm", "低",
      "set_target_joint / set_target_cart：缺必填（q/pose）、维度不符、t 非标量、目标序列非 TrajFrame 均 ValueError（消息含字段名）；拒绝后合法写入不受影响", _it65)
_item(66, "目标帧通道 deepcopy 隔离", "fsm", "低",
      "get_target_trajframe 返回深拷贝（外部改写不反哺内部）；set 目标对入参深拷贝（set 后改入参不影响内部）；None 清空通道", _it66)
_item(67, "wait_for_motion_state 四路径", "fsm", "低",
      "到达返 1（运行中等到 MOTION）；超时/FSM 未运行/非枚举返 0 且各有 warn；timeout 非正有限抛 ValueError", _it67)

# —— 07 组（safe_monitor）——
_item(70, "硬规则 error≥2→DAMPING", "monitor", "低",
      "单关节故障码触发监视器 set_event(DAMPING)，FSM 迁入 DAMP 并维持 MIT 阻尼", _it70)
_item(71, "断连/连续 None→DISABLE", "monitor", "低",
      "断连仅一次告警挂起、状态保持且禁发，重连处理保护事件后可显式恢复；持续None→DISABLE，规则仍收None、ctrl跳拍", _it71)
_item(72, "自定义规则与同拍仲裁", "monitor", "低",
      "命中合并取危险最高（硬+自定义统一取最危险，与到达顺序无关）、异常/非法返回忽略且不影响其他规则", _it72)

# —— 08 组（可视化）——
_item(80, "start/stop 离线不炸", "viz", "低",
      "未连接 start_state_viz → 页面 + /state.json 轮询端点可 GET；stop 后 HTTP 端口关闭", _it80)
_item(81, "meshcat 初始化失败降级", "viz", "低",
      "注入 MeshcatVisualizer 故障：start 不炸、warn『初始化失败→仅 HTML 状态面板』、页面 HAS_MC=false", _it81)
_item(82, "重复启停幂等 + 线程退出", "viz", "低",
      "start×2 忽略重复、stop×2 无异常、无 joyarm-viz 线程残留（daemon 均退出）", _it82)

# —— 09 组（销毁路径）——
_item(90, "close 幂等", "close", "低",
      "close×2 无未捕获异常，断连序列到达 backend 内核", _it90)
_item(91, "with __exit__ 序列", "close", "低",
      "with 退出触发 close：disable_arm/disable_end 先于 disconnect_arm/disconnect_end", _it91)
_item(92, "__del__ 冒烟", "close", "低",
      "__del__ 存在且 best-effort（del+gc 回收、进程不崩；不做执行时序承诺——正式收尾走 with/close）", _it92)

# 新增项目仍逐项打印目的与证据；原有 00~92 编号保持不变。
_item(18, '周期线程基础契约', 'init', '低',
      'prepare/未启动回收/改频重启；stop 唤醒等待后不得额外执行一拍', _it18)
_item(19, '周期线程超时与防双启动', 'init', '低',
      '阻塞 stop 保留句柄，拒绝双启动，原线程退出后可回收重启', _it19)
_item(43, '控制器选择与 FSM 模式归属', 'solvers', '低',
      'DAMP 中选择控制器不改 MIT；进入 MOTION 由 FSM 按 MODE 自愈后才恢复下发', _it43)
_item(44, '控制器切换等待在途 step', 'solvers', '低',
      '阻塞 step 与选择互斥，选择完成撤销旧许可，不发送切模式命令，无残留线程', _it44)
_item(54, '验收反馈新鲜度自检', 'feedback', '低',
      '拒绝过期/未来/维度异常/NaN/发送前反馈；新鲜缓存允许兜底', _it54)
_item(55, '运动验收判据自检', 'feedback', '低',
      '离线探针验证完全不动 FAIL、实际到位 PASS、裕量不足 SKIP，防假通过', _it55)
_item(56, '目标到后端的完整运动管线', 'fsm', '低',
      '公共目标通道→plan→ctrl→Dummy 位置指令与反馈到位，核对真实发送参数', _it56)
_item(68, '五种运动态模式/使能自愈', 'fsm', '低',
      'HOME/HOLD/MOTION/DAMP/ZERO 运动中逐态自愈；一致时不重复写', _it68)
_item(69, '自愈暂时失败后恢复', 'fsm', '低',
      '切模式失败只跳拍，下一拍恢复后下发，事件保持', _it69)
_item(73, '自愈连续失败职责边界', 'fsm', '低',
      '模式/使能各连续55拍失败不改事件、不下发；外部显式 DISABLE 后迁移', _it73)
_item(74, '首帧规划失败与异常回收', 'fsm', '低',
      '首帧 plan=False/抛异常不改事件，失败不提交 MOTION、不留线程', _it74)
_item(75, '进入失败保留旧态缓存', 'fsm', '低',
      'HOLD→HOME 切模式失败不提交新态、不清 hold_q，失败次数递增', _it75)
_item(76, 'ZERO 失能与 HOLD 锁定位形', 'fsm', '低',
      'ZERO 到位后不被自愈重新使能；HOLD 自愈不重新采集漂移位置', _it76)
_item(77, '保护事件抢占自愈', 'fsm', '低',
      '模式读取期间收到 DISABLE，旧 HOME 目标的切模式/使能/下发取消', _it77)
_item(78, '部分启动与取消保护后的恢复', 'fsm', '低',
      'ctrl 启动失败清理 plan/ctrl；HOLD 进入失败后恢复 NORMAL 可重建运动管线', _it78)
_item(79, '停止与在途自愈并发', 'fsm', '低',
      '最终收尾先禁止使能/控制，停线程后失能；在途自愈不得重新使能', _it79)

ALL_NUMS = {it['num'] for it in ITEMS}
_BY_NUM = {it['num']: it for it in ITEMS}
GROUPS: dict[str, list[int]] = {}
for _entry in ITEMS:
    GROUPS.setdefault(_entry['group'], []).append(_entry['num'])
GROUPS['all'] = list(_BY_NUM)


# ============================================================
# 真机模式（--real）：真配置真连接 backend_dm，分级危险确认
# ============================================================
_RISK_ORDER = {"低": 0, "中": 1, "高": 2, "极高": 3}

# 真机运动项保守参数（幅度小；对齐 backend_verify 运动组惯例）
REAL_SETTLE = 1.5                       # 运动后观测等待（秒）
REAL_POS_AMP = 0.05                     # 委托位置指令小幅幅度（rad）
REAL_VLIM = 0.3                         # 位置指令速度上限（rad/s）
REAL_MIT_KP, REAL_MIT_KD = 5.0, 0.5     # MIT 原地保持保守增益
REAL_VEL = 0.1                          # 速度指令幅值（rad/s）
REAL_VEL_DUR = 0.5                      # 速度指令持续时间（秒）
REAL_ZERO_TOL = 0.05                    # 设零后回读容差（rad）
REAL_WAIT_HOME = 8.0                    # FSM 启动等 HOME 迁移上限（秒）
REAL_WAIT_MOTION = 30.0                 # home 到位迁 MOTION 上限（秒，真实运动耗时）


def _confirm(level: str, args: argparse.Namespace) -> str:
    """风险确认：返回 run/skip/quit（对齐 backend_verify.py 惯例）。

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


_real_banner_done = False


def _real_banner_once():
    """进入真机段一次性总告知（对齐 backend_verify 惯例）。"""
    global _real_banner_done
    if not _real_banner_done:
        _real_banner_done = True
        print("─" * 68)
        print("⚠ 进入真机测试段：请确认机械臂工作空间已清空、急停可用、有人监护")


class _RealSession:
    """真机会话：惰性构造真实 JoyArm（无注入，backend_dm 真连接）+ 共享收尾。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._arm: Optional[JoyArm] = None

    def arm(self) -> JoyArm:
        if self._arm is None:
            self._arm = JoyArm(MODEL)   # 真实 cfg（backend_dm）；构造无总线 I/O
        return self._arm

    def close(self) -> None:
        if self._arm is not None:
            try:
                self._arm.close()
            except Exception as e:  # noqa: BLE001（验收脚本须捕获一切）
                print(f"（真机会话 close 异常（记录不吞）：{e!r}）")
            self._arm = None


def _real_prefill_fams(fam: str) -> tuple:
    return ("arm", "end") if fam == "both" else (fam,)


def _plan_real_prefill(s: _RealSession, needs: list) -> list:
    """干跑：计算将补齐哪些前置（不动状态），返回 [(动作描述, 风险)]。

    用于确认等级计算（前置风险并入该项确认）与确认文案（对齐 backend_verify）。
    """
    plans: list = []
    planned_conn: set = set()
    planned_mode: dict = {}
    planned_enabled: set = set()
    arm = s._arm
    for nd in needs:
        kind = nd[0]
        for fam in _real_prefill_fams(nd[1]):
            if fam == "end" and (arm is None or arm.n_end == 0):
                continue                          # 空族（n_end=0）无前置可言
            conn = ((getattr(arm, f"is_connected_{fam}", None) if arm is not None else None)
                    is True or fam in planned_conn)
            if kind == "connected":
                if not conn:
                    plans.append((f"connect_{fam}", "低"))
                    planned_conn.add(fam)
            elif kind == "mode":
                mode = nd[2]
                mode_cur = (getattr(arm, f"get_mode_{fam}")() if arm is not None else None)
                if mode_cur != mode and planned_mode.get(fam) != mode:
                    if not conn:
                        plans.append((f"connect_{fam}", "低"))
                        planned_conn.add(fam)
                    plans.append((f"set_mode_{fam}({mode.name})", "低"))
                    planned_mode[fam] = mode
            elif kind == "enabled":
                enabled = ((getattr(arm, f"is_enabled_{fam}", None) if arm is not None else None)
                           is True or fam in planned_enabled)
                if not enabled:
                    if not conn:
                        plans.append((f"connect_{fam}", "低"))
                        planned_conn.add(fam)
                    plans.append((f"enable_{fam}", "高"))
                    planned_enabled.add(fam)
    return plans


def _real_prefill(s: _RealSession, needs: list) -> tuple:
    """按声明顺序补齐缺失前置（已就绪则跳过；低危静默补齐，高危并入项级确认）。

    返回 (状态, 已执行的补齐动作描述, 原因)，状态：
      "ok" 全部就绪；"skip" 所需模式不可用（型号能力缺失，判 SKIP 非缺陷）；
      "fail" 连接/使能等核心前置未达成，判 FAIL。
    """
    if not needs:
        return "ok", [], ""
    arm = s.arm()
    actions: list = []
    for nd in needs:
        kind = nd[0]
        for fam in _real_prefill_fams(nd[1]):
            if fam == "end" and arm.n_end == 0:
                continue
            if kind == "connected":
                if getattr(arm, f"is_connected_{fam}") is True:
                    continue
                getattr(arm, f"connect_{fam}")()
                actions.append(f"connect_{fam}")
                if getattr(arm, f"is_connected_{fam}") is not True:
                    return "fail", actions, f"connect_{fam} 未成功（详见上方日志）"
            elif kind == "mode":
                mode = nd[2]
                if getattr(arm, f"get_mode_{fam}")() == mode:
                    continue
                if getattr(arm, f"is_connected_{fam}") is not True:   # 防御：先保连接
                    getattr(arm, f"connect_{fam}")()
                    actions.append(f"connect_{fam}")
                    if getattr(arm, f"is_connected_{fam}") is not True:
                        return "fail", actions, f"connect_{fam} 未成功（详见上方日志）"
                ret = getattr(arm, f"set_mode_{fam}")(mode)
                actions.append(f"set_mode_{fam}({mode.name})")
                if ret != 1:
                    return "skip", actions, (f"{fam} 族所需模式 {mode.name} 不可用"
                                             f"（set_mode 返回 0——型号能力缺失或硬件拒绝，非缺陷）")
            elif kind == "enabled":
                if getattr(arm, f"is_enabled_{fam}") is True:
                    continue
                if getattr(arm, f"is_connected_{fam}") is not True:
                    return "fail", actions, "连接缺失"
                ret = getattr(arm, f"enable_{fam}")()
                actions.append(f"enable_{fam}")
                if ret != 1:
                    return "fail", actions, f"enable_{fam} 未成功（详见上方日志）"
    return "ok", actions, ""


def _describe_real_needs(needs: list) -> str:
    parts = []
    for nd in needs:
        fams = "/".join(_real_prefill_fams(nd[1]))
        if nd[0] == "connected":
            parts.append(f"已连接（{fams}）")
        elif nd[0] == "enabled":
            parts.append(f"已使能（{fams}）")
        else:
            parts.append(f"{nd[2].name} 模式（{fams}）")
    return "、".join(parts)


def _real_state_fresh(arm: JoyArm, fam: str, state, after: float = 0.0) -> bool:
    """真机运动判据：整组 q 有效、帧龄在承诺周期内，必要时须为发送后的帧。"""
    if state is None:
        return False
    try:
        q = np.asarray(state.q, dtype=float)
        t = float(state.t)
        n = getattr(arm, f"n_{fam}")
        hz = float(arm.get_config()["backend"].get("state_refresh_hz", 20.0))
        return bool(q.shape == (n,) and n > 0 and np.isfinite(q).all()
                    and np.isfinite(t) and t > 0 and t >= after
                    and 0 <= time.time() - t <= 1.0 / hz)
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def _real_current_q(arm: JoyArm, fam: str = "arm", after: float = 0.0) -> tuple:
    """读时结果与缓存均校验帧龄，after 可要求发送后的反馈。"""
    try:
        st = getattr(arm, f"get_state_{fam}")()
    except Exception:
        st = None
    snap = getattr(arm, f"joint_state_{fam}")
    for candidate, src in ((st, f"get_state_{fam} 实测"), (snap, f"joint_state_{fam} 快照")):
        if _real_state_fresh(arm, fam, candidate, after):
            return np.asarray(candidate.q, dtype=float).copy(), src
    return None, "无新鲜有效状态（帧龄/时间戳/位置维度不满足要求）"


def _real_safe_target(arm: JoyArm, fam: str, q0: np.ndarray, amp: float) -> np.ndarray:
    """自当前实测位置的小幅目标：逐关节按软限位裕量选向 ±amp（至多走半裕量，不贴限位）。"""
    soft = getattr(arm, f"joint_limits_soft_{fam}")
    q_min = np.asarray(soft.q_min, dtype=float)
    q_max = np.asarray(soft.q_max, dtype=float)
    target = q0.copy()
    for i in range(target.size):
        up, dn = q_max[i] - q0[i], q0[i] - q_min[i]
        step = min(amp, 0.5 * up) if up >= dn else -min(amp, 0.5 * dn)
        target[i] = q0[i] + step
    return target


# ---------------- r0 组（真机：连接与硬件只读，低危） ----------------
def _rt100(s: _RealSession):
    """真机构建与自检：真实 cfg（backend_dm）构造 JoyArm，自检通过且无总线 I/O。"""
    i0 = _mark()
    arm = s.arm()
    cfg = arm.get_config()
    ok = (isinstance(arm, JoyArm) and cfg["backend"]["name"] == "backend_dm"
          and arm.n_arm == len(cfg["backend"]["arm"]["joints"])
          and arm.is_connected_arm is not True     # 构造契约：不做任何总线 I/O
          and _hit(_since(i0), "初始化和自检完毕"))
    ev = (f"JoyArm('{MODEL}')（真实 cfg，无注入）构造成功；backend="
          f"{cfg['backend']['name']}，n_arm={arm.n_arm}、n_end={arm.n_end}；"
          f"repr={_trunc(repr(arm), 90)}；自检完成 info="
          f"{_hit(_since(i0), '初始化和自检完毕')}；连接标志="
          f"{arm.is_connected_arm}（构造不做总线 I/O）")
    return ("PASS" if ok else "FAIL"), ev


def _rt101(s: _RealSession):
    """arm/end 连接：connect 后连接标志 True、default_mode 回填一致。"""
    arm = s.arm()
    arm.connect_arm()
    ok_a = arm.is_connected_arm is True
    m_a = arm.get_mode_arm()
    if arm.n_end:
        arm.connect_end()
        ok_e = arm.is_connected_end is True
    else:
        ok_e = True
    default = str((arm.get_config()["backend"].get("arm") or {}).get("default_mode", "")).upper()
    ok = ok_a and ok_e and (m_a is None or m_a.name == default or default == "")
    ev = (f"connect_arm() → is_connected_arm={arm.is_connected_arm}；get_mode_arm()="
          f"{getattr(m_a, 'name', m_a)}（cfg default_mode={default or '（未配）'}，"
          f"回填一致或无回读能力均不算失败）；n_end={arm.n_end}，"
          f"connect_end() → is_connected_end={arm.is_connected_end}")
    return ("PASS" if ok else "FAIL"), ev


def _rt102(s: _RealSession):
    """状态读取：get_state 输出实测值（q/dq/tau/温度/error）并校验契约（(n,) 有限、t 更新）。"""
    arm = s.arm()
    checks = []
    for fam in ("arm", "end"):
        if fam == "end" and arm.n_end == 0:
            continue
        st = getattr(arm, f"get_state_{fam}")()
        n = getattr(arm, f"n_{fam}")
        if st is None or st.q is None:
            checks.append((f"{fam} get_state 返回可用快照", False, "None"))
            continue
        q = np.asarray(st.q, dtype=float)
        fields_ok = (q.shape == (n,) and bool(np.all(np.isfinite(q)))
                     and float(getattr(st, "t", 0.0) or 0.0) > 0)
        vals = (f"q={np.round(q, 4).tolist()}"
                f"{'，dq=' + str(np.round(np.asarray(st.dq, float), 3).tolist()) if st.dq is not None else ''}"
                f"{'，error=' + str(np.asarray(st.error).reshape(-1).tolist()) if st.error is not None else ''}"
                f"，t={float(getattr(st, 't', 0.0) or 0.0):.3f}")
        checks.append((f"{fam} get_state 契约（(n,) 有限、t>0）", fields_ok, vals))
    snap_age = time.time() - float(getattr(arm.joint_state_arm, "t", 0.0) or 0.0)
    checks.append(("被动快照新鲜（age<0.5s，刷新线程在跑）",
                   0 < snap_age < 0.5, f"age={snap_age:.3f}s"))
    ok = all(c for _, c, _ in checks)
    ev = "；".join(f"{lb}={'OK' if c else 'FAIL'}（{dt}）" for lb, c, dt in checks)
    return ("PASS" if ok else "FAIL"), ev


def _rt103(s: _RealSession):
    """模式读取：get_mode 返回 default_mode 回填值（None=无回读能力，SKIP 非缺陷）。"""
    arm = s.arm()
    m_a = arm.get_mode_arm()
    m_e = arm.get_mode_end() if arm.n_end else None
    default = str((arm.get_config()["backend"].get("arm") or {}).get("default_mode", "")).upper()
    if m_a is None:
        return "SKIP", f"get_mode_arm()=None（无模式回读能力或各关节不一致，非缺陷）"
    ok = m_a.name == default
    ev = (f"get_mode_arm()={m_a.name}（cfg default_mode={default}）一致={ok}；"
          f"get_mode_end()={getattr(m_e, 'name', m_e)}")
    return ("PASS" if ok else "FAIL"), ev


def _rt104(s: _RealSession):
    """错误码读取与检查：get_error 逐关节码 + check_error 与之一致。"""
    arm = s.arm()
    checks = []
    for fam in ("arm", "end"):
        if fam == "end" and arm.n_end == 0:
            continue
        errs = getattr(arm, f"get_error_{fam}")()
        chk = getattr(arm, f"check_error_{fam}")()
        if errs is None:
            checks.append((f"{fam} get_error 可读", False, "None"))
            continue
        arr = np.asarray(errs).reshape(-1)
        consistent = chk == bool(np.all(arr < 2))
        checks.append((f"{fam} get_error/check_error 一致", consistent,
                       f"error={arr.tolist()}，check={chk}（0=失能 1=使能 ≥2=故障）"))
    ok = all(c for _, c, _ in checks)
    ev = "；".join(f"{lb}={'OK' if c else 'FAIL'}（{dt}）" for lb, c, dt in checks)
    return ("PASS" if ok else "FAIL"), ev


def _rt105(s: _RealSession):
    """参数读取：read_param('kp') 返回逐关节 list（键不可用 SKIP 非缺陷）。"""
    arm = s.arm()
    vals = arm.read_param_arm("kp")
    if vals is None:
        return "SKIP", ("read_param_arm('kp')=None（参数键不可用或无读取能力——"
                        "非缺陷；可用 --step 跑其余项）")
    ok = len(vals) == arm.n_arm and all(np.isfinite(v) for v in vals)
    ev = f"read_param_arm('kp')={np.round(np.asarray(vals, float), 3).tolist()}（n={arm.n_arm}）"
    return ("PASS" if ok else "FAIL"), ev


# ---------------- r1 组（真机：使能/模式/参数写） ----------------
def _rt110(s: _RealSession):
    """三模式设置：MIT/POSITION/VELOCITY 循环 设→读回（回读能力缺失仅记录）。"""
    arm = s.arm()
    checks = []
    for mode in (ControlMode.MIT, ControlMode.POSITION, ControlMode.VELOCITY):
        r = arm.set_mode_arm(mode)
        back = arm.get_mode_arm()
        ok = r == 1 and (back is None or back is mode)   # 回读 None=无回读能力，不算失败
        checks.append((f"set_mode({mode.name})→读回", ok,
                       f"ret={r}, 回读={getattr(back, 'name', back)}"))
    r = arm.set_mode_arm(ControlMode.POSITION)            # 收尾回 POSITION
    ok = all(c for _, c, _ in checks) and r == 1
    ev = ("；".join(f"{lb}={'OK' if c else 'FAIL'}（{dt}）" for lb, c, dt in checks)
          + f"；收尾回 POSITION ret={r}")
    return ("PASS" if ok else "FAIL"), ev


def _rt111(s: _RealSession):
    """arm 使能/失能：enable(1/True) → disable(1/False)（电机上力，高危）。"""
    arm = s.arm()
    r1 = arm.enable_arm()
    e1 = arm.is_enabled_arm
    r2 = arm.disable_arm()
    e2 = arm.is_enabled_arm
    ok = r1 == 1 and e1 is True and r2 == 1 and e2 is False
    ev = (f"enable_arm() ret={r1}，is_enabled={e1}；disable_arm() ret={r2}，"
          f"is_enabled={e2}（推导口径：全 1→True、全 0→False、混合→None）")
    return ("PASS" if ok else "FAIL"), ev


def _rt112(s: _RealSession):
    """使能/失能幂等：enable×2、disable×2 不炸且终态 False。"""
    arm = s.arm()
    r1 = arm.enable_arm()
    r2 = arm.enable_arm()
    d1 = arm.disable_arm()
    d2 = arm.disable_arm()
    e = arm.is_enabled_arm
    ok = all(isinstance(v, int) for v in (r1, r2, d1, d2)) and e is False
    ev = (f"enable×2 ret=({r1},{r2})；disable×2 ret=({d1},{d2})；"
          f"终态 is_enabled={e}（重复调用安全、无异常）")
    return ("PASS" if ok else "FAIL"), ev


def _rt113(s: _RealSession):
    """参数写回：读 kp 原值 → 写回原值 → 读回核对（无副作用）。"""
    arm = s.arm()
    vals = arm.read_param_arm("kp")
    if vals is None:
        return "SKIP", "read_param_arm('kp')=None（参数键不可用——非缺陷）"
    w = arm.write_param_arm("kp", vals)
    back = arm.read_param_arm("kp")
    same = back is not None and bool(np.allclose(np.asarray(back, float),
                                                 np.asarray(vals, float), rtol=1e-6))
    ok = w == 1 and same
    ev = (f"原值={np.round(np.asarray(vals, float), 3).tolist()} → write ret={w} → "
          f"读回={np.round(np.asarray(back, float), 3).tolist() if back else None}，"
          f"一致={same}（写回原值，无副作用）")
    return ("PASS" if ok else "FAIL"), ev


def _rt114(s: _RealSession):
    """end 使能/失能（与 111 对称；空族自动 SKIP）。"""
    arm = s.arm()
    if arm.n_end == 0:
        return "SKIP", "end 族未配置任何电机（n_end=0），空族自动跳过"
    r1 = arm.enable_end()
    e1 = arm.is_enabled_end
    r2 = arm.disable_end()
    e2 = arm.is_enabled_end
    ok = r1 == 1 and e1 is True and r2 == 1 and e2 is False
    ev = (f"enable_end() ret={r1}，is_enabled={e1}；disable_end() ret={r2}，"
          f"is_enabled={e2}")
    return ("PASS" if ok else "FAIL"), ev


# ---------------- r2 组（真机：委托指令小幅运动，高危 + 发送前参数预览） ----------------
def _rp120(s: _RealSession):
    """120 发送前参数预览：当前实测 q vs 小幅目标 q（+vlim）。"""
    arm = s.arm()
    q0, src = _real_current_q(arm, "arm")
    if q0 is None:
        print(f"（预览：{src}——无法自实际位置小幅出发）")
        return
    target = _real_safe_target(arm, "arm", q0, REAL_POS_AMP)
    print(f"  当前 q（{src}）={np.round(q0, 4).tolist()}")
    print(f"  目标 q（±{REAL_POS_AMP} rad 按软限位裕量选向）={np.round(target, 4).tolist()}")
    print(f"  vlim={REAL_VLIM} rad/s（位置指令速度上限）")


def _rt120(s: _RealSession):
    """send_position 小幅：自实测位置 ±0.05 rad（软限位裕量选向）、限速 0.3 rad/s。"""
    arm = s.arm()
    q0, src = _real_current_q(arm, "arm")
    if q0 is None:
        return "FAIL", f"{src}——取消发送，无法自实际位置小幅出发"
    target = _real_safe_target(arm, "arm", q0, REAL_POS_AMP)
    sent_at = time.time()
    arm.send_position_arm(target, np.full(arm.n_arm, REAL_VLIM))
    time.sleep(REAL_SETTLE)
    q1, src1 = _real_current_q(arm, "arm", after=sent_at)
    moved = q1 is not None and float(np.max(np.abs(q1 - q0))) > 0.005
    delta = "—" if q1 is None else f"{float(np.max(np.abs(q1 - q0))):.4f}"
    ok = moved
    ev = (f"目标（{src} ±{REAL_POS_AMP}）={np.round(target, 4).tolist()}；"
          f"{REAL_SETTLE}s 后实测 q（{src1}）"
          f"={np.round(q1, 4).tolist() if q1 is not None else None}，"
          f"|Δq|max={delta} rad（>0.005 判有响应）")
    return ("PASS" if ok else "FAIL"), ev


def _rp121(s: _RealSession):
    """121 发送前参数预览：MIT 原地保持参数。"""
    arm = s.arm()
    q0, src = _real_current_q(arm, "arm")
    if q0 is None:
        print(f"（预览：{src}）")
        return
    print(f"  当前 q（{src}）={np.round(q0, 4).tolist()}（MIT 目标=当前实测，原地保持）")
    print(f"  kp={REAL_MIT_KP}, kd={REAL_MIT_KD}（保守增益），tau=0, dq=0")


def _rt121(s: _RealSession):
    """send_mit 原地保持：当前实测位置为目标（保守增益），1s 后漂移 <0.05 rad。"""
    arm = s.arm()
    q0, src = _real_current_q(arm, "arm")
    if q0 is None:
        return "FAIL", f"{src}——取消发送，无法取当前位置作保持目标"
    n = arm.n_arm
    sent_at = time.time()
    arm.send_mit_arm(q=q0, dq=np.zeros(n), tau=np.zeros(n),
                     kp=np.full(n, REAL_MIT_KP), kd=np.full(n, REAL_MIT_KD))
    time.sleep(1.0)
    q1, _ = _real_current_q(arm, "arm", after=sent_at)
    drift = None if q1 is None else float(np.max(np.abs(q1 - q0)))
    ok = drift is not None and drift < 0.05
    ev = (f"目标 q=当前实测（{src}）{np.round(q0, 4).tolist()}，kp={REAL_MIT_KP}、"
          f"kd={REAL_MIT_KD}；1s 后 |漂移|max={drift if drift is not None else '—'} rad"
          f"（<0.05 判原地保持）")
    return ("PASS" if ok else "FAIL"), ev


def _rp122(s: _RealSession):
    """122 发送前参数预览：速度指令幅值/方向/持续时间。"""
    arm = s.arm()
    q0, src = _real_current_q(arm, "arm")
    if q0 is None:
        print(f"（预览：{src}）")
        return
    target = _real_safe_target(arm, "arm", q0, REAL_POS_AMP)
    dirs = np.sign(np.asarray(target) - q0).astype(int)
    print(f"  当前 q（{src}）={np.round(q0, 4).tolist()}")
    print(f"  dq 指令={np.round(dirs * REAL_VEL, 2).tolist()} rad/s"
          f"（按软限位裕量选向）持续 {REAL_VEL_DUR}s 后发 dq=0 停止")


def _rt122(s: _RealSession):
    """send_vel 短时低速：±0.1 rad/s 选向 ×0.5s 后 dq=0 停止。"""
    arm = s.arm()
    q0, src = _real_current_q(arm, "arm")
    if q0 is None:
        return "FAIL", f"{src}——取消发送，电机反馈异常"
    n = arm.n_arm
    target = _real_safe_target(arm, "arm", q0, REAL_POS_AMP)
    dq_cmd = np.sign(np.asarray(target) - q0) * REAL_VEL
    sent_at = time.time()
    st = None
    try:
        arm.send_vel_arm(dq_cmd)
        time.sleep(REAL_VEL_DUR)
        st = arm.get_state_arm()
    finally:
        arm.send_vel_arm(np.zeros(n))           # 反馈读取异常也必须发停止命令。
    dq_meas = None
    if _real_state_fresh(arm, "arm", st, after=sent_at) and st.dq is not None:
        candidate = np.asarray(st.dq, float)
        if candidate.shape == (n,) and np.isfinite(candidate).all():
            dq_meas = candidate
    resp = dq_meas is not None and float(np.max(np.abs(dq_meas))) > 0.01
    ev = (f"dq 指令={np.round(dq_cmd, 2).tolist()} rad/s ×{REAL_VEL_DUR}s 后已发 dq=0；"
          f"运动中实测 dq={np.round(dq_meas, 3).tolist() if dq_meas is not None else None}"
          f"（|dq|max>0.01 判有响应）")
    return ("PASS" if resp else "FAIL"), ev


def _rp123(s: _RealSession):
    """123 发送前参数预览：end 位置小幅目标。"""
    arm = s.arm()
    if arm.n_end == 0:
        print("  （end 空族）")
        return
    q0, src = _real_current_q(arm, "end")
    if q0 is None:
        print(f"  （预览：{src}）")
        return
    target = _real_safe_target(arm, "end", q0, REAL_POS_AMP)
    print(f"  end 当前 q（{src}）={np.round(q0, 4).tolist()}")
    print(f"  end 目标 q（±{REAL_POS_AMP} rad 选向）={np.round(target, 4).tolist()}")


def _rt123(s: _RealSession):
    """send_position_end 小幅：end 自实测位置 ±0.05 rad（空族 SKIP）。"""
    arm = s.arm()
    if arm.n_end == 0:
        return "SKIP", "end 族未配置任何电机（n_end=0），空族自动跳过"
    q0, src = _real_current_q(arm, "end")
    if q0 is None:
        return "FAIL", f"{src}——取消发送，无法自实际位置小幅出发"
    target = _real_safe_target(arm, "end", q0, REAL_POS_AMP)
    sent_at = time.time()
    arm.send_position_end(target, np.full(arm.n_end, REAL_VLIM))
    time.sleep(REAL_SETTLE)
    q1, _ = _real_current_q(arm, "end", after=sent_at)
    moved = q1 is not None and float(np.max(np.abs(q1 - q0))) > 0.005
    ev = (f"end 目标（{src} ±{REAL_POS_AMP}）={np.round(target, 4).tolist()}；"
          f"{REAL_SETTLE}s 后实测 q={np.round(q1, 4).tolist() if q1 is not None else None}")
    return ("PASS" if moved else "FAIL"), ev


# ---------------- r3 组（真机：状态机运动，高危——FSM 启动将使能并运动至 home） ----------------
def _rt130(s: _RealSession):
    """FSM 启动与自动迁移：start → DISABLE→HOME→MOTION（home 到位门槛放行）。"""
    arm = s.arm()
    i0 = _mark()
    arm.start_motion_fsm()                        # 幂等（已运行忽略）
    ok_home = _wait_state(arm, MotionState.HOME, timeout=REAL_WAIT_HOME)
    ok_motion = _wait_state(arm, MotionState.MOTION, timeout=REAL_WAIT_MOTION)
    msgs = _since(i0)
    seq = (_hit(msgs, "状态迁移 DISABLE → HOME") and _hit(msgs, "状态迁移 HOME → MOTION")
           and _hit(msgs, "HOME 已到位"))
    ok = ok_home and ok_motion and seq
    ths = [t for t in _joyarm_thread_names()
           if t in ("joyarm-plan", "joyarm-ctrl")]
    ev = (f"start_motion_fsm 后：HOME 迁移={ok_home}，『HOME 已到位』="
          f"{_hit(msgs, 'HOME 已到位')}（真实运动到位耗时计入），MOTION 迁移={ok_motion}；"
          f"终态 current={arm.get_current_motion_state().name}；"
          f"plan/ctrl 线程={ths or '（未清点到，见日志）'}")
    return ("PASS" if ok else "FAIL"), ev


def _rt131(s: _RealSession):
    """目标通道小幅运动：set_target_joint（当前 ±0.05）经 plan/ctrl 管线到位。"""
    arm = s.arm()
    arm.start_motion_fsm()                        # 幂等
    if not _wait_state(arm, MotionState.MOTION, timeout=REAL_WAIT_MOTION):
        return "FAIL", f"未进入 MOTION（current={arm.get_current_motion_state().name}）"
    q0, src = _real_current_q(arm, "arm")
    if q0 is None:
        return "FAIL", f"{src}——取消发送，无法取当前位置作小幅目标"
    target = _real_safe_target(arm, "arm", q0, REAL_POS_AMP)
    tol = 0.01
    if float(np.max(np.abs(target - q0))) <= 2.0 * tol:
        return "SKIP", "软限位裕量不足，无法在既定小幅范围内形成大于两倍容差的目标"
    print(f"  目标通道预览：当前 q（{src}）={np.round(q0, 4).tolist()} → "
          f"目标={np.round(target, 4).tolist()}（t+3.0s，容差 {tol} rad）")
    sent_at = time.time()
    arm.set_target_joint(sent_at + 3.0, target)
    deadline = time.monotonic() + 5.0
    arrived = False
    q1 = None
    while time.monotonic() < deadline:
        q1, _ = _real_current_q(arm, "arm", after=sent_at)
        if (q1 is not None and float(np.max(np.abs(q1 - target))) <= tol
                and float(np.max(np.abs(q1 - q0))) > tol):
            arrived = True
            break
        time.sleep(0.1)
    err = None if q1 is None else float(np.max(np.abs(q1 - target)))
    ev = (f"set_target_joint(t+3.0, ≤{REAL_POS_AMP}) 后 5s 内实际运动并到位={arrived}"
          f"（容差 {tol}、要求发送后新鲜反馈）；实测 q="
          f"{np.round(q1, 4).tolist() if q1 is not None else None}，|q−目标|max={err}")
    return ("PASS" if arrived else "FAIL"), ev


def _rt132(s: _RealSession):
    """保护事件直达：HOLDING→HOLD 原位锁定，恢复 NORMAL→MOTION。"""
    arm = s.arm()
    arm.start_motion_fsm()                        # 幂等
    if not _wait_state(arm, MotionState.MOTION, timeout=REAL_WAIT_MOTION):
        return "FAIL", f"未进入 MOTION（current={arm.get_current_motion_state().name}）"
    arm.set_event(SafetyEvent.HOLDING)
    ok_hold = _wait_state(arm, MotionState.HOLD, timeout=3.0)
    arm.set_event(SafetyEvent.NORMAL)
    ok_back = _wait_state(arm, MotionState.MOTION, timeout=8.0)
    ok = ok_hold and ok_back
    ev = (f"set_event(HOLDING) → HOLD 直达={ok_hold}（原位锁定）；"
          f"set_event(NORMAL) → MOTION 恢复={ok_back}；"
          f"终态 current={arm.get_current_motion_state().name}")
    return ("PASS" if ok else "FAIL"), ev


def _rt133(s: _RealSession):
    """FSM 停止回零：stop（阻塞含回零等待）→ 失能收尾、状态机退出。"""
    arm = s.arm()
    if arm.is_connected_arm is not True:
        return "SKIP", "arm 未连接（stop 需连接态；断连急停走 close）"
    i0 = _mark()
    t0 = time.monotonic()
    e = _exc(arm.stop_motion_fsm)                 # 阻塞：真实回零最长 ~move_wait_timeout
    dt = time.monotonic() - t0
    disabled = arm.is_enabled_arm is False
    stopped = arm.get_current_motion_state() is MotionState.DISABLE
    ok = e is None and stopped and disabled
    ev = (f"stop_motion_fsm 阻塞耗时 {dt:.2f}s（含回零等待，超时上限 "
          f"move_wait_timeout={arm.get_config()['joyarm']['runtime']['move_wait_timeout']}s）；"
          f"收尾后 current={arm.get_current_motion_state().name}、"
          f"is_enabled={arm.is_enabled_arm}；异常={e is None}；"
          f"收尾 info={_hit(_since(i0), 'FSM 已停止并清空缓存')}")
    return ("PASS" if ok else "FAIL"), ev


# ---------------- r4 组（真机：监视器/可视化，中危） ----------------
def _rt140(s: _RealSession):
    """monitor 启停与不误触发：无故障时 event 保持 NORMAL；重复启停幂等。"""
    arm = s.arm()
    i0 = _mark()
    arm.start_monitor()
    arm.start_monitor()                           # 重复启动：忽略
    ign = _hit(_since(i0), "忽略重复启动")
    time.sleep(2.0)
    ev_during = arm.get_event()
    no_fault = arm.check_error_arm() if arm.is_connected_arm is True else None
    arm.stop_monitor()
    e2 = _exc(arm.stop_monitor)                   # 重复停止：无异常
    no_false = (ev_during is SafetyEvent.NORMAL
                or no_fault is False)             # 真有故障码时触发非误报
    ok = ign and e2 is None and no_false
    ev = (f"重复 start 忽略 info={ign}；2s 观察窗 event={ev_during.name}"
          f"（check_error_arm={no_fault}：True=无故障且事件保持 NORMAL=未误触发；"
          f"False=真有故障，触发属正常保护）；重复 stop 异常={e2 is None}")
    return ("PASS" if ok else "FAIL"), ev


def _rt141(s: _RealSession):
    """自定义规则回调：收集型规则（返回 None 不 set_event）被周期调用且收到状态。"""
    arm = s.arm()
    seen: list = []

    def probe(state_arm, state_end, motion, event):
        seen.append((state_arm is not None,
                     motion.name if motion is not None else None))
        return None                               # 无意见：不触发任何事件

    arm.set_monitor_rules(probe)
    arm.start_monitor()
    time.sleep(1.5)
    arm.stop_monitor()
    arm.set_monitor_rules([])
    n_call = len(seen)
    n_state = sum(1 for has, _ in seen if has)
    hz = arm.get_config()["joyarm"]["MotionFSM"].get("monitor_hz", 50.0)
    ok = n_call >= 20 and n_state >= n_call * 0.8
    ev = (f"1.5s 内规则被调 {n_call} 次（monitor_hz={hz}，阈值 ≥20）；"
          f"其中收到非 None 状态 {n_state}/{n_call} 拍（≥80%，连接态应非 None）；"
          f"规则返回 None 全程未改 event={arm.get_event().name}")
    return ("PASS" if ok else "FAIL"), ev


def _rt142(s: _RealSession):
    """可视化面板：真机连接态 /state.json 出实测数据（arm.q 非 null）+ 幂等停止。"""
    import json as _json
    arm = s.arm()
    e = _exc(arm.start_state_viz)
    time.sleep(0.5)                               # worker 10 Hz：至少数拍填充
    urls = _viz_urls()
    ok_json, payload = _fetch(urls[-1] + "state.json") if urls else (False, b"")
    has_q = False
    if ok_json:
        try:
            data = _json.loads(payload)
            g = data.get("arm") or {}
            has_q = g.get("q") is not None
        except Exception:
            has_q = False
    e2 = _exc(arm.stop_state_viz)
    ok = e is None and e2 is None and ok_json and has_q
    ev = (f"start 异常={e is None}；GET /state.json={ok_json}，arm.q="
          f"{'非 null（连接后 backend 快照出数）' if has_q else 'null（未出数）'}；"
          f"stop 异常={e2 is None}；URL={urls[-1] if urls else '（未捕获，见日志）'}")
    return ("PASS" if ok else "FAIL"), ev


# ---------------- r5 组（真机：清错/设零） ----------------
def _rt150(s: _RealSession):
    """清错：读错误码 → clear_error → 再读核对（清后全部 <2）。"""
    arm = s.arm()
    checks = []
    for fam in ("arm", "end"):
        if fam == "end" and arm.n_end == 0:
            continue
        e0 = getattr(arm, f"get_error_{fam}")()
        r = getattr(arm, f"clear_error_{fam}")()
        e1 = getattr(arm, f"get_error_{fam}")()
        ok = (r is True and (e1 is None
                             or all(int(v) < 2 for v in np.asarray(e1).reshape(-1))))
        checks.append((f"{fam} clear_error", ok,
                       f"清前={np.asarray(e0).reshape(-1).tolist() if e0 else None}，"
                       f"ret={r}，清后={np.asarray(e1).reshape(-1).tolist() if e1 else None}"))
    ok = all(c for _, c, _ in checks)
    ev = "；".join(f"{lb}={'OK' if c else 'FAIL'}（{dt}）" for lb, c, dt in checks)
    return ("PASS" if ok else "FAIL"), ev


def _rt151(s: _RealSession):
    """设零（极高危）：每族二次确认后 set_zero 并回读 |q|≈0（容差 0.05 rad）。

    仅显式 --real --step=151 执行（任何组展开不含本项）；任一次非 yes 即跳过
    该族及后续；--yes 不放行、非终端自动跳过。
    """
    arm = s.arm()
    lines: list = []
    worst = "PASS"
    fams = [f for f in ("arm", "end") if getattr(arm, f"n_{f}") > 0]
    aborted = False
    for fam in fams:
        if aborted:
            lines.append(f"{fam}: SKIP（前族二次确认跳过，后续一并跳过）")
            worst = "SKIP" if worst == "PASS" else worst
            continue
        if not sys.stdin.isatty():
            lines.append(f"{fam}: SKIP（非终端环境自动跳过——极高危永不自动执行）")
            worst = "SKIP" if worst == "PASS" else worst
            continue
        ans = input(f"    第二次确认（yes=对 {fam} 族执行设零，其他输入=跳过该族及后续）: ").strip().lower()
        if ans not in ("yes", "y"):
            lines.append(f"{fam}: SKIP（用户在二次确认跳过）")
            worst = "SKIP" if worst == "PASS" else worst
            aborted = True
            continue
        conn = getattr(arm, f"is_connected_{fam}") is True
        if not conn:
            lines.append(f"{fam}: FAIL（未连接——设零需连接态）")
            worst = "FAIL"
            continue
        r = getattr(arm, f"set_zero_{fam}")()
        st = getattr(arm, f"get_state_{fam}")()
        qmax = (None if (st is None or st.q is None)
                else float(np.max(np.abs(np.asarray(st.q, dtype=float)))))
        ok = r == 1 and qmax is not None and qmax < REAL_ZERO_TOL
        lines.append(f"{fam}: {'PASS' if ok else 'FAIL'}（set_zero ret={r}，回读 |q|max="
                     f"{qmax if qmax is not None else '—'} rad（容差 {REAL_ZERO_TOL}））")
        if not ok:
            worst = "FAIL"
    return worst, "；".join(lines) + "【永久改写零位标定，不可逆；回读仅验证当前读数≈0】"


# ============================================================
# 真机测试项注册表（编号沿用两位数十进制分组习惯：级 r0~r5 对应
# 100/110/120/130/140/150 各组；151 极高危仅显式编号执行）
# ============================================================
REAL_ITEMS: list[dict] = []


def _ritem(num, name, group, risk, purpose, needs, fn, preview=None):
    REAL_ITEMS.append(dict(num=num, name=name, group=group, risk=risk, level=group,
                           purpose=purpose, needs=list(needs), fn=fn, preview=preview))


# —— r0 组（连接与硬件只读，低危）——
_ritem(100, "真机构建与自检", "r0", "低",
       "真实 cfg（joyarm_dm.yaml 的 backend_dm）构造 JoyArm：全套自检过、无总线 I/O、n_arm/n_end 与 cfg 一致", [], _rt100)
_ritem(101, "arm/end 连接", "r0", "低",
       "connect 后连接标志 True、default_mode 回填（信息性核对，回读能力缺失不算失败）", [], _rt101)
_ritem(102, "状态读取", "r0", "低",
       "get_state_arm/end 输出实测值（q/dq/tau/温度/error）并校验契约（(n,) 有限、t>0）+ 被动快照新鲜",
       [("connected", "both")], _rt102)
_ritem(103, "模式读取", "r0", "低",
       "get_mode 返回 default_mode 回填值（None=无回读能力，SKIP 非缺陷）",
       [("connected", "both")], _rt103)
_ritem(104, "错误码读取与检查", "r0", "低",
       "get_error 逐关节码（0=失能 1=使能 ≥2=故障）+ check_error 与之一致",
       [("connected", "both")], _rt104)
_ritem(105, "参数读取", "r0", "低",
       "read_param_arm('kp') 返回逐关节 list（键不可用 SKIP 非缺陷）",
       [("connected", "arm")], _rt105)

# —— r1 组（使能/模式/参数写：模式低危；使能/写参高危）——
_ritem(110, "三模式设置", "r1", "低",
       "MIT/POSITION/VELOCITY 循环 设→读回（切 POSITION/VELOCITY 随模式写增益；回读 None 仅记录）",
       [("connected", "arm")], _rt110)
_ritem(111, "arm 使能/失能", "r1", "高",
       "enable(1/回读 True) → disable(1/回读 False)（电机上力）",
       [("connected", "arm")], _rt111)
_ritem(112, "使能/失能幂等", "r1", "高",
       "enable×2、disable×2 均返回 int 不崩溃，终态 False",
       [("connected", "arm")], _rt112)
_ritem(113, "参数写回", "r1", "高",
       "读 kp 原值 → 写回原值 → 读回核对一致（无副作用；键不可用 SKIP）",
       [("connected", "arm")], _rt113)
_ritem(114, "end 使能/失能", "r1", "高",
       "与 111 对称的末端族使能/失能契约（空族 SKIP）",
       [("connected", "end")], _rt114)

# —— r2 组（委托指令小幅运动，高危 + 发送前参数预览 + 二次确认，--yes 不放行）——
_ritem(120, "send_position 小幅", "r2", "高",
       f"自当前实测位置 ±{REAL_POS_AMP} rad（软限位裕量自动选向）、限速 {REAL_VLIM} rad/s，观测实际响应",
       [("connected", "arm"), ("mode", "arm", ControlMode.POSITION), ("enabled", "arm")],
       _rt120, preview=_rp120)
_ritem(121, "send_mit 原地保持", "r2", "高",
       f"以当前实测位置为 MIT 目标原地保持（kp={REAL_MIT_KP}, kd={REAL_MIT_KD} 保守增益），1s 漂移 <0.05 rad",
       [("connected", "arm"), ("mode", "arm", ControlMode.MIT), ("enabled", "arm")],
       _rt121, preview=_rp121)
_ritem(122, "send_vel 短时低速", "r2", "高",
       f"{REAL_VEL} rad/s 按软限位裕量逐关节选向 ×{REAL_VEL_DUR}s 后发 dq=0 停止",
       [("connected", "arm"), ("mode", "arm", ControlMode.VELOCITY), ("enabled", "arm")],
       _rt122, preview=_rp122)
_ritem(123, "end 位置小幅", "r2", "高",
       f"end 自实测位置 ±{REAL_POS_AMP} rad 位置指令（空族 SKIP）",
       [("connected", "end"), ("mode", "end", ControlMode.POSITION), ("enabled", "end")],
       _rt123, preview=_rp123)

# —— r3 组（状态机运动，高危——FSM 启动将使能电机并运动至 home 位形）——
_ritem(130, "FSM 启动与自动迁移", "r3", "高",
       "start_motion_fsm → DISABLE→HOME→MOTION 自动迁移（HOME 已到位日志 + 真实运动耗时计入）",
       [("connected", "arm")], _rt130)
_ritem(131, "目标通道小幅运动", "r3", "高",
       f"set_target_joint(当前 ±{REAL_POS_AMP}) 经 plan/ctrl 管线实际运动并到位（tol=0.01、发送后新鲜反馈）",
       [("connected", "arm")], _rt131)
_ritem(132, "保护事件直达 HOLD", "r3", "高",
       "MOTION 下 set_event(HOLDING) 直达 HOLD 原位锁定，set_event(NORMAL) 恢复 MOTION",
       [("connected", "arm")], _rt132)
_ritem(133, "FSM 停止回零", "r3", "高",
       "stop_motion_fsm（阻塞含回零等待）→ 失能收尾、状态机退出（终态 DISABLE）",
       [("connected", "arm")], _rt133)

# —— r4 组（监视器/可视化，中危：交互回车确认）——
_ritem(140, "monitor 启停与不误触发", "r4", "中",
       "无故障时 2s 观察窗 event 保持 NORMAL（不误触发）；重复启停幂等",
       [("connected", "arm")], _rt140)
_ritem(141, "自定义规则回调", "r4", "中",
       "收集型规则（返回 None 不 set_event）被周期调用（≥20 次/1.5s）且收到非 None 状态",
       [("connected", "arm")], _rt141)
_ritem(142, "可视化面板实测数据", "r4", "中",
       "真机连接态 /state.json 出实测数据（arm.q 非 null）+ 幂等停止",
       [("connected", "arm")], _rt142)

# —— r5 组（清错高危；设零极高危）——
_ritem(150, "清除错误", "r5", "高",
       "读错误码 → clear_error（ret True）→ 再读核对全部 <2（end 对称，空族跳过）",
       [("connected", "both")], _rt150)
_ritem(151, "设置零点", "r5", "极高",
       "【永久改写零位标定，不可逆】逐族（arm/end）：二次确认后 set_zero 并回读 |q|max<0.05 rad；"
       "仅显式 --real --step=151 执行（任何组不含本项；--yes 不放行、非终端自动跳过）",
       [("connected", "both")], _rt151)

REAL_ALL_NUMS = {it["num"] for it in REAL_ITEMS}
_REAL_BY_NUM = {it["num"]: it for it in REAL_ITEMS}

REAL_LEVELS = {
    'r0': '构造、连接与硬件只读',
    'r1': '模式、使能与参数写回',
    'r2': '三模式及末端小幅指令',
    'r3': '状态机运动与保护恢复',
    'r4': '监视器与状态可视化',
    'r5': '故障清除（设零151仅显式单项）',
}


# 组名 → 编号集合（151 极高危排除出 r5/rall，仅显式编号执行——对齐 backend_verify 38 惯例）
REAL_GROUPS: dict[str, list[int]] = {
    "r0": [100, 101, 102, 103, 104, 105],
    "r1": [110, 111, 112, 113, 114],
    "r2": [120, 121, 122, 123],
    "r3": [130, 131, 132, 133],
    "r4": [140, 141, 142],
    "r5": [150],
}
REAL_GROUPS["rall"] = [n for ns in REAL_GROUPS.values() for n in ns]


# ============================================================
# CLI 与执行框架
# ============================================================
class _UsageError(Exception):
    """用法错误（退出码 2）。"""


def _item_order(it: dict) -> tuple:
    """先依赖层级、后稳定编号；编号较小的集成项不能抢在基础 FSM 前执行。"""
    return int(it['level'][1:]), it['num']


def _level_groups(real: bool = False) -> dict:
    if real:
        # 使用既有组表，151 永久标定不进入任何级别/累计选择。
        return {key: list(REAL_GROUPS[key]) for key in REAL_LEVELS}
    return {level: [it['num'] for it in ITEMS if it['level'] == level] for level in LEVELS}


def _normalize_level(value: str, real: bool = False) -> str:
    value = value.strip()
    if value.isdigit():
        value = ('r' if real else 'L') + value
    value = value.lower() if real else value.upper()
    if value not in _level_groups(real):
        raise _UsageError(f'未知层级 {value!r}；可用：{list(_level_groups(real))}')
    return value


def resolve_steps(spec: str, real: bool = False) -> list[int]:
    """解析稳定编号/功能组/级别，去重后始终按低层到高层排序。"""
    groups = REAL_GROUPS if real else GROUPS
    by_num = _REAL_BY_NUM if real else _BY_NUM
    levels = {key.lower(): nums for key, nums in _level_groups(real).items()}
    selected = []
    for token in spec.split(','):
        token = token.strip().lower()
        if not token:
            continue
        if real and token == 'all':
            token = 'rall'
        if token.isdigit():
            num = int(token)
            if num not in by_num:
                raise _UsageError(f'未知步骤编号 {num}；当前模式有效编号：{sorted(by_num)}')
            selected.append(num)
        elif token in levels:
            selected.extend(levels[token])
        elif token in groups:
            selected.extend(groups[token])
        else:
            raise _UsageError(f'未知步骤 {token!r}；组名：{list(groups)}；层级：{list(levels)}')
    if not selected:
        raise _UsageError(f'--step={spec!r} 未解析出任何测试项')
    return sorted(set(selected), key=lambda num: _item_order(by_num[num]))


def _selected_steps(args: argparse.Namespace) -> list[int]:
    levels = _level_groups(args.real)
    if args.level is not None:
        level = _normalize_level(args.level, args.real)
        return resolve_steps(','.join(map(str, levels[level])), args.real)
    if args.through is not None:
        top = _normalize_level(args.through, args.real)
        nums = [num for level, entries in levels.items()
                if int(level[1:]) <= int(top[1:]) for num in entries]
        return resolve_steps(','.join(map(str, nums)), args.real)
    return resolve_steps(args.step or 'all', args.real)


def _run_suite(items: list, execute, keep_going: bool = False) -> list:
    """逐层执行：当前层尽量测完，低层 FAIL 阻止升级；收尾层始终允许执行。"""
    results = []
    failed = set()
    current = None
    for it in sorted(items, key=_item_order):
        level = it['level']
        rank = int(level[1:])
        if current != level:
            current = level
            print(f"\n{'=' * 68}\n开始层级 {level}：{(REAL_LEVELS if level.startswith('r') else LEVELS)[level]}")
        lower_failed = sorted(low for low in failed if int(low[1:]) < rank)
        if lower_failed and not keep_going and it['group'] != 'close':
            verdict, evidence = 'SKIP', f"较低层 {lower_failed} 已失败，停止升级（可用 --keep-going 继续诊断）"
            _print_pre(it, '真机' if level.startswith('r') else '离线')
            _print_post(it, verdict, evidence)
        else:
            try:
                verdict, evidence = execute(it)
                if verdict not in ('PASS', 'FAIL', 'SKIP', 'QUIT'):
                    raise ValueError(f'非法验收结果 {verdict!r}')
            except Exception as e:
                verdict, evidence = 'FAIL', f'执行框架异常 {type(e).__name__}: {e}'
                _print_post(it, verdict, evidence)
        results.append((it, verdict, evidence))
        if verdict == 'FAIL':
            failed.add(level)
        if verdict == 'QUIT':
            break
    return results


# ---------------- 逐项 pre/post 统一打印模板（双模式共用） ----------------
def _print_pre(it: dict, mode_label: str) -> None:
    """执行前打印：本项测试内容（名称概要）+ 测试目的（双模式统一模板）。"""
    lvl = it.get("level") or _GROUP_LEVEL.get(it["group"], "-")
    print()
    print(f"[{it['num']:02d}] {it['name']} | 模式={mode_label} | 组={it['group']}"
          f" | 级={lvl} | 风险={it['risk']}")
    print(f"内容与目的：{it['purpose']}")


def _print_post(it: dict, verdict: str, evidence: str) -> None:
    """执行后打印：结果数据（实测数值/断言证据）+ 结论（双模式统一模板）。"""
    print(f"[{it['num']:02d}] {verdict} —— 结果与证据：{evidence}")


def run_item(it: dict) -> tuple[str, str]:
    """离线单项执行：pre 打印 → 执行 → post 打印（无交互）。"""
    _print_pre(it, "离线")
    start = len(_CASE_BACKENDS)
    try:
        try:
            verdict, evidence = it['fn']()
        except Exception as e:
            verdict, evidence = 'FAIL', f'未预期异常 {type(e).__name__}: {e}'
    finally:
        # 构造失败也可能已启动后台刷新；统一回收本项新建的 Dummy，防串项污染。
        _cleanup_backends(start)
        CFG_SOURCE.next = None
    _print_post(it, verdict, evidence)
    return verdict, evidence


def run_real_item(it: dict, s: _RealSession) -> tuple[str, str]:
    """真机单项执行：pre → 前置说明 → 风险确认 → 前置补齐 →（运动项）预览+二次
    确认 → 执行 → post（对齐 backend_verify 执行框架）。"""
    _print_pre(it, "真机")
    level = it["risk"]
    if it["needs"]:
        plans = _plan_real_prefill(s, it["needs"])
        print(f"前置：{_describe_real_needs(it['needs'])}"
              + (f"；将自动补齐：{'、'.join(d for d, _ in plans)}" if plans else "（已就绪）"))
        for _, lv in plans:            # 前置风险并入该项确认等级
            if _RISK_ORDER[lv] > _RISK_ORDER[level]:
                level = lv
    if level != "低":
        _real_banner_once()
    ans = _confirm(level, s.args)
    if ans == "quit":
        _print_post(it, "SKIP", "用户退出（q）——后续项不再执行")
        return "QUIT", "用户退出（q）"
    if ans == "skip":
        if not sys.stdin.isatty() and level in ("高", "极高"):
            ev = "非终端环境自动跳过（高危永不自动执行；交互终端下需输入 yes）"
        else:
            ev = f"用户跳过（风险={level}）"
        _print_post(it, "SKIP", ev)
        return "SKIP", ev
    status, actions, reason = _real_prefill(s, it["needs"])
    if status == "fail":
        ev = f"前置补齐失败：{'；'.join(actions) or '（无）'}——{reason}"
        _print_post(it, "FAIL", ev)
        return "FAIL", ev
    if status == "skip":
        ev = f"前置不可用：{reason}"
        _print_post(it, "SKIP", ev)
        return "SKIP", ev
    if actions:
        print(f"（已自动补齐前置：{'；'.join(actions)}）")
    if it.get("preview"):
        # 运动参数预览 + 二次确认（--yes 不放行；非终端自动 SKIP，不盲发）
        try:
            it["preview"](s)
        except Exception as e:  # noqa: BLE001（验收脚本须捕获一切）
            print(f"（预览失败：{type(e).__name__}: {e}）")
        if not sys.stdin.isatty():
            ans2 = "skip"
        else:
            ans2 = input("⚠ 以上为即将下发的运动参数，输入 yes 执行发送（回车/s=跳过  q=退出）: ").strip().lower()
        if ans2 in ("q", "quit", "退出"):
            _print_post(it, "SKIP", "用户在预览后退出（q）——后续项不再执行")
            return "QUIT", "用户在预览后退出（q）"
        if ans2 not in ("yes", "y"):
            _print_post(it, "SKIP", "用户在预览后跳过发送（运动参数未下发）")
            return "SKIP", "用户在预览后跳过发送"
    try:
        verdict, evidence = it["fn"](s)
    except Exception as e:  # noqa: BLE001（验收脚本须捕获一切）
        verdict, evidence = "FAIL", f"未预期异常 {type(e).__name__}: {e}"
    _print_post(it, verdict, evidence)
    return verdict, evidence


def print_list(real: bool = False, nums=None) -> None:
    by_num = _REAL_BY_NUM if real else _BY_NUM
    entries = [by_num[num] for num in nums] if nums is not None else list(by_num.values())
    print(f"JoyArm {'真机' if real else '离线'}分层验收清单（由底到高；稳定编号/功能组/层级可独立筛选）")
    current = None
    for it in sorted(entries, key=_item_order):
        level = it['level']
        if level != current:
            current = level
            print(f"\n{level}｜{(REAL_LEVELS if level.startswith('r') else LEVELS)[level]}")
        tag = '｜仅显式 --real --step=151 可执行' if it['num'] == 151 else ''
        print(f"  [{it['num']:02d}] {it['name']}｜组={it['group']}｜风险={it['risk']}{tag}")
        print(f"       目的：{it['purpose']}")
        if real:
            print(f"       前置：{_describe_real_needs(it['needs']) or '（无）'}")
    print(f"\n共 {len(entries)} 项；支持 --level 单级、--through 从底层累计、--step 单项/功能组。")
    print('默认低层 FAIL 后停止升级，当前层测完，已选收尾项仍执行；--keep-going 可继续诊断。')
    print('PASS=契约满足；FAIL=契约违反；SKIP=未验证（环境/能力/依赖/用户跳过），不能当作通过。')
    if real:
        print('高危/极高危仍需交互确认，--yes 不放行；设零151不属于任何层级或累计选择。')


EPILOG = """\
在 joyarm_code/ 目录运行（激活 .venv 或使用 uv run）：
  python joyarm_core/joyarm/joyarm_verify.py --list             # 按层查看全部离线项
  python joyarm_core/joyarm/joyarm_verify.py                    # 全部，由底到高
  python joyarm_core/joyarm/joyarm_verify.py --level L3          # 仅 FSM 基础与自愈
  python joyarm_core/joyarm/joyarm_verify.py --through L3        # L0→L3 累计验收
  python joyarm_core/joyarm/joyarm_verify.py --step 68,69,74      # 定位指定项目
  python joyarm_core/joyarm/joyarm_verify.py --step fsm          # 功能组（可能跨层）
  python joyarm_core/joyarm/joyarm_verify.py --keep-going        # 低层失败后继续诊断
  python joyarm_core/joyarm/joyarm_verify.py --real --list       # 仅列真机项，不连接
  python joyarm_core/joyarm/joyarm_verify.py --real --through r2 # 真机由 r0 验到 r2
  python joyarm_core/joyarm/joyarm_verify.py --real --step 151   # 设零：仅显式单项

离线 L0基础/构造→L1硬件委托→L2算法/目标→L3状态机→L4安全集成→L5可视化→L6收尾。
每项打印目的/结果/证据，每层汇总；原有单项编号保持兼容。默认同层继续、低层
FAIL 后跳过更高层，已选收尾项仍执行；各项自带 finally 收尾与注入恢复，不依赖前项残留。
--real 才连接真实硬件，保留逐项风险确认；高危/极高危非终端自动跳过。
退出码：任一 FAIL=1；用法错误=2；其余=0（SKIP 为未验证，不代表通过）。
SUMMARY 行与 LEVEL/GROUP 统计可用于自动化解析。
"""


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='joyarm_verify.py',
        description='JoyArm 分层逐项验收：缺省离线 L0→L6；--real 真机 r0→r5，按层递进并保留风险确认。',
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    select = parser.add_mutually_exclusive_group()
    select.add_argument('--step', help='稳定编号、功能组或层级，逗号混选；缺省 all')
    select.add_argument('--level', help='只验证某一级，如 L3；真机用 r2')
    select.add_argument('--through', help='从最低层累计验到此级，如 L3；真机用 r2')
    parser.add_argument('--keep-going', action='store_true', help='低层失败后仍继续更高层，用于诊断')
    parser.add_argument('--list', action='store_true', help='仅列出所选项目，不执行、不连接')
    parser.add_argument('--real', action='store_true', help='真机模式，真实配置与连接，逐项风险确认')
    parser.add_argument('--yes', action='store_true', help='真机仅跳过中危回车；高危/极高危仍需交互确认')
    return parser.parse_args(argv)


def _print_summary(results: list, group_order: tuple, mode_label: str) -> int:
    """结尾汇总：SUMMARY 行（机器可解析）+ 各组统计 + 失败明细；返回退出码。"""
    n_pass = sum(1 for _, v, _ in results if v == "PASS")
    n_fail = sum(1 for _, v, _ in results if v == "FAIL")
    n_skip = sum(1 for _, v, _ in results if v != "FAIL" and v != "PASS")  # SKIP/QUIT
    print()
    print("=" * 68)
    print(f"SUMMARY: total={len(results)} pass={n_pass} fail={n_fail} skip={n_skip}")
    for level in sorted({it['level'] for it, _, _ in results}, key=lambda value: int(value[1:])):
        sub = [verdict for it, verdict, _ in results if it['level'] == level]
        print(f"LEVEL {level}: total={len(sub)} pass={sub.count('PASS')} fail={sub.count('FAIL')} "
              f"skip={sum(verdict not in ('PASS', 'FAIL') for verdict in sub)}")
    for gname in group_order:
        sub = [(it, v) for it, v, _ in results if it["group"] == gname]
        if not sub:
            continue
        gp = sum(1 for _, v in sub if v == "PASS")
        gf = sum(1 for _, v in sub if v == "FAIL")
        gs = sum(1 for _, v in sub if v not in ("PASS", "FAIL"))
        print(f"GROUP {gname}: total={len(sub)} pass={gp} fail={gf} skip={gs}")
    fails = [(it, ev) for it, v, ev in results if v == "FAIL"]
    skips = [(it, v) for it, v, _ in results if v not in ("PASS", "FAIL")]
    if not fails:
        concl = f"结论：无失败项，JoyArm 层符合本次{mode_label}验收全部已测契约。"
        if skips:
            concl += f"共 {n_skip} 项未验证（环境/能力/依赖或用户跳过，不能视为通过，详见证据）。"
    else:
        print(f"结论：{n_fail} 项失败——")
        for it, ev in fails:
            print(f"  FAIL [{it['num']:02d}] {it['name']}：{_trunc(ev)}")
        concl = f"（详见上方 {n_fail} 条 FAIL 证据；其余 pass={n_pass}、skip={n_skip}）"
    print(concl)
    return 0 if n_fail == 0 else 1


def _run_real(nums: list, args: argparse.Namespace) -> int:
    """真机模式执行路径：共享会话（惰性构造真 JoyArm）→ 逐项分级确认执行 → 汇总。"""
    s = _RealSession(args)
    print("=" * 68)
    print(f"JoyArm 层真机分级验收：型号 {MODEL}（真实 cfg backend_dm 直连，无 Dummy 注入）")
    print(f"交互终端：{'是' if sys.stdin.isatty() else '否'}；--yes={args.yes}；"
          f"高危策略：{'交互 yes（--yes 不放行）' if sys.stdin.isatty() else '非终端自动 SKIP'}")
    print("危险分级：低=直执；中=回车确认（--yes 或非终端直执）；高=交互 yes；"
          "极高（151）=每族二次确认、仅显式编号执行")
    print(f"待执行 {len(nums)} 项：{[f'{n:02d}' for n in nums]}")
    try:
        results = _run_suite([_REAL_BY_NUM[num] for num in nums],
                             lambda it: run_real_item(it, s), keep_going=args.keep_going)
    finally:
        s.close()
    return _print_summary(results, ("r0", "r1", "r2", "r3", "r4", "r5"), "真机")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:  # stdout 行缓冲：与 stderr 日志正确交错（管道/重定向场景）
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        nums = _selected_steps(args)
    except _UsageError as e:
        print(f'用法错误：{e}', file=sys.stderr)
        return 2
    if args.list:
        # 完整真机清单仍展示151供发现，但它不进入默认/层级执行集合。
        list_nums = None if args.real and not any((args.step, args.level, args.through)) else nums
        print_list(real=args.real, nums=list_nums)
        return 0
    if args.real:
        return _run_real(nums, args)
    # ---- 离线模式（缺省）：Dummy 注入全自动 ----
    # 基础 cfg（注入源）+ 注入装载（finally 恢复，不触碰仓库文件）
    CFG_SOURCE.base = _tweak_cfg(load_base_cfg())
    install_injection()
    print("=" * 68)
    print(f"JoyArm 层离线分级验收：型号 {MODEL}（DummyBackend 注入 {DUMMY_KEY}）")
    print("配置：真实 joyarm_dm.yaml（backend.name 临时指向 Dummy；机器人资产与 cfg 自检用真实文件）")
    print(f"交互终端：{'是' if sys.stdin.isatty() else '否'}（离线用例全部无交互）；"
          "FSM/monitor/viz 用例逐项收尾线程；真机验收走 --real")
    print(f"待执行 {len(nums)} 项：{[f'{n:02d}' for n in nums]}")
    try:
        results = _run_suite([_BY_NUM[num] for num in nums], run_item, keep_going=args.keep_going)
    finally:
        restore_injection()
    return _print_summary(results, tuple(key for key in GROUPS if key != 'all'), '离线')


if __name__ == "__main__":
    sys.exit(main())
