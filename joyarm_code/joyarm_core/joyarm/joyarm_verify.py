"""
==============================================================================
JoyArm 层离线分级验收脚本  joyarm_core/joyarm/joyarm_verify.py
==============================================================================
【功能概要】
    对 JoyArm 整机层（joyarm/joyarm.py 的组合根：工厂/自检/只读面/委托面/求解器
    域/数学门面/MotionFSM/safe_monitor/可视化/销毁路径）做**全离线**分组分项验收
    （不连真机；真机侧验收走 backend/backend_verify.py）。组织范式与
    backend_verify.py 一致：两位数十进制分组（组号×10+组内序号）、_item 注册表
    （组/风险/用途）、PASS/SKIP/FAIL 三分判定、结尾机器可解析 SUMMARY 行：
      00 组  工厂与 listArm：未知名异常含可用列表、listArm 三条件（临时目录
             变量控制）、模板排除、Dummy 注入下的工厂成功路径；
      01 组  初始化自检：坏 cfg 变体逐条异常（缺顶层键/坏注册名/缺 plan_hz/
             非整值/home 越软限位/软限位越硬限位/限位与 URDF 不符）；
      02 组  只读属性与防篡改：writeable=False 原地写抛异常、get_config 深拷贝
             （外部篡改不反哺内部）、只读 @property 零拷贝返本体；
      03 组  委托面与互斥：DummyBackend 内核透传核对、FSM 运行中外部 arm 侧
             指令空操作 + warn、ctrl 线程豁免；
      04 组  求解器域：get/set_activated/get_<域>_solvers、坏名 warn 保持原激活、
             traj 先 plan 后换指针 + plan_hz 联动、ctrl 切换 MODE 联动；
      05 组  数学门面：显式传参离线计算 vs pinocchio 原生 API 逐项对照
             （fkine/ikine/ikine_all/jac/jac_pinv/idyn/mass/coriolis/gravity 9 方法）；
      06 组  MotionFSM（DummyBackend 推演）：start 前置 RuntimeError、
             DISABLE→HOME→MOTION 自动迁移、home 到位门槛、保护事件直达、
             stop 超时强推；
      07 组  safe_monitor：硬规则 error≥2→DAMPING、断连/连续 None→DISABLE、
             自定义规则绑定（state_arm 允许 None）与同拍仲裁（硬优先 +
             同组取危险最高）；
      08 组  可视化：start/stop_state_viz 离线降级不炸、meshcat 缺失 warn、
             HTTP /state.json 轮询端点、重复启停幂等与线程退出；
      09 组  销毁路径：close 幂等、with __exit__ 收尾序列、__del__ 冒烟
             （__del__ 不做执行时序承诺，仅验证存在与不炸）。
    DummyBackend 注入：脚本内自包含实现最小 DummyBackend(Backend)（29 个抽象
    内核全部脚本化：calls 记录 / modes·jstates 可编程 / 寄存器模拟 / 位置指令
    跟随开关），经「临时改 cfg 的 backend.name + backend.REGISTRY 原地加键
    （joyarm.py import 时值引用捕获各域 REGISTRY，替换模块属性无效）+ 临时换
    load_config 返回注入 cfg」注入（用完恢复，不触碰仓库文件）。
    交互性：全部测试项无交互（非终端可跑）；FSM/监视器/可视化线程类用例在
    每项内部 finally 收尾（stop/join），不悬挂线程、不影响退出码。

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.sh | iex

    # 2. 创建虚拟环境并同步依赖（Python 须满足 requires-python >=3.10,<3.12）
    cd <仓库>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（在 joyarm_code/ 目录下；全离线、无交互；系统 python 超
    #    requires-python 上界，一律经 uv 运行）
    python joyarm_core/joyarm/joyarm_verify.py --help        # 完整用法
    python joyarm_core/joyarm/joyarm_verify.py --list        # 全部测试项清单
    python joyarm_core/joyarm/joyarm_verify.py               # 全部（缺省 all）
    python joyarm_core/joyarm/joyarm_verify.py --step=fsm    # 单组
    python joyarm_core/joyarm/joyarm_verify.py --step=61,71  # 单项混选
    python joyarm_core/joyarm/joyarm_verify.py --step=math,monitor  # 组名混选

【命令行参数】
    --step    数字=单项编号（如 61）；组名（factory/init/readonly/delegate/
              solvers/math/fsm/monitor/viz/close/all）；逗号分隔混选；缺省 all
    --list    仅列出全部测试项（编号/组/风险/用途），不执行
    退出码：全 PASS/SKIP=0；任一 FAIL=1；用法错误=2（便于 agent 判定）
    结尾输出 SUMMARY: total=… pass=… fail=… skip=…（机器可解析）+ 各组统计行
==============================================================================
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

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
from joyarm_core.utils.types import ControlMode, MotionState, Pose, SafetyEvent

__all__ = ["main"]

MODEL = "joyarm_dm"                      # 被测型号（真实 config + robot_model 资产）
DUMMY_KEY = "backend_verify_dummy"       # DummyBackend 临时注册名（注入用，用完恢复）
VERIFY_CTRL = "verify_test_controller"   # ctrl 切换 MODE 联动验证用临时控制器注册名

# FSM / 事件类用例的状态等待上限（秒；fsm_hz=50，正常迁移 ≤ 数拍）
_WAIT_STATE = 3.0
# monitor 类用例的事件等待上限（秒；monitor_hz=50）
_WAIT_EVENT = 2.0


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
        self._hit("send_mit_arm", i)

    def _send_joint_mit_end(self, i, tau, q, dq, kp, kd):
        self._rec("send_joint_mit_end", i, tau, q, dq, kp, kd)
        self._hit("send_mit_end", i)

    def _send_joint_position_arm(self, i, q, vlim, flim):
        self._rec("send_joint_position_arm", i, q, vlim, flim)
        self._hit("send_position_arm", i)
        if self.follow:                   # 理想电机：一步到位
            self.jstates["arm"][i]["q"] = float(q)
            self.jstates["arm"][i]["dq"] = 0.0

    def _send_joint_position_end(self, i, q, vlim, flim):
        self._rec("send_joint_position_end", i, q, vlim, flim)
        self._hit("send_position_end", i)

    def _send_joint_vel_arm(self, i, dq):
        self._rec("send_joint_vel_arm", i, dq)
        self._hit("send_vel_arm", i)

    def _send_joint_vel_end(self, i, dq):
        self._rec("send_joint_vel_end", i, dq)
        self._hit("send_vel_end", i)

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


# ctrl 切换 MODE 联动验证用临时控制器（MODE=VELOCITY，与默认 POSITION 控制器可区分）
class _VerifyTestController(Controller):
    """仅验收用：切换后经 MODE 联动把电机模式置 VELOCITY 的哑控制器。"""

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
    while _PATCHES:
        _PATCHES.pop().restore()


def _tweak_cfg(cfg: dict) -> dict:
    """注入 cfg 的验收定制（不改变被测行为语义，只让判定可区分 / 时长可控）：

    - backend.name → DUMMY_KEY（Dummy 注入点）；
    - runtime.move_wait_timeout → 1.0 s（默认 10 s，stop 超时强推项时长可控）；
    - traj 两规划器 plan_hz 拉开（hold 10 / cubic 20，切换联动可判）；
    - control 追加 verify_test_controller（ctrl_hz=100，MODE 联动切换可判）。"""
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
    """FSM/monitor 项统一收尾：q 置 home + follow 开（回零路径最短）→ stop。"""
    try:
        d.follow = True
        d.set_q("arm", arm.home_arm)
        arm.stop_motion_fsm()
    except Exception:
        pass
    try:
        arm.stop_monitor()
    except Exception:
        pass
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
    """未知名：工厂与 get_backend 的异常消息均含可用列表。"""
    from joyarm_core.joyarm import _available_models
    models = _available_models()
    e1 = _exc(joyArm, "no_such_model_xyz")
    e2 = _exc(get_backend, "no_such_backend")
    ok = (isinstance(e1, ValueError) and "可用型号" in str(e1)
          and all(m in str(e1) for m in models)
          and isinstance(e2, ValueError) and "可用" in str(e2))
    ev = (f"joyArm('no_such_model_xyz') → {type(e1).__name__}，消息含可用型号 {models}="
          f"{all(m in str(e1) for m in models)}；get_backend('no_such_backend') → "
          f"{type(e2).__name__}，消息含可用注册名={('可用' in str(e2))}；"
          f"详情：{_trunc(e1, 120)}")
    return ("PASS" if ok else "FAIL"), ev


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
    "fkine": ("get_fkine_solvers", "set_activated_fkine_solver",
              "get_activated_fkine_solver", "fkine"),
    "ikine": ("get_ikine_solvers", "set_activated_ikine_solver",
              "get_activated_ikine_solver", "ikine"),
    "jac": ("get_jac_solvers", "set_activated_jac_solver",
            "get_activated_jac_solver", "jacobian"),
    "dyn": ("get_dyn_solvers", "set_activated_dyn_solver",
            "get_activated_dyn_solver", "dynamics"),
    "traj": ("get_traj_solvers", "set_activated_traj_solver",
             "get_activated_traj_solver", "traj"),
    "ctrl": ("get_ctrl_solvers", "set_activated_ctrl_solver",
             "get_activated_ctrl_solver", "control"),
}


def _it40():
    """六域查询/切换：名单=cfg、默认激活=cfg 首位、切换后激活名与 hz/MODE 联动更新。"""
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
        arm.set_activated_traj_solver("single_joint_hold_traj_planner")
        hz_hold = arm.plan_hz
        checks.append(("traj plan_hz 联动", hz_cubic == 20.0 and hz_hold == 10.0,
                       f"cubic={hz_cubic}→hold={hz_hold}（期望 20→10）"))
        # ctrl 切换 MODE 联动（直连 _backend.set_mode_arm，不经公开 set_mode）
        arm.connect_arm()
        i0 = len(d.calls)
        r = arm.set_activated_ctrl_solver(VERIFY_CTRL)
        got = d.calls_since(i0, "set_joint_mode_arm")
        checks.append(("ctrl 切换 MODE 联动",
                       r is True and arm.get_activated_ctrl_solver() == VERIFY_CTRL
                       and len(got) == arm.n_arm
                       and all(c[2] is ControlMode.VELOCITY for c in got)
                       and arm.get_mode_arm() is ControlMode.VELOCITY
                       and arm.ctrl_hz == 100.0,
                       f"set_joint_mode_arm×{len(got)}→VELOCITY，ctrl_hz={arm.ctrl_hz}"))
        r = arm.set_activated_ctrl_solver("joint_position_controller")
        checks.append(("ctrl 切回默认",
                       r is True and arm.ctrl_hz == 200.0
                       and arm.get_mode_arm() is ControlMode.POSITION,
                       f"ctrl_hz={arm.ctrl_hz}"))
        ok = all(c for _, c, _ in checks)
        ev = "；".join(f"{lb}={'OK' if c else 'FAIL'}（{dt}）" for lb, c, dt in checks)
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.close()


def _it41():
    """坏名 warn 保持原激活：六域 set_activated_*('no_such_solver') 均 False + warn。"""
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
        r = arm.set_activated_traj_solver("single_joint_cubic_traj_planner")
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
        r2 = arm.set_activated_traj_solver("single_joint_cubic_traj_planner")
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
        arm.set_activated_ikine_solver("analytic_ikine_solver")
        r_all = arm.solve_ikine_all(pose)
        errs = []
        if r_all.q is not None and np.asarray(r_all.q).size:
            for q_k in np.atleast_2d(r_all.q):
                pk = arm.solve_fkine(q_k)
                errs.append(max(float(np.linalg.norm(pk.position - pose.position)),
                                quat_err(pk.orientation, pose.orientation)))
        arm.set_activated_ikine_solver("pin_ikine_solver")
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
        ok_d = _wait_state(arm, MotionState.DAMPING)
        time.sleep(0.15)
        got = d.calls_since(i0, "send_joint_mit_arm")
        kd_cfg = float(np.asarray(
            arm.get_config()["joyarm"]["MotionFSM"]["mit_kd_damping"]).reshape(-1)[0])
        checks.append(("DAMPING→直达 + MIT 纯阻尼（kp=0, kd=cfg）",
                       ok_d and bool(got)
                       and all(abs(c[5]) < 1e-12 and abs(c[6] - kd_cfg) < 1e-9 for c in got),
                       f"{len(got)} 帧 kp=0, kd={kd_cfg}"))
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


# ---------------- 07 组（safe_monitor） ----------------
def _it70():
    """硬规则：error≥2 → DAMPING（事件与 FSM 迁移双通路）。"""
    arm, d = make_arm()
    try:
        d.follow = False
        d.set_q("arm", arm.home_arm)
        arm.connect_arm()
        arm.start_motion_fsm()
        _wait_state(arm, MotionState.HOME)
        arm.start_monitor()
        time.sleep(0.1)
        i0 = len(d.calls)
        d.jstates["arm"][0]["error"] = 2      # 单关节故障码
        ok_ev = _wait_event(arm, SafetyEvent.DAMPING)
        ok_st = _wait_state(arm, MotionState.DAMPING)
        time.sleep(0.15)                       # 等数拍维持指令（50 Hz）落地再计数
        got = d.calls_since(i0, "send_joint_mit_arm")
        d.jstates["arm"][0]["error"] = 1
        ok = ok_ev and ok_st and bool(got)
        ev = (f"error[0]=2 后：get_event()={arm.get_event().name}，"
              f"current={arm.get_current_motion_state().name}，"
              f"DAMPING 维持 MIT 帧 {len(got)} 个（>0）")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        d.jstates["arm"][0]["error"] = 1
        _settle_fsm(arm, d)
        arm.close()


def _it71():
    """断连与连续 None → DISABLE；state_arm=None 仍调自定义规则；断连收尾走 close。"""
    # ---- a) 断连 → DISABLE（事件通路 + FSM 迁移被 enter 未连接门禁拦（§6.1 规格：
    #      断连下无法迁移，急停走销毁路径 close()，见 stop_motion_fsm docstring）----
    arm, d = make_arm()
    try:
        d.follow = False
        d.set_q("arm", arm.home_arm)
        arm.connect_arm()
        arm.start_motion_fsm()
        _wait_state(arm, MotionState.HOME)
        arm.start_monitor()
        time.sleep(0.1)
        i0 = _mark()
        d.disconnect_arm()
        ok_a = _wait_event(arm, SafetyEvent.DISABLE)
        time.sleep(0.15)                       # FSM 尝试迁移：enter 撞未连接门禁 → 重试 warn
        retry_warn = _hit(_since(i0), "单步", "失败")
        stuck_state = arm.get_current_motion_state().name   # 规格预期：无法迁移，保持原态
        ev_a = (f"dummy 断连后 event={arm.get_event().name}，current 保持 {stuck_state}"
                f"（§6.1：enter 未连接抛 RuntimeError，FSM 每拍重试 warn={retry_warn}；"
                f"断连急停走销毁路径 close()）")
        arm.stop_monitor()
        e_close = _exc(arm.close)
        ok_a = ok_a and retry_warn and e_close is None
    finally:
        arm.close()
    # ---- b) 连续 None → DISABLE（整族读失败注入） ----
    arm, d = make_arm()
    try:
        d.follow = False
        d.set_q("arm", arm.home_arm)
        arm.connect_arm()
        arm.start_motion_fsm()
        _wait_state(arm, MotionState.HOME)
        seen_none = []

        def probe(sa, se, mo, ev):
            seen_none.append(sa is None)
            return None

        arm.set_monitor_rules(probe)
        arm.start_monitor()
        time.sleep(0.1)
        i0 = _mark()
        d.fail_at["read_joint_state_arm"] = None       # 整族读失败 → get_state_arm None
        ok_b = _wait_event(arm, SafetyEvent.DISABLE)
        st_b = _wait_state(arm, MotionState.DISABLE)
        ctrl_skip = _hit(_since(i0), "跳过本拍 step")
        d.fail_at.pop("read_joint_state_arm", None)
        ok = ok_a and ok_b and st_b and any(seen_none)
        ev = (f"断连：{ev_a}；读状态整族失败（连续 None>0.1s）：event={arm.get_event().name}，"
              f"current={arm.get_current_motion_state().name}；自定义规则收到 "
              f"state_arm=None（读取失败仍调用）={any(seen_none)}"
              f"（{sum(seen_none)}/{len(seen_none)} 拍为 None）；"
              f"ctrl 线程跳拍 warn={ctrl_skip}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        d.fail_at.pop("read_joint_state_arm", None)
        _settle_fsm(arm, d)
        arm.close()


def _it72():
    """自定义规则绑定 + 仲裁：同组取危险最高、硬规则优先于自定义、异常/非法返回忽略。"""
    arm, d = make_arm()
    try:
        d.follow = False
        d.set_q("arm", arm.neutral_arm)     # 非 home：门槛拦住 MOTION，驻留 HOME
        arm.connect_arm()
        arm.start_motion_fsm()
        _wait_state(arm, MotionState.HOME)
        arm.start_monitor()
        time.sleep(0.1)
        checks = []

        # 1) 同组多命中取危险最高：HOLDING(1) + DAMPING(4) → DAMPING
        arm.set_event(SafetyEvent.NORMAL)
        arm.set_monitor_rules([lambda sa, se, mo, ev: SafetyEvent.HOLDING,
                               lambda sa, se, mo, ev: SafetyEvent.DAMPING])
        ok1 = _wait_event(arm, SafetyEvent.DAMPING)
        checks.append(("同组多命中取危险最高（HOLDING+DAMPING→DAMPING）", ok1))

        # 2) 硬规则优先：硬 DAMPING(4) 压过自定义 DISABLE(5)（custom 组不参与）
        arm.set_monitor_rules([lambda sa, se, mo, ev: SafetyEvent.DISABLE])
        arm.set_event(SafetyEvent.NORMAL)
        d.jstates["arm"][0]["error"] = 2
        ok2 = _wait_event(arm, SafetyEvent.DAMPING)
        never_dis = True
        t_end = time.monotonic() + 0.2
        while time.monotonic() < t_end:      # 窗口内自定义 DISABLE 持续返回，事件须恒非 DISABLE
            if arm.get_event() is SafetyEvent.DISABLE:
                never_dis = False
            time.sleep(0.02)
        d.jstates["arm"][0]["error"] = 1
        arm.set_monitor_rules([])
        checks.append(("硬规则优先（硬 DAMPING 压过自定义 DISABLE）", ok2 and never_dis))

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
        ok = all(c for _, c in checks)
        ev = "；".join(f"{lb}={'OK' if c else 'FAIL'}" for lb, c in checks)
        return ("PASS" if ok else "FAIL"), ev
    finally:
        d.jstates["arm"][0]["error"] = 1
        arm.stop_monitor()
        _settle_fsm(arm, d)
        arm.close()


# ---------------- 08 组（可视化） ----------------
def _viz_urls() -> list[str]:
    """从已捕获日志解析监控页 URL（http://127.0.0.1:<port>/，去重保序）。"""
    import re as _re
    urls: list[str] = []
    for m in _cap.msgs:
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
        ev = (f"start 异常={e is None}；meshcat 降级 warn={degraded}（未装 meshcat 走降级）；"
              f"监控页 URL={urls[:1]}；GET /（页面含标题）={ok_page and b'JoyArm' in page}；"
              f"GET /state.json（含 model/fsm/event 键）={has_keys}；"
              f"stop 后端口关闭={gone}；stop 异常={e2 is None}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.stop_state_viz()
        arm.close()


def _it81():
    """meshcat 缺失 warn：环境未装 meshcat 时必现降级 warn；已装则 SKIP。"""
    import importlib.util
    spec = importlib.util.find_spec("meshcat")
    if spec is not None:
        return "SKIP", (f"环境已安装 meshcat（{spec.origin}），缺包降级路径无法离线触发；"
                        f"不炸与收尾已由 80 组覆盖（初始化成功/异常两分支均降级或就绪）")
    arm, _ = make_arm()
    try:
        i0 = _mark()
        arm.start_state_viz()
        time.sleep(0.15)
        arm.stop_state_viz()
        msgs = _since(i0)
        ok = _hit(msgs, "未安装 meshcat") and _hit(msgs, "降级为仅 HTML 状态面板")
        ev = (f"未安装 meshcat → warn 降级："
              f"{_trunc(next((m for m in msgs if 'meshcat' in m), ''), 120)}")
        return ("PASS" if ok else "FAIL"), ev
    finally:
        arm.stop_state_viz()
        arm.close()


def _it82():
    """重复启停幂等 + 线程退出：start×2 忽略、stop×2 无异常、无 viz 线程残留。"""
    arm, _ = make_arm()
    try:
        i0 = _mark()
        arm.start_state_viz()
        time.sleep(0.1)
        arm.start_state_viz()                  # 重复启动：忽略 + info
        ign = _hit(_since(i0), "忽略重复启动")
        e1 = _exc(arm.stop_state_viz)
        e2 = _exc(arm.stop_state_viz)          # 重复停止：无异常
        leftovers = [t for t in _joyarm_thread_names() if "viz" in t]
        ok = ign and e1 is None and e2 is None and not leftovers
        ev = (f"重复 start 忽略 info={ign}；stop×2 异常={e1 is None and e2 is None}；"
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
# 测试项注册表（编号=组号×10+组内序号；全离线低危）
# ============================================================
ITEMS: list[dict] = []


def _item(num, name, group, risk, purpose, fn):
    ITEMS.append(dict(num=num, name=name, group=group, risk=risk,
                      purpose=purpose, fn=fn))


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
      "删 robotics 顶层段 → ValueError『缺少顶层配置段』（cfg 自检①）",
      _bad_cfg_item("缺顶层键 robotics", _m_del_top, "缺少顶层配置段"))
_item(11, "坏注册名", "init", "低",
      "robotics.fkine 注册名不存在 → ValueError『未在子包 REGISTRY』（cfg 自检③）",
      _bad_cfg_item("坏注册名", _m_bad_reg, "未在子包 REGISTRY"))
_item(12, "缺 plan_hz", "init", "低",
      "traj 规划器缺 plan_hz → ValueError『缺 plan_hz（必配、无缺省回退）』（cfg 自检④）",
      _bad_cfg_item("缺 plan_hz", _m_no_plan_hz, "缺 plan_hz"))
_item(13, "非整值 plan_hz", "init", "低",
      "plan_hz=10.5 → ValueError『须为 int 或整值浮点且 ≥1』（cfg 自检④）",
      _bad_cfg_item("非整值 plan_hz", _m_frac_hz, "非法", "整值"))
_item(14, "home 越软限位", "init", "低",
      "home_arm[0]=9.9 越出软限位 → ValueError『越出软限位区间』（特征位形校验）",
      _bad_cfg_item("home 越软限位", _m_home_out, "越出软限位区间"))
_item(15, "软限位越硬限位", "init", "低",
      "软限位 q_min=-9.9 越出硬限位 → ValueError『越出对应硬限位』（cfg 自检⑥）",
      _bad_cfg_item("软限位越硬限位", _m_soft_over, "越出", "硬限位"))
_item(16, "限位与 URDF 不符", "init", "低",
      "cfg 关节 q_min 与 URDF <limit> 偏差超容差 → ValueError『与 URDF <limit> 不一致』（cfg 自检⑤）",
      _bad_cfg_item("限位与 URDF 不符", _m_urdf_lim, "与 URDF <limit> 不一致"))

# —— 02 组（只读属性与防篡改）——
_item(20, "writeable=False 防篡改", "readonly", "低",
      "全部对外数组（特征位形/硬软限位四键/tcp workspace_box）原地写均抛异常", _it20)
_item(21, "get_config 深拷贝", "readonly", "低",
      "篡改 get_config() 返回值（home/限位/帧名）不反哺内部状态与后续读取", _it21)
_item(22, "只读引用零拷贝", "readonly", "低",
      "只读 @property 多次访问返回同一本体（零拷贝）、与 get_config 值一致、pin_model.nq=n_arm", _it22)

# —— 03 组（委托面与互斥）——
_item(30, "委托面透传核对", "delegate", "低",
      "JoyArm 公开方法逐项核对 DummyBackend 对应内核（调用名 + 关键参数 + 返回值透传；arm/end 对称）", _it30)
_item(31, "FSM 运行中外部指令互斥", "delegate", "低",
      "FSM 运行中外部线程的 arm 侧指令（send_position/set_mode/enable/clear_error/set_zero）空操作 + warn；只读与 end 侧放行", _it31)
_item(32, "ctrl 线程豁免", "delegate", "低",
      "MOTION 态 ctrl 线程自身的 send_position_arm 照常到达 backend（_is_internal_thread 豁免）", _it32)

# —— 04 组（求解器域）——
_item(40, "六域查询与切换联动", "solvers", "低",
      "get_*_solvers=cfg 名单、默认激活=cfg 首位、切换后激活名/plan_hz/ctrl_hz/MODE 联动（切换直连 _backend.set_mode）", _it40)
_item(41, "坏名 warn 保持原激活", "solvers", "低",
      "六域 set_activated_*('no_such_solver') 均返回 False + warn『未在该域已加载成员中』且激活不变", _it41)
_item(42, "traj 先 plan 后换指针", "solvers", "低",
      "切换成功即新规划器已完成 plan（sample 立即可用）；断连 plan 失败保持原激活 + warn", _it42)

# —— 05 组（数学门面）——
_item(50, "9 方法 vs pinocchio 原生对照", "math", "低",
      "显式传参离线计算与 pinocchio 原生 API 同输入对照：fkine/ikine/ikine_all/jac/jac_pinv/idyn/mass/coriolis/gravity + 缺省硬件态契约", _it50)

# —— 06 组（MotionFSM：DummyBackend 推演）——
_item(60, "start 前置 RuntimeError", "fsm", "低",
      "未连接 start_motion_fsm 抛 RuntimeError『arm 未连接』（拒绝启动）", _it60)
_item(61, "自动迁移序列", "fsm", "低",
      "start 后 DISABLE→HOME→MOTION 两跳迁移（路线表经 home 中转）+『HOME 已到位』日志 + home 指令驱动", _it61)
_item(62, "home 到位门槛", "fsm", "低",
      "q 未到位时停留 HOME 不进 MOTION（§6.3 门槛）；到位后放行迁移", _it62)
_item(63, "保护事件直达", "fsm", "低",
      "MOTION 下 HOLDING/DAMPING/DISABLE 依次直达 + 各态维持指令（hold_q 重发/MIT 纯阻尼/失能轮发）核对", _it63)
_item(64, "stop 超时强推", "fsm", "低",
      "回零永不到位 → 等满 move_wait_timeout 后 warn『超时强制推进』并完成失能与线程收尾", _it64)

# —— 07 组（safe_monitor）——
_item(70, "硬规则 error≥2→DAMPING", "monitor", "低",
      "单关节故障码触发监视器 set_event(DAMPING)，FSM 迁入 DAMPING 并维持 MIT 阻尼", _it70)
_item(71, "断连/连续 None→DISABLE", "monitor", "低",
      "断连与连续 None>0.1s 双路径触发 DISABLE；state_arm=None 仍调自定义规则；ctrl 线程跳拍 warn；断连收尾走 close", _it71)
_item(72, "自定义规则与同拍仲裁", "monitor", "低",
      "同组多命中取危险最高、硬规则优先于自定义（压过更高危自定义）、异常/非法返回忽略且不影响其他规则", _it72)

# —— 08 组（可视化）——
_item(80, "start/stop 离线不炸", "viz", "低",
      "未连接 start_state_viz → 页面 + /state.json 轮询端点可 GET；stop 后 HTTP 端口关闭", _it80)
_item(81, "meshcat 缺失 warn", "viz", "低",
      "未装 meshcat 时降级 warn『未安装 meshcat→仅 HTML 状态面板』；环境已装则 SKIP（降级分支由 80 组覆盖不炸）", _it81)
_item(82, "重复启停幂等 + 线程退出", "viz", "低",
      "start×2 忽略重复、stop×2 无异常、无 joyarm-viz 线程残留（daemon 均退出）", _it82)

# —— 09 组（销毁路径）——
_item(90, "close 幂等", "close", "低",
      "close×2 无未捕获异常，断连序列到达 backend 内核", _it90)
_item(91, "with __exit__ 序列", "close", "低",
      "with 退出触发 close：disable_arm/disable_end 先于 disconnect_arm/disconnect_end", _it91)
_item(92, "__del__ 冒烟", "close", "低",
      "__del__ 存在且 best-effort（del+gc 回收、进程不崩；不做执行时序承诺——正式收尾走 with/close）", _it92)

ALL_NUMS = {it["num"] for it in ITEMS}
_BY_NUM = {it["num"]: it for it in ITEMS}

GROUPS: dict[str, list[int]] = {
    "factory": [0, 1, 2, 3],
    "init": [10, 11, 12, 13, 14, 15, 16],
    "readonly": [20, 21, 22],
    "delegate": [30, 31, 32],
    "solvers": [40, 41, 42],
    "math": [50],
    "fsm": [60, 61, 62, 63, 64],
    "monitor": [70, 71, 72],
    "viz": [80, 81, 82],
    "close": [90, 91, 92],
}
GROUPS["all"] = [n for ns in GROUPS.values() for n in ns]


# ============================================================
# CLI 与执行框架
# ============================================================
class _UsageError(Exception):
    """用法错误（退出码 2）。"""


def resolve_steps(spec: str) -> list[int]:
    """解析 --step：单项编号 / 组名逗号混选。"""
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
            raise _UsageError(
                f"未知步骤 {tok!r}；有效组名：{[k for k in GROUPS if k != 'all'] + ['all']}"
                f"，或编号 {sorted(ALL_NUMS)}")
    if not nums:
        raise _UsageError(f"--step={spec!r} 未解析出任何测试项")
    return sorted(set(nums))


def run_item(it: dict) -> tuple[str, str]:
    """单项执行：打印项与用途 → 执行 → 打印判定与证据（无交互）。"""
    print()
    print(f"[{it['num']:02d}] {it['name']} | 组={it['group']} | 风险={it['risk']}")
    print(f"用途：{it['purpose']}")
    try:
        verdict, evidence = it["fn"]()
    except Exception as e:  # noqa: BLE001（验收脚本须捕获一切）
        verdict, evidence = "FAIL", f"未预期异常 {type(e).__name__}: {e}"
    print(f"[{it['num']:02d}] {verdict} —— {evidence}")
    return verdict, evidence


def print_list():
    print("JoyArm 层离线验收测试项清单（编号=组号×10+组内序号；全离线低危、无交互）")
    print("=" * 68)
    for it in sorted(ITEMS, key=lambda x: x["num"]):
        print(f"[{it['num']:02d}] {it['name']}｜组={it['group']}｜风险={it['risk']}")
        print(f"      用途：{it['purpose']}")
    print("=" * 68)
    print("组：factory(00) / init(01) / readonly(02) / delegate(03) / solvers(04) /")
    print("    math(05) / fsm(06) / monitor(07) / viz(08) / close(09)；all=全部（缺省）")
    print("判定：PASS=符合规格；SKIP=环境条件不满足（非缺陷，如已装 meshcat）；FAIL=违反契约。")


EPILOG = """\
示例：
  python joyarm_core/joyarm/joyarm_verify.py --help       # 本说明
  python joyarm_core/joyarm/joyarm_verify.py --list       # 全部测试项
  python joyarm_core/joyarm/joyarm_verify.py              # 全部（缺省 all）
  python joyarm_core/joyarm/joyarm_verify.py --step=fsm   # 单组（06 组 60~64）
  python joyarm_core/joyarm/joyarm_verify.py --step=61,71 # 单项混选

定位：全离线（不连真机；DummyBackend 注入推演），真机侧验收走
      backend/backend_verify.py。全部测试项无交互（非终端可跑）；
      FSM/监视器/可视化线程类用例每项内部收尾（stop/join），不悬挂线程。

退出码：全 PASS/SKIP=0；任一 FAIL=1；用法错误=2。结尾输出
        SUMMARY: total=… pass=… fail=… skip=…（机器可解析）+ 各组统计行。
"""


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="joyarm_verify.py",
        description=("JoyArm 层离线分级验收脚本：工厂/自检/只读面/委托面/求解器域/"
                     "数学门面/MotionFSM/safe_monitor/可视化/销毁路径 00~09 十组分项；"
                     "DummyBackend 注入（临时改 cfg backend.name + 替换 REGISTRY，"
                     "用完恢复），PASS/SKIP/FAIL 三分判定。"),
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", default="all",
                   help="数字=单项编号（如 61）；组名（factory/init/readonly/delegate/"
                        "solvers/math/fsm/monitor/viz/close/all）；逗号分隔混选；缺省 all")
    p.add_argument("--list", action="store_true", help="仅列出全部测试项，不执行")
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
    try:
        nums = resolve_steps(args.step)
    except _UsageError as e:
        print(f"用法错误：{e}", file=sys.stderr)
        return 2
    # 基础 cfg（注入源）+ 注入装载（finally 恢复，不触碰仓库文件）
    CFG_SOURCE.base = _tweak_cfg(load_base_cfg())
    install_injection()
    print("=" * 68)
    print(f"JoyArm 层离线分级验收：型号 {MODEL}（DummyBackend 注入 {DUMMY_KEY}）")
    print("配置：真实 joyarm_dm.yaml（backend.name 临时指向 Dummy；机器人资产与 cfg 自检用真实文件）")
    print(f"交互终端：{'是' if sys.stdin.isatty() else '否'}（全部用例无交互）；"
          "FSM/monitor/viz 用例逐项收尾线程")
    print(f"待执行 {len(nums)} 项：{[f'{n:02d}' for n in nums]}")
    results: list[tuple[dict, str, str]] = []
    try:
        for num in nums:
            it = _BY_NUM[num]
            v, ev = run_item(it)
            results.append((it, v, ev))
    finally:
        restore_injection()
    n_pass = sum(1 for _, v, _ in results if v == "PASS")
    n_fail = sum(1 for _, v, _ in results if v == "FAIL")
    n_skip = sum(1 for _, v, _ in results if v == "SKIP")
    print()
    print("=" * 68)
    print(f"SUMMARY: total={len(results)} pass={n_pass} fail={n_fail} skip={n_skip}")
    for gname in ("factory", "init", "readonly", "delegate", "solvers",
                  "math", "fsm", "monitor", "viz", "close"):
        sub = [(it, v) for it, v, _ in results if it["group"] == gname]
        if not sub:
            continue
        gp = sum(1 for _, v in sub if v == "PASS")
        gf = sum(1 for _, v in sub if v == "FAIL")
        gs = sum(1 for _, v in sub if v == "SKIP")
        print(f"GROUP {gname}: total={len(sub)} pass={gp} fail={gf} skip={gs}")
    fails = [(it, ev) for it, v, ev in results if v == "FAIL"]
    skips = [(it, ev) for it, v, ev in results if v == "SKIP"]
    if not fails:
        concl = "结论：无失败项，JoyArm 层符合本次离线验收全部已测契约。"
        if skips:
            concl += f"共 {n_skip} 项跳过（环境条件不满足，非缺陷，详见各项证据）。"
    else:
        print(f"结论：{n_fail} 项失败——")
        for it, ev in fails:
            print(f"  FAIL [{it['num']:02d}] {it['name']}：{_trunc(ev)}")
        concl = f"（详见上方 {n_fail} 条 FAIL 证据；其余 pass={n_pass}、skip={n_skip}）"
    print(concl)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
