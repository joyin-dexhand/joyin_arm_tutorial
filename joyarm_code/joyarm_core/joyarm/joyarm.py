"""`JoyArm` —— 机械臂组合根（config 驱动的整机对象模型）

按 `config/<型号>.yaml` 把 `backend`（硬件通信）、`robotics`（六域求解器）与
`robot_model`（URDF 资产）组合成整机对象，对外提供统一调用入口。功能块：

- 初始化与自检（两阶段 + cfg 全套自检 + pin 缩减模型 + 完成 info）；
- 硬件后端委托面（arm/end 两组薄委托 + FSM 运行中的外部指令互斥与 ctrl 线程豁免）；
- 求解器六域成员 / 热切换 / 目标帧通道 + plan/ctrl 两线程 + 数学门面 9 方法；
- 运动状态机 MotionFSM（六状态 + 路线表 + start/stop + 销毁安全退出）；
- 安全事件 SafetyEvent 与 safe_monitor 监视器（硬规则 + 自定义规则）；
- 运行状态可视化（HTTP 轮询版：单页面监控网页 + 单一只读 daemon 线程）。

模块级设施：

- :func:`load_config`：config 加载（缺失/解析失败抛错；工厂 §2 第①步共用）；
- `_run_periodic` / `_PeriodicThread`：私有周期线程机制（§9，留在 joyarm、不入 utils）。
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import pinocchio as pin
import yaml

from ..utils.limits import (
    limits_from_joint_cfgs,
    rand_within_limits,
    soft_limits_from_cfg,
    tcp_limits_from_cfg,
)
from ..utils.transforms import quat_conj, quat_mul, quat_to_axis_angle
from ..utils.types import (
    Accel,
    ControlMode,
    IKResult,
    JointLimits,
    JointState,
    MotionState,
    Pose,
    SafetyEvent,
    TcpLimits,
    TrajFrame,
    Twist,
    Wrench,
)

from ..backend import get_backend
from ..robotics.fkine import REGISTRY as _FKINE_REGISTRY
from ..robotics.ikine import REGISTRY as _IKINE_REGISTRY
from ..robotics.jacobian import REGISTRY as _JACOBIAN_REGISTRY
from ..robotics.dynamics import REGISTRY as _DYNAMICS_REGISTRY
from ..robotics.trajectory import REGISTRY as _TRAJ_REGISTRY
from ..robotics.control import REGISTRY as _CONTROL_REGISTRY

__all__ = ["JoyArm", "load_config"]

logger = logging.getLogger("joyarm_core.joyarm")


# ============================================================
# 路径常量（内部）：config/ 与 robot_model/ 目录位置
# ============================================================
# config/ 目录（joyarm.py 位于 joyarm_core/joyarm/，上溯一级即包根）
_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
)

# robot_model/ 目录（URDF 模型 + 网格文件，运行期加载）
_ROBOT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot_model"
)

# cfg 关节限位与 URDF <limit> 的一致性核对容差（§3.2⑤）
_URDF_LIMIT_TOL = 1e-4

# 电机模式与帧名之外的 yaml 接受判定
_ACTIVE_JOINT_TYPES = ("revolute", "continuous", "prismatic")


# ============================================================
# SafetyEvent ↔ MotionState 映射与危险度（§7.1）
# ============================================================
_EVENT_TO_STATE = {
    SafetyEvent.NORMAL: MotionState.MOTION,
    SafetyEvent.HOLDING: MotionState.HOLD,
    SafetyEvent.HOMING: MotionState.HOME,
    SafetyEvent.ZEROING: MotionState.ZERO,
    SafetyEvent.DAMPING: MotionState.DAMPING,
    SafetyEvent.DISABLE: MotionState.DISABLE,
}

# 危险程度递增序（同拍多命中仲裁用，§7.3）
_EVENT_DANGER = {
    SafetyEvent.NORMAL: 0,
    SafetyEvent.HOLDING: 1,
    SafetyEvent.HOMING: 2,
    SafetyEvent.ZEROING: 3,
    SafetyEvent.DAMPING: 4,
    SafetyEvent.DISABLE: 5,
}


# ============================================================
# MotionFSM 状态转移路线表（§6.4；from 行 → to 列 → 下一跳）
# ============================================================
def _build_route() -> dict:
    """按 §6.4 路线表构造 `{from: {to: next}}`（home(3) 为中枢）。"""
    states = (MotionState.DISABLE, MotionState.ZERO, MotionState.HOME,
              MotionState.MOTION, MotionState.HOLD, MotionState.DAMPING)
    table = {
        # from disable：→zero/→home/→motion 经 home 中转；→hold/→damping 保护性直达
        MotionState.DISABLE: {
            MotionState.ZERO: MotionState.HOME, MotionState.HOME: MotionState.HOME,
            MotionState.MOTION: MotionState.HOME, MotionState.HOLD: MotionState.HOLD,
            MotionState.DAMPING: MotionState.DAMPING},
        # from zero：→home 直达；→motion 经 home；→disable/hold/damping 直达
        MotionState.ZERO: {
            MotionState.DISABLE: MotionState.DISABLE, MotionState.HOME: MotionState.HOME,
            MotionState.MOTION: MotionState.HOME, MotionState.HOLD: MotionState.HOLD,
            MotionState.DAMPING: MotionState.DAMPING},
        # from home：中枢，各态直达
        MotionState.HOME: {
            MotionState.DISABLE: MotionState.DISABLE, MotionState.ZERO: MotionState.ZERO,
            MotionState.MOTION: MotionState.MOTION, MotionState.HOLD: MotionState.HOLD,
            MotionState.DAMPING: MotionState.DAMPING},
        # from motion：→zero 经 home；→home/disable/hold/damping 直达
        MotionState.MOTION: {
            MotionState.DISABLE: MotionState.DISABLE, MotionState.ZERO: MotionState.HOME,
            MotionState.HOME: MotionState.HOME, MotionState.HOLD: MotionState.HOLD,
            MotionState.DAMPING: MotionState.DAMPING},
        # from hold：→zero/→motion 经 home；→home/disable/damping 直达
        MotionState.HOLD: {
            MotionState.DISABLE: MotionState.DISABLE, MotionState.ZERO: MotionState.HOME,
            MotionState.HOME: MotionState.HOME, MotionState.MOTION: MotionState.HOME,
            MotionState.DAMPING: MotionState.DAMPING},
        # from damping：→zero/→motion 经 home；→home/disable/hold 直达
        MotionState.DAMPING: {
            MotionState.DISABLE: MotionState.DISABLE, MotionState.ZERO: MotionState.HOME,
            MotionState.HOME: MotionState.HOME, MotionState.MOTION: MotionState.HOME,
            MotionState.HOLD: MotionState.HOLD},
    }
    assert set(table) == set(states)
    return table


_ROUTE = _build_route()


# ============================================================
# config 加载（§2 工厂第①步共用实现）
# ============================================================
def load_config(model: str) -> dict:
    """读取 `config/<model>.yaml` 型号配置文件（失败抛错，不返回 None）。

    :param model: 型号名（= yaml 文件名 = URDF 目录 `robot_model/<model>/` 同名）。
    :return: 解析后的配置 dict（顶层非空映射）。
    :raises ValueError: 文件缺失 / yaml 语法错误 / 顶层非映射或为空
        （空文件、纯注释均解析为 None，判失败）；消息含文件路径与原始解析错误。
    """
    path = os.path.join(_CONFIG_DIR, f"{model}.yaml")
    if not os.path.isfile(path):
        raise ValueError(
            f"joyarm.py - load_config：config 文件不存在：{path}")
    try:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"joyarm.py - load_config：config 解析失败：{path}（{e}）") from e
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(
            f"joyarm.py - load_config：config 顶层须为非空映射，实际："
            f"{path} 解析为 {type(cfg).__name__}"
            f"{'（空文件或纯注释）' if not cfg else ''}")
    return cfg


# ============================================================
# URDF 解析与校验助手（内部，§3.2⑤ / §3.3）
# ============================================================
def _urdf_joint_table(urdf_path: str) -> dict:
    """解析 URDF 全部 <joint> 元素（文档序），返回 `{关节名: 信息}`。

    信息键：`type`（joint 类型）、`mimic`（是否 mimic 从动）、
    `q_min/q_max/dq_max/tau_max`（<limit> 四键，属性缺失为 `None`）。

    :param urdf_path: URDF 文件路径。
    :raises ValueError: URDF 无法解析。
    """
    def _num(lim, key):
        v = None if lim is None else lim.get(key)
        return None if v is None else float(v)

    joints: dict = {}
    for j in ET.parse(urdf_path).getroot().iterfind("joint"):
        lim = j.find("limit")
        joints[j.get("name")] = {
            "type": j.get("type"),
            "mimic": j.find("mimic") is not None,
            "q_min": _num(lim, "lower"), "q_max": _num(lim, "upper"),
            "dq_max": _num(lim, "velocity"), "tau_max": _num(lim, "effort"),
        }
    return joints


def _urdf_link_names(urdf_path: str) -> List[str]:
    """解析 URDF 全部 <link> 名（重复校验用，§3.2⑤）。"""
    return [lk.get("name") for lk in ET.parse(urdf_path).getroot().iterfind("link")]


def _soft_beyond_hard(soft: JointLimits, hard: JointLimits) -> bool:
    """软限位是否越出硬限位（§3.2⑥；±∞ 软限位维度跳过，只比对有限值）。"""
    def _bad(s, h, upper: bool) -> bool:
        s, h = np.asarray(s, dtype=float), np.asarray(h, dtype=float)
        if upper:
            return bool((np.isfinite(s) & (s > h + 1e-9)).any())
        return bool((np.isfinite(s) & (s < h - 1e-9)).any())

    return (_bad(soft.q_min, hard.q_min, False)
            or _bad(soft.q_max, hard.q_max, True)
            or _bad(soft.dq_max, hard.dq_max, True)
            or _bad(soft.tau_max, hard.tau_max, True))


def _freeze_arrays(obj):
    """只读化（§3.1）：ndarray 本体或数据类内部的 ndarray 字段置 `writeable=False` 后原样返回。"""
    if isinstance(obj, np.ndarray):
        obj.flags.writeable = False
        return obj
    for v in vars(obj).values():
        if isinstance(v, np.ndarray):
            v.flags.writeable = False
    return obj


# ============================================================
# 六域算法成员的构建（内部，§5.1/§5.2）：cfg 段键 → 域缩写与注册表
# ============================================================
# 域缩写（§5.1）→ (cfg 段键, REGISTRY)；cfg 段键以 yaml 现键为准
# （yaml 实际使用 fkine/ikine/jacobian/dynamics/trajectory/control 六键）
_DOMAIN_REGISTRIES = {
    "fkine": ("fkine", _FKINE_REGISTRY),
    "ikine": ("ikine", _IKINE_REGISTRY),
    "jac": ("jacobian", _JACOBIAN_REGISTRY),
    "dyn": ("dynamics", _DYNAMICS_REGISTRY),
    "traj": ("trajectory", _TRAJ_REGISTRY),
    "ctrl": ("control", _CONTROL_REGISTRY),
}


def _parse_domain_specs(cfg_key: str, spec) -> List[tuple]:
    """把 config `robotics.<cfg_key>` 段解析成 `[(注册名, 参数字典), ...]`。

    单项可以是注册名字符串，或 `{name: 注册名, 其他键: 构造参数}` 字典。

    :raises ValueError: 规格类型非法或规格缺 `name` 键。
    """
    if not spec:
        return []
    specs = spec if isinstance(spec, (list, tuple)) else [spec]
    parsed: List[tuple] = []
    for s in specs:
        if isinstance(s, str):
            parsed.append((s, {}))
        elif isinstance(s, dict):
            params = dict(s)
            name = params.pop("name", None)
            if name is None:
                raise ValueError(
                    f"joyarm.py - _parse_domain_specs：robotics.{cfg_key} 规格缺 name 键：{s!r}")
            parsed.append((name, params))
        else:
            raise ValueError(
                f"joyarm.py - _parse_domain_specs：robotics.{cfg_key} 规格类型非法"
                f"（{type(s).__name__}），需为注册名字符串或 {{name:..., **参数}} 字典")
    return parsed


def _check_pacing_hz(cfg_key: str, name: str, key: str, value) -> float:
    """校验管线起搏键（plan_hz/ctrl_hz，§3.2④）：必配、int 或整值浮点且 ≥1。

    :return: 归一后的 float 频率。
    :raises ValueError: 缺省 / 非数值 / 非整值 / <1。
    """
    if value is None:
        raise ValueError(
            f"joyarm.py - _check_pacing_hz：robotics.{cfg_key}『{name}』缺 {key}"
            f"（必配、无缺省回退；接受 int 或整值浮点且 ≥1）")
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float("nan")
    if not (np.isfinite(v) and v >= 1.0 and float(v).is_integer()):
        raise ValueError(
            f"joyarm.py - _check_pacing_hz：robotics.{cfg_key}『{name}』{key}={value!r}"
            f" 非法（须为 int 或整值浮点且 ≥1）")
    return v


# ============================================================
# 周期线程设施（§9：私有周期线程机制，留在 joyarm、不入 utils）
# ============================================================
_ERR_LOG_INTERVAL: float = 0.5   # 同一故障反复出现时，日志的最小间隔（秒）
_JOIN_TIMEOUT: float = 1.0      # stop() 等线程退出的上限（秒）


def _as_hz(hz) -> float:
    """频率参数归一：直接给数值，或给一个"返回数值的无参函数"。"""
    return float(hz() if callable(hz) else hz)


def _run_periodic(stop: threading.Event, hz, step: Callable[[], None],
                  name: str = "") -> None:
    """以 `hz` 频率循环调用 `step()`，`stop` 置位后退出。

    节拍按绝对时间对齐：某步执行太慢只跳过单拍，之后重新对齐。
    `step()` 抛异常只记日志、下个周期照常重试；同故障持续出现时日志每 0.5 秒
    最多一条（省略的次数会补记）。

    :param stop: 停止事件（置位即退出循环）。
    :param hz:   执行频率 Hz（数值或返回数值的无参函数，每周期重新取）。
    :param step: 单步函数（无参数；内部异常视为"这步失败，下周期重来"）。
    :param name: 日志里标识本循环的名字（如线程名）。
    """
    period = 1.0 / _as_hz(hz)
    deadline = time.perf_counter()
    last_log = float("-inf")
    suppressed = 0
    while not stop.is_set():
        deadline += period
        now = time.perf_counter()
        if deadline > now:
            stop.wait(deadline - now)
        else:                              # 单步太慢错过节拍：重新对齐，不追赶
            deadline = now
        try:
            step()
        except Exception:
            if time.monotonic() - last_log >= _ERR_LOG_INTERVAL:
                detail = f"（此前 {suppressed} 次同类失败已节流省略）" \
                         if suppressed else ""
                logger.exception("joyarm.py - _run_periodic：%s单步 %s 失败，下周期重试%s",
                                 f"[{name}] " if name else "",
                                 getattr(step, "__name__", step), detail)
                last_log = time.monotonic()
                suppressed = 0
            else:
                suppressed += 1
        try:
            period = 1.0 / _as_hz(hz)      # 每周期重取（运行期改频率即时生效）
        except Exception:
            pass                           # 取频率失败：沿用上一节拍


class _PeriodicThread:
    """一个可反复启停的周期线程（`start()` 启动、`stop()` 停止回收）。

    `stop()` 最多等 1 秒；超时抛 `RuntimeError` 并保留线程引用（`is_alive()`
    仍为真、再次 `start()` 会被拒绝）——避免旧线程还卡在单步里时又启新线程，
    出现两个循环同时跑。
    """

    def __init__(self, step: Callable[[], None], hz: float, name: str = "periodic") -> None:
        """:param step: 单步函数；:param hz: 频率 Hz；:param name: 线程名。"""
        self._step = step
        self._hz = float(hz)
        self._name = str(name)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def hz(self) -> float:
        return self._hz

    @property
    def thread(self) -> Optional[threading.Thread]:
        """底层线程对象（ctrl 线程互斥豁免判定用；未启动/已回收为 `None`）。"""
        return self._thread

    def set_hz(self, hz: float) -> None:
        """修改运行频率（下个周期生效；运行中换算法时用它对齐节拍）。"""
        self._hz = float(hz)

    def is_alive(self) -> bool:
        """线程是否还在跑（从未启动过算没在跑）。"""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """启动线程；已在运行（或上次 stop 没停干净）时抛 `RuntimeError`。"""
        if self._thread is not None:
            raise RuntimeError(
                f"joyarm.py - _PeriodicThread.start：[{self._name}] 线程仍在运行"
                f"（或上次 stop 未成功退出），须先确认 stop() 完成")
        self._stop.clear()
        self._thread = threading.Thread(
            target=_run_periodic,
            args=(self._stop, lambda: self._hz, self._step, self._name),
            daemon=True, name=self._name)
        self._thread.start()

    def stop(self) -> None:
        """停止线程；超时（1s）没退出抛 `RuntimeError`（引用保留）。"""
        self._stop.set()
        th = self._thread
        if th is None:
            return
        th.join(timeout=_JOIN_TIMEOUT)
        if th.is_alive():
            raise RuntimeError(
                f"joyarm.py - _PeriodicThread.stop：[{self._name}] 线程 "
                f"{_JOIN_TIMEOUT}s 内未退出（单步可能阻塞在总线 IO 上）；"
                f"在确认其退出前请勿重新 start()")
        self._thread = None


# ============================================================
# 可视化监控页面模板（§8：HTTP 轮询、零新增依赖）
# ============================================================
# 左侧 iframe 嵌 meshcat（无 meshcat 时隐藏并显示占位），右侧状态面板；
# JS 以约 5 Hz fetch /state.json 刷新；「轨迹显示」复选框经 /traj?on=1|0 切换。
_VIZ_PAGE_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>JoyArm 运行状态监控</title>
<style>
 body { margin: 0; font-family: system-ui, sans-serif; background: #1e1e1e;
        color: #ddd; display: flex; height: 100vh; }
 #left { flex: 1; position: relative; }
 #left iframe { width: 100%; height: 100%; border: 0; }
 #placeholder { position: absolute; inset: 0; display: flex; align-items: center;
                justify-content: center; color: #777; font-size: 18px; }
 #right { width: 560px; overflow-y: auto; padding: 12px; box-sizing: border-box; }
 h2 { font-size: 15px; margin: 12px 0 6px; color: #7fd1ff; }
 table { width: 100%; border-collapse: collapse; font-size: 12px;
         font-family: monospace; }
 th, td { border: 1px solid #444; padding: 2px 5px; text-align: center; }
 .kv { font-size: 13px; margin: 4px 0; font-family: monospace; }
 .ok { color: #7fe27f; } .bad { color: #ff7f7f; } .na { color: #999; }
 label { font-size: 13px; display: block; margin: 10px 0; }
</style>
</head>
<body>
<div id="left">
  <iframe id="mc" src="__MESHCAT_SRC__"></iframe>
  <div id="placeholder" hidden>meshcat 未启用（未安装或初始化失败，仅 HTML 状态面板）</div>
</div>
<div id="right">
  <h2>FSM / 安全事件</h2><div class="kv" id="fsm">等待数据…</div>
  <h2>末端位姿与速度 frame_end（位置 m / 四元数 wxyz / 速度）</h2><div class="kv" id="tcp">等待数据…</div>
  <h2>arm 关节</h2><table id="arm"></table>
  <h2>end 关节</h2><table id="end"></table>
  <h2>backend 状态</h2><div class="kv" id="bk">等待数据…</div>
  <label><input type="checkbox" id="traj"> 轨迹显示（需 FSM 处于 MOTION 态）</label>
</div>
<script>
const HAS_MC = __HAS_MESHCAT__;
const $ = (id) => document.getElementById(id);
if (!HAS_MC) { $("mc").remove(); $("placeholder").hidden = false; }
function fmt(v, d = 4) { return (v === null || v === undefined) ? "—" : (+v).toFixed(d); }
function tri(v) {
  return v === true ? '<span class="ok">true</span>'
       : v === false ? '<span class="bad">false</span>'
       : '<span class="na">None</span>';
}
function jointTable(el, g) {
  if (!g || !g.names || !g.names.length) { el.innerHTML = "<tr><td>（无关节）</td></tr>"; return; }
  let h = "<tr><th>#</th><th>name</th><th>q rad</th><th>dq rad/s</th><th>tau N·m</th>"
        + "<th>T_mos ℃</th><th>T_rot ℃</th><th>err</th></tr>";
  g.names.forEach((n, i) => {
    const err = (g.error && g.error[i] !== undefined) ? g.error[i] : null;
    h += `<tr><td>${i + 1}</td><td>${n}</td>`
       + `<td>${fmt(g.q && g.q[i])}</td><td>${fmt(g.dq && g.dq[i])}</td>`
       + `<td>${fmt(g.tau && g.tau[i])}</td><td>${fmt(g.temp_mos && g.temp_mos[i])}</td>`
       + `<td>${fmt(g.temp_rotor && g.temp_rotor[i])}</td>`
       + `<td class="${(err !== null && err >= 2) ? 'bad' : 'ok'}">${err === null ? "—" : err}</td></tr>`;
  });
  el.innerHTML = h;
}
async function poll() {
  try {
    const s = await (await fetch("/state.json")).json();
    $("fsm").innerHTML = `current=<b>${s.fsm.current}</b> target=<b>${s.fsm.target}</b> `
                       + `event=<b>${s.event}</b>`;
    $("tcp").innerHTML = s.tcp
      ? `p=[${s.tcp.position.map(v => fmt(v, 3)).join(", ")}]<br>`
        + `q(wxyz)=[${s.tcp.orientation.map(v => fmt(v, 3)).join(", ")}]`
        + (s.tcp.twist
          ? `<br>v=[${s.tcp.twist.linear.map(v => fmt(v, 3)).join(", ")}] m/s`
            + `<br>w=[${s.tcp.twist.angular.map(v => fmt(v, 3)).join(", ")}] rad/s`
          : "")
      : "—";
    jointTable($("arm"), s.arm);
    jointTable($("end"), s.end);
    $("bk").innerHTML = `connected_arm=${tri(s.backend.connected_arm)} `
      + `enabled_arm=${tri(s.backend.enabled_arm)} `
      + `mode_arm=${s.backend.mode_arm === null ? '<span class="na">None</span>' : s.backend.mode_arm}`
      + `<br>connected_end=${tri(s.backend.connected_end)} `
      + `enabled_end=${tri(s.backend.enabled_end)} `
      + `mode_end=${s.backend.mode_end === null || s.backend.mode_end === undefined
                    ? '<span class="na">None</span>' : s.backend.mode_end}`;
    $("traj").checked = !!s.traj_enabled;
  } catch (e) { /* 下轮重试 */ }
}
$("traj").addEventListener("change", async () => {
  try { await fetch("/traj?on=" + ($("traj").checked ? 1 : 0)); } catch (e) {}
});
setInterval(poll, 200);   // 约 5 Hz 轮询（fetch /state.json）
poll();
</script>
</body>
</html>
"""

# _state_viz_worker 的数据刷新频率（Hz；面板 JS 以约 5 Hz 轮询，刷新端取 10 Hz 保证领先）
_VIZ_WORKER_HZ: float = 10.0
# 监视器「状态连续 None」判 DISABLE 的时长阈值（§7.3：连续 0.1 s）
_MONITOR_NONE_TIMEOUT: float = 0.1


def _json_round(x) -> Optional[float]:
    """单数值转 JSON 安全值（None / 非有限 → None；其余 round 4 位）。"""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, 4) if np.isfinite(v) else None


def _json_arr(x) -> Optional[List[Optional[float]]]:
    """数组转 JSON 安全列表（整组 None 保持 None；逐元素经 :func:`_json_round`）。"""
    if x is None:
        return None
    return [_json_round(v) for v in np.asarray(x).reshape(-1)]


class JoyArm:
    """JoyArm 机械臂类（本体 arm + 末端 end，config 驱动的组合根）

    两阶段初始化（§3）：先声明全部私有成员并赋安全默认值，再按依赖顺序赋真值；
    任一自检失败抛具体异常、对象不构造。类内成员一律 `_` 前缀私有，类外访问经
    只读 `@property`（防篡改；数组/限位类成员只读化后零拷贝返本体）或 `get_*`。

    私有助手清单（§9）：

    - `_fsm_worker` / `_fsm_tick`：FSM 线程体与单拍循环（固定 `_fsm_hz`）；
    - `_enter_motion_state` / `_exit_motion_state`：状态进入/退出动作分发；
    - `_maintain_motion_state`：状态维持动作分发（§6.2 每拍核心动作）；
    - `_disable_clear_disable_arm`：「失能→清错→失能」序列（DISABLE 态 enter 用）；
    - `_monitor_worker`：safe_monitor 线程体（§7.3 规则集）；
    - `_state_viz_worker`：状态可视化只读监控线程体（§8）；
    - 其余内部助手（cfg 自检 / pin 模型 / 六域 / 互斥豁免等）随用随定义。

    :param name: 型号名（= `config/<name>.yaml` 文件名 = `robot_model/<name>/` 目录名）。
    :raises ValueError: config 缺失、自检（§3.2）任一项失败、URDF/限位/特征位形校验失败。
    :raises RuntimeError: backend 或六域求解器构造失败（保留原因链）。
    """

    # ============================================================
    # 一、初始化：两阶段（§3）——私有成员声明 → 按依赖顺序赋真值
    # ============================================================

    # ------------------------------------------------------------
    # 阶段一：私有成员声明（安全默认值；None / 空容器 / 缺省数）
    # ------------------------------------------------------------
    def __init__(self, name: str):
        # ---- 基本信息 ----
        self._model: str = str(name)                     # 型号名
        self._cfg: Optional[dict] = None                 # config 全量 dict（经 §3.2 自检）
        self._frame_end: str = ""                        # 末端帧名（cfg.joyarm.frame_end）
        self._urdf_path: str = ""                        # URDF 文件路径

        # ---- 运动学模型（pinocchio，URDF 构建锁定缩减，§3.3）----
        self._pin_model: Optional[pin.Model] = None      # nq = n_arm 的缩减模型（只读共享）

        # ---- 关节数与关节名 ----
        self._n_arm: int = 0                             # 本体关节数（cfg backend.arm.joints 条数）
        self._n_end: int = 0                             # 末端电机数（cfg backend.end.joints 条数）
        self._joint_names_arm: tuple = ()                # 本体关节名（cfg 顺序 = q 向量顺序）
        self._joint_names_end: tuple = ()                # 末端电机名

        # ---- 限位（硬限位拦指令 / 软限位供上层状态判断，均只读化）----
        self._joint_limits_hard_arm: Optional[JointLimits] = None
        self._joint_limits_hard_end: Optional[JointLimits] = None
        self._joint_limits_soft_arm: Optional[JointLimits] = None
        self._joint_limits_soft_end: Optional[JointLimits] = None

        # ---- 特征位形（zero=全零 / neutral=软限位中点 / home=cfg 配置）----
        self._zero_arm: np.ndarray = np.zeros(0)
        self._zero_end: np.ndarray = np.zeros(0)
        self._neutral_arm: np.ndarray = np.zeros(0)
        self._neutral_end: np.ndarray = np.zeros(0)
        self._home_arm: np.ndarray = np.zeros(0)
        self._home_end: np.ndarray = np.zeros(0)

        # ---- 末端空间限位 ----
        self._tcp_limits: TcpLimits = TcpLimits()

        # ---- 电机通信后端 ----
        self._backend = None                             # 经 §4 委托面访问，不直接暴露

        # ---- 六域求解器（§5.1：{注册名: 实例} + 激活对象指针；未配置为 None）----
        self._fkine_solvers: Optional[dict] = None
        self._ikine_solvers: Optional[dict] = None
        self._jac_solvers: Optional[dict] = None
        self._dyn_solvers: Optional[dict] = None
        self._traj_solvers: Optional[dict] = None
        self._ctrl_solvers: Optional[dict] = None
        self._activated_fkine_solver = None
        self._activated_ikine_solver = None
        self._activated_jac_solver = None
        self._activated_dyn_solver = None
        self._activated_traj_solver = None
        self._activated_ctrl_solver = None
        self._plan_hz: Optional[float] = None            # 激活规划器 plan_hz（切换时更新）
        self._ctrl_hz: Optional[float] = None            # 激活控制器 ctrl_hz（切换时更新）
        self._plan_hz_by_name: dict = {}                 # {traj 注册名: plan_hz}
        self._ctrl_hz_by_name: dict = {}                 # {ctrl 注册名: ctrl_hz}

        # ---- 轨迹目标帧（§5.5：应用线程写，规划线程每拍整体重读）----
        self._target_trajframe: Union[TrajFrame, List[TrajFrame], None] = None

        # ---- MotionFSM（§6/§7）----
        self._current_motion_state: MotionState = MotionState.DISABLE   # FSM 线程独占写
        self._target_motion_state: MotionState = MotionState.DISABLE
        self._safe_event: SafetyEvent = SafetyEvent.DISABLE
        self._fsm_hz: float = 50.0                       # cfg.joyarm.MotionFSM.fsm_hz（默认 50）
        self._mit_kd_damping: np.ndarray = np.zeros(0)   # DAMPING 态 MIT kd（cfg 必配）
        self._monitor_hz: float = 50.0                   # 缺省 = fsm_hz（§7.3）
        self._arrival_tol_q_arm: np.ndarray = np.zeros(0)  # 到位容差（§10⑨，逐关节）
        self._arrival_tol_q_end: np.ndarray = np.zeros(0)
        self._arrival_tol_pos: float = 1e-3              # 末端位置容差 m（缺省 1e-3）
        self._arrival_tol_rot: float = 1e-2              # 末端姿态容差 rad（缺省 1e-2）
        self._move_wait_timeout: float = 10.0            # cfg joyarm.runtime.move_wait_timeout
        self._hold_q: Optional[np.ndarray] = None        # HOLD 态锁定的关节位置
        self._zero_moving: bool = False                  # ZERO 态「移动中」子阶段标志
        self._home_arrived: bool = False                 # HOME 态到位日志去重
        self._fsm_periodic: Optional[_PeriodicThread] = None

        # ---- plan/ctrl 运动线程（§5.6）----
        self._plan_periodic: Optional[_PeriodicThread] = None
        self._ctrl_periodic: Optional[_PeriodicThread] = None
        self._ctrl_thread: Optional[threading.Thread] = None  # ctrl 线程句柄（互斥豁免判定）
        self._motion_pause = threading.Event()           # 换规划器/启动期间 ctrl 暂停派发
        self._ctrl_none_warn_last: float = float("-inf")  # ctrl 线程跳拍告警节流时间戳

        # ---- safe_monitor（§7.3）----
        self._monitor_periodic: Optional[_PeriodicThread] = None
        self._monitor_rules: List[Callable] = []         # 自定义规则列表
        self._monitor_none_since: Optional[float] = None  # get_state_arm 连续 None 起算时刻

        # ---- 运行状态可视化（§8）----
        self._viz_periodic: Optional[_PeriodicThread] = None
        self._viz_httpd: Optional[ThreadingHTTPServer] = None
        self._viz_pin = None                             # pin.visualize MeshcatVisualizer（缺包为 None）
        self._viz_state: dict = {}                       # 面板数据快照（HTTP 线程读）
        self._viz_lock = threading.Lock()
        self._viz_traj_points: int = 200                 # 轨迹显示采样点数（start 可配）
        self._viz_traj_enabled: bool = False             # 轨迹显示开关（界面可切）
        self._viz_warn_last: float = float("-inf")       # meshcat 刷新失败告警节流

        # ------------------------------------------------------------
        # 阶段二：按依赖顺序赋真值（§3）
        # cfg 加载自检 → _pin_model → 关节数/名 → 限位 → 特征位形 →
        # _tcp_limits → _backend → 六域求解器与激活项 → _plan/_ctrl_hz → FSM 参数
        # ------------------------------------------------------------
        # 1) cfg 加载 + 全套自检（§3.2；frame_end/urdf_path 先行赋值供自检⑤与 pin 构建用）
        cfg = load_config(name)                          # 缺失/解析失败 → 抛错
        self._cfg = cfg
        jcfg = cfg.get("joyarm") or {}
        backend_cfg = dict(cfg.get("backend") or {})
        self._frame_end = str(jcfg.get("frame_end") or "")
        self._urdf_path = os.path.join(_ROBOT_MODEL_DIR, name, "urdf", f"{name}.urdf")
        self._check_cfg(cfg)                             # ①~⑤；⑥在限位构建后补查

        arm_joint_cfgs = list((backend_cfg.get("arm") or {}).get("joints") or [])
        end_joint_cfgs = list((backend_cfg.get("end") or {}).get("joints") or [])

        # 2) _pin_model（§3.3：URDF 全量模型 → 锁定非 cfg.arm 名单活动关节 → 缩减模型）
        self._pin_model = self._build_pin_model(self._urdf_path,
                                                [str(j.get("name")) for j in arm_joint_cfgs])

        # 3) 关节数与关节名（cfg 顺序 = URDF 顺序 = q 向量顺序，⑤已校验）
        self._joint_names_arm = tuple(str(j.get("name")) for j in arm_joint_cfgs)
        self._joint_names_end = tuple(str(j.get("name")) for j in end_joint_cfgs)
        self._n_arm = len(self._joint_names_arm)
        self._n_end = len(self._joint_names_end)

        # 4) 限位：硬限位（backend joints 四键）→ 软限位（joyarm 直配）→ ⑥ 软在硬内
        self._joint_limits_hard_arm = _freeze_arrays(limits_from_joint_cfgs(arm_joint_cfgs))
        self._joint_limits_hard_end = _freeze_arrays(limits_from_joint_cfgs(end_joint_cfgs))
        self._joint_limits_soft_arm = _freeze_arrays(
            soft_limits_from_cfg(jcfg["joint_limits_soft_arm"], self._n_arm))
        if self._n_end:
            self._joint_limits_soft_end = _freeze_arrays(
                soft_limits_from_cfg(jcfg["joint_limits_soft_end"], self._n_end))
        for tag, soft, hard in (("arm", self._joint_limits_soft_arm, self._joint_limits_hard_arm),
                                ("end", self._joint_limits_soft_end, self._joint_limits_hard_end)):
            if soft is None or hard is None:
                continue     # 无该组（如 n_end=0）不校验
            if _soft_beyond_hard(soft, hard):
                raise ValueError(
                    f"joyarm.py - JoyArm.__init__：joyarm.joint_limits_soft_{tag} 越出"
                    f"对应硬限位（软限位须位于硬限位内，供上层状态判断；±∞ 维不设软限）")

        # 5) 特征位形：zero 全零 / neutral 软限位中点 / home cfg 配置（须在软限位内）
        self._zero_arm = _freeze_arrays(np.zeros(self._n_arm))
        self._zero_end = _freeze_arrays(np.zeros(self._n_end))
        self._neutral_arm = _freeze_arrays(
            (np.asarray(self._joint_limits_soft_arm.q_min, float)
             + np.asarray(self._joint_limits_soft_arm.q_max, float)) / 2.0)
        self._neutral_end = _freeze_arrays(
            (np.asarray(self._joint_limits_soft_end.q_min, float)
             + np.asarray(self._joint_limits_soft_end.q_max, float)) / 2.0
        ) if self._n_end else _freeze_arrays(np.zeros(0))
        self._home_arm = self._parse_home(jcfg.get("home_arm"),
                                          self._joint_limits_soft_arm, "home_arm")
        self._home_end = self._parse_home(jcfg.get("home_end"),
                                          self._joint_limits_soft_end, "home_end")

        # 6) 末端空间限位（先校验后解析，§3.1；解析后 workspace_box 每行 [min,max]）
        self._tcp_limits = _freeze_arrays(self._parse_tcp_limits(jcfg.get("tcp_limits")))

        # 7) 电机通信后端（构造失败 → JoyArm 构造失败）
        backend_name = str(backend_cfg.get("name"))
        bcfg_rest = dict(backend_cfg)
        bcfg_rest.pop("name", None)
        try:
            self._backend = get_backend(backend_name)(bcfg_rest)
        except Exception as e:
            raise RuntimeError(
                f"joyarm.py - JoyArm.__init__：backend『{backend_name}』构造失败：{e}") from e

        # 8) 六域求解器（§5.2）与激活项 + 9) _plan/_ctrl_hz
        self._build_domains(cfg.get("robotics") or {})
        if self._traj_solvers:
            self._plan_hz = self._plan_hz_by_name[self._activated_name_of("traj")]
        if self._ctrl_solvers:
            self._ctrl_hz = self._ctrl_hz_by_name[self._activated_name_of("ctrl")]

        # 10) FSM / 监视 / 到位容差 / 运行超时参数（§6/§7）
        self._parse_fsm_params(jcfg)

        # 11) 初始化完成 info（§3.4）
        self._log_init_info()

    # ============================================================
    # 二、初始化助手（内部）：cfg 自检 / URDF 对比 / pin 模型 / 特征位形 / 限位解析
    # ============================================================

    # =============== cfg 内容自检（§3.2 清单） ===============
    def _check_cfg(self, cfg: dict) -> None:
        """cfg 加载后的跨成员/跨源自检（①~⑤；⑥在限位构建后由 `__init__` 补查）。

        :raises ValueError: 任一项失败；一条消息列出全部问题（指明哪一条、期望 vs 实际）。
        """
        problems: List[str] = []
        jcfg = cfg.get("joyarm") or {}
        bcfg = cfg.get("backend") or {}
        robotics_cfg = cfg.get("robotics") or {}

        # ---- ① 三顶层键存在 ----
        for key in ("joyarm", "robotics", "backend"):
            if key not in cfg:
                problems.append(f"① 缺少顶层配置段：{key}")
        if not jcfg.get("frame_end"):
            problems.append("joyarm.frame_end 缺失（末端帧名，pin 模型帧）")

        # ---- 关节数（后续维度检查基准）----
        n_arm_cfg = len((bcfg.get("arm") or {}).get("joints") or [])
        n_end_cfg = len((bcfg.get("end") or {}).get("joints") or [])

        # ---- ② home / 软限位维度 ----
        for key, n_ref in (("home_arm", n_arm_cfg), ("home_end", n_end_cfg)):
            v = jcfg.get(key)
            if v is None:
                if n_ref:
                    problems.append(f"② joyarm.{key} 缺失（须 {n_ref} 元列表）")
                continue
            size = np.asarray(v, dtype=float).reshape(-1).shape[0]
            if size != n_ref:
                problems.append(f"② joyarm.{key} 维度 {size} ≠ backend 关节数 {n_ref}")
        for key, n_ref in (("joint_limits_soft_arm", n_arm_cfg),
                           ("joint_limits_soft_end", n_end_cfg)):
            soft = jcfg.get(key)
            if soft is None:
                if n_ref:
                    problems.append(f"② joyarm.{key} 缺失（neutral 取软限位中点，必配）")
                continue
            if not isinstance(soft, dict):
                problems.append(f"② joyarm.{key} 需为四键字典（q_min/q_max/dq_max/tau_max）")
                continue
            for k in ("q_min", "q_max", "dq_max", "tau_max"):
                if k not in soft:
                    continue     # 缺键由 soft_limits_from_cfg 回退 ±∞（维度不构成问题）
                arr = np.asarray(soft[k], dtype=float).reshape(-1)
                if arr.ndim == 0 or arr.shape[0] == 1:
                    continue     # 标量：全关节统一广播
                if arr.shape[0] != n_ref:
                    problems.append(
                        f"② joyarm.{key}.{k} 长度 {arr.shape[0]} ≠ 关节数 {n_ref}"
                        f"（逐关节需 n 元列表，或写标量统一全关节）")

        # ---- ③④ 六域注册名 + 管线起搏键 ----
        for domain, (cfg_key, registry) in _DOMAIN_REGISTRIES.items():
            try:
                specs = _parse_domain_specs(cfg_key, robotics_cfg.get(cfg_key))
            except ValueError as e:
                problems.append(f"③ {e}")
                continue
            for reg_name, params in specs:
                if reg_name not in registry:
                    problems.append(
                        f"③ robotics.{cfg_key} 注册名『{reg_name}』未在子包 REGISTRY"
                        f"（可用：{sorted(registry) or '无（尚未实现注册）'}）")
                if domain == "traj":
                    try:
                        _check_pacing_hz(cfg_key, reg_name, "plan_hz", params.get("plan_hz"))
                    except ValueError as e:
                        problems.append(f"④ {e}")
                elif domain == "ctrl":
                    try:
                        _check_pacing_hz(cfg_key, reg_name, "ctrl_hz", params.get("ctrl_hz"))
                    except ValueError as e:
                        problems.append(f"④ {e}")

        # ---- ⑤ cfg.backend ↔ URDF 对比 ----
        problems.extend(self._backend_urdf_problems(bcfg))

        if problems:
            raise ValueError(
                "joyarm.py - JoyArm._check_cfg：配置自检未通过：\n"
                + "\n".join(f"  - {p}" for p in problems))

    # =============== cfg.backend ↔ URDF 对比（§3.2⑤ 细则） ===============
    def _backend_urdf_problems(self, bcfg: dict) -> List[str]:
        """cfg.backend 关节登记与 URDF 的一致性问题清单（§3.2⑤，任一不符抛异常）。

        须在 `self._urdf_path` 赋值后调用（`__init__` 已保证调用顺序）。
        """
        problems: List[str] = []
        arm_cfgs = list((bcfg.get("arm") or {}).get("joints") or [])
        end_cfgs = list((bcfg.get("end") or {}).get("joints") or [])
        if not arm_cfgs:
            return ["⑤ backend.arm.joints 为空（本体电机列表必配）"]
        if not os.path.isfile(self._urdf_path):
            return [f"⑤ URDF 文件不存在：{self._urdf_path}"]
        joints = _urdf_joint_table(self._urdf_path)
        links = _urdf_link_names(self._urdf_path)

        # URDF joint/link 名无重复
        dup_j = sorted({n for n in joints if list(joints).count(n) > 1})
        dup_l = sorted({n for n in links if links.count(n) > 1})
        if dup_j:
            problems.append(f"⑤ URDF joint 名重复：{dup_j}")
        if dup_l:
            problems.append(f"⑤ URDF link 名重复：{dup_l}")

        # cfg 关节名无重复 + 四键限位与 URDF <limit> 逐键差 ≤ 1e-4 + q_min ≤ q_max
        for part, joint_cfgs in (("arm", arm_cfgs), ("end", end_cfgs)):
            names = [str(j.get("name", "")) for j in joint_cfgs]
            dup = sorted({n for n in names if names.count(n) > 1})
            if dup:
                problems.append(f"⑤ backend.{part}.joints 关节名重复：{dup}")
            for j in joint_cfgs:
                jn = str(j.get("name", ""))
                u = joints.get(jn)
                if u is None:
                    problems.append(f"⑤ backend.{part}.joints 关节『{jn}』在 URDF 中不存在")
                    continue
                if u["type"] not in _ACTIVE_JOINT_TYPES:
                    problems.append(
                        f"⑤ backend.{part}.joints 关节『{jn}』在 URDF 中为 {u['type']}"
                        f"（fixed 关节不允许登记）")
                    continue
                if u["mimic"]:
                    problems.append(
                        f"⑤ backend.{part}.joints 关节『{jn}』在 URDF 中为 mimic 从动关节"
                        f"（不允许登记）")
                    continue
                for key in ("q_min", "q_max", "dq_max", "tau_max"):
                    cfg_v, urdf_v = j.get(key), u[key]
                    if cfg_v is None or urdf_v is None:
                        continue     # cfg 缺键由硬限位构建回退；URDF 缺属性（如 continuous）不比对
                    if abs(float(cfg_v) - urdf_v) > _URDF_LIMIT_TOL:
                        problems.append(
                            f"⑤ backend.{part}.joints 关节『{jn}』{key} 与 URDF <limit> 不一致"
                            f"（cfg={cfg_v}，URDF={urdf_v}，容差 {_URDF_LIMIT_TOL}）")
                try:
                    if float(j["q_min"]) > float(j["q_max"]):
                        problems.append(
                            f"⑤ backend.{part}.joints 关节『{jn}』q_min={j['q_min']}"
                            f" > q_max={j['q_max']}（行程上下限写反）")
                except (TypeError, ValueError, KeyError):
                    pass

        # URDF 驱动关节（非 fixed 非 mimic）按命名约定分组，须与 cfg 同名同序、编号连续
        driven = [n for n, u in joints.items()
                  if u["type"] in _ACTIVE_JOINT_TYPES and not u["mimic"]]
        arm_driven = [n for n in driven if re.fullmatch(r"joint\d+", n)]
        end_driven = [n for n in driven if re.fullmatch(r"joint_end\d+", n)]
        other = [n for n in driven if n not in arm_driven and n not in end_driven]
        if other:
            problems.append(
                f"⑤ URDF 驱动关节 {other} 命名不符约定（arm 须 joint1~jointN、"
                f"end 须 joint_end1~joint_endM）")
        expected_arm = [f"joint{i}" for i in range(1, len(arm_driven) + 1)]
        if arm_driven != expected_arm:
            problems.append(
                f"⑤ URDF arm 驱动关节编号/顺序异常：{arm_driven}（期望 {expected_arm}）")
        expected_end = [f"joint_end{i}" for i in range(1, len(end_driven) + 1)]
        if end_driven != expected_end:
            problems.append(
                f"⑤ URDF end 驱动关节编号/顺序异常：{end_driven}（期望 {expected_end}）")
        cfg_arm_names = [str(j.get("name", "")) for j in arm_cfgs]
        cfg_end_names = [str(j.get("name", "")) for j in end_cfgs]
        if cfg_arm_names != arm_driven:
            problems.append(
                f"⑤ backend.arm.joints 与 URDF arm 驱动关节不同名同序：\n"
                f"      config：{cfg_arm_names}\n      URDF  ：{arm_driven}")
        if cfg_end_names != end_driven:
            problems.append(
                f"⑤ backend.end.joints 与 URDF end 驱动关节不同名同序：\n"
                f"      config：{cfg_end_names}\n      URDF  ：{end_driven}")
        unregistered = [n for n in driven
                        if n not in cfg_arm_names and n not in cfg_end_names]
        if unregistered:
            problems.append(
                f"⑤ URDF 驱动关节 {unregistered} 未在 cfg.backend.joints 登记"
                f"（cfg 只登记非 fixed、非 mimic 的驱动关节，且须全部登记）")
        return problems

    # =============== pin 缩减模型（§3.3） ===============
    def _build_pin_model(self, urdf_path: str, arm_joint_names: List[str]) -> pin.Model:
        """URDF 全量模型 → 锁定非 cfg.arm 名单活动关节（含 end 与 mimic 等，锁定位形取零位）→ 缩减模型。

        锁定判据用 cfg 名单而非硬编码关节名集合（型号无关）；所有 link/帧保留，
        保证整臂质量惯量正确。构建后校验 `frame_end` 帧存在与关节同名同序。

        :raises ValueError: URDF 解析失败 / 缩减模型关节与 cfg 名单不符 / frame_end 帧不存在。
        """
        full = pin.buildModelFromUrdf(urdf_path)
        arm_set = set(arm_joint_names)
        lock_ids = [full.getJointId(n) for i, n in enumerate(full.names)
                    if i > 0 and n not in arm_set]
        model = pin.buildReducedModel(full, lock_ids, pin.neutral(full)) if lock_ids else full
        names = [model.names[i] for i in range(1, model.njoints)]
        if names != list(arm_joint_names) or model.nq != len(arm_joint_names):
            raise ValueError(
                f"joyarm.py - JoyArm._build_pin_model：缩减模型关节与 cfg.backend.arm.joints"
                f" 不一致（模型：{names}，nq={model.nq}；cfg：{arm_joint_names}）")
        fid = model.getFrameId(self._frame_end)
        if not 0 <= fid < model.nframes:   # getFrameId 查不到时静默返回越界哨兵，须拦截
            raise ValueError(
                f"joyarm.py - JoyArm._build_pin_model：URDF 中找不到末端帧 "
                f"'{self._frame_end}'（joyarm.frame_end）；可用帧："
                f"{[f.name for f in model.frames]}")
        return model

    # =============== 特征位形解析（§3.1 home 行） ===============
    @staticmethod
    def _parse_home(raw, soft: JointLimits, key: str) -> np.ndarray:
        """解析 home 位形并逐关节校验在软限位 `[q_min, q_max]` 内，越界抛异常。

        :param raw: cfg `joyarm.<key>` 值。
        :param soft: 对应软限位（None 时仅做维度检查，视为空组）。
        :raises ValueError: 维度不符 / 逐关节越界（指明关节、home 值、软限位区间）。
        """
        n = 0 if soft is None else int(np.asarray(soft.q_min).shape[0])
        if raw is None:
            raw = []
        home = np.asarray(raw, dtype=float).reshape(-1)
        if home.shape[0] != n:
            raise ValueError(
                f"joyarm.py - JoyArm._parse_home：joyarm.{key} 维度 {home.shape[0]}"
                f" ≠ 关节数 {n}")
        if soft is None:
            return _freeze_arrays(home)
        q_min = np.asarray(soft.q_min, dtype=float)
        q_max = np.asarray(soft.q_max, dtype=float)
        bad = np.where((home < q_min - 1e-12) | (home > q_max + 1e-12))[0]
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"joyarm.py - JoyArm._parse_home：joyarm.{key} 关节[{i + 1}]（1 基）"
                f"值 {home[i]} 越出软限位区间 [{q_min[i]}, {q_max[i]}]"
                f"（越界关节序号 {bad.tolist()}）")
        return _freeze_arrays(home)

    # =============== tcp_limits 解析（§3.1：先校验后解析） ===============
    @staticmethod
    def _parse_tcp_limits(tl: Optional[dict]) -> TcpLimits:
        """解析 `joyarm.tcp_limits` 段（先格式校验、后经 utils 解析）。

        校验：`workspace_box` 为 (3,2)（每行 `[min,max]`；兼容旧 (2,3) 布局、
        内部统一转置为 (3,2)）且每轴 min < max；`v_lin_max/v_ang_max/f_max/t_max`
        非空且 > 0；违规抛错。

        :raises ValueError: 格式或数值违规（缺 `tcp_limits` 段时回退全 inf 缺省）。
        """
        if not tl:
            return TcpLimits()     # 缺省：不限位（tcp_limits_from_cfg 对缺键回退 inf）
        if not isinstance(tl, dict):
            raise ValueError(
                f"joyarm.py - JoyArm._parse_tcp_limits：joyarm.tcp_limits 需为字典，"
                f"实际 {type(tl).__name__}")
        box = tl.get("workspace_box")
        if box is not None:
            arr = np.asarray(box, dtype=float)
            if arr.shape == (2, 3):     # 旧布局 [min 行, max 行] → 每轴 [min, max]
                arr = arr.T
            if arr.shape != (3, 2):
                raise ValueError(
                    f"joyarm.py - JoyArm._parse_tcp_limits：workspace_box 需为 3 行 2 列"
                    f"（每行 [min, max] 对应 x/y/z），实际 shape={arr.shape}")
            if not np.isfinite(arr).all() or np.any(arr[:, 0] >= arr[:, 1]):
                raise ValueError(
                    f"joyarm.py - JoyArm._parse_tcp_limits：workspace_box 每轴须满足"
                    f" min < max 且为有限值，实际 {arr.tolist()}")
        for key in ("v_lin_max", "v_ang_max", "f_max", "t_max"):
            v = tl.get(key)
            try:
                ok = v is not None and np.isfinite(float(v)) and float(v) > 0
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ValueError(
                    f"joyarm.py - JoyArm._parse_tcp_limits：tcp_limits.{key}={v!r}"
                    f" 非法（须非空且 > 0）")
        return tcp_limits_from_cfg(tl)

    # =============== FSM / 监视 / 到位容差参数解析（§6/§7/§10⑨） ===============
    def _parse_fsm_params(self, jcfg: dict) -> None:
        """解析 `joyarm.MotionFSM` / `joyarm.arrival_tol` / `joyarm.runtime.move_wait_timeout`。

        `fsm_hz` 缺省 50；`monitor_hz` 缺省 = fsm_hz；`mit_kd_damping` 必配
        （正有限数，标量广播或 n 元列表），未配置/非法 → 抛错初始化失败；
        arrival_tol 缺省 0.05 / 1e-3 / 1e-2 标量广播。

        :raises ValueError: `mit_kd_damping` 未配置或非法；其余参数显式给出但非法。
        """
        fsm = jcfg.get("MotionFSM") or {}

        def _pos(key, default, where):
            v = fsm.get(key, default)
            try:
                out = float(v)
            except (TypeError, ValueError):
                out = float("nan")
            if not (np.isfinite(out) and out > 0):
                raise ValueError(
                    f"joyarm.py - JoyArm._parse_fsm_params：{where}.{key}={v!r}"
                    f" 非法（须为正有限数，缺省 {default}）")
            return out

        self._fsm_hz = _pos("fsm_hz", 50.0, "joyarm.MotionFSM")
        self._monitor_hz = fsm.get("monitor_hz", self._fsm_hz)
        try:
            self._monitor_hz = float(self._monitor_hz)
        except (TypeError, ValueError):
            self._monitor_hz = float("nan")
        if not (np.isfinite(self._monitor_hz) and self._monitor_hz > 0):
            raise ValueError(
                f"joyarm.py - JoyArm._parse_fsm_params：joyarm.MotionFSM.monitor_hz="
                f"{fsm.get('monitor_hz')!r} 非法（须为正有限数，缺省 = fsm_hz）")

        # DAMPING 态 MIT kd（必配，§6.2）
        kd_raw = fsm.get("mit_kd_damping")
        if kd_raw is None:
            raise ValueError(
                "joyarm.py - JoyArm._parse_fsm_params：joyarm.MotionFSM.mit_kd_damping"
                " 未配置（正有限数，标量广播或 n 元列表；DAMPING 态必需）")
        try:
            kd = np.asarray(kd_raw, dtype=float)
        except (TypeError, ValueError):
            kd = np.full(1, np.nan)
        if kd.ndim == 0:
            kd = np.full(self._n_arm, float(kd))
        else:
            kd = kd.reshape(-1)
            if kd.shape[0] != self._n_arm:
                raise ValueError(
                    f"joyarm.py - JoyArm._parse_fsm_params：mit_kd_damping 长度 "
                    f"{kd.shape[0]} ≠ arm 关节数 {self._n_arm}")
        if not (np.isfinite(kd).all() and (kd > 0).all()):
            raise ValueError(
                f"joyarm.py - JoyArm._parse_fsm_params：mit_kd_damping={kd_raw!r}"
                f" 非法（须为正有限数，标量广播或 n 元列表）")
        self._mit_kd_damping = _freeze_arrays(kd)

        # 到位容差（§10⑨：缺省 0.05 / 1e-3 / 1e-2 标量广播）
        at = jcfg.get("arrival_tol") or {}

        def _tol_q(part: str, n: int) -> np.ndarray:
            v = (at.get(part) or {}).get("q", 0.05)
            arr = np.asarray(v, dtype=float)
            if arr.ndim == 0:
                arr = np.full(n, float(arr))
            else:
                arr = arr.reshape(-1)
                if arr.shape[0] != n:
                    raise ValueError(
                        f"joyarm.py - JoyArm._parse_fsm_params：arrival_tol.{part}.q 长度 "
                        f"{arr.shape[0]} ≠ 关节数 {n}（或写标量统一全关节）")
            if not (np.isfinite(arr).all() and (arr > 0).all()):
                raise ValueError(
                    f"joyarm.py - JoyArm._parse_fsm_params：arrival_tol.{part}.q={v!r}"
                    f" 非法（须为正有限数）")
            return arr

        self._arrival_tol_q_arm = _freeze_arrays(_tol_q("arm", self._n_arm))
        self._arrival_tol_q_end = _freeze_arrays(_tol_q("end", self._n_end))
        arm_at = at.get("arm") or {}
        self._arrival_tol_pos = float(arm_at.get("pos", 1e-3))
        self._arrival_tol_rot = float(arm_at.get("rot", 1e-2))
        if not (np.isfinite(self._arrival_tol_pos) and self._arrival_tol_pos > 0
                and np.isfinite(self._arrival_tol_rot) and self._arrival_tol_rot > 0):
            raise ValueError(
                f"joyarm.py - JoyArm._parse_fsm_params：arrival_tol.arm.pos/rot="
                f"{arm_at.get('pos')!r}/{arm_at.get('rot')!r} 非法（须为正有限数）")

        rt = jcfg.get("runtime") or {}
        try:
            self._move_wait_timeout = float(rt.get("move_wait_timeout", 10.0))
        except (TypeError, ValueError):
            self._move_wait_timeout = 10.0
        if not (np.isfinite(self._move_wait_timeout) and self._move_wait_timeout > 0):
            logger.warning(
                "joyarm.py - JoyArm._parse_fsm_params：runtime.move_wait_timeout=%r 非法"
                "（须为正有限数），沿用缺省 10.0 s", rt.get("move_wait_timeout"))
            self._move_wait_timeout = 10.0

    # =============== 初始化完成 info（§3.4） ===============
    def _log_init_info(self) -> None:
        """自检通过、成员装配完成后输出初始化完成 info 与配置摘要（§3.4 最低字段集）。"""

        def _r(arr, nd=4):
            return np.round(np.asarray(arr, dtype=float), nd).tolist()

        lines = [
            f"型号：{self._model}",
            f"backend：{str((self._cfg.get('backend') or {}).get('name'))}",
            f"frame_end：{self._frame_end}",
            f"arm 关节（n={self._n_arm}）：{list(self._joint_names_arm)}",
            f"end 关节（n={self._n_end}）：{list(self._joint_names_end)}",
        ]
        for i, jn in enumerate(self._joint_names_arm):
            h, s = self._joint_limits_hard_arm, self._joint_limits_soft_arm
            lines.append(
                f"  {jn}：q 硬[{_r(h.q_min[i], 2)}, {_r(h.q_max[i], 2)}] "
                f"软[{_r(s.q_min[i], 2)}, {_r(s.q_max[i], 2)}] rad；"
                f"dq_max 硬 {_r(h.dq_max[i], 1)} 软 {_r(s.dq_max[i], 1)} rad/s；"
                f"tau_max 硬 {_r(h.tau_max[i], 1)} 软 {_r(s.tau_max[i], 1)} N·m")
        if self._n_end:
            he, se = self._joint_limits_hard_end, self._joint_limits_soft_end
            lines.append(
                f"  end 硬限位 q[{_r(he.q_min, 2)}~{_r(he.q_max, 2)}] "
                f"dq_max {_r(he.dq_max, 1)} tau_max {_r(he.tau_max, 1)}；"
                f"软限位 q[{_r(se.q_min, 2)}~{_r(se.q_max, 2)}] "
                f"dq_max {_r(se.dq_max, 1)} tau_max {_r(se.tau_max, 1)}")
        lines += [
            f"zero_arm：{_r(self._zero_arm)}；neutral_arm：{_r(self._neutral_arm)}；"
            f"home_arm：{_r(self._home_arm)}",
            f"zero_end：{_r(self._zero_end)}；neutral_end：{_r(self._neutral_end)}；"
            f"home_end：{_r(self._home_end)}",
            "激活求解器：" + " ".join(
                f"{d}={self._activated_name_of(d)}" if getattr(self, f"_{d}_solvers")
                else f"{d}=（未配置）" for d in _DOMAIN_REGISTRIES),
            f"plan_hz={self._plan_hz}；ctrl_hz={self._ctrl_hz}",
            f"tcp_limits：workspace_box="
            f"{None if self._tcp_limits.workspace_box is None else _r(self._tcp_limits.workspace_box, 3)}"
            f"；v_lin_max={self._tcp_limits.v_lin_max}；v_ang_max={self._tcp_limits.v_ang_max}"
            f"；f_max={self._tcp_limits.f_max}；t_max={self._tcp_limits.t_max}",
            f"fsm_hz={self._fsm_hz}；monitor_hz={self._monitor_hz}；"
            f"mit_kd_damping={_r(self._mit_kd_damping)}",
        ]
        logger.info("%s 初始化和自检完毕（仅必要校验，不确保完全无误）！\n%s",
                    self._model, "\n".join(lines))

    # ============================================================
    # 三、六域成员管理（内部，§5.1~§5.4）
    # ============================================================
    def _build_domains(self, robotics_cfg: dict) -> None:
        """按 cfg `robotics` 段实例化六域成员并设默认激活（§5.2）。

        前置：cfg 自检③④已通过。构造参数透传（`plan_hz/ctrl_hz` 为管线键，
        由 JoyArm 剥离消费、不入求解器构造器）。域未配置任何成员时对应成员保持
        `None`（该域方法抛 `RuntimeError`）。

        :raises ValueError: 注册名不存在 / 构造参数错 / 实例化异常（列可用注册名）。
        """
        for domain, (cfg_key, registry) in _DOMAIN_REGISTRIES.items():
            specs = _parse_domain_specs(cfg_key, robotics_cfg.get(cfg_key))
            members: dict = {}
            for reg_name, params in specs:
                ctor_params = dict(params)
                if domain == "traj":
                    self._plan_hz_by_name[reg_name] = _check_pacing_hz(
                        cfg_key, reg_name, "plan_hz", ctor_params.pop("plan_hz", None))
                elif domain == "ctrl":
                    self._ctrl_hz_by_name[reg_name] = _check_pacing_hz(
                        cfg_key, reg_name, "ctrl_hz", ctor_params.pop("ctrl_hz", None))
                if reg_name not in registry:
                    raise ValueError(
                        f"joyarm.py - JoyArm._build_domains：注册名『{reg_name}』在 "
                        f"robotics.{cfg_key} 中未找到；可用："
                        f"{sorted(registry) or '无（尚未实现注册）'}")
                try:
                    members[reg_name] = registry[reg_name](**ctor_params)
                except Exception as e:
                    raise ValueError(
                        f"joyarm.py - JoyArm._build_domains：robotics.{cfg_key}"
                        f"『{reg_name}』实例化失败：{e}") from e
            if members:
                setattr(self, f"_{domain}_solvers", members)
                setattr(self, f"_activated_{domain}_solver", next(iter(members.values())))

    def _require_solvers(self, domain: str) -> dict:
        """取某域成员字典；域未配置任何成员时抛 `RuntimeError`（§5.1）。"""
        solvers = getattr(self, f"_{domain}_solvers")
        if not solvers:
            cfg_key = _DOMAIN_REGISTRIES[domain][0]
            raise RuntimeError(
                f"joyarm.py - _require_solvers：robotics.{cfg_key} 域未配置任何求解器，"
                f"该域方法不可用（config robotics.{cfg_key} 段为空）")
        return solvers

    def _activated_of(self, domain: str):
        """取某域当前激活求解器对象；域未配置时抛 `RuntimeError`。"""
        act = getattr(self, f"_activated_{domain}_solver")
        if act is None:
            self._require_solvers(domain)   # 统一抛「域未配置」
        return act

    def _activated_name_of(self, domain: str) -> str:
        """经 `_<域>_solvers` 反查激活对象对应的注册名（§5.3）。"""
        solvers = self._require_solvers(domain)
        act = getattr(self, f"_activated_{domain}_solver")
        for n, obj in solvers.items():
            if obj is act:
                return n
        raise RuntimeError(
            f"joyarm.py - _activated_name_of：{domain} 域激活指针失效（不应发生）")

    def _switch_solver(self, domain: str, name: str) -> bool:
        """热切换统一入口（§5.4 协议按域分发；名称不在成员中 warn 并保持原激活）。"""
        solvers = self._require_solvers(domain)
        if name not in solvers:
            logger.warning(
                "joyarm.py - set_activated_%s_solver：『%s』未在该域已加载成员中"
                "（保持原激活不变）；可用：%s", domain, name, sorted(solvers))
            return False
        new = solvers[name]
        if domain == "traj":
            # 固定时序：① 新规划器先 plan 一次（成功才继续）→ ② 替换激活 → ③ 更新 hz
            self._motion_pause.set()
            try:
                try:
                    ok = bool(new.plan(self, self._target_trajframe))
                except Exception as e:
                    logger.warning(
                        "joyarm.py - set_activated_traj_solver：新规划器『%s』试规划异常"
                        "（保持原激活不变）：%s", name, e)
                    return False
                if not ok:
                    logger.warning(
                        "joyarm.py - set_activated_traj_solver：新规划器『%s』试规划失败"
                        "（保持原激活不变）", name)
                    return False
                self._activated_traj_solver = new
                self._plan_hz = self._plan_hz_by_name[name]
                if self._plan_periodic is not None:
                    self._plan_periodic.set_hz(self._plan_hz)
            finally:
                self._motion_pause.clear()
        elif domain == "ctrl":
            # 替换激活 → MODE 联动（直连 backend，不经公开 set_mode、不受互斥拦截）→ 更新 hz
            self._activated_ctrl_solver = new
            self._backend.set_mode_arm(new.MODE)
            self._ctrl_hz = self._ctrl_hz_by_name[name]
            if self._ctrl_periodic is not None:
                self._ctrl_periodic.set_hz(self._ctrl_hz)
        else:
            setattr(self, f"_activated_{domain}_solver", new)
        return True

    # ============================================================
    # 四、plan/ctrl 两线程（§5.6）与 ctrl 线程互斥豁免
    # ============================================================
    def _start_motion_thread(self) -> None:
        """启动 plan/ctrl 两线程（§5.6 固定时序）。

        激活 traj 求解器 `plan(self, _target_trajframe)` 一次（保证首帧可用）→
        启动 plan 线程（`_plan_hz` 周期 plan）→ 启动 ctrl 线程（`_ctrl_hz` 每拍
        `step(self, sample(time.time()), get_state_arm() 的 JointState)`）→
        清暂停开关。方法名单数 `_thread` 但统一管理两条线程。

        :raises RuntimeError: traj/ctrl 域未配置 / 首帧规划失败 / 旧 plan/ctrl
            线程 stop 超时未确认退出（句柄保留中，防双循环）。
        """
        planner = self._activated_of("traj")
        controller = self._activated_of("ctrl")
        self._stop_motion_thread()                     # 清残留（防御性）
        leftovers = [attr for attr in ("_ctrl_periodic", "_plan_periodic")
                     if getattr(self, attr) is not None]
        if leftovers:                                  # 上次 stop 超时残留：拒绝另起新线程
            raise RuntimeError(
                f"joyarm.py - _start_motion_thread：旧 plan/ctrl 线程 stop 超时"
                f"未确认退出（{', '.join(leftovers)} 句柄保留中），拒绝另起新线程"
                f"（防双循环；MOTION 进入失败，FSM 将重试）")
        self._motion_pause.set()
        try:
            try:
                ok = bool(planner.plan(self, self._target_trajframe))
            except Exception as e:
                raise RuntimeError(
                    f"joyarm.py - _start_motion_thread：首帧规划异常（MOTION 进入失败，"
                    f"FSM 将重试）：{e}") from e
            if not ok:
                raise RuntimeError(
                    "joyarm.py - _start_motion_thread：首帧规划失败（MOTION 进入失败，"
                    "FSM 将重试）")
            self._plan_periodic = _PeriodicThread(self._tick_plan, self._plan_hz,
                                                  name="joyarm-plan")
            self._ctrl_periodic = _PeriodicThread(self._tick_ctrl, self._ctrl_hz,
                                                  name="joyarm-ctrl")
            self._plan_periodic.start()
            self._ctrl_periodic.start()
            self._ctrl_thread = self._ctrl_periodic.thread   # ctrl 线程互斥豁免判定依据
        finally:
            self._motion_pause.clear()
        logger.info("joyarm.py - _start_motion_thread：plan/ctrl 两线程已启动"
                    "（plan_hz=%s，ctrl_hz=%s，控制器 %s）",
                    self._plan_hz, self._ctrl_hz, type(controller).__name__)

    def _stop_motion_thread(self) -> None:
        """停 plan/ctrl 两线程（先 ctrl 后 plan）、超时回收（§5.6；帧缓存不在此清）。

        超时（1 s）未退出的 worker **句柄保留**（不强杀）：`_PeriodicThread`
        内部保留线程引用，本方法同样不清属性句柄与 `_ctrl_thread`（互斥豁免
        判定依据）——旧线程仍在跑期间不得另起新线程（`_start_motion_thread`
        会拒绝，防两个循环同时跑），待其退出后再次调用本方法即完成回收；
        目标帧缓存 `_target_trajframe` 仅在 FSM start/stop 边界清理（§6.5），
        MOTION 态退出不清（保护性往返后续走用户目标）。
        """
        for attr in ("_ctrl_periodic", "_plan_periodic"):
            worker = getattr(self, attr)
            if worker is None:
                continue
            try:
                worker.stop()
            except RuntimeError as e:
                logger.warning("joyarm.py - _stop_motion_thread：%s（句柄保留，"
                               "待其退出后重试）", e)
                continue                       # 超时未退出：保留句柄，不得丢弃
            setattr(self, attr, None)
        if self._ctrl_periodic is None:
            self._ctrl_thread = None           # ctrl 线程确认退出后才清（豁免判定依据）
        self._motion_pause.clear()

    def _tick_plan(self) -> None:
        """plan 线程单步：把 `_target_trajframe` 整体交给激活规划器（§5.6）。"""
        self._activated_traj_solver.plan(self, self._target_trajframe)

    def _tick_ctrl(self) -> None:
        """ctrl 线程单步：读状态 → 采样当前帧 → 控制器 step 经本对象下发（§5.6）。

        暂停开关置位期间（换规划器/启动过渡）不派发；`get_state_arm()` 为 None
        时跳过本拍 step + 节流 warn。
        """
        if self._motion_pause.is_set():
            return
        state = self._backend.get_state_arm()
        if state is None:
            now = time.monotonic()
            if now - self._ctrl_none_warn_last >= _ERR_LOG_INTERVAL:
                logger.warning(
                    "joyarm.py - _tick_ctrl：本拍 get_state_arm() 为 None"
                    "（未连接或读取失败），跳过本拍 step（下拍重试）")
                self._ctrl_none_warn_last = now
            return
        frame = self._activated_traj_solver.sample(time.time())
        self._activated_ctrl_solver.step(self, frame, state)

    def _is_internal_thread(self) -> bool:
        """当前线程是否 JoyArm 自家 ctrl 线程（互斥豁免判定）。"""
        return self._ctrl_thread is not None and threading.current_thread() is self._ctrl_thread

    def _fsm_guard(self, caller: str) -> bool:
        """FSM 外部指令互斥检查（arm 侧指令类方法用）；放行返回 True。

        FSM 运行中，**外部线程**调用 arm 侧指令方法时拦截（空操作 + warn）；
        ctrl 线程自身调用放行（照常委托 backend 下发）。
        """
        if self._is_internal_thread():
            return True
        if self._fsm_running():
            logger.warning(
                "joyarm.py - %s：FSM 运行中，外部线程的 arm 侧指令被互斥拦截"
                "（本次空操作）", caller)
            return False
        return True

    # ============================================================
    # 五、MotionFSM（§6）：线程体 / 单拍循环 / 状态进出与维持
    # ============================================================
    def _fsm_running(self) -> bool:
        """FSM 线程是否在运行。"""
        return self._fsm_periodic is not None and self._fsm_periodic.is_alive()

    def _fsm_worker(self) -> None:
        """FSM 线程体单步（§9：经 `_PeriodicThread` 以 `_fsm_hz` 周期驱动）。"""
        self._fsm_tick()

    def _fsm_tick(self) -> None:
        """FSM 单拍循环（§6.3）：消费事件 → 迁移（含门槛）→ 维持。"""
        # 1) 消费 _safe_event 变化更新目标状态（与 set_event 的同步写幂等，兜底竞态）
        self._target_motion_state = _EVENT_TO_STATE[self._safe_event]
        cur, tgt = self._current_motion_state, self._target_motion_state
        if cur is tgt:
            self._maintain_motion_state(cur)
            return
        nxt = _ROUTE[cur].get(tgt)
        if nxt is None:
            return
        # 迁移门槛：current==HOME 且 target∈{ZERO, MOTION} 须 home 到位（§6.3）；
        # 未到位时继续执行 HOME 维持动作（驱动到位），保护性直达目标不受门槛限制
        if (cur is MotionState.HOME and tgt in (MotionState.ZERO, MotionState.MOTION)
                and not self.is_in_q_arm(self._home_arm)):
            self._maintain_motion_state(cur)
            return
        self._exit_motion_state(cur)
        if not self._enter_motion_state(nxt):
            return     # HOLD 采样失败等，本次迁移推迟、下拍重试
        self._current_motion_state = nxt
        logger.info("joyarm.py - _fsm_tick：状态迁移 %s → %s（目标 %s）",
                    cur.name, nxt.name, tgt.name)

    def _enter_motion_state(self, state: MotionState) -> bool:
        """状态进入（enter）：§6.1 转移时一次检查与设置 + 状态初始化（§6.2 enter 列）。

        :return: `False` 表示本次迁移推迟（如 HOLD 采样失败），下拍重试。
        :raises RuntimeError: 转移时发现未连接（§6.1）。
        """
        if self._backend.is_connected_arm is not True:
            raise RuntimeError(
                f"joyarm.py - _enter_motion_state：转移至 {state.name} 时发现未连接"
                f"（is_connected_arm={self._backend.is_connected_arm}）")
        if state is MotionState.DISABLE:
            self._disable_clear_disable_arm()
            logger.info("joyarm.py - _enter_motion_state：进入 DISABLE（失能→清错→失能）")
        elif state is MotionState.ZERO:
            self._enter_enable_mode(ControlMode.POSITION)
            self._backend.send_position_arm(self._zero_arm)     # enter 发一次（「移动中」子阶段每拍重发）
            self._zero_moving = True
            logger.info("joyarm.py - _enter_motion_state：进入 ZERO（移动至零位后失能维持）")
        elif state is MotionState.HOME:
            self._enter_enable_mode(ControlMode.POSITION)
            self._home_arrived = False
            logger.info("joyarm.py - _enter_motion_state：进入 HOME（空闲保持）")
        elif state is MotionState.MOTION:
            self._hold_q = None                                  # 清理缓存和残留
            self._stop_motion_thread()
            controller = self._activated_of("ctrl")              # 未配置 → enter 失败重试
            self._activated_of("traj")
            self._enter_enable_mode(controller.MODE)              # 设置 mode 为控制器 MODE
            self._start_motion_thread()
            logger.info("joyarm.py - _enter_motion_state：进入 MOTION（plan/ctrl 两线程运行）")
        elif state is MotionState.HOLD:
            self._enter_enable_mode(ControlMode.POSITION)
            state_arm = self._backend.get_state_arm()
            if state_arm is None or state_arm.q is None:
                logger.warning(
                    "joyarm.py - _enter_motion_state：HOLD 采样当前关节位置失败"
                    "（get_state_arm() 为 None），本次迁移推迟下拍重试")
                return False
            self._hold_q = np.asarray(state_arm.q, dtype=float).copy()
            logger.info("joyarm.py - _enter_motion_state：进入 HOLD（原位锁定）")
        elif state is MotionState.DAMPING:
            self._enter_enable_mode(ControlMode.MIT)
            logger.info("joyarm.py - _enter_motion_state：进入 DAMPING（阻尼防护）")
        return True

    def _enter_enable_mode(self, mode: ControlMode) -> None:
        """转移时一次完成的使能与模式设置（§6.1：读标志，未达即设，不多发总线帧）。"""
        if self._backend.is_enabled_arm is not True:
            self._backend.enable_arm()
        if self._backend.get_mode_arm() != mode:
            self._backend.set_mode_arm(mode)

    def _exit_motion_state(self, state: MotionState) -> None:
        """状态退出清理（§6.2 退出清理列）。"""
        if state is MotionState.ZERO:
            self._zero_moving = False     # 清到位轮询 / 子阶段标志
        elif state is MotionState.MOTION:
            self._stop_motion_thread()    # 停线程（帧缓存不在此清，§6.5）
        elif state is MotionState.HOLD:
            self._hold_q = None
        # DISABLE / HOME / DAMPING 无退出清理

    def _maintain_motion_state(self, state: MotionState) -> None:
        """状态维持动作（§6.2 每拍核心动作；硬件操作一律直连 `self._backend`）。"""
        if state is MotionState.DISABLE:
            self._backend.disable_arm()                     # 幂等、无检查无 warn
        elif state is MotionState.ZERO:
            if self._zero_moving:
                # 「移动中」子阶段：每拍重发零位位置指令 + 到位检查（与 HOME 态一致）
                self._backend.send_position_arm(self._zero_arm)
                if self.is_in_q_arm(self._zero_arm):
                    self._backend.disable_arm()
                    self._zero_moving = False
                    logger.info("joyarm.py - _maintain_motion_state：ZERO 已到位并失能，"
                                "转入「零位失能维持」子阶段")
            else:
                self._backend.disable_arm()                 # 零位失能维持（同 DISABLE）
        elif state is MotionState.HOME:
            self._backend.send_position_arm(self._home_arm)
            if not self._home_arrived and self.is_in_q_arm(self._home_arm):
                self._home_arrived = True
                logger.info("joyarm.py - _maintain_motion_state：HOME 已到位")
        elif state is MotionState.MOTION:
            pass    # plan/ctrl 两线程自行维持；运行中热切换求解器按 §5.4
        elif state is MotionState.HOLD:
            if self._hold_q is not None:
                self._backend.send_position_arm(self._hold_q)
        elif state is MotionState.DAMPING:
            self._backend.send_mit_arm(q=np.zeros(self._n_arm),kp=np.zeros(self._n_arm), kd=self._mit_kd_damping)

    def _disable_clear_disable_arm(self) -> None:
        """「失能→清错→失能」序列（§9，DISABLE 态 enter 用；直连 backend）。"""
        self._backend.disable_arm()
        self._backend.clear_error_arm()
        self._backend.disable_arm()

    # ============================================================
    # 六、safe_monitor（§7.3）与状态可视化（§8）线程体
    # ============================================================
    def _monitor_worker(self) -> None:
        """监视器单拍（§7.3）：只读采集 → 规则判断 → 命中则 `set_event`。

        只读：绝不调用控制/下发方法；事件经 §7.2 通路驱动 FSM，天然非阻塞。
        仲裁：内置硬规则优先于自定义；同组多命中取危险程度最高者；
        只升不降：命中事件危险程度高于当前事件才 set_event。
        """
        state_arm = self._backend.get_state_arm()
        state_end = self._backend.get_state_end()
        motion, event = self._current_motion_state, self._safe_event

        hard: List[SafetyEvent] = []
        if state_arm is not None and state_arm.error is not None \
                and np.any(np.asarray(state_arm.error) >= 2):
            hard.append(SafetyEvent.DAMPING)
        if self._backend.is_connected_arm is False:
            hard.append(SafetyEvent.DISABLE)
            # 断连走本独立规则；同步清零「连续 None」计时器——重连后须重新起算
            # 连续 0.1 s 判据，防断连前旧时间戳在重连首个 None 拍立即误触发 DISABLE
            self._monitor_none_since = None
        else:
            # get_state_arm() 连续 0.1 s 返回 None（快照过期/读取失败）→ DISABLE
            if state_arm is None:
                now = time.monotonic()
                if self._monitor_none_since is None:
                    self._monitor_none_since = now
                elif now - self._monitor_none_since >= _MONITOR_NONE_TIMEOUT:
                    hard.append(SafetyEvent.DISABLE)
            else:
                self._monitor_none_since = None

        custom: List[SafetyEvent] = []
        for rule in self._monitor_rules:
            try:
                ev = rule(state_arm, state_end, motion, event)
            except Exception as e:
                logger.warning("joyarm.py - _monitor_worker：自定义规则 %s 抛异常（忽略）：%s",
                               getattr(rule, "__name__", rule), e)
                continue
            if ev is None:
                continue
            if isinstance(ev, SafetyEvent):
                custom.append(ev)
            else:
                logger.warning("joyarm.py - _monitor_worker：自定义规则 %s 返回非 "
                               "SafetyEvent/None（%r），忽略",
                               getattr(rule, "__name__", rule), ev)

        chosen: Optional[SafetyEvent] = None
        if hard:
            chosen = max(hard, key=lambda e: _EVENT_DANGER[e])
        elif custom:
            chosen = max(custom, key=lambda e: _EVENT_DANGER[e])
        # 只升不降：仅当命中事件危险程度高于当前事件才 set_event，
        # 防止覆写并发设置的更高危险事件（外部 set_event 不受限，恢复 NORMAL 走外部通路）
        if chosen is not None and _EVENT_DANGER[chosen] > _EVENT_DANGER[self._safe_event]:
            self.set_event(chosen)

    def _state_viz_worker(self) -> None:
        """可视化单拍（§8）：只读采集刷新面板数据与 meshcat 模型姿态，绝不下发指令。

        未连接（`is_connected_arm` 非 True）时跳过总线读取，面板字段置空
        （避免 backend 逐拍「未连接」告警刷屏）。
        """
        connected = self._backend.is_connected_arm is True
        state_arm = self._backend.get_state_arm() if connected else None
        state_end = self._backend.get_state_end() if connected else None
        snap = {
            "model": self._model,
            "t": round(time.time(), 3),
            "arm": self._viz_group(state_arm, self._joint_names_arm),
            "end": self._viz_group(state_end, self._joint_names_end),
            "tcp": None,
            "fsm": {"current": self._current_motion_state.name,
                    "target": self._target_motion_state.name},
            "event": self._safe_event.name,
            "backend": {
                "connected_arm": self._viz_bool(self._backend.is_connected_arm),
                "enabled_arm": self._viz_bool(self._backend.is_enabled_arm),
                "connected_end": self._viz_bool(self._backend.is_connected_end),
                "enabled_end": self._viz_bool(self._backend.is_enabled_end),
                "mode_arm": None,
                "mode_end": None,
            },
            "traj_enabled": self._viz_traj_enabled,
        }
        try:
            mode = self._backend.get_mode_arm() if connected else None
            snap["backend"]["mode_arm"] = mode.name if mode is not None else None
            mode_end = self._backend.get_mode_end() if connected else None
            snap["backend"]["mode_end"] = mode_end.name if mode_end is not None else None
        except Exception:
            pass
        # 末端位姿（激活 fkine 现算，只读）；速度 = 激活雅可比 × 当前关节速度
        q_now = None if (state_arm is None or state_arm.q is None) \
            else np.asarray(state_arm.q, dtype=float)
        if q_now is not None and self._activated_fkine_solver is not None:
            try:
                pose = self._activated_fkine_solver.solve(self, q_now, self._frame_end)
                snap["tcp"] = {"position": _json_arr(pose.position),
                               "orientation": _json_arr(pose.orientation)}
            except Exception:
                pass
            dq_now = getattr(state_arm, "dq", None)
            if dq_now is not None:
                try:
                    J = self.get_jac(q=q_now)              # (6, n) 只读雅可比
                    v = np.asarray(J, dtype=float) @ np.asarray(dq_now, dtype=float).reshape(-1)
                    if isinstance(snap["tcp"], dict):
                        snap["tcp"]["twist"] = {"linear": _json_arr(v[:3]),
                                                "angular": _json_arr(v[3:])}
                except Exception:
                    pass
        with self._viz_lock:
            self._viz_state = snap
        # meshcat 模型姿态刷新与轨迹绘制（缺包/失败降级，节流告警）
        if self._viz_pin is not None and q_now is not None:
            try:
                self._viz_pin.display(q_now)
                if (self._viz_traj_enabled and self._viz_traj_points > 0
                        and self._current_motion_state is MotionState.MOTION):
                    self._viz_draw_trajectory()
                else:
                    self._viz_hide_trajectory()
            except Exception as e:
                now = time.monotonic()
                if now - self._viz_warn_last >= 5.0:
                    logger.warning(
                        "joyarm.py - _state_viz_worker：meshcat 刷新失败（此后同类失败"
                        "已节流）：%s", e)
                    self._viz_warn_last = now

    @staticmethod
    def _viz_bool(v) -> Optional[bool]:
        """三态布尔转 JSON 安全值。"""
        return None if v is None else bool(v)

    @staticmethod
    def _viz_group(state: Optional[JointState], names: tuple) -> dict:
        """关节状态快照 → 面板分组 dict（JSON 安全；NaN/缺测 → None）。"""
        return {
            "names": list(names),
            "q": _json_arr(getattr(state, "q", None)),
            "dq": _json_arr(getattr(state, "dq", None)),
            "tau": _json_arr(getattr(state, "tau", None)),
            "temp_mos": _json_arr(getattr(state, "temp_mos", None)),
            "temp_rotor": _json_arr(getattr(state, "temp_rotor", None)),
            "error": None if getattr(state, "error", None) is None
            else [int(e) for e in np.asarray(state.error).reshape(-1)],
        }

    def _viz_draw_trajectory(self) -> None:
        """轨迹显示（§8 可选功能）：从激活规划器采样轨迹点 + 当前/目标位姿绘制。

        采样规则：单帧目标取区间 [当前时刻, 目标 t]、多帧目标取 [当前时刻, 末帧 t]；
        取 `_viz_traj_points` 点（start 可配，默认 200）；区间长度 ≤ 0.1 s 不采样
        不绘制。需运动状态机已启动且处于 MOTION 态（由调用方判定）。
        """
        import meshcat
        import meshcat.geometry as mg

        planner = self._activated_traj_solver
        target = self._target_trajframe
        frames = [target] if isinstance(target, TrajFrame) else list(target or [])
        viewer = self._viz_pin.viewer
        if not frames:
            viewer["traj"].delete()
            return
        t_end = frames[-1].time
        t_now = time.time()
        if t_end is None or t_end - t_now <= 0.1:
            viewer["traj"].delete()
            return
        ts = np.linspace(t_now, float(t_end), max(int(self._viz_traj_points), 2))
        fk = self._activated_fkine_solver
        pts = []
        for t in ts:
            frame = planner.sample(float(t))
            if frame is None or frame.q is None:
                continue
            pts.append(np.asarray(
                fk.solve(self, frame.q, self._frame_end).position, dtype=float))
        if len(pts) >= 2:
            viewer["traj"].set_object(
                mg.Line(mg.PointsGeometry(position=np.asarray(pts, dtype=np.float32))))
        # frame_end 当前位姿与目标位姿标记（球）
        q_cur = self._backend.get_state_arm()
        if q_cur is not None and q_cur.q is not None:
            p_cur = np.asarray(
                fk.solve(self, q_cur.q, self._frame_end).position, dtype=float)
            viewer["traj_cur"].set_object(mg.Sphere(0.012))
            viewer["traj_cur"].set_transform(
                np.vstack([np.hstack([np.eye(3), p_cur.reshape(3, 1)]),
                           np.array([0.0, 0.0, 0.0, 1.0])]))
        tgt_pose = frames[-1].pose
        if tgt_pose is not None:
            p_tgt = np.asarray(tgt_pose.position, dtype=float)
            viewer["traj_tgt"].set_object(mg.Sphere(0.012))
            viewer["traj_tgt"].set_transform(
                np.vstack([np.hstack([np.eye(3), p_tgt.reshape(3, 1)]),
                           np.array([0.0, 0.0, 0.0, 1.0])]))

    def _viz_hide_trajectory(self) -> None:
        """清除 meshcat 上的轨迹显示。"""
        if self._viz_pin is None:
            return
        try:
            viewer = self._viz_pin.viewer
            for key in ("traj", "traj_cur", "traj_tgt"):
                viewer[key].delete()
        except Exception:
            pass

    # ============================================================
    # 七、公开接口区——按「拿到一台机械臂后怎么用」的顺序分类
    # ============================================================

    # ============================================================
    # 7.1 配置与基本信息：get_config / 只读 @property（§3.1）
    # ============================================================
    def get_config(self) -> dict:
        """返回当前 config 配置的深拷贝（教学数据如 MDH 参数的读取口，约束8）。"""
        return copy.deepcopy(self._cfg)

    @property
    def pin_model(self) -> pin.Model:
        """pinocchio 缩减构型模型（`nq = n_arm`，只读共享）。"""
        return self._pin_model

    @property
    def n_arm(self) -> int:
        """本体关节数。"""
        return self._n_arm

    @property
    def n_end(self) -> int:
        """末端电机数（无 end 段为 0）。"""
        return self._n_end

    @property
    def joint_limits_hard_arm(self) -> Optional[JointLimits]:
        """本体硬限位（backend 下发越限裁剪的唯一依据）。

        只读引用：构造后内部数组已置 `writeable=False`，返本体零拷贝，外部原地改抛异常。
        """
        return self._joint_limits_hard_arm

    @property
    def joint_limits_hard_end(self) -> Optional[JointLimits]:
        """末端硬限位（backend 下发越限裁剪的唯一依据；无 end 段为 `None`）。

        只读引用：构造后内部数组已置 `writeable=False`，返末端零拷贝，外部原地改抛异常。
        """
        return self._joint_limits_hard_end

    @property
    def joint_limits_soft_arm(self) -> Optional[JointLimits]:
        """本体软限位（上层状态判断依据，不拦指令；只读引用）。"""
        return self._joint_limits_soft_arm

    @property
    def joint_limits_soft_end(self) -> Optional[JointLimits]:
        """末端软限位（上层状态判断依据，不拦指令；无 end 段为 `None`；只读引用）。"""
        return self._joint_limits_soft_end

    @property
    def zero_arm(self) -> np.ndarray:
        """本体零位 `(n,)`（全零，编码器标零基准；只读引用）。"""
        return self._zero_arm

    @property
    def zero_end(self) -> np.ndarray:
        """末端零位 `(n_end,)`（全零，编码器标零基准；只读引用）。"""
        return self._zero_end

    @property
    def neutral_arm(self) -> np.ndarray:
        """本体数值求解默认初值 `(n,)`（软限位逐关节中点；只读引用）。"""
        return self._neutral_arm

    @property
    def neutral_end(self) -> np.ndarray:
        """末端数值求解默认初值 `(n_end,)`（软限位中点；只读引用）。"""
        return self._neutral_end

    @property
    def home_arm(self) -> np.ndarray:
        """本体 home 位形 `(n,)`（cfg `joyarm.home_arm`；只读引用）。"""
        return self._home_arm

    @property
    def home_end(self) -> np.ndarray:
        """末端 home 位形 `(n_end,)`（cfg `joyarm.home_end`；只读引用）。"""
        return self._home_end

    @property
    def tcp_limits(self) -> TcpLimits:
        """末端空间限位（只读引用，内部数组已只读化）。"""
        return self._tcp_limits

    @property
    def plan_hz(self) -> Optional[float]:
        """激活规划器的规划线程频率 Hz（切换求解器时更新；traj 域未配置为 `None`）。"""
        return self._plan_hz

    @property
    def ctrl_hz(self) -> Optional[float]:
        """激活控制器的控制线程频率 Hz（切换求解器时更新；ctrl 域未配置为 `None`）。"""
        return self._ctrl_hz

    def __repr__(self) -> str:
        act = {d: (self._activated_name_of(d) if getattr(self, f"_{d}_solvers") else "-")
               for d in _DOMAIN_REGISTRIES}
        return (f"<{type(self).__name__} model={self._model!r} n_arm={self._n_arm} "
                f"n_end={self._n_end} frame_end={self._frame_end!r} "
                f"solvers={act} fsm={self._current_motion_state.name}>")

    # ============================================================
    # 7.2 关节查询与随机采样（§3.5）
    # ============================================================
    def get_joint_names_arm(self, joint: Union[int, str] = "all") -> Optional[Union[str, tuple]]:
        """查本体关节名。

        :param joint: `"all"`（默认）返回内部关节名 tuple 本体（cfg 顺序 = q 向量顺序）；
            传入 1 基序号（cfg 列表位置）返回该关节名。
        :return: 关节名或名 tuple；越界 `[1, n]` 或类型非法 warn 并返回 `None`。
        """
        return self._get_joint_names(self._joint_names_arm, joint, "arm")

    def get_joint_names_end(self, joint: Union[int, str] = "all") -> Optional[Union[str, tuple]]:
        """查末端电机名。

        :param joint: `"all"`（默认）返回内部电机名 tuple 末端（cfg 顺序 = q 向量顺序）；
            传入 1 基序号（cfg 列表位置）返回该电机名。
        :return: 电机名或名 tuple；越界 `[1, n_end]` 或类型非法 warn 并返回 `None`。
        """
        return self._get_joint_names(self._joint_names_end, joint, "end")

    @staticmethod
    def _get_joint_names(names: tuple, joint, part: str):
        """关节名查询实现（1 基序号口径，§3.5）。"""
        if isinstance(joint, str) and joint == "all":
            return names
        if isinstance(joint, (int, np.integer)) and not isinstance(joint, bool) \
                and 1 <= joint <= len(names):
            return names[int(joint) - 1]
        logger.warning("joyarm.py - get_joint_names_%s：joint=%r 非法（须为 \"all\" 或 "
                       "1 基序号 ∈ [1, %d]），返回 None", part, joint, len(names))
        return None

    def rand_q_arm(self, joint_limits: Optional[JointLimits] = None) -> np.ndarray:
        """本体关节角随机采样 `(n,)`：在 `[q_min, q_max]` 均匀采样。

        :param joint_limits: 采样限位；缺省 `None` 在方法体内回退**软限位**
            （签名默认值不能引用 self）。仅此单参数。
        """
        limits = joint_limits if joint_limits is not None else self._joint_limits_soft_arm
        return rand_within_limits(limits)

    def rand_q_end(self, joint_limits: Optional[JointLimits] = None) -> np.ndarray:
        """末端电机角随机采样 `(n_end,)`：在 `[q_min, q_max]` 均匀采样。

        :param joint_limits: 采样限位；缺省 `None` 在方法体内回退**软限位**
            （签名默认值不能引用 self）。仅此单参数。无 end 段型号（`n_end=0`，
            软限位不存在）返回空数组 `(0,)`。
        """
        limits = joint_limits if joint_limits is not None else self._joint_limits_soft_end
        if limits is None:
            return np.zeros(self._n_end)
        return rand_within_limits(limits)

    # ============================================================
    # 7.3 硬件后端委托面（§4）：连接/使能/参数/模式/状态/零点与错误/指令/动作
    # ============================================================

    # =============== 连接与断开（§4.2） ===============
    def connect_arm(self, channel: Optional[str] = None,
                    protocol: Optional[str] = None) -> Optional[bool]:
        """连接 arm 硬件总线。

        透传可选参（缺省用 backend cfg 成员值，显式传入则同步更新）。backend 副作用：
        连接成功后自动设 cfg `default_mode` 并等 `write_settle` 秒生效。

        :return: `is_connected_arm`（True 成功 / False 失败 / None 意外）。
        """
        self._backend.connect_arm(channel, protocol)
        return self._backend.is_connected_arm

    def connect_end(self, channel: Optional[str] = None,
                    protocol: Optional[str] = None) -> Optional[bool]:
        """连接 end 硬件总线。

        透传可选参（缺省用 backend cfg 成员值，显式传入则同步更新）。backend 副作用：
        连接成功后自动设 cfg `default_mode` 并等 `write_settle` 秒生效。

        :return: `is_connected_end`（True 成功 / False 失败 / None 意外）。
        """
        self._backend.connect_end(channel, protocol)
        return self._backend.is_connected_end

    def disconnect_arm(self) -> None:
        """断开 arm 硬件总线（backend 副作用：先尽力失能再断总线、断开后清模式缓存）。"""
        self._backend.disconnect_arm()

    def disconnect_end(self) -> None:
        """断开 end 硬件总线（backend 副作用：先尽力失能再断总线、断开后清模式缓存）。"""
        self._backend.disconnect_end()

    @property
    def is_connected_arm(self) -> Optional[bool]:
        """arm 连接状态（True / False / None 未知；与 backend 同名直连）。"""
        return self._backend.is_connected_arm

    @property
    def is_connected_end(self) -> Optional[bool]:
        """end 连接状态（True / False / None 未知；与 backend 同名直连）。"""
        return self._backend.is_connected_end

    # =============== 使能与失能（§4.2；arm 侧受 FSM 互斥） ===============
    def enable_arm(self) -> int:
        """使能 arm 全部关节电机。

        :return: 1 = 全部成功、0 = 任一失败或未连接（失败细节经日志 warn，不抛；
            结果另经 `is_enabled_arm` 核实）。FSM 运行中被外部线程调用时空操作并
            warn（FSM 互斥拦截时返回 0）。
        """
        if not self._fsm_guard("enable_arm"):
            return 0
        return self._backend.enable_arm()

    def enable_end(self) -> int:
        """使能 end 全部电机。

        :return: 1 = 全部成功、0 = 任一失败或未连接（失败细节经日志 warn，不抛；
            结果另经 `is_enabled_end` 核实）。end 侧不受互斥（FSM 互斥仅约束 arm）。
        """
        return self._backend.enable_end()

    def disable_arm(self) -> int:
        """失能 arm 全部关节电机（FSM 运行中被外部线程调用时空操作并 warn，互斥拦截时返回 0）。"""
        if not self._fsm_guard("disable_arm"):
            return 0
        return self._backend.disable_arm()

    def disable_end(self) -> int:
        """失能 end 全部电机。

        :return: 1 = 全部成功、0 = 任一失败或未连接（失败细节经日志 warn，不抛）。
            end 侧不受互斥（FSM 互斥仅约束 arm）。
        """
        return self._backend.disable_end()

    @property
    def is_enabled_arm(self) -> Optional[bool]:
        """arm 使能状态（按各关节 error 码推导：全 1→True、全 0→False、混合/含 ≥2→None，可自愈）。"""
        return self._backend.is_enabled_arm

    @property
    def is_enabled_end(self) -> Optional[bool]:
        """end 使能状态（按各电机 error 码推导：全 1→True、全 0→False、混合/含 ≥2→None，可自愈）。"""
        return self._backend.is_enabled_end

    # =============== 参数读写（§4.3；不受 FSM 互斥） ===============
    def read_param_arm(self, key: str) -> Optional[list]:
        """读 arm 全部关节的电机参数（逐关节值列表；任一失败/未连接返回 `None`）。"""
        return self._backend.read_param_arm(key)

    def read_param_end(self, key: str) -> Optional[list]:
        """读 end 全部电机的电机参数（逐电机值列表；任一失败/未连接返回 `None`）。"""
        return self._backend.read_param_end(key)

    def write_param_arm(self, key: str, values: Sequence[float]) -> int:
        """写 arm 全部关节的电机参数（`values` 长度 = 关节数且全有限；成功 1、失败 0）。"""
        return self._backend.write_param_arm(key, values)

    def write_param_end(self, key: str, values: Sequence[float]) -> int:
        """写 end 全部电机的电机参数（`values` 长度 = 电机数且全有限；成功 1、失败 0）。"""
        return self._backend.write_param_end(key, values)

    # =============== 控制模式（§4.4；set 受 FSM 互斥） ===============
    def get_mode_arm(self) -> Optional[ControlMode]:
        """读 arm 全组控制模式（全组一致才返回该模式；不一致/失败/未连接返回 `None`）。"""
        return self._backend.get_mode_arm()

    def get_mode_end(self) -> Optional[ControlMode]:
        """读 end 全组控制模式（全组一致才返回该模式；不一致/失败/未连接返回 `None`）。"""
        return self._backend.get_mode_end()

    def set_mode_arm(self, mode: ControlMode = ControlMode.MIT) -> int:
        """设置 arm 全组控制模式（MIT / POSITION / VELOCITY）。

        入参须为 `ControlMode` 枚举（非枚举直接返回 0）；成功返回 1 并**乐观更新**
        模式缓存（发送门禁立即放行），硬件一致性经 `get_mode_arm` 读回纠正。切
        POSITION / VELOCITY 自动随写 POS_VEL 四增益（缺配置跳过）。FSM 运行中被
        外部线程调用时空操作 + warn（FSM 互斥拦截时返回 0）。
        """
        if not self._fsm_guard("set_mode_arm"):
            return 0
        return self._backend.set_mode_arm(mode)

    def set_mode_end(self, mode: ControlMode = ControlMode.MIT) -> int:
        """设置 end 全组控制模式（MIT / POSITION / VELOCITY）。

        入参须为 `ControlMode` 枚举（非枚举直接返回 0）；成功返回 1 并**乐观更新**
        模式缓存（发送门禁立即放行），硬件一致性经 `get_mode_end` 读回纠正。切
        POSITION / VELOCITY 自动随写 POS_VEL 四增益（缺配置跳过）。end 侧不受
        互斥（FSM 互斥仅约束 arm）。
        """
        return self._backend.set_mode_end(mode)

    # =============== 状态读取（§4.5；不受 FSM 互斥） ===============
    def get_state_arm(self) -> Optional[JointState]:
        """读 arm 关节状态快照（读时刷新；`t` = 本批数据帧最早到达时刻 Unix 秒）。

        任一关节读取失败/数据异常时跳过本轮更新返回 `None` 并保持上次快照；
        backend 低频刷新线程持续维护快照。`error` 码约定：0=失能、1=使能、≥2=故障。
        """
        return self._backend.get_state_arm()

    def get_state_end(self) -> Optional[JointState]:
        """读 end 电机状态快照（读时刷新；`t` = 本批数据帧最早到达时刻 Unix 秒）。

        任一电机读取失败/数据异常时跳过本轮更新返回 `None` 并保持上次快照；
        backend 低频刷新线程持续维护快照。`error` 码约定：0=失能、1=使能、≥2=故障。
        """
        return self._backend.get_state_end()

    # =============== 零点与错误（§4.6；set/clear 受 FSM 互斥） ===============
    def set_zero_arm(self) -> int:
        """将 arm 当前位置记为零点（影响后续 q 读数基准；1=全部成功、0=任一失败）。"""
        if not self._fsm_guard("set_zero_arm"):
            return 0
        return self._backend.set_zero_arm()

    def set_zero_end(self) -> int:
        """将 end 当前位置记为零点（影响后续 q 读数基准；1=全部成功、0=任一失败）。"""
        return self._backend.set_zero_end()

    def get_error_arm(self) -> Optional[list]:
        """读 arm 各关节状态码缓存（快照过期、未连接或 error 段缺失返回 `None`）。"""
        return self._backend.get_error_arm()

    def get_error_end(self) -> Optional[list]:
        """读 end 各电机状态码缓存（快照过期、未连接或 error 段缺失返回 `None`）。"""
        return self._backend.get_error_end()

    def check_error_arm(self) -> bool:
        """检查 arm 状态码（全 0/1 判正常 True；含 ≥2 / 未连接 / 快照过期 False）。"""
        return self._backend.check_error_arm()

    def check_error_end(self) -> bool:
        """检查 end 状态码（全 0/1 判正常 True；含 ≥2 / 未连接 / 快照过期 False）。"""
        return self._backend.check_error_end()

    def clear_error_arm(self) -> bool:
        """清除 arm 各关节硬件错误（逐关节发清错指令、无读回验证；受 FSM 互斥，拦截时返回 False）。"""
        if not self._fsm_guard("clear_error_arm"):
            return False
        return self._backend.clear_error_arm()

    def clear_error_end(self) -> bool:
        """清除 end 各电机硬件错误（逐电机发清错指令、无读回验证；end 侧不受互斥）。"""
        return self._backend.clear_error_end()

    # =============== 关节指令下发 ===============
    def send_mit_arm(self, q, dq=None, tau=None, kp=None, kd=None) -> None:
        """arm MIT 指令整组下发。

        `q` 必填 `(n,)` rad；`dq`/`tau` 缺省 `zeros(n)`；`kp`/`kd` 缺省取 `_kp/_kd_mit_default_arm`。
        FSM 运行中被外部线程调用时空操作 + warn。
        """
        if not self._fsm_guard("send_mit_arm"):
            return
        self._backend.send_mit_arm(q, dq, tau, kp, kd)

    def send_mit_end(self, q, dq=None, tau=None, kp=None, kd=None) -> None:
        """end MIT 指令整组下发。

        `q` 必填 `(n_end,)` rad；`dq`/`tau` 缺省 `zeros(n_end)`；`kp`/`kd` 缺省取
        backend `_kp/_kd_mit_default_end`。end 侧不受互斥。
        """
        self._backend.send_mit_end(q, dq, tau, kp, kd)

    def send_position_arm(self, q, vlim=None, flim=None) -> None:
        """arm 位置指令整组下发。

        :param q: 位置目标 `(n,)` rad；越硬限位就近裁剪 + warn。
        :param vlim: 速度上限 rad/s；缺省取 cfg `POS_VEL.vlim`，超 `[0, dq_max]` 裁剪。
        :param flim: 归一化力矩电流上限 0~1；缺省取 cfg `POS_VEL.flim`。
        FSM 运行中被外部线程调用时空操作 + warn（ctrl 线程自身豁免放行）。
        """
        if not self._fsm_guard("send_position_arm"):
            return
        self._backend.send_position_arm(q, vlim, flim)

    def send_position_end(self, q, vlim=None, flim=None) -> None:
        """end 位置指令整组下发。

        :param q: 位置目标 `(n_end,)` rad；越硬限位就近裁剪 + warn。
        :param vlim: 速度上限 rad/s；缺省取 cfg `POS_VEL.vlim`，超 `[0, dq_max]` 裁剪。
        :param flim: 归一化力矩电流上限 0~1；缺省取 cfg `POS_VEL.flim`。
        end 侧不受互斥（FSM 互斥仅约束 arm）。
        """
        self._backend.send_position_end(q, vlim, flim)

    def send_vel_arm(self, dq) -> None:
        """arm 速度指令整组下发（`dq` `(n,)` rad/s；幅值越 `|dq_max|` 裁剪；受 FSM 互斥）。"""
        if not self._fsm_guard("send_vel_arm"):
            return
        self._backend.send_vel_arm(dq)

    def send_vel_end(self, dq) -> None:
        """end 速度指令整组下发（`dq` `(n_end,)` rad/s；幅值越 `|dq_max|` 裁剪；end 侧不受互斥）。"""
        self._backend.send_vel_end(dq)

    # =============== 末端离散动作（§4.8；end 侧不受互斥） ===============
    def set_action_end(self, action: str, **kw) -> None:
        """末端离散动作（纯透传 `backend.send_action_end`，连接校验与分发由 backend 完成）。

        未知动作或缺 `position` 参数：backend 抛 `ValueError` 经基类转 warn
        ——本层可观察行为为 warn 不抛。
        """
        self._backend.send_action_end(action, **kw)

    # ============================================================
    # 7.4 求解器六域管理（§5.3/§5.4）：查询 / 切换（×6 域）
    # ============================================================
    def get_fkine_solvers(self) -> List[str]:
        """fkine 域已加载注册名列表（深拷贝）。"""
        return copy.deepcopy(sorted(self._require_solvers("fkine")))

    def set_activated_fkine_solver(self, name: str) -> bool:
        """切换 fkine 域激活求解器（名称不存在 warn 并保持原激活；traj/ctrl 域另有切换协议 §5.4，ikine/jac/dyn 直接换指针）。"""
        return self._switch_solver("fkine", name)

    def get_activated_fkine_solver(self) -> str:
        """fkine 域当前激活求解器注册名。"""
        return self._activated_name_of("fkine")

    def get_ikine_solvers(self) -> List[str]:
        """ikine 域已加载注册名列表（深拷贝）。"""
        return copy.deepcopy(sorted(self._require_solvers("ikine")))

    def set_activated_ikine_solver(self, name: str) -> bool:
        """切换 ikine 域激活求解器。"""
        return self._switch_solver("ikine", name)

    def get_activated_ikine_solver(self) -> str:
        """ikine 域当前激活求解器注册名。"""
        return self._activated_name_of("ikine")

    def get_jac_solvers(self) -> List[str]:
        """jacobian 域已加载注册名列表（深拷贝）。"""
        return copy.deepcopy(sorted(self._require_solvers("jac")))

    def set_activated_jac_solver(self, name: str) -> bool:
        """切换 jacobian 域激活求解器。"""
        return self._switch_solver("jac", name)

    def get_activated_jac_solver(self) -> str:
        """jacobian 域当前激活求解器注册名。"""
        return self._activated_name_of("jac")

    def get_dyn_solvers(self) -> List[str]:
        """dynamics 域已加载注册名列表（深拷贝）。"""
        return copy.deepcopy(sorted(self._require_solvers("dyn")))

    def set_activated_dyn_solver(self, name: str) -> bool:
        """切换 dynamics 域激活求解器。"""
        return self._switch_solver("dyn", name)

    def get_activated_dyn_solver(self) -> str:
        """dynamics 域当前激活求解器注册名。"""
        return self._activated_name_of("dyn")

    def get_traj_solvers(self) -> List[str]:
        """traj 域已加载注册名列表（深拷贝）。"""
        return copy.deepcopy(sorted(self._require_solvers("traj")))

    def set_activated_traj_solver(self, name: str) -> bool:
        """切换 traj 域激活规划器（§5.4 固定时序：新规划器先 plan 成功才替换）。"""
        return self._switch_solver("traj", name)

    def get_activated_traj_solver(self) -> str:
        """traj 域当前激活规划器注册名。"""
        return self._activated_name_of("traj")

    def get_ctrl_solvers(self) -> List[str]:
        """control 域已加载注册名列表（深拷贝）。"""
        return copy.deepcopy(sorted(self._require_solvers("ctrl")))

    def set_activated_ctrl_solver(self, name: str) -> bool:
        """切换 control 域激活控制器（替换指针 → MODE 联动电机模式 → 更新 ctrl 线程频率）。"""
        return self._switch_solver("ctrl", name)

    def get_activated_ctrl_solver(self) -> str:
        """control 域当前激活控制器注册名。"""
        return self._activated_name_of("ctrl")

    # ============================================================
    # 7.5 目标帧通道（§5.5）：set/get + 上层便捷封装
    # ============================================================
    def set_target_trajframe(self, target) -> None:
        """写入轨迹目标（单帧或帧序列；`None` / 空 = 清空目标）。

        `copy.deepcopy` 后单条赋值语句整体原子替换引用（规划线程每拍整体重读，
        无锁原子；发布后外部原地修改不影响内部）。

        :param target: :class:`TrajFrame` 单帧、帧列表，或 `None`/空列表（清空）。
        :raises ValueError: 序列元素非 :class:`TrajFrame`。
        """
        if target is None:
            frames = None
        elif isinstance(target, TrajFrame):
            frames = copy.deepcopy(target)
        else:
            seq = list(target)
            if not seq:
                frames = None
            else:
                for fr in seq:
                    if not isinstance(fr, TrajFrame):
                        raise ValueError(
                            f"joyarm.py - set_target_trajframe：目标序列含非 TrajFrame "
                            f"元素（{type(fr).__name__}）")
                frames = copy.deepcopy(seq)
        self._target_trajframe = frames

    def get_target_trajframe(self):
        """读当前轨迹目标（只读引用；规划期间请勿原地修改）。"""
        return self._target_trajframe

    def set_target_joint(self, t, q, dq=None, ddq=None, tau=None) -> None:
        """单帧关节目标便捷封装（内部组帧后走 :meth:`set_target_trajframe` 同一原子通道）。

        :param t: 目标到达时刻，Unix 秒标量（必填）。
        :param q: 关节位置目标 `(n,)` rad（必填；list 或 ndarray）。
        :param dq: 关节速度目标 rad/s（可选）；:param ddq: 加速度 rad/s²（可选）；
            :param tau: 前馈力矩 N·m（可选）。
        :raises ValueError: `t`/`q` 缺失、维度不符 `(n,)`。
        """
        q = self._check_target_vector(q, self._n_arm, "q", required=True)
        dq = self._check_target_vector(dq, self._n_arm, "dq", required=False)
        ddq = self._check_target_vector(ddq, self._n_arm, "ddq", required=False)
        tau = self._check_target_vector(tau, self._n_arm, "tau", required=False)
        self.set_target_trajframe(TrajFrame(time=self._check_target_time(t),
                                            q=q, dq=dq, ddq=ddq, tau=tau))

    def set_target_cart(self, t, pose, twist=None, accel=None, wrench=None) -> None:
        """单帧笛卡尔目标便捷封装（内部组帧后走 :meth:`set_target_trajframe` 同一原子通道）。

        :param t: 目标到达时刻，Unix 秒标量（必填）。
        :param pose: 末端位姿目标（必填；:class:`Pose` 或 `(7,)` list/ndarray
            `[x, y, z, qw, qx, qy, qz]`）。
        :param twist: 末端速度（可选；:class:`Twist` 或 `(6,)` `[vx,vy,vz,wx,wy,wz]`）；
            :param accel: 加速度（:class:`Accel` 或 `(6,)`）；
            :param wrench: 力/力矩（:class:`Wrench` 或 `(6,)` `[fx,fy,fz,tx,ty,tz]`）。
        :raises ValueError: 必填缺失、类型/维度不符。
        """
        if isinstance(pose, Pose):
            pose_out = copy.deepcopy(pose)
        else:
            arr = self._check_target_vector(pose, 7, "pose", required=True)
            pose_out = Pose(position=np.array(arr[:3], dtype=float),
                            orientation=np.array(arr[3:], dtype=float))
        twist_out = self._check_target_cart6(twist, Twist, "twist")
        accel_out = self._check_target_cart6(accel, Accel, "accel")
        wrench_out = self._check_target_cart6(wrench, Wrench, "wrench")
        self.set_target_trajframe(TrajFrame(time=self._check_target_time(t),
                                            pose=pose_out, twist=twist_out,
                                            accel=accel_out, wrench=wrench_out))

    @staticmethod
    def _check_target_time(t) -> float:
        """目标时刻校验（Unix 秒标量）。"""
        if isinstance(t, bool) or not isinstance(t, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"joyarm.py - set_target：t 需为 Unix 秒标量，实际 {type(t).__name__}")
        return float(t)

    @staticmethod
    def _check_target_vector(v, n: int, key: str, required: bool):
        """便捷封装的 `(n,)` 向量校验（None 且非必填 → None；否则转 float 数组并验维）。"""
        if v is None:
            if required:
                raise ValueError(f"joyarm.py - set_target：{key} 必填（None）")
            return None
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(
                f"joyarm.py - set_target：{key} 维度 {arr.shape[0]} ≠ 期望 {n}")
        return arr

    @classmethod
    def _check_target_cart6(cls, v, cls6, key: str):
        """便捷封装的笛卡尔 6 维量校验（类型对象或 `(6,)` 序列 → 对应数据类）。"""
        if v is None:
            return None
        if isinstance(v, cls6):
            return copy.deepcopy(v)
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.shape[0] != 6:
            raise ValueError(
                f"joyarm.py - set_target：{key} 需为 {cls6.__name__} 或 (6,) 序列，"
                f"实际长度 {arr.shape[0]}")
        if cls6 is Wrench:
            return Wrench(force=np.array(arr[:3]), torque=np.array(arr[3:]))
        return cls6(linear=np.array(arr[:3]), angular=np.array(arr[3:]))

    # ============================================================
    # 7.6 数学门面 9 方法（§5.7）：显式传参离线可用，缺省取硬件状态
    # ============================================================
    def _current_q(self, caller: str) -> np.ndarray:
        """取 arm 当前 q（缺省值统一入口；无可用状态抛 `RuntimeError`）。"""
        st = self._backend.get_state_arm()
        if st is None or st.q is None:
            raise RuntimeError(
                f"joyarm.py - {caller}：get_state_arm() 无可用 q（未连接或读取失败）；"
                f"显式传参可离线计算")
        return np.asarray(st.q, dtype=float)

    def _current_qdq(self, caller: str) -> tuple:
        """取 arm 当前 (q, dq)（无可用状态抛 `RuntimeError`）。"""
        st = self._backend.get_state_arm()
        if st is None or st.q is None or st.dq is None:
            raise RuntimeError(
                f"joyarm.py - {caller}：get_state_arm() 无可用 q/dq（未连接或读取失败）；"
                f"显式传参可离线计算")
        return np.asarray(st.q, dtype=float), np.asarray(st.dq, dtype=float)

    def solve_fkine(self, q=None, frame=None) -> Pose:
        """正运动学：`q` → `frame` 帧位姿（返回 `Pose`）。

        :param q: `(n,)` 关节角 rad；缺省取 `get_state_arm().q`（无状态抛 `RuntimeError`）。
        :param frame: 帧名；缺省取 cfg `joyarm.frame_end`。
        """
        if q is None:
            q = self._current_q("solve_fkine")
        return self._activated_of("fkine").solve(self, np.asarray(q, dtype=float),
                                                 frame or self._frame_end)

    def solve_ikine(self, pose, frame=None, q0=None, **kw) -> IKResult:
        """逆运动学单解：目标位姿 → 关节角（数值法自 `q0` 迭代；解析法选最近解），返回 :class:`IKResult`（q 为 `(n,)` 单解，无可行解时 success=False）。

        :param pose: 目标位姿 :class:`Pose`。
        :param frame: 目标帧名；缺省取 cfg `joyarm.frame_end`。
        :param q0: `(n,)` 参考关节角；缺省当前状态 q（无状态抛 `RuntimeError`）。
        :param kw: 求解器特有参数（`tol` / `iters` 等，无法识别的忽略）。
        """
        if q0 is None:
            q0 = self._current_q("solve_ikine")
        return self._activated_of("ikine").solve(self, pose, frame or self._frame_end,
                                                 np.asarray(q0, dtype=float), **kw)

    def solve_ikine_all(self, pose, frame=None, **kw):
        """逆运动学全部解（`q` 为 `(K, n)`；仅解析法实现）。

        :raises RuntimeError: 激活求解器未实现 `solve_all`（数值法——提示换解析法）。
        """
        try:
            return self._activated_of("ikine").solve_all(self, pose, frame=frame, **kw)
        except NotImplementedError as e:
            raise RuntimeError(
                f"joyarm.py - solve_ikine_all：当前激活的 ikine 求解器不支持求全部解"
                f"（数值法/未实现 solve_all，请切换解析法实现）：{e}") from e

    def get_jac(self, q=None, frame=None) -> np.ndarray:
        """雅可比矩阵 `J(q)`（`(6, n)` 基座系）。"""
        if q is None:
            q = self._current_q("get_jac")
        return self._activated_of("jac").jac(self, np.asarray(q, dtype=float),
                                             frame or self._frame_end)

    def get_jac_pinv_damped(self, q=None, frame=None, damping: float = 1e-3) -> np.ndarray:
        """雅可比的阻尼最小二乘广义逆 `J*`（`(n, 6)`；`J* = Jᵀ(JJᵀ + λ²I)⁻¹`）。"""
        if q is None:
            q = self._current_q("get_jac_pinv_damped")
        return self._activated_of("jac").jac_pinv_damped(self, np.asarray(q, dtype=float),
                                                         frame or self._frame_end, damping)

    def solve_idyn(self, q=None, dq=None, ddq=None) -> np.ndarray:
        """逆动力学（RNEA）：`(q, dq, ddq)` → 各关节力矩 `(n,)` N·m。

        `q`/`dq` 缺省取 `get_state_arm()` 的 q/dq；`ddq` 缺省零向量。
        """
        if q is None or dq is None:
            q_cur, dq_cur = self._current_qdq("solve_idyn")
            q = q if q is not None else q_cur
            dq = dq if dq is not None else dq_cur
        q, dq = np.asarray(q, dtype=float), np.asarray(dq, dtype=float)
        ddq = np.zeros_like(q) if ddq is None else np.asarray(ddq, dtype=float)
        return self._activated_of("dyn").idyn(self, q, dq, ddq)

    def get_mass_matrix(self, q=None) -> np.ndarray:
        """关节空间惯量矩阵 `M(q)`（`(n, n)`）。"""
        if q is None:
            q = self._current_q("get_mass_matrix")
        return self._activated_of("dyn").mass_matrix(self, np.asarray(q, dtype=float))

    def get_coriolis(self, q=None, dq=None) -> np.ndarray:
        """科氏/向心项 `C(q, q̇)q̇`（`(n,)` N·m）。"""
        if q is None or dq is None:
            q_cur, dq_cur = self._current_qdq("get_coriolis")
            q = q if q is not None else q_cur
            dq = dq if dq is not None else dq_cur
        return self._activated_of("dyn").coriolis(self, np.asarray(q, dtype=float),
                                                  np.asarray(dq, dtype=float))

    def get_gravity(self, q=None) -> np.ndarray:
        """重力项 `G(q)`（`(n,)` N·m）。"""
        if q is None:
            q = self._current_q("get_gravity")
        return self._activated_of("dyn").gravity(self, np.asarray(q, dtype=float))

    # ============================================================
    # 7.7 到位判断三函数（§6.7）
    # ============================================================
    def is_in_pose_arm(self, pose, tol_pos: Optional[float] = None,
                       tol_rot: Optional[float] = None, frame=None) -> bool:
        """末端位姿到位判断：位置误差 ≤ `tol_pos` 且姿态误差角 ≤ `tol_rot`。

        末端位姿由激活 fkine 对 `get_state_arm().q` 现算（`frame` 缺省取
        cfg `joyarm.frame_end`）；tol 缺省从 cfg `joyarm.arrival_tol` 取。

        :param pose: 目标位姿（须为 :class:`Pose` 类型）。
        :raises TypeError: `pose` 非 :class:`Pose`。
        """
        if not isinstance(pose, Pose):
            raise TypeError(
                f"joyarm.py - is_in_pose_arm：pose 需为 Pose 类型，实际 "
                f"{type(pose).__name__}")
        tol_pos = self._arrival_tol_pos if tol_pos is None else float(tol_pos)
        tol_rot = self._arrival_tol_rot if tol_rot is None else float(tol_rot)
        st = self._backend.get_state_arm()
        if st is None or st.q is None:
            logger.warning("joyarm.py - is_in_pose_arm：无可用关节状态（未连接/读取失败），"
                           "判为未到位")
            return False
        cur = self._activated_of("fkine").solve(self, st.q, frame or self._frame_end)
        d_pos = float(np.linalg.norm(
            np.asarray(cur.position, float) - np.asarray(pose.position, float)))
        q_err = quat_mul(quat_conj(np.asarray(cur.orientation, float)),
                         np.asarray(pose.orientation, float))
        d_rot = float(quat_to_axis_angle(q_err)[1])
        return bool(d_pos <= tol_pos and d_rot <= tol_rot)

    def is_in_q_arm(self, q, tol_q=None) -> bool:
        """arm 关节到位判断：`np.max(|q_cur − q|) ≤ tol_q` 全关节同时满足。

        :param q: 目标关节角 `(n,)` rad（必传）。
        :param tol_q: 容差 rad；缺省从 cfg `joyarm.arrival_tol.arm.q` 取（n 元逐关节）。
        :raises ValueError: 目标维度与关节数不符。
        """
        return self._is_in_q(q, tol_q, self._arrival_tol_q_arm,
                             self._backend.get_state_arm, self._n_arm, "arm")

    def is_in_q_end(self, q, tol_q=None) -> bool:
        """end 电机到位判断：`np.max(|q_cur − q|) ≤ tol_q` 全电机同时满足。

        :param q: 目标电机角 `(n_end,)` rad（必传）。
        :param tol_q: 容差 rad；缺省从 cfg `joyarm.arrival_tol.end.q` 取（n_end 元逐电机）。
        :raises ValueError: 目标维度与电机数不符。
        """
        return self._is_in_q(q, tol_q, self._arrival_tol_q_end,
                             self._backend.get_state_end, self._n_end, "end")

    @staticmethod
    def _is_in_q(q, tol_q, tol_member, get_state, n: int, part: str) -> bool:
        """关节到位判断实现（§6.7）。"""
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != n:
            raise ValueError(
                f"joyarm.py - is_in_q_{part}：目标维度 {q.shape[0]} ≠ 关节数 {n}")
        st = get_state()
        if st is None or st.q is None:
            logger.warning("joyarm.py - is_in_q_%s：无可用关节状态（未连接/读取失败），"
                           "判为未到位", part)
            return False
        q_cur = np.asarray(st.q, dtype=float)
        tol = tol_member if tol_q is None else np.asarray(tol_q, dtype=float)
        return bool(np.all(np.abs(q_cur - q) <= np.broadcast_to(tol, q.shape)))

    # ============================================================
    # 7.8 MotionFSM 启停（§6.5）与安全事件（§7.2）
    # ============================================================
    def get_current_motion_state(self) -> MotionState:
        """FSM 当前状态（`_current_motion_state` 只读查询）。"""
        return self._current_motion_state

    def get_target_motion_state(self) -> MotionState:
        """FSM 目标状态。"""
        return self._target_motion_state

    def _set_target_motion_state(self, state: MotionState) -> None:
        """私有目标状态设置（原子赋值）。"""
        self._target_motion_state = state

    def start_motion_fsm(self) -> None:
        """启动运动状态机（§6.5，非阻塞）。

        前置：`is_connected_arm is True`（否则抛 `RuntimeError`）。若规划/控制
        线程尚在运行先停止，幂等清理 FSM 全部缓存（`_hold_q`、plan/ctrl 线程句柄
        与轨迹帧缓存；`_safe_event` 置 NORMAL、目标置 MOTION），启动 FSM 线程
        （固定 `_fsm_hz`）：初始 `current=DISABLE`、按路线表自动走 1→3→4 迁移
        至 MOTION（3→4 受 home 到位门槛）。已在运行时重复调用无副作用。

        :raises RuntimeError: 未连接；traj/ctrl 域未配置（MOTION 进入时抛出并重试）。
        """
        if self._backend.is_connected_arm is not True:
            raise RuntimeError(
                f"joyarm.py - start_motion_fsm：arm 未连接"
                f"（is_connected_arm={self._backend.is_connected_arm}），拒绝启动")
        if self._fsm_running():
            logger.info("joyarm.py - start_motion_fsm：FSM 已在运行，忽略重复启动")
            return
        self._stop_motion_thread()                 # 规划/控制线程尚在运行则先停止
        # 幂等清理 FSM 全部缓存
        self._hold_q = None
        self._zero_moving = False
        self._home_arrived = False
        self._target_trajframe = None
        self._monitor_none_since = None
        self._safe_event = SafetyEvent.NORMAL      # 与 stop 置 ZEROING 对称（统一事件与目标）
        self._target_motion_state = MotionState.MOTION
        self._current_motion_state = MotionState.DISABLE
        self._fsm_periodic = _PeriodicThread(self._fsm_worker, self._fsm_hz,
                                             name="joyarm-fsm")
        self._fsm_periodic.start()
        logger.info("joyarm.py - start_motion_fsm：FSM 已启动（fsm_hz=%s，"
                    "DISABLE → HOME → MOTION 自动迁移，3→4 受 home 到位门槛）",
                    self._fsm_hz)

    def stop_motion_fsm(self) -> None:
        """停止运动状态机（§6.5，阻塞含回零等待）。

        ① 前置 `is_connected_arm is True`，否则抛 `RuntimeError`（断连急停走
        销毁路径 `close()`）；② 置 `set_event(SafetyEvent.ZEROING)`（统一
        事件与目标），阻塞等待经 HOME 中转迁移至 ZERO 并完成到位失能，超时沿用
        cfg `joyarm.runtime.move_wait_timeout`（10 s），超时 warn 并强制推进；
        ③ `self._backend.disable_arm()` 失能 arm 全部电机（end 失能由调用方或
        销毁路径处理）；④ 停 FSM 线程（join 带超时、超时 warn 不强杀——此时句柄
        保留、跳过缓存清理与状态复位（互斥与 ctrl 豁免判定继续生效），待其退出后再次调用
        本方法完成清理）、清空全部 FSM 缓存、状态机退出（再用需重新
        `start_motion_fsm`）。

        :raises RuntimeError: 未连接。
        """
        if self._backend.is_connected_arm is not True:
            raise RuntimeError(
                f"joyarm.py - stop_motion_fsm：arm 未连接（断连急停不走本方法——"
                f"无法下发任何指令，请走销毁路径 close()）")
        if self._fsm_running():
            self.set_event(SafetyEvent.ZEROING)    # 统一事件与目标（与 start 置 NORMAL 对称）
            deadline = time.monotonic() + self._move_wait_timeout
            done = False
            while time.monotonic() < deadline:
                if (self._current_motion_state is MotionState.ZERO
                        and not self._zero_moving):
                    done = True
                    break
                time.sleep(0.02)
            if not done:
                logger.warning(
                    "joyarm.py - stop_motion_fsm：%s s 内未完成回零失能"
                    "（超时强制推进：disable → 清缓存 → 退出）", self._move_wait_timeout)
        else:
            logger.info("joyarm.py - stop_motion_fsm：FSM 未在运行，直接执行失能与清理")
        # ③ 失能 arm 全部电机
        self._backend.disable_arm()
        # ④ 停 FSM 线程 + 清空全部 FSM 缓存 + 状态机退出
        if self._fsm_periodic is not None:
            try:
                self._fsm_periodic.stop()
            except RuntimeError as e:
                # 超时未退出：保留 _fsm_periodic 句柄（_fsm_running 仍为 True，
                # 互斥与 ctrl 豁免判定继续生效）；FSM 独占写的状态与缓存此时
                # 不得由用户线程改写——跳过清理返回，待线程退出后再次调用本
                # 方法完成清理
                logger.warning(
                    "joyarm.py - stop_motion_fsm：FSM 线程未按时退出（不强杀，"
                    "句柄保留、跳过缓存清理与状态复位；待其退出后再次调用本方法"
                    "收尾）：%s", e)
                self._stop_motion_thread()
                return
            self._fsm_periodic = None
        self._stop_motion_thread()
        self._hold_q = None
        self._zero_moving = False
        self._home_arrived = False
        self._target_trajframe = None      # 目标帧缓存仅在 FSM start/stop 边界清理
        self._safe_event = SafetyEvent.DISABLE
        self._target_motion_state = MotionState.DISABLE
        self._current_motion_state = MotionState.DISABLE
        logger.info("joyarm.py - stop_motion_fsm：FSM 已停止并清空缓存"
                    "（再用需重新 start_motion_fsm）")

    def set_event(self, event: SafetyEvent) -> None:
        """设置安全事件（§7.2）：原子赋值 `_safe_event` 并同步目标状态为事件映射状态。

        FSM 线程每拍优先消费事件变化，响应延迟 ≤ 1/fsm_hz（默认 ≤ 20 ms）。

        :param event: :class:`SafetyEvent` 枚举成员。
        :raises TypeError: 非 `SafetyEvent` 枚举。
        """
        if not isinstance(event, SafetyEvent):
            raise TypeError(
                f"joyarm.py - set_event：event 需为 SafetyEvent 枚举，实际 "
                f"{type(event).__name__}")
        self._safe_event = event
        self._target_motion_state = _EVENT_TO_STATE[event]

    def get_event(self) -> SafetyEvent:
        """读当前安全事件。"""
        return self._safe_event

    # ============================================================
    # 7.9 safe_monitor 监视器（§7.3）
    # ============================================================
    def start_monitor(self) -> None:
        """启动安全监视器（独立 daemon 线程；频率 cfg `MotionFSM.monitor_hz`，缺省 = fsm_hz）。

        每拍只读采集（`get_state_arm/end`、当前状态、当前事件）→ 按规则判断 →
        命中则 `set_event`；重复调用无副作用。
        """
        if self._monitor_periodic is not None and self._monitor_periodic.is_alive():
            logger.info("joyarm.py - start_monitor：监视器已在运行，忽略重复启动")
            return
        self._monitor_none_since = None
        self._monitor_periodic = _PeriodicThread(self._monitor_worker, self._monitor_hz,
                                                 name="joyarm-monitor")
        self._monitor_periodic.start()
        logger.info("joyarm.py - start_monitor：safe_monitor 已启动（monitor_hz=%s）",
                    self._monitor_hz)

    def stop_monitor(self) -> None:
        """停止安全监视器（join 带超时，超时 warn 不强杀、句柄保留待重试；重复调用无副作用）。"""
        if self._monitor_periodic is not None:
            try:
                self._monitor_periodic.stop()
            except RuntimeError as e:
                # 超时未退出：保留 _monitor_periodic 句柄（口径同 stop_motion_fsm/
                # _stop_motion_thread 的超时分支）——句柄若被置 None，本方法的
                # 判空分支将不再生效，仍存活的监视器线程沦为无法停止的僵尸
                # （每拍采集并 set_event 改写安全事件/目标状态）；跳过置空与
                # "已停止"日志，待线程退出后再次调用本方法完成收尾
                logger.warning(
                    "joyarm.py - stop_monitor：监视器线程未按时退出（不强杀，"
                    "句柄保留、暂未停止；待其退出后再次调用本方法收尾）：%s", e)
                return
            self._monitor_periodic = None
            logger.info("joyarm.py - stop_monitor：safe_monitor 已停止")
    def set_monitor_rules(self, rules) -> None:
        """绑定自定义监视规则（单条或列表）。

        规则签名 `rule(state_arm, state_end, motion, event) -> SafetyEvent | None`
        ——`state_arm/state_end` 允许为 `None`（状态读取失败时仍被调用），
        返回 `None` = 无意见。仲裁：内置硬规则优先于自定义；同组多命中取危险程度
        最高者（§7.3）。

        :param rules: 可调用对象或其列表。
        :raises ValueError: 存在非可调用项。
        """
        seq = [rules] if callable(rules) else list(rules or [])
        bad = [r for r in seq if not callable(r)]
        if bad:
            raise ValueError(
                f"joyarm.py - set_monitor_rules：存在非可调用规则项：{bad!r}")
        self._monitor_rules = seq

    # ============================================================
    # 7.10 运行状态可视化（§8：HTTP 轮询）
    # ============================================================
    def start_state_viz(self, traj_points: int = 200) -> None:
        """启动运行状态可视化（§8：单页面监控网页 + 单一只读 daemon 线程）。

        左侧 iframe 嵌 meshcat 3D 视图（**可选依赖**：此处才 import；缺包 warn 并
        降级为仅 HTML 状态面板、3D 区占位），右侧自写 HTML 状态面板；轻量 HTTP
        服务托管页面，面板 JS 定时 `fetch /state.json`（约 5 Hz，零新增依赖）。
        `_state_viz_worker` 只调只读接口刷新 meshcat 模型姿态与面板数据，绝不
        调用控制/下发方法。

        :param traj_points: 轨迹显示采样点数（start 时可配，默认 200；0 = 关闭）。
            轨迹显示需 FSM 已启动且处于 MOTION 态才绘制，亦可在界面勾选切换。
        """
        if self._viz_periodic is not None and self._viz_periodic.is_alive():
            logger.info("joyarm.py - start_state_viz：可视化已在运行，忽略重复启动")
            return
        self._viz_traj_points = max(int(traj_points), 0)
        self._viz_traj_enabled = bool(self._viz_traj_points)
        # ---- meshcat 可选依赖（降级路径）----
        self._viz_pin = None
        iframe_src = "about:blank"
        try:
            import meshcat
            from pinocchio.visualize import MeshcatVisualizer as _PinViz
            collision = pin.buildGeomFromUrdf(self._pin_model, self._urdf_path,
                                              pin.GeometryType.COLLISION)
            visual = pin.buildGeomFromUrdf(self._pin_model, self._urdf_path,
                                           pin.GeometryType.VISUAL)
            viewer = meshcat.Visualizer()
            viz = _PinViz(self._pin_model, collision, visual)
            viz.initViewer(meshcat_visualizer=viewer)
            viz.loadViewerModel()
            self._viz_pin = viz
            iframe_src = viewer.url()
            logger.info("joyarm.py - start_state_viz：meshcat 3D 视图已就绪（%s）",
                        iframe_src)
        except ImportError:
            logger.warning(
                "joyarm.py - start_state_viz：未安装 meshcat（可选依赖，"
                "`uv sync --extra vis` 安装），降级为仅 HTML 状态面板（3D 区占位）")
        except Exception as e:
            logger.warning(
                "joyarm.py - start_state_viz：meshcat 初始化失败（%s），"
                "降级为仅 HTML 状态面板", e)
        # ---- 轻量 HTTP 托管（单页面 + /state.json 轮询端点 + /traj 开关）----
        page = (_VIZ_PAGE_HTML
                .replace("__MESHCAT_SRC__", iframe_src)
                .replace("__HAS_MESHCAT__", "true" if self._viz_pin is not None else "false")
                .encode("utf-8"))
        arm_ref = self

        class _VizHandler(BaseHTTPRequestHandler):
            """可视化页面/数据请求处理（只读 + 界面开关；不触任何控制接口）。"""

            def do_GET(self) -> None:  # noqa: N802（http.server 约定）
                path, _, query = self.path.partition("?")
                if path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                elif path == "/state.json":
                    with arm_ref._viz_lock:
                        payload = json.dumps(arm_ref._viz_state).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif path == "/traj":
                    arm_ref._viz_traj_enabled = query.strip().endswith("on=1")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"ok")
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *args) -> None:
                pass    # 静默访问日志（轮询高频，避免刷屏）

        self._viz_httpd = ThreadingHTTPServer(("127.0.0.1", 0), _VizHandler)
        port = self._viz_httpd.server_address[1]
        threading.Thread(target=self._viz_httpd.serve_forever,
                         name="joyarm-viz-http", daemon=True).start()
        # ---- 单一只读 daemon 线程：刷新 meshcat 姿态与面板数据 ----
        with self._viz_lock:
            self._viz_state = {}
        self._viz_periodic = _PeriodicThread(self._state_viz_worker, _VIZ_WORKER_HZ,
                                             name="joyarm-viz")
        self._viz_periodic.start()
        logger.info("joyarm.py - start_state_viz：监控页面已启动 "
                    "http://127.0.0.1:%s/（面板 JS 约 5 Hz 轮询 /state.json；"
                    "轨迹点数 %s）", port, self._viz_traj_points)

    def stop_state_viz(self) -> None:
        """停止运行状态可视化（停 worker、关 HTTP 服务；重复调用无副作用）。

        worker 停止超时（`RuntimeError`）时保留句柄并提前返回——口径同
        `stop_monitor` / `_stop_motion_thread` 的超时分支：句柄若被置
        `None`，仍存活的 worker 沦为无法停止的僵尸，且再次 `start_state_viz`
        的存活守卫将失效、创建第二套 worker 与 HTTP 服务；待线程退出后再次
        调用本方法完成收尾。
        """
        was_running = self._viz_periodic is not None or self._viz_httpd is not None
        if self._viz_periodic is not None:
            try:
                self._viz_periodic.stop()
            except RuntimeError as e:
                logger.warning(
                    "joyarm.py - stop_state_viz：viz worker 未按时退出（不强杀，"
                    "句柄保留、暂未停止；待其退出后再次调用本方法收尾）：%s", e)
                return
            self._viz_periodic = None
        httpd, self._viz_httpd = self._viz_httpd, None
        if httpd is not None:
            try:
                httpd.shutdown()
                httpd.server_close()
            except Exception as e:
                logger.warning("joyarm.py - stop_state_viz：HTTP 服务关闭异常：%s", e)
        self._viz_pin = None
        if was_running:
            logger.info("joyarm.py - stop_state_viz：可视化已停止")

    # ============================================================
    # 7.11 销毁安全退出（§6.6）
    # ============================================================
    def close(self) -> None:
        """安全退出：`stop_motion_fsm()`（best-effort）→ `disable_arm` →
        `disable_end` → `disconnect_arm/end`。

        各步异常捕获转 warn 不阻断收尾；同时停监视器与可视化线程。请用 `with`
        语句或显式调用本方法收尾（`__del__` 不做执行时序承诺）。
        """
        steps = (
            ("stop_motion_fsm", self.stop_motion_fsm),
            ("stop_monitor", self.stop_monitor),
            ("stop_state_viz", self.stop_state_viz),
            ("disable_arm", self._backend.disable_arm),
            ("disable_end", self._backend.disable_end),
            ("disconnect_arm", self._backend.disconnect_arm),
            ("disconnect_end", self._backend.disconnect_end),
        )
        for name, fn in steps:
            try:
                fn()
            except Exception as e:
                logger.warning("joyarm.py - close：%s 失败（继续收尾）：%s", name, e)
        logger.info("joyarm.py - close：安全退出完成（%s）", self._model)

    def __enter__(self) -> "JoyArm":
        """支持 `with arm:` 写法（进入返回自身；连接需显式 `connect_arm`）。"""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """`with` 退出收尾（§6.6）：第一步 `stop_motion_fsm()`（best-effort），
        其后 `disable_arm → disable_end → disconnect`。"""
        self.close()

    def __del__(self) -> None:
        """析构（**不做执行时序承诺**）：best-effort 调 :meth:`close`。

        请用 `with` 语句或显式 `close()` 收尾，勿依赖本方法。
        """
        try:
            self.close()
        except Exception:
            pass
