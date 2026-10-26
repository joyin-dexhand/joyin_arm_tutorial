"""`JoyArm` —— 机械臂组合根（config 驱动的整机对象）

按 `config/<型号>.yaml` 把 `backend`、`robotics` 与 `robot_model`组合成整机对象，
对外提供统一调用入口。一级功能块包括：

- 初始化与自检（定义与赋值两阶段 + cfg 全套自检 + pin 锁末端缩减模型 + 完成后 info）；
- 硬件后端委托面（arm/end 两组薄委托 + FSM 运行中的外部指令互斥与 ctrl 线程豁免）；
- 求解器六域成员 /方法 热切换 / 目标帧通道 + plan/ctrl 两线程 + 数学门面；
- 运动状态机 MotionFSM（六状态 + 路线表 + start/stop + 销毁安全退出）；
- 安全事件 SafetyEvent 与 safe_monitor 监视器（硬规则 + 自定义规则）；
- 运行状态可视化（HTTP 轮询：单页面监控网页 + 单一只读 daemon 线程）。

"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional, Sequence, Union

import meshcat
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import yaml
import zmq

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
# 路径与校验常量
# ============================================================
# config/ 目录（joyarm.py 位于 joyarm_core/joyarm/）
_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
)

# robot_model/ 目录（URDF 模型 + 网格文件，运行期加载）
_ROBOT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot_model"
)

# cfg 关节限位与 URDF <limit> 的一致性核对容差
_URDF_LIMIT_TOL = 1e-4

# 电机模式与帧名之外的 yaml 接受判定
_ACTIVE_JOINT_TYPES = ("revolute", "continuous", "prismatic")


# ============================================================
# SafetyEvent ↔ MotionState 映射与危险度
# ============================================================
_EVENT_TO_STATE = {
    SafetyEvent.NORMAL: MotionState.MOTION,
    SafetyEvent.HOLDING: MotionState.HOLD,
    SafetyEvent.HOMING: MotionState.HOME,
    SafetyEvent.ZEROING: MotionState.ZERO,
    SafetyEvent.DAMPING: MotionState.DAMPING,
    SafetyEvent.DISABLE: MotionState.DISABLE,
}

# 危险程度递增序（同拍多命中仲裁用）
_EVENT_DANGER = {
    SafetyEvent.NORMAL: 0,
    SafetyEvent.HOLDING: 1,
    SafetyEvent.HOMING: 2,
    SafetyEvent.ZEROING: 3,
    SafetyEvent.DAMPING: 4,
    SafetyEvent.DISABLE: 5,
}


# ============================================================
# MotionFSM 状态转移路线表
# ============================================================
def _build_route() -> dict:
    """按状态转移路线表构造 `{from: {to: next}}`（`home` 为中枢）。"""
    states = (MotionState.DISABLE, MotionState.ZERO, MotionState.HOME, MotionState.MOTION, MotionState.HOLD, MotionState.DAMPING)
    table = {
        # from disable：→zero/→motion 经 home 中转；→home/→hold/→damping 保护性直达
        MotionState.DISABLE: {
            MotionState.ZERO: MotionState.HOME, MotionState.HOME: MotionState.HOME,
            MotionState.MOTION: MotionState.HOME, MotionState.HOLD: MotionState.HOLD,
            MotionState.DAMPING: MotionState.DAMPING},
        # from zero：→motion 经 home；→home/disable/hold/damping 直达
        MotionState.ZERO: {
            MotionState.DISABLE: MotionState.DISABLE, MotionState.HOME: MotionState.HOME,
            MotionState.MOTION: MotionState.HOME, MotionState.HOLD: MotionState.HOLD,
            MotionState.DAMPING: MotionState.DAMPING},
        # from home：中枢，各状态直达
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
# config 加载
# ============================================================
def load_config(model: str) -> dict:
    """读取 `config/<model>.yaml` 型号配置文件，返回解析后的 dict。

    :param model: 型号名（= yaml 文件名 = URDF 目录 `robot_model/<model>/` 名）。
    :return: 解析后的配置 dict（非空映射）。
    :raises ValueError: 文件缺失 / yaml 语法错误 / 顶层非映射或为空；消息含文件路径与原始解析错误。
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
# cfg 与 URDF 校验助手
# ============================================================
def _urdf_joint_table(urdf_path: str) -> dict:
    """解析 URDF 全部 <joint> 元素，返回 `{关节名: 信息}`。

    信息键：`type`（关节类型）、`mimic`（是否从动）、`q_min/q_max/dq_max/tau_max`（<limit> 四键）。

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
    """解析 URDF 全部 <link> 名。"""
    return [lk.get("name") for lk in ET.parse(urdf_path).getroot().iterfind("link")]


def _soft_beyond_hard(soft: JointLimits, hard: JointLimits) -> bool:
    """软限位是否越出硬限位。"""
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
    """只读化：ndarray 本体或数据类内部的 ndarray 字段置 `writeable=False` 后返回。"""
    if isinstance(obj, np.ndarray):
        obj.flags.writeable = False
        return obj
    for v in vars(obj).values():
        if isinstance(v, np.ndarray):
            v.flags.writeable = False
    return obj


def _cfg_leaf_problems(section, root: str) -> List[str]:
    """cfg 段递归叶校验：None 叶子一律拒、数值叶（int/float，bool 除外）须有限。"""
    problems: List[str] = []

    def _walk(node, path: str) -> None:
        if node is None:
            problems.append(f"{path} 为 null（{root} 段不允许空值）")
        elif isinstance(node, bool):
            pass                              # bool 不按数值叶校验
        elif isinstance(node, (int, float)):
            if not bool(np.isfinite(node)):
                problems.append(f"{path}={node} 非有限数值（{root} 段数值叶须有限）")
        elif isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                _walk(v, f"{path}[{i}]")

    _walk(section, root)
    return problems


def _backend_joint_cfgs(bcfg, part: str) -> List[dict]:
    """backend.<part>.joints 安全提取。"""
    seg = bcfg.get(part)
    joints = seg.get("joints") if isinstance(seg, dict) else None
    if not isinstance(joints, list):
        return []
    return [j for j in joints if isinstance(j, dict)]


# ============================================================
# 六域算法成员的构建：cfg 段六键 → 域缩写与注册表
# ============================================================
# 域缩写 → (cfg 段键, REGISTRY)；cfg 段六键以 yaml 为准（fkine/ikine/jacobian/dynamics/trajectory/control 六键）
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
    """校验管线起搏键（plan_hz/ctrl_hz）：必配且 ≥1。

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
# 周期线程设施（私有周期线程机制）
# ============================================================
_JOIN_TIMEOUT: float = 1.0       # stop() 等线程退出的上限（秒）


def _as_hz(hz) -> float:
    """频率参数归一：直接给数值，或给一个"返回数值的无参函数"。"""
    return float(hz() if callable(hz) else hz)


def _run_periodic(stop: threading.Event, hz, step: Callable[[], None], name: str = "") -> None:
    """以 `hz` 频率循环调用 `step()`，`stop` 置位后退出。

    节拍按绝对时间对齐：某步执行太慢只跳过单拍，之后重新对齐。
    `step()` 抛异常只记日志、下个周期照常重试。

    :param stop: 停止事件（置位即退出循环）。
    :param hz:   执行频率 Hz（数值或返回数值的无参函数，每周期重新取）。
    :param step: 单步函数（无参数；内部异常视为"此步失败，下周期重来"）。
    :param name: 日志里标识本循环的名字（如线程名）。
    """
    period = 1.0 / _as_hz(hz)
    deadline = time.perf_counter()
    while not stop.is_set():
        deadline += period
        now = time.perf_counter()
        if deadline > now:
            if stop.wait(deadline - now):
                break
        else:                              # 单步太慢错过节拍：重新对齐，不追赶
            deadline = now
        try:
            step()
        except Exception:
            logger.exception("joyarm.py - _run_periodic：%s单步 %s 失败，下周期重试",
                             f"[{name}] " if name else "",
                             getattr(step, "__name__", step))
        try:
            period = 1.0 / _as_hz(hz)      # 每周期重取（运行期改频率即时生效）
        except Exception:
            pass                           # 取频率失败：沿用上一节拍


class _PeriodicThread:
    """一个可反复启停的周期线程（`start()` 启动、`stop()` 停止并回收）。

    `stop()` 最多等 1 秒；超时抛 `RuntimeError` 并保留线程引用（`is_alive()`仍为真、再次 `start()` 会被拒绝）
    ——避免旧线程还卡在单步里时又启动新线程，出现两个循环同时运行的情况。
    """

    def __init__(self, step: Callable[[], None], hz: float, name: str = "periodic") -> None:
        """
        :param step: 单步函数。
        :param hz: 频率 Hz。
        :param name: 线程名。
        """
        self._step = step               # 单步函数（循环体）
        self._hz = float(hz)            # 执行频率 Hz
        self._name = str(name)          # 线程名（日志标识）
        self._stop = threading.Event()  # 停止事件（置位即退出）
        self._thread: Optional[threading.Thread] = None  # 底层线程句柄（未启动/已回收为 None）

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

    def is_running(self) -> bool:
        """线程存活且未请求停止；用于保护迁移取消后判断运动管线是否需要重建。"""
        return self.is_alive() and not self._stop.is_set()

    def prepare(self) -> threading.Thread:
        """预创建底层线程对象并返回（未启动）。

        属主把线程句柄登记到外部判定槽位（如 ctrl 线程互斥豁免判定）后再 :meth:`start`。

        :return: 已创建未启动的底层线程对象（与 :attr:`thread` 同一引用）。
        :raises RuntimeError: 已有句柄（运行中或上次 stop 未确认退出）。
        """
        if self._thread is not None:
            raise RuntimeError(
                f"joyarm.py - _PeriodicThread.prepare：[{self._name}] 线程仍在运行"
                f"（或上次 stop 未成功退出），须先确认 stop() 完成")
        self._stop.clear()
        self._thread = threading.Thread(
            target=_run_periodic,
            args=(self._stop, lambda: self._hz, self._step, self._name),
            daemon=True, name=self._name)
        return self._thread

    def start(self) -> None:
        """启动线程；已在运行（或上次 stop 没停干净）时抛 `RuntimeError`。"""
        th = self._thread
        if th is None:
            th = self.prepare()
        elif th.ident is not None:      # ident 仅在线程起跑后为非 None（预创建未启动放行）
            raise RuntimeError(
                f"joyarm.py - _PeriodicThread.start：[{self._name}] 线程仍在运行"
                f"（或上次 stop 未成功退出），须先确认 stop() 完成")
        th.start()

    def stop(self) -> None:
        """停止线程；超时（1s）没退出抛 `RuntimeError`（引用保留）。"""
        self._stop.set()
        th = self._thread
        if th is None:
            return
        if th.ident is None:                 # prepare 后启动失败：没有线程可 join
            self._thread = None
            return
        th.join(timeout=_JOIN_TIMEOUT)
        if th.is_alive():
            raise RuntimeError(
                f"joyarm.py - _PeriodicThread.stop：[{self._name}] 线程 "
                f"{_JOIN_TIMEOUT}s 内未退出（单步可能阻塞在总线 IO 上）；"
                f"在确认其退出前请勿重新 start()")
        self._thread = None


# ============================================================
# 可视化监控页面模板（HTTP 轮询）：三栏布局（左侧静态参考 - 右侧动态状态 - 中间 3D 视图）
# ============================================================
_VIZ_PAGE_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>JoyArm 运行状态监控</title>
<style>
 :root { --colh: min(100vh, calc(60vw - 16px)); } /* 三栏总宽 = 5/3·colh + 2×8px gap，
        上限须扣 16px gap，否则居中布局下左栏左缘溢出不可达（永久裁切） */
 body { margin: 0; font-family: system-ui, sans-serif; background: #1e1e1e;
        color: #ddd; height: 100vh; display: flex; justify-content: center;
        align-items: center; gap: 8px; }
 .col { height: var(--colh); box-sizing: border-box; padding: 10px; }
 #col-left { flex: none; aspect-ratio: 1/3; overflow-y: auto; }   /* 竖长条，内容内部纵滚 */
 #col-mid { flex: none; aspect-ratio: 1/1; position: relative; overflow: hidden; }
 #col-right { flex: none; aspect-ratio: 1/3; overflow-y: auto; }
 #col-mid iframe { width: 100%; height: 100%; border: 0; }
 #placeholder { position: absolute; inset: 0; display: flex; align-items: center;
                justify-content: center; color: #777; font-size: 15px; }
 #placeholder[hidden] { display: none; } /* display:flex 会覆盖 hidden 属性，须显式压制 */
 #mid-overlay { position: absolute; top: 8px; right: 8px; z-index: 5;
                background: rgba(30, 30, 30, .72); padding: 4px 8px;
                border-radius: 4px; }
 h2 { font-size: 15px; margin: 12px 0 6px; color: #7fd1ff; }
 table { width: 100%; border-collapse: collapse; font-size: 12px;
         font-family: monospace; }
 th, td { border: 1px solid #444; padding: 2px 5px; text-align: center; }
 .kv { font-size: 13px; margin: 4px 0; font-family: monospace; word-break: break-all; }
 .ok { color: #7fe27f; } .bad { color: #ff7f7f; } .na { color: #999; }
 .warn { color: #ffd97f; }
 details { margin: 12px 0 4px; }
 summary { font-size: 14px; color: #7fd1ff; cursor: pointer; }
 pre { overflow-x: auto; font-size: 11px; margin: 4px 0; } /* cfg 原文长行横向滚动 */
 label { font-size: 13px; }
</style>
</head>
<body>
<div class="col" id="col-left">
  <h2>标识与关节</h2><div class="kv" id="ident">等待数据…</div>
  <h2>限位</h2><div class="kv" id="limits">等待数据…</div>
  <h2>特征位形</h2><div class="kv" id="poses">等待数据…</div>
  <h2>控制默认值</h2><div class="kv" id="ctrl-defaults">等待数据…</div>
  <details>
    <summary>cfg 原文（已滤除通信拓扑/名义运动学参数）</summary>
    <pre id="cfg-raw">等待数据…</pre>
  </details>
</div>
<div class="col" id="col-mid">
  <iframe id="mc" src="__MESHCAT_SRC__"></iframe>
  <div id="placeholder" hidden>meshcat 未启用（初始化失败）</div>
  <div id="mid-overlay"><label><input type="checkbox" id="traj"> 轨迹（MOTION 态）</label></div>
</div>
<div class="col" id="col-right">
  <h2>运行态</h2><div class="kv" id="fsm">等待数据…</div>
  <div class="kv" id="snap">等待数据…</div>
  <div class="kv" id="bk">等待数据…</div>
  <h2 id="tcp-title">末端位姿与速度</h2><div class="kv" id="tcp">等待数据…</div>
  <h2 id="arm-title">arm 关节</h2><table id="arm"></table>
  <h2 id="end-title">end 关节</h2><table id="end"></table>
  <h2>求解器与管线</h2><div class="kv" id="solvers">等待数据…</div>
</div>
<script>
const HAS_MC = __HAS_MESHCAT__;
const $ = (id) => document.getElementById(id);
if (!HAS_MC) { $("mc").remove(); $("placeholder").hidden = false; }
function fmt(v, d = 4) { return (v === null || v === undefined) ? "—" : (+v).toFixed(d); }
function fmtInf(v, d = 3) { return (v === null || v === undefined) ? "∞" : (+v).toFixed(d); }
function arrOrDash(a, d = 3) {   // 数组 → [v, v, …]（null/undefined 整组显「—」）
  return (a === null || a === undefined) ? "—" : `[${a.map(v => fmt(v, d)).join(", ")}]`;
}
function tri(v) {
  return v === true ? '<span class="ok">true</span>'
       : v === false ? '<span class="bad">false</span>'
       : '<span class="na">None</span>';
}
function fmtClock(t) {   // Unix 秒 → 本地 HH:MM:SS.mmm（仅 time.time() 域；单调钟原值禁用）
  if (t === null || t === undefined) return "—";
  const d = new Date(t * 1000), p = (x, n = 2) => String(x).padStart(n, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)}`;
}
function jointTable(el, g, emptyText) {
  if (!g || !g.names || !g.names.length) { el.innerHTML = `<tr><td>${emptyText}</td></tr>`; return; }
  let h = "<tr><th>#</th><th>name</th><th>q rad</th><th>dq rad/s</th><th>tau N·m</th>"
        + "<th>T_mos ℃</th><th>T_rot ℃</th><th>err</th></tr>";
  g.names.forEach((n, i) => {
    const err = (g.error && g.error[i] !== undefined) ? g.error[i] : null;
    const errCell = (err === null || err === undefined) ? '<td class="na">—</td>'
                  : err === -1 ? '<td class="na">未读</td>'
                  : `<td class="${err >= 2 ? 'bad' : 'ok'}">${err}</td>`;
    h += `<tr><td>${i + 1}</td><td>${n}</td>`
       + `<td>${fmt(g.q && g.q[i])}</td><td>${fmt(g.dq && g.dq[i])}</td>`
       + `<td>${fmt(g.tau && g.tau[i])}</td><td>${fmt(g.temp_mos && g.temp_mos[i])}</td>`
       + `<td>${fmt(g.temp_rotor && g.temp_rotor[i])}</td>${errCell}</tr>`;
  });
  el.innerHTML = h;
}
function snapLine(label, m, noMotor) {   // 快照标注行：陈旧仅标注、不隐藏数据
  if (noMotor) return `${label} 快照（无电机）`;
  const badge = m.state === "新鲜" ? '<span class="ok">［新鲜］</span>'
              : m.state === "陈旧" ? '<span class="warn">［陈旧］</span>'
              : '<span class="na">［未读取］</span>';
  const age = (m.age === null || m.age === undefined) ? "—" : (+m.age).toFixed(2) + "s";
  return `${label} 快照 t=${fmtClock(m.t)} age=${age} ${badge}`;
}
function renderSnap(s) {
  const nEnd = (s.end && s.end.names) ? s.end.names.length : 0;
  $("snap").innerHTML = snapLine("arm", s.snapshot.arm, false) + "<br>"
                      + snapLine("end", s.snapshot.end, nEnd === 0);
}
function renderSolvers(s) {
  const d = s.solvers;
  let h = ["fkine", "ikine", "jac", "dyn", "traj", "ctrl"]
    .map(k => `${k}=<b>${d[k] || "（未配置）"}</b>`).join(" ");
  h += `<br>plan_hz=${fmt(d.plan_hz, 1)} ctrl_hz=${fmt(d.ctrl_hz, 1)}`;
  let tgt = "未设置";
  if (d.target) {
    const tl = (d.target.t_last === null || d.target.t_last === undefined)
             ? "—" : (+d.target.t_last).toFixed(3);
    tgt = d.target.kind === "seq"
        ? `序列 ×${d.target.frames} 末帧 t=${tl}` : `单帧 t=${tl}`;
  }
  return h + `<br>目标帧：${tgt}`;
}
function limArr(v, side) {   // 限位区间数组（null 元素 = 不限位 ±∞）
  return "[" + v.map(x => (x === null || x === undefined)
    ? (side === "min" ? "−∞" : "∞") : fmt(x)).join(", ") + "]";
}
// ---- 左栏静态组渲染（/static.json；组装配失败 _error 显「装配失败」）----
function renderIdent(st) {
  if (!st || st._error) { $("ident").innerHTML = (st && st._error) ? "装配失败" : "—"; return; }
  $("ident").innerHTML = `型号=<b>${st.model}</b><br>frame_end=<b>${st.frame_end}</b><br>`
    + `urdf=${st.urdf}<br>n_arm=${st.n_arm} n_end=${st.n_end}<br>`
    + `arm 关节名：${st.joint_names_arm.join(", ") || "（无关节）"}<br>`
    + `end 关节名：${st.joint_names_end.join(", ") || "（无电机）"}`;
}
function renderLimits(lm) {
  if (!lm || lm._error) { $("limits").innerHTML = (lm && lm._error) ? "装配失败" : "—"; return; }
  let h = "";
  for (const fam of ["arm", "end"]) {
    const g = lm[fam];
    if (!g) { h += `${fam} 限位：—（无电机）<br>`; continue; }
    h += `${fam} 硬限位 q∈${limArr(g.hard_q_min, "min")}~${limArr(g.hard_q_max, "max")} `
       + `dq_max≤${limArr(g.hard_dq_max, "max")} tau_max≤${limArr(g.hard_tau_max, "max")}<br>`;
    h += `${fam} 软限位 q∈${limArr(g.soft_q_min, "min")}~${limArr(g.soft_q_max, "max")} `
       + `dq_max≤${limArr(g.soft_dq_max, "max")} tau_max≤${limArr(g.soft_tau_max, "max")}<br>`;
  }
  const T = lm.tcp;
  h += T.workspace_box
     ? "工作空间盒 m（x/y/z）：" + T.workspace_box.map(r => `[${fmt(r[0], 3)}, ${fmt(r[1], 3)}]`).join(" ") + "<br>"
     : "工作空间盒：—（不限位）<br>";
  h += `v_lin_max=${fmtInf(T.v_lin_max)} v_ang_max=${fmtInf(T.v_ang_max)} `
     + `f_max=${fmtInf(T.f_max)} t_max=${fmtInf(T.t_max)}`;
  $("limits").innerHTML = h;
}
function renderPoses(st) {
  if (!st || st._error) { $("poses").innerHTML = (st && st._error) ? "装配失败" : "—"; return; }
  let h = "";
  for (const p of ["zero", "neutral", "home"]) {
    for (const fam of ["arm", "end"]) {
      const arr = st[`${p}_${fam}`];
      if (!arr || !arr.length) continue;
      h += `${p}_${fam}=[${arr.map(v => fmt(v)).join(", ")}]<br>`;
    }
  }
  $("poses").innerHTML = h || "—";
}
function renderCtrl(cd) {
  if (!cd || cd._error) { $("ctrl-defaults").innerHTML = (cd && cd._error) ? "装配失败" : "—"; return; }
  let h = "";
  for (const fam of ["arm", "end"]) {
    const m = cd.mit ? cd.mit[fam] : null;
    h += `MIT.${fam} kp=${arrOrDash(m && m.kp)} kd=${arrOrDash(m && m.kd)}<br>`;
  }
  for (const fam of ["arm", "end"]) {
    const p = cd.pos_vel ? cd.pos_vel[fam] : null;
    h += `POS_VEL.${fam} pos_kp=${arrOrDash(p && p.pos_kp)} pos_ki=${arrOrDash(p && p.pos_ki)} `
       + `vel_kp=${arrOrDash(p && p.vel_kp)} vel_ki=${arrOrDash(p && p.vel_ki)} `
       + `vlim=${arrOrDash(p && p.vlim)} flim=${arrOrDash(p && p.flim)}<br>`;
  }
  const dm = cd.default_mode || {}, bp = cd.backend_params || {}, at = cd.arrival_tol || {};
  h += `default_mode：arm=${dm.arm || "—"} end=${dm.end || "—"}<br>`;
  h += `write_settle=${fmt(bp.write_settle, 3)} state_refresh_hz=${fmt(bp.state_refresh_hz, 1)}<br>`;
  h += `fsm_hz=${fmt(cd.fsm_hz, 1)} monitor_hz=${fmt(cd.monitor_hz, 1)} `
     + `damping_kd=${arrOrDash(cd.damping_kd)} damping_k_gravity=${arrOrDash(cd.damping_k_gravity)}<br>`;
  h += `到位容差 arm_q=${arrOrDash(at.arm_q)} end_q=${arrOrDash(at.end_q)} `
     + `pos=${fmt(at.pos, 4)} rot=${fmt(at.rot, 4)}<br>`;
  h += `move_wait_timeout=${fmt(cd.move_wait_timeout, 2)}`;
  $("ctrl-defaults").innerHTML = h;
}
let STATIC = null;          // /static.json 载荷（null = 未就绪，poll 内重试）
let staticLoading = false;  // in-flight 防护（fetch 挂起时不重复发起）
async function loadStatic() {
  if (staticLoading) return;
  staticLoading = true;
  try {
    STATIC = await (await fetch("/static.json")).json();
    renderIdent(STATIC.ident);
    renderLimits(STATIC.limits);
    renderPoses(STATIC.poses);
    renderCtrl(STATIC.ctrl_defaults);
    $("cfg-raw").textContent = (typeof STATIC.cfg_raw === "string") ? STATIC.cfg_raw : "装配失败";
  } catch (e) { STATIC = null; }   // 拉取/渲染失败保持未就绪，随轮询重试
  finally { staticLoading = false; }
}
async function poll() {
  if (STATIC === null) loadStatic();
  try {
    const s = await (await fetch("/state.json")).json();
    $("fsm").innerHTML = `current=<b>${s.fsm.current}</b> target=<b>${s.fsm.target}</b> `
                       + `event=<b>${s.event}</b>`;
    renderSnap(s);
    $("bk").innerHTML = `connected_arm=${tri(s.backend.connected_arm)} `
                      + `enabled_arm=${tri(s.backend.enabled_arm)}<br>`
                      + `connected_end=${tri(s.backend.connected_end)} `
                      + `enabled_end=${tri(s.backend.enabled_end)}`;
    const fe = (STATIC && STATIC.ident && STATIC.ident.frame_end)
      ? STATIC.ident.frame_end : "—";
    $("tcp-title").textContent = `末端位姿与速度 frame_end=${fe}（位置 m / 四元数 wxyz / 速度）`;
    $("tcp").innerHTML = s.tcp
      ? `p=[${s.tcp.position.map(v => fmt(v, 3)).join(", ")}]<br>`
        + `q(wxyz)=[${s.tcp.orientation.map(v => fmt(v, 3)).join(", ")}]`
        + (s.tcp.twist
          ? `<br>v=[${s.tcp.twist.linear.map(v => fmt(v, 3)).join(", ")}] m/s`
            + `<br>w=[${s.tcp.twist.angular.map(v => fmt(v, 3)).join(", ")}] rad/s`
          : "")
      : "—（未读取或求解器未配置）";
    $("arm-title").textContent = `arm 关节（${(s.arm && s.arm.names) ? s.arm.names.length : 0}）`;
    $("end-title").textContent = `end 关节（${(s.end && s.end.names) ? s.end.names.length : 0}）`;
    jointTable($("arm"), s.arm, "（无关节）");
    jointTable($("end"), s.end, "（无电机）");
    $("solvers").innerHTML = renderSolvers(s);
    $("traj").checked = !!s.traj_enabled;
  } catch (e) { /* 下轮重试 */ }
}
$("traj").addEventListener("change", async () => {
  try { await fetch("/traj?on=" + ($("traj").checked ? 1 : 0)); } catch (e) {}
});
setInterval(poll, 200);   // 约 5 Hz 轮询（fetch /state.json；静态未就绪时顺带重试）
poll();
</script>
</body>
</html>
"""

# _state_viz_worker 的数据刷新频率（Hz；面板 JS 以约 5 Hz 轮询，刷新端取 10 Hz 保证领先）
_VIZ_WORKER_HZ: float = 10.0
# meshcat 刷新 zmq 收包超时（毫秒）：套接字置 RCVTIMEO，server 死亡时 recv 不永久阻塞
_VIZ_ZMQ_TIMEOUT_MS: int = 1000
# meshcat 刷新连续失败降级阈值（zmq 超时/状态错误累计达此数即降级仅 HTML 面板并停步进）
_VIZ_ZMQ_FAIL_MAX: int = 3
# 监视器「状态连续 None」判 DISABLE 的时长阈值（连续 0.1 s）
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


# ============================================================
# 原生状态可视化窗口（PySide6 子进程）
# ============================================================
class _VizWindow:
    """监控页原生窗口封装（PySide6 QtWebEngine 子进程，父/子进程两侧入口）。

    父进程侧：`launch()` 拉起子进程并挂早夭看护，`stop()` 终止；
    子进程侧经 `python -c` 调 `child_main()` 内联执行 QApplication + QWebEngineView 加载监控页。
    """

    def __init__(self, url: str, title: str = "JoyArm 运行状态监控") -> None:
        """记录参数并置进程句柄占位（进程由 `launch()` 创建）。

        :param url: 监控页 URL（`http://127.0.0.1:<port>/`）。
        :param title: 窗口标题。
        """
        self._url = url
        self._title = title
        self._proc: Optional[subprocess.Popen] = None   # 窗口子进程句柄
        self._stop_requested = threading.Event()        # 主动停止标志

    def launch(self) -> bool:
        """拉起窗口子进程并挂早夭看护（PySide6 预检失败返回 False 降级 URL）。

        :return: 是否成功创建子进程（早夭由看护线程事后告警）。
        """
        try:
            from PySide6 import QtWebEngineWidgets  # noqa: F401  导入即预检（子进程内正式使用）
        except Exception as e:
            logger.warning("joyarm.py - _VizWindow.launch：PySide6 预检失败（%s），"
                           "请用浏览器打开 %s", e, self._url)
            return False
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-c",
                 "from joyarm_core.joyarm.joyarm import _VizWindow; import sys; "
                 "sys.exit(_VizWindow.child_main(sys.argv[1:]))",
                 self._url, self._title],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            self._proc = None
            logger.warning("joyarm.py - _VizWindow.launch：监控窗口启动失败（%s），"
                           "请用浏览器打开 %s", e, self._url)
            return False
        threading.Thread(target=self._watchdog, name="joyarm-viz-win",
                         daemon=True).start()
        return True

    def _watchdog(self) -> None:
        """子进程早夭看护：启动短窗内非零退出则告警并弃置句柄（回退 URL）。

        对 `_proc` 取局部快照并做 None 防护（与 `stop()` 并发置空时不误触）；
        主动停止标志已置时不告警、不动句柄（收尾由 `stop()` 完成）。
        """
        proc = self._proc                          # 局部快照：stop 并发置 None 不致 None.wait
        if proc is None:
            return
        try:
            rc = proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            return    # 3 s 仍存活视为窗口已正常打开
        if self._stop_requested.is_set():
            return    # 用户主动收尾（terminate 致 -15 等）不当作故障告警
        if rc != 0:
            self._proc = None
            logger.warning("joyarm.py - _VizWindow.launch：监控窗口进程已退出"
                           "（码 %s，可能无显示器），请用浏览器打开 %s", rc, self._url)

    def stop(self) -> None:
        """终止窗口子进程（幂等；先置停止标志防看护误告警，超时 kill 兜底，失败均静默）。"""
        self._stop_requested.set()                 # 主动停止标志先行：看护线程不再当故障告警
        proc = self._proc
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()                        # terminate 2 s 未退出（挂起）：强杀兜底
                proc.wait()
            except Exception:
                pass    # 强杀失败静默弃句柄（幂等口径）
        except Exception:
            pass    # 窗口已自行关闭等无需处理（幂等口径）
        finally:
            self._proc = None

    @staticmethod
    def child_main(argv: List[str]) -> int:
        """子进程入口：`QMainWindow` + `QWebEngineView` 加载监控页并进入事件循环。

        :param argv: `[url]` 或 `[url, title]`（父进程经 `python -c` 透传）。
        :return: 进程退出码（参数缺失返回 2，正常关窗返回 0）。
        """
        from PySide6.QtCore import Qt, QUrl
        from PySide6.QtWidgets import QApplication, QMainWindow
        from PySide6.QtWebEngineWidgets import QWebEngineView

        if len(argv) < 1:
            print("用法：_VizWindow.child_main([url] 或 [url, title])")
            return 2
        url = argv[0]
        title = argv[1] if len(argv) > 1 else "JoyArm 运行状态监控"
        # QtWebEngine 官方建议：QApplication 创建前开启 OpenGL 上下文共享（WebGL 稳定）。
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        app = QApplication([])
        win = QMainWindow()
        win.setWindowTitle(title)
        win.resize(1440, 820)
        view = QWebEngineView()
        view.load(QUrl(url))
        win.setCentralWidget(view)
        win.show()
        return app.exec()


class JoyArm:
    """JoyArm 机械臂类（本体 arm + 末端 end，config 驱动的组合根）

    两阶段初始化：先声明全部私有成员并赋安全默认值，再按依赖顺序赋真值；任一自检失败抛具体异常、对象不构造。
    类内成员一律 `_` 前缀私有，类外访问经 `@property`（只读防篡改）或 `get_*`（返回深拷贝）。

    :param name: 型号名（= `config/<name>.yaml` 文件名 = `robot_model/<name>/` 目录名）。
    :raises ValueError: config 缺失、自检任一项失败、URDF/限位/特征位形校验失败。
    :raises RuntimeError: backend 或六域求解器构造失败（保留原因链）。
    """

    # ============================================================
    # 一、初始化：两阶段 —— 私有成员声明 + 按依赖顺序赋真值
    # ============================================================

    # ------------------------------------------------------------
    # 子阶段一：私有成员声明（安全默认值）
    # ------------------------------------------------------------
    def __init__(self, name: str):
        # ---- 基本信息 ----
        self._model: str = str(name)                     # 型号名
        self._cfg: Optional[dict] = None                 # config 全量 dict
        self._frame_end: str = ""                        # 末端帧名（cfg.joyarm.frame_end）
        self._urdf_path: str = ""                        # URDF 文件路径

        # ---- 运动学模型（pinocchio，URDF 末端锁定缩减）----
        self._pin_model: Optional[pin.Model] = None      # nq = n_arm 的缩减模型（只读共享）

        # ---- 关节数与关节名 ----
        self._n_arm: int = 0                             # 本体关节数（cfg backend.arm.joints 条数）
        self._n_end: int = 0                             # 末端电机数（cfg backend.end.joints 条数）
        self._joint_names_arm: tuple = ()                # 本体关节名（cfg 顺序 = q 向量顺序）
        self._joint_names_end: tuple = ()                # 末端电机名（cfg 顺序 = q 向量顺序）

        # ---- 限位（硬限位拦指令 / 软限位供上层状态判断，均只读化）----
        self._joint_limits_hard_arm: Optional[JointLimits] = None  # 本体硬限位
        self._joint_limits_hard_end: Optional[JointLimits] = None  # 末端硬限位
        self._joint_limits_soft_arm: Optional[JointLimits] = None  # 本体软限位
        self._joint_limits_soft_end: Optional[JointLimits] = None  # 末端软限位

        # ---- 特征位形（zero=全零 / neutral=软限位中点 / home=cfg 配置）----
        self._zero_arm: np.ndarray = np.zeros(0)         # 本体零位位形 (n_arm,)
        self._zero_end: np.ndarray = np.zeros(0)         # 末端零位位形 (n_end,)
        self._neutral_arm: np.ndarray = np.zeros(0)      # 本体中性位形 (n_arm,)
        self._neutral_end: np.ndarray = np.zeros(0)      # 末端中性位形 (n_end,)
        self._home_arm: np.ndarray = np.zeros(0)         # 本体home位形 (n_arm,)
        self._home_end: np.ndarray = np.zeros(0)         # 末端home位形 (n_end,)

        # ---- 末端空间限位 ----
        self._tcp_limits: TcpLimits = TcpLimits()        # 工作空间盒 + 速度/力上限

        # ---- 电机通信后端 ----
        self._backend = None                             # 经委托访问，不直接暴露

        # ---- 六域求解器（{注册名: 实例} + 激活对象指针）----
        self._fkine_solvers: Optional[dict] = None       # fkine 域成员 {注册名: 实例}
        self._ikine_solvers: Optional[dict] = None       # ikine 域成员 {注册名: 实例}
        self._jac_solvers: Optional[dict] = None         # jacobian 域成员 {注册名: 实例}
        self._dyn_solvers: Optional[dict] = None         # dynamics 域成员 {注册名: 实例}
        self._traj_solvers: Optional[dict] = None        # trajectory 域成员 {注册名: 实例}
        self._ctrl_solvers: Optional[dict] = None        # control 域成员 {注册名: 实例}
        self._activated_fkine_solver = None              # fkine 域激活求解器对象
        self._activated_ikine_solver = None              # ikine 域激活求解器对象
        self._activated_jac_solver = None                # jacobian 域激活求解器对象
        self._activated_dyn_solver = None                # dynamics 域激活求解器对象
        self._activated_traj_solver = None               # trajectory 域激活求解器对象
        self._activated_ctrl_solver = None               # control 域激活求解器对象
        self._plan_hz: Optional[float] = None            # 激活规划器 plan_hz（切换时更新）
        self._ctrl_hz: Optional[float] = None            # 激活控制器 ctrl_hz（切换时更新）
        self._plan_hz_by_name: dict = {}                 # {traj 注册名: plan_hz}
        self._ctrl_hz_by_name: dict = {}                 # {ctrl 注册名: ctrl_hz}

        # ---- 轨迹目标帧（外部应用线程写，规划线程每拍整体重读）----
        self._target_trajframe: Union[TrajFrame, List[TrajFrame], None] = None  # 轨迹目标帧（单帧或帧序列）

        # ---- 运动状态机 MotionFSM ----
        self._current_motion_state: MotionState = MotionState.DISABLE   # FSM 线程独占写
        self._target_motion_state: MotionState = MotionState.DISABLE    # 安全事件响应设置
        self._safe_event: SafetyEvent = SafetyEvent.DISABLE             # 当前安全事件（set_event 原子写入）
        self._event_lock = threading.RLock()             # 事件发布与监视器升档仲裁原子化；不持锁做 I/O
        self._fsm_hz: float = 50.0                       # cfg.joyarm.MotionFSM.fsm_hz
        self._damping_kd: np.ndarray = np.zeros(0)       # DAMPING 态 MIT kd（cfg 必配）
        self._damping_k_gravity: np.ndarray = np.zeros(0)  # DAMPING 态重力补偿率（cfg 必配）
        self._monitor_hz: float = 50.0                   # 缺省 = fsm_hz
        self._arrival_tol_q_arm = np.zeros(0)            # 到位容差（逐关节）
        self._arrival_tol_q_end = np.zeros(0)            # 到位容差（逐电机）
        self._arrival_tol_pos: float = 1e-3              # 末端位置容差 m
        self._arrival_tol_rot: float = 1e-2              # 末端姿态容差 rad
        self._move_wait_timeout: float = 10.0            # cfg.joyarm.runtime.move_wait_timeout
        self._hold_q: Optional[np.ndarray] = None        # HOLD 态锁定的关节位置
        self._zero_moving: bool = False                  # ZERO 态「移动中」子阶段标志
        self._home_arrived: bool = False                 # HOME 态到位日志去重
        self._fsm_disc_warned: bool = False              # 断连挂起告警去重（重连后复位）
        self._fsm_periodic: Optional[_PeriodicThread] = None  # FSM 周期线程句柄（fsm_hz）

        # ---- plan/ctrl 运动线程 ----
        self._plan_periodic: Optional[_PeriodicThread] = None  # plan 周期线程句柄
        self._ctrl_periodic: Optional[_PeriodicThread] = None  # ctrl 周期线程句柄
        self._ctrl_thread: Optional[threading.Thread] = None   # ctrl 线程句柄（互斥豁免判定）
        self._motion_lock = threading.RLock()           # 算法选择、模式自愈与单拍下发互斥；join 不持锁
        self._motion_ready = None                       # FSM 已确认就绪的控制器对象；None 禁止 ctrl 下发
        self._fsm_shutdown = threading.Event()          # 最终收尾开始后禁止自愈/使能/运动下发
        self._closed: bool = False                      # close() 已完成标志（重复 close 静默跳过）

        # ---- safe_monitor ----
        self._monitor_periodic: Optional[_PeriodicThread] = None  # 监视器周期线程句柄
        self._monitor_rules: List[Callable] = []         # 自定义规则列表
        self._monitor_none_since = None                  # 连续 None 的起算时刻

        # ---- 运行状态可视化 ----
        self._viz_periodic: Optional[_PeriodicThread] = None   # 可视化周期线程句柄
        self._viz_httpd: Optional[ThreadingHTTPServer] = None  # 监控页面 HTTP 服务句柄
        self._viz_pin = None                             # pin.visualize MeshcatVisualizer
        self._viz_zmq_fails: int = 0                     # meshcat 刷新连续失败计数（zmq 看门狗降级用）
        self._viz_state: dict = {}                       # 面板数据快照（HTTP 线程读）
        self._viz_traj_points: int = 200                 # 轨迹显示采样点数（start 可配）
        self._viz_traj_enabled: bool = False             # 轨迹显示开关（界面可切）
        self._viz_window: Optional[_VizWindow] = None    # 监控窗口对象（start 时创建拉起、stop 终止）
        # 以下三成员真值初始化推迟到 start_state_viz
        self._viz_model: Optional[pin.Model] = None      # 可视化专属 pin 模型（arm+end 联动）
        self._viz_static_bytes: bytes = b""              # /static.json 预装配载荷（start 时装配一次）
        self._viz_q_cache: dict = {}                     # display q 族级缓存 {arm, end}（start 时初始化）

        # ------------------------------------------------------------
        # 子阶段二：按依赖顺序赋真值（cfg 加载自检 → _pin_model → 关节数/名 → 限位 → 特征位形 → _tcp_limits → _backend → 六域solvers与激活项 → _plan/ctrl_hz → FSM 参数）
        # ------------------------------------------------------------
        # 1) cfg 加载 + 全套自检（frame_end/urdf_path 先行赋值供自检与 pin 构建用）
        self._cfg = load_config(name)                          # 缺失/解析失败 → 抛错
        jcfg = self._cfg.get("joyarm") if isinstance(self._cfg.get("joyarm"), dict) else {}
        backend_cfg = dict(self._cfg["backend"]) if isinstance(self._cfg.get("backend"), dict) else {}
        self._frame_end = str(jcfg.get("frame_end") or "")
        self._urdf_path = os.path.join(_ROBOT_MODEL_DIR, name, "urdf", f"{name}.urdf")
        self._check_cfg(self._cfg)                             # 全套自检（软限位越出硬限位的交叉检查在限位构建后补做）

        arm_joint_cfgs = list((backend_cfg.get("arm") or {}).get("joints") or [])
        end_joint_cfgs = list((backend_cfg.get("end") or {}).get("joints") or [])

        # 2) _pin_model（URDF 全量模型 → 锁定非 cfg.arm 名单活动关节 → 缩减模型）
        self._pin_model = self._build_pin_model(self._urdf_path, [str(j.get("name")) for j in arm_joint_cfgs])

        # 3) 关节数与关节名（cfg 顺序 = URDF 顺序 = q 向量顺序）
        self._joint_names_arm = tuple(str(j.get("name")) for j in arm_joint_cfgs)
        self._joint_names_end = tuple(str(j.get("name")) for j in end_joint_cfgs)
        self._n_arm = len(self._joint_names_arm)
        self._n_end = len(self._joint_names_end)

        # 4) 限位：硬限位（backend joints 四键）→ 软限位（cfg.joyarm 直配）→ 软在硬内校验
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
        # zero（全零位形，ZERO 态回零目标）须在硬限位内：不含零位的型号回零永不到位、
        # stop 每次等满超时（边界贴合按 1e-12 容差放行，与 home 校验口径一致）
        for tag, hard in (("arm", self._joint_limits_hard_arm),
                          ("end", self._joint_limits_hard_end)):
            if hard is None:
                continue     # 无该组（如 n_end=0）不校验
            q_min = np.asarray(hard.q_min, dtype=float)
            q_max = np.asarray(hard.q_max, dtype=float)
            bad = np.where((q_min > 1e-12) | (q_max < -1e-12))[0]
            if bad.size:
                i = int(bad[0])
                raise ValueError(
                    f"joyarm.py - JoyArm.__init__：backend.{tag}.joints 关节[{i + 1}]（1 基）"
                    f"硬限位 [{q_min[i]}, {q_max[i]}] 不含零位（zero=全零为 ZERO 态回零目标；"
                    f"越界关节序号 {bad.tolist()}）")
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

        # 6) 末端空间限位（先校验后解析；解析后 workspace_box 每行 [min,max]）
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

        # 8) 六域求解器与激活项 + _plan/_ctrl_hz
        self._build_domains(self._cfg.get("robotics") or {})
        if self._traj_solvers:
            self._plan_hz = self._plan_hz_by_name[self._activated_name_of("traj")]
        if self._ctrl_solvers:
            self._ctrl_hz = self._ctrl_hz_by_name[self._activated_name_of("ctrl")]

        # 10) FSM / 监视 / 到位容差 / 运行超时参数
        self._parse_fsm_params(jcfg)

        # 11) 初始化完成 info
        self._log_init_info()

    # ============================================================
    # 二、初始化助手：cfg 自检 / URDF 对比 / pin 模型 / 特征位形 / 限位解析
    # ============================================================

    # =============== cfg 内容自检 ===============
    def _check_cfg(self, cfg: dict) -> None:
        """cfg 加载后的跨成员/跨源自检。

        :raises ValueError: 任一项失败；一条消息列出全部问题。
        """
        problems: List[str] = []
        # ---- 段类型前置校验（段写成列表等结构错误）----
        for key in ("joyarm", "robotics", "backend"):
            v = cfg.get(key)
            if v is not None and not isinstance(v, dict):
                problems.append(f"顶层段 {key} 需为字典，实际 {type(v).__name__}")
        jcfg = cfg.get("joyarm") if isinstance(cfg.get("joyarm"), dict) else {}
        bcfg = cfg.get("backend") if isinstance(cfg.get("backend"), dict) else {}
        robotics_cfg = cfg.get("robotics") if isinstance(cfg.get("robotics"), dict) else {}

        # ---- 三顶层键存在 ----
        for key in ("joyarm", "robotics", "backend"):
            if key not in cfg:
                problems.append(f"缺少顶层配置段：{key}")
        # ---- 三顶层段值递归校验：None 叶直接拒、数值叶（bool 除外）须有限 ----
        for key in ("joyarm", "robotics", "backend"):
            if isinstance(cfg.get(key), (dict, list, tuple)):
                problems.extend(_cfg_leaf_problems(cfg[key], key))
        if not jcfg.get("frame_end"):
            problems.append("joyarm.frame_end 缺失（末端帧名，pin 模型帧）")

        # ---- backend.arm/end 段与 joints 条目类型前置校验 ----
        for part in ("arm", "end"):
            seg = bcfg.get(part)
            if seg is None:
                continue
            if not isinstance(seg, dict):
                problems.append(f"backend.{part} 需为字典，实际 {type(seg).__name__}")
                continue
            joints = seg.get("joints")
            if joints is None:
                continue
            if not isinstance(joints, list):
                problems.append(
                    f"backend.{part}.joints 需为列表，实际 {type(joints).__name__}")
                continue
            bad = [i for i, j in enumerate(joints) if not isinstance(j, dict)]
            if bad:
                problems.append(
                    f"backend.{part}.joints 条目需为字典（关节 name/限位四键），"
                    f"非字典条目下标 {bad}")

        # ---- 关节数（后续维度检查基准；结构不符的段取空表，类型问题已在上报）----
        n_arm_cfg = len(_backend_joint_cfgs(bcfg, "arm"))
        n_end_cfg = len(_backend_joint_cfgs(bcfg, "end"))

        # ---- home / 软限位维度 ----
        for key, n_ref in (("home_arm", n_arm_cfg), ("home_end", n_end_cfg)):
            v = jcfg.get(key)
            if v is None:
                if n_ref:
                    problems.append(f"joyarm.{key} 缺失（须 {n_ref} 元列表）")
                continue
            size = np.asarray(v, dtype=float).reshape(-1).shape[0]
            if size != n_ref:
                problems.append(f"joyarm.{key} 维度 {size} ≠ backend 关节数 {n_ref}")
        for key, n_ref in (("joint_limits_soft_arm", n_arm_cfg),
                           ("joint_limits_soft_end", n_end_cfg)):
            soft = jcfg.get(key)
            if soft is None:
                if n_ref:
                    problems.append(f"joyarm.{key} 缺失（neutral 取软限位中点，必配）")
                continue
            if not isinstance(soft, dict):
                problems.append(f"joyarm.{key} 需为四键字典（q_min/q_max/dq_max/tau_max）")
                continue
            for k in ("q_min", "q_max", "dq_max", "tau_max"):
                if k not in soft:
                    if k in ("q_min", "q_max"):
                        problems.append(
                            f"joyarm.{key}.{k} 缺键（位置软限位必配；缺键会使"
                            f"随机采样与数值 IK 溢出崩溃、neutral 发散）")
                    continue     # dq_max/tau_max 缺键由 soft_limits_from_cfg 回退 ±∞（无崩溃路径）
                arr = np.asarray(soft[k], dtype=float).reshape(-1)
                if arr.ndim == 0 or arr.shape[0] == 1:
                    continue     # 标量：全关节统一广播
                if arr.shape[0] != n_ref:
                    problems.append(
                        f"joyarm.{key}.{k} 长度 {arr.shape[0]} ≠ 关节数 {n_ref}"
                        f"（逐关节需 n 元列表，或写标量统一全关节）")

        # ---- 六域注册名 + 管线起搏键 ----
        for domain, (cfg_key, registry) in _DOMAIN_REGISTRIES.items():
            try:
                specs = _parse_domain_specs(cfg_key, robotics_cfg.get(cfg_key))
            except ValueError as e:
                problems.append(f"{e}")
                continue
            for reg_name, params in specs:
                if reg_name not in registry:
                    problems.append(
                        f"robotics.{cfg_key} 注册名『{reg_name}』未在子包 REGISTRY"
                        f"（可用：{sorted(registry) or '无（尚未实现注册）'}）")
                if domain == "traj":
                    try:
                        _check_pacing_hz(cfg_key, reg_name, "plan_hz", params.get("plan_hz"))
                    except ValueError as e:
                        problems.append(f"{e}")
                elif domain == "ctrl":
                    try:
                        _check_pacing_hz(cfg_key, reg_name, "ctrl_hz", params.get("ctrl_hz"))
                    except ValueError as e:
                        problems.append(f"{e}")

        # ---- cfg.backend ↔ URDF 对比 ----
        problems.extend(self._backend_urdf_problems(bcfg))

        if problems:
            raise ValueError(
                "joyarm.py - JoyArm._check_cfg：配置自检未通过：\n"
                + "\n".join(f"  - {p}" for p in problems))

    # =============== cfg.backend ↔ URDF 对比 ===============
    def _backend_urdf_problems(self, bcfg: dict) -> List[str]:
        """cfg.backend 关节登记与 URDF 的一致性问题清单（任一不符抛异常）。
        """
        problems: List[str] = []
        arm_cfgs = _backend_joint_cfgs(bcfg, "arm")
        end_cfgs = _backend_joint_cfgs(bcfg, "end")
        if not arm_cfgs:
            return ["backend.arm.joints 为空（本体电机列表必配）"]
        if not os.path.isfile(self._urdf_path):
            return [f"URDF 文件不存在：{self._urdf_path}"]
        joints = _urdf_joint_table(self._urdf_path)
        links = _urdf_link_names(self._urdf_path)

        # URDF joint/link 名无重复
        dup_j = sorted({n for n in joints if list(joints).count(n) > 1})
        dup_l = sorted({n for n in links if links.count(n) > 1})
        if dup_j:
            problems.append(f"URDF joint 名重复：{dup_j}")
        if dup_l:
            problems.append(f"URDF link 名重复：{dup_l}")

        # cfg 关节名无重复 + 四键限位与 URDF <limit> 逐键容差 + q_min ≤ q_max
        for part, joint_cfgs in (("arm", arm_cfgs), ("end", end_cfgs)):
            names = [str(j.get("name", "")) for j in joint_cfgs]
            dup = sorted({n for n in names if names.count(n) > 1})
            if dup:
                problems.append(f"backend.{part}.joints 关节名重复：{dup}")
            for j in joint_cfgs:
                jn = str(j.get("name", ""))
                u = joints.get(jn)
                if u is None:
                    problems.append(f"backend.{part}.joints 关节『{jn}』在 URDF 中不存在")
                    continue
                if u["type"] not in _ACTIVE_JOINT_TYPES:
                    problems.append(
                        f"backend.{part}.joints 关节『{jn}』在 URDF 中为 {u['type']}"
                        f"（fixed 关节不允许登记）")
                    continue
                if u["mimic"]:
                    problems.append(
                        f"backend.{part}.joints 关节『{jn}』在 URDF 中为 mimic 从动关节"
                        f"（不允许登记）")
                    continue
                for key in ("q_min", "q_max", "dq_max", "tau_max"):
                    cfg_v, urdf_v = j.get(key), u[key]
                    if cfg_v is None or urdf_v is None:
                        continue     # cfg 缺键由硬限位构建回退；URDF 缺属性（如 continuous）不比对
                    if abs(float(cfg_v) - urdf_v) > _URDF_LIMIT_TOL:
                        problems.append(
                            f"backend.{part}.joints 关节『{jn}』{key} 与 URDF <limit> 不一致"
                            f"（cfg={cfg_v}，URDF={urdf_v}，容差 {_URDF_LIMIT_TOL}）")
                try:
                    if float(j["q_min"]) > float(j["q_max"]):
                        problems.append(
                            f"backend.{part}.joints 关节『{jn}』q_min={j['q_min']}"
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
                f"URDF 驱动关节 {other} 命名不符约定（arm 须 joint1~jointN、"
                f"end 须 joint_end1~joint_endM）")
        expected_arm = [f"joint{i}" for i in range(1, len(arm_driven) + 1)]
        if arm_driven != expected_arm:
            problems.append(
                f"URDF arm 驱动关节编号/顺序异常：{arm_driven}（期望 {expected_arm}）")
        expected_end = [f"joint_end{i}" for i in range(1, len(end_driven) + 1)]
        if end_driven != expected_end:
            problems.append(
                f"URDF end 驱动关节编号/顺序异常：{end_driven}（期望 {expected_end}）")
        cfg_arm_names = [str(j.get("name", "")) for j in arm_cfgs]
        cfg_end_names = [str(j.get("name", "")) for j in end_cfgs]
        if cfg_arm_names != arm_driven:
            problems.append(
                f"backend.arm.joints 与 URDF arm 驱动关节不同名同序：\n"
                f"      config：{cfg_arm_names}\n      URDF  ：{arm_driven}")
        if cfg_end_names != end_driven:
            problems.append(
                f"backend.end.joints 与 URDF end 驱动关节不同名同序：\n"
                f"      config：{cfg_end_names}\n      URDF  ：{end_driven}")
        unregistered = [n for n in driven
                        if n not in cfg_arm_names and n not in cfg_end_names]
        if unregistered:
            problems.append(
                f"URDF 驱动关节 {unregistered} 未在 cfg.backend.joints 登记"
                f"（cfg 只登记非 fixed、非 mimic 的驱动关节，且须全部登记）")
        return problems

    # =============== pin 缩减模型 ===============
    def _build_pin_model(self, urdf_path: str, arm_joint_names: List[str]) -> pin.Model:
        """URDF 全量模型 → 锁定非 cfg.arm 名单活动关节（含 end 与 mimic 等，锁定位形取零位）→ 缩减模型。

        锁定判据用 cfg 名单而非硬编码关节名集合（型号无关）；所有 link/帧保留，保证整臂质量惯量正确。
        构建后校验 `frame_end` 帧存在与关节同名同序。

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

    # =============== 特征位形解析（home 行） ===============
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

    # =============== tcp_limits 解析（先校验后解析） ===============
    @staticmethod
    def _parse_tcp_limits(tl: Optional[dict]) -> TcpLimits:
        """解析 `joyarm.tcp_limits` 段（先格式校验、后经 utils 解析）。

        校验：`workspace_box` 为 (3,2)（每行 `[min,max]`；兼容 (2,3) 布局、内部统一转置为 (3,2)）且每轴 min < max；
        `v_lin_max/v_ang_max/f_max/t_max` 非空且 > 0；违规抛错。

        :raises ValueError: 格式或数值违规。
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
            if arr.shape == (2, 3):     # [min 行, max 行] → 每轴 [min, max]
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

    # =============== FSM / 监视 / 到位容差参数解析 ===============
    def _parse_fsm_params(self, jcfg: dict) -> None:
        """解析 `joyarm.MotionFSM` / `joyarm.arrival_tol` / `joyarm.runtime.move_wait_timeout`。

        :raises ValueError: `damping_kd` / `damping_k_gravity` 未配置或非法；其余参数显式给出但非法。
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

        # DAMPING 状态增益（kd 速度阻尼、k_gravity 重力补偿率）
        def _broadcast_cfg(key: str, ok: Callable[[np.ndarray], bool], req: str) -> np.ndarray:
            raw = fsm.get(key)
            if raw is None:
                raise ValueError(
                    f"joyarm.py - JoyArm._parse_fsm_params：joyarm.MotionFSM.{key}"
                    f" 未配置（{req}；DAMPING 态必需）")
            try:
                arr = np.asarray(raw, dtype=float)
            except (TypeError, ValueError):
                arr = np.full(1, np.nan)
            if arr.ndim == 0:
                arr = np.full(self._n_arm, float(arr))
            else:
                arr = arr.reshape(-1)
                if arr.shape[0] != self._n_arm:
                    raise ValueError(
                        f"joyarm.py - JoyArm._parse_fsm_params：{key} 长度 "
                        f"{arr.shape[0]} ≠ arm 关节数 {self._n_arm}")
            if not ok(arr):
                raise ValueError(
                    f"joyarm.py - JoyArm._parse_fsm_params：{key}={raw!r}"
                    f" 非法（{req}）")
            return _freeze_arrays(arr)

        self._damping_kd = _broadcast_cfg(
            "damping_kd", lambda a: bool(np.isfinite(a).all() and (a > 0).all()),
            "正有限数，标量广播或 n_arm 元列表")
        self._damping_k_gravity = _broadcast_cfg(
            "damping_k_gravity",
            lambda a: bool(np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()),
            "[0,1] 有限数（1=完全补偿），标量广播或 n_arm 元列表")

        # 到位容差（缺省 0.05 / 1e-3 / 1e-2 标量广播）
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
        try:
            self._arrival_tol_pos = float(arm_at.get("pos", 1e-3))
            self._arrival_tol_rot = float(arm_at.get("rot", 1e-2))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"joyarm.py - JoyArm._parse_fsm_params：arrival_tol.arm.pos/rot="
                f"{arm_at.get('pos')!r}/{arm_at.get('rot')!r} 非法（须为正有限数）") from e
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

    # =============== 初始化完成 info ===============
    def _log_init_info(self) -> None:
        """自检通过、成员装配完成后输出初始化完成 info 与配置摘要（最低字段集）。"""

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
            f"damping_kd={_r(self._damping_kd)}；damping_k_gravity={_r(self._damping_k_gravity)}",
        ]
        logger.info("%s 初始化和自检完毕（仅必要校验，不确保完全无误）！\n%s",
                    self._model, "\n".join(lines))

    # ============================================================
    # 三、六域成员管理
    # ============================================================
    def _build_domains(self, robotics_cfg: dict) -> None:
        """按 cfg `robotics` 段实例化六域成员并设默认激活。

        前置：cfg 自检已通过。构造参数透传。

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
        """取某域成员字典；域未配置任何成员时抛 `RuntimeError`。"""
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
        """经 `_<域>_solvers` 反查激活对象对应的注册名。"""
        solvers = self._require_solvers(domain)
        act = getattr(self, f"_activated_{domain}_solver")
        for n, obj in solvers.items():
            if obj is act:
                return n
        raise RuntimeError(
            f"joyarm.py - _activated_name_of：{domain} 域激活指针失效（不应发生）")

    def _switch_solver(self, domain: str, name: str) -> bool:
        """热切换统一入口（协议按域分发；名称不在成员中 warn 并保持原激活）。"""
        solvers = self._require_solvers(domain)
        if name not in solvers:
            logger.warning(
                "joyarm.py - set_%s_solver：『%s』未在该域已加载成员中"
                "（保持原激活不变）；可用：%s", domain, name, sorted(solvers))
            return False
        new = solvers[name]
        if domain == "traj":
            # 与 plan/sample/step 串行：试规划成功后才替换，不共享可被他处提前清除的暂停事件。
            with self._motion_lock:
                try:
                    ok = bool(new.plan(self, self._target_trajframe))
                except Exception as e:
                    logger.warning("joyarm.py - set_traj_solver：新规划器『%s』试规划异常"
                                   "（保持原激活不变）：%s", name, e)
                    return False
                if not ok:
                    logger.warning("joyarm.py - set_traj_solver：新规划器『%s』试规划失败"
                                   "（保持原激活不变）", name)
                    return False
                self._activated_traj_solver = new
                self._plan_hz = self._plan_hz_by_name[name]
                if self._plan_periodic is not None:
                    self._plan_periodic.set_hz(self._plan_hz)
        elif domain == "ctrl":
            # 这里只选择算法；MOTION 的进入/维持拍负责核验 MODE 与使能。
            # 先等待在途 step 完成，再撤销许可并换指针，防新控制器沿用旧模式下发。
            if not isinstance(new.MODE, ControlMode):
                logger.warning("joyarm.py - set_ctrl_solver：控制器『%s』MODE 非 ControlMode", name)
                return False
            with self._motion_lock:
                self._motion_ready = None
                self._activated_ctrl_solver = new
                self._ctrl_hz = self._ctrl_hz_by_name[name]
                if self._ctrl_periodic is not None:
                    self._ctrl_periodic.set_hz(self._ctrl_hz)
        else:
            setattr(self, f"_activated_{domain}_solver", new)
        return True

    # ============================================================
    # 四、plan/ctrl 两线程与 ctrl 线程互斥豁免
    # ============================================================
    def _start_motion_thread(self, event: SafetyEvent) -> bool:
        """首帧规划 → 硬件就绪检查 → 启动 plan/ctrl；仅 FSM 调用。

        失败时回收部分启动的线程，不修改事件；异常交给周期线程记录。
        停旧线程不持 `_motion_lock`，避免 join 等待正在取同一把锁的 worker。
        """
        self._stop_motion_thread()
        if self._ctrl_periodic is not None or self._plan_periodic is not None:
            return False                       # 旧线程尚未退出，下拍重试
        started = False
        try:
            with self._motion_lock:
                planner = self._activated_of("traj")
                controller = self._activated_of("ctrl")
                if not planner.plan(self, self._target_trajframe):
                    return False               # 首帧尚未就绪，不提交 MOTION
                if not self._ensure_arm_ready(controller.MODE, event):
                    return False
                self._plan_periodic = _PeriodicThread(self._tick_plan, self._plan_hz, name="joyarm-plan")
                self._ctrl_periodic = _PeriodicThread(self._tick_ctrl, self._ctrl_hz, name="joyarm-ctrl")
                self._plan_periodic.start()
                self._ctrl_thread = self._ctrl_periodic.prepare()  # 起跑前登记 ctrl 豁免句柄
                self._ctrl_periodic.start()
                self._motion_ready = controller
                started = True
        finally:
            if not started:
                self._stop_motion_thread()     # 仅做资源清理，join 时不持运动锁
        logger.info("joyarm.py - _start_motion_thread：plan/ctrl 两线程已启动"
                    "（plan_hz=%s，ctrl_hz=%s，控制器 %s）",
                    self._plan_hz, self._ctrl_hz, type(controller).__name__)
        return True

    def _stop_motion_thread(self) -> None:
        """撤销下发许可并停 plan/ctrl（先 ctrl 后 plan，join 不持运动锁）。

        超时保留线程句柄与 ctrl 豁免句柄，确认退出前不允许另起新线程；目标帧保留。
        """
        self._motion_ready = None
        for attr in ("_ctrl_periodic", "_plan_periodic"):
            worker = getattr(self, attr)
            if worker is None:
                continue
            try:
                worker.stop()
            except RuntimeError as e:
                logger.warning("joyarm.py - _stop_motion_thread：%s（句柄保留，待其退出后重试）", e)
                continue
            setattr(self, attr, None)
        if self._ctrl_periodic is None:
            self._ctrl_thread = None

    def _tick_plan(self) -> None:
        """与规划器切换及采样串行；单拍失败不修改事件，异常由周期线程记录。"""
        with self._motion_lock:
            if not self._fsm_shutdown.is_set() and self._backend.is_connected_arm is True:
                self._activated_traj_solver.plan(self, self._target_trajframe)

    def _ctrl_ready(self) -> bool:
        """只由 FSM 授予当前控制器许可；保护事件、断连或最终收尾立即禁止下发。"""
        return (not self._fsm_shutdown.is_set()
                and self._backend.is_connected_arm is True
                and self._safe_event is SafetyEvent.NORMAL
                and self._current_motion_state is MotionState.MOTION
                and self._motion_ready is not None
                and self._motion_ready is self._activated_ctrl_solver)

    def _tick_ctrl(self) -> None:
        """就绪时读状态、采样、执行控制；缺少输入则跳拍，不修改安全事件。"""
        with self._motion_lock:
            if not self._ctrl_ready():
                return
            state = self._backend.get_state_arm()
            if state is None:
                return
            frame = self._activated_traj_solver.sample(time.time())
            if frame is None:
                return
            if self._ctrl_ready():              # 读取期间可能收到保护事件
                self._activated_ctrl_solver.step(self, frame, state)

    def _is_internal_thread(self) -> bool:
        """当前线程是否为 JoyArm 自身内部的 ctrl 线程（互斥豁免判定）。"""
        return self._ctrl_thread is not None and threading.current_thread() is self._ctrl_thread

    def _fsm_guard(self, caller: str) -> bool:
        """FSM 外部指令互斥检查（arm 侧指令类方法用）；放行返回 True。

        FSM 运行时外部调用被拦截；ctrl 线程仅在当前控制器已就绪且处于正常 MOTION 时放行。
        """
        if self._is_internal_thread():
            return self._ctrl_ready()
        if self._fsm_running():
            logger.warning(
                "joyarm.py - %s：FSM 运行中，外部线程的 arm 侧指令被互斥拦截"
                "（本次空操作）", caller)
            return False
        return True

    # ============================================================
    # 五、MotionFSM：线程体 / 单拍循环 / 状态进出与维持
    # ============================================================
    def _fsm_running(self) -> bool:
        """FSM 线程是否在运行。"""
        return self._fsm_periodic is not None and self._fsm_periodic.is_alive()

    def _fsm_worker(self) -> None:
        """FSM 单拍；收尾期间在途自愈返回后再失能，防其晚于 stop 的失能完成。"""
        try:
            self._fsm_tick()
        finally:
            if self._fsm_shutdown.is_set():
                self._backend.disable_arm()

    def _fsm_tick(self) -> None:
        """消费事件及目标 → 进入/维持 → 成功才提交状态；不作安全事件判定。"""
        if self._fsm_shutdown.is_set():
            return
        if self._backend.is_connected_arm is not True:
            if not self._fsm_disc_warned:
                logger.warning(
                    "joyarm.py - _fsm_tick：arm 未连接（is_connected_arm=%s），"
                    "FSM 挂起：暂停迁移与维持下发，等待重连后自动恢复",
                    self._backend.is_connected_arm)
                self._fsm_disc_warned = True
            return
        self._fsm_disc_warned = False
        with self._event_lock:
            event, tgt = self._safe_event, self._target_motion_state
        cur = self._current_motion_state
        nxt = _ROUTE[cur].get(tgt, cur)
        waiting_home = (cur is MotionState.HOME and tgt in (MotionState.ZERO, MotionState.MOTION)
                        and (not self._home_arrived or not self.is_in_q_arm(self._home_arm)))
        if cur is tgt or waiting_home:
            if (cur is MotionState.MOTION
                    and any(worker is None or not worker.is_running()
                            for worker in (self._ctrl_periodic, self._plan_periodic))):
                # 重建管线。
                self._start_motion_thread(event)
            else:
                self._maintain_motion_state(cur, event)
            return
        self._exit_motion_state(cur)
        if cur is MotionState.MOTION and self._ctrl_periodic is not None:
            return                             # 控制线程尚未退出，不进入下一状态
        if not self._enter_motion_state(nxt, event):
            return
        with self._motion_lock:
            allowed = self._arm_action_allowed(event)
            if allowed:
                if cur is MotionState.ZERO:
                    self._zero_moving = False
                elif cur is MotionState.HOLD:
                    self._hold_q = None
                self._current_motion_state = nxt
            else:
                self._motion_ready = None
        if not allowed:
            if nxt is MotionState.MOTION:
                self._stop_motion_thread()
            return
        logger.info("joyarm.py - _fsm_tick：状态迁移 %s → %s（目标 %s）",
                    cur.name, nxt.name, tgt.name)

    def _arm_action_allowed(self, event: SafetyEvent) -> bool:
        """阻塞 I/O 前后复查事件及最终停机标志，防按过期目标继续使能/下发。"""
        return not self._fsm_shutdown.is_set() and self._safe_event is event

    def _ensure_arm_ready(self, mode: ControlMode, event: SafetyEvent) -> bool:
        """状态内模式/使能自愈（调用方持运动锁）；成功才允许运动下发。

        每拍只读一次模式；不一致才写，先模式后使能。失败交由 FSM 下拍重试，
        不清故障、不修改安全事件；未知使能不得当作已使能。
        """
        self._motion_ready = None
        if not self._arm_action_allowed(event):
            return False
        if self._backend.is_connected_arm is not True:
            return False
        actual = self._backend.get_mode_arm()
        if not self._arm_action_allowed(event):
            return False
        if actual != mode:
            if self._backend.set_mode_arm(mode) != 1:
                return False
        if not self._arm_action_allowed(event):
            return False
        if self._backend.is_enabled_arm is not True:
            if self._backend.enable_arm() != 1:
                return False
        return (self._arm_action_allowed(event)
                and self._backend.is_enabled_arm is True)

    def _enter_motion_state(self, state: MotionState, event: SafetyEvent) -> bool:
        """进入动作成功才提交状态；失败不清理旧 HOLD/ZERO 缓存，不盲目恢复旧指令。"""
        if state is MotionState.MOTION:
            return self._start_motion_thread(event)
        with self._motion_lock:
            if not self._arm_action_allowed(event):
                return False
            if self._backend.is_connected_arm is not True:
                return False
            if state is MotionState.DISABLE:
                return self._disable_clear_disable_arm()
            hold_q = None
            if state is MotionState.HOLD:
                st = self._backend.get_state_arm()
                if st is None or st.q is None:
                    return False
                hold_q = np.asarray(st.q, dtype=float).copy()
                if hold_q.shape != (self._n_arm,) or not np.isfinite(hold_q).all():
                    return False
            mode = ControlMode.MIT if state is MotionState.DAMPING else ControlMode.POSITION
            if not self._ensure_arm_ready(mode, event):
                return False
            if state is MotionState.ZERO:
                self._zero_moving = True
                self._backend.send_position_arm(self._zero_arm)
            elif state is MotionState.HOME:
                self._home_arrived = False
            elif state is MotionState.HOLD:
                self._hold_q = hold_q
            elif (state is MotionState.DAMPING and self._activated_dyn_solver is None
                    and bool((self._damping_k_gravity > 0).any())):
                logger.warning("joyarm.py - _enter_motion_state：dynamics 域未配置，"
                               "DAMPING 重力补偿不可用，退化为纯阻尼")
            return True

    def _exit_motion_state(self, state: MotionState) -> None:
        """MOTION 先撤销许可、停线程；ZERO/HOLD 缓存仅在迁移成功后清理。"""
        if state is MotionState.MOTION:
            self._stop_motion_thread()       # join 不得持运动锁

    def _maintain_motion_state(self, state: MotionState, event: SafetyEvent) -> bool:
        """每拍先确认模式/使能再维持；ZERO 到位与 DISABLE 只保持失能（确知已失能则静默不发帧）。"""
        with self._motion_lock:
            if not self._arm_action_allowed(event):
                return False
            if state is MotionState.DISABLE or (state is MotionState.ZERO and not self._zero_moving):
                self._motion_ready = None
                if self._backend.is_enabled_arm is False:
                    return True          # 确知已失能：静默维持，免每拍刷屏
                return self._backend.disable_arm() == 1
            controller = self._activated_of("ctrl") if state is MotionState.MOTION else None
            mode = (controller.MODE if controller is not None else
                    ControlMode.MIT if state is MotionState.DAMPING else ControlMode.POSITION)
            if not self._ensure_arm_ready(mode, event):
                return False
            if state is MotionState.ZERO:
                self._backend.send_position_arm(self._zero_arm)
                if self.is_in_q_arm(self._zero_arm):
                    if self._backend.disable_arm() != 1:
                        return False
                    self._zero_moving = False
                    logger.info("joyarm.py - _maintain_motion_state：ZERO 已到位并失能")
            elif state is MotionState.HOME:
                self._backend.send_position_arm(self._home_arm)
                arrived = self.is_in_q_arm(self._home_arm)
                if arrived and not self._home_arrived:
                    logger.info("joyarm.py - _maintain_motion_state：HOME 已到位")
                self._home_arrived = arrived
            elif state is MotionState.MOTION:
                self._motion_ready = controller
            elif state is MotionState.HOLD:
                if self._hold_q is None:
                    return False
                self._backend.send_position_arm(self._hold_q)
            elif state is MotionState.DAMPING:
                tau_ff = np.zeros(self._n_arm)            # 重力补偿前馈 k_gravity·G(q)；缺求解器或快照无效时置零（纯阻尼）
                dyn = self._activated_dyn_solver
                q = np.asarray(self._backend.joint_state_arm.q, dtype=float)
                if dyn is not None and bool(np.isfinite(q).all()):
                    tau_ff = self._damping_k_gravity * dyn.gravity(self, q)
                self._backend.send_mit_arm(q=np.zeros(self._n_arm), kp=np.zeros(self._n_arm), kd=self._damping_kd, tau=tau_ff)
            return True

    def _disable_clear_disable_arm(self) -> bool:
        """既有 DISABLE 进入序列；最后一次失能必须成功，不能仅凭调用完成提交状态。"""
        self._motion_ready = None
        self._backend.disable_arm()
        self._backend.clear_error_arm()
        return self._backend.disable_arm() == 1

    # ============================================================
    # 六、safe_monitor 与状态可视化线程体
    # ============================================================
    def _fresh_joint_state(self, family: str) -> Optional[JointState]:
        """缓存快照过期过滤：槽位未读取（t≤0）或 age 超 max(2×刷新周期) → None。

        被动只读 backend `joint_state_{family}` 快照槽位（不触发总线帧）；
        供监视器/可视化等消费点使用，需实时者需走 `get_state_*`（读时刷新）。
        """
        snap = getattr(self._backend, f"joint_state_{family}")
        hz = float(self._backend.cfg.get("state_refresh_hz", 20.0))
        if (snap is None or not np.isfinite(snap.t) or snap.t <= 0
                or not 0 <= time.time() - snap.t <= max(_MONITOR_NONE_TIMEOUT, 2.0 / hz)):
            return None
        return snap

    def _monitor_worker(self) -> None:
        """监视器单拍：只读采集 → 规则判断 → 命中则 `set_event`。

        只读：绝不调用控制/下发方法；事件经 set_event 通路驱动 FSM。
        仲裁：内置规则与自定义规则命中合并、统一取危险程度最高者；
        只升不降：命中事件危险程度高于当前事件才 set_event。
        """
        state_arm = self._fresh_joint_state("arm")
        state_end = self._fresh_joint_state("end")
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
            # 连续无新鲜快照（过期过滤判 None：读取失败/断连致快照陈旧）→ DISABLE
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

        # 合并取最危险：内置规则与自定义规则命中合并、统一取危险度最高
        chosen: Optional[SafetyEvent] = None
        if hard or custom:
            chosen = max(hard + custom, key=lambda e: _EVENT_DANGER[e])
        # 只升不降：仅当命中事件危险程度高于当前事件才 set_event，
        # 防止覆写并发设置的更高危险事件（外部 set_event 不受限，恢复 NORMAL 走外部通路）
        with self._event_lock:
            if chosen is not None and _EVENT_DANGER[chosen] > _EVENT_DANGER[self._safe_event]:
                self.set_event(chosen)

    # =============== 状态可视化线程体与面板数据 ===============
    def _state_viz_worker(self) -> None:
        """可视化单拍：被动采集刷新面板数据与 meshcat 模型姿态。

        数据一律取 backend 缓存快照槽位（被动只读）；静态参考不在此装配（`start_state_viz` 时一次性装配为 `/static.json`）。
        """
        state_arm = self._backend.joint_state_arm
        state_end = self._backend.joint_state_end
        # 快照 q/dq 与有效性单次读取判定（tcp 现算、动力学现算、轨迹绘制共用）
        q_arr = np.asarray(getattr(state_arm, "q", None), dtype=float).reshape(-1)
        dq_arr = np.asarray(getattr(state_arm, "dq", None), dtype=float).reshape(-1)
        snap_t = float(getattr(state_arm, "t", 0.0) or 0.0)
        q_ok = (snap_t > 0 and q_arr.size > 0
                and bool(np.all(np.isfinite(q_arr))))
        dq_ok = (snap_t > 0 and dq_arr.size > 0
                 and bool(np.all(np.isfinite(dq_arr))))

        snap = {
            "model": self._model,
            "t": round(time.time(), 3),
            "snapshot": {"arm": self._viz_snap_meta(state_arm),
                         "end": self._viz_snap_meta(state_end)},
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
            },
            "solvers": self._viz_solvers(),
            "traj_enabled": self._viz_traj_enabled,
        }
        # 末端位姿（激活 fkine 对快照 q 现算，只读）；速度 = 激活雅可比 × 快照关节速度；
        # q/dq 含 NaN（未读取占位）或域未配置 → 对应项占位（try/except 防 10 Hz 刷屏）
        if q_ok and self._activated_fkine_solver is not None:
            try:
                pose = self._activated_fkine_solver.solve(self, q_arr, self._frame_end)
                snap["tcp"] = {"position": _json_arr(pose.position),
                               "orientation": _json_arr(pose.orientation)}
            except Exception:
                pass
            if self._jac_solvers and dq_ok:
                try:
                    J = self.get_jac(q=q_arr)              # (6, n_arm) 只读雅可比
                    v = np.asarray(J, dtype=float) @ dq_arr
                    if isinstance(snap["tcp"], dict):
                        snap["tcp"]["twist"] = {"linear": _json_arr(v[:3]),
                                                "angular": _json_arr(v[3:])}
                except Exception:
                    pass
        self._viz_state = snap
        # meshcat 模型姿态刷新与轨迹绘制（初始化失败降级；q 无效时保持上一帧）。
        # display q 按可视化模型（arm+end）关节序从快照按名映射拼装——其有效性与 q_ok（arm 快照当拍有效）经族级缓存解耦；
        # 轨迹绘制入口因此保留 q_ok 与 fkine 已配置两条件：q_arr 无效或 fkine 域未配置时
        # _viz_draw_trajectory 内 fk.solve 会抛异常，必须整块跳过（否则 10 Hz warn 无限刷）。
        if self._viz_pin is not None:
            try:
                q_viz = self._viz_display_q(state_arm, state_end)
                if q_viz is not None:
                    self._viz_pin.display(q_viz)
                if (self._viz_traj_enabled and self._viz_traj_points > 0
                        and self._current_motion_state is MotionState.MOTION
                        and q_ok
                        and self._activated_fkine_solver is not None):
                    self._viz_draw_trajectory(q_arr)
                else:
                    self._viz_hide_trajectory()
                self._viz_zmq_fails = 0
            except zmq.error.ZMQError as e:
                # zmq 看门狗：套接字已置 RCVTIMEO，server 死亡时 recv 超时抛 Again 而非永久阻塞；
                # 连续失败计数达阈值则降级为仅 HTML 面板并停止 meshcat 步进
                self._viz_zmq_fails += 1
                if self._viz_zmq_fails >= _VIZ_ZMQ_FAIL_MAX:
                    logger.warning(
                        "joyarm.py - _state_viz_worker：meshcat 刷新连续 %s 次"
                        "zmq 超时/失败（server 可能已死亡），降级为仅 HTML 状态面板"
                        "并停止 meshcat 刷新", self._viz_zmq_fails)
                    self._viz_pin = None
                else:
                    logger.warning(
                        "joyarm.py - _state_viz_worker：meshcat 刷新 zmq 超时/失败"
                        "（%s/%s）：%s", self._viz_zmq_fails, _VIZ_ZMQ_FAIL_MAX, e)
            except Exception as e:
                logger.warning(
                    "joyarm.py - _state_viz_worker：meshcat 刷新失败：%s", e)

    @staticmethod
    def _viz_bool(v) -> Optional[bool]:
        """三状态布尔转 JSON 安全值。"""
        return None if v is None else bool(v)

    @staticmethod
    def _viz_group(state: JointState, names: tuple) -> dict:
        """被动关节快照槽位 → 面板分组 dict。

        :param state: 被动关节快照槽位（backend `joint_state_*` 属性本体）。
        :param names: 该组关节名 tuple（cfg 顺序 = q 向量顺序）。
        :return: `{names, q, dq, tau, temp_mos, temp_rotor, error}` 面板分组 dict（JSON 安全值）。
        """
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

    def _viz_snap_meta(self, snapshot: JointState) -> dict:
        """被动关节快照 → 面板元数据 `{t, age, state}`（快照时刻/距今时长/新鲜度标注）。

        纯算术标注、不过期隐藏（陈旧数据照常显示）；
        新鲜/陈旧阈值沿用 `max(0.1 s, 2/state_refresh_hz)`。

        :param snapshot: 本拍读取并用于填充面板数据的同一快照对象（单次读取）。
        :return: `{t, age, state}`——t≤0 时 `t/age` 为 `None`、state 为「未读取」。
        """
        t = float(getattr(snapshot, "t", 0.0) or 0.0)
        if t <= 0:
            return {"t": None, "age": None, "state": "未读取"}
        try:
            hz = float(self._backend.cfg.get("state_refresh_hz", 20.0))
            fresh_win = max(_MONITOR_NONE_TIMEOUT, 2.0 / hz)
        except (TypeError, ValueError, ZeroDivisionError):
            fresh_win = _MONITOR_NONE_TIMEOUT
        age = time.time() - t
        return {"t": _json_round(t), "age": _json_round(age),
                "state": "新鲜" if age <= fresh_win else "陈旧"}

    def _viz_solvers(self) -> dict:
        """组装求解器与管线分组：六域激活注册名 + plan/ctrl 频率 + 目标帧摘要。

        域未配置显示「（未配置）」占位；`_target_trajframe` 单次读入局部引用后只读访问。

        :return: `{六域: 激活注册名, plan_hz, ctrl_hz, target}` 分组 dict——target 为 `{kind, frames, t_last}` 摘要或 `None`。
        """
        out: dict = {}
        for d in _DOMAIN_REGISTRIES:
            if getattr(self, f"_{d}_solvers"):
                try:
                    out[d] = self._activated_name_of(d)
                except Exception:
                    out[d] = "（未配置）"
            else:
                out[d] = "（未配置）"
        out["plan_hz"] = _json_round(self._plan_hz)
        out["ctrl_hz"] = _json_round(self._ctrl_hz)
        target = self._target_trajframe
        if isinstance(target, TrajFrame):
            out["target"] = {"kind": "frame", "frames": 1,
                             "t_last": _json_round(target.time)}
        elif target:
            out["target"] = {"kind": "seq", "frames": len(target),
                             "t_last": _json_round(target[-1].time)}
        else:
            out["target"] = None
        return out

    def _viz_display_q(self, state_arm, state_end) -> Optional[np.ndarray]:
        """按可视化模型关节序拼装 display q（快照按关节名映射，无效族沿用缓存）。

        可视化模型为arm+end联动缩减模型（`start_state_viz` 构建）；
        每拍对快照做有效判定，有效则更新`_viz_q_cache`、无效沿用缓存「保持上一帧」。

        :param state_arm: arm 被动快照（`_state_viz_worker` 单次读取的局部引用）。
        :param state_end: end 被动快照（`_state_viz_worker` 单次读取的局部引用）。
        :return: 可视化模型关节序 display q `(nq_viz,)`；模型未构建或 arm 无可用值时 `None`。
        """
        if self._viz_model is None:
            return None
        # 名字 → q 向量下标：关节 0 为 "universe" 无自由度，i-1 才是 q 序
        #（_build_pin_model 已保证 nq == 关节数-1，全部单自由度关节）
        name_to_idx = {self._viz_model.names[i]: i - 1
                       for i in range(1, self._viz_model.njoints)}
        q_viz = np.zeros(self._viz_model.nq)
        for fam, state, joint_names in (
                ("arm", state_arm, self._joint_names_arm),
                ("end", state_end, self._joint_names_end)):
            arr = np.asarray(getattr(state, "q", None), dtype=float).reshape(-1)
            if (float(getattr(state, "t", 0.0) or 0.0) > 0
                    and arr.size > 0 and bool(np.all(np.isfinite(arr)))):
                self._viz_q_cache[fam] = arr.copy()    # 族级有效才更新缓存
            cached = self._viz_q_cache.get(fam)
            if cached is None:
                continue
            for name, val in zip(joint_names,
                                 np.asarray(cached, dtype=float).reshape(-1)):
                idx = name_to_idx.get(name)
                if idx is not None:
                    q_viz[idx] = val
        # arm 族从未有效（无缓存）→ 整拍不刷新（等价旧行为的 q_ok=False 分支）
        return q_viz if self._viz_q_cache.get("arm") is not None else None

    def _viz_static_payload(self) -> dict:
        """`start_state_viz` 时一次性装配 `/static.json` 五组静态载荷。

        限位四键（q_min/q_max/dq_max/tau_max）均逐关节全数组。
        cfg 深拷贝仅此一次，`ctrl_defaults` 与 `cfg_raw` 两组共用。

        :return: `{ident, limits, poses, ctrl_defaults, cfg_raw}` 五组 dict（纯 JSON 值）。
        """
        cfg = self.get_config()   # 深拷贝一次，供控制默认值与 cfg 原文两组共用

        def _group(key: str, build) -> dict:
            """单组装配兜底：异常置 `{"_error": ...}`（不影响其余组）。"""
            try:
                return build()
            except Exception as e:
                logger.warning("joyarm.py - _viz_static_payload：%s 组装配失败：%s", key, e)
                return {"_error": str(e)}

        return {
            "ident": _group("ident", self._viz_ident),
            "limits": _group("limits", self._viz_limits),
            "poses": _group("poses", self._viz_poses),
            "ctrl_defaults": _group("ctrl_defaults",
                                    lambda: self._viz_ctrl_defaults(cfg)),
            "cfg_raw": _group("cfg_raw", lambda: self._viz_cfg_raw(cfg)),
        }

    def _viz_ident(self) -> dict:
        """标识与关节分组：型号 / 末端帧 / URDF 路径 / 关节数与关节名（构造后不变量）。"""
        return {
            "model": self._model,
            "frame_end": self._frame_end,
            "urdf": self._urdf_path,
            "n_arm": self._n_arm,
            "n_end": self._n_end,
            "joint_names_arm": list(self._joint_names_arm),
            "joint_names_end": list(self._joint_names_end),
        }

    def _viz_limits(self) -> dict:
        """限位分组：arm/end 硬/软限位（四键逐关节全数组）+ tcp_limits。

        :return: `{arm, end, tcp}` 分组 dict。
        """
        def _lim_group(hard: Optional[JointLimits], soft: Optional[JointLimits]) -> Optional[dict]:
            """硬/软限位 → 对称摘要分组（限位四键均逐关节全数组）。"""
            if hard is None and soft is None:
                return None
            def _arr(lim: Optional[JointLimits], key: str) -> Optional[list]:
                return _json_arr(getattr(lim, key) if lim is not None else None)
            return {
                "hard_q_min": _arr(hard, "q_min"),
                "hard_q_max": _arr(hard, "q_max"),
                "hard_dq_max": _arr(hard, "dq_max"),
                "hard_tau_max": _arr(hard, "tau_max"),
                "soft_q_min": _arr(soft, "q_min"),
                "soft_q_max": _arr(soft, "q_max"),
                "soft_dq_max": _arr(soft, "dq_max"),
                "soft_tau_max": _arr(soft, "tau_max"),
            }

        tcp = self._tcp_limits
        box = tcp.workspace_box
        return {
            "arm": _lim_group(self._joint_limits_hard_arm, self._joint_limits_soft_arm),
            "end": None if not self._n_end
            else _lim_group(self._joint_limits_hard_end,
                            self._joint_limits_soft_end),
            "tcp": {
                "workspace_box": None if box is None else [
                    [_json_round(box[i][0]), _json_round(box[i][1])]
                    for i in range(3)],
                "v_lin_max": _json_round(tcp.v_lin_max),
                "v_ang_max": _json_round(tcp.v_ang_max),
                "f_max": _json_round(tcp.f_max),
                "t_max": _json_round(tcp.t_max),
            },
        }

    def _viz_poses(self) -> dict:
        """特征位形分组：zero/neutral/home × arm/end（空数组照传，页面跳过）。"""
        return {
            "zero_arm": _json_arr(self._zero_arm),
            "neutral_arm": _json_arr(self._neutral_arm),
            "home_arm": _json_arr(self._home_arm),
            "zero_end": _json_arr(self._zero_end),
            "neutral_end": _json_arr(self._neutral_end),
            "home_end": _json_arr(self._home_end),
        }

    def _viz_ctrl_defaults(self, cfg) -> dict:
        """控制默认值分组：cfg（深拷贝）+ backend.cfg 映射 + 已解析成员。

        只走公开面取数，且只含跨型号通用键（型号专用运行键不入本组）。

        :param cfg: `get_config()` 深拷贝（`_viz_static_payload` 传入）。
        :return: `{mit, pos_vel, default_mode, backend_params, fsm_hz, monitor_hz,
            damping_kd, damping_k_gravity, arrival_tol, move_wait_timeout}` 分组 dict（纯 JSON 值）。
        """
        def _joint_gain(fam: str, seg: str, keys: tuple) -> dict:
            """walk cfg `backend.<fam>.joints[].<seg>.<keys>` → 逐关节数组。

            条目/键缺失该元素 `null`；整族无条目 → 各键整组 `null`。
            """
            out = {k: [] for k in keys}
            joints = (((cfg.get("backend") or {}).get(fam) or {})
                      .get("joints") or [])
            for entry in joints:
                segcfg = (entry.get(seg) or {}) if isinstance(entry, dict) else {}
                for k in keys:
                    out[k].append(_json_round(segcfg.get(k)
                                              if isinstance(segcfg, dict) else None))
            return {k: (v if v else None) for k, v in out.items()}

        def _default_mode(fam: str) -> Optional[str]:
            """cfg `backend.<fam>.default_mode`（缺省 `null`）。"""
            v = ((cfg.get("backend") or {}).get(fam) or {}).get("default_mode")
            return None if v is None else str(v)

        bcfg_pub = self._backend.cfg               # 公开 MappingProxy（只读）
        return {
            "mit": {fam: _joint_gain(fam, "MIT", ("kp", "kd"))
                    for fam in ("arm", "end")},
            "pos_vel": {fam: _joint_gain(
                fam, "POS_VEL",
                ("pos_kp", "pos_ki", "vel_kp", "vel_ki", "vlim", "flim"))
                for fam in ("arm", "end")},
            "default_mode": {"arm": _default_mode("arm"),
                             "end": _default_mode("end")},
            "backend_params": {
                "write_settle": _json_round(bcfg_pub.get("write_settle")),
                "state_refresh_hz": _json_round(bcfg_pub.get("state_refresh_hz")),
            },
            "fsm_hz": _json_round(self._fsm_hz),
            "monitor_hz": _json_round(self._monitor_hz),
            "damping_kd": _json_arr(self._damping_kd),
            "damping_k_gravity": _json_arr(self._damping_k_gravity),
            "arrival_tol": {
                "arm_q": _json_arr(self._arrival_tol_q_arm),
                "end_q": _json_arr(self._arrival_tol_q_end),
                "pos": _json_round(self._arrival_tol_pos),
                "rot": _json_round(self._arrival_tol_rot),
            },
            "move_wait_timeout": _json_round(self._move_wait_timeout),
        }

    def _viz_cfg_raw(self, cfg) -> str:
        """cfg 原文分组（净化版）。

        :param cfg: `get_config()` 深拷贝（`_viz_static_payload` 传入）。
        :return: yaml 字符串（页面折叠区原文显示，长行横向滚动）。
        """
        return yaml.dump(self._sanitize_cfg_for_viz(cfg),
                         allow_unicode=True, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _sanitize_cfg_for_viz(cfg: dict) -> dict:
        """cfg 净化副本。

        删除项：`joyarm.arm_mdh`、`joyarm.T_linkn_endtcp`、backend.arm/end的 `channel/protocol/baud_rate`、
        backend.arm/end.joints[] 各条目的 `motor_id/feedback_id/model`；缺键/缺段静默跳过（兼容缺段型号）。

        :param cfg: `get_config()` 深拷贝。
        :return: 净化后的新 dict（入参不被修改）。
        """
        clean = copy.deepcopy(cfg)
        jcfg = clean.get("joyarm")
        if isinstance(jcfg, dict):
            jcfg.pop("arm_mdh", None)
            jcfg.pop("T_linkn_endtcp", None)
        bcfg = clean.get("backend")
        if isinstance(bcfg, dict):
            for fam in ("arm", "end"):
                fcfg = bcfg.get(fam)
                if not isinstance(fcfg, dict):
                    continue
                for key in ("channel", "protocol", "baud_rate"):
                    fcfg.pop(key, None)
                joints = fcfg.get("joints")
                if isinstance(joints, list):
                    for entry in joints:
                        if isinstance(entry, dict):
                            for key in ("motor_id", "feedback_id", "model"):
                                entry.pop(key, None)
        return clean

    def _viz_draw_trajectory(self, q_cur: Optional[np.ndarray] = None) -> None:
        """轨迹显示（可选）：从激活规划器采样轨迹点 + 当前/目标位姿绘制。

        采样规则：单帧目标取区间 [当前时刻, 目标 t]、多帧目标取 [当前时刻, 末帧 t]；
        取 `_viz_traj_points` 点（start 可配，默认 200）；区间长度 ≤ 0.1 s 不采样不绘制。
        需运动状态机已启动且处于 MOTION 态（由调用方判定）。

        :param q_cur: 调用方传入的被动快照 q `(n_arm,)`（当前位姿球现算用；`None` 时跳过）。
        """
        planner = self._activated_traj_solver
        target = self._target_trajframe
        frames = [target] if isinstance(target, TrajFrame) else list(target or [])
        viewer = self._viz_pin.viewer
        if not frames:
            for key in ("traj", "traj_cur", "traj_tgt"):   # 标记球一并清（防旧冻结在屏上）
                viewer[key].delete()
            return
        t_end = frames[-1].time
        t_now = time.time()
        if t_end is None or t_end - t_now <= 0.1:
            for key in ("traj", "traj_cur", "traj_tgt"):   # 同上：早退也全清三键
                viewer[key].delete()
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
        # frame_end 当前位姿与目标位姿标记（球；当前位姿取调用方传入的被动快照 q）
        if q_cur is not None:
            p_cur = np.asarray(
                fk.solve(self, q_cur, self._frame_end).position, dtype=float)
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
        """清除 meshcat 上的轨迹显示。

        正常抛异常：delete 走 zmq send+recv，server 死亡时的超时异常必须到达 zmq 看门狗才能计数降级。
        """
        if self._viz_pin is None:
            return
        viewer = self._viz_pin.viewer
        for key in ("traj", "traj_cur", "traj_tgt"):
            viewer[key].delete()

    # ============================================================
    # 七、公开接口区
    # ============================================================

    # ============================================================
    # 7.1 配置与基本信息：get_config / 只读 @property
    # ============================================================
    def get_config(self) -> dict:
        """返回当前 config 配置的深拷贝。"""
        return copy.deepcopy(self._cfg)

    @property
    def pin_model(self) -> pin.Model:
        """pinocchio 缩减构型模型（`nq = n_arm`，只读共享）。"""
        return self._pin_model

    @property
    def n_arm(self) -> int:
        """本体关节数（只读）。"""
        return self._n_arm

    @property
    def n_end(self) -> int:
        """末端电机数（只读）。"""
        return self._n_end

    @property
    def joint_limits_hard_arm(self) -> Optional[JointLimits]:
        """本体硬限位。

        只读引用：返本体零拷贝，外部改动抛异常。
        """
        return self._joint_limits_hard_arm

    @property
    def joint_limits_hard_end(self) -> Optional[JointLimits]:
        """末端硬限位。

        只读引用：返末端零拷贝，外部改动抛异常。
        """
        return self._joint_limits_hard_end

    @property
    def joint_limits_soft_arm(self) -> Optional[JointLimits]:
        """本体软限位（只读引用）。"""
        return self._joint_limits_soft_arm

    @property
    def joint_limits_soft_end(self) -> Optional[JointLimits]:
        """末端软限位（只读引用）。"""
        return self._joint_limits_soft_end

    @property
    def zero_arm(self) -> np.ndarray:
        """本体零位 `(n_arm,)`（只读引用）。"""
        return self._zero_arm

    @property
    def zero_end(self) -> np.ndarray:
        """末端零位 `(n_end,)`（只读引用）。"""
        return self._zero_end

    @property
    def neutral_arm(self) -> np.ndarray:
        """本体数值求解默认初值 `(n_arm,)`（只读引用）。"""
        return self._neutral_arm

    @property
    def neutral_end(self) -> np.ndarray:
        """末端数值求解默认初值 `(n_end,)`（只读引用）。"""
        return self._neutral_end

    @property
    def home_arm(self) -> np.ndarray:
        """本体 home 位形 `(n_arm,)`（只读引用）。"""
        return self._home_arm

    @property
    def home_end(self) -> np.ndarray:
        """末端 home 位形 `(n_end,)`（只读引用）。"""
        return self._home_end

    @property
    def tcp_limits(self) -> TcpLimits:
        """末端空间限位（只读引用，内部数组已只读化）。"""
        return self._tcp_limits

    @property
    def plan_hz(self) -> Optional[float]:
        """激活规划器的规划线程频率 Hz（切换求解器时更新；外部只读）。"""
        return self._plan_hz

    @property
    def ctrl_hz(self) -> Optional[float]:
        """激活控制器的控制线程频率 Hz（切换求解器时更新；外部只读）。"""
        return self._ctrl_hz

    def __repr__(self) -> str:
        act = {d: (self._activated_name_of(d) if getattr(self, f"_{d}_solvers") else "-")
               for d in _DOMAIN_REGISTRIES}
        return (f"<{type(self).__name__} model={self._model!r} n_arm={self._n_arm} "
                f"n_end={self._n_end} frame_end={self._frame_end!r} "
                f"solvers={act} fsm={self._current_motion_state.name}>")

    # ============================================================
    # 7.2 关节查询与随机采样
    # ============================================================
    def get_joint_names_arm(self, joint: Union[int, str] = "all") -> Optional[Union[str, tuple]]:
        """查本体关节名。

        :param joint: `"all"`（默认）返回全部关节名 tuple（cfg 顺序 = q 向量顺序）；传入 1 基序号（cfg 列表位置）返回该关节名。
        :return: 关节名或名 tuple；越界 `[1, n_arm]` 或类型非法 warn 并返回 `None`。
        """
        return self._get_joint_names(self._joint_names_arm, joint, "arm")

    def get_joint_names_end(self, joint: Union[int, str] = "all") -> Optional[Union[str, tuple]]:
        """查末端电机名。

        :param joint: `"all"`（默认）返回全部电机名 tuple（cfg 顺序 = q 向量顺序）；传入 1 基序号（cfg 列表位置）返回该电机名。
        :return: 电机名或名 tuple；越界 `[1, n_end]` 或类型非法 warn 并返回 `None`。
        """
        return self._get_joint_names(self._joint_names_end, joint, "end")

    @staticmethod
    def _get_joint_names(names: tuple, joint, part: str):
        """关节名查询实现（1 基序号口径）。"""
        if isinstance(joint, str) and joint == "all":
            return names
        if isinstance(joint, (int, np.integer)) and not isinstance(joint, bool) and 1 <= joint <= len(names):
            return names[int(joint) - 1]
        logger.warning("joyarm.py - get_joint_names_%s：joint=%r 非法（须为 \"all\" 或 "
                       "1 基序号 ∈ [1, %d]），返回 None", part, joint, len(names))
        return None

    def rand_q_arm(self, joint_limits: Optional[JointLimits] = None) -> np.ndarray:
        """本体关节角随机采样：在 `[q_min, q_max]` 均匀采样。

        :param joint_limits: 采样限位；缺省 `None` 在方法体内回退**软限位**。
        :return: 随机关节角 `(n_arm,)`。
        """
        limits = joint_limits if joint_limits is not None else self._joint_limits_soft_arm
        return rand_within_limits(limits)

    def rand_q_end(self, joint_limits: Optional[JointLimits] = None) -> np.ndarray:
        """末端电机角随机采样：在 `[q_min, q_max]` 均匀采样。

        :param joint_limits: 采样限位；缺省 `None` 在方法体内回退**软限位**。
        :return: 随机电机角 `(n_end,)`；无 end 段型号返回空数组 `(0,)`。
        """
        limits = joint_limits if joint_limits is not None else self._joint_limits_soft_end
        if limits is None:
            return np.zeros(self._n_end)
        return rand_within_limits(limits)

    # ============================================================
    # 7.3 硬件后端委托面：连接/使能/参数/模式/状态/零点与错误/指令/动作
    # ============================================================

    # =============== 连接与断开 ===============
    def connect_arm(self, channel: Optional[str] = None, protocol: Optional[str] = None) -> Optional[bool]:
        """连接 arm 硬件总线。

        连接成功后自动设 cfg `default_mode` 并等 `write_settle` 秒生效。

        :param channel: 总线通道（可选；缺省用 backend cfg 成员值，显式传入则同步更新）。
        :param protocol: 总线协议（可选；缺省用 backend cfg 成员值，显式传入则同步更新）。
        :return: `is_connected_arm`（True 成功 / False 失败 / None 意外）。
        """
        self._backend.connect_arm(channel, protocol)
        return self._backend.is_connected_arm

    def connect_end(self, channel: Optional[str] = None, protocol: Optional[str] = None) -> Optional[bool]:
        """连接 end 硬件总线。

        连接成功后自动设 cfg `default_mode` 并等 `write_settle` 秒生效。

        :param channel: 总线通道（可选；缺省用 backend cfg 成员值，显式传入则同步更新）。
        :param protocol: 总线协议（可选；缺省用 backend cfg 成员值，显式传入则同步更新）。
        :return: `is_connected_end`（True 成功 / False 失败 / None 意外）。
        """
        self._backend.connect_end(channel, protocol)
        return self._backend.is_connected_end

    def disconnect_arm(self) -> None:
        """断开 arm 硬件总线（先尽力失能再断总线、断开后清模式缓存）。"""
        self._backend.disconnect_arm()

    def disconnect_end(self) -> None:
        """断开 end 硬件总线（先尽力失能再断总线、断开后清模式缓存）。"""
        self._backend.disconnect_end()

    @property
    def is_connected_arm(self) -> Optional[bool]:
        """arm 连接状态（True / False / None 未知；与 backend 同名直连）。"""
        return self._backend.is_connected_arm

    @property
    def is_connected_end(self) -> Optional[bool]:
        """end 连接状态（True / False / None 未知；与 backend 同名直连）。"""
        return self._backend.is_connected_end

    # =============== 使能与失能（arm 侧受 FSM 互斥） ===============
    def enable_arm(self) -> int:
        """使能 arm 全部关节电机：FSM 运行中被外部调用时空操作 + warn（互斥拦截时返回 0）。

        :return: 1 = 全部成功、0 = 任一失败或未连接。
        """
        if not self._fsm_guard("enable_arm"):
            return 0
        return self._backend.enable_arm()

    def enable_end(self) -> int:
        """使能 end 全部电机。

        :return: 1 = 全部成功、0 = 任一失败或未连接。
        """
        return self._backend.enable_end()

    def disable_arm(self) -> int:
        """失能 arm 全部关节电机：FSM 运行中被外部调用时空操作 + warn（互斥拦截时返回 0）。"""
        if not self._fsm_guard("disable_arm"):
            return 0
        return self._backend.disable_arm()

    def disable_end(self) -> int:
        """失能 end 全部电机。

        :return: 1 = 全部成功、0 = 任一失败或未连接（失败细节经日志 warn）。
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

    # =============== 参数读写（不受 FSM 互斥） ===============
    def read_param_arm(self, key: str) -> Optional[list]:
        """读 arm 全部关节的电机参数。

        :param key: 电机参数名（必填，backend子类映射到电机参数寄存器）。
        :return: 逐关节值列表；任一失败/未连接返回 `None`。
        """
        return self._backend.read_param_arm(key)

    def read_param_end(self, key: str) -> Optional[list]:
        """读 end 全部电机的电机参数。

        :param key: 电机参数名（必填，backend子类映射到电机参数寄存器）。
        :return: 逐电机值列表；任一失败/未连接返回 `None`。
        """
        return self._backend.read_param_end(key)

    def write_param_arm(self, key: str, values: Sequence[float]) -> int:
        """写 arm 全部关节的电机参数。

        :param key: 电机参数名（必填，backend子类映射到电机参数寄存器）。
        :param values: 参数值序列（必填）；长度 = 关节数且全有限。
        :return: 成功 1、失败 0。
        """
        return self._backend.write_param_arm(key, values)

    def write_param_end(self, key: str, values: Sequence[float]) -> int:
        """写 end 全部电机的电机参数。

        :param key: 电机参数名（必填，backend子类映射到电机参数寄存器）。
        :param values: 参数值序列（必填）；长度 = 电机数且全有限。
        :return: 成功 1、失败 0。
        """
        return self._backend.write_param_end(key, values)

    # =============== 控制模式（set 受 FSM 互斥） ===============
    def get_mode_arm(self) -> Optional[ControlMode]:
        """读 arm 全组控制模式（全组一致才返回该模式；不一致/失败/未连接返回 `None`）。"""
        return self._backend.get_mode_arm()

    def get_mode_end(self) -> Optional[ControlMode]:
        """读 end 全组控制模式（全组一致才返回该模式；不一致/失败/未连接返回 `None`）。"""
        return self._backend.get_mode_end()

    def set_mode_arm(self, mode: ControlMode = ControlMode.MIT) -> int:
        """设置 arm 控制模式（MIT/POSITION/VELOCITY）：FSM 运行中被外部调用时空操作 + warn。

        成功返回 1 并**乐观更新**模式缓存，硬件一致性经 `get_mode_arm` 读回纠正；
        切 POSITION / VELOCITY 自动随写 POS_VEL 四增益（backend 子类实现）。

        :param mode: 控制模式（缺省 MIT；:class:`ControlMode` 枚举，非枚举直接返回 0）。
        :return: 成功 1、失败 0。
        """
        if not self._fsm_guard("set_mode_arm"):
            return 0
        return self._backend.set_mode_arm(mode)

    def set_mode_end(self, mode: ControlMode = ControlMode.MIT) -> int:
        """设置 end 全组控制模式（MIT / POSITION / VELOCITY）。

        成功返回 1 并**乐观更新**模式缓存，硬件一致性经 `get_mode_end` 读回纠正；
        切 POSITION / VELOCITY 自动随写 POS_VEL 四增益（backend 子类实现）。

        :param mode: 控制模式（缺省 MIT；:class:`ControlMode` 枚举，非枚举直接返回 0）。
        :return: 成功 1、失败 0。
        """
        return self._backend.set_mode_end(mode)

    # =============== 状态读取（不受 FSM 互斥） ===============
    @property
    def joint_state_arm(self) -> JointState:
        """arm 整组关节状态快照（被动只读、不触发读帧）。"""
        return self._backend.joint_state_arm

    @property
    def joint_state_end(self) -> JointState:
        """end 整组关节状态快照（被动只读、不触发读帧）。"""
        return self._backend.joint_state_end

    def get_state_arm(self) -> Optional[JointState]:
        """读 arm 关节状态快照（读时刷新）。

        backend 低频刷新线程持续维护快照；`error` 码约定：0=失能、1=使能、≥2=故障。

        :return: :class:`JointState` 快照；任一关节读取失败/数据异常时跳过本轮更新返回 `None`。
        """
        return self._backend.get_state_arm()

    def get_state_end(self) -> Optional[JointState]:
        """读 end 电机状态快照（读时刷新）。

        backend 低频刷新线程持续维护快照；`error` 码约定：0=失能、1=使能、≥2=故障。

        :return: :class:`JointState` 快照；任一电机读取失败/数据异常时跳过本轮更新返回 `None`。
        """
        return self._backend.get_state_end()

    # =============== 零点与错误（set/clear 受 FSM 互斥） ===============
    def set_zero_arm(self) -> int:
        """将 arm 当前位置记为零点（1=全部成功、0=任一失败）：FSM 运行中被外部调用时空操作 + warn（拦截时返回 0）。"""
        if not self._fsm_guard("set_zero_arm"):
            return 0
        return self._backend.set_zero_arm()

    def set_zero_end(self) -> int:
        """将 end 当前位置记为零点（1=全部成功、0=任一失败）。"""
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
        """清除 arm 各关节硬件错误（逐关节发清错指令、无读回验证）：FSM 运行中被外部调用时空操作 + warn（拦截时返回 False）。"""
        if not self._fsm_guard("clear_error_arm"):
            return False
        return self._backend.clear_error_arm()

    def clear_error_end(self) -> bool:
        """清除 end 各电机硬件错误（逐电机发清错指令、无读回验证）。"""
        return self._backend.clear_error_end()

    # =============== 关节指令下发 ===============
    def send_mit_arm(self, q, dq=None, tau=None, kp=None, kd=None) -> None:
        """arm MIT 指令整组下发：FSM 运行中被外部调用时空操作 + warn（ctrl 线程自身豁免放行）。

        :param q: 位置目标 `(n_arm,)` rad，必填。
        :param dq: 速度前馈 `(n_arm,)` rad/s；缺省 `zeros(n_arm)`。
        :param tau: 力矩前馈 `(n_arm,)` N·m；缺省 `zeros(n_arm)`。
        :param kp: 位置增益 `(n_arm,)`；缺省取 `_kp_mit_default_arm`。
        :param kd: 速度增益 `(n_arm,)`；缺省取 `_kd_mit_default_arm`。
        """
        if not self._fsm_guard("send_mit_arm"):
            return
        self._backend.send_mit_arm(q, dq, tau, kp, kd)

    def send_mit_end(self, q, dq=None, tau=None, kp=None, kd=None) -> None:
        """end MIT 指令整组下发。

        :param q: 位置目标 `(n_end,)` rad，必填。
        :param dq: 速度前馈 `(n_end,)` rad/s；缺省 `zeros(n_end)`。
        :param tau: 力矩前馈 `(n_end,)` N·m；缺省 `zeros(n_end)`。
        :param kp: 位置增益 `(n_end,)`；缺省取 `_kp_mit_default_end`。
        :param kd: 速度增益 `(n_end,)`；缺省取 `_kd_mit_default_end`。
        """
        self._backend.send_mit_end(q, dq, tau, kp, kd)

    def send_position_arm(self, q, vlim=None, flim=None) -> None:
        """arm 位置指令整组下发：FSM 运行中被外部调用时空操作 + warn（ctrl 线程自身豁免放行）。

        :param q: 位置目标 `(n_arm,)` rad；越硬限位就近裁剪 + warn。
        :param vlim: 速度上限 `(n_arm,)` rad/s；缺省取 cfg `POS_VEL.vlim`，超 `[0, dq_max]` 裁剪。
        :param flim: 归一化力矩电流上限 `(n_arm,)` 0~1；缺省取 cfg `POS_VEL.flim`。
        """
        if not self._fsm_guard("send_position_arm"):
            return
        self._backend.send_position_arm(q, vlim, flim)

    def send_position_end(self, q, vlim=None, flim=None) -> None:
        """end 位置指令整组下发。

        :param q: 位置目标 `(n_end,)` rad；越硬限位就近裁剪 + warn。
        :param vlim: 速度上限 `(n_end,)` rad/s；缺省取 cfg `POS_VEL.vlim`，超 `[0, dq_max]` 裁剪。
        :param flim: 归一化力矩电流上限 `(n_end,)` 0~1；缺省取 cfg `POS_VEL.flim`。
        """
        self._backend.send_position_end(q, vlim, flim)

    def send_vel_arm(self, dq) -> None:
        """arm 速度指令整组下发：FSM 运行中被外部调用时空操作 + warn（ctrl 线程自身豁免放行）。

        :param dq: 速度目标 `(n_arm,)` rad/s；幅值越 `|dq_max|` 裁剪。
        """
        if not self._fsm_guard("send_vel_arm"):
            return
        self._backend.send_vel_arm(dq)

    def send_vel_end(self, dq) -> None:
        """end 速度指令整组下发。

        :param dq: 速度目标 `(n_end,)` rad/s；幅值越 `|dq_max|` 裁剪。
        """
        self._backend.send_vel_end(dq)

    # =============== 末端离散动作（end 侧不受互斥） ===============
    def set_action_end(self, action: str, **kw) -> None:
        """末端离散动作（纯透传 `_backend.send_action_end`）。

        未知动作或缺 `position` 参数：backend 抛 `ValueError` 经基类转 warn。

        :param action: 离散动作名（必填）。
        :param kw: 动作附加参数（如 `position`）。
        """
        self._backend.send_action_end(action, **kw)

    # ============================================================
    # 7.4 求解器六域管理：查询 / 切换
    # ============================================================
    def get_fkine_solvers(self) -> List[str]:
        """fkine 域已加载的注册名列表（深拷贝）。"""
        return copy.deepcopy(sorted(self._require_solvers("fkine")))

    def set_fkine_solver(self, name: str) -> bool:
        """切换 fkine 域激活求解器

        :param name: 注册名（必填）。
        :return: 成功 `True`；名称不在已加载成员中 warn 并保持原激活，返回 `False`。
        """
        return self._switch_solver("fkine", name)

    def get_activated_fkine_solver(self) -> str:
        """fkine 域当前激活求解器注册名。"""
        return self._activated_name_of("fkine")

    def get_ikine_solvers(self) -> List[str]:
        """ikine 域已加载的注册名列表（深拷贝）。"""
        return copy.deepcopy(sorted(self._require_solvers("ikine")))

    def set_ikine_solver(self, name: str) -> bool:
        """切换 ikine 域激活求解器

        :param name: 注册名（必填）。
        :return: 成功 `True`；名称不在已加载成员中 warn 并保持原激活，返回 `False`。
        """
        return self._switch_solver("ikine", name)

    def get_activated_ikine_solver(self) -> str:
        """ikine 域当前激活求解器注册名。"""
        return self._activated_name_of("ikine")

    def get_jac_solvers(self) -> List[str]:
        """jacobian 域已加载的注册名列表（深拷贝）。"""
        return copy.deepcopy(sorted(self._require_solvers("jac")))

    def set_jac_solver(self, name: str) -> bool:
        """切换 jacobian 域激活求解器

        :param name: 注册名（必填）。
        :return: 成功 `True`；名称不在已加载成员中 warn 并保持原激活，返回 `False`。
        """
        return self._switch_solver("jac", name)

    def get_activated_jac_solver(self) -> str:
        """jacobian 域当前激活求解器注册名。"""
        return self._activated_name_of("jac")

    def get_dyn_solvers(self) -> List[str]:
        """dynamics 域已加载的注册名列表（深拷贝）。"""
        return copy.deepcopy(sorted(self._require_solvers("dyn")))

    def set_dyn_solver(self, name: str) -> bool:
        """切换 dynamics 域激活求解器

        :param name: 注册名（必填）。
        :return: 成功 `True`；名称不在已加载成员中 warn 并保持原激活，返回 `False`。
        """
        return self._switch_solver("dyn", name)

    def get_activated_dyn_solver(self) -> str:
        """dynamics 域当前激活求解器注册名。"""
        return self._activated_name_of("dyn")

    def get_traj_solvers(self) -> List[str]:
        """traj 域已加载的注册名列表（深拷贝）。"""
        return copy.deepcopy(sorted(self._require_solvers("traj")))

    def set_traj_solver(self, name: str) -> bool:
        """切换 traj 域激活规划器（新规划器先 plan 成功才替换）

        :param name: 注册名（必填）。
        :return: 成功 `True`；名称不在已加载成员中 warn 并保持原激活，返回 `False`。
        """
        return self._switch_solver("traj", name)

    def get_activated_traj_solver(self) -> str:
        """traj 域当前激活规划器注册名。"""
        return self._activated_name_of("traj")

    def get_ctrl_solvers(self) -> List[str]:
        """control 域已加载的注册名列表（深拷贝）。"""
        return copy.deepcopy(sorted(self._require_solvers("ctrl")))

    def set_ctrl_solver(self, name: str) -> bool:
        """选择 control 域控制器并更新频率；MOTION 进入/维持时由 FSM 自愈电机模式。

        :param name: 注册名（必填）；离线也可选择，不触碰硬件。
        :return: 选择成功 `True`（不代表硬件就绪）；无效名称/MODE 返回 `False`。
        """
        return self._switch_solver("ctrl", name)

    def get_activated_ctrl_solver(self) -> str:
        """control 域当前激活控制器注册名。"""
        return self._activated_name_of("ctrl")

    # ============================================================
    # 7.5 目标帧通道：set/get + 上层便捷封装
    # ============================================================
    def set_target_trajframe(self, target) -> None:
        """写入轨迹目标（单帧或帧序列）。

        `copy.deepcopy` 后单条赋值语句整体原子替换引用。

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
        """读当前轨迹目标（深拷贝）。"""
        return copy.deepcopy(self._target_trajframe)

    def set_target_joint(self, t, q, dq=None, ddq=None, tau=None) -> None:
        """单帧关节目标便捷封装（内部组帧后走 :meth:`set_target_trajframe` 同一原子通道）。

        :param t: 目标到达时刻，Unix 秒标量（必填）。
        :param q: 关节位置目标 `(n_arm,)` rad（必填；list 或 ndarray）。
        :param dq: 关节速度目标 `(n_arm,)` rad/s（可选）。
        :param ddq: 关节加速度目标 `(n_arm,)` rad/s²（可选）。
        :param tau: 前馈力矩 `(n_arm,)` N·m（可选）。
        :raises ValueError: `t`/`q` 缺失、维度不符 `(n_arm,)`。
        """
        q = self._check_target_vector(q, self._n_arm, "q", required=True)
        dq = self._check_target_vector(dq, self._n_arm, "dq", required=False)
        ddq = self._check_target_vector(ddq, self._n_arm, "ddq", required=False)
        tau = self._check_target_vector(tau, self._n_arm, "tau", required=False)
        self.set_target_trajframe(TrajFrame(time=self._check_target_time(t), q=q, dq=dq, ddq=ddq, tau=tau))

    def set_target_cart(self, t, pose, twist=None, accel=None, wrench=None) -> None:
        """单帧笛卡尔目标便捷封装（内部组帧后走 :meth:`set_target_trajframe` 同一原子通道）。

        :param t: 目标到达时刻，Unix 秒标量（必填）。
        :param pose: 末端位姿目标（必填；:class:`Pose` 或 `(7,)` `[x, y, z, qw, qx, qy, qz]`）。
        :param twist: 末端速度（可选；:class:`Twist` 或 `(6,)` `[vx, vy, vz, wx, wy, wz]`）。
        :param accel: 末端加速度（可选；:class:`Accel` 或 `(6,)` `[ax, ay, az, αx, αy, αz]`）。
        :param wrench: 末端力/力矩（可选；:class:`Wrench` 或 `(6,)` `[fx, fy, fz, tx, ty, tz]`）。
        :raises ValueError: 必填缺失、类型/维度不符。
        """
        if isinstance(pose, Pose):
            pose_out = copy.deepcopy(pose)
        else:
            arr = self._check_target_vector(pose, 7, "pose", required=True)
            pose_out = Pose(position=np.array(arr[:3], dtype=float), orientation=np.array(arr[3:], dtype=float))
        twist_out = self._check_target_cart6(twist, Twist, "twist")
        accel_out = self._check_target_cart6(accel, Accel, "accel")
        wrench_out = self._check_target_cart6(wrench, Wrench, "wrench")
        self.set_target_trajframe(TrajFrame(time=self._check_target_time(t), pose=pose_out, twist=twist_out, accel=accel_out, wrench=wrench_out))

    @staticmethod
    def _check_target_time(t) -> float:
        """目标时刻校验（Unix 秒标量，须有限）。"""
        if isinstance(t, bool) or not isinstance(t, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"joyarm.py - set_target：t 需为 Unix 秒标量，实际 {type(t).__name__}")
        out = float(t)
        if not np.isfinite(out):
            raise ValueError(f"joyarm.py - set_target：t={t!r} 非有限（NaN/inf 拒绝）")
        return out

    @staticmethod
    def _check_target_vector(v, n: int, key: str, required: bool):
        """便捷封装的 `(n,)` 向量校验（None 且非必填 → None；否则转 float 数组、验维且须全有限）。"""
        if v is None:
            if required:
                raise ValueError(f"joyarm.py - set_target：{key} 必填（None）")
            return None
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(
                f"joyarm.py - set_target：{key} 维度 {arr.shape[0]} ≠ 期望 {n}")
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError(f"joyarm.py - set_target：{key} 含非有限值（NaN/inf 拒绝）")
        return arr

    @classmethod
    def _check_target_cart6(cls, v, cls6, key: str):
        """便捷封装的笛卡尔 6 维量校验（类型对象深拷贝；`(6,)` 序列转对应数据类，须全有限）。"""
        if v is None:
            return None
        if isinstance(v, cls6):
            return copy.deepcopy(v)
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.shape[0] != 6:
            raise ValueError(
                f"joyarm.py - set_target：{key} 需为 {cls6.__name__} 或 (6,) 序列，"
                f"实际长度 {arr.shape[0]}")
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError(f"joyarm.py - set_target：{key} 含非有限值（NaN/inf 拒绝）")
        if cls6 is Wrench:
            return Wrench(force=np.array(arr[:3]), torque=np.array(arr[3:]))
        return cls6(linear=np.array(arr[:3]), angular=np.array(arr[3:]))

    # ============================================================
    # 7.6 数学门面方法：显式传参离线可用，缺省取硬件状态快照
    # ============================================================
    def _current_q(self, caller: str) -> np.ndarray:
        """取 arm 当前 q（joint_state_arm 快照）。"""
        st = self._backend.joint_state_arm
        q = np.asarray(st.q, dtype=float)
        if not bool(np.all(np.isfinite(q))):
            raise RuntimeError(
                f"joyarm.py - {caller}：joint_state_arm 快照无可用 q（未收到状态帧）；"
                f"显式传参可离线计算")
        return q

    def _current_qdq(self, caller: str) -> tuple:
        """取 arm 当前 (q, dq)（joint_state_arm 快照）。"""
        st = self._backend.joint_state_arm
        q = np.asarray(st.q, dtype=float)
        dq = np.asarray(st.dq, dtype=float)
        if not (bool(np.all(np.isfinite(q))) and bool(np.all(np.isfinite(dq)))):
            raise RuntimeError(
                f"joyarm.py - {caller}：joint_state_arm 无可用 q/dq（未收到状态帧）；"
                f"显式传参可离线计算")
        return q, dq

    def _check_math_vector(self, v, key: str, caller: str) -> np.ndarray:
        """数学门面的 `(n_arm,)` 向量校验。"""
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.shape[0] != self._n_arm:
            raise ValueError(
                f"joyarm.py - {caller}：{key} 维度 {arr.shape[0]} ≠ 期望 {self._n_arm}（n_arm）")
        return arr

    def solve_fkine(self, q=None, frame=None) -> Pose:
        """正运动学：`q` → `frame` 帧位姿。

        :param q: `(n_arm,)` 关节角 rad；缺省取 `joint_state_arm.q` 快照。
        :param frame: 帧名；缺省取 cfg `joyarm.frame_end`。
        :return: :class:`Pose`——`frame` 帧位姿。
        :raises ValueError: `q` 维度不符 `(n_arm,)`。
        """
        if q is None:
            q = self._current_q("solve_fkine")
        q = self._check_math_vector(q, "q", "solve_fkine")
        return self._activated_of("fkine").solve(self, q, frame if frame is not None else self._frame_end)

    def solve_ikine(self, pose, frame=None, q0=None, **kw) -> IKResult:
        """逆运动学单解：目标位姿 → 关节角（数值法自 `q0` 迭代；解析法选最近解）。

        :param pose: 目标位姿（必填；:class:`Pose`）。
        :param frame: 目标帧名；缺省取 cfg `joyarm.frame_end`。
        :param q0: `(n_arm,)` 参考关节角；缺省取当前状态 q。
        :param kw: 求解器特有参数（`tol` / `iters` 等，无法识别的忽略）。
        :return: :class:`IKResult`——`q` 为 `(n_arm,)` 单解，无可行解时 `success=False`。
        :raises TypeError: `pose` 非 :class:`Pose`。
        """
        if not isinstance(pose, Pose):
            raise TypeError(
                f"joyarm.py - solve_ikine：pose 需为 Pose 类型，实际 {type(pose).__name__}")
        if q0 is None:
            q0 = self._current_q("solve_ikine")
        return self._activated_of("ikine").solve(self, pose, frame if frame is not None else self._frame_end, np.asarray(q0, dtype=float), **kw)

    def solve_ikine_all(self, pose, frame=None, **kw):
        """逆运动学全部解（仅解析法实现）。

        :param pose: 目标位姿（必填；:class:`Pose`）。
        :param frame: 目标帧名；缺省取 cfg `joyarm.frame_end`。
        :param kw: 求解器特有参数（无法识别的忽略）。
        :return: :class:`IKResult`——`q` 为 `(K, n_arm)` 全部解。
        :raises TypeError: `pose` 非 :class:`Pose`。
        :raises RuntimeError: 激活求解器未实现 `solve_all`（数值法——提示换解析法）。
        """
        if not isinstance(pose, Pose):
            raise TypeError(
                f"joyarm.py - solve_ikine_all：pose 需为 Pose 类型，实际 "
                f"{type(pose).__name__}")
        try:
            return self._activated_of("ikine").solve_all(self, pose, frame=frame, **kw)
        except NotImplementedError as e:
            raise RuntimeError(
                f"joyarm.py - solve_ikine_all：当前激活的 ikine 求解器不支持求全部解"
                f"（数值法/未实现 solve_all，请切换解析法实现）：{e}") from e

    def get_jac(self, q=None, frame=None) -> np.ndarray:
        """雅可比矩阵 `J(q)`。

        :param q: `(n_arm,)` 关节角 rad；缺省取 `joint_state_arm.q` 快照。
        :param frame: 帧名；缺省取 cfg `joyarm.frame_end`。
        :return: 基座系雅可比 `(6, n_arm)`。
        :raises ValueError: `q` 维度不符 `(n_arm,)`。
        """
        if q is None:
            q = self._current_q("get_jac")
        q = self._check_math_vector(q, "q", "get_jac")
        return self._activated_of("jac").jac(self, q, frame if frame is not None else self._frame_end)

    def get_jac_pinv_damped(self, q=None, frame=None, damping: float = 1e-3) -> np.ndarray:
        """雅可比的阻尼最小二乘广义逆（`J* = Jᵀ(JJᵀ + λ²I)⁻¹`）。

        :param q: `(n_arm,)` 关节角 rad；缺省取 `joint_state_arm.q` 快照。
        :param frame: 帧名；缺省取 cfg `joyarm.frame_end`。
        :param damping: 阻尼系数 λ；缺省 1e-3。
        :return: 广义逆 `(n_arm, 6)`。
        :raises ValueError: `q` 维度不符 `(n_arm,)`。
        """
        if q is None:
            q = self._current_q("get_jac_pinv_damped")
        q = self._check_math_vector(q, "q", "get_jac_pinv_damped")
        return self._activated_of("jac").jac_pinv_damped(self, q, frame if frame is not None else self._frame_end, damping)

    def solve_idyn(self, q=None, dq=None, ddq=None) -> np.ndarray:
        """逆动力学（RNEA）：`(q, dq, ddq)` → 各关节力矩。

        :param q: `(n_arm,)` 关节角 rad；缺省取 `joint_state_arm.q` 快照。
        :param dq: `(n_arm,)` 关节速度 rad/s；缺省取 `joint_state_arm.dq` 快照。
        :param ddq: `(n_arm,)` 关节加速度 rad/s²；缺省零向量。
        :return: 各关节力矩 `(n_arm,)` N·m。
        :raises ValueError: `q`/`dq`/`ddq` 维度不符 `(n_arm,)`。
        """
        if q is None or dq is None:
            q_cur, dq_cur = self._current_qdq("solve_idyn")
            q = q if q is not None else q_cur
            dq = dq if dq is not None else dq_cur
        q = self._check_math_vector(q, "q", "solve_idyn")
        dq = self._check_math_vector(dq, "dq", "solve_idyn")
        ddq = np.zeros_like(q) if ddq is None else self._check_math_vector(ddq, "ddq", "solve_idyn")
        return self._activated_of("dyn").idyn(self, q, dq, ddq)

    def get_mass_matrix(self, q=None) -> np.ndarray:
        """关节空间惯量矩阵 `M(q)`。

        :param q: `(n_arm,)` 关节角 rad；缺省取 `joint_state_arm.q` 快照。
        :return: 惯量阵 `(n_arm, n_arm)`。
        :raises ValueError: `q` 维度不符 `(n_arm,)`。
        """
        if q is None:
            q = self._current_q("get_mass_matrix")
        q = self._check_math_vector(q, "q", "get_mass_matrix")
        return self._activated_of("dyn").mass_matrix(self, q)

    def get_coriolis(self, q=None, dq=None) -> np.ndarray:
        """科氏/向心项 `C(q, q̇)q̇`。

        :param q: `(n_arm,)` 关节角 rad；缺省取 `joint_state_arm.q` 快照。
        :param dq: `(n_arm,)` 关节速度 rad/s；缺省取 `joint_state_arm.dq` 快照。
        :return: 各关节科氏/向心力矩 `(n_arm,)` N·m。
        """
        if q is None or dq is None:
            q_cur, dq_cur = self._current_qdq("get_coriolis")
            q = q if q is not None else q_cur
            dq = dq if dq is not None else dq_cur
        return self._activated_of("dyn").coriolis(self, np.asarray(q, dtype=float),
                                                  np.asarray(dq, dtype=float))

    def get_gravity(self, q=None) -> np.ndarray:
        """重力项 `G(q)`。

        :param q: `(n_arm,)` 关节角 rad；缺省取 `joint_state_arm.q` 快照。
        :return: 各关节重力力矩 `(n_arm,)` N·m。
        """
        if q is None:
            q = self._current_q("get_gravity")
        return self._activated_of("dyn").gravity(self, np.asarray(q, dtype=float))

    # ============================================================
    # 7.7 到位判断三函数
    # ============================================================
    def is_in_pose_arm(self, pose, tol_pos: Optional[float] = None, tol_rot: Optional[float] = None, frame=None) -> bool:
        """末端位姿到位判断：位置误差 ≤ `tol_pos` 且姿态误差角 ≤ `tol_rot`。

        :param pose: 目标位姿（必填；:class:`Pose`）。
        :param tol_pos: 位置容差 m（可选）；缺省取 cfg `joyarm.arrival_tol.arm.pos`。
        :param tol_rot: 姿态容差 rad（可选）；缺省取 cfg `joyarm.arrival_tol.arm.rot`。
        :param frame: 帧名；缺省取 cfg `joyarm.frame_end`。
        :return: 到位 `True`；未到位或无有效状态快照 `False`。
        :raises TypeError: `pose` 非 :class:`Pose`。
        """
        if not isinstance(pose, Pose):
            raise TypeError(
                f"joyarm.py - is_in_pose_arm：pose 需为 Pose 类型，实际 "
                f"{type(pose).__name__}")
        tol_pos = self._arrival_tol_pos if tol_pos is None else float(tol_pos)
        tol_rot = self._arrival_tol_rot if tol_rot is None else float(tol_rot)
        snapshot = self._fresh_joint_state("arm")
        q = np.asarray(snapshot.q if snapshot is not None else None, dtype=float)
        if q.shape != (self._n_arm,) or not bool(np.all(np.isfinite(q))):
            logger.warning("joyarm.py - is_in_pose_arm：无可用关节状态快照（未收到状态帧），判为未到位")
            return False
        cur = self._activated_of("fkine").solve(self, q, frame if frame is not None else self._frame_end)
        d_pos = float(np.linalg.norm(
            np.asarray(cur.position, float) - np.asarray(pose.position, float)))
        q_err = quat_mul(quat_conj(np.asarray(cur.orientation, float)), np.asarray(pose.orientation, float))
        d_rot = float(quat_to_axis_angle(q_err)[1])
        return bool(d_pos <= tol_pos and d_rot <= tol_rot)

    def is_in_q_arm(self, q, tol_q=None) -> bool:
        """arm 关节到位判断：`np.max(|q_cur − q|) ≤ tol_q` 全关节同时满足。

        :param q: 目标关节角 `(n_arm,)` rad（必填）。
        :param tol_q: 容差 rad（标量或 `(n_arm,)`）；缺省从 cfg `joyarm.arrival_tol.arm.q` 取。
        :return: 全关节同时满足容差 `True`；未到位或无有效状态快照 `False`。
        :raises ValueError: 目标维度与关节数不符；`tol_q` 非标量且长度与关节数不符。
        """
        return self._is_in_q(q, tol_q, self._arrival_tol_q_arm, self._fresh_joint_state("arm"), self._n_arm, "arm")

    def is_in_q_end(self, q, tol_q=None) -> bool:
        """end 电机到位判断：`np.max(|q_cur − q|) ≤ tol_q` 全电机同时满足。

        :param q: 目标电机角 `(n_end,)` rad（必填）。
        :param tol_q: 容差 rad（标量或 `(n_end,)`）；缺省从 cfg `joyarm.arrival_tol.end.q` 取。
        :return: 全电机同时满足容差 `True`；未到位或无有效状态快照 `False`。
        :raises ValueError: 目标维度与电机数不符；`tol_q` 非标量且长度与电机数不符。
        """
        return self._is_in_q(q, tol_q, self._arrival_tol_q_end, self._fresh_joint_state("end"), self._n_end, "end")

    @staticmethod
    def _is_in_q(q, tol_q, tol_member, state, n: int, part: str) -> bool:
        """关节到位判断实现（读快照）；`tol_q` 显式给出时须为标量或 n_arm/n_end 长度的列表。"""
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != n:
            raise ValueError(
                f"joyarm.py - is_in_q_{part}：目标维度 {q.shape[0]} ≠ 关节数 {n}")
        q_cur = np.asarray(state.q if state is not None else None, dtype=float)
        if q_cur.shape != (n,) or not bool(np.all(np.isfinite(q_cur))):
            logger.warning("joyarm.py - is_in_q_%s：无可用关节状态快照（未收到状态帧），判为未到位", part)
            return False
        if tol_q is None:
            tol = tol_member
        else:
            tol = np.asarray(tol_q, dtype=float)
            if tol.ndim == 0:
                tol = np.full(n, float(tol))
            else:
                tol = tol.reshape(-1)
                if tol.shape[0] != n:
                    raise ValueError(
                        f"joyarm.py - is_in_q_{part}：tol_q 长度 {tol.shape[0]} ≠ 关节数 {n}"
                        f"（标量统一全关节或 n 元逐关节）")
        return bool(np.all(np.abs(q_cur - q) <= np.broadcast_to(tol, q.shape)))

    # ============================================================
    # 7.8 MotionFSM 启停与安全事件
    # ============================================================
    def get_current_motion_state(self) -> MotionState:
        """FSM 当前状态（只读查询）。"""
        return self._current_motion_state

    def get_target_motion_state(self) -> MotionState:
        """FSM 目标状态（只读查询）。"""
        return self._target_motion_state

    def wait_for_motion_state(self, fsm_state: MotionState, timeout: float = 5.0) -> int:
        """阻塞等待 FSM 到达指定状态（只读轮询）。

        :param fsm_state: 目标状态（:class:`MotionState` 枚举）。
        :param timeout: 等待超时（秒，须为正有限数）。
        :return: 1=已到达目标状态；0=超时、FSM 未运行、FSM 中途停止或 fsm_state 非枚举。
        :raises ValueError: timeout 非法（非正有限秒数）。
        """
        if not isinstance(fsm_state, MotionState):
            logger.warning("joyarm.py - wait_for_motion_state：fsm_state 需为 MotionState"
                           " 枚举，实际 %s", type(fsm_state).__name__)
            return 0
        try:
            timeout = float(timeout)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"joyarm.py - wait_for_motion_state：timeout={timeout!r} 非法（须为正有限秒数）") from e
        if not (np.isfinite(timeout) and timeout > 0):
            raise ValueError(
                f"joyarm.py - wait_for_motion_state：timeout={timeout!r} 非法（须为正有限秒数）")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._current_motion_state is fsm_state:
                return 1
            if not self._fsm_running():
                logger.warning("joyarm.py - wait_for_motion_state：FSM 未运行（当前 %s，未达 %s），提前返回 0",
                               self._current_motion_state.name, fsm_state.name)
                return 0
            time.sleep(0.02)                      # 轮询步进（对齐 stop_motion_fsm 内部等待节奏）
        if self._current_motion_state is fsm_state:
            return 1
        logger.warning("joyarm.py - wait_for_motion_state：等待 %s 超时 %.3g s"
                       "（当前 %s）", fsm_state.name, timeout,
                       self._current_motion_state.name)
        return 0

    def start_motion_fsm(self) -> None:
        """启动运动状态机（非阻塞）。

        前置：已连接、反馈新鲜有效、无电机故障、traj/ctrl 已配置。
        不执行监视规则；运行期安全判定始终由独立监视线程负责。
        若规划/控制线程尚在运行则先停止，清理 FSM 全部缓存（`_hold_q`、plan/ctrl 线程句柄与轨迹帧缓存；`_safe_event` 置 NORMAL、目标置 MOTION），
        启动 FSM 线程（固定 `_fsm_hz`）：初始 `current=DISABLE`、按路线表自动迁移 DISABLE → HOME → MOTION。

        :raises RuntimeError: 连接/反馈/电机故障前置检查不通过；traj/ctrl 域未配置。
        """
        if self._backend.is_connected_arm is not True:
            raise RuntimeError(
                f"joyarm.py - start_motion_fsm：arm 未连接"
                f"（is_connected_arm={self._backend.is_connected_arm}），拒绝启动")
        if self._fsm_running():
            logger.info("joyarm.py - start_motion_fsm：FSM 已在运行，忽略重复启动")
            return
        # traj/ctrl 域配置检查：MOTION 态依赖 plan/ctrl 管线，未配置时启动则失败
        for domain in ("traj", "ctrl"):
            if not getattr(self, f"_{domain}_solvers"):
                raise RuntimeError(
                    f"joyarm.py - start_motion_fsm：robotics."
                    f"{_DOMAIN_REGISTRIES[domain][0]} 域未配置任何求解器"
                    f"（MOTION 态依赖 plan/ctrl 管线），拒绝启动")
        # 仅检查启动所需的即时反馈，不调用 monitor，也不改写其计时器或规则状态。
        state = self._backend.get_state_arm()
        hz = float(self._backend.cfg.get("state_refresh_hz", 20.0))
        if (state is None or not np.isfinite(state.t) or state.t <= 0
                or not 0 <= time.time() - state.t <= 1.0 / hz):
            raise RuntimeError("joyarm.py - start_motion_fsm：arm 无新鲜反馈，拒绝启动")
        q = np.asarray(state.q, dtype=float)
        error = np.asarray(state.error)
        if q.shape != (self._n_arm,) or not np.all(np.isfinite(q)):
            raise RuntimeError("joyarm.py - start_motion_fsm：arm 位置反馈无效，拒绝启动")
        if error.shape != (self._n_arm,) or not np.all(np.isin(error, (0, 1))):
            raise RuntimeError("joyarm.py - start_motion_fsm：arm 电机故障或故障码无效，拒绝启动")
        self._stop_motion_thread()                        # 规划/控制线程尚在运行则先停止
        if self._ctrl_periodic is not None or self._plan_periodic is not None:
            raise RuntimeError("joyarm.py - start_motion_fsm：旧运动线程未确认退出")
        self._fsm_shutdown.clear()
        # 幂等清理 FSM 全部缓存
        self._hold_q = None
        self._zero_moving = False
        self._home_arrived = False
        self._target_trajframe = None
        with self._event_lock:
            self._current_motion_state = MotionState.DISABLE
            self.set_event(SafetyEvent.NORMAL)          # 同步 target=MOTION，生命周期初始化
        self._fsm_periodic = _PeriodicThread(self._fsm_worker, self._fsm_hz, name="joyarm-fsm")
        self._fsm_periodic.start()
        logger.info("joyarm.py - start_motion_fsm：FSM 已启动（fsm_hz=%s，"
                    "DISABLE → HOME → MOTION 自动迁移，HOME → MOTION 受 home 到位门槛）", self._fsm_hz)

    def stop_motion_fsm(self) -> None:
        """停止运动状态机（阻塞含回零等待）。

        ① 前置 `is_connected_arm is True`；
        ② 置 `set_event(SafetyEvent.ZEROING)`，阻塞等待经 HOME 中转迁移至 ZERO 并完成到位失能；
           当前危险度高于 ZEROING（DAMPING / DISABLE）时改置 DISABLE 直接失能停机；
        ③ 禁止后续自愈/下发，停 FSM/plan/ctrl 线程，确认退出后清缓存并复位；
           join 超时仅告警，保留句柄与原状态、外部互斥继续生效，ctrl 保持禁止下发；
        ④ 最后失能 arm（end 由调用方或销毁路径处理）。

        :raises RuntimeError: 未连接。
        """
        if self._backend.is_connected_arm is not True:
            raise RuntimeError(
                f"joyarm.py - stop_motion_fsm：arm 未连接（断连急停不走本方法——"
                f"无法下发任何指令，请走销毁路径 close()）")
        if self._fsm_running():
            # 与监视器升档仲裁串行，不能用旧事件覆盖并发 DISABLE。
            with self._event_lock:
                direct = _EVENT_DANGER[self._safe_event] > _EVENT_DANGER[SafetyEvent.ZEROING]
                self.set_event(SafetyEvent.DISABLE if direct else SafetyEvent.ZEROING)
            deadline = time.monotonic() + self._move_wait_timeout
            done = False
            while time.monotonic() < deadline:
                # 回零过程中出现更高危事件或失连，立即结束等待并进入最终失能收尾。
                if (self._backend.is_connected_arm is not True
                        or _EVENT_DANGER[self._safe_event] > _EVENT_DANGER[SafetyEvent.ZEROING]):
                    self.set_event(SafetyEvent.DISABLE)
                    done = True
                    break
                if self._current_motion_state is MotionState.ZERO and not self._zero_moving:
                    done = True
                    break
                time.sleep(0.02)
            if not done:
                logger.warning("joyarm.py - stop_motion_fsm：%s s 内未完成回零失能，超时强制推进最终失能收尾",
                               self._move_wait_timeout)
        else:
            logger.info("joyarm.py - stop_motion_fsm：FSM 未在运行，直接执行失能与清理")
        # 回零结束/超时后进入最终收尾：禁止自愈 → 停线程 → 最后失能。
        self._fsm_shutdown_cleanup()
        if self._backend.disable_arm() != 1:
            logger.warning("joyarm.py - stop_motion_fsm：最终失能未确认成功")

    def _fsm_shutdown_cleanup(self) -> None:
        """FSM 统一收尾：停 FSM/plan/ctrl 线程 + 清空全部缓存与状态复位。

        FSM 线程停止超时：保留 `_fsm_periodic` 句柄（`_fsm_running` 仍为 True，外部互斥继续生效、ctrl 禁止下发）、
            FSM 独占写的状态与缓存不得由用户线程改写——跳过清理提前返回，待线程退出后再次调用完成收尾。
        """
        self._fsm_shutdown.set()
        self._motion_ready = None
        if self._fsm_periodic is not None:
            try:
                self._fsm_periodic.stop()
            except RuntimeError as e:
                logger.warning(
                    "joyarm.py - _fsm_shutdown_cleanup：FSM 线程未按时退出（不强杀，"
                    "句柄保留、跳过缓存清理与状态复位；待其退出后再次调用本方法"
                    "收尾）：%s", e)
                self._stop_motion_thread()
                return
            self._fsm_periodic = None
        self._stop_motion_thread()
        self._hold_q = None
        self._zero_moving = False
        self._home_arrived = False
        self._target_trajframe = None                     # 目标帧缓存仅在 FSM start/stop 边界清理
        with self._event_lock:
            self._current_motion_state = MotionState.DISABLE
            self.set_event(SafetyEvent.DISABLE)
        logger.info("joyarm.py - _fsm_shutdown_cleanup：FSM 已停止并清空缓存")

    def set_event(self, event: SafetyEvent) -> None:
        """设置安全事件：原子赋值 `_safe_event` 并同步目标状态为事件映射状态。

        :param event: 安全事件（必填；:class:`SafetyEvent` 枚举成员）。
        :raises TypeError: 非 `SafetyEvent` 枚举。
        """
        if not isinstance(event, SafetyEvent):
            raise TypeError(
                f"joyarm.py - set_event：event 需为 SafetyEvent 枚举，实际 "
                f"{type(event).__name__}")
        with self._event_lock:
            self._safe_event = event
            self._target_motion_state = _EVENT_TO_STATE[event]
            if event is not SafetyEvent.NORMAL:
                self._motion_ready = None         # 不等运动锁，立即阻止在途 ctrl 后续下发

    def get_event(self) -> SafetyEvent:
        """读当前安全事件。"""
        return self._safe_event

    # ============================================================
    # 7.9 safe_monitor 监视器
    # ============================================================
    def start_monitor(self) -> None:
        """启动安全监视器（独立 daemon 线程）。

        每拍只读采集（`joint_state_arm/end`、当前状态、当前事件）→ 按规则判断 → 命中则 `set_event`。
        """
        if self._monitor_periodic is not None and self._monitor_periodic.is_alive():
            logger.info("joyarm.py - start_monitor：监视器已在运行，忽略重复启动")
            return
        self._monitor_none_since = None
        self._monitor_periodic = _PeriodicThread(self._monitor_worker, self._monitor_hz, name="joyarm-monitor")
        self._monitor_periodic.start()
        logger.info("joyarm.py - start_monitor：safe_monitor 已启动（monitor_hz=%s）", self._monitor_hz)

    def stop_monitor(self) -> None:
        """停止安全监视器（join 带超时，超时 warn 不强杀、句柄保留待重试）。"""
        if self._monitor_periodic is not None:
            try:
                self._monitor_periodic.stop()
            except RuntimeError as e:
                # 超时未退出：保留 _monitor_periodic 句柄（同 stop_motion_fsm/thread 的超时分支）
                # ——句柄若被置 None，本方法的判空分支将不再生效，仍存活的监视器线程沦为无法停止的僵尸；
                # 跳过置空与"已停止"日志，待线程退出后再次调用本方法完成收尾
                logger.warning(
                    "joyarm.py - stop_monitor：监视器线程未按时退出（不强杀，"
                    "句柄保留、暂未停止；待其退出后再次调用本方法收尾）：%s", e)
                return
            self._monitor_periodic = None
            logger.info("joyarm.py - stop_monitor：safe_monitor 已停止")
    def set_monitor_rules(self, rules) -> None:
        """绑定自定义监视规则（单条或列表）。

        规则签名 `rule(state_arm, state_end, motion, event) -> SafetyEvent | None`
        仲裁：硬规则与自定义规则命中合并、统一取危险程度最高者（结果与到达顺序无关，返回 `None` 为“无意见”。）。

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
    # 7.10 运行状态可视化（HTTP 轮询）
    # ============================================================
    def start_state_viz(self, traj_points: int = 200) -> None:
        """启动运行状态可视化（单页面监控网页 + 单一只读 daemon 线程）。

        三栏页面：左侧静态参考 - 中间 meshcat 3D 视图 - 右侧动态状态面板；
        轻量 HTTP 服务托管页面，面板 JS 定时 `fetch /state.json`（约 5 Hz）。
        启动后默认弹出监控窗口（`_VizWindow` 子进程；`JOYARM_VIZ_NOWINDOW=1`可禁用）。
        `_state_viz_worker` 只调只读接口刷新 meshcat 模型姿态与面板数据。

        :param traj_points: 轨迹显示采样点数（默认 200；0 = 关闭）。轨迹显示需 FSM 已启动且处于 MOTION 态。
        :raises RuntimeError: 上次 stop 超时残留句柄未收尾（防旧服务泄漏），须先调用 stop_state_viz()。
        """
        if self._viz_periodic is not None and self._viz_periodic.is_alive():
            logger.info("joyarm.py - start_state_viz：可视化已在运行，忽略重复启动")
            return
        leftovers = [attr for attr in ("_viz_httpd", "_viz_window", "_viz_pin")
                     if getattr(self, attr) is not None]
        if leftovers:                  # 上次 stop 超时残留：拒绝覆盖句柄（防泄漏）
            raise RuntimeError(
                f"joyarm.py - start_state_viz：上次可视化 stop 超时残留未收尾"
                f"（{', '.join(leftovers)} 句柄保留中），拒绝另起；"
                f"请先调用 stop_state_viz() 完成收尾再重试"
                f"（防旧 HTTP 服务/监控窗口/meshcat-server 被覆盖引用后永久泄漏）")
        self._viz_traj_points = max(int(traj_points), 0)
        self._viz_traj_enabled = bool(self._viz_traj_points)
        # ---- 可视化专属 pin 模型（arm+end 联动；仅依赖 pinocchio）----
        # 独立于下方 meshcat 初始化块：几何构建失败不影响模型本体（另有 arm-only 兜底）。
        # 数学求解模型 _pin_model 保持不动（可视化专属缩减模型，meshcat 几何也按本模型构建）。
        self._viz_model = None
        try:
            self._viz_model = self._build_pin_model(
                self._urdf_path,
                list(self._joint_names_arm) + list(self._joint_names_end))
        except Exception as e:
            logger.warning(
                "joyarm.py - start_state_viz：arm+end 可视化模型构建失败（%s），"
                "降级回 arm-only 模型", e)
        if self._viz_model is None:
            self._viz_model = self._pin_model          # arm-only 兜底（= 现行为）
        # ---- 族级 display q 缓存 + 静态载荷一次性装配（此时 _zero_end 已为真值）----
        self._viz_q_cache = {"arm": None, "end": np.array(self._zero_end)}
        self._viz_static_bytes = json.dumps(
            self._viz_static_payload()).encode("utf-8")
        # ---- meshcat 3D 初始化（直接依赖；失败降级路径）----
        self._viz_pin = None
        self._viz_zmq_fails = 0
        iframe_src = "about:blank"
        try:
            # package_dirs 指向 URDF 所在目录：URDF 内 mesh 全为相对路径（../meshes/*.STL）
            pkg_dirs = [os.path.dirname(self._urdf_path)]
            collision = pin.buildGeomFromUrdf(self._viz_model, self._urdf_path,
                                              pin.GeometryType.COLLISION,
                                              package_dirs=pkg_dirs)
            visual = pin.buildGeomFromUrdf(self._viz_model, self._urdf_path,
                                           pin.GeometryType.VISUAL,
                                           package_dirs=pkg_dirs)
            viewer = meshcat.Visualizer()
            # zmq 看门狗：meshcat 每次 send 后跟阻塞 recv，套接字置收包超时，
            # server 死亡时 recv 超时抛异常而非永久阻塞（_state_viz_worker 捕获降级）
            viewer.window.zmq_socket.setsockopt(zmq.RCVTIMEO, _VIZ_ZMQ_TIMEOUT_MS)
            viz = MeshcatVisualizer(self._viz_model, collision, visual)
            viz.initViewer(viewer=viewer)
            viz.loadViewerModel()
            self._viz_pin = viz
            iframe_src = viewer.url()
            logger.info("joyarm.py - start_state_viz：meshcat 3D 视图已就绪（%s）",
                        iframe_src)
        except Exception as e:
            logger.warning(
                "joyarm.py - start_state_viz：meshcat 初始化失败（%s），"
                "降级为仅 HTML 状态面板", e)
        # ---- 轻量 HTTP 托管（单页面 + /static.json 静态端点 + /state.json 轮询端点 + /traj 开关）----
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
                    payload = json.dumps(arm_ref._viz_state).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif path == "/static.json":
                    # 静态载荷：start 时预装配 bytes（不可变，HTTP 线程只读）
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length",
                                     str(len(arm_ref._viz_static_bytes)))
                    self.end_headers()
                    self.wfile.write(arm_ref._viz_static_bytes)
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
        threading.Thread(target=self._viz_httpd.serve_forever, name="joyarm-viz-http", daemon=True).start()
        # ---- 单一只读 daemon 线程：刷新 meshcat 姿态与面板数据 ----
        self._viz_state = {}
        self._viz_periodic = _PeriodicThread(self._state_viz_worker, _VIZ_WORKER_HZ, name="joyarm-viz")
        self._viz_periodic.start()
        url = f"http://127.0.0.1:{port}/"
        logger.info("joyarm.py - start_state_viz：监控页面已启动 %s（面板 JS 约 5 Hz 轮询 "
                    "/state.json；轨迹点数 %s；不要求连接——未连接可启动，连接后 backend "
                    "快照自动出数）", url, self._viz_traj_points)
        # ---- 原生监控窗口（PySide6 QtWebEngine 子进程加载面板；失败/无显示器降级回 URL）----
        self._viz_window = None
        if os.environ.get("JOYARM_VIZ_NOWINDOW"):
            logger.info("joyarm.py - start_state_viz：JOYARM_VIZ_NOWINDOW 已设，"
                        "跳过监控窗口（仅 URL 方式）")
        else:
            self._viz_window = _VizWindow(url)
            self._viz_window.launch()

    def stop_state_viz(self) -> None:
        """停止运行状态可视化（停 worker、关 HTTP 服务）。

        worker 停止超时（`RuntimeError`）时保留句柄并提前返回；
        待线程退出后再次调用本方法完成收尾。
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
        if self._viz_window is not None:
            self._viz_window.stop()
            self._viz_window = None
        # meshcat-server 子进程收尾：直接终止其内部句柄（Visualizer() 拉起的独立 server）
        viz_pin, self._viz_pin = self._viz_pin, None
        viewer_window = getattr(getattr(viz_pin, "viewer", None), "window", None)
        server_proc = getattr(viewer_window, "server_proc", None)
        if server_proc is not None:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=2.0)
            except Exception as e:
                logger.warning(
                    "joyarm.py - stop_state_viz：meshcat-server 终止异常"
                    "（弃置句柄，进程退出时由 atexit 兜底）：%s", e)
        self._viz_model = None          # 下次 start 重新构建（arm+end 联动）
        self._viz_q_cache = {}
        self._viz_static_bytes = b""    # 下次 start 重新装配
        if was_running:
            logger.info("joyarm.py - stop_state_viz：可视化已停止")

    # ============================================================
    # 7.11 销毁安全退出
    # ============================================================
    def close(self) -> None:
        """安全退出：`stop_motion_fsm()`（best-effort）→ `disable_arm/end` → `disconnect_arm/end`。

        arm 已断连时走豁免路径：跳过 stop 的回零/失能等下发类步骤，停 FSM/plan/ctrl 线程并清空 FSM 缓存；
        `disconnect_arm` 保留。各步异常捕获不阻断收尾；同时停监视器与可视化线程。
        最后关闭 backend 的刷新线程和残留总线；close 后请新建对象再连接。
        本方法幂等：已完成收尾后再次调用（含 `__del__` 兜底）静默跳过。
        """
        if self._closed:                                # 幂等：已完成收尾的重复调用直接返回
            return
        if self._backend.is_connected_arm is not True:
            # 断连豁免：stop_motion_fsm、disable_arm 均跳过；线程收尾与缓存清理照常执行
            logger.warning(
                "joyarm.py - close：arm 未连接（is_connected_arm=%s），"
                "stop_motion_fsm/disable_arm 豁免跳过（仍停线程并清缓存）",
                self._backend.is_connected_arm)
            try:
                self._fsm_shutdown_cleanup()
            except Exception as e:
                logger.warning("joyarm.py - close：断连豁免收尾失败（继续收尾）：%s", e)
            steps = (
                ("stop_monitor", self.stop_monitor),
                ("stop_state_viz", self.stop_state_viz),
                ("disable_end", self._backend.disable_end),
                ("disconnect_arm", self._backend.disconnect_arm),
                ("disconnect_end", self._backend.disconnect_end),
            )
        else:
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
                if name == "stop_motion_fsm":
                    try:
                        self._fsm_shutdown_cleanup()
                    except Exception as cleanup_error:
                        logger.warning("joyarm.py - close：强制线程收尾失败：%s", cleanup_error)
        try:
            self._backend.close()                      # 显式回收后台刷新与残留总线
        except Exception as e:
            logger.warning("joyarm.py - close：backend.close 失败：%s", e)
        self._closed = True                            # 收尾完成（置位后重复 close 静默跳过）
        logger.info("joyarm.py - close：安全退出完成（%s）", self._model)

    def __enter__(self) -> "JoyArm":
        """支持 `with arm:` 写法（进入返回自身；连接需显式 `connect_arm`）。"""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """`with` 退出收尾。"""
        if exc_type is not None:
            self.set_event(SafetyEvent.DISABLE)   # 异常/中断退出不得自动回零
        self.close()

    def __del__(self) -> None:
        """析构（**不做执行时序承诺**）：best-effort 调 :meth:`close`。

        请用 `with` 语句或显式 `close()` 收尾，勿依赖本方法。
        """
        try:
            self.close()
        except Exception:
            pass
