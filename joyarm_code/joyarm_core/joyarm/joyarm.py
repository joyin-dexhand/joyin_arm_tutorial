"""``JoyArm`` —— 机械臂类

推荐用工厂创建（见 ``joyarm/__init__.py``）：``joyarm_factory("joyarm_dm")``。

快速上手：

    from joyarm_core import joyarm_factory
    arm = joyarm_factory("joyarm_dm")   # 创建（默认不连真机）
    arm.connect()                       # 连接（见"二、连接与断开"）
    arm.enable_arm()                    # 使能本体电机（见"三、使能与控制模式"）
    arm.move_j(q)                       # 关节运动到目标角（见"七、运动"）
    arm.disconnect()                    # 断开

"""
from __future__ import annotations

import copy
import logging
import os
import threading
import time
from typing import Callable, List, Optional, Union

import numpy as np
import pinocchio as pin

from ..utils.limits import clamp_to_limits, limits_from_joint_cfgs, rand_within_limits, soft_limits_from_cfg
from ..utils.transforms import quat_conj, quat_mul, quat_to_axis_angle
from ..utils.types import ArmState, ControlMode, JointLimits, Pose, TcpLimits, TrajFrame

from ..backend import Backend, get_backend
from ..robotics.fkine import REGISTRY as _FKINE_REGISTRY
from ..robotics.ikine import REGISTRY as _IKINE_REGISTRY
from ..robotics.jacobian import REGISTRY as _JACOBIAN_REGISTRY
from ..robotics.dynamics import REGISTRY as _DYNAMICS_REGISTRY
from ..robotics.trajectory import REGISTRY as _TRAJ_REGISTRY
from ..robotics.control import REGISTRY as _CONTROL_REGISTRY

__all__ = ["JoyArm", "load_config"]

logger = logging.getLogger("joyarm_core.joyarm")


# ============================================================
# 路径常量（内部）：configs/ 与 robot_model/ 目录位置
# ============================================================
# configs/ 目录（joyarm.py 位于 joyarm_core/joyarm/，上溯一级即包根）
_CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
)

# robot_model/ 目录（URDF 模型 + 网格文件，运行期加载）
_ROBOT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot_model"
)

# 本体关节命名约定：URDF 里名字是 joint1~joint9 的才算本体关节，
# 其余（末端/手指等）不参与关节数与顺序校验
_ARM_JOINT_NAMES = {f"joint{i}" for i in range(1, 10)}


def load_config(model: str, strict: bool = False) -> Optional[dict]:
    """读取 ``configs/<model>.yaml`` 型号配置文件。

    :param model: 型号名（= yaml 文件名 = yaml 里 ``basic.name`` 字段）。
    :param strict: ``True`` 时文件缺失/解析出错直接抛 ``ValueError``（并列出
        可用型号）；默认 ``False``，出错返回 ``None``（工厂用这种方式，
        由工厂自己把 ``None`` 转成失败提示）。
    """
    try:
        import yaml

        with open(os.path.join(_CONFIGS_DIR, f"{model}.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"joyarm.py - load_config：顶层应为映射，实际 {type(cfg).__name__}")
        return cfg
    except Exception as e:
        if not strict:
            return None
        try:
            avail = sorted(f[:-5] for f in os.listdir(_CONFIGS_DIR) if f.endswith(".yaml"))
        except OSError:
            avail = []
        raise ValueError(
            f"joyarm.py - load_config：『{model}』型号在 configs 中未找到；可用：{avail}"
        ) from e


# ============================================================
# 六域算法成员的构建（内部）：按 config 的 robotics 段创建求解器实例
# ============================================================
_DOMAIN_REGISTRIES = {
    "fkine": _FKINE_REGISTRY,
    "ikine": _IKINE_REGISTRY,
    "jacobian": _JACOBIAN_REGISTRY,
    "dynamics": _DYNAMICS_REGISTRY,
    "traj": _TRAJ_REGISTRY,
    "control": _CONTROL_REGISTRY,
}

def _parse_domain_specs(domain: str, spec) -> List[tuple]:
    """把 config ``robotics.<domain>`` 段解析成 ``[(注册名, 参数字典), ...]``。

    单项可以是注册名字符串，或 ``{name: 注册名, 其他键: 构造参数}`` 字典，否则抛 ``ValueError``。
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
                    f"joyarm.py - _parse_domain_specs：robotics.{domain} 规格缺 name 键：{s!r}")
            parsed.append((name, params))
        else:
            raise ValueError(
                f"joyarm.py - _parse_domain_specs：robotics.{domain} 规格类型非法"
                f"（{type(s).__name__}），需为注册名字符串或 {{name:..., **参数}} 字典")
    return parsed


def _build_domain(domain: str, registry: dict, spec) -> dict:
    """按 config 规格实例化一域算法成员，返回 ``{注册名: 实例}``。

    域没配置就返回空字典；注册名不存在、参数非法或构造出错都直接抛``ValueError``（宁可失败，不带病运行）。
    """
    members: dict = {}
    for name, params in _parse_domain_specs(domain, spec):
        if name not in registry:
            raise ValueError(
                f"joyarm.py - _build_domain：注册名『{name}』在 robotics.{domain} 中"
                f"未找到；可用：{sorted(registry) or '无（尚未实现注册）'}")
        try:
            members[name] = registry[name](**params)
        except Exception as e:
            raise ValueError(
                f"joyarm.py - _build_domain：robotics.{domain}『{name}』实例化失败：{e}"
            ) from e
    return members


# ============================================================
# 周期线程（内部）：按固定频率反复执行一个函数，直到让它停止
# ============================================================
_ERR_LOG_INTERVAL: float = 0.5   # 同一故障反复出现时，日志的最小间隔（秒）
_JOIN_TIMEOUT: float = 1.0      # stop() 等线程退出的上限（秒）


def _as_hz(hz) -> float:
    """频率参数归一：直接给数值，或给一个"返回数值的无参函数"。"""
    return float(hz() if callable(hz) else hz)


def _run_periodic(stop: threading.Event, hz, step: Callable[[], None],
                  name: str = "") -> None:
    """以 ``hz`` 频率循环调用 ``step()``，``stop`` 置位后退出。

    节拍按绝对时间对齐：某步执行太慢只跳过单拍，之后重新对齐。
    ``step()`` 抛异常只记日志、下个周期照常重试；同故障持续出现时日志每 0.5 秒最多一条（省略的次数会补记）。

    :param stop: 停止事件（置位即退出循环）。
    :param hz:   执行频率 Hz（数值或返回数值的无参函数，每周期重新取，改频率下个周期就生效）。
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
    """一个可反复启停的周期线程（``start()`` 启动、``stop()`` 停止回收）。

    ``stop()`` 最多等 1 秒；超时抛 ``RuntimeError`` 并保留线程引用（``is_alive()`` 仍为真、再次 ``start()`` 会被拒绝）
    ——避免旧线程还卡在单步里时又启新线程，出现两个循环同时跑。
    """

    def __init__(self, step: Callable[[], None], hz: float,name: str = "periodic") -> None:
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

    def set_hz(self, hz: float) -> None:
        """修改运行频率（下个周期生效；运行中换算法时用它对齐节拍）。"""
        self._hz = float(hz)

    def is_alive(self) -> bool:
        """线程是否还在跑（从未启动过算没在跑）。"""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """启动线程；已在运行（或上次 stop 没停干净）时抛 ``RuntimeError``。"""
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
        """停止线程；超时（默认 1s）没退出抛 ``RuntimeError``（引用保留）。"""
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


def _cubic_traj(q0, q1, t: float, rate: float = 100.0):
    """三次多项式插值（只有 move_j 和 _safe_move 用）：
    ``q0 → q1`` 走 ``t`` 秒，起末速度为零，按 ``rate`` 采样。

    :param q0: 起始关节角 ``(n,)``；:param q1: 目标关节角 ``(n,)``。
    :param t: 总时长（秒）；``t ≤ 0`` 不插值，直接给目标（单帧）。
    :param rate: 采样率（Hz）。
    :return: ``(ts, q, dq)``——时间戳 ``(N,)``、关节角 ``(N, n)``、关节速度 ``(N, n)``。
    """
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    t, rate = float(t), float(rate)
    if t <= 0 or rate <= 0:
        return (np.zeros(1), q1.reshape(1, -1), np.zeros((1, q0.size)))
    n = max(int(round(t * rate)) + 1, 2)          # 至少首末两帧
    ts = np.arange(n) / (n - 1) * t
    s = ts / t
    q = q0 + (q1 - q0) * (3.0 * s ** 2 - 2.0 * s ** 3)[:, None]
    dq = (q1 - q0) * (6.0 * s * (1.0 - s) / t)[:, None]
    return ts, q, dq


class JoyArm:
    """JoyArm 机械臂类（本体 arm + 末端 end）

    :param model: 型号名（= ``configs/<model>.yaml`` 文件名 = yaml ``basic.name``）。
    :param config: 型号配置字典；默认自动加载 ``configs/<model>.yaml``。config 缺失或 :meth:`check_config` 自检不通过时抛异常。
    """

    # ============================================================
    # 内部实现区：初始化 + 各类私有助手（仅调用JoyArm时直接看"公开接口区"即可）
    # ============================================================

    # ============================================================
    # 初始化：读配置并自检 → 解析 URDF → 核对关节数/顺序/限位 →
    # 装配特征位形与末端限位 → 创建通信后端 → 创建六域算法成员
    # ============================================================
    def __init__(self, model: str, config: Optional[dict] = None):
        # ---- 读取 config：不传就自动加载 configs/<model>.yaml ----
        if config is None:
            config = load_config(model, strict=True)   # 文件缺失/解析错 → 直接抛错
        cfg = copy.deepcopy(config)   # 拷贝一份：调用方之后改原字典不影响本对象
        self.check_config(model, cfg)  # 先整体自检，通过才继续装配

        basic = cfg.get("basic") or {}
        jcfg = cfg.get("joyarm") or {}
        backend_cfg = cfg.get("backend") or {}
        arm_joint_cfgs = (backend_cfg.get("arm") or {}).get("joints") or []
        end_joint_cfgs = (backend_cfg.get("end") or {}).get("joints") or []

        # ---- 成员变量先全部定义一遍（能赋的先赋值，其余先给默认值再逐段赋真值）----
        # ---- 基本信息 ----
        self.model: str = model                          # 型号名
        self._config: dict = cfg                         # 配置快照（get_config 返回它的深拷贝）
        self._urdf_path: str = ""                        # URDF 文件路径
        self.connected: bool = False                     # 是否已连真机（创建后默认不连）
        self.ee_frame_name: str = str(basic.get("ee_frame") or "ee")  # 末端坐标系名

        # ---- 运动学模型（pinocchio，由 URDF 解析而来）----
        self.pin_model: Optional[pin.Model] = None       # 构型模型（只读共享）
        self.pin_data: Optional[pin.Data] = None         # 给用户直接用的 pin.Data（内部求解另建私有的，不共用）
        self.ee_frame_id: int = -1                       # 末端坐标系在模型里的编号

        # ---- 本体（arm）----
        self.n_arm: int = 0                              # 本体关节数（config arm.joints 条数）
        self.arm_limits: Optional[JointLimits] = None    # 硬限位（真正拦指令的）
        self.arm_limits_soft: Optional[JointLimits] = None   # 软限位（只供上层状态判断，不拦指令）
        self._arm_joint_names: List[str] = []            # 关节名（config 顺序 = q 向量顺序）
        self._arm_zero: np.ndarray = np.zeros(0)         # 零位（全零）
        self._arm_home: np.ndarray = np.zeros(0)         # home 位形（config joyarm.arm_home）
        self._arm_neutral: np.ndarray = np.zeros(0)      # 数值求解默认初值（全零）

        # ---- 末端（end）----
        self.n_end: int = 0                              # 末端电机数（无末端为 0）
        self.end_limits: Optional[JointLimits] = None    # 末端硬限位（电机行程）
        self.end_limits_soft: Optional[JointLimits] = None   # 末端软限位（同 arm，备用）
        self._end_joint_names: List[str] = []            # 末端电机名（config 顺序）
        self._end_zero: np.ndarray = np.zeros(0)         # 末端零位（全零）
        self._end_home: np.ndarray = np.zeros(0)         # 末端初始位形（config joyarm.end_home）
        self._end_neutral: np.ndarray = np.zeros(0)      # 末端求解默认初值（全零）

        # ---- 末端空间限位 ----
        self.tcp_limits: TcpLimits = TcpLimits()         # 末端空间限位（默认占位，未配置）
        
        # ---- 六域算法成员（config 选型：全部创建、第一个激活）----
        self._fkine_solvers: dict = {}                   # {注册名: 实例}
        self._ikine_solvers: dict = {}
        self._jacobian_solvers: dict = {}
        self._dynamics_solvers: dict = {}
        self._traj_planners: dict = {}
        self._controllers: dict = {}
        self._active_name: dict = {}                     # 各域当前使用的注册名（有且仅一个）

        # ---- 电机通信后端 ----
        self._backend: Optional[Backend] = None          # 整机通信后端
        # ---- 轨迹数据（应用线程写目标 → 规划线程产当前帧 → 控制线程消费）----
        self._target_traj: Optional[List[TrajFrame]] = None    # 目标序列（应用写）
        self._current_frame: Optional[TrajFrame] = None        # 当前帧（规划线程写）
        
        # ---- 运行状态与后台线程 ----
        self.is_normal: bool = True                     # 运行正常标志（False 时控制线程不下发指令；由应用置位）
        self._state_thread: Optional[_PeriodicThread] = None    # 状态后台刷新线程（connect 时启动）
        self._motion_running: bool = False              # 运动管线是否在跑（防重复启停）
        self._motion_threads: dict = {}                 # 管线的三个线程 {线程名: _PeriodicThread}
        self._traj_planned_by = None                    # 最近完成规划的规划器实例（运行中换规划器的过渡保护）
        self._switching = threading.Event()             # 换算法时的暂停开关（置位期间管线线程暂停派发）

        # ---- 逐步换成真值：URDF → 关节数/顺序 → 硬/软限位 → 特征位形 → 末端空间限位 → 通信后端 → 六域算法 ----
        # ---- 解析 URDF，构建 pinocchio 模型 ----
        self._urdf_path = self._resolve_robot_urdf(str(basic["robot"]))
        self.pin_model = pin.buildModelFromUrdf(self._urdf_path)
        self.pin_data = self.pin_model.createData()

        # ---- 末端坐标系（getFrameId 对不存在的名字不报错、返回越界值，据此判断缺失）----
        self.ee_frame_id = self.pin_model.getFrameId(self.ee_frame_name)
        if self.ee_frame_id >= len(self.pin_model.frames):
            raise ValueError(
                f"joyarm.py - JoyArm.__init__：URDF 中找不到末端帧 '{self.ee_frame_name}'；"
                f"可用帧：{[f.name for f in self.pin_model.frames]}"
            )

        # ---- 关节数、关节名（config 顺序即 q 向量顺序）----
        self._arm_joint_names = [
            str(j.get("name", f"joint{i + 1}"))
            for i, j in enumerate(arm_joint_cfgs)
        ]
        self._end_joint_names = [
            str(j.get("name", f"end_joint{i + 1}"))
            for i, j in enumerate(end_joint_cfgs)
        ]
        self.n_arm = len(self._arm_joint_names)
        self.n_end = len(self._end_joint_names)
        arm_nq_urdf = self._urdf_arm_nq(self.pin_model)
        if self.n_arm != arm_nq_urdf:
            raise ValueError(
                f"joyarm.py - JoyArm.__init__：backend.arm.joints 数量 {self.n_arm} "
                f"≠ URDF arm 关节数 {arm_nq_urdf}（仅支持 arm 的关节数比对校验（config 与 urdf），"
                f"name 为 joint1~joint9 的 arm 的 joint 才会被识别；end 关节不参与校验）")
        # 顺序校验：q 向量按 config 顺序排、求解时按 URDF 顺序喂给模型——
        # 顺序不一致 = 关节角装错轴，正运动学和指令会悄悄算错，必须拦下
        pin_arm_order = [self.pin_model.names[i]
                         for i in range(1, len(self.pin_model.names))
                         if self.pin_model.names[i] in _ARM_JOINT_NAMES]
        if self._arm_joint_names != pin_arm_order:
            raise ValueError(
                f"joyarm.py - JoyArm.__init__：backend.arm.joints 顺序与 URDF 关节"
                f"顺序不一致\n  config：{self._arm_joint_names}\n  URDF  ："
                f"{pin_arm_order}\n（q 指令向量按 config 顺序索引、求解按 URDF 序"
                f"喂给 pin 模型，顺序错位会导致关节角装错轴、正运动学/指令静默"
                f"错乱；请按 URDF 关节顺序排列 config 的 joints 列表）")

        # ---- 硬限位（config backend.*.joints 四键；拦指令用，初始化后固定）----
        # 约定：四键数值与 URDF limit 保持一致（下一行会核对，不一致只告警）
        self.arm_limits = limits_from_joint_cfgs(arm_joint_cfgs)
        self.end_limits = limits_from_joint_cfgs(end_joint_cfgs)
        self._check_arm_limits_vs_urdf(self._urdf_path, arm_joint_cfgs)

        # ---- 软限位（config joyarm.*_soft_limits，不填就软=硬）----
        # 只加载备用（供上层"超软限位→报警/急停"用），不参与指令裁剪
        self.arm_limits_soft = (
            soft_limits_from_cfg(jcfg["arm_soft_limits"], self.n_arm)
            if jcfg.get("arm_soft_limits") is not None
            else copy.deepcopy(self.arm_limits))
        self.end_limits_soft = (
            soft_limits_from_cfg(jcfg["end_soft_limits"], self.n_end)
            if (self.end_limits is not None and jcfg.get("end_soft_limits") is not None)
            else copy.deepcopy(self.end_limits))
        # 软限位必须落在硬限位里面（超出=配置写错，直接抛错）
        for tag, soft, hard in (("arm", self.arm_limits_soft, self.arm_limits),
                                ("end", self.end_limits_soft, self.end_limits)):
            if soft is None or hard is None:
                continue
            beyond = ((np.asarray(soft.q_min) < np.asarray(hard.q_min) - 1e-9).any()
                      or (np.asarray(soft.q_max) > np.asarray(hard.q_max) + 1e-9).any()
                      or (np.asarray(soft.dq_max) > np.asarray(hard.dq_max) + 1e-9).any()
                      or (np.asarray(soft.tau_max) > np.asarray(hard.tau_max) + 1e-9).any())
            if beyond:
                raise ValueError(
                    f"joyarm.py - JoyArm.__init__：joyarm.{tag}_soft_limits 超出对应"
                    f"硬限位（软限位须位于硬限位内，供上层状态判断）")

        # ---- 特征位形（zero/neutral 恒全零；home 来自 config，越限自动裁剪并告警）----

        def _clip_pose(arr, limits, key):
            arr = np.asarray(arr, dtype=float).reshape(-1)
            if limits is None:
                return arr
            clipped = np.clip(arr, limits.q_min, limits.q_max)
            if not np.array_equal(arr, clipped):
                logger.warning("config joyarm.%s 越限位，已裁剪 %s → %s（以裁剪后为准）",
                               key, np.round(arr, 4).tolist(), np.round(clipped, 4).tolist())
            return clipped

        self._arm_zero = np.zeros(self.n_arm)
        self._arm_neutral = np.zeros(self.n_arm)
        self._arm_home = _clip_pose(
            jcfg.get("arm_home") if jcfg.get("arm_home") is not None else np.zeros(self.n_arm),
            self.arm_limits, "arm_home")
        self._end_zero = np.zeros(self.n_end)
        self._end_neutral = np.zeros(self.n_end)
        self._end_home = _clip_pose(
            jcfg.get("end_home") if jcfg.get("end_home") is not None else np.zeros(self.n_end),
            self.end_limits, "end_home")

        # ---- 末端空间限位（config joyarm 段配了就覆盖默认占位）----
        if jcfg.get("tcp_limits"):
            self._apply_tcp_limits(jcfg["tcp_limits"])

        # ---- 创建电机通信后端（失败即构造失败）----
        backend_name = str(backend_cfg["name"])          # check_config 已确保存在
        bcfg_rest = dict(backend_cfg)
        bcfg_rest.pop("name", None)
        self._backend = get_backend(backend_name)(bcfg_rest)

        # ---- 创建六域算法成员（任一创建失败即构造失败；第一个设为激活）----
        robotics_cfg = cfg.get("robotics") or {}
        for domain, registry in _DOMAIN_REGISTRIES.items():
            members = _build_domain(domain, registry, robotics_cfg.get(domain))
            self._members(domain).update(members)
            self._active_name[domain] = next(iter(members), None)

    # ============================================================
    # 初始化助手（内部）：URDF 查找 / 关节数统计 / 限位解析与核对 / 末端空间限位
    # ============================================================
    @staticmethod
    def _resolve_robot_urdf(robot: str) -> str:
        """找到随包 URDF 文件：``robot_model/<robot>/urdf/<robot>.urdf``。

        :raises ValueError: 型号目录/URDF 不存在（消息里列出可用型号）。
        """
        path = os.path.join(_ROBOT_MODEL_DIR, robot, "urdf", f"{robot}.urdf")
        if os.path.isfile(path):
            return path
        try:
            avail = sorted(d for d in os.listdir(_ROBOT_MODEL_DIR)
                           if os.path.isdir(os.path.join(_ROBOT_MODEL_DIR, d)))
        except OSError:
            avail = []
        raise ValueError(
            f"joyarm.py - _resolve_robot_urdf：『{robot}』型号在 robot_model 中未找到；可用：{avail}")

    @staticmethod
    def _urdf_arm_nq(pin_model) -> int:
        """数一数 pin 模型里本体关节（名字为 joint1~joint9）共有几个自由度。

        第 0 个是根节点（universe）要跳过；mimic 从动关节、末端/手指关节
        名字不在约定集合内，不计入（只核对本体，末端不校验）。

        :param pin_model: pinocchio 模型（URDF 解析产物）。
        :return: 本体关节自由度总数。
        """
        nq = 0
        for i in range(1, len(pin_model.names)):
            if pin_model.names[i] in _ARM_JOINT_NAMES:
                nq += pin_model.joints[i].nq
        return nq

    @staticmethod
    def _urdf_joint_limits(urdf_path: str) -> dict:
        """解析 URDF 里各活动关节的限位，供与 config 核对。

        只收有限位的活动关节（revolute/continuous/prismatic）。``<limit>`` 缺的属性记 ``None``，如 continuous。

        :param urdf_path: URDF 文件路径。
        :return: ``{关节名: {q_min, q_max, dq_max, tau_max, mimic}}``，
            前四值为 float 或 ``None``，``mimic`` 表示是否为从动关节。
        """
        import xml.etree.ElementTree as ET

        def _num(lim, key: str):
            v = None if lim is None else lim.get(key)
            return None if v is None else float(v)

        joints: dict = {}
        for j in ET.parse(urdf_path).getroot().iterfind("joint"):
            if j.get("type") not in ("revolute", "continuous", "prismatic"):
                continue
            lim = j.find("limit")
            joints[j.get("name")] = {
                "q_min": _num(lim, "lower"), "q_max": _num(lim, "upper"),
                "dq_max": _num(lim, "velocity"), "tau_max": _num(lim, "effort"),
                "mimic": j.find("mimic") is not None,
            }
        return joints

    @staticmethod
    def _check_arm_limits_vs_urdf(urdf_path: str, arm_joint_cfgs: list) -> None:
        """config 的关节限位与 URDF 的关节限位进行对比（只告警、不拦初始化）。

        真正生效的是 config 四键，以 config 为准。查三种情况：

        ① config 里的关节在 URDF 中不存在（没法核对）；
        ② 同名关节的限位数值和 URDF 不一致（容差 1e-4，逐键比）；
        ③ URDF 里有 config 没定义的本体活动关节（joint1~joint9 的非 mimic 关节）。

        :param urdf_path: URDF 文件路径（``basic.robot`` 解析产物）。
        :param arm_joint_cfgs: config ``backend.arm.joints`` 条目列表。
        """
        urdf = JoyArm._urdf_joint_limits(urdf_path)
        for j in arm_joint_cfgs or []:
            name = str(j.get("name", ""))
            u = urdf.get(name)
            if u is None:
                logger.warning(
                    "URDF 限位自检：config 关节 %r 在 URDF 中不存在（限位无法与 URDF 标定核对）",
                    name)
                continue
            for key in ("q_min", "q_max", "dq_max", "tau_max"):
                cfg_v, urdf_v = j.get(key), u[key]
                if cfg_v is None or urdf_v is None:
                    continue     # config 缺键由 check_config 把关；URDF 缺属性不比对
                if abs(float(cfg_v) - urdf_v) > 1e-6:
                    logger.warning(
                        "URDF 限位自检：关节 %r 的 %s 不一致——config=%s，URDF=%s"
                        "（数值以 config 为准，请核对标定）", name, key, cfg_v, urdf_v)
        defined = {str(j.get("name", "")) for j in arm_joint_cfgs or []}
        for name, u in urdf.items():
            if name not in _ARM_JOINT_NAMES:
                continue    # 非 arm 关节（end/手指等）不校验
            if name not in defined and not u["mimic"]:
                logger.warning(
                    "URDF 限位自检：URDF 活动关节 %r 未在 config backend.arm.joints 中定义",
                    name)

    def _apply_tcp_limits(self, tl: dict) -> None:
        """用 config 的 tcp_limits 段覆盖默认末端空间限位。

        ``workspace_box`` 两种写法均可： ``[[xmin,ymin,zmin],[xmax,ymax,zmax]]``
        （两行，yaml 常用）或每轴一行 ``[min,max]`` 的 ``(3,2)``；内部统一成``(3,2)``。
        """
        box = np.asarray(tl.get("workspace_box",
                                [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.8]]), dtype=float)
        if box.shape == (2, 3):          # [min 行, max 行] → 每轴 [min, max]
            box = box.T
        self.tcp_limits = TcpLimits(
            workspace_box=box,
            v_lin_max=float(tl.get("v_lin_max", 0.0)),
            v_ang_max=float(tl.get("v_ang_max", 0.0)),
            f_max=float(tl.get("f_max", 0.0)),
            t_max=float(tl.get("t_max", 0.0)),
        )

    # ============================================================
    # 六域成员管理（内部）：取某域的成员字典 / 当前使用的成员 / 统一求解入口
    # ============================================================
    def _members(self, domain: str) -> dict:
        """域 → 成员字典。"""
        return {
            "fkine": self._fkine_solvers,
            "ikine": self._ikine_solvers,
            "jacobian": self._jacobian_solvers,
            "dynamics": self._dynamics_solvers,
            "traj": self._traj_planners,
            "control": self._controllers,
        }[domain]

    def _active(self, domain: str):
        """取某域当前使用的成员；未配置成员时抛 ``RuntimeError``。"""
        d = self._members(domain)
        name = self._active_name.get(domain)
        if name is None or name not in d:
            raise RuntimeError(
                f"joyarm.py - _active：robotics.{domain} 成员未加载"
                f"（config 未配置或注册名未实现）；已加载：{sorted(d) or '无'}。"
            )
        return d[name]

    def _pick(self, domain: str, name: Optional[str]):
        """按注册名取已加载成员（``None`` = 当前使用的成员）。"""
        if name is None:
            return self._active(domain)
        d = self._members(domain)
        if name not in d:
            raise ValueError(
                f"joyarm.py - _pick：『{name}』未在 robotics.{domain} 已加载成员中；"
                f"已加载：{sorted(d)}"
            )
        return d[name]

    def _solve(self, domain: str, method: str, *args, **kw):
        """统一求解入口：取当前使用的成员，调它的 ``method``（纯转发）。

        内置 Pin 求解器每次计算都新建私有的 ``pin.Data``（微秒级），多线程
        并发调用天然安全，不需要排队。
        """
        solver = self._active(domain)
        return getattr(solver, method)(*args, **kw)

    # ============================================================
    # 调用前置校验（内部）：没连接 / 没使能 / 模式不对时，给出明确的报错
    # ============================================================
    def _require_connected(self) -> None:
        """没连接真机就抛 ``RuntimeError``。"""
        if not self.connected:
            raise RuntimeError(
                f"joyarm.py - _require_connected：[{self.model}] 未连接真机（离线）；"
                f"请先 connect()。")

    def _require_enabled_arm(self) -> None:
        """本体电机没全部使能就抛 ``RuntimeError``（在线查使能位）。

        用于运动/下发指令前（move_j、safe_*、hold_position 等）；
        急停类操作（damping_mode/lock_position）不需要先使能。
        """
        enabled = np.asarray(self.get_arm_state().joint.enabled, dtype=bool).reshape(-1)
        if not bool(enabled.all()):
            bad = np.where(~enabled)[0].tolist()
            raise RuntimeError(
                f"joyarm.py - _require_enabled_arm：本体关节 {bad} 未使能；请先 enable_arm()")

    def _require_enabled_end(self) -> None:
        """末端电机没全部使能就抛 ``RuntimeError``；无末端（n_end=0）跳过。"""
        if self.n_end == 0:
            return
        enabled = list(self.get_end_state().get("enabled", []))
        if not all(enabled):
            bad = [i for i, ok in enumerate(enabled) if not ok]
            raise RuntimeError(
                f"joyarm.py - _require_enabled_end：末端电机 {bad} 未使能；请先 enable_end()")

    def _require_mode_arm(self, mode: ControlMode, joint: Optional[int] = None) -> None:
        """本体当前控制模式 ≠ ``mode`` 就抛 ``RuntimeError``（查本地缓存）。
        """
        cur = self.read_mode_arm(joint)
        if cur != mode:
            raise RuntimeError(
                f"joyarm.py - _require_mode_arm：本体当前控制模式为 {cur}"
                f"（None=未设置/各关节不一致），与指令模式 {mode} 不符；"
                f"请先 set_mode_arm({mode})")

    # ============================================================
    # 状态后台刷新线程（内部）：connect 后自动启动，空闲时定期刷新状态缓存，
    # 保证 get_arm_state 随时能拿到新数据（发指令期间状态随指令自动更新）
    # ============================================================
    _STATE_KEEPALIVE_HZ: float = 10.0   # 巡检频率（Hz）
    _STATE_STALE_AFTER: float = 0.15    # 数据超过这么久没更新就算"旧"（秒）

    def _start_state_keepalive(self) -> None:
        """启动状态后台刷新线程（connect 自动调用；已在跑就不重复启动）。"""
        if self._state_thread is not None and self._state_thread.is_alive():
            return
        worker = _PeriodicThread(self._keepalive_step, self._STATE_KEEPALIVE_HZ,
                                name="joyarm-state")
        worker.start()
        self._state_thread = worker

    def _stop_state_keepalive(self) -> None:
        """停止状态后台刷新线程（disconnect 自动调用；1 秒内没退出抛
        ``RuntimeError``，引用保留，再次 connect 不会重复启动）。"""
        if self._state_thread is not None:
            self._state_thread.stop()
            self._state_thread = None

    def _keepalive_step(self) -> None:
        """后台刷新的单步：缓存数据旧了（说明没有指令在发）就主动查一次电机。

        正在发指令时状态随指令自动更新（新旧≈指令周期），这一步什么都不做、
        不占总线。后端不支持查数据新旧时静默跳过（其他异常由周期循环统一
        记日志、下周期重试）。
        """
        if not self.connected:
            return
        try:
            if self._backend.state_age_arm() > self._STATE_STALE_AFTER:
                self._backend.read_state_arm()
            if self.n_end and \
                    self._backend.state_age_end() > self._STATE_STALE_AFTER:
                self._backend.read_state_end()   # 无末端型号跳过（末端查询会抛错）
        except NotImplementedError:        # 后端不支持查数据新旧 → 没法保活
            pass

    # ============================================================
    # 运动管线内部（三个后台线程的单步逻辑）
    # ============================================================
    def _tick_plan(self) -> None:
        """规划线程单步：交给当前规划器；**规划成功**才记下这个规划器
        （plan_once 返回 False 表示本周期跳过，不更新记录）。"""
        planner = self._active("traj")
        if planner.plan_once(self):
            self._traj_planned_by = planner

    def _tick_sample(self) -> None:
        """采样线程单步：当前规划器和最近完成规划的是同一个才发布帧
        （防止运行中换规划器时，新规划器还没算出系数就发布）。"""
        planner = self._active("traj")
        if self._traj_planned_by is not planner:
            return
        self.set_current_frame(planner.sample_frame(time.time()))

    def _tick_ctrl(self) -> None:
        """控制线程单步：换算法的暂停开关置位期间先不发（切换流程自己补一拍）。"""
        if self._switching.is_set():
            return
        self._dispatch_ctrl()

    def _dispatch_ctrl(self) -> None:
        """控制派发一步（不带线程语义，运行中换算法时同步调用也用它）：
        运行正常？→ 读当前帧 → 控制器算指令 → 下发。"""
        if not self.is_normal:
            return
        frame = self.get_current_frame()
        if frame is None:
            return
        state = self.get_arm_state()
        mode, cmd = self._active("control").compute(self, frame, state)
        self.set_arm_command(mode, **cmd)

    def _sync_traj_tick(self) -> None:
        """同步跑一次"规划 + 采样发布"（启动/运行中换算法时用，当前帧立刻可用）。

        规划失败或跳过时保持原有帧不动，plan 线程会按节拍重试自愈——start_motion 不会因此抛错。
        """
        planner = self._active("traj")
        try:
            ok = planner.plan_once(self)
        except Exception as e:
            logger.warning("joyarm.py - _sync_traj_tick：同步规划失败，沿用现有帧"
                           "（plan 线程将按节拍重试）：%s", e)
            return
        if not ok:
            return                             # 本周期跳过（如回退帧构造失败）
        self._traj_planned_by = planner
        try:
            self.set_current_frame(planner.sample_frame(time.time()))
        except Exception as e:
            logger.warning("joyarm.py - _sync_traj_tick：同步采样发布失败，沿用"
                           "现有帧（sample 线程将按节拍重试）：%s", e)

    # ============================================================
    # 紧急阻尼的指令序列（内部；damping_mode 的实现细节）
    # ============================================================
    def _damping_frames(self, kd: float = 5.0) -> None:
        """发一套阻尼指令：切 MIT → 发阻尼帧 → 尽力使能 → 再发一帧。"""
        z = np.zeros(self.n_arm)

        def _send_arm(tag: str) -> None:
            try:
                self._backend.send_mit_arm(z, z, z, kp=z, kd=np.full(self.n_arm, kd))
            except Exception as e:
                logger.warning("damping_mode：%s阻尼指令失败：%s", tag, e)

        def _send_end(tag: str) -> None:
            try:
                if self.n_end:
                    ez = np.zeros(self.n_end)
                    self._backend.send_mit_end(ez, ez, ez, kp=ez,kd=np.full(self.n_end, kd))
            except Exception as e:
                logger.warning("damping_mode：%s阻尼指令失败：%s", tag, e)

        for setter, tag in ((self._backend.set_mode_arm, "本体切 MIT"),
                            (self._backend.set_mode_end, "末端切 MIT")):
            try:
                setter(ControlMode.MIT)
            except Exception as e:
                logger.warning("damping_mode：%s失败：%s", tag, e)
        _send_arm("本体")
        _send_end("末端")
        try:
            self._backend.enable_arm()
            self._backend.enable_end()
        except Exception as e:
            logger.warning("damping_mode：使能失败（可能部分电机故障，已失能者将自由）：%s", e)
        _send_arm("本体（使能后）")
        _send_end("末端（使能后）")

    # ============================================================
    # 直连运动内核（内部）：精确定时逐帧下发 / MIT 阻抗安全运动 / 末端与到位判断
    # ============================================================
    # 定时余量（秒）：先 sleep 到目标时刻前 1ms，再自旋等准点（兼顾 CPU 占用与亚毫秒精度）
    _SPIN_MARGIN: float = 0.001

    def _paced_send(self, ts, send) -> None:
        """按时间表逐帧下发（move_j/_safe_move 共用）。

        第 ``i`` 帧在"起点 + ts[i]"这个时刻调用 ``send(i)``；某帧迟了就跳过
        这帧的等待、继续按表走（不追赶）。
        """
        start = time.perf_counter()
        for i, ti in enumerate(ts):
            deadline = start + float(ti)
            now = time.perf_counter()
            if deadline - now > self._SPIN_MARGIN:
                time.sleep(deadline - now - self._SPIN_MARGIN)
            while time.perf_counter() < deadline:
                pass
            send(i)

    def _safe_move(self, q_arm, q_end, t=None, *, rate=None,
                   wait_tol: float = 0.05, wait_timeout: float = 10.0) -> None:
        """安全运动内核（safe_home/safe_zero 用）

        本体走 MIT 阻抗模式的三次多项式指令流（跟踪 q/dq + config MIT 增益 + 有重力补偿就用），末端走位置模式；末段等到位。

        末端不用 MIT 阻抗（config 里末端 MIT 增益标定为 0），所以走位置模式。
        """
        # 入口裁硬限位（与 move_j 同一判定基准；软限位归上层状态判断）
        q_arm = clamp_to_limits(np.asarray(q_arm, dtype=float).reshape(-1),
                                self.arm_limits)
        q0 = np.asarray(self.get_arm_state().joint.q, dtype=float).reshape(-1)
        if rate is None:
            rate = float(((self._config.get("backend") or {}).get("arm") or {})
                         .get("control_rate", 100.0))
        rate = min(float(rate), 1000.0)
        if t is None:
            t = float(np.max(np.abs(q_arm - q0)))
        ts, qs, dqs = _cubic_traj(q0, q_arm, float(t), rate)
        # 末端：位置模式发单个目标（电机角度）
        if self.n_end:
            q_end = clamp_to_limits(np.asarray(q_end, dtype=float).reshape(-1),
                                    self.end_limits)
            self._backend.set_mode_end(ControlMode.POSITION)
            self._backend.send_position_end(q_end)
        # 本体：MIT 阻抗指令流（kp/kd 传 None → 后端用 config 的 MIT 增益；
        # 重力前馈取起点的常值，没配 dynamics 就置零）
        try:
            tau_ff = np.asarray(self.gravity(q0), dtype=float).reshape(-1)
        except RuntimeError:
            tau_ff = np.zeros(self.n_arm)
        self.set_mode_arm(ControlMode.MIT)
        self._paced_send(ts, lambda i: self._backend.send_mit_arm(
            qs[i], dqs[i], tau_ff))
        if not self._wait_in_position(q_arm, wait_tol, wait_timeout):
            raise RuntimeError(
                f"joyarm.py - _safe_move：到位超时（{wait_timeout}s 内未达容差 "
                f"{wait_tol} rad；请增大 t 或检查 config 增益/限速）")

    def _is_end_in_position(self, q_end, tol_q: float = 0.05) -> bool:
        """末端到位判断（每个电机都进入容差才算；无末端恒 ``True``）。"""
        if self.n_end == 0:
            return True
        cur = np.asarray(self.get_end_state().get("q", []), dtype=float).reshape(-1)
        if cur.size == 0:
            return False
        q_end = np.asarray(q_end, dtype=float).reshape(-1)
        return bool(np.max(np.abs(cur - q_end)) <= tol_q)

    def _wait_in_position(self, q, tol_q: float, timeout: float) -> bool:
        """每隔 50ms 查一次是否到位；超时前再查最后一次并返回结果。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_in_position(q=q, tol_q=tol_q):
                return True
            time.sleep(0.05)
        return self.is_in_position(q=q, tol_q=tol_q)

    # ============================================================
    # 公开接口区（用户调用）——按"拿到一台机械臂后怎么用"的顺序分类
    # ============================================================

    # ============================================================
    # 一、配置与自检：看配置内容、检查配置文件、检查硬件好坏
    # ============================================================
    def get_config(self) -> dict:
        """返回当前 config 配置的深拷贝

        用于外部获取型号数据（如 MDH 参数）时调用，不建议重新读 yaml。
        """
        return copy.deepcopy(self._config)

    @staticmethod
    def check_config(model: str, config: dict) -> None:
        """检查配置内容是否合格（不接硬件；``__init__`` 第一步自动调用）。

        检查项：basic 段齐全、命名一致、URDF 文件存在、backend 必配且本体
        关节四键限位齐全、home 位形长度与关节数一致、robotics 段格式合法。

        :raises ValueError: 有问题时抛出，一条消息列出全部问题。
        """
        problems: List[str] = []
        cfg = config or {}
        if "basic" not in cfg:
            problems.append("缺少配置段：basic")
        basic = cfg.get("basic") or {}
        if basic.get("name") not in (None, model):
            problems.append(f"basic.name={basic.get('name')!r} 与 model={model!r} 不一致")
        if not basic.get("robot"):
            problems.append("basic.robot 缺失（robot_model URDF 索引，无法解析 URDF）")
        else:
            try:
                JoyArm._resolve_robot_urdf(str(basic["robot"]))
            except ValueError as e:
                problems.append(str(e))
        if not basic.get("ee_frame"):
            problems.append("basic.ee_frame 缺失（末端帧名）")
        if "backend" not in cfg:
            problems.append("缺少配置段：backend（必配：不配置则无法指定通信与执行）")
        bcfg = cfg.get("backend") or {}
        if not bcfg.get("name"):
            problems.append("backend.name 缺失（整机后端选型键）")
        for i, j in enumerate((bcfg.get("arm") or {}).get("joints") or []):
            if not isinstance(j, dict) or not j.get("name"):
                problems.append(f"backend.arm.joints[{i}] 缺少 name 键")
            else:
                missing = [k for k in ("q_min", "q_max", "dq_max", "tau_max")
                           if j.get(k) is None]
                if missing:
                    problems.append(
                        f"backend.arm.joints[{i}]（{j.get('name')}）缺少限位键 {missing}"
                        f"（四键数值须与 URDF limit 标定保持一致）")
        for part in ("arm", "end"):
            joints = (bcfg.get(part) or {}).get("joints") or []
            if part == "arm" and not joints:
                problems.append("backend.arm.joints 为空（本体电机列表必配）")
            for i, j in enumerate(joints):
                if not isinstance(j, dict) or not j.get("name"):
                    if part == "end":
                        problems.append(f"backend.end.joints[{i}] 缺少 name 键")
        n_arm_cfg = len((bcfg.get("arm") or {}).get("joints") or [])
        n_end_cfg = len((bcfg.get("end") or {}).get("joints") or [])
        jcfg = cfg.get("joyarm") or {}
        for key, n_ref in (("arm_home", n_arm_cfg), ("end_home", n_end_cfg)):
            v = jcfg.get(key)
            if v is None:
                continue
            size = np.asarray(v, dtype=float).size
            if size != n_ref:
                problems.append(f"joyarm.{key} 长度 {size} ≠ 关节数 {n_ref}")
        for key, n_ref in (("arm_soft_limits", n_arm_cfg), ("end_soft_limits", n_end_cfg)):
            soft = jcfg.get(key)
            if soft is None:
                continue
            if not isinstance(soft, dict):
                problems.append(f"joyarm.{key} 需为四键字典（q_min/q_max/dq_max/tau_max）")
                continue
            for k, v in soft.items():
                if k in ("q_min", "q_max", "dq_max", "tau_max"):
                    if np.asarray(v, dtype=float).ndim > 0 and np.asarray(v, dtype=float).reshape(-1).shape[0] != n_ref:
                        problems.append(
                            f"joyarm.{key}.{k} 长度 "
                            f"{np.asarray(v, dtype=float).reshape(-1).shape[0]} ≠ 关节数 {n_ref}")
        robotics_cfg = cfg.get("robotics") or {}
        for domain in _DOMAIN_REGISTRIES:
            try:
                _parse_domain_specs(domain, robotics_cfg.get(domain))
            except ValueError as e:
                problems.append(str(e).split("：", 1)[1] if "：" in str(e) else str(e))
        if problems:
            raise ValueError(
                "joyarm.py - check_config：配置自检未通过：\n"
                + "\n".join(f"  - {p}" for p in problems))

    def check_hardware(self) -> None:
        """检查硬件

        连接硬件，电机失能状态下逐个检查：arm 关节通讯/故障/编码器、end 通讯/故障；
        查完沿用之前的连接状态。一切正常就静默返回；存在问题抛``RuntimeError`` 。

        :raises RuntimeError: 连接失败或存在硬件故障（消息列出全部问题）。
        """
        own_conn = not self.connected
        if own_conn:
            self.connect()                       # 临时连接（不使能电机）
        try:
            problems: List[str] = []
            st = self.get_arm_state()
            js = st.joint
            for i in range(self.n_arm):
                if not bool(np.asarray(js.comm_ok)[i]):
                    problems.append(f"本体关节[{i}] 通讯无应答")
                if bool(np.asarray(js.error)[i]):
                    problems.append(f"本体关节[{i}] 电机故障（故障标志置位）")
                if not bool(np.asarray(js.angle_ok)[i]):
                    problems.append(f"本体关节[{i}] 编码器角度无效")
            try:
                es = self.get_end_state()
            except Exception as e:
                es = {}
                problems.append(f"末端状态读取失败：{e}")
            for i, ok in enumerate(es.get("comm_ok", [])):
                if not ok:
                    problems.append(f"末端电机[{i}] 通讯无应答")
            for i, err in enumerate(es.get("error", [])):
                if err:
                    problems.append(f"末端电机[{i}] 电机故障（故障标志置位）")
        finally:
            if own_conn:
                self.disconnect()
        if problems:
            raise RuntimeError(
                "joyarm.py - check_hardware：硬件自检未通过：\n"
                + "\n".join(f"  - {p}" for p in problems))

    # ============================================================
    # 二、连接与断开：connect 之后才能做下面所有真机操作
    # ============================================================
    def connect(self) -> None:
        """连接真机（本体 + 末端一起连；只连接不使能）。

        连接后自动启动后台小线程：空闲时定期刷新状态缓存，保证``get_arm_state()`` 随时拿到新数据。
        """
        self._backend.connect()
        self.connected = bool(self._backend.connected)
        self._start_state_keepalive()

    def disconnect(self) -> None:
        """断开真机

        各收尾步骤都尽力执行：某步失败只告警、断开流程继续（防止总线卡死时串口泄漏）。
        """
        try:
            self.stop_motion(damping=False)
        except Exception as e:
            logger.warning("joyarm.py - disconnect：停止运动管线失败（继续断开）：%s", e)
        try:
            self._stop_state_keepalive()
        except Exception as e:
            logger.warning("joyarm.py - disconnect：停止状态保活线程失败（继续断开；"
                           "线程可能卡在总线 IO 上，其引用已保留待排查）：%s", e)
        self._backend.disconnect()
        self.connected = False

    def __enter__(self) -> "JoyArm":
        """支持 ``with arm:`` 写法：进入时没连接就自动 ``connect()``。"""
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """``with`` 退出收尾
        
        停管线 → 失能本体/末端 → 断开；
        """
        for name, fn in (("stop_motion", lambda: self.stop_motion(damping=False)),
                         ("disable_arm", self.disable_arm),
                         ("disable_end", self.disable_end),
                         ("disconnect", self.disconnect)):
            try:
                fn()
            except Exception as e:
                logger.warning("__exit__：%s 失败：%s", name, e)

    # ============================================================
    # 三、使能与控制模式：电机要先使能才能动
    # ============================================================
    def enable_arm(self, joint: Optional[int] = None) -> None:
        """使能本体电机（``joint=None`` 全部）"""
        self._require_connected()
        self._backend.enable_arm(joint)

    def disable_arm(self, joint: Optional[int] = None) -> None:
        """失能本体电机（``joint=None`` 全部）"""
        self._require_connected()
        self._backend.disable_arm(joint)

    def enable_end(self, joint: Optional[int] = None) -> None:
        """使能末端电机（``joint=None`` 全部）"""
        self._require_connected()
        self._backend.enable_end(joint)

    def disable_end(self, joint: Optional[int] = None) -> None:
        """失能末端电机（``joint=None`` 全部）"""
        self._require_connected()
        self._backend.disable_end(joint)

    def set_mode_arm(self, mode: ControlMode = ControlMode.POSITION, joint: Optional[int] = None) -> None:
        """切换本体控制模式（位置/速度/MIT）

        默认位置模式，``joint=None`` 全部关节。
        """
        self._require_connected()
        self._backend.set_mode_arm(mode, joint)

    def set_mode_end(self, mode: ControlMode = ControlMode.POSITION,
            joint: Optional[int] = None) -> None:
        """切换末端控制模式（位置/速度/MIT）

        默认位置模式，``joint=None`` 全部关节。
        """
        self._require_connected()
        self._backend.set_mode_end(mode, joint)

    def read_mode_arm(self, joint: Optional[int] = None) -> Optional[ControlMode]:
        """查本体当前控制模式（读本地缓存、不发总线）

        ``joint=None`` 时，整臂各关节模式一致才返回该模式，否则 ``None``。
        """
        return self._backend.read_mode_arm(joint)

    def read_mode_end(self, joint: Optional[int] = None) -> Optional[ControlMode]:
        """查末端当前控制模式（读本地缓存、不发总线）

        ``joint=None`` 时各电机模式一致才返回该模式，否则 ``None``。
        """
        return self._backend.read_mode_end(joint)

    # ============================================================
    # 四、读状态：关节角/速度/力矩/温度、末端位姿
    # ============================================================
    def get_arm_state(self) -> ArmState: # ！改为:param joint: 末端电机索引，``None`` 表示全部。
        """读取本体当前状态（关节角/速度/力矩、使能、故障、通讯、温度等）。

        数据新（0.15 秒内更新过）则直接用缓存、不发总线请求，太旧就先查询一次再返回。
        空闲时由连接后自动启动的后台线程维持数据新鲜。

        :raises RuntimeError: 还没 ``connect()``。
        """
        self._require_connected()
        try:
            if self._backend.state_age_arm() <= self._STATE_STALE_AFTER:
                state = self._backend.read_state_cache_arm()
            else:
                state = self._backend.read_state_arm()   # 太旧 → 先查一次
        except NotImplementedError:
            state = self._backend.read_state_arm()
        if self._active_name.get("fkine") is not None:
            # 填 tcp.pose 前先拷贝外壳：缓存读到的可能是共享对象，直接改会
            # 污染缓存（joint 不动、保持共享）
            state = copy.copy(state)
            state.tcp = copy.copy(state.tcp)
            state.tcp.pose = self.fkine(state.joint.q, self.ee_frame_name)   # 默认 rep="pose" → Pose
        return state

    def get_end_state(self, joint: Optional[int] = None) -> dict:
        """读取末端状态（q/dq/tau、使能、故障、通讯、温度，字典形式；
        
        数据新（0.15 秒内更新过）则直接用缓存、不发总线请求，太旧就先查询一次再返回。
        空闲时由连接后自动启动的后台线程维持数据新鲜。

        :param joint: 末端电机索引，``None`` 表示全部。
        :raises RuntimeError: 还没 ``connect()``。
        """
        self._require_connected()
        try:
            if self._backend.state_age_end(joint) <= self._STATE_STALE_AFTER:
                return self._backend.read_state_cache_end(joint)
            return self._backend.read_state_end(joint)     # 太旧 → 先查一次
        except NotImplementedError:
            return self._backend.read_state_end(joint)

    # ============================================================
    # 五、手动下发指令（底层单步）
    # ============================================================
    def set_arm_command(self,
        mode: ControlMode = ControlMode.POSITION,
        q: Optional[np.ndarray] = None,
        dq: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        """手动下发一条运动指令
        
        三种模式：POSITION 要 ``q``；VELOCITY 要 ``dq``；MIT 要 ``q/dq/tau``；
        其中 ``kp/kd`` 默认值为 config 的 MIT 增益。

        ``joint`` 指定单关节控制；``joint=None`` 全部关节。

        :raises RuntimeError: 未连接 / 当前模式与 ``mode`` 不符（先 :meth:`set_mode_arm`）。
        :raises ValueError: 该模式需要的参数没给全。
        """
        self._require_connected()
        self._require_mode_arm(mode, joint)           # 收指令前必须已切到对应模式
        if mode == ControlMode.POSITION:
            if q is None:
                raise ValueError("joyarm.py - set_arm_command：POSITION 模式需要 q")
            self._backend.send_position_arm(q, joint)
        elif mode == ControlMode.VELOCITY:
            if dq is None:
                raise ValueError("joyarm.py - set_arm_command：VELOCITY 模式需要 dq")
            self._backend.send_velocity_arm(dq, joint)
        elif mode == ControlMode.MIT:
            missing = [
                name for name, val in (("q", q), ("dq", dq), ("tau", tau)) if val is None
            ]
            if missing:
                raise ValueError(f"joyarm.py - set_arm_command：MIT 模式缺少参数：{missing}")
            self._backend.send_mit_arm(q, dq, tau, kp=kp, kd=kd, joint=joint)
        else:
            raise ValueError(f"joyarm.py - set_arm_command：未知控制模式：{mode}")

    # ============================================================
    # 六、末端执行器（夹爪）：开/闭/归零/位置/力矩
    # ============================================================
    def set_end_open(self, joint: Optional[int] = None) -> None:
        """张开末端到最大（默认行程/力度；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_action_end("open", joint)

    def set_end_close(self, joint: Optional[int] = None) -> None:
        """闭合末端（夹到默认力度即停；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_action_end("close", joint)

    def set_end_zero(self, joint: Optional[int] = None) -> None:
        """末端归零（目标 = 电机 0 弧度，越行程会自动裁剪）。"""
        self._require_connected()
        self._backend.send_action_end("zero", joint)

    def set_end_position(self, position, joint: Optional[int] = None) -> None:
        """末端位置控制（连续量，如夹爪电机弧度；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_position_end(position, joint)

    def set_end_tau(self, tau, joint: Optional[int] = None) -> None:
        """末端力矩控制（直接给电机力矩 N·m；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_tau_end(tau, joint)

    # ============================================================
    # 七、运动（最常用）：move_j 一把梭；safe_* 柔和回位；后四个为占位
    # ============================================================
    def move_j(self, q, t=None, *, rate=None,
               wait_tol: float = 0.05, wait_timeout: float = 10.0) -> None:
        """关节运动到目标角（阻塞到到位/超时）

        内部用三次多项式把轨迹铺平滑，按固定频率逐帧发位置指令，末段轮询等到位。
        独立的直连通道，**不经**「规划器→控制器」管线。
        峰值速度可能被 config ``POS_VEL.vlim``限住，实际时长可能大于 ``t``。

        :param q: 目标关节角 ``(n_arm,)``，弧度。
        :param t: 运动时长（秒）；不填按路程自动估计（峰值约 1.5 rad/s）；``t ≤ 0`` 不插值直接发目标。
        :param rate: 指令发送频率 Hz，默认取 config ``control_rate``（≤1000）。
        :param wait_tol: 到位判定容差（rad）。
        :param wait_timeout: 到位等待上限（秒），超时抛 ``RuntimeError``。
        :raises ValueError: 目标维度与关节数不符。
        :raises RuntimeError: 未连接 / 未使能 / 到位超时。
        """
        self._require_connected()
        self._require_enabled_arm()
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.n_arm:
            raise ValueError(
                f"joyarm.py - move_j：目标维度 {q.shape[0]} 与关节数 {self.n_arm} 不符")
        # 入口裁硬限位：规划、下发、到位判定统一用裁剪后的目标，避免
        # "后端裁剪到的位置 ≠ 判到位用的目标" 造成假超时
        q_c = clamp_to_limits(q, self.arm_limits)
        if not np.array_equal(q, q_c):
            logger.warning("move_j：目标关节角越硬限位，已裁剪 %s → %s"
                           "（规划与到位判定以裁剪后为准）",
                           np.round(q, 4).tolist(), np.round(q_c, 4).tolist())
        q = q_c
        q0 = np.asarray(self.get_arm_state().joint.q, dtype=float).reshape(-1)
        if rate is None:
            rate = float(((self._config.get("backend") or {}).get("arm") or {})
                         .get("control_rate", 100.0))
        rate = min(float(rate), 1000.0)
        if t is None:
            t = float(np.max(np.abs(q - q0)))
        ts, qs, _ = _cubic_traj(q0, q, float(t), rate)
        self.set_mode_arm(ControlMode.POSITION)          # 不在位置模式先切过来
        self._paced_send(ts, lambda i: self._backend.send_position_arm(qs[i]))
        if not self._wait_in_position(q, wait_tol, wait_timeout):
            raise RuntimeError(
                f"joyarm.py - move_j：到位超时（{wait_timeout}s 内未达容差 "
                f"{wait_tol} rad；可能受 vlim 限速，请增大 t 或检查 config）")

    def safe_home(self, t=None, *, wait_tol: float = 0.05,
                  wait_timeout: float = 10.0) -> None:
        """安全回到 home 姿态（本体 + 末端）

        本体走 MIT 阻抗模式，末端走位置模式。
        """
        self._require_connected()
        self._require_enabled_arm()
        self._safe_move(self.arm_home, self.end_home, t,
                        wait_tol=wait_tol, wait_timeout=wait_timeout)

    def home_to_zero(self, t=None, *, wait_tol: float = 0.05, wait_timeout: float = 10.0) -> None:
        """从 home 走到零位（本体 + 末端）。
        会先确认当前确实在 home（否则报错），再走到零位。

        :raises RuntimeError: 当前不在 home 位形（请先 :meth:`safe_home` /
            :meth:`safe_zero`）。
        """
        self._require_connected()
        arm_ok = self.is_in_position(
            q=clamp_to_limits(self.arm_home, self.arm_limits), tol_q=wait_tol)
        end_ok = self._is_end_in_position(
            clamp_to_limits(self.end_home, self.end_limits), tol_q=wait_tol) \
            if self.n_end else True
        if not (arm_ok and end_ok):
            raise RuntimeError(
                f"joyarm.py - home_to_zero：当前不在 home 位形（容差 {wait_tol} rad）；"
                f"请先 safe_home() 或 safe_zero()")
        self._safe_move(self.arm_zero, self.end_zero, t,
                        wait_tol=wait_tol, wait_timeout=wait_timeout)

    def safe_zero(self, t=None, *, wait_tol: float = 0.05, wait_timeout: float = 10.0) -> None:
        """安全回零（本体 + 末端）
        
        先``safe_home()`` 回 home，再 ``home_to_zero()`` 走到零位
        """
        self._require_connected()
        self.safe_home(t, wait_tol=wait_tol, wait_timeout=wait_timeout)
        self.home_to_zero(t, wait_tol=wait_tol, wait_timeout=wait_timeout)

    def move_l(self, pose, t=None, **kw) -> None:
        """末端走直线到目标位姿（**占位，未实现**）
        
        需要逆运动学 + 笛卡尔轨迹规划实现后落地。当前请用 ``move_j`` 或运动管线。
        """
        raise NotImplementedError(
            "joyarm.py - move_l：笛卡尔直线运动需 ikine + 笛卡尔规划，尚未实现；"
            "常规运动请走 轨迹桥 → 规划器 → 控制器 管线")

    def teach_start(self) -> None:
        """开始拖动示教（**占位，未实现**）：需要重力补偿实现后落地。

        目标效果：MIT 模式 + 重力前馈 + 零刚度（kp=0）——臂只剩重力补偿，可徒手拖动，同时录制关节轨迹。
        """
        raise NotImplementedError(
            "joyarm.py - teach_start：拖动示教需 dynamics 重力补偿实现"
            "（MIT 模式 + gravity 前馈 + 零刚度 kp=0）")

    def teach_play(self) -> None:
        """示教轨迹回放（**占位，未实现**）：需要 :meth:`teach_start` 的录制。

        目标效果：退出拖动示教，把录到的关节轨迹做时间参数化后回放。
        """
        raise NotImplementedError(
            "joyarm.py - teach_play：示教回放需先实现 teach_start 的录制能力"
            "（录制关节轨迹 → 时间参数化 → 回放）")

    def teleop_keyboard(self) -> None:
        """笛卡尔键盘遥操作（**占位，未实现**）

        """
        raise NotImplementedError(
            "joyarm.py - teleop_keyboard：笛卡尔键盘遥操作需键盘映射 + "
            "微分逆解速度指令流实现")

    # ============================================================
    # 八、安全与急停：任何状态都能用；出事先想到 damping_mode
    # ============================================================
    def damping_mode(self, kd: float = 5.0) -> None:
        """紧急阻尼：**任何状态**下把全部电机（含末端）切成"只抵抗运动"的模式。

        依次做：切 MIT 模式 → 发阻尼指令 → 尽力使能 → 再发一帧。每步都
        尽力执行：某步失败只告警，不影响其他电机收到指令。

        :param kd: 阻尼强度（N·m·s/rad），默认 5.0（DM 电机能编码的上限，
            越大越"黏"；给更大的值会被后端钳到 5 并告警）。
        :raises RuntimeError: 未连接真机。
        """
        self._require_connected()
        self._damping_frames(kd)

    def hold_position(self, kp=None, kd=None, tau=None) -> None:
        """原地保持：把当前位置设为目标，用 MIT 模式"抓"住不放。

        ``tau`` 不填时自动加重力补偿（需配置 dynamics 域；否则置零并告警——纯靠刚度硬撑）。
        ``kp/kd`` 不填写则用 config 的 MIT 增益。
        """
        self._require_connected()
        self._require_enabled_arm()
        q = np.asarray(self.get_arm_state().joint.q, dtype=float).reshape(-1)
        if tau is None:
            try:
                tau = np.asarray(self.gravity(q), dtype=float).reshape(-1)
            except RuntimeError:
                tau = np.zeros(self.n_arm)
                logger.warning("hold_position：dynamics 域未配置，tau 前馈置零（退化保持）")
        self.set_mode_arm(ControlMode.MIT)
        self.set_arm_command(ControlMode.MIT, q=q, dq=np.zeros(self.n_arm), tau=tau,
                             kp=kp, kd=kd)

    def lock_position(self) -> None:
        """急停锁定：不管当前什么模式，立刻切位置模式锁死当前关节角。"""
        self._require_connected()
        q = np.asarray(self.get_arm_state().joint.q, dtype=float).reshape(-1)
        self.set_mode_arm(ControlMode.POSITION)
        self.set_arm_command(ControlMode.POSITION, q=q)

    def is_in_position(self, q=None, pose=None, frame=None,
            tol_q: float = 0.05, tol_pos: float = 1e-3, tol_rot: float = 1e-2) -> bool:
        """到位判断：``q``（关节目标）和 ``pose``（笛卡尔目标）二选一传入。

        关节：每个关节都进入 ``tol_q``（rad）容差内算到位；
        位姿：末端位置误差 ≤ ``tol_pos``（m）且姿态误差角 ≤ ``tol_rot``（rad）。

        :param frame: pose 分支的参考帧名，缺省末端坐标系。
        :raises ValueError: ``q``/``pose`` 双空或双给、维度不符。
        :raises TypeError: ``pose`` 不是 :class:`Pose` 类型。
        :raises RuntimeError: 未连接；pose 分支未配置 fkine（算不了末端位姿）。
        """
        if (q is None) == (pose is None):
            raise ValueError("joyarm.py - is_in_position：q 与 pose 必须恰给其一")
        if pose is not None and not isinstance(pose, Pose):
            raise TypeError(
                f"joyarm.py - is_in_position：pose 需为 Pose 类型，实际 "
                f"{type(pose).__name__}（传入非 Pose 对象会与恒等位姿比较、"
                f"几乎必然判不到位，故直接拒绝）")
        st = self.get_arm_state()                        # 未连接在此抛 RuntimeError
        if q is not None:
            q = np.asarray(q, dtype=float).reshape(-1)
            if q.shape[0] != self.n_arm:
                raise ValueError(
                    f"joyarm.py - is_in_position：目标维度 {q.shape[0]} "
                    f"与关节数 {self.n_arm} 不符")
            cur = np.asarray(st.joint.q, dtype=float).reshape(-1)
            return bool(np.max(np.abs(cur - q)) <= tol_q)
        frame = frame or self.ee_frame_name
        # 现算当前末端位姿（不用 st.tcp.pose——没配 fkine 时它是恒等默认值，
        # 拿来比较会出错）
        cur_pose = self.fkine(st.joint.q, frame)
        tgt = pose
        d_pos = float(np.linalg.norm(
            np.asarray(cur_pose.position, float) - np.asarray(tgt.position, float)))
        q_err = quat_mul(quat_conj(np.asarray(cur_pose.orientation, float)),
                         np.asarray(tgt.orientation, float))
        d_rot = float(quat_to_axis_angle(q_err)[1])
        return bool(d_pos <= tol_pos and d_rot <= tol_rot)

    # ============================================================
    # 九、标零、故障清除与电机参数读写
    # ============================================================
    def set_zero_arm(self, joint: Optional[int] = None) -> None:
        """本体零位标定（把当前位置记为零点；先失能、确认无故障再标）。"""
        self._require_connected()
        self._backend.set_zero_arm(joint)

    def set_zero_end(self, joint: Optional[int] = None) -> None:
        """末端零位标定（流程同 :meth:`set_zero_arm`）。"""
        self._require_connected()
        self._backend.set_zero_end(joint)

    def clear_fault_arm(self, joint: Optional[int] = None) -> None:
        """清除本体关节的电机故障（失能清错 → 核对 → 重新使能 → 再核对）。

        还有故障没恢复时，后端抛 ``RuntimeError`` 汇总（电机名 + 故障码）。

        :param joint: 关节索引，``None`` 表示全部电机。
        """
        self._require_connected()
        self._backend.clear_fault_arm(joint)

    def clear_fault_end(self, joint: Optional[int] = None) -> None:
        """清除末端电机故障（流程同 :meth:`clear_fault_arm`）。"""
        self._require_connected()
        self._backend.clear_fault_end(joint)

    def read_param_arm(self, key: str, joint: Optional[int] = None):
        """读本体电机参数（key 如 ``"pos_kp"``，含义由后端定义）。

        :param joint: 关节索引，``None`` 表示全部电机。
        :return: 指定 ``joint`` 时返回该电机的值；``joint=None`` 返回逐电机列表。
        """
        self._require_connected()
        return self._backend.read_param_arm(key, joint)

    def write_param_arm(self, key: str, value, joint: Optional[int] = None,
                        persist: bool = False) -> None:
        """写本体电机参数。

        :param value: 参数值，标量（作用于所选全部电机）或与电机数一致的列表。
        :param joint: 关节索引，``None`` 表示全部电机。
        :param persist: ``True`` 时同时存进电机闪存（掉电不丢）。
        """
        self._require_connected()
        return self._backend.write_param_arm(key, value, joint, persist=persist)

    def read_param_end(self, key: str, joint: Optional[int] = None):
        """读末端电机参数（key 含义由后端定义）。"""
        self._require_connected()
        return self._backend.read_param_end(key, joint)

    def write_param_end(self, key: str, value, joint: Optional[int] = None,
                        persist: bool = False) -> None:
        """写末端电机参数（参数含义同 :meth:`write_param_arm`）。"""
        self._require_connected()
        return self._backend.write_param_end(key, value, joint, persist=persist)

    # ============================================================
    # 十、基本信息与只读属性：关节数/关节名/限位/特征位形（返回的都是拷贝）
    # ============================================================
    @property
    def arm_zero(self) -> np.ndarray:
        """本体零位 ``(n_arm,)``（全零，编码器标零基准）。"""
        return self._arm_zero.copy()

    @property
    def arm_home(self) -> np.ndarray:
        """本体 home 位形 ``(n_arm,)``（config ``joyarm.arm_home``）。"""
        return self._arm_home.copy()

    @property
    def arm_neutral(self) -> np.ndarray:
        """本体数值求解默认初值 ``(n_arm,)``（全零，如 IK 迭代起点）。"""
        return self._arm_neutral.copy()

    @property
    def end_zero(self) -> np.ndarray:
        """末端零位 ``(n_end,)``（全零；无末端为空数组）。"""
        return self._end_zero.copy()

    @property
    def end_home(self) -> np.ndarray:
        """末端初始位形 ``(n_end,)``（config ``joyarm.end_home``，电机角度）。"""
        return self._end_home.copy()

    @property
    def end_neutral(self) -> np.ndarray:
        """末端数值求解默认初值 ``(n_end,)``（全零；无末端为空数组）。"""
        return self._end_neutral.copy()

    @property
    def joint_names_arm(self) -> List[str]:
        """本体关节名列表（config 顺序，即 q 向量顺序）。"""
        return list(self._arm_joint_names)

    @property
    def joint_names_end(self) -> List[str]:
        """末端电机名列表（config 顺序；无末端为空）。"""
        return list(self._end_joint_names)

    def joint_index_arm(self, name: str) -> int:
        """本体关节名 → 索引（按名字找第几个关节）。

        :raises ValueError: 名字不在列表中（消息列出可用名）。
        """
        try:
            return self._arm_joint_names.index(str(name))
        except ValueError:
            raise ValueError(
                f"joyarm.py - joint_index_arm：关节名 {name!r} 未找到；"
                f"可用：{self._arm_joint_names}") from None

    def joint_index_end(self, name: str) -> int:
        """末端电机名 → 索引。

        :raises ValueError: 名字不在列表中（消息列出可用名）。
        """
        try:
            return self._end_joint_names.index(str(name))
        except ValueError:
            raise ValueError(
                f"joyarm.py - joint_index_end：电机名 {name!r} 未找到；"
                f"可用：{self._end_joint_names}") from None

    def rand_q_arm(self,
        size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """在本体**硬限位**内随机采关节角（``(n_arm,)``；``size=N`` 给
        ``(N, n_arm)``）——教程里做工作空间采样等实验用。"""
        return rand_within_limits(self.arm_limits, size=size, rng=rng)

    def __repr__(self) -> str:
        domains = {d: (self._active_name.get(d) or "-") for d in _DOMAIN_REGISTRIES}
        return (
            f"{type(self).__name__}(model={self.model!r}, n_arm={self.n_arm}, "
            f"n_end={self.n_end}, ee_frame={self.ee_frame_name!r}, "
            f"solvers={domains}, "
            f"{'connected' if self.connected else 'offline'})"
        )

    # ============================================================
    # 十一、数学计算（不连真机也能用）：给关节角算位姿、给位姿算关节角等
    # ============================================================
    def fkine(self, q: np.ndarray, frame: Union[str, int], rep: str = "pose"):
        """正运动学：给关节角，算 ``frame`` 帧的位姿（``frame`` 必填，帧名或
        编号；``rep`` 取 ``pose``（默认，xyz+四元数）/ ``T``（4×4 矩阵）/
        ``se3``（pin.SE3））。"""
        return self._solve("fkine", "solve", self, q, frame=frame, rep=rep)

    def ikine(self, target: Pose, frame: Union[str, int], q0: np.ndarray, **kw):
        """逆运动学单解：给目标位姿，算关节角。``q0`` ``(n_arm,)`` 必填
        （数值法的迭代起点；``tol``/``iters`` 等为求解器特有参数）。"""
        return self._solve("ikine", "solve", self, target, frame, q0, **kw)

    def ikine_all(self, target: Pose, frame: Union[str, int], **kw):
        """逆运动学全部解（``q`` 为 ``(K, n_arm)``，逐解尝试 ±2π 平移尽量落
        进限位；数值法求不了多解，会抛 ``RuntimeError``）。"""
        try:
            return self._solve("ikine", "solve_all", self, target, frame, **kw)
        except NotImplementedError as e:
            raise RuntimeError(
                f"joyarm.py - ikine_all：当前激活的 ikine 成员不支持求全部解"
                f"（数值法/未实现 solve_all）：{e}") from e

    def jac(self, q: np.ndarray, frame: Union[str, int], ref: str = "base"):
        """雅可比 J(q)（``ref`` 取 ``local``/``base``：末端坐标系 / 基座系）。"""
        return self._solve("jacobian", "jac", self, q, frame=frame, ref=ref)

    def manipulability(self, q: np.ndarray, frame: Union[str, int]) -> float:
        """可操作度（衡量当前位形离奇异有多远，越大越灵活）。"""
        return self._solve("jacobian", "manipulability", self, q, frame=frame)

    def statics(self, q: np.ndarray, F: np.ndarray, frame: Union[str, int]):
        """静力学 ``τ = JᵀF``：给末端六维力 ``F``（力 N + 力矩 N·m），
        算平衡所需的各关节力矩。"""
        return self._solve("jacobian", "statics", self, q, F, frame=frame)

    # （idyn / coriolis / cartesian_inertia / fkine_vel / ikine_vel 不设门面——
    #   经激活求解器调用：求解器实例.method(arm, ...)，实例可由 set_solver 返回值取得）
    def mass_matrix(self, q):
        """关节空间惯量矩阵 M(q)。"""
        return self._solve("dynamics", "mass_matrix", self, q)

    def gravity(self, q):
        """重力项 G(q)（保持当前位形各关节需要克服的重力力矩）。"""
        return self._solve("dynamics", "gravity", self, q)

    # ============================================================
    # 十二、进阶：运动管线（三个后台线程自动跑）与运行中换算法求解器
    # 数据流：set_target_traj 写目标 → 规划线程算系数 → 采样线程产当前帧
    #        → 控制线程算指令并自动下发；set_solver 运行中可换规划器/控制器
    # ============================================================
    def set_target_traj(self, targets) -> None:
        """写入运动目标（写进去的是拷贝，之后改原对象不影响）。

        :param targets: :class:`TrajFrame` 单帧或帧列表；``None``/空列表 = 清空
            目标（规划器回退规划回 q_home）。目标是否有效、是否来不及执行，
            由规划线程检查、无效的会剔除。
        """
        if targets is None:
            frames = []
        elif isinstance(targets, TrajFrame):
            frames = [targets]
        else:
            frames = list(targets)
        self._target_traj = copy.deepcopy(frames)

    def get_target_traj(self) -> Optional[List[TrajFrame]]:
        """读取当前目标序列（返回内部快照引用，规划期间请勿改动）。"""
        return self._target_traj

    def set_current_frame(self, frame: TrajFrame) -> None:
        """写入当前轨迹帧（规划线程用；写的是拷贝）"""
        self._current_frame = copy.deepcopy(frame)

    def get_current_frame(self) -> Optional[TrajFrame]:
        """读取当前轨迹帧（控制线程用；第一帧发布前是 ``None``）。"""
        return self._current_frame

    def start_motion(self) -> None:
        """启动常规运动管线三个线程：规划 plan_hz / 采样 sample_hz / 控制 ctrl_hz

        启动时按控制器的要求自动切电机模式，并立刻算出第一帧。已在运行时再调用没有副作用。

        注意：启动时若还没写过目标，第一个周期就会规划"回家 q_home"——
        请先 ``set_target_traj`` 写目标，或确认往 home 走是安全的。

        :raises RuntimeError: 未连接真机，或 config 没配 robotics.traj /robotics.control 域。
        """
        if self._motion_running:
            return
        self._require_connected()
        planner = self._active("traj")
        controller = self._active("control")
        self.set_mode_arm(controller.MODE)      # 按控制器声明自动切电机模式
        self._traj_planned_by = None
        self._motion_threads = {
            "traj-plan": _PeriodicThread(self._tick_plan, planner.plan_hz,
                                        name="traj-plan"),
            "traj-sample": _PeriodicThread(self._tick_sample, planner.sample_hz,
                                          name="traj-sample"),
            "ctrl-step": _PeriodicThread(self._tick_ctrl, controller.ctrl_hz,
                                        name="ctrl-step"),
        }
        self._switching.set()                   # 先置暂停开关再启线程（与运行中换算法的时序一致）
        try:                                    # 线程起步期间同步算出首帧
            for worker in self._motion_threads.values():
                worker.start()
            self._motion_running = True
            self._sync_traj_tick()
        finally:
            self._switching.clear()

    def stop_motion(self, *, damping: bool = True) -> None:
        """停掉三个管线线程；默认紧接着切纯阻尼（防止停流后臂快速下坠，
        不想要这个行为传 ``damping=False``）。重复调用没有副作用。

        :raises RuntimeError: 线程卡住 1 秒还没退出（多半是阻塞在总线 IO 上）；
            此时管线视为仍在运行，排查后可重试本方法。
        """
        if not self._motion_running:
            return
        stuck = []
        for wname, worker in self._motion_threads.items():
            try:
                worker.stop()
            except RuntimeError as e:
                stuck.append(f"{wname}: {e}")
        if stuck:
            raise RuntimeError(
                f"joyarm.py - stop_motion：部分线程未按时退出（{'；'.join(stuck)}）；"
                f"请排查阻塞原因后重试 stop_motion()")
        self._motion_threads = {}
        self._traj_planned_by = None
        self._current_frame = None             # 回到"第一帧发布前"的初始状态
        self._motion_running = False
        if damping and self.connected:
            try:
                self.damping_mode()             # 安全默认：切阻尼防下坠
            except Exception as e:
                logger.warning("joyarm.py - stop_motion：纯阻尼切换失败：%s", e)

    def set_solver(self, domain: str, name: str):
        """切换某域当前使用的算法（按注册名；域 ∈ fkine/ikine/jacobian/
        dynamics/traj/control）。每个域同一时刻只有一个在用。

        平时只是换指针；若运动管线正在跑、且切的是 control 或 traj，会先让
        管线暂停一拍 → 完成切换并立刻用新算法算一拍 → 再恢复，运行中切换
        也平滑生效。返回切到的算法实例（可直接调用它的方法）。
        """
        if domain not in _DOMAIN_REGISTRIES:
            raise ValueError(
                f"joyarm.py - set_solver：未知域 {domain!r}；可用：{sorted(_DOMAIN_REGISTRIES)}")
        inst = self._pick(domain, name)   # 校验已加载
        if self._motion_running and domain == "control":
            self._require_connected()     # 切电机模式需在线
            self._switching.set()
            try:
                self.set_mode_arm(inst.MODE)      # 先切模式（暂停期间不发指令）
                self._active_name[domain] = name
                worker = self._motion_threads.get("ctrl-step")
                if worker is not None:            # 并发 stop 极端下优雅降级
                    worker.set_hz(inst.ctrl_hz)
                try:
                    self._dispatch_ctrl()          # 新控制器立即下发首拍指令
                except Exception as e:
                    logger.warning("joyarm.py - set_solver：切换后首拍指令下发失败"
                                   "（ctrl 线程将按节拍重试）：%s", e)
            finally:
                self._switching.clear()
        elif self._motion_running and domain == "traj":
            self._switching.set()
            try:
                self._active_name[domain] = name
                self._sync_traj_tick()             # 当前帧立即来自新规划器
                for wname, hz in (("traj-plan", inst.plan_hz),
                                  ("traj-sample", inst.sample_hz)):
                    worker = self._motion_threads.get(wname)
                    if worker is not None:
                        worker.set_hz(hz)
            finally:
                self._switching.clear()
        else:
            self._active_name[domain] = name
        return inst

    def list_solvers(self, domain: str) -> list:
        """列出某域已加载的算法名（当前在用的排第一个，其余按字母序）。"""
        names = sorted(self._members(domain))
        active = self._active_name.get(domain)
        if active in names:
            names.remove(active)
            names.insert(0, active)
        return names
