"""``JoyArm`` —— 完整机械臂基类（arm本体 + end执行器），**组合根**。

持有 pinocchio ``model``/``data`` + 限位；一个整机通信后端（``backend``，按 yaml ``backend:`` 段的 ``name`` 经 backends REGISTRY 选型构建）
与六个策略成员（``_fkine_solver``/``_ikine_solver``/``_jacobian_solver``/``_dynamics_solver``/``_traj_planner``/``_controller``，
按 yaml ``solvers:`` 段经各域 REGISTRY 选型组装）；公开门面（``fkine``/``plan_joint_p2p``/``play_joint`` 等）全部委托私有成员，换配置即换算法。
离线（``connected=False`` 默认）：计算类（``fkine``/``jac``/…）随时可用；执行类（``get_arm_state``/``set_arm_command``/``end_open``/…）``connect()`` 前抛 RuntimeError。
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import List, Optional, Union

import numpy as np

from ..utils.limits import clamp_to_limits
from ..utils.types import (
    ArmState,
    ControlMode,
    JointLimits,
    Pose,
    TcpLimits,
)

try:
    import pinocchio as pin
except ImportError:
    pin = None

from ..backends import get_backend
from ..robotics.fkine import REGISTRY as _FKINE_REGISTRY
from ..robotics.ikine import REGISTRY as _IKINE_REGISTRY
from ..robotics.jacobian import REGISTRY as _JACOBIAN_REGISTRY
from ..robotics.dynamics import REGISTRY as _DYNAMICS_REGISTRY
from ..robotics.trajectory import REGISTRY as _TRAJ_REGISTRY
from ..robotics.control import REGISTRY as _CONTROL_REGISTRY, play_joint_trajectory, play_cart_trajectory

__all__ = ["JoyArm", "load_config"]


# configs/ 目录（joyarm.py 位于 joyarm_core/joyarms/，上溯一级即包根）
_CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
)


def load_config(model: str, strict: bool = False) -> Optional[dict]:
    """加载 ``configs/<model>.yaml`` 型号配置。

    :param model: 型号名（与 yaml 文件名、yaml ``name`` 字段、joyarms 注册名一致）。
    :param strict: 严格模式（``JoyArmFactory`` 路径）：文件缺失/解析失败抛
        ``ValueError``（列出可用型号）；缺省容错返回 ``None``（调用方回退类常量）。
    """
    try:
        import yaml

        with open(os.path.join(_CONFIGS_DIR, f"{model}.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"顶层应为映射，实际 {type(cfg).__name__}")
        return cfg
    except Exception as e:
        if not strict:
            return None
        try:
            avail = sorted(f[:-5] for f in os.listdir(_CONFIGS_DIR) if f.endswith(".yaml"))
        except OSError:
            avail = []
        raise ValueError(f"『{model}』型号在 configs 中未找到；可用：{avail}") from e


# config solvers 段 "default" 别名对应的各域默认注册名
_DOMAIN_DEFAULTS = {
    "fkine": "pin",
    "ikine": "pin",
    "jacobian": "pin",
    "dynamics": "pin",
    "traj": "default",
    "control": "position",
}


def _build_component(registry: dict, spec, domain: str):
    """按 config 规格实例化策略成员：值为注册名字符串（``"default"`` 转各域默认），或 ``{name: ..., **参数}``。"""
    if isinstance(spec, str):
        name, params = spec, {}
    else:
        params = dict(spec)
        name = params.pop("name", None)
        if name is None:
            raise ValueError(f"solvers.{domain} 需为名字字符串或含 name 键的映射")
    if name == "default":
        name = _DOMAIN_DEFAULTS.get(domain, name)
    if name not in registry:
        raise ValueError(f"『{name}』型号在 solvers.{domain} 中未找到；可用：{sorted(registry)}")
    return registry[name](**params)


class JoyArm:
    """完整机械臂基类。

    :param name: 名称；``urdf_path``: URDF 路径；``ee_frame_name``: 末端帧名（默认 ``"ee"``）。
    :param mesh_dirs: mesh 搜索目录；``load_geometry``: 是否加载 visual/collision 几何。
    :param config: 型号 YAML 字典（``solvers``/``backend`` 等段）；缺省无后端（离线）。
    """

    def __init__(self,
        name: str,
        urdf_path: str,
        ee_frame_name: str = "ee",
        mesh_dirs: Optional[List[str]] = None,
        load_geometry: bool = False,
        config: Optional[dict] = None,
    ):
        if pin is None:
            raise ImportError(
                "JoyArm 需要 pinocchio 才能加载 URDF 与计算运动学。请安装：\n"
                "  uv pip install pin\n"
                "  # 或：pip install pin"
            )

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(
                f"未找到 URDF 文件：{urdf_path}\n"
                f"请将正式 URDF 放入 joyarm_core/robots/。"
            )

        cfg = config or {}

        # ---- 构建 pinocchio 模型 ----
        if mesh_dirs:
            self.model = pin.buildModelFromUrdf(urdf_path, package_dirs=mesh_dirs)
        else:
            self.model = pin.buildModelFromUrdf(urdf_path)

        self.collision_model: Optional[object] = None
        self.visual_model: Optional[object] = None
        if load_geometry:
            try:
                geo_dirs = mesh_dirs or [os.path.dirname(urdf_path)]
                self.collision_model = pin.buildGeomFromUrdf(
                    self.model, urdf_path, pin.GeometryType.COLLISION, package_dirs=geo_dirs
                )
                self.visual_model = pin.buildGeomFromUrdf(
                    self.model, urdf_path, pin.GeometryType.VISUAL, package_dirs=geo_dirs
                )
            except Exception as e:
                import warnings

                warnings.warn(f"几何模型加载失败：{e}")
        self.data = self.model.createData()

        # ---- 基本属性 ----
        self.n: int = self.model.nq
        self.nv: int = self.model.nv
        self.name: str = name
        self.connected: bool = False

        # ---- 末端帧 ----
        self.ee_frame_name: str = ee_frame_name
        # pinocchio getFrameId 对未知名不抛异常而是返回 nframes，据此判缺
        self.ee_frame_id: int = self.model.getFrameId(ee_frame_name)
        if self.ee_frame_id >= len(self.model.frames):
            raise ValueError(
                f"URDF 中找不到末端帧 '{ee_frame_name}'；"
                f"可用帧：{[f.name for f in self.model.frames]}"
            )

        # ---- 关节限位（硬 + 软）----
        # margin 具体值由 yaml 的 joint_limits_soft_margin 提供；未配置时取 0（软=硬）
        self.joint_limits: JointLimits = self._build_joint_limits(self.model)
        self.joint_limits_soft: JointLimits = self._build_soft_limits(
            self.joint_limits, margin=float(cfg.get("joint_limits_soft_margin", 0.0))
        )
        self.qlow: np.ndarray = self.joint_limits_soft.q_min
        self.qhigh: np.ndarray = self.joint_limits_soft.q_max

        # ---- 特征位形（config 优先，缺失回退 q_zero）----
        self._q_zero: np.ndarray = np.clip(
            np.asarray(cfg.get("q_zero", np.zeros(self.n)), dtype=float).reshape(-1),
            self.joint_limits.q_min,
            self.joint_limits.q_max,
        )
        self._q_home: np.ndarray = np.asarray(
            cfg.get("q_home", self._q_zero), dtype=float
        ).reshape(-1)
        self._q_neutral: np.ndarray = np.asarray(
            cfg.get("q_neutral", self._q_zero), dtype=float
        ).reshape(-1)

        self.tcp_limits: TcpLimits = TcpLimits()  # 末端限位（占位）
        self.T_base: np.ndarray = np.eye(4)  # 基坐标系偏移

        # ---- 整机通信后端：config backend 段 name 选型（本体+末端一体，私有）----
        self._backend = None
        backend_cfg = cfg.get("backend")
        if backend_cfg:
            backend_cfg = dict(backend_cfg)
            backend_name = backend_cfg.pop("name", None)
            if backend_name is None:
                raise ValueError("config backend 段缺少选型键 name")
            self._backend = get_backend(backend_name)(backend_cfg)

        # ---- 策略成员：按 config solvers 段 + 各域注册表组装（算法可换）----
        solvers_cfg = cfg.get("solvers", {})
        self._fkine_solver = _build_component(_FKINE_REGISTRY, solvers_cfg.get("fkine", "default"), "fkine")
        self._ikine_solver = _build_component(_IKINE_REGISTRY, solvers_cfg.get("ikine", "default"), "ikine")
        self._jacobian_solver = _build_component(_JACOBIAN_REGISTRY, solvers_cfg.get("jacobian", "default"), "jacobian")
        self._dynamics_solver = _build_component(_DYNAMICS_REGISTRY, solvers_cfg.get("dynamics", "default"), "dynamics")
        self._traj_planner = _build_component(_TRAJ_REGISTRY, solvers_cfg.get("traj", "default"), "traj")
        self._controller = _build_component(_CONTROL_REGISTRY, solvers_cfg.get("control", "default"), "control")

    # ----------------------------------------------------------
    # 特征位形（只读，返回拷贝）
    # ----------------------------------------------------------
    @property
    def q_zero(self) -> np.ndarray:
        """硬件零位 ``(n,)``（编码器标零基准）。"""
        return self._q_zero.copy()

    @property
    def q_home(self) -> np.ndarray:
        """上电初始位形 ``(n,)``。"""
        return self._q_home.copy()

    @property
    def q_neutral(self) -> np.ndarray:
        """数值求解默认初值 ``(n,)``（如 IK 迭代起点）。"""
        return self._q_neutral.copy()

    # ----------------------------------------------------------
    # 真机连接（connect 后才可执行 arm_*/end_*）
    # ----------------------------------------------------------
    def connect(self) -> None:
        """连接真机（整机后端：本体 + 末端）。"""
        if self._backend is not None:
            self._backend.connect()
        self.connected = True

    def disconnect(self) -> None:
        """断开真机（整机后端：本体 + 末端）。"""
        if self._backend is not None:
            self._backend.disconnect()
        self.connected = False

    def _require_connected(self) -> None:
        """执行类方法前置：未连接真机（离线）时抛 ``RuntimeError``。"""
        if not self.connected:
            raise RuntimeError(f"[{self.name}] 未连接真机（离线）；请先 connect()。")

    # ----------------------------------------------------------
    # 内部：限位构建
    # ----------------------------------------------------------
    @staticmethod
    def _build_joint_limits(model) -> JointLimits:
        """从 pinocchio model 解析硬限位（位置/速度/力矩来自 URDF；其余手动赋值）。"""
        n = model.nq
        q_min = np.asarray(model.lowerPositionLimit, dtype=float).reshape(n)
        q_max = np.asarray(model.upperPositionLimit, dtype=float).reshape(n)
        nv = model.nv
        dq_max = (
            np.asarray(model.velocityLimit, dtype=float).reshape(nv)
            if hasattr(model, "velocityLimit")
            else np.full(nv, np.inf)
        )
        tau_max = (
            np.asarray(model.effortLimit, dtype=float).reshape(nv)
            if hasattr(model, "effortLimit")
            else np.full(nv, np.inf)
        )
        if nv != n:
            dq_max = np.full(n, np.inf)
            tau_max = np.full(n, np.inf)
        return JointLimits(
            q_min=q_min,
            q_max=q_max,
            dq_max=dq_max,
            tau_max=tau_max,
            ddq_max=np.full(n, np.inf),
            voltage_min=np.zeros(n),
            voltage_max=np.zeros(n),
            current_max=np.zeros(n),
        )

    @staticmethod
    def _build_soft_limits(hard: JointLimits, margin: float) -> JointLimits:
        """由硬限位内缩 ``margin`` 比例生成软限位（``margin=0`` 时软=硬）。"""
        span = hard.q_max - hard.q_min
        return replace(hard, q_min=hard.q_min + margin * span, q_max=hard.q_max - margin * span)

    # ----------------------------------------------------------
    # 打印表示
    # ----------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, n={self.n}, "
            f"ee_frame={self.ee_frame_name!r}, "
            f"backend={'yes' if self._backend else 'no'}, "
            f"{'connected' if self.connected else 'offline'})"
        )

    # ----------------------------------------------------------
    # 关节角采样 / 裁剪 / 校验（基于软限位）
    # ----------------------------------------------------------
    def rand_q(self,
        size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """在**软限位**内均匀采样关节角。"""
        rng = rng if rng is not None else np.random.default_rng()
        low, high = self.qlow, self.qhigh
        if size is None:
            return rng.uniform(low, high)
        return rng.uniform(low, high, size=(size, self.n))

    def clamp_q(self, q: np.ndarray) -> np.ndarray:
        """将关节角裁剪到**软限位**内（薄委托 :func:`joyarm_core.utils.limits.clamp_to_limits`）。"""
        return clamp_to_limits(q, self.joint_limits_soft)

    def is_q_valid(self, q: np.ndarray) -> bool:
        """关节角是否在**软限位**内（单点）。"""
        q = np.asarray(q, dtype=float).reshape(-1)
        return bool(np.all(q >= self.qlow - 1e-9) and np.all(q <= self.qhigh + 1e-9))

    # ----------------------------------------------------------
    # robotics 求解算法（门面 → 私有策略成员，config solvers 段可换实现）
    # ----------------------------------------------------------
    def fkine(self, q: np.ndarray, frame: Optional[Union[str, int]] = None, rep: str = "T"):
        """正运动学（``rep`` 取 ``quat``/``T``/``se3``：Pose / 4×4 矩阵 / pin.SE3）。"""
        return self._fkine_solver.solve(self, q, frame=frame, rep=rep)

    def ikine(self, T_target, q0=None, frame=None, **kw):
        """逆运动学（委托 ``_ikine_solver``；数值/解析可换）。"""
        return self._ikine_solver.solve(self, T_target, q0=q0, frame=frame, **kw)

    def ikine_constrained(self, T_target, q0=None, frame=None, **kw):
        """带关节限位约束的逆运动学。"""
        return self._ikine_solver.solve_constrained(self, T_target, q0=q0, frame=frame, **kw)

    def jac(self, q: np.ndarray, frame: Optional[Union[str, int]] = None, ref: str = "local"):
        """雅可比 J(q)（``ref`` 取 ``local``/``base``：末端帧系 / 基座系）。"""
        return self._jacobian_solver.jac(self, q, frame=frame, ref=ref)

    def manipulability(self, q: np.ndarray, frame: Optional[Union[str, int]] = None) -> float:
        """Yoshikawa 可操作度（雅可比衍生量）。"""
        return self._jacobian_solver.manipulability(self, q, frame=frame)

    def cond_number(self, q: np.ndarray, frame: Optional[Union[str, int]] = None) -> float:
        """雅可比条件数（雅可比衍生量）。"""
        return self._jacobian_solver.cond_number(self, q, frame=frame)

    def statics(self, q: np.ndarray, F: np.ndarray, frame: Optional[Union[str, int]] = None):
        """静力学 τ = JᵀF（雅可比衍生量）。"""
        return self._jacobian_solver.statics(self, q, F, frame=frame)

    def idyn(self, q, dq, ddq, f_ext=None):
        """逆动力学（委托 ``_dynamics_solver``）。"""
        return self._dynamics_solver.idyn(self, q, dq, ddq, f_ext=f_ext)

    def mass_matrix(self, q):
        """关节空间惯量矩阵 M(q)（委托 ``_dynamics_solver``）。"""
        return self._dynamics_solver.mass_matrix(self, q)

    def coriolis(self, q, dq):
        """科氏+向心项 C(q,q̇)（委托 ``_dynamics_solver``）。"""
        return self._dynamics_solver.coriolis(self, q, dq)

    def gravity(self, q):
        """重力项 G(q)（委托 ``_dynamics_solver``）。"""
        return self._dynamics_solver.gravity(self, q)

    def cartesian_inertia(self, q, frame=None):
        """笛卡尔惯量 Λ=J⁻ᵀMJ⁻¹（M ⊕ ``arm.jac`` 模板）。"""
        return self._dynamics_solver.cartesian_inertia(self, q, frame=frame)

    def plan_joint_p2p(self, q0, qf, *, method="quintic", **kw):
        """关节空间点到点轨迹（``method`` 选 cubic/quintic/lspb）。"""
        return self._traj_planner.plan_joint_p2p(self, q0, qf, method=method, **kw)

    def plan_joint_waypoints(self, qs, Ts, **kw):
        """关节空间多点途经轨迹（段间平滑拼接）。"""
        return self._traj_planner.plan_joint_waypoints(self, qs, Ts, **kw)

    def plan_cart_p2p(self, *, method="line", **kw):
        """笛卡尔点到点（``method`` 选 line/arc；line 需 T0/Tf，arc 需 center/radius/T_start/angle）。"""
        return self._traj_planner.plan_cart_p2p(self, method=method, **kw)

    def plan_cart_waypoints(self, poses, Ts, **kw):
        """笛卡尔多点途经轨迹（段间平滑拼接）。"""
        return self._traj_planner.plan_cart_waypoints(self, poses, Ts, **kw)

    def set_controller(self, name):
        """运行期切换控制律（按注册名，如 ``"position"``）。"""
        self._controller = _build_component(_CONTROL_REGISTRY, name, "control")
        return self._controller

    def play_joint(self, traj, mode=ControlMode.POSITION, hz: int = 200):
        """按时间序列回放关节轨迹（ControlLoop 驱动 ``_controller``）。"""
        return play_joint_trajectory(self, traj, mode=mode, hz=hz)

    def play_cart(self, traj, hz: int = 200, **kw):
        """回放笛卡尔轨迹（OSC：任务空间 PD → JᵀF + 重力补偿 → MIT 下发）。"""
        return play_cart_trajectory(self, traj, hz=hz, **kw)

    # ----------------------------------------------------------
    # 本体执行类方法（依赖 _backend；未连接 raise；arm_* 与 end_* 对应）
    # ----------------------------------------------------------
    def enable_arm(self, joint: Optional[int] = None) -> None:
        """使能本体关节电机（``joint=None`` 全部）。"""
        self._require_connected()
        self._backend.enable_arm(joint)

    def disable_arm(self, joint: Optional[int] = None) -> None:
        """失能本体关节电机。"""
        self._require_connected()
        self._backend.disable_arm(joint)

    def set_zero_arm(self, joint: Optional[int] = None) -> None:
        """本体零位标定（先失能，反馈无故障后再标零）。"""
        self._require_connected()
        self._backend.set_zero_arm(joint)

    def set_mode_arm(self, mode: ControlMode = ControlMode.POSITION,
                     joint: Optional[int] = None) -> None:
        """切换本体控制模式（收指令前必须先切到对应模式；默认位置模式，``joint=None`` 全部）。"""
        self._require_connected()
        self._backend.set_mode_arm(mode, joint)

    def get_arm_state(self) -> ArmState:
        """读取本体状态快照（委托 ``_backend.read_state_arm()``；与 ``get_end_state`` 对应）。

        :raises RuntimeError: 未连接真机（``connected=False``）时抛出。
        """
        self._require_connected()
        state = self._backend.read_state_arm()
        state.tcp.pose = Pose.from_T(self.fkine(state.joint.q))
        return state

    def set_arm_command(self,
        mode: ControlMode = ControlMode.POSITION,
        q: Optional[np.ndarray] = None,
        dq: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> None:
        """按控制模式下发运动指令（委托 ``_backend``；三态 POSITION/VELOCITY/MIT）。

        MIT 模式 ``kp/kd`` 可缺省（``None`` 透传后端回退 config 增益）；
        纯力矩不设独立模式，经 MIT（``kp=kd=0``）实现。

        :raises RuntimeError: 未连接真机时抛出。
        :raises ValueError: 对应模式所需参数缺失时抛出。
        """
        self._require_connected()
        if mode == ControlMode.POSITION:
            if q is None:
                raise ValueError("POSITION 模式需要 q")
            self._backend.send_position_arm(q)
        elif mode == ControlMode.VELOCITY:
            if dq is None:
                raise ValueError("VELOCITY 模式需要 dq")
            self._backend.send_velocity_arm(dq)
        elif mode == ControlMode.MIT:
            missing = [
                name for name, val in (("q", q), ("dq", dq), ("tau", tau)) if val is None
            ]
            if missing:
                raise ValueError(f"MIT 模式缺少参数：{missing}")
            self._backend.send_mit_arm(q, dq, tau, kp, kd)
        else:
            raise ValueError(f"未知控制模式：{mode}")

    # ----------------------------------------------------------
    # 电机参数读写（依赖 _backend；未连接 raise）
    # ----------------------------------------------------------
    def read_param_arm(self, key: str, joint: Optional[int] = None):
        """读本体关节电机参数（key 为参数名，语义由后端定义）。

        :param key: 参数名（如 ``"pos_kp"``）。
        :param joint: 关节索引，``None`` 表示全部电机。
        :return: 指定 ``joint`` 时返回该电机参数值（float 或 int）；
            ``joint=None`` 时返回逐电机参数值列表。
        """
        self._require_connected()
        return self._backend.read_param_arm(key, joint)

    def write_param_arm(self, key: str, value, joint: Optional[int] = None,
                        persist: bool = False) -> None:
        """写本体关节电机参数。

        :param key: 参数名。
        :param value: 参数值，标量（作用于所选全部电机）或与所选电机数一致的列表。
        :param joint: 关节索引，``None`` 表示全部电机。
        :param persist: ``True`` 时写入并持久化到非易失存储。
        """
        self._require_connected()
        self._backend.write_param_arm(key, value, joint, persist=persist)

    def read_param_end(self, key: str, joint: Optional[int] = None):
        """读末端电机参数（key 为参数名，语义由后端定义）。

        :param key: 参数名（如 ``"pos_kp"``）。
        :param joint: 末端电机索引，``None`` 表示全部电机。
        :return: 指定 ``joint`` 时返回该电机参数值；``joint=None`` 时返回逐电机列表。
        """
        self._require_connected()
        return self._backend.read_param_end(key, joint)

    def write_param_end(self, key: str, value, joint: Optional[int] = None,
                        persist: bool = False) -> None:
        """写末端电机参数。

        :param key: 参数名。
        :param value: 参数值，标量（作用于所选全部电机）或与所选电机数一致的列表。
        :param joint: 末端电机索引，``None`` 表示全部电机。
        :param persist: ``True`` 时写入并持久化到非易失存储。
        """
        self._require_connected()
        self._backend.write_param_end(key, value, joint, persist=persist)

    # ----------------------------------------------------------
    # 末端执行类方法（依赖 _backend；未连接 raise）
    # ----------------------------------------------------------
    def enable_end(self, joint: Optional[int] = None) -> None:
        """使能末端执行器电机（``joint=None`` 全部）。"""
        self._require_connected()
        self._backend.enable_end(joint)

    def disable_end(self, joint: Optional[int] = None) -> None:
        """失能末端执行器电机。"""
        self._require_connected()
        self._backend.disable_end(joint)

    def set_zero_end(self, joint: Optional[int] = None) -> None:
        """末端零位标定（先失能，反馈无故障后再标零）。"""
        self._require_connected()
        self._backend.set_zero_end(joint)

    def set_mode_end(self, mode: ControlMode = ControlMode.POSITION,
                     joint: Optional[int] = None) -> None:
        """切换末端控制模式（收指令前必须先切到对应模式；默认位置模式，``joint=None`` 全部）。"""
        self._require_connected()
        self._backend.set_mode_end(mode, joint)

    def end_open(self, joint: Optional[int] = None) -> None:
        """张开末端到最大（默认行程/力度；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_action_end("open", joint)

    def end_close(self, joint: Optional[int] = None) -> None:
        """闭合末端（夹到默认力度即停；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_action_end("close", joint)

    def set_end_position(self, position, joint: Optional[int] = None) -> None:
        """末端位置控制（连续量，如夹爪电机弧度；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_position_end(position, joint)

    def set_end_force(self, force, joint: Optional[int] = None) -> None:
        """末端力度控制（如夹持力 N；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_force_end(force, joint)

    def get_end_state(self, joint: Optional[int] = None) -> dict:
        """读取末端状态（字段由后端定义；值为所选电机的逐电机序列）。

        :param joint: 末端电机索引，``None`` 表示全部。
        :raises RuntimeError: 未连接真机时抛出。
        """
        self._require_connected()
        return self._backend.read_state_end(joint)
