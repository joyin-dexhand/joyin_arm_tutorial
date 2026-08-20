"""``JoyArm`` —— 完整机械臂基类（多轴本体 + 末端执行器），**组合根**。

持有 pinocchio ``model``/``data`` + 限位；两个通信后端（``backend_arm``/``backend_end``，
子类绑 ``*_cls``）与六个策略成员（``_fkine_solver``/``_ikine_solver``/``_jacobian_solver``/
``_dynamics_solver``/``_traj_planner``/``_controller``）——后者按 yaml ``solvers:`` 段经各域
REGISTRY 选型组装；公开门面（``fkine``/``plan_joint``/``play`` 等）全部委托私有成员，
换配置即换算法。离线（``connected=False`` 默认）：计算类（``fkine``/``jac``/…）随时可用；
执行类（``get_arm_state``/``set_arm_command``/``end_open``/…）``connect()`` 前抛 RuntimeError。
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from ..utils.types import (
    ArmState,
    ControlMode,
    JointLimits,
    TcpLimits,
)

try:
    import pinocchio as pin
except ImportError:
    pin = None

from ..backends.backend_arm import BackendArm
from ..backends.backend_end import BackendEnd
from ..robotics.fkine import REGISTRY as _FKINE_REGISTRY
from ..robotics.ikine import REGISTRY as _IKINE_REGISTRY
from ..robotics.jacobian import REGISTRY as _JACOBIAN_REGISTRY
from ..robotics.dynamics import REGISTRY as _DYNAMICS_REGISTRY
from ..robotics.trajectory import REGISTRY as _TRAJ_REGISTRY
from ..robotics.control import REGISTRY as _CONTROL_REGISTRY

__all__ = ["JoyArm"]


def _build_component(registry: dict, spec, domain: str):
    """按 config 规格实例化策略成员：值为注册名字符串，或 ``{name: ..., **参数}``。"""
    if isinstance(spec, str):
        name, params = spec, {}
    else:
        params = dict(spec)
        name = params.pop("name", None)
        if name is None:
            raise ValueError(f"solvers.{domain} 需为名字字符串或含 name 键的映射")
    if name not in registry:
        raise ValueError(f"未知 solvers.{domain}={name!r}，可用：{sorted(registry)}")
    return registry[name](**params)


class JoyArm:
    """完整机械臂基类。

    :param name: 名称；``urdf_path``: URDF 路径；``ee_frame_name``: 末端帧名（默认 ``"ee"``）。
    :param mesh_dirs: mesh 搜索目录；``load_geometry``: 是否加载 visual/collision 几何。
    :param config: 型号 YAML 字典（``solvers``/``backend_arm``/``backend_end`` 等段）；缺省无后端。
    """

    # ---- 型号绑定，构造时按 config 实例化 ----
    backend_arm_cls: Optional[type] = None
    backend_end_cls: Optional[type] = None

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

        import os

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
        try:
            self.ee_frame_id: int = self.model.getFrameId(ee_frame_name)
        except Exception:
            self.ee_frame_id = -1
            for i, f in enumerate(self.model.frames):
                if f.name == ee_frame_name:
                    self.ee_frame_id = i
                    break
            if self.ee_frame_id < 0:
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

        # ---- 中性位形 ----
        self.q_neutral: np.ndarray = np.clip(
            pin.neutral(self.model),
            self.joint_limits.q_min,
            self.joint_limits.q_max,
        )

        self.tcp_limits: TcpLimits = TcpLimits()  # 末端限位（占位）
        self.T_base: np.ndarray = np.eye(4)  # 基坐标系偏移

        # ---- 两个通信后端：子类绑 *_cls，按 config 对应段实例化 ----
        self.backend_arm: Optional[BackendArm] = (
            self.backend_arm_cls(**cfg.get("backend_arm", {}))
            if self.backend_arm_cls is not None
            else None
        )
        self.backend_end: Optional[BackendEnd] = (
            self.backend_end_cls(**cfg.get("backend_end", {}))
            if self.backend_end_cls is not None
            else None
        )

        # ---- 策略成员：按 config solvers 段 + 各域注册表组装（算法可换）----
        solvers_cfg = cfg.get("solvers", {})
        self._fkine_solver = _build_component(_FKINE_REGISTRY, solvers_cfg.get("fkine", "pin"), "fkine")
        self._ikine_solver = _build_component(_IKINE_REGISTRY, solvers_cfg.get("ikine", "pin"), "ikine")
        self._jacobian_solver = _build_component(_JACOBIAN_REGISTRY, solvers_cfg.get("jacobian", "pin"), "jacobian")
        self._dynamics_solver = _build_component(_DYNAMICS_REGISTRY, solvers_cfg.get("dynamics", "pin"), "dynamics")
        self._traj_planner = _build_component(_TRAJ_REGISTRY, solvers_cfg.get("traj", "default"), "traj")
        self._controller = _build_component(_CONTROL_REGISTRY, solvers_cfg.get("control", "position"), "control")

    # ----------------------------------------------------------
    # 真机连接（connect 后才可执行 arm_*/end_*）
    # ----------------------------------------------------------
    def connect(self) -> None:
        """连接本体 + 末端真机。"""
        if self.backend_arm is not None:
            self.backend_arm.connect()
        if self.backend_end is not None:
            self.backend_end.connect()
        self.connected = True

    def disconnect(self) -> None:
        """断开本体 + 末端真机。"""
        if self.backend_arm is not None:
            self.backend_arm.disconnect()
        if self.backend_end is not None:
            self.backend_end.disconnect()
        self.connected = False

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
            temp_motor_max=np.zeros(n),
            temp_driver_max=np.zeros(n),
            voltage_min=np.zeros(n),
            voltage_max=np.zeros(n),
            current_max=np.zeros(n),
        )

    @staticmethod
    def _build_soft_limits(hard: JointLimits, margin: float) -> JointLimits:
        """由硬限位内缩 ``margin`` 比例生成软限位（``margin=0`` 时软=硬）。"""
        span = hard.q_max - hard.q_min
        qlow = hard.q_min + margin * span
        qhigh = hard.q_max - margin * span
        return JointLimits(
            q_min=qlow,
            q_max=qhigh,
            dq_max=hard.dq_max.copy(),
            tau_max=hard.tau_max.copy(),
            ddq_max=hard.ddq_max.copy(),
            temp_motor_max=hard.temp_motor_max.copy(),
            temp_driver_max=hard.temp_driver_max.copy(),
            voltage_min=hard.voltage_min.copy(),
            voltage_max=hard.voltage_max.copy(),
            current_max=hard.current_max.copy(),
        )

    # ----------------------------------------------------------
    # 打印表示
    # ----------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, n={self.n}, "
            f"ee_frame={self.ee_frame_name!r}, "
            f"end={'yes' if self.backend_end else 'no'}, "
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
        from ..utils.limits import clamp_to_limits

        return clamp_to_limits(q, self.joint_limits_soft)

    def is_q_valid(self, q: np.ndarray) -> bool:
        """关节角是否在**软限位**内（单点）。"""
        q = np.asarray(q, dtype=float).reshape(-1)
        return bool(np.all(q >= self.qlow - 1e-9) and np.all(q <= self.qhigh + 1e-9))

    # ----------------------------------------------------------
    # robotics 求解算法（门面 → 私有策略成员，config solvers 段可换实现）
    # ----------------------------------------------------------
    def fkine(self, q: np.ndarray, frame: Optional[Union[str, int]] = None, rep: str = "T"):
        """正运动学（委托 ``_fkine_solver``；默认 pinocchio 黑盒，config ``solvers.fkine`` 可换）。"""
        return self._fkine_solver.solve(self, q, frame=frame, rep=rep)

    def ikine(self, T_target, q0=None, frame=None, **kw):
        """逆运动学（委托 ``_ikine_solver``；数值/解析可换，Ch3 实现）。"""
        return self._ikine_solver.solve(self, T_target, q0=q0, frame=frame, **kw)

    def ikine_constrained(self, T_target, q0=None, frame=None, **kw):
        """带关节限位约束的逆运动学（Ch3 实现）。"""
        return self._ikine_solver.solve_constrained(self, T_target, q0=q0, frame=frame, **kw)

    def jac(self, q: np.ndarray, frame: Optional[Union[str, int]] = None, ref: str = "local"):
        """雅可比 J(q)（委托 ``_jacobian_solver``，Ch4 实现）。"""
        return self._jacobian_solver.jac(self, q, frame=frame, ref=ref)

    def manipulability(self, q: np.ndarray, frame: Optional[Union[str, int]] = None) -> float:
        """Yoshikawa 可操作度（雅可比衍生量，Ch4 实现）。"""
        return self._jacobian_solver.manipulability(self, q, frame=frame)

    def cond_number(self, q: np.ndarray, frame: Optional[Union[str, int]] = None) -> float:
        """雅可比条件数（雅可比衍生量，Ch4 实现）。"""
        return self._jacobian_solver.cond_number(self, q, frame=frame)

    def statics(self, q: np.ndarray, F: np.ndarray, frame: Optional[Union[str, int]] = None):
        """静力学 τ = JᵀF（雅可比衍生量，Ch4 实现）。"""
        return self._jacobian_solver.statics(self, q, F, frame=frame)

    def fdyn(self, q, dq, tau, f_ext=None):
        """正动力学（委托 ``_dynamics_solver``，Ch8 实现）。"""
        return self._dynamics_solver.fdyn(self, q, dq, tau, f_ext=f_ext)

    def idyn(self, q, dq, ddq, f_ext=None):
        """逆动力学（委托 ``_dynamics_solver``，Ch8 实现）。"""
        return self._dynamics_solver.idyn(self, q, dq, ddq, f_ext=f_ext)

    def mass_matrix(self, q):
        """关节空间惯量矩阵 M(q)（委托 ``_dynamics_solver``，Ch8 实现）。"""
        return self._dynamics_solver.mass_matrix(self, q)

    def coriolis(self, q, dq):
        """科氏+向心项 C(q,q̇)（委托 ``_dynamics_solver``，Ch8 实现）。"""
        return self._dynamics_solver.coriolis(self, q, dq)

    def gravity(self, q):
        """重力项 G(q)（委托 ``_dynamics_solver``，Ch8 实现）。"""
        return self._dynamics_solver.gravity(self, q)

    def cartesian_inertia(self, q, frame=None):
        """笛卡尔惯量 Λ=J⁻ᵀMJ⁻¹（M ⊕ ``arm.jac`` 模板，Ch8/9 实现）。"""
        return self._dynamics_solver.cartesian_inertia(self, q, frame=frame)

    def plan_joint(self, q0, qf, *, method="quintic", **kw):
        """关节空间点到点轨迹（委托 ``_traj_planner``；method 选 cubic/quintic/lspb）。"""
        return self._traj_planner.plan_joint(self, q0, qf, method=method, **kw)

    def plan_waypoints(self, qs, Ts, **kw):
        """关节空间多点途经轨迹（段间平滑拼接）。"""
        return self._traj_planner.plan_waypoints(self, qs, Ts, **kw)

    def plan_cart(self, *, method="line", **kw):
        """笛卡尔空间轨迹（method 选 line/arc；line 需 T0/Tf，arc 需 center/radius/T_start/angle）。"""
        return self._traj_planner.plan_cart(self, method=method, **kw)

    def set_controller(self, name):
        """运行期切换控制律（按注册名，如 ``"position"``）。"""
        self._controller = _build_component(_CONTROL_REGISTRY, name, "control")
        return self._controller

    def play(self, traj, mode=ControlMode.POSITION, hz: int = 200):
        """按时间序列回放轨迹（ControlLoop 驱动 ``_controller``，Ch6 实现）。"""
        from ..robotics.control import play_trajectory

        return play_trajectory(self, traj, mode=mode, hz=hz)

    # ----------------------------------------------------------
    # 本体执行类方法（依赖 backend_arm；未连接 raise；arm_* 与 end_* 对应）
    # ----------------------------------------------------------
    def get_arm_state(self) -> ArmState:
        """读取本体状态快照（委托 ``backend_arm.read_state()``；与 ``get_end_state`` 对应）。

        :raises RuntimeError: 未连接真机（``connected=False``）时抛出。
        """
        if not self.connected:
            raise RuntimeError(f"[{self.name}] 未连接真机（离线）；请先 connect()。")
        return self.backend_arm.read_state()

    def set_arm_command(self,
        mode: ControlMode = ControlMode.POSITION,
        q: Optional[np.ndarray] = None,
        dq: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> None:
        """按控制模式下发运动指令（委托 ``backend_arm``；三态 POSITION/VELOCITY/MIT）。

        MIT 模式 ``kp/kd`` 可缺省（``None`` 透传后端回退 config 增益）；
        纯力矩不设独立模式，经 MIT（``kp=kd=0``）实现。

        :raises RuntimeError: 未连接真机时抛出。
        :raises ValueError: 对应模式所需参数缺失时抛出。
        """
        if not self.connected:
            raise RuntimeError(f"[{self.name}] 未连接真机（离线）；请先 connect()。")
        if mode == ControlMode.POSITION:
            if q is None:
                raise ValueError("POSITION 模式需要 q")
            self.backend_arm.send_position(q)
        elif mode == ControlMode.VELOCITY:
            if dq is None:
                raise ValueError("VELOCITY 模式需要 dq")
            self.backend_arm.send_velocity(dq)
        elif mode == ControlMode.MIT:
            missing = [
                name for name, val in (("q", q), ("dq", dq), ("tau", tau)) if val is None
            ]
            if missing:
                raise ValueError(f"MIT 模式缺少参数：{missing}")
            self.backend_arm.send_mit(q, dq, tau, kp, kd)
        else:
            raise ValueError(f"未知控制模式：{mode}")

    # ----------------------------------------------------------
    # 末端执行类方法（依赖 backend_end；未连接 raise）
    # ----------------------------------------------------------
    def end_open(self) -> None:
        """张开末端到最大（默认行程/力度）。"""
        if not self.connected:
            raise RuntimeError(f"[{self.name}] 未连接真机（离线）；请先 connect()。")
        self.backend_end.send_action("open")

    def end_close(self) -> None:
        """闭合末端（夹到默认力度即停）。"""
        if not self.connected:
            raise RuntimeError(f"[{self.name}] 未连接真机（离线）；请先 connect()。")
        self.backend_end.send_action("close")

    def set_end_position(self, position: float) -> None:
        """末端位置控制（如两指间距 mm）。"""
        if not self.connected:
            raise RuntimeError(f"[{self.name}] 未连接真机（离线）；请先 connect()。")
        self.backend_end.send_position(position)

    def set_end_force(self, force: float) -> None:
        """末端力度控制（如夹持力 N）。"""
        if not self.connected:
            raise RuntimeError(f"[{self.name}] 未连接真机（离线）；请先 connect()。")
        self.backend_end.send_force(force)

    def get_end_state(self) -> dict:
        """读取末端状态（如夹爪 ``{"width_mm","force_N","is_grasping"}``）。

        :raises RuntimeError: 未连接真机时抛出。
        """
        if not self.connected:
            raise RuntimeError(f"[{self.name}] 未连接真机（离线）；请先 connect()。")
        return self.backend_end.read_state()
