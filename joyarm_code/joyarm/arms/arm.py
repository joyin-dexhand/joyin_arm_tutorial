"""``Arm`` 基类 —— 完整机械臂（多轴本体 + 末端执行器）。

``Arm`` 持有 pinocchio 模型 ``model`` + 数据 ``data`` + 限位，并直接持有两个通信后端：``backend_mas``（本体）/``backend_end``（末端）。
后端绑定（``backend_mas_cls`` / ``backend_end_cls``），构造时按 ``configs/*.yaml`` 的``backend_mas`` / ``backend_end`` 段实例化。

``connected=False``（默认，离线）时：
- 计算类方法（``fkine``/``jac``/...）不依赖真机，无硬件也能跑
- 执行类方法（``get_state``/``command``/``end_open``/...）``raise RuntimeError``；当``connect()`` 后才可用
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

from ..backends.backend_mas import BackendMas 
from ..backends.backend_end import BackendEnd 

__all__ = ["Arm"]


class Arm:
    """完整机械臂基类

    :param urdf_path: URDF 文件路径。
    :param ee_frame_name: 末端参考帧名（默认 ``"ee"``）。
    :param mesh_dirs: URDF 引用的 mesh 搜索目录列表；缺省时不加载几何。
    :param load_geometry: 是否加载 visual/collision 几何。
    :param name: 名称。
    :param config: 型号 YAML 配置字典（含 ``backend_mas`` / ``backend_end`` 等段）；缺省无后端。
    """

    # ---- 型号绑定，构造时按 config 实例化 ----
    backend_mas_cls: Optional[type] = None
    backend_end_cls: Optional[type] = None

    def __init__(self,
        urdf_path: str,
        ee_frame_name: str = "ee",
        mesh_dirs: Optional[List[str]] = None,
        load_geometry: bool = False,
        name: str = "Arm",
        config: Optional[dict] = None,
    ):
        if pin is None:
            raise ImportError(
                "Arm 需要 pinocchio 才能加载 URDF 与计算运动学。请安装：\n"
                "  conda install -c conda-forge pinocchio\n"
                "  # 或：pip install pin"
            )

        import os

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(
                f"未找到 URDF 文件：{urdf_path}\n"
                f"请将正式 URDF 放入 joyarm/robots/。"
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
        self.joint_limits: JointLimits = self._build_joint_limits(self.model)
        self.joint_limits_soft: JointLimits = self._build_soft_limits(
            self.joint_limits, margin=cfg.get("joint_limits_soft_margin", 0.05)
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
        self.backend_mas: Optional[BackendMas] = (
            self.backend_mas_cls(**cfg.get("backend_mas", {}))
            if self.backend_mas_cls is not None
            else None
        )
        self.backend_end: Optional[BackendEnd] = (
            self.backend_end_cls(**cfg.get("backend_end", {}))
            if self.backend_end_cls is not None
            else None
        )

    # ----------------------------------------------------------
    # 真机连接（connect 后才可执行 get_state/command/end_*）
    # ----------------------------------------------------------
    def connect(self) -> None:
        """连接本体 + 末端真机。"""
        if self.backend_mas is not None:
            self.backend_mas.connect()
        if self.backend_end is not None:
            self.backend_end.connect()
        self.connected = True

    def disconnect(self) -> None:
        """断开本体 + 末端真机。"""
        if self.backend_mas is not None:
            self.backend_mas.disconnect()
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
            temp_coil_max=np.zeros(n),
            temp_driver_max=np.zeros(n),
            voltage_min=np.zeros(n),
            voltage_max=np.zeros(n),
            current_max=np.zeros(n),
        )

    @staticmethod
    def _build_soft_limits(hard: JointLimits, margin: float = 0.05) -> JointLimits:
        """由硬限位内缩 ``margin`` 比例生成软限位。"""
        span = hard.q_max - hard.q_min
        qlow = hard.q_min + margin * span
        qhigh = hard.q_max - margin * span
        return JointLimits(
            q_min=qlow,
            q_max=qhigh,
            dq_max=hard.dq_max.copy(),
            tau_max=hard.tau_max.copy(),
            ddq_max=hard.ddq_max.copy(),
            temp_coil_max=hard.temp_coil_max.copy(),
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
    def rand_q(
        self,
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
        """将关节角裁剪到**软限位**内。"""
        return clamp_to_limits(q, self.joint_limits_soft)

    def is_q_valid(self, q: np.ndarray) -> bool:
        """关节角是否在**软限位**内（单点）。"""
        q = np.asarray(q, dtype=float).reshape(-1)
        return bool(np.all(q >= self.qlow - 1e-9) and np.all(q <= self.qhigh + 1e-9))

    # ----------------------------------------------------------
    # 正运动学（底层 + 门面委托；默认 pinocchio，手写请于子类覆盖）
    # ----------------------------------------------------------
    def frame_placement(self, q: np.ndarray, frame: Optional[Union[str, int]] = None) -> np.ndarray:
        """底层单次 FK：返回指定帧在基坐标系下的 ``(4,4)`` 位姿。"""
        fid = self._resolve_frame(frame)
        q_arr = np.asarray(q, dtype=float).reshape(self.n)
        pin.forwardKinematics(self.model, self.data, q_arr)
        pin.updateFramePlacement(self.model, self.data, fid)
        T = self.data.oMf[fid].homogeneous
        return self.T_base @ T

    def _resolve_frame(self, frame) -> int:
        """将帧名/索引/None 解析为 pinocchio frame id。"""
        if frame is None:
            return self.ee_frame_id
        if isinstance(frame, (int, np.integer)):
            return int(frame)
        for i, f in enumerate(self.model.frames):
            if f.name == frame:
                return i
        raise ValueError(f"找不到帧 '{frame}'")

    def fkine(self, q: np.ndarray, frame: Optional[Union[str, int]] = None, rep: str = "T"):
        """正运动学（薄委托 :func:`joyarm.robotics.fkine.fkine`，默认 pinocchio）。"""
        from ..robotics.fkine import fkine

        return fkine(self, q, frame=frame, rep=rep)

    # ----------------------------------------------------------
    # 占位门面委托（Ch3+ 实现；默认 pinocchio，手写请于子类覆盖）
    # ----------------------------------------------------------
    def ikine(self, T_target, q0=None, frame=None, **kw):
        """逆运动学（薄委托，Ch3 实现）。"""
        from ..robotics.ikine import ikine

        return ikine(self, T_target, q0=q0, frame=frame, **kw)

    def jac(self, q: np.ndarray, frame: Optional[Union[str, int]] = None, ref: str = "local"):
        """雅可比（薄委托，Ch4 实现）。"""
        from ..robotics.jacobian import jac

        return jac(self, q, frame=frame, ref=ref)

    def fdyn(self, q, dq, tau, **kw):
        """正动力学（薄委托，Ch8 实现）。"""
        from ..robotics.dyn import fdyn

        return fdyn(self, q, dq, tau, **kw)

    def idyn(self, q, dq, ddq, **kw):
        """逆动力学（薄委托，Ch8 实现）。"""
        from ..robotics.dyn import idyn

        return idyn(self, q, dq, ddq, **kw)

    def mass_matrix(self, q):
        """关节空间惯量矩阵 M(q)（薄委托，Ch8 实现）。"""
        from ..robotics.dyn import mass_matrix

        return mass_matrix(self, q)

    def coriolis(self, q, dq):
        """科氏+向心项 C(q,q̇)（薄委托，Ch8 实现）。"""
        from ..robotics.dyn import coriolis

        return coriolis(self, q, dq)

    def gravity(self, q):
        """重力项 G(q)（薄委托，Ch8 实现）。"""
        from ..robotics.dyn import gravity

        return gravity(self, q)

    def cartesian_inertia(self, q, frame=None):
        """笛卡尔惯量 Λ=J⁻ᵀMJ⁻¹（薄委托，Ch8/9 实现）。"""
        from ..robotics.dyn import cartesian_inertia

        return cartesian_inertia(self, q, frame=frame)

    # ----------------------------------------------------------
    # 安全校验（薄委托 safety；不依赖 backend，离线可用）
    # ----------------------------------------------------------
    def check_joint_limits(self, state):
        """关节层安全校验（薄委托，Ch11 实现）。"""
        from ..safety.safety import joint_limits_check

        return joint_limits_check(state, self.joint_limits)

    def check_tcp_limits(self, state):
        """末端层安全校验（薄委托，Ch11 实现）。"""
        from ..safety.safety import tcp_limits_check

        return tcp_limits_check(state, self.tcp_limits)

    # ----------------------------------------------------------
    # 本体执行类方法（依赖 backend_mas；未连接 raise）
    # ----------------------------------------------------------
    def get_state(self) -> ArmState:
        """读取本体状态快照（委托 ``backend_mas.read_state()``）。

        :raises RuntimeError: 未连接真机（``connected=False``）时抛出。
        """
        if not self.connected:
            raise RuntimeError(f"[{self.name}] 未连接真机（离线）；请先 connect()。")
        return self.backend_mas.read_state()

    def command(
        self,
        q: Optional[np.ndarray] = None,
        dq: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        mode: ControlMode = ControlMode.POSITION,
    ) -> None:
        """按控制模式下发运动指令（委托 ``backend_mas``）。

        :raises RuntimeError: 未连接真机时抛出。
        :raises ValueError: 对应模式所需参数缺失时抛出。
        """
        if not self.connected:
            raise RuntimeError(f"[{self.name}] 未连接真机（离线）；请先 connect()。")
        if mode == ControlMode.POSITION:
            if q is None:
                raise ValueError("POSITION 模式需要 q")
            self.backend_mas.send_position(q)
        elif mode == ControlMode.VELOCITY:
            if dq is None:
                raise ValueError("VELOCITY 模式需要 dq")
            self.backend_mas.send_velocity(dq)
        elif mode == ControlMode.TORQUE:
            if tau is None:
                raise ValueError("TORQUE 模式需要 tau")
            self.backend_mas.send_torque(tau)
        elif mode == ControlMode.MIT:
            missing = [
                name for name, val in (("q", q), ("dq", dq), ("tau", tau),
                                       ("kp", kp), ("kd", kd)) if val is None
            ]
            if missing:
                raise ValueError(f"MIT 模式缺少参数：{missing}")
            self.backend_mas.send_mit(q, dq, tau, kp, kd)
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
