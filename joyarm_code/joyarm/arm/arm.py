"""``Arm`` 基类 —— 组合根（运动学模型层）。
``Arm`` 持有 pinocchio 模型 ``model`` + 数据 ``data`` + 真机后端

``backend=None`` 时：
- 计算类方法不依赖 backend，无硬件也能跑
- 执行类方法``raise RuntimeError("离线模式不可执行")``

"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from ..utils.types import (
    ArmState,
    ControlMode,
    JointLimits,
    TcpLimits,
    clamp_to_limits,
)

try:  # pinocchio 为重依赖，惰性导入：缺失时仅 Arm 实例化报错
    import pinocchio as pin
except ImportError:  # pragma: no cover
    pin = None

# Backend 类型注解（实际定义在 backends/arm_backend.py）；放于 types 之后避免循环导入
from ..backends.arm_backend import ArmBackend  # noqa: E402  (type-only, no cycle: arm_backend 依赖 types)

__all__ = ["Arm"]


class Arm:
    """机械臂基类（组合根）。

    持有 pinocchio ``model`` + ``data`` + 可选 ``backend`` + 限位声明。
    子类（如 :class:`joyarm.arm.joyarm_rebot_dm.JoyArmRebotDM`）固化特有默认值
    （URDF 路径、末端帧、home 位形等）。

    :param urdf_path: URDF 文件路径。
    :param backend: 真机后端实例；``None`` = 离线模式（仅计算+可视化）。
    :param ee_frame_name: 末端参考帧名（默认 ``"ee"``）。
    :param mesh_dirs: URDF 引用的 mesh 搜索目录列表；缺省时不加载几何。
    :param load_geometry: 是否加载 visual/collision 几何（meshcat 可视化需要）。
    :param name: 机械臂名称。
    """

    def __init__(
        self,
        urdf_path: str,
        backend: Optional["ArmBackend"] = None,
        ee_frame_name: str = "ee",
        mesh_dirs: Optional[List[str]] = None,
        load_geometry: bool = False,
        name: str = "Arm",
    ):
        if pin is None:
            # pinocchio 是重依赖：只在实例化 Arm 时才强制要求（顶层 import joyarm 不需要）
            raise ImportError(
                "Arm 需要 pinocchio 才能加载 URDF 与计算运动学。请安装：\n"
                "  conda install -c conda-forge pinocchio\n"
                "  # 或：pip install pin\n"
                "（meshcat 可视化另需：pip install meshcat）"
            )

        import os

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(
                f"未找到 URDF 文件：{urdf_path}\n"
                f"请将正式 URDF 放入 joyarm/robots/ "
                f"（见 架构设计.md §十七；第七章导出正式版）。"
            )

        # ---- 构建 pinocchio 模型 ----
        # pin.buildModelFromUrdf 返回单个 Model（非元组）；package_dirs 用于解析 mesh 路径
        if mesh_dirs:
            self.model = pin.buildModelFromUrdf(urdf_path, package_dirs=mesh_dirs)
        else:
            self.model = pin.buildModelFromUrdf(urdf_path)

        # 几何（visual/collision）加载：meshcat 可视化需要。
        # 用 buildGeomFromUrdf 单独加载（buildModelFromUrdf 不含几何）。
        # 当前为可选能力，加载失败时仅记录、不阻断运动学计算。
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
                # 几何缺失不应阻断运动学/限位等核心功能
                import warnings

                warnings.warn(f"几何模型加载失败（meshcat 可视化将不可用）：{e}")
        self.data = self.model.createData()

        # ---- 基本属性 ----
        self.n: int = self.model.nq  # 自由度（配置维度；对纯转动关节 == nv）
        self.nv: int = self.model.nv
        self.name: str = name
        self.backend: Optional["ArmBackend"] = backend

        # ---- 末端帧 ----
        self.ee_frame_name: str = ee_frame_name
        try:
            # 用帧名查 pinocchio 的帧 ID（后续 FK 用 ID 取位姿，比每次按名查找快）
            self.ee_frame_id: int = self.model.getFrameId(ee_frame_name)
        except Exception:
            # pinocchio 旧版本无 getFrameId 时回退遍历
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
            self.joint_limits, margin=0.05
        )
        # 便捷别名（软限位位置上下限）
        self.qlow: np.ndarray = self.joint_limits_soft.q_min
        self.qhigh: np.ndarray = self.joint_limits_soft.q_max

        # ---- 中性位形（pinocchio neutral，非 home）----
        self.q_neutral: np.ndarray = np.clip(
            pin.neutral(self.model),
            self.joint_limits.q_min,
            self.joint_limits.q_max,
        )

        # ---- 末端限位（Ch11 才正式填；此处占位）----
        self.tcp_limits: TcpLimits = TcpLimits()

        # ---- 基坐标系偏移（默认单位阵）----
        self.T_base: np.ndarray = np.eye(4)

    # ----------------------------------------------------------
    # 内部：限位构建
    # ----------------------------------------------------------
    @staticmethod
    def _build_joint_limits(model) -> JointLimits:
        """从 pinocchio model 解析硬限位。

        - 位置/速度/力矩：来自 URDF（``lowerPositionLimit`` 等）。
        - 加速度/温度/电压/电流：URDF 不含，置零（Ch11 从电机规格填）。
        """
        n = model.nq
        # 位置限位直接来自 URDF 的 <limit lower/upper>，维度 = nq
        q_min = np.asarray(model.lowerPositionLimit, dtype=float).reshape(n)
        q_max = np.asarray(model.upperPositionLimit, dtype=float).reshape(n)
        # 速度/力矩上限：pinocchio 维度 = nv
        nv = model.nv
        # hasattr 兜底：某些模型未声明 velocity/effort 时给无限大
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
        # 对齐到 nq 维度（绝大多数纯转动关节 nq == nv；含移动关节时会不等）
        if nv != n:
            dq_max = np.full(n, np.inf)
            tau_max = np.full(n, np.inf)
        return JointLimits(
            q_min=q_min,
            q_max=q_max,
            dq_max=dq_max,
            tau_max=tau_max,
            ddq_max=np.full(n, np.inf),  # Ch11 填
            temp_coil_max=np.zeros(n),  # Ch11 填
            temp_driver_max=np.zeros(n),  # Ch11 填
            voltage_min=np.zeros(n),  # Ch11 填
            voltage_max=np.zeros(n),  # Ch11 填
            current_max=np.zeros(n),  # Ch11 填
        )

    @staticmethod
    def _build_soft_limits(hard: JointLimits, margin: float = 0.05) -> JointLimits:
        """由硬限位内缩 ``margin`` 比例生成软限位。"""
        # 软限位 = 硬限位向内缩 margin 比例（如 5%），给控制系统留缓冲、避免撞硬限位
        span = hard.q_max - hard.q_min
        qlow = hard.q_min + margin * span
        qhigh = hard.q_max - margin * span
        # 软限位其余字段同硬限位
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
    # 表示
    # ----------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, n={self.n}, "
            f"ee_frame={self.ee_frame_name!r}, "
            f"backend={'real' if self.backend is not None else 'offline'})"
        )

    # ----------------------------------------------------------
    # 关节角采样 / 裁剪 / 校验（基于软限位）
    # ----------------------------------------------------------
    def rand_q(
        self,
        size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """在**软限位**内均匀采样关节角。

        :param size: ``None`` → ``(n,)`` 单个采样；``int`` → ``(size, n)`` 批量。
        :param rng: ``numpy.random.Generator``；缺省新建默认生成器。
        :return: 采样关节角，弧度。
        """
        rng = rng if rng is not None else np.random.default_rng()
        low, high = self.qlow, self.qhigh          # 软限位区间（留了安全余量）
        if size is None:                           # 单个采样
            return rng.uniform(low, high)
        return rng.uniform(low, high, size=(size, self.n))   # 批量采样

    def clamp_q(self, q: np.ndarray) -> np.ndarray:
        """将关节角裁剪到**软限位**内。

        :param q: ``(n,)`` 或 ``(N,n)``。
        :return: 同形状裁剪结果。
        """
        return clamp_to_limits(q, self.joint_limits_soft)

    def is_q_valid(self, q: np.ndarray) -> bool:
        """关节角是否在**软限位**内（单点）。

        :param q: ``(n,)``。
        :return: 全部分量在 ``[qlow, qhigh]`` 内则为 ``True``。
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        # 1e-9 容差：允许浮点边界误差，避免"刚好等于限位却被判非法"
        return bool(
            np.all(q >= self.qlow - 1e-9) and np.all(q <= self.qhigh + 1e-9)
        )

    # ----------------------------------------------------------
    # 正运动学（底层 + 薄委托）
    # ----------------------------------------------------------
    def frame_placement(
        self, q: np.ndarray, frame: Optional[Union[str, int]] = None
    ) -> np.ndarray:
        """底层单次 FK：返回指定帧在基坐标系下的 ``(4,4)`` 位姿。

        :param q: ``(n,)`` 关节角，弧度。
        :param frame: 帧名（``str``）或帧索引（``int``）；缺省为末端帧。
        :return: ``(4,4)`` 齐次变换矩阵（含 ``T_base`` 基坐标系偏移）。
        """
        fid = self._resolve_frame(frame)             # 帧 ID（末端默认）
        q_arr = np.asarray(q, dtype=float).reshape(self.n)   # 整理成 (n,) 向量
        # pinocchio 正运动学两步走：先算所有关节，再更新指定帧的位姿
        pin.forwardKinematics(self.model, self.data, q_arr)
        pin.updateFramePlacement(self.model, self.data, fid)
        T = self.data.oMf[fid].homogeneous  # pinocchio 返回的 (4,4)
        # 应用基坐标系偏移（默认单位阵，透明）
        return self.T_base @ T

    def _resolve_frame(self, frame) -> int:
        """将帧名/索引/None 解析为 pinocchio frame id。"""
        if frame is None:
            return self.ee_frame_id
        if isinstance(frame, (int, np.integer)):
            return int(frame)
        # str：按名查找
        for i, f in enumerate(self.model.frames):
            if f.name == frame:
                return i
        raise ValueError(f"找不到帧 '{frame}'")

    def fkine(
        self,
        q: np.ndarray,
        frame: Optional[Union[str, int]] = None,
        rep: str = "T",
        method: str = "auto",
    ):
        """正运动学（薄委托到 :func:`joyarm.robotics.fkine.fkine`）。"""
        # 局部 import 避免顶层循环依赖
        from ..robotics.fkine import fkine

        return fkine(self, q, frame=frame, rep=rep, method=method)

    # ----------------------------------------------------------
    # 占位薄委托（Ch3+ 实现；调用时触发 NotImplementedError）
    # ----------------------------------------------------------
    def ikine(self, T_target, q0=None, frame=None, method: str = "auto", **kw):
        """逆运动学（薄委托，Ch3 实现）。"""
        from ..robotics.ikine import ikine

        return ikine(self, T_target, q0=q0, frame=frame, method=method, **kw)

    def jac(
        self,
        q: np.ndarray,
        frame: Optional[Union[str, int]] = None,
        ref: str = "local",
        method: str = "auto",
    ):
        """雅可比（薄委托，Ch4 实现）。"""
        from ..robotics.jacobian import jac

        return jac(self, q, frame=frame, ref=ref, method=method)

    def fdyn(self, q, dq, tau, method: str = "auto", **kw):
        """正动力学（薄委托，Ch8 实现）。"""
        from ..robotics.dyn import fdyn

        return fdyn(self, q, dq, tau, method=method, **kw)

    def idyn(self, q, dq, ddq, method: str = "auto", **kw):
        """逆动力学（薄委托，Ch8 实现）。"""
        from ..robotics.dyn import idyn

        return idyn(self, q, dq, ddq, method=method, **kw)

    # ----------------------------------------------------------
    # 执行类方法（依赖 backend；离线模式 raise）
    # ----------------------------------------------------------
    def get_state(self) -> ArmState:
        """读取整机状态快照（委托 ``backend.read_state()``）。

        :raises RuntimeError: ``backend=None``（离线模式）时抛出。
        """
        if self.backend is None:
            raise RuntimeError(
                f"[{self.name}] 离线模式不可执行 get_state()："
                f"未挂载真机后端（backend=None）。"
            )
        return self.backend.read_state()

    def command(
        self,
        q: Optional[np.ndarray] = None,
        dq: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        mode: ControlMode = ControlMode.POSITION,
    ) -> None:
        """按控制模式下发运动指令（委托 ``backend``）。

        - :attr:`ControlMode.POSITION`：用 ``q``。
        - :attr:`ControlMode.VELOCITY`：用 ``dq``。
        - :attr:`ControlMode.TORQUE`：用 ``tau``。
        - :attr:`ControlMode.MIT`：用 ``(q, dq, tau, kp, kd)``，
          满足 ``τ = kp·(q_des−q) + kd·(dq_des−dq) + tau_ff``。

        :raises RuntimeError: ``backend=None``（离线模式）时抛出。
        :raises ValueError: 对应模式所需参数缺失时抛出。
        """
        if self.backend is None:
            raise RuntimeError(
                f"[{self.name}] 离线模式不可执行 command()："
                f"未挂载真机后端（backend=None）。"
            )
        # 四种控制模式各下发不同的物理量，分别调用 backend 的不同方法
        if mode == ControlMode.POSITION:            # 位置模式：直接告诉电机转到哪个角度
            if q is None:
                raise ValueError("POSITION 模式需要 q")
            self.backend.send_position(q)
        elif mode == ControlMode.VELOCITY:          # 速度模式：告诉电机以多快速度转
            if dq is None:
                raise ValueError("VELOCITY 模式需要 dq")
            self.backend.send_velocity(dq)
        elif mode == ControlMode.TORQUE:            # 力矩模式：直接输出多大力
            if tau is None:
                raise ValueError("TORQUE 模式需要 tau")
            self.backend.send_torque(tau)
        elif mode == ControlMode.MIT:               # MIT 阻抗模式：同时给目标+增益，电机自己做 PD
            # MIT 模式需要全部 5 个参数（q,dq 为目标，tau 为前馈，kp/kd 为刚度/阻尼）
            missing = [
                name
                for name, val in (("q", q), ("dq", dq), ("tau", tau),
                                  ("kp", kp), ("kd", kd))
                if val is None
            ]
            if missing:
                raise ValueError(f"MIT 模式缺少参数：{missing}")
            self.backend.send_mit(q, dq, tau, kp, kd)
        else:
            raise ValueError(f"未知控制模式：{mode}")
