"""``FakeArm`` —— JoyArm 的无硬件最小替身（"arm 协议"的鸭子类型参照实现）。

**FakeArm 是什么**

与 :class:`~joyarm_core.joyarm.JoyArm` **无继承关系**的独立小类：不创建
通信后端、不起任何线程、不接触硬件（仅构造期读 config 与 URDF），只在内存里
维护一份**理想执行器**关节状态——位置指令即时到位、无延迟、无动力学。
它与 JoyArm 遵守同一套属性/方法约定（下称"arm 协议"：``pin_model`` /
``arm_limits`` / ``arm_home`` / ``get_arm_state()`` / ``get_target_traj()``
等；robotics 六域算法只按这套约定消费 ``arm``，不查类型）。因此**凡按协议
消费 ``arm`` 的代码，拿到 FakeArm 与拿到 JoyArm 同样能跑**。

**有什么用**

1. robotics 算法测试验证：无硬件、无后端即可对 fkine / ikine / jacobian /
   dynamics / trajectory / control 六域做测试与回归（test/ 套件的标准化
   替身，免去每个测试手写哑对象）；
2. 教程各章最小示例：chapt 脚本 ``from joyarm_core import FakeArm`` 一行
   即得一台"可算可动"的机械臂，无真机的读者也能完整复现；
3. 无硬件环境运行：CI、演示与教学环境的统一载体；
4. 快速原型与实验：REPL / notebook 即开即用，验证想法不用接设备；
5. 教学参照：作为"arm 协议"的最小范本，演示鸭子类型 + 策略模式的接口契约
   （robotics 求解器第一参数都是 ``arm``，本类即该参数的最小合法实现）。

**与 JoyArm 的差异（最小化裁剪；需要硬件语义时请用 JoyArm）**

- 无通信后端、无线程、无运动管线：``start_motion`` / ``stop_motion`` /
  ``set_solver`` / ``teach_*`` / ``move_l`` / ``safe_*`` / ``is_in_position`` /
  电机参数读写等未实现（调用了会 ``AttributeError``）；
- 求解器固定为四个默认 Pin 实现（fkine / ikine / jacobian / dynamics
  门面可用），不支持运行期切换；
- 指令语义为理想执行器：位置 / MIT（``kp`` 非零）指令即时把关节状态置为
  裁剪后的目标并清零速度；速度与纯力矩指令只更新状态数组，不做时间积分；
- :meth:`FakeArm.move_j` 同步逐帧更新内部状态后**返回 ``(ts, q, dq)`` 轨迹
  数组**（JoyArm 返回 ``None``），默认不睡眠（结果确定、利于测试），
  ``realtime=True`` 时按时间戳实时回放；
- :meth:`FakeArm.get_arm_state` 不要求先 :meth:`~FakeArm.connect`（算法
  测试随时可读状态）。

快速上手::

    from joyarm_core import FakeArm

    arm = FakeArm()                                # 默认型号 joyarm_dm，初始停在 home
    pose = arm.fkine(arm.arm_home, "link_end")     # 数学门面：不连"机"也能算
    arm.connect()                                  # 生命周期标志（无实际 IO）
    arm.enable_arm()
    ts, qs, dqs = arm.move_j(arm.rand_q_arm())     # 理想插值运动，返回轨迹数组
"""
from __future__ import annotations

import copy
import logging
import time
from typing import List, Optional, Union

import numpy as np
import pinocchio as pin

from ..robotics import (
    PinDynamicsSolver,
    PinFkineSolver,
    PinIkineSolver,
    PinJacobianSolver,
)
from ..utils.limits import (
    clamp_to_limits,
    limits_from_joint_cfgs,
    rand_within_limits,
    soft_limits_from_cfg,
)
from ..utils.types import (
    ArmState,
    ControlMode,
    JointState,
    Pose,
    TcpLimits,
    TcpState,
    TrajFrame,
)

from .joyarm import _ARM_JOINT_NAMES, _cubic_traj, JoyArm, load_config

__all__ = ["FakeArm"]

logger = logging.getLogger("joyarm_core.joyarm.fakearm")


def _clip_abs(values, max_abs) -> np.ndarray:
    """对称幅值裁剪（速度/力矩指令用）：逐元素夹到 ±max_abs。

    :param values: 待裁剪数组；:param max_abs: 幅值上限数组；``None`` 原样返回。
    """
    v = np.asarray(values, dtype=float)
    if max_abs is None:
        return v
    return np.clip(v, -np.asarray(max_abs, dtype=float), np.asarray(max_abs, dtype=float))


class FakeArm:
    """JoyArm 的无硬件最小替身（"arm 协议"的鸭子类型实现，详见模块 docstring）。

    :param model: 型号名（= ``configs/<model>.yaml``，默认 ``"joyarm_dm"``）。
    :param config: 型号配置字典；不传就自动加载（与 JoyArm 同一解析路径）。
    :param q0: 初始关节角 ``(n_arm,)``；缺省停在 home 位形。

    公开接口分组（对齐 JoyArm 的常用子集）：连接与上下文 → 使能与模式 →
    读状态 → 下发指令 → 末端 → 运动 → 目标轨迹 → 数学计算 → 基本信息。
    """

    # ============================================================
    # 内部实现区：初始化（读配置 → 解析 URDF → 核对关节 → 限位/特征位形 →
    # 内部仿真状态 → 固定默认求解器）+ 私有助手
    # ============================================================
    def __init__(self, model: str = "joyarm_dm", config: Optional[dict] = None,
                 q0: Optional[np.ndarray] = None):
        # ---- 读取并自检 config（复用 JoyArm 的加载与校验，出错直接抛）----
        if config is None:
            config = load_config(model, strict=True)
        cfg = copy.deepcopy(config)
        JoyArm.check_config(model, cfg)

        basic = cfg.get("basic") or {}
        jcfg = cfg.get("joyarm") or {}
        backend_cfg = cfg.get("backend") or {}
        arm_joint_cfgs = (backend_cfg.get("arm") or {}).get("joints") or []
        end_joint_cfgs = (backend_cfg.get("end") or {}).get("joints") or []

        # ---- 基本信息 ----
        self.model: str = model
        self._config: dict = cfg
        self.connected: bool = False        # 连接标志（connect 只置标志，无 IO）
        self.ee_frame_name: str = str(basic.get("ee_frame") or "ee")
        self.is_normal: bool = True

        # ---- 运动学模型（pinocchio，与 JoyArm 同一 URDF 解析路径）----
        self._urdf_path: str = JoyArm._resolve_robot_urdf(str(basic["robot"]))
        self.pin_model: pin.Model = JoyArm._build_pin_model(self._urdf_path)
        self.pin_data: pin.Data = self.pin_model.createData()
        self.ee_frame_id: int = self.pin_model.getFrameId(self.ee_frame_name)
        if self.ee_frame_id >= len(self.pin_model.frames):
            raise ValueError(
                f"fakearm.py - FakeArm.__init__：URDF 中找不到末端帧 '{self.ee_frame_name}'；"
                f"可用帧：{[f.name for f in self.pin_model.frames]}")

        # ---- 关节数、关节名、与 URDF 核对（q 向量顺序 = config 顺序）----
        self._arm_joint_names: List[str] = [
            str(j.get("name", f"joint{i + 1}")) for i, j in enumerate(arm_joint_cfgs)
        ]
        self._end_joint_names: List[str] = [
            str(j.get("name", f"end_joint{i + 1}")) for i, j in enumerate(end_joint_cfgs)
        ]
        self.n_arm: int = len(self._arm_joint_names)
        self.n_end: int = len(self._end_joint_names)
        if self.n_arm != JoyArm._urdf_arm_nq(self.pin_model):
            raise ValueError(
                f"fakearm.py - FakeArm.__init__：backend.arm.joints 数量 {self.n_arm} "
                f"≠ URDF arm 关节数 {JoyArm._urdf_arm_nq(self.pin_model)}")
        pin_arm_order = [self.pin_model.names[i]
                         for i in range(1, len(self.pin_model.names))
                         if self.pin_model.names[i] in _ARM_JOINT_NAMES]
        if self._arm_joint_names != pin_arm_order:
            raise ValueError(
                f"fakearm.py - FakeArm.__init__：backend.arm.joints 顺序与 URDF 关节"
                f"顺序不一致\n  config：{self._arm_joint_names}\n  URDF  ：{pin_arm_order}")

        # ---- 限位（硬限位来自 config backend.*.joints，软限位直配；同 JoyArm）----
        self.arm_limits = limits_from_joint_cfgs(arm_joint_cfgs)
        self.end_limits = limits_from_joint_cfgs(end_joint_cfgs)
        self.arm_limits_soft = (
            soft_limits_from_cfg(jcfg["arm_soft_limits"], self.n_arm)
            if jcfg.get("arm_soft_limits") is not None
            else copy.deepcopy(self.arm_limits))
        self.end_limits_soft = (
            soft_limits_from_cfg(jcfg["end_soft_limits"], self.n_end)
            if (self.end_limits is not None and jcfg.get("end_soft_limits") is not None)
            else copy.deepcopy(self.end_limits))

        # ---- 特征位形（zero/neutral 恒全零；home 来自 config，越限裁剪告警）----

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

        # ---- 末端空间限位（config joyarm.tcp_limits 配了就加载）----
        self.tcp_limits: TcpLimits = TcpLimits()
        if jcfg.get("tcp_limits"):
            self._apply_tcp_limits(jcfg["tcp_limits"])

        # ---- 内部仿真状态：理想执行器（指令即时生效，无延迟无动力学）----
        self._mode: ControlMode = ControlMode.POSITION     # 本体模式（全关节共用）
        self._end_mode: ControlMode = ControlMode.POSITION
        self._enabled = np.zeros(self.n_arm, dtype=bool)   # 使能标志（先使能才能 move_j）
        self._end_enabled = np.zeros(self.n_end, dtype=bool)
        self._q = self.arm_home.copy() if q0 is None \
            else _clip_pose(q0, self.arm_limits, "q0(初始关节角)")
        self._dq = np.zeros(self.n_arm)
        self._tau = np.zeros(self.n_arm)
        self._end_q = self._end_home.copy()
        self._end_dq = np.zeros(self.n_end)
        self._end_tau = np.zeros(self.n_end)
        self._target_traj: Optional[List[TrajFrame]] = None

        # ---- 固定默认求解器（四个 Pin 默认实现；不支持运行期切换）----
        self._solvers: dict = {
            "fkine": PinFkineSolver(),
            "ikine": PinIkineSolver(),
            "jacobian": PinJacobianSolver(),
            "dynamics": PinDynamicsSolver(),
        }

    # === 初始化助手（内部）===
    def _apply_tcp_limits(self, tl: dict) -> None:
        """解析 config ``tcp_limits`` 段（``workspace_box`` 兼容 (2,3)/(3,2)）。"""
        box = np.asarray(tl.get("workspace_box",
                                [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.8]]), dtype=float)
        if box.shape == (2, 3):
            box = box.T
        self.tcp_limits = TcpLimits(
            workspace_box=box,
            v_lin_max=float(tl.get("v_lin_max", 0.0)),
            v_ang_max=float(tl.get("v_ang_max", 0.0)),
            f_max=float(tl.get("f_max", 0.0)),
            t_max=float(tl.get("t_max", 0.0)),
        )

    # === 前置校验（内部）===
    def _require_connected(self) -> None:
        """没 connect 就抛 ``RuntimeError``（FakeArm 只查标志，connect 无实际 IO）。"""
        if not self.connected:
            raise RuntimeError(
                f"fakearm.py - _require_connected：[{self.model}] 尚未 connect()；"
                f"FakeArm 无硬件，connect() 仅置连接标志。")

    def _require_enabled_arm(self) -> None:
        """本体未全部使能就抛 ``RuntimeError``（move_j 等运动前用）。"""
        if not self._enabled.all():
            raise RuntimeError(
                f"fakearm.py - _require_enabled_arm：本体电机未全部使能；请先 enable_arm()")

    def _require_mode_arm(self, mode: ControlMode) -> None:
        """本体当前控制模式 ≠ ``mode`` 就抛 ``RuntimeError``。"""
        if self._mode != mode:
            raise RuntimeError(
                f"fakearm.py - _require_mode_arm：本体当前控制模式为 {self._mode}，"
                f"与指令模式 {mode} 不符；请先 set_mode_arm({mode})")

    def _require_end(self, what: str) -> None:
        """无末端电机（n_end=0）时抛 ``RuntimeError``。"""
        if self.n_end == 0:
            raise RuntimeError(f"fakearm.py - {what}：本型号无末端电机（n_end=0）")

    def _joint_indices(self, joint: Optional[int], n: int, what: str) -> np.ndarray:
        """``joint=None`` → 全部索引；否则校验范围后返回单元素索引数组。"""
        if joint is None:
            return np.arange(n)
        if not 0 <= joint < n:
            raise ValueError(
                f"fakearm.py - {what}：joint 索引 {joint} 越界（共 {n} 个）")
        return np.array([joint])

    def _merge_scalar(self, value, joint: int, base: np.ndarray, what: str):
        """单关节控制的标量参数 → 全关节向量（该关节给新值，其余保持 ``base``）。

        ``None`` 透传（该参数未给）；非标量形状抛 ``ValueError``。
        """
        if value is None:
            return None
        v = np.asarray(value, dtype=float).reshape(-1)
        if v.size != 1:
            raise ValueError(
                f"fakearm.py - {what}：指定 joint={joint} 时参数须为标量，"
                f"实际形状 {v.shape}（全关节向量请用 joint=None）")
        out = np.asarray(base, dtype=float).copy()
        out[joint] = v[0]
        return out

    def _apply_position(self, q) -> None:
        """位置指令生效：裁硬限位（越限告警）后即时置位、速度清零。"""
        q = np.asarray(q, dtype=float).reshape(-1)
        q_c = clamp_to_limits(q, self.arm_limits)
        if not np.array_equal(q, q_c):
            logger.warning("FakeArm 位置指令越硬限位，已裁剪 %s → %s（以裁剪后为准）",
                           np.round(q, 4).tolist(), np.round(q_c, 4).tolist())
        self._q = q_c
        self._dq = np.zeros(self.n_arm)

    def _solve(self, domain: str, method: str, *args, **kw):
        """统一求解入口：转发给固定默认求解器（JoyArm 的精简版，无切换）。"""
        return getattr(self._solvers[domain], method)(*args, **kw)

    # ============================================================
    # 公开接口区
    # ============================================================

    # ===
    # 一、连接与上下文：connect 只置标志（教学时保持与 JoyArm 相同的调用顺序）
    # ===
    def connect(self) -> None:
        """置连接标志（无任何 IO；语义上"接入"了这台理想机械臂）。"""
        self.connected = True

    def disconnect(self) -> None:
        """清除连接标志。"""
        self.connected = False

    def __enter__(self) -> "FakeArm":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    # ===
    # 二、使能与控制模式：标志位仿真（全关节共用一个模式，无逐关节差异）
    # ===
    def enable_arm(self, joint: Optional[int] = None) -> None:
        """置本体使能标志（``joint=None`` 全部）。"""
        self._enabled[self._joint_indices(joint, self.n_arm, "enable_arm")] = True

    def disable_arm(self, joint: Optional[int] = None) -> None:
        """清除本体使能标志（``joint=None`` 全部）。"""
        self._enabled[self._joint_indices(joint, self.n_arm, "disable_arm")] = False

    def enable_end(self, joint: Optional[int] = None) -> None:
        """置末端使能标志（``joint=None`` 全部）。"""
        self._require_end("enable_end")
        self._end_enabled[self._joint_indices(joint, self.n_end, "enable_end")] = True

    def disable_end(self, joint: Optional[int] = None) -> None:
        """清除末端使能标志（``joint=None`` 全部）。"""
        self._require_end("disable_end")
        self._end_enabled[self._joint_indices(joint, self.n_end, "disable_end")] = False

    def set_mode_arm(self, mode: ControlMode = ControlMode.POSITION,
                     joint: Optional[int] = None) -> None:
        """切换本体控制模式（全关节共用；``joint`` 形参仅为对齐 JoyArm 签名）。"""
        self._mode = mode

    def read_mode_arm(self, joint: Optional[int] = None) -> ControlMode:
        """读本体当前控制模式（全关节共用，恒一致）。"""
        return self._mode

    def set_mode_end(self, mode: ControlMode = ControlMode.POSITION,
                     joint: Optional[int] = None) -> None:
        """切换末端控制模式（全电机共用；``joint`` 形参仅为对齐 JoyArm 签名）。"""
        self._end_mode = mode

    def read_mode_end(self, joint: Optional[int] = None) -> ControlMode:
        """读末端当前控制模式。"""
        return self._end_mode

    # ===
    # 三、读状态：快照来自内部仿真状态（不要求 connect，算法测试随时可读）
    # ===
    def get_arm_state(self) -> ArmState:
        """本体当前状态快照（关节 q/dq/tau、使能标志，恒"通讯正常/无故障"）；
        末端位姿 ``tcp.pose`` 用内置 fkine 现算填充。"""
        n = self.n_arm
        return ArmState(
            joint=JointState(
                control_mode=self._mode,
                q=self._q.copy(), dq=self._dq.copy(), tau=self._tau.copy(),
                enabled=self._enabled.copy(),
                error=np.zeros(n, dtype=bool),
                comm_ok=np.ones(n, dtype=bool),
                angle_ok=np.ones(n, dtype=bool),
                temp_mos=np.zeros(n), temp_rotor=np.zeros(n),
            ),
            tcp=TcpState(pose=self.fkine(self._q, self.ee_frame_name)),
            mode=self._mode,
            timestamp=time.time(),
            errors=[],
        )

    def get_end_state(self, joint: Optional[int] = None) -> dict:
        """末端状态快照（字典形式，键 ``q/dq/tau/enabled/error/comm_ok``）。

        :param joint: 末端电机索引，``None`` 表示全部。
        :raises RuntimeError: 本型号无末端电机。
        """
        self._require_end("get_end_state")
        idx = self._joint_indices(joint, self.n_end, "get_end_state")
        return {
            "q": self._end_q[idx].copy(),
            "dq": self._end_dq[idx].copy(),
            "tau": self._end_tau[idx].copy(),
            "enabled": self._end_enabled[idx].copy(),
            "error": np.zeros(idx.size, dtype=bool),
            "comm_ok": np.ones(idx.size, dtype=bool),
        }

    # ===
    # 四、手动下发指令（理想执行器语义，见模块 docstring"差异"节）
    # ===
    def set_arm_command(self,
        mode: ControlMode = ControlMode.POSITION,
        q: Optional[np.ndarray] = None,
        dq: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        """手动下发一条运动指令（参数语义同 JoyArm；生效规则为理想执行器）。

        - POSITION：``q`` 即时置位（裁硬限位）；
        - VELOCITY：``dq`` 记入状态（裁 ``dq_max``），位置不动、不做积分；
        - MIT：``q/dq/tau`` 必填；``kp`` 非零时 ``q`` 按位置目标生效，
          否则（纯力矩/前馈）只记录 ``dq/tau``；
        - ``joint`` 指定即单关节控制（该关节参数须为标量，其余关节保持现值）。

        :raises RuntimeError: 未 connect / 当前模式与 ``mode`` 不符。
        :raises ValueError: 该模式需要的参数没给全 / 形状不符。
        """
        self._require_connected()
        self._require_mode_arm(mode)
        if joint is not None:
            self._joint_indices(joint, self.n_arm, "set_arm_command")   # 校验范围
            q = self._merge_scalar(q, joint, self._q, "set_arm_command")
            dq = self._merge_scalar(dq, joint, self._dq, "set_arm_command")
            tau = self._merge_scalar(tau, joint, self._tau, "set_arm_command")
            kp = self._merge_scalar(kp, joint, np.zeros(self.n_arm), "set_arm_command")
        if mode == ControlMode.POSITION:
            if q is None:
                raise ValueError("fakearm.py - set_arm_command：POSITION 模式需要 q")
            self._apply_position(q)
        elif mode == ControlMode.VELOCITY:
            if dq is None:
                raise ValueError("fakearm.py - set_arm_command：VELOCITY 模式需要 dq")
            self._dq = _clip_abs(dq, self.arm_limits.dq_max if self.arm_limits else None)
        elif mode == ControlMode.MIT:
            missing = [name for name, val in (("q", q), ("dq", dq), ("tau", tau))
                       if val is None]
            if missing:
                raise ValueError(f"fakearm.py - set_arm_command：MIT 模式缺少参数：{missing}")
            self._tau = _clip_abs(tau, self.arm_limits.tau_max if self.arm_limits else None)
            self._dq = _clip_abs(dq, self.arm_limits.dq_max if self.arm_limits else None)
            if kp is not None and np.any(np.asarray(kp, dtype=float) > 0):
                self._apply_position(q)     # 有位置增益 → 视作位置目标
        else:
            raise ValueError(f"fakearm.py - set_arm_command：未知控制模式：{mode}")

    # ===
    # 五、末端执行器（夹爪）：位置即时生效、力矩只记录（同本体语义）
    # ===
    def _apply_end_position(self, target, joint: Optional[int], what: str) -> None:
        """末端位置指令 → 全电机目标向量 → 裁行程后即时置位。"""
        self._require_connected()
        self._require_end(what)
        t = np.asarray(target, dtype=float).reshape(-1)
        if joint is None and t.size == 1:
            t = np.full(self.n_end, t[0])
        elif joint is not None:
            t = self._merge_scalar(t, joint, self._end_q, what)
        if t.size != self.n_end:
            raise ValueError(
                f"fakearm.py - {what}：参数长度 {t.size} 与末端电机数 {self.n_end} 不符")
        t_c = clamp_to_limits(t, self.end_limits)
        if not np.array_equal(t, t_c):
            logger.warning("FakeArm 末端指令越行程，已裁剪 %s → %s",
                           np.round(t, 4).tolist(), np.round(t_c, 4).tolist())
        self._end_q = t_c
        self._end_dq = np.zeros(self.n_end)

    def set_end_open(self, joint: Optional[int] = None) -> None:
        """张开末端到行程下限（= ``q_min``，夹爪张开位）。"""
        lim = self.end_limits
        self._apply_end_position(lim.q_min if joint is None else lim.q_min[joint],
                                 joint, "set_end_open")

    def set_end_close(self, joint: Optional[int] = None) -> None:
        """闭合末端到行程上限（= ``q_max``，夹爪闭合位）。"""
        lim = self.end_limits
        self._apply_end_position(lim.q_max if joint is None else lim.q_max[joint],
                                 joint, "set_end_close")

    def set_end_zero(self, joint: Optional[int] = None) -> None:
        """末端归零（目标 = 电机 0 弧度，越行程自动裁剪）。"""
        self._apply_end_position(0.0, joint, "set_end_zero")

    def set_end_position(self, position, joint: Optional[int] = None) -> None:
        """末端位置控制（连续量，如夹爪电机弧度）。"""
        self._apply_end_position(position, joint, "set_end_position")

    def set_end_tau(self, tau, joint: Optional[int] = None) -> None:
        """末端力矩控制（只记录，裁 ``tau_max``；不产生运动）。"""
        self._require_connected()
        self._require_end("set_end_tau")
        t = np.asarray(tau, dtype=float).reshape(-1)
        if joint is None and t.size == 1:
            t = np.full(self.n_end, t[0])
        elif joint is not None:
            t = self._merge_scalar(t, joint, self._end_tau, "set_end_tau")
        if t.size != self.n_end:
            raise ValueError(
                f"fakearm.py - set_end_tau：参数长度 {t.size} 与末端电机数 {self.n_end} 不符")
        self._end_tau = _clip_abs(t, self.end_limits.tau_max if self.end_limits else None)

    # ===
    # 六、运动：move_j 理想插值（同步逐帧更新状态，返回轨迹数组）
    # ===
    def move_j(self, q, t=None, *, rate=None, realtime: bool = False):
        """关节运动到目标角（理想执行器：同步逐帧把状态铺到插值轨迹上）。

        与 JoyArm 的差异：不阻塞等到位（状态即时跟随，必到）、**返回
        ``(ts, qs, dqs)``**（时间戳 ``(N,)``、关节角/速度 ``(N, n)``，可直接
        供教程绘图）；默认不睡眠（结果确定），``realtime=True`` 按时间戳
        实时回放。目标越硬限位自动裁剪并告警。

        :param q: 目标关节角 ``(n_arm,)``，弧度。
        :param t: 运动时长（秒）；不填按路程自动估计（峰值约 1.5 rad/s）。
        :param rate: 轨迹采样率 Hz（默认 config ``backend.arm.control_rate``，≤1000）。
        :param realtime: ``True`` 时按时间戳逐帧 sleep（演示动画用）。
        :raises ValueError: 目标维度与关节数不符。
        :raises RuntimeError: 未 connect / 未使能。
        """
        self._require_connected()
        self._require_enabled_arm()
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.n_arm:
            raise ValueError(
                f"fakearm.py - move_j：目标维度 {q.shape[0]} 与关节数 {self.n_arm} 不符")
        q_c = clamp_to_limits(q, self.arm_limits)
        if not np.array_equal(q, q_c):
            logger.warning("move_j：目标关节角越硬限位，已裁剪 %s → %s"
                           "（轨迹与到位状态以裁剪后为准）",
                           np.round(q, 4).tolist(), np.round(q_c, 4).tolist())
        q = q_c
        q0 = self._q.copy()
        if rate is None:
            rate = float(((self._config.get("backend") or {}).get("arm") or {})
                         .get("control_rate", 100.0))
        rate = min(float(rate), 1000.0)
        if t is None:
            t = float(np.max(np.abs(q - q0)))
        ts, qs, dqs = _cubic_traj(q0, q, float(t), rate)
        self._mode = ControlMode.POSITION          # 不在位置模式先切过来（同 JoyArm）
        for i in range(len(ts)):
            self._q = qs[i].copy()
            self._dq = dqs[i].copy()
            if realtime and i > 0:
                time.sleep(max(0.0, float(ts[i] - ts[i - 1])))
        return ts, qs, dqs

    # ===
    # 七、目标轨迹（规划器协议）：ToJointTrajPlanner 等规划器经这三对方法消费 arm
    # ===
    def set_target_traj(self, targets) -> None:
        """写入运动目标（写进去的是拷贝；语义同 JoyArm）。

        :param targets: :class:`TrajFrame` 一帧或列表；``None``/空列表 = 清空。
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

    # ===
    # 八、数学计算（固定默认 Pin 求解器；签名与 JoyArm 门面一致）
    # ===
    def fkine(self, q: np.ndarray, frame: Union[str, int], rep: str = "pose"):
        """正运动学：给关节角，算 ``frame`` 帧的位姿（``rep``：``pose``/``T``/``se3``）。"""
        return self._solve("fkine", "solve", self, q, frame=frame, rep=rep)

    def ikine(self, target: Pose, frame: Union[str, int], q0: np.ndarray, **kw):
        """逆运动学单解：给目标位姿，算关节角（``q0`` 为迭代起点，必填）。"""
        return self._solve("ikine", "solve", self, target, frame, q0, **kw)

    def ikine_all(self, target: Pose, frame: Union[str, int], **kw):
        """逆运动学全部解（数值法默认实现不支持，会抛 ``RuntimeError``）。"""
        try:
            return self._solve("ikine", "solve_all", self, target, frame, **kw)
        except NotImplementedError as e:
            raise RuntimeError(
                f"fakearm.py - ikine_all：默认数值求解器不支持求全部解：{e}") from e

    def jac(self, q: np.ndarray, frame: Union[str, int], ref: str = "base"):
        """雅可比 J(q)（``ref`` 取 ``local``/``base``：末端坐标系 / 基座系）。"""
        return self._solve("jacobian", "jac", self, q, frame=frame, ref=ref)

    def manipulability(self, q: np.ndarray, frame: Union[str, int]) -> float:
        """可操作度（衡量当前位形离奇异有多远，越大越灵活）。"""
        return self._solve("jacobian", "manipulability", self, q, frame=frame)

    def statics(self, q: np.ndarray, F: np.ndarray, frame: Union[str, int]):
        """静力学 ``τ = JᵀF``：给末端六维力，算平衡所需的各关节力矩。"""
        return self._solve("jacobian", "statics", self, q, F, frame=frame)

    def mass_matrix(self, q):
        """关节空间惯量矩阵 M(q)。"""
        return self._solve("dynamics", "mass_matrix", self, q)

    def gravity(self, q):
        """重力项 G(q)（保持当前位形各关节需要克服的重力力矩）。"""
        return self._solve("dynamics", "gravity", self, q)

    # ===
    # 九、基本信息与只读属性（返回的都是拷贝）
    # ===
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
        """本体关节名 → 索引。

        :raises ValueError: 名字不在列表中（消息列出可用名）。
        """
        try:
            return self._arm_joint_names.index(str(name))
        except ValueError:
            raise ValueError(
                f"fakearm.py - joint_index_arm：关节名 {name!r} 未找到；"
                f"可用：{self._arm_joint_names}") from None

    def joint_index_end(self, name: str) -> int:
        """末端电机名 → 索引。

        :raises ValueError: 名字不在列表中（消息列出可用名）。
        """
        try:
            return self._end_joint_names.index(str(name))
        except ValueError:
            raise ValueError(
                f"fakearm.py - joint_index_end：电机名 {name!r} 未找到；"
                f"可用：{self._end_joint_names}") from None

    def rand_q_arm(self,
        size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """在本体**硬限位**内随机采关节角（``(n_arm,)``；``size=N`` 给
        ``(N, n_arm)``）——教程里做工作空间采样等实验用。"""
        return rand_within_limits(self.arm_limits, size=size, rng=rng)

    def get_config(self) -> dict:
        """配置快照（深拷贝；教学数据如 MDH 参数经此读取）。"""
        return copy.deepcopy(self._config)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, n_arm={self.n_arm}, "
            f"n_end={self.n_end}, ee_frame={self.ee_frame_name!r}, "
            f"{'connected' if self.connected else 'offline'})"
        )
