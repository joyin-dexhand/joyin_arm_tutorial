"""``Backend`` —— 整机硬件通信后端抽象基类（定义多厂商关节电机的通用功能）。

一个后端 = 一台完整设备的硬件（多轴本体 + 末端执行器，共用或各用通信总线）；
子类按**电机厂商/型号**派生（结构相同、参数不同），以 ``_arm`` / ``_end``后缀区分本体与末端两组。

配置结构即代码结构（yaml ``backend:`` 段，``name`` 选型键经 ``REGISTRY`` 解析）::

    backend:
      name: backend_dm          # 选型键（JoyArm 弹出后按 REGISTRY 构建子类）
      arm: {channel, protocol, baud_rate, control_rate, joints}   # 本体子段
      end: {channel, protocol, baud_rate, joints}   # 末端子段（单帧锁存，无控制率）

接口按功能分类：生命周期（connect/disconnect）、使能失能（enable/disable/set_zero）、状态读取（read_state）、
模式切换（set_mode）、指令下发（send_*，「限位守卫模板 + 子类内核」）、电机参数读写（read_param/write_param）。

"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..utils.limits import limits_from_joint_cfgs
from ..utils.types import ArmState, ControlMode, JointLimits

__all__ = ["Backend"]

logger = logging.getLogger("joyarm_core.backend")

# 越限告警节流间隔（秒）：控制周期（如 200 Hz）内持续越限至多每 0.5s 告警一条，
# 裁剪本身不受节流影响、始终执行
_WARN_INTERVAL = 0.5


class Backend(ABC):
    """整机硬件通信后端抽象基类（本体 + 末端一体）。

    ``joint`` 形参：电机索引，``None`` 表示全部电机。

    :param cfg: yaml ``backend:`` 段字典（``name`` 已由 JoyArm 弹出），含
        ``arm:`` / ``end:`` 两个子段，各含 ``channel`` / ``protocol`` /
        ``baud_rate`` / ``joints``（电机配置列表，条目四键限位**自解析**为
        arm/end 硬限位）。
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        # 越限告警节流：上次告警的 time.monotonic 时刻（读写加锁，指令可能从控制线程与应用线程并发）
        self._warn_lock = threading.Lock()
        self._warn_last = 0.0
        # ---- 硬限位构建（守卫依据）----
        self._arm_limits: Optional[JointLimits] = self.arm_limits_from_cfg(cfg)
        self._end_limits: Optional[JointLimits] = self.end_limits_from_cfg(cfg)

    # ----------------------------------------------------------
    # 硬限位属性（arm_limits / end_limits；下发守卫唯一依据）
    # ----------------------------------------------------------
    @property
    def arm_limits(self) -> Optional[JointLimits]:
        """本体硬限位（自 cfg ``arm.joints`` 解析；无 arm 段为 ``None``，守卫放行）。"""
        return self._arm_limits

    @property
    def end_limits(self) -> Optional[JointLimits]:
        """末端硬限位（自 cfg ``end.joints`` 解析；无末端段为 ``None``）。"""
        return self._end_limits

    @staticmethod
    def arm_limits_from_cfg(cfg: dict) -> Optional[JointLimits]:
        """从 backend config 解析**本体硬限位**（与 :meth:`end_limits_from_cfg` 同构）。

        arm joint 条目的 ``q_min`` / ``q_max`` / ``dq_max`` / ``tau_max`` 四键
        为 ABC 契约标准限位键（弧度 / rad/s / N·m；四键齐全性由
        ``JoyArm.check_config`` 保证，缺键量纲置 ±∞ 不参与守卫）。无 ``arm``
        段或 ``joints`` 为空 → ``None``（本体守卫放行）。

        :param cfg: yaml ``backend:`` 段字典（含 ``arm.joints`` 列表）。
        """
        return limits_from_joint_cfgs(((cfg or {}).get("arm") or {}).get("joints") or [])

    @staticmethod
    def end_limits_from_cfg(cfg: dict) -> Optional[JointLimits]:
        """从 backend config 解析**末端硬限位**（逐电机，多执行器末端通用）。

        末端 joint 条目的 ``q_min`` / ``q_max`` / ``dq_max`` / ``tau_max`` 四键
        为 ABC 契约标准限位键（弧度 / rad/s / N·m，位置语义由子类约定）；缺键
        的量纲置 ±∞（不参与守卫）。无 ``end`` 段或 ``joints`` 为空 → ``None``。

        :param cfg: yaml ``backend:`` 段字典（含 ``end.joints`` 列表）。
        """
        return limits_from_joint_cfgs(((cfg or {}).get("end") or {}).get("joints") or [])

    # ----------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------
    @abstractmethod
    def connect(self) -> None:
        """建立通信链路（本体 + 末端）。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开通信链路（失能电机 → 停接收线程 → 关总线）。"""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """通信链路是否已建立（``connect()`` 后 / ``disconnect()`` 前为 ``True``）。"""

    # ----------------------------------------------------------
    # 本体（关节电机）：_arm 后缀
    # ----------------------------------------------------------
    @abstractmethod
    def enable_arm(self, joint: Optional[int] = None) -> None:
        """使能本体关节电机（默认失能）。"""

    @abstractmethod
    def disable_arm(self, joint: Optional[int] = None) -> None:
        """失能本体关节电机。"""

    @abstractmethod
    def set_zero_arm(self, joint: Optional[int] = None) -> None:
        """本体零位标定（先失能，反馈无故障后再标零）。"""

    @abstractmethod
    def clear_fault_arm(self, joint: Optional[int] = None) -> None:
        """本体关节故障清除（验证式复位：失能清错 → 核对 → 使能 → 核对）。

        语义：逐电机 best-effort 执行协议规定的清错流程并**验证结果**；
        仍存在未恢复故障（如需断电的硬故障）时抛 ``RuntimeError`` 汇总
        （电机名 + 剩余故障码），全部恢复则静默返回。

        :param joint: 关节索引，``None`` 表示全部电机。
        """

    @abstractmethod
    def set_mode_arm(self, mode: ControlMode = ControlMode.POSITION, joint: Optional[int] = None) -> None:
        """切换本体控制模式（收指令前必须先切到对应模式；默认位置模式）。

        ``POSITION → 电机 POS_VEL``、``VELOCITY → 电机 VEL``、``MIT → 电机 MIT``；
        切换所需增益（POS_VEL 闭环参数 / MIT kp·kd）回退 config 对应段。

        :param mode: 目标控制模式，默认 ``ControlMode.POSITION``（电机 POS_VEL）。
        :param joint: 关节索引，``None`` 表示全部；仅对未处于目标模式的电机补切。
        """

    @abstractmethod
    def read_mode_arm(self, joint: Optional[int] = None) -> Optional[ControlMode]:
        """查询本体关节当前控制模式（本地缓存，不发总线帧，离线可查）。

        :param joint: 关节索引；``None`` 表示整臂。
        :return: 指定 ``joint`` 时返回该关节控制模式（未设置为 ``None``）；
            ``joint=None`` 时整臂各关节模式唯一才返回该模式，否则 ``None``。
        """

    @abstractmethod
    def read_state_arm(self, joint: Optional[int] = None) -> ArmState:
        """读取本体状态快照（关节角/速度/力矩/使能与错误状态等）。

        :param joint: 关节索引，``None`` 表示全部（数组 ``(n,)``）；指定关节时
            各数组仅含该关节（长度 1）。
        :return: :class:`ArmState` 快照。
        """

    # ----------------------------------------------------------
    # 指令限位守卫（arm/end 两族共用裁剪核；send_* 模板前置；只裁硬限位）
    # ----------------------------------------------------------
    def _guard_arm(self, name: str, arr: np.ndarray,joint: Optional[int]) -> np.ndarray:
        """本体指令守卫：``q`` 硬限位裁剪、``dq``/``tau`` 幅值裁剪。

        无硬限位（cfg 未配置 ``arm.joints``）/ 维度不符时原样放行。
        """
        return self._clip_within(self._arm_limits, name, arr, joint,family="arm", broadcast=False)

    def _guard_end(self, name: str, arr: np.ndarray,joint: Optional[int]) -> np.ndarray:
        """末端指令守卫（逐电机硬限位；多执行器末端支持标量广播裁剪）。"""
        return self._clip_within(self._end_limits, name, arr, joint,family="end", broadcast=True)

    def _clip_within(self, limits: Optional[JointLimits], name: str,arr: Optional[np.ndarray], 
            joint: Optional[int],family: str, broadcast: bool) -> Optional[np.ndarray]:
        """裁剪核：``q`` → [q_min, q_max]，``dq``/``tau`` → ±幅值上限，其余放行。

        ``joint`` 指定时取该关节限位切片；指令长度与限位不符时放行交内核校验；
        ``broadcast=True``（末端）且为标量/单元素整体指令时，按逐电机限位
        **广播裁剪**为 ``(n_end,)``（每电机各自就近裁剪，支持多执行器末端）。
        越限告警经节流（每 ``_WARN_INTERVAL`` 秒至多一条，防控制周期高频刷屏；
        裁剪本身始终执行），返回裁剪值。
        """
        if limits is None or arr is None:
            return arr
        if name == "q":
            lo, hi = limits.q_min, limits.q_max
        elif name == "dq":
            lo, hi = -limits.dq_max, limits.dq_max
        elif name == "tau":
            lo, hi = -limits.tau_max, limits.tau_max
        else:
            return arr
        if joint is not None:               # 单关节：取该关节限位（arr 长度 1）
            lo, hi = lo[joint:joint + 1], hi[joint:joint + 1]
        arr = np.asarray(arr).reshape(-1)  # 归一 1-D：(1,n)/(n,1) 等同长形态一并参与裁剪，防旁路
        if arr.shape != np.shape(lo):       # 长度 ≠ 限位数 → 交内核报维度错误
            if not (broadcast and arr.size == 1 and lo.size > 1):
                return arr
        clipped = np.clip(arr, lo, hi)      # 末端标量广播：→ (n_end,) 逐电机裁剪
        if not np.array_equal(np.broadcast_to(arr, clipped.shape), clipped):
            now = time.monotonic()
            with self._warn_lock:
                if now - self._warn_last >= _WARN_INTERVAL:
                    self._warn_last = now
                    logger.warning("send_*_%s: %s 指令越限，已就近裁剪 %s → %s",family, name, 
                                   np.round(arr, 4).tolist(), np.round(clipped, 4).tolist())
        return clipped

    @staticmethod
    def _vec(x) -> Optional[np.ndarray]:
        """指令升维：标量/列表 → ``(k,)`` float 数组（``None`` 透传）。"""
        if x is None:
            return None
        return np.atleast_1d(np.asarray(x, dtype=float))

    # ----------------------------------------------------------
    # 本体指令下发（send_*_arm 基类守卫模板 → 子类抽象内核 _send_*_arm）
    # ----------------------------------------------------------
    def send_position_arm(self, q: np.ndarray, joint: Optional[int] = None) -> None:
        """本体位置指令（守卫模板：``q`` 硬限位裁剪 → 内核限速下发）。

        :param q: 关节角目标 ``(n,)``，弧度；越限时就近裁剪到硬限位并告警。
        """
        self._send_position_arm(self._guard_arm("q", self._vec(q), joint), joint)

    @abstractmethod
    def _send_position_arm(self, q: np.ndarray, joint: Optional[int] = None) -> None:
        """本体位置指令内核（实现须按 config ``POS_VEL.vlim`` 限速）。

        :param q: 关节角目标 ``(n,)``，弧度（已经守卫裁剪）。
        """

    def send_velocity_arm(self, dq: np.ndarray, joint: Optional[int] = None) -> None:
        """本体速度指令（守卫模板：``dq`` 幅值裁剪 → 内核下发）。

        :param dq: 关节速度目标 ``(n,)``，rad/s；越限时裁剪到硬限位幅值并告警。
        """
        self._send_velocity_arm(self._guard_arm("dq", self._vec(dq), joint), joint)

    @abstractmethod
    def _send_velocity_arm(self, dq: np.ndarray, joint: Optional[int] = None) -> None:
        """本体速度指令内核。

        :param dq: 关节速度目标 ``(n,)``，rad/s（已经守卫裁剪）。
        """

    def send_mit_arm(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_ff: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        """本体 MIT 阻抗/前馈指令（守卫模板：``τ = kp·(q_des−q) + kd·(dq_des−dq) + tau_ff``）。

        ``q``/``dq``/``tau_ff`` 越限时就近裁剪并告警（kp/kd为增益，不裁剪），再委托内核下发。

        :param q: 位置目标 ``(n,)``，弧度。
        :param dq: 速度目标 ``(n,)``，rad/s。
        :param tau_ff: 前馈力矩 ``(n,)``，N·m；纯力矩用 ``kp=kd=0`` + ``tau_ff``。
        :param kp: 位置增益 ``(n,)``；``None`` 回退 config ``MIT.kp``。
        :param kd: 速度阻尼 ``(n,)``；``None`` 回退 config ``MIT.kd``。
        """
        self._send_mit_arm(
            self._guard_arm("q", self._vec(q), joint),
            self._guard_arm("dq", self._vec(dq), joint),
            self._guard_arm("tau", self._vec(tau_ff), joint),
            kp=self._vec(kp),
            kd=self._vec(kd),
            joint=joint)

    @abstractmethod
    def _send_mit_arm(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_ff: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        """本体 MIT 指令内核（q/dq/tau_ff 已经守卫裁剪）。"""


    @abstractmethod
    def read_param_arm(self, key: str, joint: Optional[int] = None):
        """读本体关节电机参数。

        :param key: 参数名（字符串，语义由子类映射到厂商寄存器，
            如 DM 的 ``"pos_kp"`` → 寄存器 27）。
        :param joint: 关节索引，``None`` 表示全部电机。
        :return: 指定 ``joint`` 时返回该电机参数值（float 或 int）；
            ``joint=None`` 时返回逐电机参数值列表。
        """

    @abstractmethod
    def write_param_arm(self, key: str, value, joint: Optional[int] = None,
                        persist: bool = False) -> None:
        """写本体关节电机参数。

        :param key: 参数名（语义由子类定义）。
        :param value: 参数值，标量（作用于所选全部电机）或与所选电机数
            一致的列表（逐电机）。
        :param joint: 关节索引，``None`` 表示全部电机。
        :param persist: ``True`` 时写入并持久化到非易失存储（如 DM 需先失能再存闪存）。
        """

    # ----------------------------------------------------------
    # 末端（执行器电机组）：_end 后缀（可多电机，如灵巧手）；
    # 连续量三法为「基类守卫模板 → 子类抽象内核」，离散动作不模板化
    # ----------------------------------------------------------
    @abstractmethod
    def enable_end(self, joint: Optional[int] = None) -> None:
        """使能末端执行器电机（``joint=None`` 全部）。"""

    @abstractmethod
    def disable_end(self, joint: Optional[int] = None) -> None:
        """失能末端执行器电机（``joint=None`` 全部）。"""

    @abstractmethod
    def set_zero_end(self, joint: Optional[int] = None) -> None:
        """末端零位标定（先失能，反馈无故障后再标零；``joint=None`` 全部）。"""

    @abstractmethod
    def clear_fault_end(self, joint: Optional[int] = None) -> None:
        """末端电机故障清除（语义同 :meth:`clear_fault_arm`，验证式复位）。"""

    @abstractmethod
    def set_mode_end(self, mode: ControlMode = ControlMode.POSITION,joint: Optional[int] = None) -> None:
        """切换末端控制模式（语义同 :meth:`set_mode_arm`，默认位置模式）。

        :param mode: 目标控制模式，默认 ``ControlMode.POSITION``（电机 POS_VEL）。
        :param joint: 末端电机索引，``None`` 表示全部。
        """

    @abstractmethod
    def read_mode_end(self, joint: Optional[int] = None) -> Optional[ControlMode]:
        """查询末端电机当前控制模式（语义同 :meth:`read_mode_arm`）。

        :param joint: 末端电机索引；``None`` 表示整个末端。
        :return: 指定 ``joint`` 时返回该电机控制模式（未设置为 ``None``）；
            ``joint=None`` 时整个末端各电机模式唯一才返回该模式，否则 ``None``。
        """

    @abstractmethod
    def read_state_end(self, joint: Optional[int] = None) -> dict:
        """读取末端状态快照（值为所选电机的逐电机序列）。

        :param joint: 末端电机索引，``None`` 表示全部。
        :return: 状态字典，字段由子类定义；值为逐电机序列（单电机末端为
            单元素序列）。例如 DM 夹爪：``{"q": [...], "dq": [...], "tau": [...],
            "enabled": [...], "error": [...], "comm_ok": [...], "temp_mos": [...],
            "temp_rotor": [...]}``。
        """

    def read_state_cache_arm(self, joint: Optional[int] = None) -> ArmState:
        """读取本体状态快照的**缓存视图**（q/dq/tau 换算关节空间，同 :meth:`read_state_arm`）。

        数据来自最近一次状态应答写入的缓存槽——控制流期间随指令帧同频刷新（一发一收），
        空闲期由上层保活刷新维持新鲜（见 :meth:`state_age_arm`）。

        :param joint: 关节索引，``None`` 表示全部。
        :raises NotImplementedError: 子类未实现缓存读。
        """
        raise NotImplementedError(
            "backend.py - Backend.read_state_cache_arm：本后端未实现缓存读，"
            "请回退 read_state_arm（请求-应答式同步刷新）")

    def read_state_cache_end(self, joint: Optional[int] = None) -> dict:
        """读取末端状态快照的**缓存视图**（同 :meth:`read_state_cache_arm`）。

        :param joint: 末端电机索引，``None`` 表示全部。
        :raises NotImplementedError: 子类未实现缓存读。
        """
        raise NotImplementedError(
            "backend.py - Backend.read_state_cache_end：本后端未实现缓存读，"
            "请回退 read_state_end（请求-应答式同步刷新）")

    def state_age_arm(self, joint: Optional[int] = None) -> float:
        """本体缓存状态的陈旧度（秒）：距最近一次状态应答的时间，取最旧电机。

        :param joint: 关节索引，``None`` 表示全部。
        :raises NotImplementedError: 子类未实现。
        """
        raise NotImplementedError(
            "backend.py - Backend.state_age_arm：本后端未实现陈旧度查询")

    def state_age_end(self, joint: Optional[int] = None) -> float:
        """末端缓存状态的陈旧度（秒；语义同 :meth:`state_age_arm`）。

        :param joint: 末端电机索引，``None`` 表示全部。
        :raises NotImplementedError: 子类未实现。
        """
        raise NotImplementedError(
            "backend.py - Backend.state_age_end：本后端未实现陈旧度查询")

    def send_position_end(self, position, joint: Optional[int] = None) -> None:
        """末端位置控制（守卫模板：``q`` 逐电机行程裁剪 → 内核下发）。

        :param position: 位置目标，标量（作用于所选全部电机，越限时按逐电机
            限位**广播就近裁剪**）或与所选电机数一致的序列；单位语义由子类
            约定（如夹爪电机弧度）。
        :param joint: 末端电机索引，``None`` 表示全部。
        """
        self._send_position_end(self._guard_end("q", self._vec(position), joint),
                                joint)

    @abstractmethod
    def _send_position_end(self, position, joint: Optional[int] = None) -> None:
        """末端位置指令内核（position 已经守卫裁剪）。"""

    def send_tau_end(self, tau, joint: Optional[int] = None) -> None:
        """末端力矩控制（守卫模板：tau 裁剪到逐电机 ±tau_max → 内核）。

        :param tau: 力矩目标，标量（作用于所选全部电机，越限时按逐电机
            限位**广播就近裁剪**）或与所选电机数一致的序列（N·m）。
        :param joint: 末端电机索引，``None`` 表示全部。
        """
        self._send_tau_end(self._guard_end("tau", self._vec(tau), joint),joint)

    @abstractmethod
    def _send_tau_end(self, tau, joint: Optional[int] = None) -> None:
        """末端力矩指令内核（tau 已经 ±tau_max 守卫裁剪）。"""

    def send_mit_end(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_ff: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        """末端 MIT 阻抗/前馈指令（守卫模板：``τ = kp·(q_des−q) + kd·(dq_des−dq) + tau_ff``）。

        ``q``/``dq``/``tau_ff`` 越限时就近裁剪并告警（``kp``/``kd`` 为标定增益，
        不裁剪），再委托内核。紧急阻尼（``JoyArm.damping_mode``）的末端通道：
        ``q=dq=tau=kp=0, kd>0`` 即纯黏滞阻尼；``kp/kd`` 为 ``None`` 时回退
        config 末端 ``MIT`` 增益。

        :param q: 位置目标（与所选电机数一致；``kp=0`` 时固件忽略）。
        :param dq: 速度目标。
        :param tau_ff: 前馈力矩。
        :param kp: 位置增益；``None`` 回退 config。
        :param kd: 速度阻尼；``None`` 回退 config。
        :param joint: 末端电机索引，``None`` 表示全部。
        """
        self._send_mit_end(
            self._guard_end("q", self._vec(q), joint),
            self._guard_end("dq", self._vec(dq), joint),
            self._guard_end("tau", self._vec(tau_ff), joint),
            kp=self._vec(kp),
            kd=self._vec(kd),
            joint=joint)

    @abstractmethod
    def _send_mit_end(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_ff: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        """末端 MIT 指令内核（q/dq/tau_ff 已经守卫裁剪）。"""

    @abstractmethod
    def send_action_end(self, action: str, joint: Optional[int] = None) -> None:
        """末端离散动作（预设目标由子类按 config 定义，作用于所选电机）。

        :param action: 动作名，常见 ``"open"`` / ``"close"`` / ``"zero"``；子类可扩展。
        :param joint: 末端电机索引，``None`` 表示全部。
        """

    @abstractmethod
    def read_param_end(self, key: str, joint: Optional[int] = None):
        """读末端电机参数。

        :param key: 参数名（语义由子类映射到厂商寄存器）。
        :param joint: 末端电机索引，``None`` 表示全部电机。
        :return: 指定 ``joint`` 时返回该电机参数值（float 或 int）；
            ``joint=None`` 时返回逐电机参数值列表。
        """

    @abstractmethod
    def write_param_end(self, key: str, value, joint: Optional[int] = None,
                        persist: bool = False) -> None:
        """写末端电机参数（``persist=True`` 持久化到非易失存储）。

        :param key: 参数名（语义由子类定义）。
        :param value: 参数值，标量（作用于所选全部电机）或与所选电机数
            一致的列表（逐电机）。
        :param joint: 末端电机索引，``None`` 表示全部电机。
        :param persist: ``True`` 时写入并持久化到非易失存储。
        """
