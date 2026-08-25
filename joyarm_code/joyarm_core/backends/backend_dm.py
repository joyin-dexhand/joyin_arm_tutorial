"""``BackendDM`` —— joyarm_dm 真机后端。

USB-CAN 串口桥（如 ``/dev/ttyACM0``，921600）驱动整机 7 个 DM 电机：本体 6 关节
（joint1~3 = 4340P，joint4~6 = 4310）+ 两指夹爪（``gripper``，4310）。DM 通讯协议
参照 ``u2can/``（厂商参考库）的原理**重新实现**。

**总线共享规则**（config ``arm.channel`` 与 ``end.channel`` 决定）::

    channel 相同 → 共享单总线：一个串口句柄 + 一个 RX 读线程 + 一个 TX 锁，end 借用 arm 的总线；
    channel 不同 → 两条独立总线，互不干扰（支持末端独立通道的硬件形态）。

**DM 协议要点**（详见下文各常量/函数处注释）::

    发送帧（串口桥 → CAN）  30 字节定长模板，仅 [13:15]=CAN ID（LE）与
                            [21:29]=8 字节载荷随指令变化；
    应答帧（CAN → 串口桥）  16 字节定长：[0]=0xAA、[15]=0x55、[1]=CMD（0x11）、
                            [3:7]=CAN ID（LE32）、[7:15]=8 字节载荷；
    CAN ID 仲裁             MIT 指令=SlaveID；POS_VEL=0x100+SlaveID；
                            VEL=0x200+SlaveID；使能/失能/标零=SlaveID +
                            ``FF×7+cmd``（0xFC/0xFD/0xFE）；参数通道=0x7FF +
                            载荷内嵌 [slave_id_l, slave_id_h, 子码, RID, 值]，
                            子码 0x33 读 / 0x55 写 / 0xAA 存闪存 / 0xCC 状态请求；
    一发一收                每个指令帧/请求帧触发一帧应答，状态仅在收发后刷新；
    状态帧载荷              D0 高 4 位=错误码、D0 低 4 位=ID，D1~2=位置(16bit)，
                            D3~4=速度(12bit)，D4~5=力矩(12bit)，D6~7 未用；
    错误码语义              0=使能正常，8=失能正常，其余（1 过压/2 欠压/3 过流/
                            4 功率故障/5 超温/6 通信丢失/7 过载等）=故障。

**实现结构**（本文件内三层，``DmMotor``/``DmCanBus`` 为私有协议层、不导出）::

    协议常量 + 编解码纯函数   帧封装/提取、MIT 位打包、状态解包、float↔uint；
    DmMotor                   单电机配置（config joint 条目）+ 状态/参数槽 + 事件；
    DmCanBus                  串口桥总线：TX 锁 + RX 守护线程 + 帧分发 + 电机原语；
    BackendDM                 :class:`~joyarm_core.backends.backend.Backend` 适配层。

时序约束（依 u2can 标定）：控制帧间隔建议 ≥1ms；使能应答 ~100ms 内；参数读写
带重试确认（写回读比对）；标零前须失能且反馈无故障；存闪存（0xAA）前必须失能。
"""
from __future__ import annotations

import threading
import time
from struct import pack, unpack
from typing import Optional

import numpy as np

from ..utils.types import ArmState, ControlMode, JointState
from .backend import Backend

__all__ = ["BackendDM"]


# ============================================================
# 协议常量
# ============================================================
# 型号 → (PMAX rad, VMAX rad/s, TMAX N·m)，MIT 打包与状态解包的线性标度极限；
# 仅收录本项目所用型号（4340P 为 4340 的命名变体，限值相同）
_MOTOR_LIMITS: dict[str, tuple[float, float, float]] = {
    "4310": (12.5, 30.0, 10.0),
    "4340": (12.5, 10.0, 28.0),
    "4340P": (12.5, 10.0, 28.0),
}

# MIT 帧固定位宽：q 16bit / dq 12bit / tau 12bit（按型号极限缩放），kp/kd 12bit（固定量程）
_KP_MAX, _KD_MAX = 500.0, 5.0

# 错误码：0=使能正常、8=失能正常、其余=故障
_ERR_ENABLED, _ERR_DISABLED = 0, 8

# 电机寄存器（RID）：10=控制模式（1 MIT / 2 POS_VEL / 3 VEL），25~28=POS_VEL 闭环增益
_RID_CTRL_MODE = 10

# 参数名（基类/用户侧字符串 key）→ DM 寄存器 RID
_PARAM_RIDS: dict[str, int] = {
    "ctrl_mode": 10,
    "vel_kp": 25,
    "vel_ki": 26,
    "pos_kp": 27,
    "pos_ki": 28,
}

# uint32 型寄存器 RID 集合（其余按 float32 收发）
_UINT_RIDS = frozenset(range(7, 11)) | frozenset(range(13, 17)) | {35, 36}

# ControlMode（三态）→ DM 电机控制模式寄存器值
_DM_MODE: dict[ControlMode, int] = {
    ControlMode.MIT: 1,
    ControlMode.POSITION: 2,
    ControlMode.VELOCITY: 3,
}


# ============================================================
# 编解码纯函数
# ============================================================
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """浮点 → 定点整数（先裁剪到量程，线性映射到 ``[0, 2^bits-1]``）。"""
    span = x_max - x_min
    return int(round((_clamp(x, x_min, x_max) - x_min) / span * ((1 << bits) - 1)))


def _uint_to_float(x: int, x_min: float, x_max: float, bits: int) -> float:
    """定点整数 → 浮点（:func:`_float_to_uint` 的逆映射）。"""
    span = x_max - x_min
    return x / ((1 << bits) - 1) * span + x_min


def _pack_mit(q: float, dq: float, tau: float, kp: float, kd: float,
              limits: tuple[float, float, float]) -> bytes:
    """打包 MIT 指令帧载荷（8 字节）：``q16 | dq12 | kp12 | kd12 | tau12`` 共 64bit。"""
    pmax, vmax, tmax = limits
    q_u = _float_to_uint(q, -pmax, pmax, 16)
    dq_u = _float_to_uint(dq, -vmax, vmax, 12)
    kp_u = _float_to_uint(kp, 0.0, _KP_MAX, 12)
    kd_u = _float_to_uint(kd, 0.0, _KD_MAX, 12)
    tau_u = _float_to_uint(tau, -tmax, tmax, 12)
    b = bytearray(8)
    b[0] = (q_u >> 8) & 0xFF
    b[1] = q_u & 0xFF
    b[2] = (dq_u >> 4) & 0xFF
    b[3] = ((dq_u & 0xF) << 4) | ((kp_u >> 8) & 0xF)
    b[4] = kp_u & 0xFF
    b[5] = (kd_u >> 4) & 0xFF
    b[6] = ((kd_u & 0xF) << 4) | ((tau_u >> 8) & 0xF)
    b[7] = tau_u & 0xFF
    return bytes(b)


def _unpack_status(data: bytes, limits: tuple[float, float, float]):
    """解包状态帧载荷（8 字节）→ ``(q, dq, tau, err)``。"""
    pmax, vmax, tmax = limits
    err = (data[0] >> 4) & 0x0F
    q = _uint_to_float((data[1] << 8) | data[2], -pmax, pmax, 16)
    dq = _uint_to_float((data[3] << 4) | (data[4] >> 4), -vmax, vmax, 12)
    tau = _uint_to_float(((data[4] & 0xF) << 8) | data[5], -tmax, tmax, 12)
    return float(q), float(dq), float(tau), int(err)


def _pack_tx(can_id: int, data: bytes) -> bytes:
    """封装串口桥发送帧（30 字节定长模板，仅 CAN ID 与载荷段可变）。"""
    if len(data) != 8:
        # 切片赋值 f[21:29] 不保长：短载荷会把 30 字节帧缩成 <30，桥接固件丢帧
        raise ValueError(f"CAN 载荷须为 8 字节，收到 {len(data)}")
    f = bytearray(30)
    f[0], f[1] = 0x55, 0xAA       # 帧头
    f[2] = 0x1E                    # 帧长 30
    f[3] = 0x03                    # 帧类型（TX）
    f[4] = 0x01
    f[8] = 0x0A
    f[13] = can_id & 0xFF          # CAN ID 低 / 高字节（LE）
    f[14] = (can_id >> 8) & 0xFF
    f[18] = 0x08                   # DLC=8
    f[21:29] = data                # 8 字节 CAN 载荷
    return bytes(f)


def _extract_frames(buf: bytes) -> tuple[list[bytes], bytes]:
    """从缓冲中滑动扫描 16 字节应答帧（``0xAA ... 0x55``），返回 (帧列表, 残余)。

    残余为最后一个有效帧之后的字节（可能含跨读的半帧），由调用方与下次数据拼接。
    """
    frames, i, last = [], 0, 0
    while i <= len(buf) - 16:
        if buf[i] == 0xAA and buf[i + 15] == 0x55:
            frames.append(buf[i:i + 16])
            i += 16
            last = i
        else:
            i += 1
    return frames, buf[last:]


def _rid_is_uint(rid: int) -> bool:
    return rid in _UINT_RIDS


def _value_matches(got, want: float) -> bool:
    """写参数回读比对（float32 精度容差；int 精确）。"""
    if isinstance(want, int) and not isinstance(want, bool):
        return int(got) == want
    return abs(got - want) <= max(1e-4, 1e-3 * abs(want))


# ============================================================
# DmMotor：单电机配置 + 状态/参数槽
# ============================================================
class DmMotor:
    """单个 DM 电机：config 一个 joint 条目 → 一个实例。

    持有通讯参数（``motor_id``/``feedback_id``/型号限值）、控制增益（config
    ``MIT`` / ``POS_VEL`` 段回退源）、末端行程语义（``q_min``/``q_max``/
    ``force_to_tau``，仅 end 关节配置），以及 RX 线程回填的状态槽与参数槽
    （配到达 Event，供"发请求 → 等应答"同步）。
    """

    def __init__(self, jcfg: dict) -> None:
        self.name = str(jcfg.get("name", "motor"))
        self.motor_id = int(jcfg["motor_id"])
        self.feedback_id = int(jcfg["feedback_id"])
        model = str(jcfg.get("model", ""))
        if model not in _MOTOR_LIMITS:
            raise ValueError(
                f"关节 {self.name} 电机型号 {model!r} 不受支持；可用：{sorted(_MOTOR_LIMITS)}"
            )
        self.model = model
        self.limits = _MOTOR_LIMITS[model]

        mit = jcfg.get("MIT") or {}
        self.mit_kp = float(mit.get("kp", 0.0))
        self.mit_kd = float(mit.get("kd", 0.0))
        pv = jcfg.get("POS_VEL") or {}
        self.gains = {
            "vel_kp": float(pv.get("vel_kp", 0.0)),
            "vel_ki": float(pv.get("vel_ki", 0.0)),
            "pos_kp": float(pv.get("pos_kp", 0.0)),
            "pos_ki": float(pv.get("pos_ki", 0.0)),
        }
        self.vlim = float(pv.get("vlim", 0.0))

        # 末端行程语义（可选；arm 关节不配置）
        self.q_min = None if jcfg.get("q_min") is None else float(jcfg["q_min"])
        self.q_max = None if jcfg.get("q_max") is None else float(jcfg["q_max"])
        self.force_to_tau = float(jcfg.get("force_to_tau", 0.1))

        # 状态槽 + 参数槽（RX 线程写、指令线程读，Event 通知应答到达）
        self.q, self.dq, self.tau, self.err = 0.0, 0.0, 0.0, _ERR_DISABLED
        self._state_event = threading.Event()
        self.params: dict[int, float | int] = {}
        self._param_events: dict[int, threading.Event] = {}
        self.bus: Optional[DmCanBus] = None  # connect 注册时回填

    # ---- 状态槽 ----
    def update_state(self, q: float, dq: float, tau: float, err: int) -> None:
        self.q, self.dq, self.tau, self.err = q, dq, tau, err
        self._state_event.set()

    def clear_state(self) -> None:
        self._state_event.clear()

    def wait_state(self, timeout: float) -> bool:
        """等待状态应答到达（发请求帧前先 :meth:`clear_state`）。"""
        return self._state_event.wait(timeout)

    # ---- 参数槽 ----
    def update_param(self, rid: int, value) -> None:
        self.params[rid] = value
        self._param_events.setdefault(rid, threading.Event()).set()

    def clear_param(self, rid: int) -> None:
        self.params.pop(rid, None)
        self._param_events.pop(rid, None)

    def wait_param(self, rid: int, timeout: float) -> bool:
        ev = self._param_events.get(rid)
        return ev is not None and ev.wait(timeout)


# ============================================================
# DmCanBus：USB-CAN 串口桥上的 DM 电机总线
# ============================================================
class DmCanBus:
    """一条串口 CAN 总线：TX 锁 + RX 守护线程 + 帧分发 + 电机原语。

    pyserial 延迟到 :meth:`open` 导入（离线环境无需安装）；RX 线程持续收帧、
    按应答 CAN ID（正常=电机 feedback_id，``CANID==0`` 回退 D0 低 4 位）分发
    状态到电机状态槽；参数应答（载荷 ``[2]`` 为 0x33/0x55 且载荷内 slave ID
    已注册）分发到参数槽——双重判别防止状态帧位置低字节撞上参数子码。
    """

    def __init__(self, channel: str, baud_rate: int = 921600) -> None:
        self.channel = channel
        self.baud_rate = int(baud_rate)
        self._ser = None
        self._rx_thread: Optional[threading.Thread] = None
        self._stop = False
        self._tx_lock = threading.Lock()
        self._rx_buf = bytearray()
        self._by_fid: dict[int, DmMotor] = {}        # 状态帧：CAN ID(=feedback_id) → 电机
        self._by_fid_low: dict[int, DmMotor] = {}    # CANID==0 回退：低 4 位 → 电机
        self._by_mid: dict[int, DmMotor] = {}        # 参数应答：载荷内 slave_id → 电机

    # ---- 生命周期 ----
    def open(self) -> None:
        import serial  # 延迟导入

        if self._ser is not None and self._ser.is_open:
            return
        self._ser = serial.Serial(self.channel, self.baud_rate, timeout=0.02)
        self._stop = False
        self._rx_thread = threading.Thread(
            target=self._rx_loop, daemon=True, name=f"dm-rx:{self.channel}"
        )
        self._rx_thread.start()

    def close(self) -> None:
        self._stop = True
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def add_motor(self, motor: DmMotor) -> None:
        motor.bus = self
        self._by_fid[motor.feedback_id] = motor
        self._by_fid_low[motor.feedback_id & 0x0F] = motor
        self._by_mid[motor.motor_id] = motor

    # ---- 收发 ----
    def send(self, can_id: int, data: bytes) -> None:
        """发送一帧 CAN 报文（30 字节桥帧封装，TX 锁串行化）。"""
        frame = _pack_tx(can_id, bytes(data))
        with self._tx_lock:
            self._ser.write(frame)

    def _rx_loop(self) -> None:
        while not self._stop:
            try:
                data = self._ser.read(4096)
            except Exception:
                break  # 串口关闭/拔出
            if data:
                self._feed(data)

    def _feed(self, data: bytes) -> None:
        self._rx_buf += data
        frames, rest = _extract_frames(bytes(self._rx_buf))
        self._rx_buf = bytearray(rest)
        for f in frames:
            self._dispatch(f)

    def _dispatch(self, frame: bytes) -> None:
        if frame[1] != 0x11:
            return
        can_id = int.from_bytes(frame[3:7], "little")
        d = frame[7:15]
        slave = (d[1] << 8) | d[0]
        # 参数应答：[slave_l, slave_h, 子码, RID, 值4B]。除子码 0x33/0x55 外还须
        # slave 为已注册电机 ID：状态帧的 d[2] 是位置低字节，恰取 0x33/0x55 时会被
        # 误判成参数应答（保持固定位形时确定性复现）；而状态帧的 d[0]/d[1]（错误码
        # +ID / 位置高字节）拼出有效 slave 仅在位置贴近 -PMAX 量程底的个别码点，
        # 实际关节限位（约 ±3 rad << PMAX=12.5 rad）不可达
        if d[2] in (0x33, 0x55) and slave in self._by_mid:
            motor = self._by_fid.get(can_id) or self._by_mid[slave]
            rid = d[3]
            value = (
                int.from_bytes(d[4:8], "little")
                if _rid_is_uint(rid)
                else unpack("<f", d[4:8])[0]
            )
            motor.update_param(rid, value)
        else:  # 状态帧
            motor = self._by_fid.get(can_id)
            if motor is None and can_id == 0:
                motor = self._by_fid_low.get(d[0] & 0x0F)
            if motor is None:
                return
            motor.update_state(*_unpack_status(d, motor.limits))

    # ---- 电机原语：指令 ----
    def send_mit(self, m: DmMotor, q: float, dq: float, tau: float, kp: float, kd: float) -> None:
        self.send(m.motor_id, _pack_mit(q, dq, tau, kp, kd, m.limits))

    def send_pos_vel(self, m: DmMotor, p: float, v: float) -> None:
        self.send(0x100 + m.motor_id, pack("<ff", p, v))

    def send_vel(self, m: DmMotor, v: float) -> None:
        # 速度 float32 占前 4 字节、后 4 字节补零（对齐厂商 control_Vel 布局）
        self.send(0x200 + m.motor_id, pack("<f", v) + b"\x00" * 4)

    # ---- 电机原语：使能/失能/标零/状态请求 ----
    def enable(self, m: DmMotor) -> None:
        self.send(m.motor_id, b"\xff" * 7 + b"\xfc")

    def disable(self, m: DmMotor) -> None:
        self.send(m.motor_id, b"\xff" * 7 + b"\xfd")

    def set_zero(self, m: DmMotor) -> None:
        self.send(m.motor_id, b"\xff" * 7 + b"\xfe")

    def refresh(self, m: DmMotor) -> None:
        """请求一次状态应答（0x7FF + 0xCC，一发一收）。"""
        self._send_param_frame(m, 0xCC)

    # ---- 电机原语：参数通道 ----
    def _send_param_frame(self, m: DmMotor, code: int, rid: int = 0,
                          payload: bytes = b"") -> None:
        data = (
            bytes([m.motor_id & 0xFF, (m.motor_id >> 8) & 0xFF, code, rid])
            + payload
            + b"\x00" * (4 - len(payload))
        )
        self.send(0x7FF, data)

    def read_param(self, m: DmMotor, rid: int, timeout: float = 0.1, retries: int = 5):
        """读电机参数（0x33，重试等待应答）。"""
        for _ in range(retries):
            m.clear_param(rid)
            self._send_param_frame(m, 0x33, rid)
            if m.wait_param(rid, timeout):
                return m.params[rid]
        raise TimeoutError(f"电机 {m.name} 读参数 RID={rid} 无应答（{self.channel}）")

    def write_param(self, m: DmMotor, rid: int, value, timeout: float = 0.1,
                    retries: int = 5) -> None:
        """写电机参数（0x55，写回应答回读比对确认）。"""
        payload = (
            pack("<I", int(value)) if _rid_is_uint(rid) else pack("<f", float(value))
        )
        for _ in range(retries):
            m.clear_param(rid)
            self._send_param_frame(m, 0x55, rid, payload)
            if m.wait_param(rid, timeout) and _value_matches(m.params[rid], value):
                return
        raise RuntimeError(f"电机 {m.name} 写参数 RID={rid}={value} 未确认")

    def save_params(self, m: DmMotor) -> None:
        """参数存闪存（0xAA；**须先失能**，此处自动失能且保持失能态）。"""
        self.disable(m)
        self._send_param_frame(m, 0xAA)
        time.sleep(0.001)

    def switch_mode(self, m: DmMotor, dm_mode: int) -> None:
        """切换控制模式（写 RID10 并经写确认校验）。"""
        self.write_param(m, _RID_CTRL_MODE, dm_mode)


# ============================================================
# BackendDM：Backend 适配层
# ============================================================
class BackendDM(Backend):
    """JoyArm（joyarm_dm）整机真机通信后端（达妙 DM，USB-CAN 串口桥）。

    :param cfg: yaml ``backend:`` 段字典（``name`` 已由 JoyArm 弹出），含
        ``arm:`` / ``end:`` 子段（``channel`` / ``baud_rate`` / ``joints``，
        joints 各含 ``motor_id`` / ``feedback_id`` / ``model`` / ``MIT`` /
        ``POS_VEL``，end 关节另含 ``q_min`` / ``q_max`` / ``force_to_tau``）。
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._arm_cfg = cfg.get("arm") or {}
        self._end_cfg = cfg.get("end") or {}
        self._arm_motors = [DmMotor(j) for j in self._arm_cfg.get("joints") or []]
        self._end_motors = [DmMotor(j) for j in self._end_cfg.get("joints") or []]
        self._n = len(self._arm_motors)
        self._buses: dict[str, DmCanBus] = {}  # channel → 总线（同 channel 共享）
        self._mode_arm: Optional[ControlMode] = None
        self._mode_end: Optional[ControlMode] = None

    # ----------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------
    def _check_open(self) -> None:
        if not self._buses:
            raise RuntimeError("BackendDM 未连接；请先 connect()")

    def _motors_for(self, joint: Optional[int]) -> list[DmMotor]:
        if joint is None:
            return self._arm_motors
        if not 0 <= joint < self._n:
            raise ValueError(f"关节索引 {joint} 超出 [0, {self._n})")
        return [self._arm_motors[joint]]

    def _rows(self, cmd, name: str, joint: Optional[int]) -> list[tuple[DmMotor, float]]:
        """指令数组 → (电机, 标量值) 对齐列表（维度校验）。"""
        motors = self._motors_for(joint)
        a = np.asarray(cmd, dtype=float).reshape(-1)
        if a.size != len(motors):
            raise ValueError(f"{name} 维度 {a.size} ≠ 关节数 {len(motors)}")
        return list(zip(motors, a))

    def _require_mode_arm(self, mode: ControlMode) -> None:
        if self._mode_arm != mode:
            cur = "未设置" if self._mode_arm is None else self._mode_arm.value
            raise RuntimeError(f"当前本体模式为 {cur}；请先 set_mode_arm({mode.value})")

    def _end_motors_for(self, joint: Optional[int]) -> list[DmMotor]:
        """取末端电机子集（``joint=None`` 全部）；未配置末端时抛 ``RuntimeError``。"""
        if not self._end_motors:
            raise RuntimeError("本型号未配置末端（config backend.end 缺失）")
        if joint is None:
            return self._end_motors
        if not 0 <= joint < len(self._end_motors):
            raise ValueError(f"末端电机索引 {joint} 超出 [0, {len(self._end_motors)})")
        return [self._end_motors[joint]]

    def _require_mode_end(self, mode: ControlMode) -> None:
        if not self._end_motors:
            raise RuntimeError("本型号未配置末端（config backend.end 缺失）")
        if self._mode_end != mode:
            cur = "未设置" if self._mode_end is None else self._mode_end.value
            raise RuntimeError(f"当前末端模式为 {cur}；请先 set_mode_end({mode.value})")

    @staticmethod
    def _end_values(cmd, name: str, motors: list[DmMotor]) -> np.ndarray:
        """末端连续量指令 → 与所选电机数对齐的向量（标量广播 / 序列等长校验）。"""
        a = np.asarray(cmd, dtype=float).reshape(-1)
        if a.size == 1:
            return np.full(len(motors), float(a[0]))
        if a.size != len(motors):
            raise ValueError(f"{name} 维度 {a.size} ≠ 所选末端电机数 {len(motors)}")
        return a

    def _enable_motors(self, motors: list[DmMotor]) -> None:
        for m in motors:
            m.clear_state()
            m.bus.enable(m)
            if not m.wait_state(0.5):
                raise RuntimeError(f"电机 {m.name} 使能无应答")
            if m.err != _ERR_ENABLED:
                raise RuntimeError(f"电机 {m.name} 使能失败（错误码 {m.err}）")

    def _set_zero_motors(self, motors: list[DmMotor]) -> None:
        """标零流程：失能 → 轮询反馈至无故障（错误码 0/8）→ 发标零帧。"""
        for m in motors:
            m.bus.disable(m)
            ok = False
            for _ in range(10):
                m.clear_state()
                m.bus.refresh(m)
                if m.wait_state(0.05) and m.err in (_ERR_ENABLED, _ERR_DISABLED):
                    ok = True
                    break
            if not ok:
                raise RuntimeError(f"电机 {m.name} 标零前存在故障（错误码 {m.err}）")
            m.clear_state()
            m.bus.set_zero(m)
            m.wait_state(0.5)

    def _switch_group_mode(self, motors: list[DmMotor], mode: ControlMode) -> None:
        """逐电机切模式：POSITION 先写 POS_VEL 闭环增益（RID25~28）再切模式确认。"""
        dm_mode = _DM_MODE.get(mode)
        if dm_mode is None:
            raise ValueError(f"未知控制模式：{mode}")
        for m in motors:
            if mode == ControlMode.POSITION:
                for rid, key in ((25, "vel_kp"), (26, "vel_ki"), (27, "pos_kp"), (28, "pos_ki")):
                    m.bus.write_param(m, rid, m.gains[key])
            m.bus.switch_mode(m, dm_mode)

    @staticmethod
    def _read_group_state(motors: list[DmMotor], timeout: float = 0.05) -> list[bool]:
        """两段式反馈：清事件 → 批量发刷新帧 → 逐电机等应答，返回应答到达标志。"""
        for m in motors:
            m.clear_state()
        for m in motors:
            m.bus.refresh(m)
        return [m.wait_state(timeout) for m in motors]

    # ----------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------
    def connect(self) -> None:
        for sec, motors in ((self._arm_cfg, self._arm_motors), (self._end_cfg, self._end_motors)):
            if not sec:
                continue
            channel = sec.get("channel")
            bus = self._buses.get(channel)
            if bus is None:
                bus = DmCanBus(channel, sec.get("baud_rate", 921600))
                bus.open()
                self._buses[channel] = bus
            for m in motors:
                bus.add_motor(m)

    def disconnect(self) -> None:
        # 顺序：失能电机 → 停 RX 线程 → 关串口（共享总线只关一次）
        for m in self._arm_motors + self._end_motors:
            if m.bus is not None:
                m.bus.disable(m)
        for bus in self._buses.values():
            bus.close()
        self._buses.clear()
        for m in self._arm_motors + self._end_motors:
            m.bus = None
        self._mode_arm = None
        self._mode_end = None

    # ----------------------------------------------------------
    # 本体：_arm
    # ----------------------------------------------------------
    def enable_arm(self, joint: Optional[int] = None) -> None:
        self._check_open()
        self._enable_motors(self._motors_for(joint))

    def disable_arm(self, joint: Optional[int] = None) -> None:
        self._check_open()
        for m in self._motors_for(joint):
            m.bus.disable(m)

    def set_zero_arm(self, joint: Optional[int] = None) -> None:
        self._check_open()
        self._set_zero_motors(self._motors_for(joint))

    def set_mode_arm(self, mode: ControlMode) -> None:
        self._check_open()
        if mode == self._mode_arm:
            return
        self._switch_group_mode(self._arm_motors, mode)
        self._mode_arm = mode

    def read_state_arm(self) -> ArmState:
        self._check_open()
        arrived = self._read_group_state(self._arm_motors)
        errs = [m.err for m in self._arm_motors]
        errors = [
            f"{m.name}: 错误码 {e}" for m, ok, e in zip(self._arm_motors, arrived, errs)
            if ok and e not in (_ERR_ENABLED, _ERR_DISABLED)
        ] + [f"{m.name}: 通讯无应答" for m, ok in zip(self._arm_motors, arrived) if not ok]
        return ArmState(
            joint=JointState(
                control_mode=self._mode_arm or ControlMode.POSITION,
                q=np.array([m.q for m in self._arm_motors]),
                dq=np.array([m.dq for m in self._arm_motors]),
                ddq=np.zeros(self._n),
                tau=np.array([m.tau for m in self._arm_motors]),
                enabled=np.array([e == _ERR_ENABLED for e in errs]),
                error=np.array([e not in (_ERR_ENABLED, _ERR_DISABLED) for e in errs]),
                comm_ok=np.array(arrived, dtype=bool),
                # DM 反馈无独立编码器状态位，以通讯正常近似
                angle_ok=np.array(arrived, dtype=bool),
                voltage=0.0,
                current=0.0,
            ),
            mode=self._mode_arm or ControlMode.POSITION,
            timestamp=time.time(),
            errors=errors,
        )

    def send_position_arm(self, q: np.ndarray, joint: Optional[int] = None) -> None:
        self._require_mode_arm(ControlMode.POSITION)
        for m, qi in self._rows(q, "q", joint):
            m.bus.send_pos_vel(m, float(qi), m.vlim)  # 限速取 config POS_VEL.vlim

    def send_velocity_arm(self, dq: np.ndarray, joint: Optional[int] = None) -> None:
        self._require_mode_arm(ControlMode.VELOCITY)
        for m, dqi in self._rows(dq, "dq", joint):
            m.bus.send_vel(m, float(dqi))

    def send_mit_arm(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_ff: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        self._require_mode_arm(ControlMode.MIT)
        motors = self._motors_for(joint)

        def _vec(v, name, default_per_motor=None):
            if v is None:
                return np.array(default_per_motor, dtype=float)
            a = np.asarray(v, dtype=float).reshape(-1)
            if a.size != len(motors):
                raise ValueError(f"{name} 维度 {a.size} ≠ 关节数 {len(motors)}")
            return a

        q_v = _vec(q, "q")
        dq_v = _vec(dq, "dq")
        tau_v = _vec(tau_ff, "tau_ff")
        kp_v = _vec(kp, "kp", [m.mit_kp for m in motors])  # None → config MIT.kp
        kd_v = _vec(kd, "kd", [m.mit_kd for m in motors])
        for m, qi, dqi, ti, kpi, kdi in zip(motors, q_v, dq_v, tau_v, kp_v, kd_v):
            m.bus.send_mit(m, qi, dqi, ti, kpi, kdi)

    def read_param_arm(self, joint: int, key: str):
        self._check_open()
        rid = _PARAM_RIDS.get(key)
        if rid is None:
            raise ValueError(f"未知参数名 {key!r}；可用：{sorted(_PARAM_RIDS)}")
        m = self._motors_for(joint)[0]
        return m.bus.read_param(m, rid)

    def write_param_arm(self, joint: int, key: str, value, persist: bool = False) -> None:
        self._check_open()
        rid = _PARAM_RIDS.get(key)
        if rid is None:
            raise ValueError(f"未知参数名 {key!r}；可用：{sorted(_PARAM_RIDS)}")
        m = self._motors_for(joint)[0]
        m.bus.write_param(m, rid, value)
        if persist:
            m.bus.save_params(m)  # 自动失能并保持失能（DM 存闪存硬约束）

    # ----------------------------------------------------------
    # 末端：_end（电机组，本型号=单夹爪电机；位置语义为电机弧度，0=张开、q_max=闭合）
    # ----------------------------------------------------------
    def enable_end(self, joint: Optional[int] = None) -> None:
        self._check_open()
        self._enable_motors(self._end_motors_for(joint))

    def disable_end(self, joint: Optional[int] = None) -> None:
        self._check_open()
        for m in self._end_motors_for(joint):
            m.bus.disable(m)

    def set_zero_end(self, joint: Optional[int] = None) -> None:
        self._check_open()
        self._set_zero_motors(self._end_motors_for(joint))

    def set_mode_end(self, mode: ControlMode) -> None:
        self._check_open()
        if mode == self._mode_end:
            return
        self._switch_group_mode(self._end_motors_for(None), mode)
        self._mode_end = mode

    def read_state_end(self) -> dict:
        self._check_open()
        motors = self._end_motors_for(None)
        arrived = self._read_group_state(motors)
        return {
            "q": [m.q for m in motors],
            "dq": [m.dq for m in motors],
            "tau": [m.tau for m in motors],
            "enabled": [m.err == _ERR_ENABLED for m in motors],
            "error": [m.err not in (_ERR_ENABLED, _ERR_DISABLED) for m in motors],
            "comm_ok": arrived,
        }

    def send_position_end(self, position, joint: Optional[int] = None) -> None:
        self._require_mode_end(ControlMode.POSITION)
        motors = self._end_motors_for(joint)
        for m, p in zip(motors, self._end_values(position, "position", motors)):
            if m.q_min is None or m.q_max is None:
                raise ValueError(f"末端关节 {m.name} 缺少 q_min/q_max 行程配置（config backend.end.joints）")
            m.bus.send_pos_vel(m, _clamp(float(p), m.q_min, m.q_max), m.vlim)

    def send_force_end(self, force, joint: Optional[int] = None) -> None:
        """末端力度控制（近似）：MIT 闭合至各电机 ``q_max``，前馈 ``force×force_to_tau``。

        DM 夹爪无力控通道与力反馈，``force_to_tau`` 为 N→N·m 近似换算系数
        （config 逐电机标定）；本方法须先 ``set_mode_end(MIT)``。
        """
        self._require_mode_end(ControlMode.MIT)
        motors = self._end_motors_for(joint)
        for m, f in zip(motors, self._end_values(force, "force", motors)):
            if m.q_max is None:
                raise ValueError(f"末端关节 {m.name} 缺少 q_max 行程配置（config backend.end.joints）")
            m.bus.send_mit(m, m.q_max, 0.0, float(f) * m.force_to_tau, m.mit_kp, m.mit_kd)

    def send_action_end(self, action: str) -> None:
        """末端整组离散动作：``open``→各电机 ``q_min``、``close``→各电机 ``q_max``。"""
        self._require_mode_end(ControlMode.POSITION)
        if action not in ("open", "close"):
            raise ValueError(f"未知末端动作 {action!r}；可用：['open', 'close']")
        motors = self._end_motors_for(None)
        for m in motors:
            if m.q_min is None or m.q_max is None:
                raise ValueError(f"末端关节 {m.name} 缺少 q_min/q_max 行程配置（config backend.end.joints）")
            target = m.q_min if action == "open" else m.q_max
            m.bus.send_pos_vel(m, target, m.vlim)

    def read_param_end(self, joint: int, key: str):
        self._check_open()
        rid = _PARAM_RIDS.get(key)
        if rid is None:
            raise ValueError(f"未知参数名 {key!r}；可用：{sorted(_PARAM_RIDS)}")
        m = self._end_motors_for(joint)[0]
        return m.bus.read_param(m, rid)

    def write_param_end(self, joint: int, key: str, value, persist: bool = False) -> None:
        self._check_open()
        rid = _PARAM_RIDS.get(key)
        if rid is None:
            raise ValueError(f"未知参数名 {key!r}；可用：{sorted(_PARAM_RIDS)}")
        m = self._end_motors_for(joint)[0]
        m.bus.write_param(m, rid, value)
        if persist:
            m.bus.save_params(m)
