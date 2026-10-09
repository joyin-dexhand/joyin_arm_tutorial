"""``BackendDM`` —— DM（达妙）机械臂硬件后端：7 电机经 U2CAN 串口转 CAN 桥接入。

型号说明（协议实现自包含，参照 ``u2can/DM_CAN.py`` 厂商库重写、不导入）：

- **总线**：U2CAN 盒串口封装帧（发 30B ``55 AA .. 00``、收 16B ``AA .. 55``，CAN id + 8B 数据内嵌）；
  一条串口一条总线，arm/end 同 channel 自动共享，不同 channel 各开各的；
- **模式**：模式寄存器 RID10（RAM 态、断电丢失）：基类 MIT / VELOCITY / POSITION ↔ DM 1 / 3 / 4
  （4 = Torque_Pos 力位混合，控制帧走 ``0x300+SlaveID`` 的 pos_force 帧：pos float32 + vel×100 + 电流标幺×10000）；
  切 POSITION/VELOCITY 时随模式写 4 个增益寄存器（RID 25-28，缺配置 NaN 跳过），切 MIT 不写；
- **状态**：反馈帧带 q/dq/tau/error 与两路温度（data[6]=T_MOS、data[7]=T_Rotor，厂商库未解析的温度在此补齐）；
  error 码 DM 原生 0=失能 / 1=使能 / ≥2=故障，与全库约定一致直接透传；
- **参数**：``_PARAM_RIDS`` 键表读写 RAM 寄存器（无 save 功能；只读键见 ``_READONLY_KEYS``）；
- **末端动作**：``open/home/close/position`` 离散动作（目标 -5.0/-3.0/0.0 rad / 按需 kwarg ``position``）——
  自动切 POSITION 模式，vlim/flim 取 cfg 发送默认，位置经基类管道裁硬限位后下发。

线程模型：应用/控制线程（``send_*`` 高频 + ``get_state_*`` 非阻塞）｜基类低频刷新线程（10Hz）｜
每总线一条 RX 收取线程（默认 1000Hz 节拍阻塞读、cfg 顶层 ``rx_hz`` 可配；只收数据+识别+原子更新
电机状态快照，不主动发帧）。

可靠性（读回验证与重试均在内核闭环）：参数写等 0x55 回显核对（uint 精确 / float 容差）、参数读等
0x33 应答、使能/失能/设零/清错（0xFC/0xFD/0xFE/0xFB）等新状态帧核对 error 码，均重试 3 次（单次超时
0.05s）；应答按「电机+RID+应答时刻 ≥ 发送时刻」配对，防陈旧应答洗白。状态读取非阻塞：缓存新鲜
（≤50ms）直返，陈旧发一帧 0xCC 刷新后立即返回当前快照（下一拍自愈）；连续 >1s 无反馈先发刷新帧再按
数据异常上抛（电机掉线防冻结值洗白，电机恢复后下一拍自愈）。
"""
from __future__ import annotations

import logging
import math
import struct
import threading
import time
from typing import Optional

import serial

from ..utils.types import ControlMode
from .backend import Backend

__all__ = ["BackendDM"]

logger = logging.getLogger("joyarm_core.backend_dm")

# ---------------- DM 协议常量 ----------------
_CMD_ENABLE, _CMD_DISABLE, _CMD_ZERO, _CMD_CLEAR = 0xFC, 0xFD, 0xFE, 0xFB  # 使能/失能/设零/清错帧（FF×7+cmd → SlaveID）
_QUERY_HOST_ID = 0x7FF      # 0xCC 状态查询 / 0x33 参数读 / 0x55 参数写帧的固定主机 CAN id
_MODE_RID = 10              # 控制模式寄存器（CTRL_MODE，uint32）
_KP_MAX, _KD_MAX = 500.0, 5.0  # MIT 帧 kp / kd 位域量程（12bit）

# 基类模式 ↔ DM 模式寄存器码（POSITION=4 即 Torque_Pos/pos_force 力位混合；枚举外码 2/5/6/7 读回映射 None）
_MODE_CODE = {ControlMode.MIT: 1, ControlMode.VELOCITY: 3, ControlMode.POSITION: 4}
_CODE_MODE = {1: ControlMode.MIT, 3: ControlMode.VELOCITY, 4: ControlMode.POSITION}

# 参数寄存器组织三件套（读写共用；写入均为 RAM 态、断电丢失）
_PARAM_RIDS: dict = {"pos_kp": 27, "pos_ki": 28, "vel_kp": 25, "vel_ki": 26,
                     "pmax": 21, "vmax": 22, "tmax": 23, "hw_ver": 13, "sw_ver": 14, "sn": 15}
_READONLY_KEYS = frozenset({"hw_ver", "sw_ver", "sn"})          # 只读参数（写侧上抛拒绝）
_UINT_RIDS = frozenset(range(7, 11)) | frozenset(range(13, 17)) | {35, 36}  # uint32 寄存器 RID，其余 float32
_GAIN_RIDS = (("pos_kp", 27), ("pos_ki", 28), ("vel_kp", 25), ("vel_ki", 26))  # 切 POSITION/VELOCITY 随模式写入
_END_ACTION_POS = {"open": -5.0, "home": -3.0, "close": 0.0}  # 末端离散动作 → 目标位置 rad（夹爪固定位，均在行程内）

# 型号 → MIT 位域量程 (PMAX, VMAX, TMAX)（rad / rad/s / N·m，取自厂商 DM_Motor_Type 量程表，键小写匹配）
_MODEL_LIMITS = {"4310": (12.5, 30.0, 10.0), "4310_48v": (12.5, 50.0, 10.0),
                 "4340": (12.5, 10.0, 28.0), "4340p": (12.5, 10.0, 28.0),
                 "6006": (12.5, 45.0, 20.0), "8006": (12.5, 45.0, 40.0), "8009": (12.5, 45.0, 54.0)}

# ---------------- 时序常量 ----------------
_RX_HZ_DEFAULT = 1000.0  # RX 收取线程默认节拍 Hz（cfg 顶层 rx_hz 可覆盖）
_RX_HZ_MAX = 5000.0      # rx_hz 上限（过高 = 串口空转轮询烧 CPU，挤占控制线程）
_TX_TIMEOUT = 0.05       # 单次应答等待 s（正常回显毫秒级，50ms ≈ 10 倍余量）
_TX_RETRIES = 3          # 验证失败重试次数
_STATE_STALE = 0.05      # 电机状态缓存新鲜阈值 s：新鲜直返，陈旧发 0xCC 刷新
_STATE_DEAD = 1.0        # 硬陈旧阈值 s：超过按数据异常上抛（电机掉线防冻结值洗白）


def _f2u(x: float, x_min: float, x_max: float, bits: int) -> int:
    """float → 位域 uint（与厂商 ``float_to_uint`` 位级一致：钳位 → 归一化×满量程 → 向零截断）。"""
    if x <= x_min:
        x = x_min
    elif x > x_max:
        x = x_max
    return int((x - x_min) / (x_max - x_min) * ((1 << bits) - 1))


def _u2can_prefix(can_id: int) -> bytes:
    """U2CAN 发送帧 21B 前缀（帧头 55AA + 长度/命令字 + CAN id 小端嵌 13-14 + DLC=8），尾部 1B 组帧时补。"""
    return bytes((0x55, 0xAA, 0x1E, 0x03, 0x01, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00,
                  can_id & 0xFF, (can_id >> 8) & 0xFF, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00))


def _param_echo_ok(echo, value: float, is_uint: bool) -> bool:
    """参数写回显核对：uint 精确等值；float 与写入值的 float32 位型比（rel/abs 容差 1e-3）。"""
    if echo is None:
        return False
    if is_uint:
        return echo == int(value)
    v32 = struct.unpack("<f", struct.pack("<f", value))[0]
    return abs(echo - v32) <= 1e-3 * max(1.0, abs(v32))


class _DMMotor:
    """DM 单电机协议数据（模块私有）：总线标识、位域量程、预构建发送帧、状态快照与参数应答槽。"""

    def __init__(self, slave_id: int, master_id: int, pmax: float, vmax: float, tmax: float) -> None:
        self.slave_id = slave_id      # 指令帧 CAN id（MIT/使能/失能/设零/清错）
        self.master_id = master_id    # 应答帧 CAN id（不为 0）
        self.pmax, self.vmax, self.tmax = pmax, vmax, tmax  # MIT 位域量程（发帧打包用）
        # 反馈帧解码常数（RX 线程热路径：q = _q_off + q_u * _q_s）
        self._q_off, self._q_s = -pmax, 2.0 * pmax / 65535.0
        self._dq_off, self._dq_s = -vmax, 2.0 * vmax / 4095.0
        self._tau_off, self._tau_s = -tmax, 2.0 * tmax / 4095.0
        # 预构建 U2CAN 前缀/整帧（高频路径零组帧开销）
        self.prefix = _u2can_prefix(slave_id)              # MIT 与使能/失能/设零/清错帧
        self.prefix_pos = _u2can_prefix(0x300 + slave_id)  # pos_force 帧
        self.prefix_vel = _u2can_prefix(0x200 + slave_id)  # vel 帧
        self.query_frame = (_u2can_prefix(_QUERY_HOST_ID)
                            + bytes((slave_id & 0xFF, (slave_id >> 8) & 0xFF, 0xCC, 0, 0, 0, 0, 0)) + b"\x00")
        # 状态快照（RX 线程单写者，单一 tuple 原子换入、读侧一次取引用）：
        #   (t, q, dq, tau, err, temp_mos, temp_rotor)；初值 t=0 即视为陈旧
        self.state = (0.0, 0.0, 0.0, 0.0, -1, -1, -1)
        self.state_event = threading.Event()  # 每收到状态帧置位（在岸确认/指令验证等阻塞事务用）
        # 最近参数应答槽（RX 线程写）：RID / 值 / 应答时刻（事务按时刻防陈旧洗白）
        self.param_rid, self.param_val, self.param_t = -1, None, 0.0


def _pack_mit(m: _DMMotor, tau: float, q: float, dq: float, kp: float, kd: float) -> bytes:
    """MIT 帧 8B 数据：``[q:16][dq:12][kp:12][kd:12][tau:12]``（量程按电机型号 + kp 0-500 / kd 0-5，越界就近钳位）。"""
    q_u = _f2u(q, -m.pmax, m.pmax, 16)
    dq_u = _f2u(dq, -m.vmax, m.vmax, 12)
    kp_u = _f2u(kp, 0.0, _KP_MAX, 12)
    kd_u = _f2u(kd, 0.0, _KD_MAX, 12)
    tau_u = _f2u(tau, -m.tmax, m.tmax, 12)
    return bytes((q_u >> 8, q_u & 0xFF, dq_u >> 4, ((dq_u & 0xF) << 4) | (kp_u >> 8),
                  kp_u & 0xFF, kd_u >> 4, ((kd_u & 0xF) << 4) | (tau_u >> 8), tau_u & 0xFF))


def _pack_pos(q: float, vlim: float, flim: float) -> bytes:
    """pos_force 帧 8B 数据：pos float32 + vel×100 uint16 + 电流标幺×10000 uint16（LE；入参基类已裁剪）。"""
    return struct.pack("<fHH", q, int(vlim * 100.0), int(flim * 10000.0))


def _pack_vel(dq: float) -> bytes:
    """vel 帧 8B 数据：速度 float32 + 4B 零填充。"""
    return struct.pack("<f", dq) + b"\x00" * 4


class _DMBus:
    """单条 U2CAN 串口总线（模块私有）：串口 + 发送锁 + RX 收取线程 + 帧分发 + 验证型参数/指令事务。"""

    def __init__(self, channel, baud: int, rx_hz: float) -> None:
        self._channel = channel
        self._baud = baud
        self._rx_period = 1.0 / rx_hz
        self._serial = None
        self._send_lock = threading.Lock()  # 串口写互斥（高频发送路径仅持锁数微秒）
        self._tx_lock = threading.Lock()    # 参数/指令验证事务串行化（低频；锁序恒为 tx → send，无死锁）
        self._rx_stop = threading.Event()
        self._rx_thread = None
        self._rx_buf = b""                  # 收取残尾（跨读拼接，恒 <16B）
        self._fb_map: dict = {}             # CAN id（master_id 与 slave_id 双键）→ 电机
        self._tx_event = threading.Event()  # 当前参数事务的应答到达置位
        self._tx_motor = None               # 当前等待参数应答的电机（与 _tx_rid 配对，RX 线程消费）
        self._tx_rid = -1

    # ---------------- 生命周期 ----------------
    def open(self) -> None:
        """打开串口并启动 RX 收取线程（daemon）。"""
        self._serial = serial.Serial(self._channel, self._baud, timeout=self._rx_period)
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True,
                                           name=f"backend-dm-rx（{self._channel}）")
        self._rx_thread.start()

    def close(self) -> None:
        """幂等关停：置停止标志 → join RX 线程（1s 超时）→ 关串口。"""
        self._rx_stop.set()
        th, self._rx_thread = self._rx_thread, None
        if th is not None:
            th.join(timeout=1.0)
        ser, self._serial = self._serial, None
        if ser is not None and ser.is_open:
            ser.close()

    @property
    def is_open(self) -> bool:
        """总线可用：串口已开**且** RX 收取线程存活（RX 死亡 = 僵尸总线，重连时须先关再重建）。"""
        return (self._serial is not None and self._serial.is_open
                and self._rx_thread is not None and self._rx_thread.is_alive())

    def register(self, motors) -> None:
        """注册电机应答路由（master_id 与 slave_id 双键；同总线键冲突即配置错误上抛）。"""
        for m in motors:
            for key in (m.master_id, m.slave_id):
                other = self._fb_map.get(key)
                if other is not None and other is not m:
                    raise ValueError(f"CAN id 0x{key:02X} 被多个电机占用（channel={self._channel}，配置错误）")
                self._fb_map[key] = m

    def unregister(self, motors) -> None:
        for m in motors:
            self._fb_map.pop(m.master_id, None)
            self._fb_map.pop(m.slave_id, None)

    # ---------------- 发送 ----------------
    def write(self, frame: bytes) -> None:
        """发送锁内写一帧完整 U2CAN 帧（30B）。"""
        with self._send_lock:
            self._serial.write(frame)

    # ---------------- 验证型事务（应答配对：电机 + RID + 应答时刻 ≥ 发送时刻，防陈旧洗白） ----------------
    def param_read(self, motor: _DMMotor, rid: int):
        """读参数寄存器：发 0x33 → 等应答 → 重试。

        :param motor: 目标电机。
        :param rid: 寄存器 RID。
        :return: 参数值（uint32 寄存器为 int，其余为 float）。
        :raises TimeoutError: 重试耗尽仍无应答。
        """
        lo, hi = motor.slave_id & 0xFF, (motor.slave_id >> 8) & 0xFF
        frame = _u2can_prefix(_QUERY_HOST_ID) + bytes((lo, hi, 0x33, rid, 0, 0, 0, 0)) + b"\x00"
        with self._tx_lock:
            for _ in range(_TX_RETRIES):
                self._tx_motor, self._tx_rid = motor, rid
                self._tx_event.clear()
                t0 = time.monotonic()
                self.write(frame)
                if self._tx_event.wait(_TX_TIMEOUT) and motor.param_t >= t0 and motor.param_rid == rid:
                    return motor.param_val
        raise TimeoutError(f"参数 RID{rid} 读取无应答（motor 0x{motor.slave_id:02X}）")

    def param_write(self, motor: _DMMotor, rid: int, value) -> None:
        """写参数寄存器：发 0x55 → 等回显核对（uint 精确 / float 容差）→ 重试。

        :raises TimeoutError: 重试耗尽仍无应答或回显不符。
        """
        is_uint = rid in _UINT_RIDS
        raw = struct.pack("<I" if is_uint else "<f", int(value) if is_uint else value)
        lo, hi = motor.slave_id & 0xFF, (motor.slave_id >> 8) & 0xFF
        frame = _u2can_prefix(_QUERY_HOST_ID) + bytes((lo, hi, 0x55, rid)) + raw + b"\x00"
        with self._tx_lock:
            for _ in range(_TX_RETRIES):
                self._tx_motor, self._tx_rid = motor, rid
                self._tx_event.clear()
                t0 = time.monotonic()
                self.write(frame)
                if self._tx_event.wait(_TX_TIMEOUT) and motor.param_t >= t0:
                    if _param_echo_ok(motor.param_val, value, is_uint):
                        return
        raise TimeoutError(f"参数 RID{rid} 写入未确认（motor 0x{motor.slave_id:02X}，"
                           f"目标 {value}，回显 {motor.param_val}）")

    def cmd_verified(self, motor: _DMMotor, cmd: int, expect) -> bool:
        """发使能/失能/设零/清错帧并验证：等新状态帧（超时补发一次 0xCC 查询）→ 按 expect(err) 核对 → 重试。

        :param cmd: 指令字节（0xFC 使能 / 0xFD 失能 / 0xFE 设零 / 0xFB 清错）。
        :param expect: error 码判定函数（使能 ``==1`` / 失能 ``!=1`` / 清错 ``<2``）；``None`` 仅链路 ACK。
        :return: 验证通过 True。
        :raises TimeoutError: 重试耗尽仍无新状态帧或核对不过。
        """
        frame = motor.prefix + b"\xff" * 7 + bytes((cmd,)) + b"\x00"
        with self._tx_lock:
            for _ in range(_TX_RETRIES):
                motor.state_event.clear()
                t0 = time.monotonic()
                self.write(frame)
                if not self._wait_state(motor, t0):  # 无应答 → 补发一次 0xCC 查询再等
                    motor.state_event.clear()
                    t0 = time.monotonic()
                    self.write(motor.query_frame)
                    if not self._wait_state(motor, t0):
                        continue
                err = motor.state[4]
                if expect is None or expect(err):
                    return True
        raise TimeoutError(f"指令 0x{cmd:02X} 未验证通过（motor 0x{motor.slave_id:02X}，err={motor.state[4]}）")

    @staticmethod
    def _wait_state(motor: _DMMotor, t0: float) -> bool:
        """等电机新状态帧（快照时刻 ≥ t0）至超时。"""
        return motor.state_event.wait(_TX_TIMEOUT) and motor.state[0] >= t0

    # ---------------- RX 收取线程 ----------------
    def _rx_loop(self) -> None:
        ser = self._serial
        while not self._rx_stop.is_set():
            try:
                chunk = ser.read(1)  # 阻塞至多一个节拍（timeout=1/rx_hz）
                n = ser.in_waiting  # 有数据即刻排空处理（延迟≈0，空闲时节拍即 rx_hz）
                if n:
                    chunk += ser.read(n)
            except Exception:
                if not self._rx_stop.is_set():
                    logger.warning("backend_dm.py - RX 收取线程：串口读取异常退出（channel=%s，重连可恢复）",
                                   self._channel)
                break
            if chunk:
                self._feed(chunk)

    def _feed(self, chunk: bytes) -> None:
        """残尾拼接 → 提取 16B 帧（头 AA、尾 55、CMD=0x11）→ 逐帧分发；残尾恒 <16B（垃圾字节逐个跳过）。"""
        buf = self._rx_buf + chunk
        i, end = 0, len(buf) - 16
        while i <= end:
            f = buf[i:i + 16]
            if f[0] == 0xAA and f[15] == 0x55 and f[1] == 0x11:
                self._dispatch((f[6] << 24) | (f[5] << 16) | (f[4] << 8) | f[3], f[7:15])
                i += 16
            else:
                i += 1
        self._rx_buf = buf[i:]

    def _dispatch(self, can_id: int, data: bytes) -> None:
        """按 CAN id 路由（can_id=0 时按 data[0] 低 4 位回退），按 data[2] 判别参数应答（0x33/0x55）与状态帧。

        状态帧 data[2] 恰为 0x33/0x55（dq 高字节，约 0.8% 概率）会被判为参数应答——仅少一次缓存刷新，无害；
        参数应答永不误判为状态帧（反向无污染）。
        """
        motor = self._fb_map.get(can_id) if can_id else self._fb_map.get(data[0] & 0x0F)
        if motor is None:
            return
        if data[2] == 0x33 or data[2] == 0x55:
            rid = data[3]
            motor.param_rid = rid
            motor.param_val = struct.unpack("<I" if rid in _UINT_RIDS else "<f", data[4:8])[0]
            motor.param_t = time.monotonic()
            if motor is self._tx_motor and rid == self._tx_rid:
                self._tx_event.set()
        else:
            q_u = (data[1] << 8) | data[2]
            dq_u = (data[3] << 4) | (data[4] >> 4)
            tau_u = ((data[4] & 0x0F) << 8) | data[5]
            motor.state = (time.monotonic(),
                           motor._q_off + q_u * motor._q_s,
                           motor._dq_off + dq_u * motor._dq_s,
                           motor._tau_off + tau_u * motor._tau_s,
                           data[0] >> 4, data[6], data[7])  # err=高 4 位；data[6]/[7]=T_MOS/T_Rotor（℃）
            motor.state_event.set()


class BackendDM(Backend):
    """DM 整机后端（型号说明见模块 docstring；arm 6 电机 + end 1 电机同协议，end 常为夹爪）。"""

    def __init__(self, cfg: dict) -> None:
        """先 ``super().__init__(cfg)`` 解析基类成员，再建电机表（无总线 I/O，总线在 ``_connect_*`` 打开）。

        :param cfg: yaml ``backend:`` 段整体（顶层 ``rx_hz`` 可配 RX 收取线程节拍，默认 1000Hz）。
        :raises ValueError: 型号不在量程表 / motor_id/feedback_id 缺失或非 int / feedback_id=0 / 组内 id 重复。
        """
        super().__init__(cfg)
        rx_hz = self._pos_float(self._cfg, "rx_hz", _RX_HZ_DEFAULT)
        if rx_hz > _RX_HZ_MAX:
            logger.warning("backend_dm.py - BackendDM.__init__：rx_hz=%s 过高（上限 %s Hz），已按上限执行",
                           rx_hz, _RX_HZ_MAX)
            rx_hz = _RX_HZ_MAX
        self._rx_hz = rx_hz
        self._motors_arm = self._build_motors("arm")  # 电机句柄表（下标对齐 _jointscfg_arm）
        self._motors_end = self._build_motors("end")
        self._bus_arm = None   # 本族当前使用的总线（connect 置位 / disconnect 清空）
        self._bus_end = None
        self._buses: dict = {}  # channel → 总线（arm/end 同 channel 共享一条）

    def close(self) -> None:
        """先停基类刷新线程，再关停全部总线（RX 线程 + 串口）；不可重启。"""
        super().close()
        for bus in list(self._buses.values()):
            bus.close()
        self._buses.clear()
        self._bus_arm = self._bus_end = None

    # ---------------- 私有助手 ----------------
    def _build_motors(self, family: str) -> list:
        """构建电机句柄表（型号 → 位域量程；只建对象不做总线 I/O），下标严格对齐 ``_jointscfg_{family}``。"""
        motors = []
        for i in range(self._n(family)):
            jn = self._jname(family, i)
            model = getattr(self, f"_joint_model_{family}")[i]
            try:
                pmax, vmax, tmax = _MODEL_LIMITS[str(model).lower()]
            except KeyError:
                raise ValueError(f"backend_dm.py - BackendDM.__init__：joint『{jn}』型号 {model!r} "
                                 f"不在 DM 型号量程表（可用：{sorted(_MODEL_LIMITS)}）")
            sid = getattr(self, f"_joint_motor_id_{family}")[i]
            mid = getattr(self, f"_joint_feedback_id_{family}")[i]
            if not isinstance(sid, int) or not isinstance(mid, int) or mid == 0:
                raise ValueError(f"backend_dm.py - BackendDM.__init__：joint『{jn}』motor_id/feedback_id "
                                 f"须为 int 且 feedback_id 非 0（得到 {sid!r} / {mid!r}）")
            motors.append(_DMMotor(sid, mid, pmax, vmax, tmax))
        ids = [k for m in motors for k in (m.slave_id, m.master_id)]
        if len(set(ids)) != len(ids):
            raise ValueError(f"backend_dm.py - BackendDM.__init__：{family} 组电机 motor_id/feedback_id 存在重复（配置错误）")
        return motors

    def _connect_family(self, family: str, channel) -> bool:
        """打开（或复用）channel 总线并启动 RX 线程（僵尸总线先关再重建），注册路由后逐电机 0xCC 在岸确认。"""
        bus = self._buses.get(channel)
        if bus is not None and not bus.is_open:  # 僵尸总线（串口已关或 RX 线程死亡）：先关再重建，避免同口双开
            bus.close()
            self._buses.pop(channel, None)
            bus = None
        if bus is None:
            baud = getattr(self, f"_baud_{family}")
            if not isinstance(baud, int) or baud <= 0:
                raise ValueError(f"baud_rate={baud!r} 非法（须为正整数）")
            bus = _DMBus(channel, baud, self._rx_hz)
            bus.open()
            self._buses[channel] = bus
        motors = getattr(self, f"_motors_{family}")
        setattr(self, f"_bus_{family}", bus)
        try:
            bus.register(motors)
            for i, m in enumerate(motors):
                m.state_event.clear()
                t0 = time.monotonic()
                bus.write(m.query_frame)
                if not _DMBus._wait_state(m, t0):
                    raise TimeoutError(f"在岸确认无应答（joint『{self._jname(family, i)}』"
                                       f"motor 0x{m.slave_id:02X}，channel={channel}）")
        except Exception:
            bus.unregister(motors)
            setattr(self, f"_bus_{family}", None)
            raise
        return True

    def _disconnect_family(self, family: str) -> None:
        """注销本族路由；另一族仍连着共享总线则保留总线本体，最后一族断开才真正关停；清扫无主总线（幂等）。"""
        bus = getattr(self, f"_bus_{family}")
        if bus is not None:
            setattr(self, f"_bus_{family}", None)
            bus.unregister(getattr(self, f"_motors_{family}"))
            other = "end" if family == "arm" else "arm"
            if not (getattr(self, f"_bus_{other}") is bus
                    and getattr(self, f"_is_connected_{other}") is not False):
                for ch, b in list(self._buses.items()):  # 最后一族断开：关停并移除本总线
                    if b is bus:
                        del self._buses[ch]
                        bus.close()
                        break
        for ch, b in list(self._buses.items()):  # 清扫无主总线（connect 中途失败遗留的重连缓存）
            if b is not self._bus_arm and b is not self._bus_end:
                del self._buses[ch]
                b.close()

    def _rid_of(self, key: str) -> int:
        """参数键 → RID（未知键上抛，基类转 warn）。"""
        try:
            return _PARAM_RIDS[key]
        except (KeyError, TypeError):
            raise ValueError(f"未知参数键 {key!r}（可用：{sorted(_PARAM_RIDS)}）")

    def _set_mode_core(self, family: str, i: int, mode: ControlMode) -> None:
        """写模式寄存器（回显验证+重试）；切 POSITION/VELOCITY 一并写 4 增益（NaN 跳过），MIT 不写。"""
        bus = getattr(self, f"_bus_{family}")
        motor = getattr(self, f"_motors_{family}")[i]
        bus.param_write(motor, _MODE_RID, _MODE_CODE[mode])
        if mode is ControlMode.MIT:
            return
        for key, rid in _GAIN_RIDS:
            gain = getattr(self, f"_{key}_{family}")[i]  # 基类已解析的 (n,) 增益成员（_pos_kp_arm 等）
            if not math.isnan(gain):
                bus.param_write(motor, rid, gain)

    def _read_state(self, family: str, i: int) -> dict:
        """非阻塞读单电机状态：新鲜（≤50ms）直返缓存；陈旧先发一帧 0xCC 刷新——软陈旧立即返当前快照
        （下一拍自愈），硬陈旧（>1s）先发刷新帧再上抛（电机在岸则下一拍自愈，不会永久锁死）。

        :raises TimeoutError: 状态 >1s 无刷新（电机掉线/接线异常，防冻结值洗白）。
        """
        motor = getattr(self, f"_motors_{family}")[i]
        st = motor.state
        age = time.monotonic() - st[0]
        if age > _STATE_STALE:
            getattr(self, f"_bus_{family}").write(motor.query_frame)
            if age > _STATE_DEAD:
                raise TimeoutError(f"状态 {age:.1f}s 无刷新（motor 0x{motor.slave_id:02X} 在岸/接线异常）")
        return {"q": st[1], "dq": st[2], "tau": st[3], "error": st[4],
                "temp_mos": st[5], "temp_rotor": st[6]}

    # ============================================================
    # 第一部分：生命周期——连接与使能（8 个内核）
    # ============================================================
    def _connect_arm(self, channel, protocol) -> bool:
        """连接 arm 总线（打开串口 + RX 线程 + 路由注册 + 逐电机在岸确认；``protocol`` 信息性忽略）。"""
        return self._connect_family("arm", channel)

    def _connect_end(self, channel, protocol) -> bool:
        """连接 end 总线（与 arm 同 channel 时复用同一条总线）。"""
        return self._connect_family("end", channel)

    def _disconnect_arm(self) -> None:
        """断开 arm：注销路由，共享总线按最后一族断开才关停的规则处理（幂等）。"""
        self._disconnect_family("arm")

    def _disconnect_end(self) -> None:
        """断开 end（同 :meth:`_disconnect_arm`）。"""
        self._disconnect_family("end")

    def _enable_joint_arm(self, i: int) -> bool:
        """使能 arm 第 ``i`` 关节（0xFC 帧 + 新状态帧核对 err==1，重试 3 次）。"""
        return self._bus_arm.cmd_verified(self._motors_arm[i], _CMD_ENABLE, lambda e: e == 1)

    def _enable_joint_end(self, i: int) -> bool:
        """使能 end 第 ``i`` 电机（同 :meth:`_enable_joint_arm`）。"""
        return self._bus_end.cmd_verified(self._motors_end[i], _CMD_ENABLE, lambda e: e == 1)

    def _disable_joint_arm(self, i: int) -> bool:
        """失能 arm 第 ``i`` 关节（0xFD 帧 + 新状态帧核对 err!=1——故障码在场仍视为已失能）。"""
        return self._bus_arm.cmd_verified(self._motors_arm[i], _CMD_DISABLE, lambda e: e != 1)

    def _disable_joint_end(self, i: int) -> bool:
        """失能 end 第 ``i`` 电机（同 :meth:`_disable_joint_arm`）。"""
        return self._bus_end.cmd_verified(self._motors_end[i], _CMD_DISABLE, lambda e: e != 1)

    # ============================================================
    # 第二部分：读取（6 个内核）
    # ============================================================
    def _read_joint_param_arm(self, i: int, key: str):
        """读 arm 第 ``i`` 关节参数寄存器（0x33 事务，键表 ``_PARAM_RIDS``）。"""
        return self._bus_arm.param_read(self._motors_arm[i], self._rid_of(key))

    def _read_joint_param_end(self, i: int, key: str):
        """读 end 第 ``i`` 电机参数寄存器。"""
        return self._bus_end.param_read(self._motors_end[i], self._rid_of(key))

    def _read_joint_mode_arm(self, i: int) -> Optional[ControlMode]:
        """读 arm 第 ``i`` 关节模式寄存器 RID10 → 基类枚举（DM 码 2/5/6/7 在枚举外 → ``None``）。"""
        return _CODE_MODE.get(self._bus_arm.param_read(self._motors_arm[i], _MODE_RID))

    def _read_joint_mode_end(self, i: int) -> Optional[ControlMode]:
        """读 end 第 ``i`` 电机模式寄存器 → 基类枚举。"""
        return _CODE_MODE.get(self._bus_end.param_read(self._motors_end[i], _MODE_RID))

    def _read_joint_state_arm(self, i: int) -> dict:
        """非阻塞读 arm 第 ``i`` 关节状态（新鲜直返 / 陈旧发 0xCC 后返当前快照，见 :meth:`_read_state`）。"""
        return self._read_state("arm", i)

    def _read_joint_state_end(self, i: int) -> dict:
        """非阻塞读 end 第 ``i`` 电机状态。"""
        return self._read_state("end", i)

    # ============================================================
    # 第三部分：写入（6 个内核）
    # ============================================================
    def _write_joint_param_arm(self, i: int, key: str, value: float) -> None:
        """写 arm 第 ``i`` 关节参数寄存器（0x55 事务 + 回显核对；只读键上抛拒绝）。"""
        if key in _READONLY_KEYS:
            raise ValueError(f"参数 {key!r} 只读")
        self._bus_arm.param_write(self._motors_arm[i], self._rid_of(key), value)

    def _write_joint_param_end(self, i: int, key: str, value: float) -> None:
        """写 end 第 ``i`` 电机参数寄存器。"""
        if key in _READONLY_KEYS:
            raise ValueError(f"参数 {key!r} 只读")
        self._bus_end.param_write(self._motors_end[i], self._rid_of(key), value)

    def _set_joint_mode_arm(self, i: int, mode: ControlMode) -> None:
        """设 arm 第 ``i`` 关节模式（写 RID10 验证；POSITION/VELOCITY 随写 4 增益，见 :meth:`_set_mode_core`）。"""
        self._set_mode_core("arm", i, mode)

    def _set_joint_mode_end(self, i: int, mode: ControlMode) -> None:
        """设 end 第 ``i`` 电机模式。"""
        self._set_mode_core("end", i, mode)

    def _set_joint_zero_arm(self, i: int) -> bool:
        """arm 第 ``i`` 关节设零（0xFE 帧 + 新状态帧链路 ACK）。"""
        return self._bus_arm.cmd_verified(self._motors_arm[i], _CMD_ZERO, None)

    def _set_joint_zero_end(self, i: int) -> bool:
        """end 第 ``i`` 电机设零。"""
        return self._bus_end.cmd_verified(self._motors_end[i], _CMD_ZERO, None)

    # ============================================================
    # 第四部分：发送（7 个内核，入参均已由基类校验裁剪——只组帧 + 锁内写串口）
    # ============================================================
    def _send_joint_mit_arm(self, i: int, tau: float, q: float, dq: float, kp: float, kd: float) -> None:
        """arm 第 ``i`` 关节 MIT 帧下发（位域打包见 :func:`_pack_mit`）。"""
        m = self._motors_arm[i]
        self._bus_arm.write(m.prefix + _pack_mit(m, tau, q, dq, kp, kd) + b"\x00")

    def _send_joint_mit_end(self, i: int, tau: float, q: float, dq: float, kp: float, kd: float) -> None:
        """end 第 ``i`` 电机 MIT 帧下发。"""
        m = self._motors_end[i]
        self._bus_end.write(m.prefix + _pack_mit(m, tau, q, dq, kp, kd) + b"\x00")

    def _send_joint_position_arm(self, i: int, q: float, vlim: float, flim: float) -> None:
        """arm 第 ``i`` 关节 pos_force 帧下发（pos float32 + vel×100 + 电流标幺×10000，见 :func:`_pack_pos`）。"""
        self._bus_arm.write(self._motors_arm[i].prefix_pos + _pack_pos(q, vlim, flim) + b"\x00")

    def _send_joint_position_end(self, i: int, q: float, vlim: float, flim: float) -> None:
        """end 第 ``i`` 电机 pos_force 帧下发。"""
        self._bus_end.write(self._motors_end[i].prefix_pos + _pack_pos(q, vlim, flim) + b"\x00")

    def _send_joint_vel_arm(self, i: int, dq: float) -> None:
        """arm 第 ``i`` 关节 vel 帧下发（速度 float32，见 :func:`_pack_vel`）。"""
        self._bus_arm.write(self._motors_arm[i].prefix_vel + _pack_vel(dq) + b"\x00")

    def _send_joint_vel_end(self, i: int, dq: float) -> None:
        """end 第 ``i`` 电机 vel 帧下发。"""
        self._bus_end.write(self._motors_end[i].prefix_vel + _pack_vel(dq) + b"\x00")

    def _send_action_end(self, action: str, **kwargs) -> None:
        """末端离散动作（DM 末端 = 夹爪电机，动作在 POSITION/pos_force 模式下执行）。

        动作集：``"open"``→-5.0、``"home"``→-3.0、``"close"``→0.0（rad 固定行程位）、
        ``"position"``→按需目标位置（kwarg ``position``，rad，JoyArm 层透传）。
        模式不一致时自动切 POSITION（随写增益寄存器）；运动限速/限流取 cfg ``POS_VEL`` 的
        ``vlim/flim`` 发送默认成员（缺配置被基类空值门禁拦截、warn 不发送）；
        位置经 ``send_position_end`` 基类管道裁硬限位（越限就近裁剪 + warn）后整组下发。

        :param action: 动作名（open / home / close / position）。
        :param kwargs: ``position`` 动作的目标位置（rad）。
        :raises ValueError: 未知动作、或 position 动作缺 ``position`` 参数（基类转 warn）。
        :raises RuntimeError: 切 POSITION 模式失败（基类转 warn）。
        """
        if action == "position":
            pos = kwargs.get("position")
            if pos is None:
                raise ValueError("position 动作需提供 position 参数（rad）")
        else:
            pos = _END_ACTION_POS.get(action)
            if pos is None:
                raise ValueError(f"未知动作 {action!r}（可用：open / home / close / position）")
        if self._mode_end is not ControlMode.POSITION:  # 动作保证位置模式：不一致自动切换（含增益写入）
            if self.set_mode_end(ControlMode.POSITION) != 1:
                raise RuntimeError("末端动作前置：切 POSITION 模式失败")
        n = self._n("end")
        self.send_position_end([pos] * n, self._vlim_default_end, self._flim_default_end)

    # ============================================================
    # 第五部分：清错（2 个内核）
    # ============================================================
    def _clear_joint_error_arm(self, i: int) -> bool:
        """清 arm 第 ``i`` 关节错误（0xFB 清错帧 + 新状态帧核对 err<2；清错后自动失能属正常）。"""
        return self._bus_arm.cmd_verified(self._motors_arm[i], _CMD_CLEAR, lambda e: e < 2)

    def _clear_joint_error_end(self, i: int) -> bool:
        """清 end 第 ``i`` 电机错误。"""
        return self._bus_end.cmd_verified(self._motors_end[i], _CMD_CLEAR, lambda e: e < 2)
