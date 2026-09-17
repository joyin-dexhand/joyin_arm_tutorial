"""``BackendDM`` —— joyarm_dm 真机后端。

USB-CAN 串口桥（如 ``/dev/ttyACM0``，921600）驱动整机 7 个 DM 电机：本体 6 关节（joint1~3 = 4340P，joint4~6 = 4310）
+ 两指夹爪（``gripper``，4310）。DM 通讯协议参照 ``u2can/``（厂商参考库）的原理**重新实现**。

**总线共享规则**（config ``arm.channel`` 与 ``end.channel`` 决定）::

    channel 相同 → 共享单总线：一个串口句柄 + 一个 RX 读线程 + 一个 TX 锁，end 借用 arm 的总线；
    channel 不同 → 两条独立总线，互不干扰（支持末端独立通道的硬件形态）。

**DM 协议要点**（详见各常量/函数处注释）::

    发送帧（串口桥 → CAN）  30 字节定长模板，仅 [13:15]=CAN ID（LE）与[21:29]=8 字节载荷随指令变化；
    应答帧（CAN → 串口桥）  16 字节定长：[0]=0xAA、[15]=0x55、[1]=CMD（0x11）、[3:7]=CAN ID（LE32）、[7:15]=8 字节载荷；
    CAN ID 仲裁             MIT 指令=SlaveID；POS_VEL=0x100+SlaveID；VEL=0x200+SlaveID；使能/失能/标零=SlaveID +
                            ``FF×7+cmd``（0xFC/0xFD/0xFE）；参数通道=0x7FF +载荷内嵌 [slave_id_l, slave_id_h, 子码, RID, 值]，
                            子码 0x33 读 / 0x55 写 / 0xAA 存闪存 / 0xCC 状态请求；
    一发一收                每个指令帧/请求帧触发一帧应答，状态仅在收发后刷新；
    状态帧载荷              D0 高 4 位=错误码、D0 低 4 位=ID，D1~2=位置(16bit)，D3~4=速度(12bit)，D4~5=力矩(12bit)，D6~7=驱动板/转子
                            温度（各 1 字节 ℃，旧固件未用恒 0）；
    错误码语义              0=失能正常，1=使能正常，其余（8 超压/9 欠压/A 过流/B MOS超温/C 线圈超温/D 通信丢失/E 过载）=故障。

**实现结构**（本文件内三层，``DmMotor``/``DmCanBus`` 为私有协议层、不导出）::

    协议常量 + 编解码纯函数   帧封装/提取、MIT 位打包、状态解包、float↔uint；
    DmMotor                   单电机配置（config joint 条目）+ 状态/参数槽 + 事件；
    DmCanBus                  串口桥总线：TX 锁 + RX 守护线程 + 帧分发 + 电机原语；
    BackendDM                 :class:`~joyarm_core.backend.backend.Backend` 适配层。

时序约束（依 u2can 标定）：控制帧间隔建议 ≥1ms；使能应答 ~100ms 内；参数读写带重试确认（写回读比对）；
标零前须失能且反馈无故障；存闪存（0xAA）前必须失能。
"""
from __future__ import annotations

import logging
import threading
import time
from struct import pack, unpack
from typing import Optional

import numpy as np

from ..utils.types import ArmState, ControlMode, JointState
from .backend import Backend

__all__ = ["BackendDM"]

logger = logging.getLogger("joyarm_core.backend_dm")


# ============================================================
# 协议常量
# ============================================================
# 型号 → (PMAX rad, VMAX rad/s, TMAX N·m, KP_MAX, KD_MAX)：MIT 打包与状态解包的线性标度极限；
# kp/kd 为 MIT 帧 12bit 定点编码量程（随型号可异，超量程会被就近钳位并告警）；
# 仅收录本项目所用型号（4340P 为 4340 的命名变体，限值相同）
_MOTOR_LIMITS: dict[str, tuple[float, float, float, float, float]] = {
    "4310": (12.5, 30.0, 10.0, 500.0, 5.0),
    "4340": (12.5, 10.0, 28.0, 500.0, 5.0),
    "4340P": (12.5, 10.0, 28.0, 500.0, 5.0),
}

# 串口 CAN 桥默认波特率（config 缺 baud_rate 键时的回退值）
_DEFAULT_BAUD_RATE: int = 921600

# 接收残余缓冲上限（字节）与截断告警节流间隔（秒）：见 _extract_frames
_RX_RESIDUAL_MAX = 4096
_RX_CAP_WARN_INTERVAL = 5.0
_RX_CAP_WARN_LAST = float("-inf")

# kp/kd 越量程钳位告警节流间隔（秒）：见 _pack_mit（200Hz 指令流防刷屏）
_MIT_RANGE_WARN_INTERVAL = 0.5
_MIT_RANGE_WARN_LAST = float("-inf")

# 错误码：0=失能正常、1=使能正常、8~E=故障（依达妙协议：8 超压/9 欠压/A 过流/
# B MOS超温/C 线圈超温/D 通信丢失/E 过载）
_ERR_DISABLED, _ERR_ENABLED = 0, 1

# 故障码 → 名称（read_state 的 errors 列表可读化；仅 8~E）
_ERR_FAULT_NAMES: dict[int, str] = {
    8: "超压", 9: "欠压", 10: "过流", 11: "MOS超温",
    12: "线圈超温", 13: "通信丢失", 14: "过载",
}

# 电机寄存器（RID）：10=控制模式（1 MIT / 2 POS_VEL / 3 VEL），25~28=POS_VEL 闭环增益，
# 另收录保护阈值/运动参数/版本身份/物理特性（全集见 u2can/DM_CAN.py 的 DM_variable）
_RID_CTRL_MODE = 10

# 参数名（基类/用户侧字符串 key）→ DM 寄存器 RID
_PARAM_RIDS: dict[str, int] = {
    # 控制与闭环增益（可写）
    "ctrl_mode": 10,   # 1=MIT / 2=POS_VEL / 3=VEL（uint32）
    "vel_kp": 25,      # 速度环 Kp
    "vel_ki": 26,      # 速度环 Ki
    "pos_kp": 27,      # 位置环 Kp
    "pos_ki": 28,      # 位置环 Ki
    # 保护阈值（可写，慎改）
    "uv": 0,           # 欠压阈值 V
    "ot": 2,           # 过温阈值 ℃
    "oc": 3,           # 过流阈值 A
    "ov": 29,          # 过压阈值 V
    "timeout": 9,      # CAN 掉线保护超时（uint32）
    # 运动参数（可写）
    "acc": 4,          # 加速时间
    "dec": 5,          # 减速时间
    "max_spd": 6,      # 最大速度 rad/s
    # 版本身份（只读，uint32）
    "hw_ver": 13,
    "sw_ver": 14,
    "sn": 15,
    "sub_ver": 36,
    # 物理特性（只读）
    "kt": 1,           # 转矩常数
    "gr": 20,          # 减速比
    "pmax": 21,        # 位置量程极限 rad
    "vmax": 22,        # 速度极限 rad/s
    "tmax": 23,        # 力矩极限 N·m
}

# 只读参数名（版本/序列号/出厂物理特性事实量）：write_param_* 拒绝写入
_READONLY_KEYS = frozenset(
    {"hw_ver", "sw_ver", "sn", "sub_ver", "kt", "gr", "pmax", "vmax", "tmax"}
)

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
              limits: tuple[float, float, float, float, float]) -> bytes:
    """打包 MIT 指令帧载荷（8 字节）：``q16 | dq12 | kp12 | kd12 | tau12`` 共 64bit。

    kp/kd 超出该型号 12bit 编码量程（``0 ≤ kp ≤ kp_max``、``0 ≤ kd ≤ kd_max``，
    即 ``_MOTOR_LIMITS`` 型号表第 4/5 元）时就近钳位并节流告警——定点编码
    本身会静默截断，此处显式留痕防"增益名不副实"。
    """
    global _MIT_RANGE_WARN_LAST
    pmax, vmax, tmax, kp_max, kd_max = limits
    kp_c, kd_c = _clamp(kp, 0.0, kp_max), _clamp(kd, 0.0, kd_max)
    now = time.monotonic()
    clipped = [f"{n}={v:g} → {c:g}（量程 0~{hi:g}）"
               for n, v, c, hi in (("kp", kp, kp_c, kp_max), ("kd", kd, kd_c, kd_max))
               if v != c]
    if clipped and now - _MIT_RANGE_WARN_LAST >= _MIT_RANGE_WARN_INTERVAL:
        _MIT_RANGE_WARN_LAST = now
        logger.warning(
            "backend_dm.py - _pack_mit：kp/kd 超出 MIT 帧编码量程，已就近钳位"
            f"：{'；'.join(clipped)}；实际生效以钳位后为准，请核对 config 增益"
            "或指令参数")
    q_u = _float_to_uint(q, -pmax, pmax, 16)
    dq_u = _float_to_uint(dq, -vmax, vmax, 12)
    kp_u = _float_to_uint(kp_c, 0.0, kp_max, 12)
    kd_u = _float_to_uint(kd_c, 0.0, kd_max, 12)
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


def _unpack_status(data: bytes, limits: tuple[float, float, float, float, float]):
    """解包状态帧载荷（8 字节）→ ``(q, dq, tau, err, t_mos, t_rotor)``。

    D6~7 为驱动板 MOS / 转子温度（1 字节，℃）；旧固件该两字节未用（恒 0）。
    """
    pmax, vmax, tmax = limits[:3]
    err = (data[0] >> 4) & 0x0F
    q = _uint_to_float((data[1] << 8) | data[2], -pmax, pmax, 16)
    dq = _uint_to_float((data[3] << 4) | (data[4] >> 4), -vmax, vmax, 12)
    tau = _uint_to_float(((data[4] & 0xF) << 8) | data[5], -tmax, tmax, 12)
    return float(q), float(dq), float(tau), int(err), int(data[6]), int(data[7])


def _pack_tx(can_id: int, data: bytes) -> bytes:
    """封装串口桥发送帧（30 字节定长模板，仅 CAN ID 与载荷段可变）。"""
    if len(data) != 8:
        # 切片赋值 f[21:29] 不保长：短载荷会把 30 字节帧缩成 <30，桥接固件丢帧
        raise ValueError(f"backend_dm.py - _pack_tx：CAN 载荷须为 8 字节，收到 {len(data)}")
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
    残余超过 ``_RX_RESIDUAL_MAX`` 时丢弃更早的字节只留尾部——持续收到无有效
    帧的乱码（如波特率不匹配）时防止缓冲无限增长；未提取的帧早已不可能在
    被丢弃的字节里成帧，尾部足以接住跨读的半帧。
    """
    frames, i, last = [], 0, 0
    while i <= len(buf) - 16:
        if buf[i] == 0xAA and buf[i + 15] == 0x55:
            frames.append(buf[i:i + 16])
            i += 16
            last = i
        else:
            i += 1
    residual = buf[last:]
    global _RX_CAP_WARN_LAST
    if len(residual) > _RX_RESIDUAL_MAX:
        residual = residual[-_RX_RESIDUAL_MAX:]
        now = time.monotonic()
        if now - _RX_CAP_WARN_LAST >= _RX_CAP_WARN_INTERVAL:
            _RX_CAP_WARN_LAST = now
            logger.warning("backend_dm.py - _extract_frames：接收残余已超 "
                           f"{_RX_RESIDUAL_MAX} 字节并截断保尾；总线持续收到无法"
                           "解析的数据，请检查波特率/串口桥接是否匹配")
    return frames, residual


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
    ``MIT`` / ``POS_VEL`` 段回退源）、关节限位（``q_min``/``q_max``/
    ``dq_max``/``tau_max``，四键 arm/end 同构必配，齐全性由
    ``JoyArm.check_config`` 把关；q_min/q_max/dq_max 供 end 行程语义复用，
    tau_max 供上层读取，arm 守卫走基类限位），以及 RX 线程回填的状态槽与
    参数槽（配到达 Event，供"发请求 → 等应答"同步）。
    """

    def __init__(self, jcfg: dict) -> None:
        self.name = str(jcfg.get("name", "motor"))
        for key in ("motor_id", "feedback_id"):
            v = jcfg.get(key)
            try:
                vi = int(v)
            except (TypeError, ValueError):
                vi = None
            if vi is None or not 0 <= vi <= 0x1FFFFFFF:
                raise ValueError(
                    f"backend_dm.py - DmMotor.__init__：关节 {self.name} 的 "
                    f"{key}={v!r} 缺失或非法（CAN ID 须为非负整数；"
                    f"现有键：{sorted(jcfg)}）")
        self.motor_id = int(jcfg["motor_id"])
        self.feedback_id = int(jcfg["feedback_id"])
        model = str(jcfg.get("model", ""))
        if model not in _MOTOR_LIMITS:
            raise ValueError(
                f"backend_dm.py - DmMotor.__init__：关节 {self.name} 电机型号 {model!r} "
                f"不受支持；可用：{sorted(_MOTOR_LIMITS)}"
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
        if self.vlim <= 0.0:
            # POS_VEL/VEL 指令的限速上限为 0 → 电机收到指令也不会动
            logger.warning("backend_dm.py - DmMotor.__init__：关节 %s 未配置 "
                           "POS_VEL.vlim（速度上限），位置/速度指令将以速度 0 "
                           "下发（电机不动作），请在 config 补该键", self.name)

        # 关节限位（四键 arm/end 同构必配，check_config 把关齐全性；
        # q_min/q_max/dq_max 供 end 行程语义复用，tau_max 供上层读取——
        # arm 守卫走基类 limits_from_joint_cfgs）
        self.q_min = None if jcfg.get("q_min") is None else float(jcfg["q_min"])
        self.q_max = None if jcfg.get("q_max") is None else float(jcfg["q_max"])
        self.dq_max = None if jcfg.get("dq_max") is None else float(jcfg["dq_max"])
        self.tau_max = None if jcfg.get("tau_max") is None else float(jcfg["tau_max"])

        # 状态槽 + 参数槽（RX 线程写、指令线程读，Event 通知应答到达）。
        # 槽字段无锁：GIL 下单字段读写原子，但读侧可能混到相邻两帧的数值（如 q 取第 N 帧、tau 取第 N+1 帧）；
        # 需要精确一致性时走 read_state_* 请求-应答路径（Event 同步保证成帧完整）。
        self.q, self.dq, self.tau, self.err = 0.0, 0.0, 0.0, _ERR_DISABLED
        self.t_mos, self.t_rotor = 0, 0  # ℃，旧固件不反馈（恒 0）
        self.t_state = 0.0              # 最近状态应答时刻（monotonic；0=从未收到）
        self._state_event = threading.Event()
        self.params: dict[int, float | int] = {}
        self._param_events: dict[int, threading.Event] = {}
        self.bus: Optional[DmCanBus] = None  # connect 注册时回填

    # ---- 状态槽 ----
    def update_state(self, q: float, dq: float, tau: float, err: int,
                     t_mos: int, t_rotor: int) -> None:
        """回填状态槽并置位应答事件（RX 线程调用）。"""
        self.q, self.dq, self.tau, self.err = q, dq, tau, err
        self.t_mos, self.t_rotor = t_mos, t_rotor
        self.t_state = time.monotonic()
        self._state_event.set()

    def clear_state(self) -> None:
        """清状态应答事件（发请求帧前调用，防 wait 读到旧应答）。"""
        self._state_event.clear()

    def wait_state(self, timeout: float) -> bool:
        """等待状态应答到达（发请求帧前先 :meth:`clear_state`）。"""
        return self._state_event.wait(timeout)

    # ---- 参数槽 ----
    def update_param(self, rid: int, value) -> None:
        """回填参数槽并置位该 RID 应答事件（RX 线程调用）。"""
        self.params[rid] = value
        self._param_events.setdefault(rid, threading.Event()).set()

    def clear_param(self, rid: int) -> None:
        """清参数应答事件并丢弃旧值（发请求帧前调用）。"""
        self.params.pop(rid, None)
        # 事件常驻（同 _state_event 的 clear 而非移除）：否则发请求后事件不存在，
        # wait_param 无法在应答到达前阻塞等待，参数读写必然超时
        self._param_events.setdefault(rid, threading.Event()).clear()

    def wait_param(self, rid: int, timeout: float) -> bool:
        """等待参数应答到达（发请求帧前先 :meth:`clear_param`）。"""
        return self._param_events.setdefault(rid, threading.Event()).wait(timeout)


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

    def __init__(self, channel: str, baud_rate: int = _DEFAULT_BAUD_RATE) -> None:
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
        """打开串口并启动 RX 守护线程（已打开则幂等返回）。"""
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
        """停 RX 线程并关闭串口（幂等；共享总线只关一次，由 disconnect 统一调度）。"""
        self._stop = True
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def add_motor(self, motor: DmMotor) -> None:
        """注册电机到三个分发表（feedback_id / 低 4 位回退 / motor_id 参数通道）。"""
        motor.bus = self
        self._by_fid[motor.feedback_id] = motor
        self._by_fid_low[motor.feedback_id & 0x0F] = motor
        self._by_mid[motor.motor_id] = motor

    # ---- 收发 ----
    def send(self, can_id: int, data: bytes) -> None:
        """发送一帧 CAN 报文（30 字节桥帧封装，TX 锁串行化）。"""
        if self._ser is None:
            raise RuntimeError(
                f"backend_dm.py - DmCanBus.send：总线 {self.channel} 已关闭"
                f"（close/disconnect 之后不可发送），请先 connect() 再发送")
        frame = _pack_tx(can_id, bytes(data))
        with self._tx_lock:
            self._ser.write(frame)

    def _rx_loop(self) -> None:
        while not self._stop:
            try:
                data = self._ser.read(4096)
            except Exception as e:
                # 串口关闭/拔出：线程退出，此后所有应答等待将超时——必须留痕可排查
                logger.warning("RX 线程退出（%s）：%s", self.channel, e)
                break
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
        """下发 MIT 指令帧（CAN ID = motor_id，8 字节位打包）。"""
        self.send(m.motor_id, _pack_mit(q, dq, tau, kp, kd, m.limits))

    def send_pos_vel(self, m: DmMotor, p: float, v: float) -> None:
        """下发 POS_VEL 位置速度帧（CAN ID = 0x100 + motor_id）。"""
        self.send(0x100 + m.motor_id, pack("<ff", p, v))

    def send_vel(self, m: DmMotor, v: float) -> None:
        """下发纯速度帧（CAN ID = 0x200 + motor_id；速度 float32 + 4 字节补零）。"""
        # 速度 float32 占前 4 字节、后 4 字节补零（对齐厂商 control_Vel 布局）
        self.send(0x200 + m.motor_id, pack("<f", v) + b"\x00" * 4)

    # ---- 电机原语：使能/失能/标零/状态请求 ----
    def enable(self, m: DmMotor) -> None:
        """使能电机（``FF×7 + 0xFC``）。"""
        self.send(m.motor_id, b"\xff" * 7 + b"\xfc")

    def disable(self, m: DmMotor) -> None:
        """失能电机（``FF×7 + 0xFD``）。"""
        self.send(m.motor_id, b"\xff" * 7 + b"\xfd")

    def set_zero(self, m: DmMotor) -> None:
        """零位标定帧（``FF×7 + 0xFE``；须先失能且反馈无故障）。"""
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
        raise TimeoutError(
            f"backend_dm.py - read_param：电机 {m.name} 读参数 RID={rid} "
            f"无应答（{self.channel}）")

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
        raise RuntimeError(f"backend_dm.py - write_param：电机 {m.name} 写参数 RID={rid}={value} 未确认")

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
        ``POS_VEL`` 与四键限位 ``q_min`` / ``q_max`` / ``dq_max`` /
        ``tau_max``——arm/end 同构必配，守卫限位由基类自 cfg 解析，本类
        DmMotor 的 q_min/q_max/dq_max 供 end 行程语义复用，tau_max 供上层
        读取）。
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._arm_cfg = cfg.get("arm") or {}
        self._end_cfg = cfg.get("end") or {}
        self._arm_motors = [DmMotor(j) for j in self._arm_cfg.get("joints") or []]
        self._end_motors = [DmMotor(j) for j in self._end_cfg.get("joints") or []]
        self._n = len(self._arm_motors)
        self._buses: dict[str, DmCanBus] = {}  # channel → 总线（同 channel 共享）
        self._mode_arm: dict[str, ControlMode] = {}  # 电机名 → 已切换模式（空=未设置）
        self._mode_end: dict[str, ControlMode] = {}

    # ----------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------
    def _check_open(self) -> None:
        if not self._buses:
            raise RuntimeError("backend_dm.py - _check_open：BackendDM 未连接；请先 connect()")

    def _motors_for(self, joint: Optional[int]) -> list[DmMotor]:
        if joint is None:
            return self._arm_motors
        if not 0 <= joint < self._n:
            raise ValueError(f"backend_dm.py - _motors_for：关节索引 {joint} 超出 [0, {self._n})")
        return [self._arm_motors[joint]]

    def _rows(self, cmd, name: str, joint: Optional[int]) -> list[tuple[DmMotor, float]]:
        """指令数组 → (电机, 标量值) 对齐列表（维度校验）。"""
        motors = self._motors_for(joint)
        a = np.asarray(cmd, dtype=float).reshape(-1)
        if a.size != len(motors):
            raise ValueError(f"backend_dm.py - _rows：{name} 维度 {a.size} ≠ 关节数 {len(motors)}")
        return list(zip(motors, a))

    def _require_mode_arm(self, mode: ControlMode, motors: list[DmMotor]) -> None:
        """指令前置：已连接（未连接报清晰错误，而非误导性的模式错误）。"""
        self._check_open()
        bad = [m.name for m in motors if self._mode_arm.get(m.name) != mode]
        if bad:
            raise RuntimeError(
                f"backend_dm.py - _require_mode_arm：关节 {bad} 未处于 {mode.value} 模式；"
                f"请先 set_mode_arm({mode.value})")

    def _end_motors_for(self, joint: Optional[int]) -> list[DmMotor]:
        """取末端电机子集（``joint=None`` 全部）；未配置末端时抛 ``RuntimeError``。"""
        if not self._end_motors:
            raise RuntimeError("backend_dm.py - _end_motors_for：本型号未配置末端（config backend.end 缺失）")
        if joint is None:
            return self._end_motors
        if not 0 <= joint < len(self._end_motors):
            raise ValueError(
                f"backend_dm.py - _end_motors_for：末端电机索引 {joint} "
                f"超出 [0, {len(self._end_motors)})")
        return [self._end_motors[joint]]

    def _require_mode_end(self, mode: ControlMode, motors: list[DmMotor]) -> None:
        """指令前置：已连接（未连接报清晰错误，而非误导性的模式错误）。"""
        self._check_open()
        bad = [m.name for m in motors if self._mode_end.get(m.name) != mode]
        if bad:
            raise RuntimeError(
                f"backend_dm.py - _require_mode_end：末端电机 {bad} 未处于 {mode.value} 模式；"
                f"请先 set_mode_end({mode.value})")

    @staticmethod
    def _end_speed(m: DmMotor) -> float:
        """末端 POS_VEL 指令速度：config ``vlim``，不超过电机 ``dq_max``。"""
        return min(m.vlim, m.dq_max) if m.dq_max is not None else m.vlim

    @staticmethod
    def _values_for(cmd, name: str, motors: list[DmMotor]) -> np.ndarray:
        """连续量指令/参数值 → 与所选电机数对齐的向量（标量广播 / 序列等长校验）。"""
        a = np.asarray(cmd, dtype=float).reshape(-1)
        if a.size == 1:
            return np.full(len(motors), float(a[0]))
        if a.size != len(motors):
            raise ValueError(
                f"backend_dm.py - _values_for：{name} 维度 {a.size} "
                f"≠ 所选电机数 {len(motors)}")
        return a

    @staticmethod
    def _vec_values(v, name: str, motors: list[DmMotor],
                    default_per_motor=None) -> np.ndarray:
        """指令向量 → 与所选电机数对齐的向量（``None`` 回退逐电机默认值，如 config 增益）。"""
        if v is None:
            return np.array(default_per_motor, dtype=float)
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size != len(motors):
            raise ValueError(
                f"backend_dm.py - _vec_values：{name} 维度 {a.size} "
                f"≠ 所选电机数 {len(motors)}")
        return a

    def _send_mit_group(self, motors: list[DmMotor], q, dq, tau_ff,
                        kp, kd) -> None:
        """MIT 指令逐电机下发（arm/end 共用；``kp/kd`` 为 ``None`` 时回退 config 增益）。"""
        q_v = self._vec_values(q, "q", motors)
        dq_v = self._vec_values(dq, "dq", motors)
        tau_v = self._vec_values(tau_ff, "tau_ff", motors)
        kp_v = self._vec_values(kp, "kp", motors, [m.mit_kp for m in motors])
        kd_v = self._vec_values(kd, "kd", motors, [m.mit_kd for m in motors])
        for m, qi, dqi, ti, kpi, kdi in zip(motors, q_v, dq_v, tau_v, kp_v, kd_v):
            m.bus.send_mit(m, qi, dqi, ti, kpi, kdi)

    def _read_params(self, motors: list[DmMotor], key: str) -> list:
        """参数名校验 + 逐电机读（0x33 通道，arm/end 共用）。"""
        self._check_open()
        rid = _PARAM_RIDS.get(key)
        if rid is None:
            raise ValueError(
                f"backend_dm.py - _read_params：未知参数名 {key!r}；可用：{sorted(_PARAM_RIDS)}")
        return [m.bus.read_param(m, rid) for m in motors]

    def _write_params(self, motors: list[DmMotor], key: str, value,
                      persist: bool) -> None:
        """参数名/只读校验 + 逐电机写确认（0x55 通道，arm/end 共用）。"""
        self._check_open()
        rid = _PARAM_RIDS.get(key)
        if rid is None:
            raise ValueError(
                f"backend_dm.py - _write_params：未知参数名 {key!r}；可用：{sorted(_PARAM_RIDS)}")
        if key in _READONLY_KEYS:
            raise ValueError(
                f"backend_dm.py - _write_params：参数 {key!r} 只读；"
                f"只读参数：{sorted(_READONLY_KEYS)}")
        for m, v in zip(motors, self._values_for(value, "value", motors)):
            m.bus.write_param(m, rid, v)
            if persist:
                m.bus.save_params(m)  # 自动失能并保持失能（DM 存闪存硬约束）

    def _enable_motors(self, motors: list[DmMotor]) -> None:
        """逐电机使能：清事件 → 发使能帧 → 等应答并校验错误码。"""
        for m in motors:
            m.clear_state()
            m.bus.enable(m)
            if not m.wait_state(0.5):
                raise RuntimeError(f"backend_dm.py - _enable_motors：电机 {m.name} 使能无应答")
            if m.err != _ERR_ENABLED:
                fault = ("仍处于失能态（错误码 0x0，非故障；使能帧可能未生效）"
                         if m.err == _ERR_DISABLED
                         else f"故障（{_ERR_FAULT_NAMES.get(m.err, '未知')}({m.err:#x})）")
                raise RuntimeError(
                    f"backend_dm.py - _enable_motors：电机 {m.name} 使能失败（{fault}）")

    def _set_zero_motors(self, motors: list[DmMotor]) -> None:
        """标零流程：失能 → 轮询反馈至无故障（错误码 0/1）→ 发标零帧。"""
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
                raise RuntimeError(
                    f"backend_dm.py - _set_zero_motors：电机 {m.name} 标零前存在故障"
                    f"（{_ERR_FAULT_NAMES.get(m.err, '故障')}({m.err:#x})）"
                )
            m.clear_state()
            m.bus.set_zero(m)
            if not m.wait_state(0.5):
                raise RuntimeError(f"backend_dm.py - _set_zero_motors：电机 {m.name} 标零无应答")

    def _clear_fault_motors(self, motors: list[DmMotor]) -> None:
        """验证式清错：失能帧（DM 协议清错手段）→ 轮询核对故障码清除 → 重新
        使能并核对使能态；逐电机 best-effort，收尾汇总未恢复项。"""
        bad: list[str] = []
        for m in motors:
            fault = _ERR_FAULT_NAMES.get(m.err, "故障")   # 槽内旧值仅作初值兜底
            m.bus.disable(m)                      # ① 失能 = DM 协议的故障清除
            cleared = False
            for _ in range(10):
                m.clear_state()
                m.bus.refresh(m)
                if m.wait_state(0.05):
                    fault = _ERR_FAULT_NAMES.get(m.err, "故障")   # 以最新应答为准
                    if m.err in (_ERR_ENABLED, _ERR_DISABLED):
                        cleared = True
                        break
            if not cleared:
                bad.append(f"{m.name}：故障未清除（{fault}({m.err:#x})，可能需断电排查）")
                continue
            try:                                   # ② 重新使能并核对
                m.clear_state()
                m.bus.enable(m)
                if (not m.wait_state(0.5)) or m.err != _ERR_ENABLED:
                    bad.append(f"{m.name}：使能未恢复（err={m.err:#x}）")
            except Exception as e:
                bad.append(f"{m.name}：使能失败（{e}）")
        if bad:
            raise RuntimeError(
                "backend_dm.py - clear_fault：部分电机未恢复：\n"
                + "\n".join(f"  - {b}" for b in bad))

    def _switch_group_mode(self, motors: list[DmMotor], mode: ControlMode) -> None:
        """逐电机切模式：POSITION 先写 POS_VEL 闭环增益（RID25~28）再切模式确认。"""
        dm_mode = _DM_MODE.get(mode)
        if dm_mode is None:
            raise ValueError(f"backend_dm.py - _switch_group_mode：未知控制模式：{mode}")
        for m in motors:
            if mode == ControlMode.POSITION:
                for key in ("vel_kp", "vel_ki", "pos_kp", "pos_ki"):
                    m.bus.write_param(m, _PARAM_RIDS[key], m.gains[key])
            m.bus.switch_mode(m, dm_mode)

    def _set_group_mode(self, motors: list[DmMotor], cache: dict,
                        mode: ControlMode) -> None:
        """按模式缓存切电机组模式（arm/end 共用）：仅对未到位的电机写增益/模式。"""
        need = [m for m in motors if cache.get(m.name) != mode]
        if need:
            self._switch_group_mode(need, mode)
        cache.update({m.name: mode for m in motors})

    @staticmethod
    def _read_group_mode(motors: list[DmMotor], cache: dict) -> Optional[ControlMode]:
        """查电机组当前模式（本地缓存，不发总线帧，离线可查；组内模式唯一才
        返回该模式，否则 ``None``——arm/end 共用）。"""
        modes = {cache.get(m.name) for m in motors}
        return modes.pop() if len(modes) == 1 else None

    @staticmethod
    def _disable_motors(motors: list[DmMotor]) -> None:
        """逐电机发失能帧（不等应答；arm/end 共用）。"""
        for m in motors:
            m.bus.disable(m)

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
        """语义见 :meth:`Backend.connect`；DM 实现：逐段建总线（同 channel 共享）并注册电机。

        任一段失败时回滚：关闭本次已打开的总线后再抛出原异常，避免半连接状态泄漏串口句柄
            （调用方 connect 失败后往往不会再调 disconnect）。
        """
        opened: dict[str, DmCanBus] = {}
        try:
            for sec_name, sec, motors in (
                ("arm", self._arm_cfg, self._arm_motors),
                ("end", self._end_cfg, self._end_motors),
            ):
                if not sec:
                    continue
                channel = sec.get("channel")
                if not channel:
                    raise ValueError(
                        f"backend_dm.py - connect：config backend.{sec_name}.channel 缺失"
                        f"（串口设备路径，如 /dev/ttyACM0）")
                bus = self._buses.get(channel)
                if bus is None:
                    bus = DmCanBus(channel, sec.get("baud_rate", _DEFAULT_BAUD_RATE))
                    bus.open()
                    self._buses[channel] = bus
                    opened[channel] = bus
                for m in motors:
                    bus.add_motor(m)
        except Exception:
            for channel, bus in opened.items():
                try:
                    bus.close()
                except Exception as e:
                    logger.warning("backend_dm.py - connect：回滚关闭总线 %s 失败：%s",
                                   channel, e)
                self._buses.pop(channel, None)
            raise

    def disconnect(self) -> None:
        """语义见 :meth:`Backend.disconnect`；DM 实现：失能电机 → 停 RX 关串口 → 清模式缓存。

        失能/关闭单条总线失败只记告警不中断（如设备已被拔出时写帧必失败），
        确保其余总线仍被关闭、串口句柄不泄漏。
        """
        # 顺序：失能电机 → 停 RX 线程 → 关串口（共享总线只关一次）
        for m in self._arm_motors + self._end_motors:
            if m.bus is not None:
                try:
                    m.bus.disable(m)
                except Exception as e:
                    logger.warning("backend_dm.py - disconnect：失能电机 %s 失败"
                                   "（继续关闭总线）：%s", m.name, e)
        for channel, bus in self._buses.items():
            try:
                bus.close()
            except Exception as e:
                logger.warning("backend_dm.py - disconnect：关闭总线 %s 失败：%s",
                               channel, e)
        self._buses.clear()
        for m in self._arm_motors + self._end_motors:
            m.bus = None
        self._mode_arm.clear()
        self._mode_end.clear()

    @property
    def connected(self) -> bool:
        """通信链路是否已建立（有已打开的总线即 ``True``）。"""
        return bool(self._buses)

    # ----------------------------------------------------------
    # 本体：_arm
    # ----------------------------------------------------------
    def enable_arm(self, joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend.enable_arm`；DM 实现：逐电机使能并等应答、校验错误码。"""
        self._check_open()
        self._enable_motors(self._motors_for(joint))

    def disable_arm(self, joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend.disable_arm`；DM 实现：逐电机发失能帧（不等应答）。"""
        self._check_open()
        self._disable_motors(self._motors_for(joint))

    def set_zero_arm(self, joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend.set_zero_arm`；DM 实现：失能 → 轮询至无故障 → 标零并等应答。"""
        self._check_open()
        self._set_zero_motors(self._motors_for(joint))

    def clear_fault_arm(self, joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend.clear_fault_arm`；DM 实现：失能帧清错 + 轮询核对 + 重新使能核对。"""
        self._check_open()
        self._clear_fault_motors(self._motors_for(joint))

    def set_mode_arm(self, mode: ControlMode = ControlMode.POSITION,
                     joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend.set_mode_arm`；DM 实现：仅对需切换的电机写增益/模式并确认。"""
        self._check_open()
        self._set_group_mode(self._motors_for(joint), self._mode_arm, mode)

    def read_mode_arm(self, joint: Optional[int] = None) -> Optional[ControlMode]:
        """查询本体关节当前控制模式（本地缓存，不发总线帧，离线可查）。

        指定 ``joint`` 返回该关节模式（未设置为 ``None``）；``joint=None`` 时
        整臂各关节模式唯一才返回该模式，否则 ``None``。
        """
        return self._read_group_mode(self._motors_for(joint), self._mode_arm)

    def read_state_arm(self, joint: Optional[int] = None) -> ArmState:
        """语义见 :meth:`Backend.read_state_arm`；DM 实现：两段式批量刷新产出完整 ``ArmState``。"""
        self._check_open()
        motors = self._motors_for(joint)
        arrived = self._read_group_state(motors)
        return self._assemble_arm_state(motors, arrived, joint)

    def read_state_cache_arm(self, joint: Optional[int] = None) -> ArmState:
        """语义见 :meth:`Backend.read_state_cache_arm`；DM 实现：直接组装电机
        状态槽（零总线帧、零等待，不发刷新请求）。"""
        self._check_open()
        motors = self._motors_for(joint)
        return self._assemble_arm_state(motors, [m.t_state > 0.0 for m in motors], joint)

    def state_age_arm(self, joint: Optional[int] = None) -> float:
        """语义见 :meth:`Backend.state_age_arm`；DM 实现：取最旧电机槽的陈旧度。"""
        return max(time.monotonic() - m.t_state for m in self._motors_for(joint))

    def _assemble_arm_state(self, motors: list[DmMotor],
                            arrived: list[bool], joint: Optional[int] = None) -> ArmState:
        """由电机状态槽组装 ``ArmState``（read_state_arm 与缓存读共用；不碰总线）。"""
        errs = [m.err for m in motors]
        errors = [
            f"{m.name}: {_ERR_FAULT_NAMES.get(e, '故障')}({e:#x})"
            for m, ok, e in zip(motors, arrived, errs)
            if ok and e not in (_ERR_ENABLED, _ERR_DISABLED)
        ] + [f"{m.name}: 通讯无应答" for m, ok in zip(motors, arrived) if not ok]
        # 所选电机模式一致时取该值；未设置/混合时显示 POSITION（信息性字段，
        # 指令门槛以 _require_mode_* 按电机校验为准）
        mode = self.read_mode_arm(joint)
        return ArmState(
            joint=JointState(
                control_mode=mode or ControlMode.POSITION,
                q=np.array([m.q for m in motors]),
                dq=np.array([m.dq for m in motors]),
                tau=np.array([m.tau for m in motors]),
                enabled=np.array([e == _ERR_ENABLED for e in errs]),
                error=np.array([e not in (_ERR_ENABLED, _ERR_DISABLED) for e in errs]),
                comm_ok=np.array(arrived, dtype=bool),
                # DM 反馈无独立编码器状态位，以通讯正常近似
                angle_ok=np.array(arrived, dtype=bool),
                temp_mos=np.array([m.t_mos for m in motors], dtype=float),
                temp_rotor=np.array([m.t_rotor for m in motors], dtype=float),
            ),
            mode=mode or ControlMode.POSITION,
            timestamp=time.time(),
            errors=errors,
        )

    def _send_position_arm(self, q: np.ndarray, joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend._send_position_arm`；DM 实现：POS_VEL 帧，速度取 config ``vlim``。"""
        self._require_mode_arm(ControlMode.POSITION, self._motors_for(joint))
        for m, qi in self._rows(q, "q", joint):
            m.bus.send_pos_vel(m, float(qi), m.vlim)  # 限速取 config POS_VEL.vlim

    def _send_velocity_arm(self, dq: np.ndarray, joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend._send_velocity_arm`；DM 实现：VEL 帧并按 ``vlim`` 限幅。"""
        self._require_mode_arm(ControlMode.VELOCITY, self._motors_for(joint))
        for m, dqi in self._rows(dq, "dq", joint):
            m.bus.send_vel(m, float(np.clip(dqi, -m.vlim, m.vlim)))  # 限速：config POS_VEL.vlim

    def _send_mit_arm(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_ff: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        """语义见 :meth:`Backend._send_mit_arm`；DM 实现：MIT 帧，``kp/kd`` 缺省回退 config 增益。"""
        motors = self._motors_for(joint)
        self._require_mode_arm(ControlMode.MIT, motors)
        self._send_mit_group(motors, q, dq, tau_ff, kp, kd)

    def read_param_arm(self, key: str, joint: Optional[int] = None):
        """语义见 :meth:`Backend.read_param_arm`；DM 实现：0x33 参数通道（arm/end 共用助手）。"""
        values = self._read_params(self._motors_for(joint), key)
        return values[0] if joint is not None else values

    def write_param_arm(self, key: str, value, joint: Optional[int] = None,
                        persist: bool = False) -> None:
        """语义见 :meth:`Backend.write_param_arm`；DM 实现：0x55 写回读比对确认。"""
        self._write_params(self._motors_for(joint), key, value, persist)

    # ----------------------------------------------------------
    # 末端：_end（电机组，本型号=单夹爪电机；位置语义为电机弧度，q_min=张开、q_max=闭合）
    # ----------------------------------------------------------
    def enable_end(self, joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend.enable_end`；DM 实现：同 ``enable_arm``（共用使能流程）。"""
        self._check_open()
        self._enable_motors(self._end_motors_for(joint))

    def disable_end(self, joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend.disable_end`；DM 实现：逐电机发失能帧（不等应答）。"""
        self._check_open()
        self._disable_motors(self._end_motors_for(joint))

    def set_zero_end(self, joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend.set_zero_end`；DM 实现：同 ``set_zero_arm``（共用标零流程）。"""
        self._check_open()
        self._set_zero_motors(self._end_motors_for(joint))

    def clear_fault_end(self, joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend.clear_fault_end`；DM 实现：同 ``clear_fault_arm``（共用清错流程）。"""
        self._check_open()
        self._clear_fault_motors(self._end_motors_for(joint))

    def set_mode_end(self, mode: ControlMode = ControlMode.POSITION,
                     joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend.set_mode_end`；DM 实现：同 ``set_mode_arm``（写增益/模式确认）。"""
        self._check_open()
        self._set_group_mode(self._end_motors_for(joint), self._mode_end, mode)

    def read_mode_end(self, joint: Optional[int] = None) -> Optional[ControlMode]:
        """查询末端电机当前控制模式（本地缓存，语义同 :meth:`read_mode_arm`）。"""
        return self._read_group_mode(self._end_motors_for(joint), self._mode_end)

    def read_state_end(self, joint: Optional[int] = None) -> dict:
        """语义见 :meth:`Backend.read_state_end`；DM 实现：两段式刷新，返回逐电机字段字典。"""
        self._check_open()
        motors = self._end_motors_for(joint)
        arrived = self._read_group_state(motors)
        return self._assemble_end_state(motors, arrived)

    def read_state_cache_end(self, joint: Optional[int] = None) -> dict:
        """语义见 :meth:`Backend.read_state_cache_end`；DM 实现：直接组装电机
        状态槽（零总线帧、零等待，不发刷新请求）。"""
        self._check_open()
        motors = self._end_motors_for(joint)
        return self._assemble_end_state(motors, [m.t_state > 0.0 for m in motors])

    def state_age_end(self, joint: Optional[int] = None) -> float:
        """语义见 :meth:`Backend.state_age_end`；DM 实现：取最旧电机槽的陈旧度。"""
        return max(time.monotonic() - m.t_state for m in self._end_motors_for(joint))

    @staticmethod
    def _assemble_end_state(motors: list[DmMotor], arrived: list[bool]) -> dict:
        """由电机状态槽组装末端状态字典（read_state_end 与缓存读共用；不碰总线）。"""
        return {
            "q": [m.q for m in motors],
            "dq": [m.dq for m in motors],
            "tau": [m.tau for m in motors],
            "enabled": [m.err == _ERR_ENABLED for m in motors],
            "error": [m.err not in (_ERR_ENABLED, _ERR_DISABLED) for m in motors],
            "comm_ok": arrived,
            "temp_mos": [m.t_mos for m in motors],
            "temp_rotor": [m.t_rotor for m in motors],
        }

    def _send_position_end(self, position, joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend._send_position_end`；DM 实现：POS_VEL 帧 + ``_end_speed`` 限速。"""
        motors = self._end_motors_for(joint)
        self._require_mode_end(ControlMode.POSITION, motors)
        for m, p in zip(motors, self._values_for(position, "position", motors)):
            if m.q_min is None or m.q_max is None:
                raise ValueError(
                    f"backend_dm.py - _send_position_end：末端关节 {m.name} "
                    f"缺少 q_min/q_max 行程配置（config backend.end.joints）")
            m.bus.send_pos_vel(m, float(p), self._end_speed(m))

    def _send_tau_end(self, tau, joint: Optional[int] = None) -> None:
        """语义见 :meth:`Backend._send_tau_end`；DM 实现：MIT 闭合至 ``q_max``，前馈即传入 ``tau``。

        DM 夹爪无力控通道与力反馈，以 MIT 帧（``q=q_max`` 闭合目标 + 前馈
        tau）近似力矩控制；值已守卫裁剪到 ±tau_max（N·m）；本方法须先 ``set_mode_end(MIT)``。
        """
        motors = self._end_motors_for(joint)
        self._require_mode_end(ControlMode.MIT, motors)
        for m, ti in zip(motors, self._values_for(tau, "tau", motors)):
            if m.q_max is None:
                raise ValueError(
                    f"backend_dm.py - _send_tau_end：末端关节 {m.name} "
                    f"缺少 q_max 行程配置（config backend.end.joints）")
            m.bus.send_mit(m, m.q_max, 0.0, float(ti), m.mit_kp, m.mit_kd)

    def _send_mit_end(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_ff: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        """语义见 :meth:`Backend._send_mit_end`（紧急阻尼通道）；DM 实现：与 ``_send_mit_arm`` 共用下发助手。"""
        motors = self._end_motors_for(joint)
        self._require_mode_end(ControlMode.MIT, motors)
        self._send_mit_group(motors, q, dq, tau_ff, kp, kd)

    def send_action_end(self, action: str, joint: Optional[int] = None) -> None:
        """末端离散动作：``open``→``q_min``、``close``→``q_max``、``zero``→电机弧度 0。"""
        motors = self._end_motors_for(joint)
        self._require_mode_end(ControlMode.POSITION, motors)
        if action not in ("open", "close", "zero"):
            raise ValueError(
                f"backend_dm.py - send_action_end：未知末端动作 {action!r}；"
                f"可用：['open', 'close', 'zero']")
        for m in motors:
            if m.q_min is None or m.q_max is None:
                raise ValueError(
                    f"backend_dm.py - send_action_end：末端关节 {m.name} "
                    f"缺少 q_min/q_max 行程配置（config backend.end.joints）")
            if action == "zero":
                # 行程不含 0 的末端（如 q_min>0）直接发 0 会越限，裁剪保安全
                target = _clamp(0.0, m.q_min, m.q_max)
            else:
                target = m.q_min if action == "open" else m.q_max
            m.bus.send_pos_vel(m, target, self._end_speed(m))

    def read_param_end(self, key: str, joint: Optional[int] = None):
        """语义见 :meth:`Backend.read_param_end`；DM 实现：0x33 参数通道（arm/end 共用助手）。"""
        values = self._read_params(self._end_motors_for(joint), key)
        return values[0] if joint is not None else values

    def write_param_end(self, key: str, value, joint: Optional[int] = None,
                        persist: bool = False) -> None:
        """语义见 :meth:`Backend.write_param_end`；DM 实现：0x55 写回读比对确认。"""
        self._write_params(self._end_motors_for(joint), key, value, persist)
