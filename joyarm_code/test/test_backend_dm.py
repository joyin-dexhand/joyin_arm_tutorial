"""BackendDM 协议层离线单测（无需硬件）。

覆盖：float↔uint 映射、MIT 位打包（含 kp/kd 量程钳位 + 告警节流）、状态帧
解包、30B 桥发送帧封装、16B 应答帧提取（含跨读残余、乱码残余上限截断）、
DmCanBus 帧分发（状态帧/参数应答/CANID==0 回退/未知 ID 忽略）与指令原语
字节级正确性、send 关闭防护、DmMotor config 校验（motor_id/型号）、BackendDM
离线实例化与约束检查、公共查询接口（connected / read_mode_* 三态语义）、
错误码语义回归（0=失能正常/1=使能正常/8~E=故障）、验证式清错/标零/切模流程
（替身总线）、指令帧 vlim 限速与 MIT 增益回退、末端离散动作目标、缓存读
read_state_cache_* 与陈旧度 state_age_*、参数通道（只读拒绝/persist 存闪存/
未知 key）、连接失败回滚、共享通道单总线与拔线断开保护（资源安全契约）。

运行：``python test/test_backend_dm.py``（或 ``pytest test/test_backend_dm.py``）。
"""
from __future__ import annotations

import inspect
import struct
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core.backend.backend import Backend  # noqa: E402
from joyarm_core.backend.backend_dm import (  # noqa: E402
    _PARAM_RIDS,
    _READONLY_KEYS,
    _MOTOR_LIMITS,
    _extract_frames,
    _float_to_uint,
    _pack_mit,
    _pack_tx,
    _uint_to_float,
    _unpack_status,
    _value_matches,
    BackendDM,
    DmCanBus,
    DmMotor,
)
from joyarm_core.utils.types import ControlMode  # noqa: E402


def _mk_rx_frame(can_id: int, payload: bytes) -> bytes:
    """构造 16 字节应答帧（与 DmCanBus._dispatch 的解析布局一致）。"""
    assert len(payload) == 8
    return bytes([0xAA, 0x11, 0x00]) + can_id.to_bytes(4, "little") + payload + b"\x55"


def _mk_status_payload(q: float, dq: float, tau: float, err: int,
                       limits: tuple[float, float, float], fid_low: int = 1,
                       t_mos: int = 0, t_rotor: int = 0) -> bytes:
    """按状态帧位布局构造 8 字节载荷（q16|dq12|tau12，D0=err<<4|id，D6~7=温度）。"""
    pmax, vmax, tmax = limits
    q_u = _float_to_uint(q, -pmax, pmax, 16)
    dq_u = _float_to_uint(dq, -vmax, vmax, 12)
    tau_u = _float_to_uint(tau, -tmax, tmax, 12)
    return bytes([
        (err << 4) | (fid_low & 0x0F),
        (q_u >> 8) & 0xFF, q_u & 0xFF,
        (dq_u >> 4) & 0xFF,
        ((dq_u & 0xF) << 4) | ((tau_u >> 8) & 0xF),
        tau_u & 0xFF,
        t_mos & 0xFF, t_rotor & 0xFF,
    ])


# ----------------------------------------------------------------
# 编解码纯函数
# ----------------------------------------------------------------
def test_float_uint_roundtrip_and_clamp():
    cases = [(-12.5, 12.5, 16), (-30.0, 30.0, 12), (0.0, 500.0, 12), (0.0, 5.0, 12), (-10.0, 10.0, 12)]
    for x_min, x_max, bits in cases:
        res = (x_max - x_min) / ((1 << bits) - 1)
        for x in np.linspace(x_min, x_max, 21):
            y = _uint_to_float(_float_to_uint(x, x_min, x_max, bits), x_min, x_max, bits)
            assert abs(y - x) <= res / 2 + 1e-9, (x_min, x_max, bits, x, y)
        # 越界裁剪
        assert _float_to_uint(x_max * 10, x_min, x_max, bits) == (1 << bits) - 1
        assert _float_to_uint(x_min - 10, x_min, x_max, bits) == 0


def test_pack_mit_bit_layout():
    limits = _MOTOR_LIMITS["4310"]
    q, dq, tau, kp, kd = 1.23, -4.5, 2.0, 120.0, 0.8
    b = _pack_mit(q, dq, tau, kp, kd, limits)
    assert len(b) == 8
    pmax, vmax, tmax = limits
    q_u = (b[0] << 8) | b[1]
    dq_u = (b[2] << 4) | (b[3] >> 4)
    kp_u = ((b[3] & 0xF) << 8) | b[4]
    kd_u = (b[5] << 4) | (b[6] >> 4)
    tau_u = ((b[6] & 0xF) << 8) | b[7]
    assert q_u == _float_to_uint(q, -pmax, pmax, 16)
    assert dq_u == _float_to_uint(dq, -vmax, vmax, 12)
    assert kp_u == _float_to_uint(kp, 0.0, 500.0, 12)
    assert kd_u == _float_to_uint(kd, 0.0, 5.0, 12)
    assert tau_u == _float_to_uint(tau, -tmax, tmax, 12)


def test_unpack_status_roundtrip():
    for model, limits in _MOTOR_LIMITS.items():
        q, dq, tau, err, t_mos, t_rotor = -3.2, 5.5, -7.0, 8, 42, 57
        payload = _mk_status_payload(q, dq, tau, err, limits, t_mos=t_mos, t_rotor=t_rotor)
        rq, rdq, rtau, rerr, rtm, rtr = _unpack_status(payload, limits)
        pmax, vmax, tmax = limits
        assert rerr == err
        assert (rtm, rtr) == (t_mos, t_rotor)
        assert abs(rq - q) <= 2 * pmax / 0xFFFF + 1e-9
        assert abs(rdq - dq) <= 2 * vmax / 0xFFF + 1e-9
        assert abs(rtau - tau) <= 2 * tmax / 0xFFF + 1e-9


def test_pack_tx_template():
    f = _pack_tx(0x0102, bytes(range(8)))
    assert len(f) == 30
    assert f[0] == 0x55 and f[1] == 0xAA and f[2] == 0x1E
    assert f[3] == 0x03 and f[4] == 0x01 and f[8] == 0x0A and f[18] == 0x08
    assert f[13] == 0x02 and f[14] == 0x01          # CAN ID 小端
    assert f[21:29] == bytes(range(8))               # 8 字节载荷
    assert f[29] == 0x00


def test_pack_tx_rejects_short_payload():
    # 回归：非 8 字节载荷必须拒绝（切片赋值不保长，30 字节帧会缩成 26 字节）
    try:
        _pack_tx(0x201, struct.pack("<f", -3.5))
        raise AssertionError("非 8 字节载荷应抛 ValueError")
    except ValueError:
        pass


def test_extract_frames_with_garbage_and_split():
    f1, f2 = _mk_rx_frame(0x11, b"\x01" * 8), _mk_rx_frame(0x12, b"\x02" * 8)
    frames, rest = _extract_frames(b"\x00\xff\xaa" + f1 + f2[:10])   # 前导垃圾 + 半帧
    assert frames == [f1]
    assert rest == f2[:10]
    frames, rest = _extract_frames(rest + f2[10:])                   # 跨读拼接
    assert frames == [f2] and rest == b""


# ----------------------------------------------------------------
# DmCanBus 分发与指令原语（不打开串口）
# ----------------------------------------------------------------
def _mk_bus_with_motor(model="4310"):
    bus = DmCanBus("dummy")
    m = DmMotor({"name": "j1", "motor_id": 0x01, "feedback_id": 0x11, "model": model})
    bus.add_motor(m)
    return bus, m


def test_bus_dispatch_status_frame():
    bus, m = _mk_bus_with_motor()
    limits = m.limits
    m.clear_state()
    assert not m.wait_state(0)
    bus._feed(_mk_rx_frame(0x11, _mk_status_payload(1.0, 2.0, 3.0, 0, limits, t_mos=35, t_rotor=48)))
    assert m.wait_state(0)
    assert abs(m.q - 1.0) < 1e-3 and abs(m.dq - 2.0) < 2e-2 and abs(m.tau - 3.0) < 1e-2
    assert m.err == 0
    assert (m.t_mos, m.t_rotor) == (35, 48)


def test_bus_dispatch_canid0_fallback():
    bus, m = _mk_bus_with_motor()
    # CANID==0 → 以 D0 低 4 位（feedback_id&0xF=1）回退匹配
    bus._feed(_mk_rx_frame(0x0000, _mk_status_payload(0.5, 0.0, 0.0, 8, m.limits, fid_low=1)))
    assert m.wait_state(0)
    assert m.err == 8 and abs(m.q - 0.5) < 1e-3


def test_bus_dispatch_param_reply():
    bus, m = _mk_bus_with_motor()
    rid, val = 25, 1.5
    payload = bytes([0x01, 0x00, 0x33, rid]) + struct.pack("<f", val)
    bus._feed(_mk_rx_frame(0x11, payload))
    assert m.wait_param(rid, 0)
    assert abs(m.params[rid] - val) < 1e-6
    # uint 型寄存器（RID10）按 uint32 解
    payload = bytes([0x01, 0x00, 0x55, 10]) + struct.pack("<I", 2)
    bus._feed(_mk_rx_frame(0x11, payload))
    assert m.params[10] == 2


def test_read_param_waits_for_delayed_reply():
    """回归：参数应答在 wait_param 等待期间才到达（模拟真实链路延迟）也必须等到。

    修复前 clear_param 会移除事件，wait_param 在应答未到时立即返回 False
    而非阻塞等待，参数读写必然"无应答"超时（重试瞬间烧完）。
    """
    import threading as _th

    bus, m = _mk_bus_with_motor()
    reply = _mk_rx_frame(0x11, bytes([0x01, 0x00, 0x33, 25]) + struct.pack("<f", 1.5))

    def send_and_reply_later(cid, data):
        t = _th.Timer(0.03, lambda: bus._feed(reply))  # 应答延迟 30ms 投递
        t.daemon = True
        t.start()

    bus.send = send_and_reply_later  # 截获发送：应答落在 wait 窗口内
    assert abs(bus.read_param(m, 25) - 1.5) < 1e-6


def test_bus_dispatch_status_lowbyte_hits_subcode():
    """回归：状态帧位置低字节=0x33/0x55 时不得误入参数分支。

    状态帧 d[2] 是 16bit 位置低字节，保持固定位形时撞上参数子码是确定性事件；
    修复后参数分支还要求载荷内 slave ID 已注册，此类帧仍按状态帧处理。
    """
    bus, m = _mk_bus_with_motor()
    pmax = m.limits[0]
    # 取 q_u=0x3033：d[2]=0x33（撞子码）、d[1]=0x30（拼出的 slave=0x3001 非注册 ID）
    q = _uint_to_float(0x3033, -pmax, pmax, 16)
    m.clear_state()
    bus._feed(_mk_rx_frame(0x11, _mk_status_payload(q, 1.0, 2.0, 0, m.limits)))
    assert m.wait_state(0)
    assert abs(m.q - q) < 1e-9
    assert not m.params                       # 参数槽未被污染
    # 对称验证 d[2]=0x55 的分支
    q = _uint_to_float(0x3055, -pmax, pmax, 16)
    m.clear_state()
    bus._feed(_mk_rx_frame(0x11, _mk_status_payload(q, 0.0, 0.0, 0, m.limits)))
    assert m.wait_state(0)
    assert abs(m.q - q) < 1e-9
    assert not m.params


def test_bus_command_primitives_bytes():
    bus, m = _mk_bus_with_motor()
    sent = []
    bus.send = lambda cid, data: sent.append((cid, bytes(data)))  # 截获发送帧

    bus.enable(m)
    assert sent[-1] == (0x01, b"\xff" * 7 + b"\xfc")
    bus.disable(m)
    assert sent[-1] == (0x01, b"\xff" * 7 + b"\xfd")
    bus.set_zero(m)
    assert sent[-1] == (0x01, b"\xff" * 7 + b"\xfe")
    bus.send_pos_vel(m, 1.0, 2.0)
    assert sent[-1] == (0x101, struct.pack("<ff", 1.0, 2.0))
    bus.send_vel(m, -3.5)
    # VEL 载荷：速度 float32 前 4 字节 + 补零至 8 字节（短载荷会缩帧、桥固件丢帧）
    assert sent[-1] == (0x201, struct.pack("<f", -3.5) + b"\x00" * 4)
    bus.refresh(m)
    assert sent[-1][0] == 0x7FF and sent[-1][1][2] == 0xCC and sent[-1][1][0] == 0x01
    bus.save_params(m)
    assert sent[-1][0] == 0x7FF and sent[-1][1][2] == 0xAA

    # 写参数：无应答时抛错，但发送帧字节正确（float32 LE 载荷）
    try:
        bus.write_param(m, 25, 1.5, timeout=0.01, retries=1)
        raise AssertionError("应抛 RuntimeError")
    except RuntimeError:
        pass
    cid, data = sent[-1]
    assert cid == 0x7FF
    assert data[0] == 0x01 and data[1] == 0x00 and data[2] == 0x55 and data[3] == 25
    assert data[4:8] == struct.pack("<f", 1.5)


# ----------------------------------------------------------------
# BackendDM 离线实例化与约束
# ----------------------------------------------------------------
def _mk_backend() -> BackendDM:
    cfg = yaml.safe_load((_ROOT / "joyarm_core/configs/joyarm_dm.yaml").read_text(encoding="utf-8"))
    bcfg = dict(cfg["backend"])
    bcfg.pop("name")
    return BackendDM(bcfg)


def test_backenddm_offline_construction():
    be = _mk_backend()
    assert len(be._arm_motors) == 6
    assert [m.model for m in be._arm_motors] == ["4340P"] * 3 + ["4310"] * 3
    assert [m.motor_id for m in be._arm_motors] == [1, 2, 3, 4, 5, 6]
    assert [m.feedback_id for m in be._arm_motors] == [0x11, 0x12, 0x13, 0x14, 0x15, 0x16]
    assert len(be._end_motors) == 1
    em = be._end_motors[0]
    assert em.model == "4310"
    assert (em.q_min, em.q_max, em.dq_max, em.tau_max) == (-1.8, 3.8, 2.0, 1.0)
    assert _PARAM_RIDS == {
        # 控制与闭环增益（可写）
        "ctrl_mode": 10, "vel_kp": 25, "vel_ki": 26, "pos_kp": 27, "pos_ki": 28,
        # 保护阈值（可写）
        "uv": 0, "ot": 2, "oc": 3, "ov": 29, "timeout": 9,
        # 运动参数（可写）
        "acc": 4, "dec": 5, "max_spd": 6,
        # 版本身份（只读）
        "hw_ver": 13, "sw_ver": 14, "sn": 15, "sub_ver": 36,
        # 物理特性（只读）
        "kt": 1, "gr": 20, "pmax": 21, "vmax": 22, "tmax": 23,
    }
    assert _READONLY_KEYS == {"hw_ver", "sw_ver", "sn", "sub_ver", "kt", "gr",
                              "pmax", "vmax", "tmax"}


def test_backenddm_multi_motor_end():
    """多电机末端（如灵巧手）按电机组建模：索引寻址 / 标量广播 / 序列校验。"""
    cfg = yaml.safe_load((_ROOT / "joyarm_core/configs/joyarm_dm.yaml").read_text(encoding="utf-8"))
    bcfg = dict(cfg["backend"])
    bcfg.pop("name")
    j0 = dict(bcfg["end"]["joints"][0])
    j1 = dict(j0, name="gripper2", motor_id=0x08, feedback_id=0x18, q_max=1.2)
    bcfg["end"]["joints"] = [j0, j1]
    be = BackendDM(bcfg)
    assert len(be._end_motors) == 2
    assert [m.name for m in be._end_motors] == ["gripper", "gripper2"]
    assert be._end_motors_for(None) == be._end_motors
    assert be._end_motors_for(1) == [be._end_motors[1]]
    try:
        be._end_motors_for(2)
        raise AssertionError("越界索引应抛 ValueError")
    except ValueError:
        pass
    # 标量广播 / 序列等长 / 维度不匹配
    motors = be._end_motors_for(None)
    assert list(BackendDM._values_for(0.5, "position", motors)) == [0.5, 0.5]
    assert list(BackendDM._values_for([0.1, 0.9], "position", motors)) == [0.1, 0.9]
    try:
        BackendDM._values_for([0.1, 0.2, 0.3], "position", motors)
        raise AssertionError("维度不匹配应抛 ValueError")
    except ValueError:
        pass
    # 指令前必须先切模式（多电机末端同样受模式约束）
    try:
        be.send_position_end(0.5)
        raise AssertionError("未设模式应抛 RuntimeError")
    except RuntimeError:
        pass


def test_joint_addressing_contract():
    """契约：joint 形参方法缺省 None=全部；set_mode 默认位置模式；参数读写 key 为首参。"""
    methods = ("set_mode_arm", "set_mode_end", "read_mode_arm", "read_mode_end",
               "read_state_arm", "read_state_end",
               "send_action_end", "read_param_arm", "read_param_end",
               "write_param_arm", "write_param_end")
    for cls in (Backend, BackendDM):
        for name in methods:
            assert inspect.signature(getattr(cls, name)).parameters["joint"].default is None, (cls.__name__, name)
        for name in ("set_mode_arm", "set_mode_end"):
            sig = inspect.signature(getattr(cls, name))
            assert sig.parameters["mode"].default is ControlMode.POSITION, (cls.__name__, name)
        for name in ("read_param_arm", "write_param_arm", "read_param_end", "write_param_end"):
            assert list(inspect.signature(getattr(cls, name)).parameters)[1] == "key", (cls.__name__, name)


def test_backenddm_offline_guards():
    be = _mk_backend()
    guards = [
        lambda: be.enable_arm(),
        lambda: be.disable_arm(),
        lambda: be.set_mode_arm(),          # 默认参形式（POSITION）同样受连接门槛
        lambda: be.set_mode_end(),
        lambda: be.read_state_arm(),
        lambda: be.read_state_arm(0),       # joint 子集读取同样须先连接
        lambda: be.read_param_arm("pos_kp", 0),
        lambda: be.write_param_arm("pos_kp", 1.0),
        lambda: be.enable_end(),
        lambda: be.read_state_end(),
        lambda: be.read_state_end(0),
        lambda: be.read_param_end("pos_kp", 0),
        lambda: be.write_param_end("pos_kp", 1.0),
        lambda: be.send_action_end("open"),
    ]
    for call in guards:
        try:
            call()
            raise AssertionError("未连接应抛 RuntimeError")
        except RuntimeError:
            pass
    # 指令前必须先切模式
    try:
        be.send_position_arm(np.zeros(6))
        raise AssertionError("未设模式应抛 RuntimeError")
    except RuntimeError:
        pass
    # 不支持的电机型号
    try:
        DmMotor({"name": "x", "motor_id": 1, "feedback_id": 2, "model": "8006"})
        raise AssertionError("未知型号应抛 ValueError")
    except ValueError:
        pass


def test_write_param_readonly_rejected():
    """只读参数（版本/序列号/物理特性）write_param_* 拒绝写入（守卫先于总线访问）。"""
    be = _mk_backend()
    be._buses["dummy"] = DmCanBus("dummy")  # 仅过 _check_open，不实际收发
    for call in (lambda: be.write_param_arm("sw_ver", 1),
                 lambda: be.write_param_end("pmax", 1.0)):
        try:
            call()
            raise AssertionError("只读参数写入应抛 ValueError")
        except ValueError:
            pass
    # 可写参数不受守卫影响（发送帧由总线截获验证字节，不接真机）
    bus = be._arm_motors[0].bus = DmCanBus("dummy")
    sent = []
    bus.send = lambda cid, data: sent.append((cid, bytes(data)))
    be._arm_motors[0].bus = bus
    try:
        be.write_param_arm("acc", 0.5, 0)
        raise AssertionError("无应答应抛 RuntimeError 而非 ValueError")
    except RuntimeError:
        pass
    assert sent[-1][0] == 0x7FF and sent[-1][1][3] == 4  # RID=4（acc）


def test_backenddm_query_api():
    """公共查询接口：connected 三态 + read_mode_* 一致/未设置/混合三态（离线可用）。"""
    be = _mk_backend()

    # connected：未连接 False → 注入假总线 True → 清空后 False
    assert be.connected is False
    be._buses["dummy"] = DmCanBus("dummy")
    assert be.connected is True
    be._buses.clear()
    assert be.connected is False

    # read_mode_arm：未设置 None；全部一致返回该模式；混合 None；joint 子集独立判断
    assert be.read_mode_arm() is None
    be._mode_arm.update({f"joint{i}": ControlMode.MIT for i in range(1, 7)})
    assert be.read_mode_arm() == ControlMode.MIT
    assert be.read_mode_arm(0) == ControlMode.MIT
    be._mode_arm["joint1"] = ControlMode.POSITION
    assert be.read_mode_arm() is None                 # 混合
    assert be.read_mode_arm(0) == ControlMode.POSITION  # 子集一致
    try:
        be.read_mode_arm(6)
        raise AssertionError("越界索引应抛 ValueError")
    except ValueError:
        pass

    # read_mode_end：同构三态
    assert be.read_mode_end() is None
    be._mode_end["gripper"] = ControlMode.VELOCITY
    assert be.read_mode_end() == ControlMode.VELOCITY
    assert be.read_mode_end(0) == ControlMode.VELOCITY

    # 未配置末端：与其它 _end 方法同抛 RuntimeError
    cfg = yaml.safe_load((_ROOT / "joyarm_core/configs/joyarm_dm.yaml").read_text(encoding="utf-8"))
    bcfg = dict(cfg["backend"])
    bcfg.pop("name")
    bcfg.pop("end")
    be_noend = BackendDM(bcfg)
    assert be_noend.read_mode_arm() is None
    try:
        be_noend.read_mode_end()
        raise AssertionError("未配置末端应抛 RuntimeError")
    except RuntimeError:
        pass


def test_backenddm_err_semantics():
    """回归：错误码语义 0=失能正常、1=使能正常、8~E=故障（曾误设 0=使能/8=失能，
    致真机失能显示"使能"、使能成功被报"错误码 1"故障）。"""
    be = _mk_backend()
    bus = DmCanBus("dummy")
    be._buses["dummy"] = bus  # 跳过 connect() 打开串口，仅注册总线
    m = be._arm_motors[0]
    bus.add_motor(m)

    def _read_with_err(err: int):
        p = _mk_status_payload(0.0, 0.0, 0.0, err, m.limits)
        bus.refresh = lambda motor: bus._feed(_mk_rx_frame(m.feedback_id, p))
        return be.read_state_arm(0)

    s = _read_with_err(0)                       # 失能正常
    assert not s.joint.enabled[0] and not s.joint.error[0] and not s.errors
    s = _read_with_err(1)                       # 使能正常
    assert s.joint.enabled[0] and not s.joint.error[0] and not s.errors
    s = _read_with_err(9)                       # 欠压故障
    assert not s.joint.enabled[0] and s.joint.error[0]
    assert any("欠压(0x9)" in msg for msg in s.errors)

    # 使能应答 err=1 视为成功；err=8（超压）应抛带故障名的 RuntimeError
    p_ok = _mk_status_payload(0.0, 0.0, 0.0, 1, m.limits)
    bus.enable = lambda motor: bus._feed(_mk_rx_frame(m.feedback_id, p_ok))
    be.enable_arm(0)
    p_ov = _mk_status_payload(0.0, 0.0, 0.0, 8, m.limits)
    bus.enable = lambda motor: bus._feed(_mk_rx_frame(m.feedback_id, p_ov))
    try:
        be.enable_arm(0)
        raise AssertionError("超压应答应抛 RuntimeError")
    except RuntimeError as exc:
        assert "超压(0x8)" in str(exc)


# ============================================================
# 帧缓冲上限 + 连接回滚 / 断开保护（资源安全契约）
# ============================================================
def test_extract_frames_residual_cap():
    """持续乱码：残余被截断到上限且保尾部；跨读半帧仍可在下一轮成帧。"""
    from joyarm_core.backend.backend_dm import _RX_RESIDUAL_MAX
    garbage = bytes([0x00]) * (_RX_RESIDUAL_MAX + 6000)     # 10KB 无帧乱码
    frames, rest = _extract_frames(garbage)
    assert frames == [] and len(rest) == _RX_RESIDUAL_MAX   # 截断保尾
    head = bytes([0xAA]) + bytes(range(1, 8))               # 帧头+7字节（跨读半帧）
    frames, rest = _extract_frames(garbage + head)
    assert frames == [] and len(rest) == _RX_RESIDUAL_MAX
    completion = bytes(range(8, 15)) + bytes([0x55])        # 剩余 9 字节（尾 0x55）
    frames2, _ = _extract_frames(bytes(rest) + completion)
    assert len(frames2) == 1                                 # 半帧跨读不丢


class _FakeDmBus:
    """替身总线：open/disable 可控失败；记录 open/close/disable 调用。"""

    fail_open_channels: set = set()
    fail_disable = False
    opened: list = []
    closed: list = []
    disabled: list = []

    def __init__(self, channel, baud_rate=921600):
        self.channel = channel

    def open(self):
        if self.channel in _FakeDmBus.fail_open_channels:
            raise OSError(f"cannot open {self.channel}")
        _FakeDmBus.opened.append(self.channel)

    def close(self):
        _FakeDmBus.closed.append(self.channel)

    def add_motor(self, motor):
        motor.bus = self

    def disable(self, motor):
        if _FakeDmBus.fail_disable:
            raise OSError("device unplugged")
        _FakeDmBus.disabled.append((self.channel, motor.name))


def _make_dm_backend():
    """从 joyarm_dm 真实 config 构建 BackendDM（离线：不碰串口）。"""
    from joyarm_core.joyarm import load_config
    import copy
    cfg = copy.deepcopy(load_config("joyarm_dm")["backend"])
    cfg.pop("name", None)
    return BackendDM(cfg)


def _with_fake_bus(fn):
    """测试期间把 backend_dm.DmCanBus 换成替身，用毕还原。"""
    from joyarm_core.backend import backend_dm as dm_mod

    def wrapper():
        saved = dm_mod.DmCanBus
        dm_mod.DmCanBus = _FakeDmBus
        try:
            fn()
        finally:
            dm_mod.DmCanBus = saved
    wrapper.__name__ = fn.__name__
    return wrapper


@_with_fake_bus
def test_dm_connect_failure_rolls_back():
    """end 总线打开失败：arm 已开总线被回滚关闭，无句柄残留。"""
    be = _make_dm_backend()
    be._end_cfg["channel"] = "/dev/ttyFAKE_END"             # 独立通道触发第二段失败
    _FakeDmBus.opened.clear(), _FakeDmBus.closed.clear()
    _FakeDmBus.fail_open_channels = {"/dev/ttyFAKE_END"}
    try:
        be.connect()
        raise AssertionError("end 总线打开失败应抛")
    except OSError:
        pass
    assert len(_FakeDmBus.opened) == 1                      # arm 总线曾打开
    assert _FakeDmBus.closed == _FakeDmBus.opened           # 已被回滚关闭
    assert not be.connected                                 # 无残留总线


@_with_fake_bus
def test_dm_disconnect_disable_failure_still_closes():
    """拔线场景：disable 抛错不断开收尾，串口仍被关闭、状态被清。"""
    be = _make_dm_backend()
    _FakeDmBus.opened.clear(), _FakeDmBus.closed.clear(), _FakeDmBus.disabled.clear()
    _FakeDmBus.fail_open_channels = set()
    _FakeDmBus.fail_disable = True
    be.connect()
    assert be.connected
    be.disconnect()                                         # 不抛：失能失败仅告警
    assert _FakeDmBus.disabled == []                        # 全部 disable 失败
    assert len(_FakeDmBus.closed) == len(_FakeDmBus.opened)  # 总线仍全被关闭
    assert not be.connected
    assert all(m.bus is None for m in be._arm_motors + be._end_motors)



# ============================================================
# 新增协议行为（kp/kd 量程钳位 / config 校验 / send 关闭防护）
# ============================================================
def _kp_of(b: bytes) -> int:
    return ((b[3] & 0xF) << 8) | b[4]


def _kd_of(b: bytes) -> int:
    return (b[5] << 4) | (b[6] >> 4)


def test_pack_mit_kp_kd_range_clamp():
    """kp/kd 超 12bit 编码量程（0≤kp≤500、0≤kd≤5）：编码前就近钳位（位级等价）。"""
    limits = _MOTOR_LIMITS["4310"]
    for kp_bad, kp_exp in ((800.0, 500.0), (-3.0, 0.0)):
        assert _kp_of(_pack_mit(0, 0, 0, kp_bad, 1.0, limits)) \
            == _kp_of(_pack_mit(0, 0, 0, kp_exp, 1.0, limits))
    for kd_bad, kd_exp in ((7.0, 5.0), (-1.0, 0.0)):
        assert _kd_of(_pack_mit(0, 0, 0, 10.0, kd_bad, limits)) \
            == _kd_of(_pack_mit(0, 0, 0, 10.0, kd_exp, limits))
    # 边界值不钳：kp=500 / kd=5 恰为满量程编码
    b = _pack_mit(0, 0, 0, 500.0, 5.0, limits)
    assert _kp_of(b) == (1 << 12) - 1 and _kd_of(b) == (1 << 12) - 1


def test_pack_mit_range_warn_throttle():
    """kp/kd 钳位告警 0.5s 节流：首条告警、窗口内静默、过后再告警（伪时钟）。"""
    import logging
    from types import SimpleNamespace

    import joyarm_core.backend.backend_dm as dm_mod

    class _Cap(logging.Handler):
        def __init__(self):
            super().__init__()
            self.msgs = []

        def emit(self, record):
            self.msgs.append(record.getMessage())

    cap = _Cap()
    log = logging.getLogger("joyarm_core.backend_dm")
    log.addHandler(cap)
    clock = {"t": 100.0}
    orig_time, orig_last = dm_mod.time, dm_mod._MIT_RANGE_WARN_LAST
    dm_mod.time = SimpleNamespace(monotonic=lambda: clock["t"])
    dm_mod._MIT_RANGE_WARN_LAST = float("-inf")
    try:
        limits = _MOTOR_LIMITS["4310"]
        _pack_mit(0, 0, 0, 800.0, 1.0, limits)          # 首次越限 → 告警
        assert len(cap.msgs) == 1 and "量程" in cap.msgs[0]
        clock["t"] += 0.2
        _pack_mit(0, 0, 0, 900.0, 1.0, limits)          # 0.2s 内 → 节流静默
        assert len(cap.msgs) == 1
        clock["t"] += 0.4                               # 累计 0.6s → 再告警
        _pack_mit(0, 0, 0, 800.0, 9.0, limits)          # kp/kd 同时越限同条告警
        assert len(cap.msgs) == 2
        _pack_mit(0, 0, 0, 100.0, 1.0, limits)          # 限内打包不告警
        assert len(cap.msgs) == 2
    finally:
        dm_mod.time = orig_time
        dm_mod._MIT_RANGE_WARN_LAST = orig_last
        log.removeHandler(cap)


def test_dm_motor_id_validation():
    """motor_id/feedback_id 缺失或非法：ValueError 含关节名与现有键（config 手误定位）。"""
    base = {"name": "jx", "motor_id": 1, "feedback_id": 2, "model": "4310"}
    for missing in ("motor_id", "feedback_id"):
        jcfg = dict(base)
        jcfg.pop(missing)
        try:
            DmMotor(jcfg)
            raise AssertionError(f"缺 {missing} 应抛 ValueError")
        except ValueError as e:
            assert "jx" in str(e) and "缺失或非法" in str(e)
            assert "model" in str(e)                     # 现有键列表辅助定位
    for field in ("motor_id", "feedback_id"):
        for bad_val in (None, "abc", [1]):
            try:
                DmMotor(dict(base, **{field: bad_val}))
                raise AssertionError(f"{field}={bad_val!r} 应抛 ValueError")
            except ValueError:
                pass


def test_bus_send_closed_runtimeerror():
    """send 关闭防护：未 open / close 之后发送 → RuntimeError（防静默丢帧）。"""
    from types import SimpleNamespace

    bus = DmCanBus("dummy")
    try:
        bus.send(0x01, b"\x00" * 8)
        raise AssertionError("未打开总线 send 应抛 RuntimeError")
    except RuntimeError as e:
        assert "已关闭" in str(e) and "dummy" in str(e)
    bus2 = DmCanBus("dummy")
    bus2._ser = SimpleNamespace(is_open=True, close=lambda: None)   # 模拟已打开
    bus2.close()
    try:
        bus2.send(0x01, b"\x00" * 8)
        raise AssertionError("close 后 send 应抛 RuntimeError")
    except RuntimeError:
        pass


def test_dispatch_unknown_can_id_ignored():
    """未知 CAN ID 应答帧：安全忽略（不崩溃、不误更新任何电机槽）。"""
    bus, m = _mk_bus_with_motor()
    m.clear_state()
    bus._feed(_mk_rx_frame(0x99, _mk_status_payload(1.0, 0.0, 0.0, 8, m.limits)))
    assert not m.wait_state(0) and m.params == {}


# ============================================================
# BackendDM 流程（替身总线：记录原语调用 + 脚本化状态应答）
# ============================================================
class _StubBus:
    """离线替身总线：实现 BackendDM/DmMotor 消费的原语面。

    enable/set_zero 即喂 err=1/0 应答（模拟一发一收）；refresh 按
    ``refresh_err_seq`` 逐次喂 err（耗尽取末值，缺省 0）——可脚本化
    『先故障后恢复』等序列；其余原语仅记录调用。
    """

    def __init__(self):
        self.calls = []                 # (方法名, 参数...)
        self.refresh_err_seq: list[int] = []
        self._refresh_n = 0

    def _feed_status(self, m, err=0):
        m.update_state(0.0, 0.0, 0.0, err, 0, 0)

    def send(self, can_id, data):
        self.calls.append(("send", can_id, bytes(data)))

    def enable(self, m):
        self.calls.append(("enable", m.name))
        self._feed_status(m, err=1)

    def disable(self, m):
        self.calls.append(("disable", m.name))

    def set_zero(self, m):
        self.calls.append(("set_zero", m.name))
        self._feed_status(m, err=0)

    def refresh(self, m):
        self.calls.append(("refresh", m.name))
        self._refresh_n += 1
        seq = self.refresh_err_seq
        err = seq[min(self._refresh_n - 1, len(seq) - 1)] if seq else 0
        self._feed_status(m, err=err)

    def send_mit(self, m, q, dq, tau, kp, kd):
        self.calls.append(("send_mit", m.name, q, dq, tau, kp, kd))

    def send_pos_vel(self, m, p, v):
        self.calls.append(("send_pos_vel", m.name, p, v))

    def send_vel(self, m, v):
        self.calls.append(("send_vel", m.name, v))

    def read_param(self, m, rid, timeout=0.1, retries=5):
        self.calls.append(("read_param", m.name, rid))
        return 42.0

    def write_param(self, m, rid, value, timeout=0.1, retries=5):
        self.calls.append(("write_param", m.name, rid, value))

    def save_params(self, m):
        self.calls.append(("save_params", m.name))

    def switch_mode(self, m, dm_mode):
        self.calls.append(("switch_mode", m.name, dm_mode))


def _be_with_stub() -> tuple[BackendDM, _StubBus]:
    """真实 config 构造 BackendDM 并挂替身总线（过 _check_open，不碰串口）。"""
    be = _mk_backend()
    bus = _StubBus()
    be._buses["stub"] = bus
    for m in be._arm_motors + be._end_motors:
        m.bus = bus
    return be, bus


def _mk_backend_end(**overrides) -> BackendDM:
    """末端关节条目覆盖构造（单电机末端变体）。"""
    cfg = yaml.safe_load(
        (_ROOT / "joyarm_core/configs/joyarm_dm.yaml").read_text(encoding="utf-8"))
    bcfg = dict(cfg["backend"])
    bcfg.pop("name")
    bcfg["end"]["joints"] = [dict(bcfg["end"]["joints"][0], **overrides)]
    return BackendDM(bcfg)


def test_clear_fault_success_and_latest_fault_reported():
    """验证式清错：故障→失能→轮询核对→重新使能核对；全恢复静默、持续故障按
    **最新应答**报故障名（旧槽值仅兜底）。"""
    be, bus = _be_with_stub()
    m = be._arm_motors[0]
    m.update_state(0, 0, 0, 13, 0, 0)                    # 槽内旧故障：通信丢失
    bus.refresh_err_seq = [13, 0]                        # 先仍故障、再已清除
    be.clear_fault_arm(0)                                # 全恢复：静默返回
    kinds = [c[0] for c in bus.calls]
    assert "disable" in kinds and "refresh" in kinds and "enable" in kinds
    # 持续故障：RuntimeError 汇总含最新应答故障名（非旧槽值）
    be2, bus2 = _be_with_stub()
    be2._arm_motors[1].update_state(0, 0, 0, 9, 0, 0)    # 旧故障：欠压
    bus2.refresh_err_seq = [8]                           # 最新应答：超压
    try:
        be2.clear_fault_arm(1)
        raise AssertionError("持续故障应抛 RuntimeError")
    except RuntimeError as e:
        assert "超压(0x8)" in str(e) and "断电" in str(e)


def test_clear_fault_enable_not_restored():
    """清错成功但使能未恢复（应答 err≠1）：汇总 RuntimeError『使能未恢复』。"""
    be, bus = _be_with_stub()
    bus.refresh_err_seq = [0]                            # 故障已清除
    bus.enable = lambda m: (bus.calls.append(("enable", m.name)),
                            m.update_state(0, 0, 0, 0, 0, 0))   # 使能后仍失能态
    try:
        be.clear_fault_arm(0)
        raise AssertionError("使能未恢复应抛 RuntimeError")
    except RuntimeError as e:
        assert "使能未恢复" in str(e)


def test_set_zero_flow():
    """标零流程：失能→轮询至无故障→标零帧；带故障时拒绝并报故障名。"""
    be, bus = _be_with_stub()
    be.set_zero_arm(0)
    kinds = [c[0] for c in bus.calls]
    assert kinds[0] == "disable" and "refresh" in kinds and kinds[-1] == "set_zero"
    be2, bus2 = _be_with_stub()
    bus2.refresh_err_seq = [11]                          # 持续 MOS 超温
    try:
        be2.set_zero_arm(0)
        raise AssertionError("带故障标零应抛 RuntimeError")
    except RuntimeError as e:
        assert "MOS超温(0xb)" in str(e)


def test_set_mode_position_gains_then_confirm():
    """POSITION 切模：先写四项 POS_VEL 增益（RID25~28、config 值）再确认模式；
    已到位电机经缓存跳过；MIT 只切模式不写位置环增益。"""
    be, bus = _be_with_stub()
    m = be._arm_motors[0]
    be.set_mode_arm(ControlMode.POSITION, 0)
    writes = [c for c in bus.calls if c[0] == "write_param"]
    assert [w[2] for w in writes] == [25, 26, 27, 28]    # vel_kp/vel_ki/pos_kp/pos_ki
    assert [w[3] for w in writes] == [m.gains[k] for k in
                                      ("vel_kp", "vel_ki", "pos_kp", "pos_ki")]
    assert [c for c in bus.calls if c[0] == "switch_mode"] == \
        [("switch_mode", m.name, 2)]
    assert be.read_mode_arm(0) == ControlMode.POSITION
    n0 = len(bus.calls)                                  # 已到位：再次切同模式零写入
    be.set_mode_arm(ControlMode.POSITION, 0)
    assert len(bus.calls) == n0
    bus.calls.clear()
    be.set_mode_arm(ControlMode.MIT, 0)
    assert [c for c in bus.calls if c[0] == "write_param"] == []
    assert [c for c in bus.calls if c[0] == "switch_mode"][0][2] == 1


def test_command_frames_vlimit_and_mit_gain_fallback():
    """指令路径（替身截获）：pos_vel 速度=config vlim；vel 按 ±vlim 限幅；
    MIT kp/kd=None 回退 config 增益；模式不符显性拒绝。"""
    be, bus = _be_with_stub()
    m = be._arm_motors[0]
    assert 0 < m.vlim < be.arm_limits.dq_max[0]          # 前置：vlim 在速度限内
    be._mode_arm.update({mm.name: ControlMode.POSITION
                         for mm in be._arm_motors})
    be._mode_arm[m.name] = ControlMode.VELOCITY
    be.send_velocity_arm(np.array([m.vlim * 2]), joint=0)   # 超 vlim → 限幅
    assert bus.calls[-1] == ("send_vel", m.name, m.vlim)
    be._mode_arm[m.name] = ControlMode.POSITION
    be.send_position_arm(np.array([0.5]), joint=0)
    assert bus.calls[-1] == ("send_pos_vel", m.name, 0.5, m.vlim)
    be._mode_arm[m.name] = ControlMode.MIT
    be.send_mit_arm(np.array([0.1]), np.array([0.2]), np.array([0.3]), joint=0)
    assert bus.calls[-1] == ("send_mit", m.name, 0.1, 0.2, 0.3,
                             m.mit_kp, m.mit_kd)         # None → config MIT 增益
    be._mode_arm[m.name] = ControlMode.POSITION          # 模式不符：显性拒绝
    try:
        be.send_mit_arm(np.array([0.1]), np.array([0.2]), np.array([0.3]), joint=0)
        raise AssertionError("模式不符应抛 RuntimeError")
    except RuntimeError as e:
        assert "mit" in str(e)


def test_send_action_end_targets():
    """末端离散动作：open→q_min、close→q_max、zero→行程内钳位（不含 0 时取
    下界）；未知动作 / 缺行程配置显性拒绝。"""
    be, bus = _be_with_stub()
    em = be._end_motors[0]
    be._mode_end[em.name] = ControlMode.POSITION
    speed = BackendDM._end_speed(em)
    be.send_action_end("open")
    assert bus.calls[-1] == ("send_pos_vel", em.name, em.q_min, speed)
    be.send_action_end("close")
    assert bus.calls[-1][2] == em.q_max
    be.send_action_end("zero")
    assert bus.calls[-1][2] == max(em.q_min, min(0.0, em.q_max))
    try:
        be.send_action_end("grasp")
        raise AssertionError("未知动作应抛 ValueError")
    except ValueError as e:
        assert "grasp" in str(e)
    # 行程不含 0（q_min>0）：zero 钳位到 q_min 而非直发 0
    be2, bus2 = _be_with_stub()
    em2 = be2._end_motors[0]
    em2.q_min, em2.q_max = 0.5, 1.0
    be2._mode_end[em2.name] = ControlMode.POSITION
    be2.send_action_end("zero")
    assert bus2.calls[-1][2] == 0.5
    # 缺行程配置：指令前显性报错
    em2.q_min = em2.q_max = None
    try:
        be2.send_action_end("open")
        raise AssertionError("缺行程应抛 ValueError")
    except ValueError as e:
        assert "q_min/q_max" in str(e)
    try:
        be2.send_position_end(0.5)
        raise AssertionError("位置指令缺行程应抛 ValueError")
    except ValueError:
        pass


def test_state_cache_read_and_age():
    """缓存读：零总线帧组装状态槽（refresh 不被调用）；陈旧度=最旧电机；
    从未应答电机 comm_ok=False + errors『通讯无应答』；末端同构字典。"""
    be, bus = _be_with_stub()
    m0, m1 = be._arm_motors[0], be._arm_motors[1]
    m0.update_state(0.1, 0.2, 0.3, 1, 35, 40)
    m1.update_state(0.4, 0.5, 0.6, 0, 30, 33)
    m1.t_state = time.monotonic() - 5.0                 # 人为老化 5s
    s = be.read_state_cache_arm()
    assert not any(c[0] in ("refresh", "send") for c in bus.calls)   # 零总线帧
    assert np.allclose(s.joint.q[:2], [0.1, 0.4])
    assert s.joint.enabled[0] and not s.joint.enabled[1]   # err=1 使能 / err=0 失能
    assert not s.joint.error[:2].any()
    assert s.joint.temp_mos[0] == 35 and s.joint.temp_rotor[1] == 33
    assert s.joint.comm_ok[0] and s.joint.comm_ok[1]
    assert not s.joint.comm_ok[2:].any()                # 其余电机从未应答
    assert np.array_equal(s.joint.angle_ok, s.joint.comm_ok)
    assert sum("通讯无应答" in msg for msg in s.errors) == 4
    assert s.mode == ControlMode.POSITION               # 未设置模式缺省显示
    assert be.state_age_arm() > 4.9                      # 含从未应答电机 → 巨大
    assert 4.9 < be.state_age_arm(1) < 6.0               # 子集最旧 = m1（≈5s）
    assert be.state_age_arm(0) < 1.0                     # m0 刚应答
    # 单关节从未应答：comm_ok=False + 无应答错误
    s2 = be.read_state_cache_arm(5)
    assert not s2.joint.comm_ok[0]
    assert any("通讯无应答" in msg for msg in s2.errors)
    # 末端缓存读：逐电机字段字典
    em = be._end_motors[0]
    em.update_state(1.5, 0.0, 0.2, 1, 20, 21)
    d = be.read_state_cache_end()
    assert d["q"] == [1.5] and d["enabled"] == [True] and d["comm_ok"] == [True]
    assert be.state_age_end() >= 0.0


def test_param_channel_details():
    """参数通道：未知 key ValueError（列可用名）；joint=None 读返逐电机列表；
    persist=True 触发存闪存；_value_matches int 精确 / float 容差。"""
    be, bus = _be_with_stub()
    try:
        be.read_param_arm("bogus", 0)
        raise AssertionError("未知参数名应抛 ValueError")
    except ValueError as e:
        assert "bogus" in str(e) and "可用" in str(e)
    assert be.read_param_arm("pos_kp", 0) == 42.0
    assert be.read_param_arm("pos_kp") == [42.0] * 6     # joint=None → 列表
    be.write_param_arm("acc", 0.5, 0, persist=True)
    kinds = [c[0] for c in bus.calls]
    assert "write_param" in kinds and "save_params" in kinds
    assert _value_matches(2, 2) and not _value_matches(2, 3)
    assert _value_matches(1.0000499, 1.0) and not _value_matches(1.002, 1.0)


@_with_fake_bus
def test_dm_connect_shared_channel_single_bus():
    """arm/end 同 channel（真实配置 /dev/ttyACM0）：仅开一条总线、两组电机挂
    同一实例（共享单总线规则）。"""
    be = _make_dm_backend()
    _FakeDmBus.opened.clear(), _FakeDmBus.closed.clear()
    _FakeDmBus.fail_open_channels = set()
    be.connect()
    assert _FakeDmBus.opened == ["/dev/ttyACM0"]         # 仅一次 open
    assert len({m.bus for m in be._arm_motors + be._end_motors}) == 1
    be.disconnect()
    assert _FakeDmBus.closed == ["/dev/ttyACM0"] and not be.connected


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"全部通过：{len(tests)} 项")
