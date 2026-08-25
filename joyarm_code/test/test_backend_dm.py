"""BackendDM 协议层离线单测（无需硬件）。

覆盖：float↔uint 映射、MIT 位打包、状态帧解包、30B 桥发送帧封装、16B 应答帧
提取（含跨读残余）、DmCanBus 帧分发（状态帧/参数应答/CANID==0 回退）与指令
原语字节级正确性、BackendDM 离线实例化与约束检查。

运行：``python test/test_backend_dm.py``（或 ``pytest test/test_backend_dm.py``）。
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core.backends.backend_dm import (  # noqa: E402
    _PARAM_RIDS,
    _MOTOR_LIMITS,
    _extract_frames,
    _float_to_uint,
    _pack_mit,
    _pack_tx,
    _uint_to_float,
    _unpack_status,
    BackendDM,
    DmCanBus,
    DmMotor,
)


def _mk_rx_frame(can_id: int, payload: bytes) -> bytes:
    """构造 16 字节应答帧（与 DmCanBus._dispatch 的解析布局一致）。"""
    assert len(payload) == 8
    return bytes([0xAA, 0x11, 0x00]) + can_id.to_bytes(4, "little") + payload + b"\x55"


def _mk_status_payload(q: float, dq: float, tau: float, err: int,
                       limits: tuple[float, float, float], fid_low: int = 1) -> bytes:
    """按状态帧位布局构造 8 字节载荷（q16|dq12|tau12，D0=err<<4|id）。"""
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
        0x00, 0x00,
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
        q, dq, tau, err = -3.2, 5.5, -7.0, 8
        payload = _mk_status_payload(q, dq, tau, err, limits)
        rq, rdq, rtau, rerr = _unpack_status(payload, limits)
        pmax, vmax, tmax = limits
        assert rerr == err
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
    bus._feed(_mk_rx_frame(0x11, _mk_status_payload(1.0, 2.0, 3.0, 0, limits)))
    assert m.wait_state(0)
    assert abs(m.q - 1.0) < 1e-3 and abs(m.dq - 2.0) < 2e-2 and abs(m.tau - 3.0) < 1e-2
    assert m.err == 0


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
    assert (em.q_min, em.q_max, em.force_to_tau) == (0.0, 0.8, 0.1)
    assert _PARAM_RIDS == {"ctrl_mode": 10, "vel_kp": 25, "vel_ki": 26, "pos_kp": 27, "pos_ki": 28}


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
    assert list(BackendDM._end_values(0.5, "position", motors)) == [0.5, 0.5]
    assert list(BackendDM._end_values([0.1, 0.9], "position", motors)) == [0.1, 0.9]
    try:
        BackendDM._end_values([0.1, 0.2, 0.3], "position", motors)
        raise AssertionError("维度不匹配应抛 ValueError")
    except ValueError:
        pass
    # 指令前必须先切模式（多电机末端同样受模式约束）
    try:
        be.send_position_end(0.5)
        raise AssertionError("未设模式应抛 RuntimeError")
    except RuntimeError:
        pass


def test_backenddm_offline_guards():
    be = _mk_backend()
    guards = [
        lambda: be.enable_arm(),
        lambda: be.disable_arm(),
        lambda: be.read_state_arm(),
        lambda: be.read_param_arm(0, "pos_kp"),
        lambda: be.enable_end(),
        lambda: be.read_state_end(),
        lambda: be.read_param_end(0, "pos_kp"),
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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"全部通过：{len(tests)} 项")
