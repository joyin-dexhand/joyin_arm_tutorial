"""BackendDM 协议层离线单测（无需硬件）。

覆盖：float↔uint 映射、MIT 位打包、状态帧解包、30B 桥发送帧封装、16B 应答帧
提取（含跨读残余）、DmCanBus 帧分发（状态帧/参数应答/CANID==0 回退）与指令
原语字节级正确性、BackendDM 离线实例化与约束检查、公共查询接口
（connected / read_mode_* 三态语义）、错误码语义回归
（0=失能正常/1=使能正常/8~E=故障）。

运行：``python test/test_backend_dm.py``（或 ``pytest test/test_backend_dm.py``）。
"""
from __future__ import annotations

import inspect
import struct
import sys
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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"全部通过：{len(tests)} 项")
