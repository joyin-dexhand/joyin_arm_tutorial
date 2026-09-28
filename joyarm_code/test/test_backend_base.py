"""Backend 基类测试：上层公开调用 + 单 joint 内核调用面（Dummy 全内核模拟）。

覆盖：cfg 预解析（限位/发送默认/POS_VEL 增益/baud）、三值状态标志、族内同步读取、
写后读回验证、三模式发送流水线（维度→空值/裁剪→连接/模式门禁→逐 joint 内核）、
非 MIT 增益校验（空值仅 warn）、错误码获取/检查/清错（失能门禁）、空族跳过、低频保活线程（mode 已知不重读）、断连编排、
刷新 0.8 阈值与 abled 推导、定时器销毁双路径（close() 显式 / 随对象销毁自动）。

注意：test_04 起的多数用例共享模块级主 Dummy ``b`` 的演化状态（与被移植的冒烟脚本一致），
依赖 pytest 同文件按定义顺序执行；warn/info 按语义通道独立限频（``_warn_last``/``_info_last`` 字典），
连续断言多类节流日志须逐处复位 ``b._warn_last = {}``（需要处补 ``b._info_last = {}``）。
test_24 起为本轮审阅新增（修复回归 / end 族对称 / 属性与边界），均用独立实例，不依赖 ``b`` 的演化状态。
"""
import copy
import gc
import logging
import threading
import time
import weakref
from pathlib import Path

import numpy as np
import pytest
import yaml

from joyarm_core.backend.backend import Backend
from joyarm_core.utils.types import ArmState, ControlMode, JointState

ROOT = Path(__file__).resolve().parents[1]


# ---- 日志捕获（joyarm_core.backend logger） ----
class _Cap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


cap = _Cap()
_lg = logging.getLogger("joyarm_core.backend")
_lg.addHandler(cap)
_lg.propagate = False


def warns(sub=""):
    return [m for m in cap.msgs if sub in m]


# ---- Dummy 子类：实现全部抽象内核，行为可脚本化 ----
class DummyBackend(Backend):
    def __init__(self, cfg):
        self.calls = []
        self.fail_at = {}      # 内核名 -> {joint 下标}：命中抛异常
        self.ret_false = {}    # 内核名 -> {joint 下标}：命中返回 False
        self.registers = {}    # (family, i, key) -> 值
        self.readback_scale = 1.0
        super().__init__(cfg)
        self.modes = {f: [ControlMode.POSITION] * self._n(f) for f in ("arm", "end")}
        self.jstates = {f: [{"q": 0.1 * (i + 1), "dq": 0.0, "tau": 0.0,
                             "temp_mos": 30.0 + i, "temp_rotor": 30.0 + i,
                             "error": 0} for i in range(self._n(f))]
                        for f in ("arm", "end")}

    def _hit(self, name, i=None):
        if name in self.fail_at and (i is None or i in self.fail_at[name]):
            raise RuntimeError(f"{name} scripted fail")
        return name in self.ret_false and (i is None or i in self.ret_false[name])

    # ---- 生命周期内核 ----
    def _connect_arm(self, channel, protocol):
        self.calls.append(("connect_arm", channel, protocol))
        return not self._hit("connect_arm")

    def _connect_end(self, channel, protocol):
        self.calls.append(("connect_end", channel, protocol))
        return not self._hit("connect_end")

    def _disconnect_arm(self):
        self.calls.append(("disconnect_arm",))
        self._hit("disconnect_arm")

    def _disconnect_end(self):
        self.calls.append(("disconnect_end",))
        self._hit("disconnect_end")

    def _enable_joint_arm(self, i):
        self.calls.append(("enable_joint_arm", i))
        return not self._hit("enable_joint_arm", i)

    def _enable_joint_end(self, i):
        self.calls.append(("enable_joint_end", i))
        return not self._hit("enable_joint_end", i)

    def _disable_joint_arm(self, i):
        self.calls.append(("disable_joint_arm", i))
        return not self._hit("disable_joint_arm", i)

    def _disable_joint_end(self, i):
        self.calls.append(("disable_joint_end", i))
        return not self._hit("disable_joint_end", i)

    # ---- 读取内核 ----
    def _read_joint_param_arm(self, i, key):
        self.calls.append(("read_joint_param_arm", i, key))
        self._hit("read_joint_param_arm", i)
        return self.registers.get(("arm", i, key), 0.0) * self.readback_scale

    def _read_joint_param_end(self, i, key):
        self.calls.append(("read_joint_param_end", i, key))
        self._hit("read_joint_param_end", i)
        return self.registers.get(("end", i, key), 0.0) * self.readback_scale

    def _read_joint_mode_arm(self, i):
        self.calls.append(("read_joint_mode_arm", i))
        self._hit("read_joint_mode_arm", i)
        return self.modes["arm"][i]

    def _read_joint_mode_end(self, i):
        self.calls.append(("read_joint_mode_end", i))
        self._hit("read_joint_mode_end", i)
        return self.modes["end"][i]

    def _read_joint_state_arm(self, i):
        self.calls.append(("read_joint_state_arm", i))
        self._hit("read_joint_state_arm", i)
        return dict(self.jstates["arm"][i])

    def _read_joint_state_end(self, i):
        self.calls.append(("read_joint_state_end", i))
        self._hit("read_joint_state_end", i)
        return dict(self.jstates["end"][i])

    # ---- 写入内核 ----
    def _write_joint_param_arm(self, i, key, value):
        self.calls.append(("write_joint_param_arm", i, key, value))
        self._hit("write_joint_param_arm", i)
        self.registers[("arm", i, key)] = value

    def _write_joint_param_end(self, i, key, value):
        self.calls.append(("write_joint_param_end", i, key, value))
        self._hit("write_joint_param_end", i)
        self.registers[("end", i, key)] = value

    def _set_joint_mode_arm(self, i, mode):
        self.calls.append(("set_joint_mode_arm", i, mode))
        self._hit("set_joint_mode_arm", i)
        self.modes["arm"][i] = mode

    def _set_joint_mode_end(self, i, mode):
        self.calls.append(("set_joint_mode_end", i, mode))
        self._hit("set_joint_mode_end", i)
        self.modes["end"][i] = mode

    def _set_joint_zero_arm(self, i):
        self.calls.append(("set_joint_zero_arm", i))
        return not self._hit("set_joint_zero_arm", i)

    def _set_joint_zero_end(self, i):
        self.calls.append(("set_joint_zero_end", i))
        return not self._hit("set_joint_zero_end", i)

    # ---- 发送内核（单 joint 一帧，逐 joint 被基类循环调用） ----
    def _send_joint_mit_arm(self, i, tau, q, dq, kp, kd):
        self.calls.append(("send_joint_mit_arm", i, tau, q, dq, kp, kd))
        self._hit("send_mit_arm", i)

    def _send_joint_mit_end(self, i, tau, q, dq, kp, kd):
        self.calls.append(("send_joint_mit_end", i, tau, q, dq, kp, kd))
        self._hit("send_mit_end", i)

    def _send_joint_position_arm(self, i, q, vlim, flim):
        self.calls.append(("send_joint_position_arm", i, q, vlim, flim))
        self._hit("send_position_arm", i)

    def _send_joint_position_end(self, i, q, vlim, flim):
        self.calls.append(("send_joint_position_end", i, q, vlim, flim))
        self._hit("send_position_end", i)

    def _send_joint_vel_arm(self, i, dq):
        self.calls.append(("send_joint_vel_arm", i, dq))
        self._hit("send_vel_arm", i)

    def _send_joint_vel_end(self, i, dq):
        self.calls.append(("send_joint_vel_end", i, dq))
        self._hit("send_vel_end", i)

    def _send_action_end(self, action, **kwargs):
        self.calls.append(("send_action_end", action, kwargs))
        self._hit("send_action_end")

    # ---- 清错内核 ----
    def _clear_joint_error_arm(self, i):
        self.calls.append(("clear_joint_error_arm", i))
        return not self._hit("clear_joint_error_arm", i)

    def _clear_joint_error_end(self, i):
        self.calls.append(("clear_joint_error_end", i))
        return not self._hit("clear_joint_error_end", i)


@pytest.fixture(scope="module")
def cfg():
    with open(ROOT / "joyarm_core" / "config" / "joyarm_dm.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["backend"]


@pytest.fixture(scope="module")
def b(cfg):
    """主 Dummy：各用例按文件顺序共享其演化状态（连接于 test_06）。"""
    bb = DummyBackend(cfg)
    yield bb
    bb._refresh_stop.set()


# ---- T1 初始化（真实 yaml） ----
def test_01_init(b, cfg):
    arm_j = cfg["arm"]["joints"]
    assert b.name == "backend_dm"
    assert b._channel_arm == "/dev/ttyACM0" and b._protocol_arm == "serial_can"
    assert b._baud_arm == 921600 and b._baud_end == 921600
    assert b._joint_motor_id_arm == [j["motor_id"] for j in arm_j] and b._joint_motor_id_end == [0x07]
    assert b._joint_feedback_id_arm == [j["feedback_id"] for j in arm_j] and b._joint_feedback_id_end == [0x17]
    assert b._joint_model_arm == [j["model"] for j in arm_j] and b._joint_model_end == ["4310"]
    assert b._n_joints_arm == 6 and b._n_joints_end == 1
    assert b.n_joints_arm == 6 and b.n_joints_end == 1
    assert np.allclose(b.joint_limits_arm.dq_max, [5, 5, 5, 8, 8, 8])
    assert b.joint_limits_arm.q_min[0] == -2.8
    assert all(getattr(b, a) is None for a in
               ("is_connected_arm", "is_connected_end", "is_abled_arm", "is_abled_end"))
    assert b.mode_arm is None and b.mode_end is None
    assert b._refresh_hz == 10.0
    # 状态槽（公开成员）已定维：物理量 NaN / error -1 占位 / t=0；模式缓存逐 joint None
    assert b.joint_state_arm.q.shape == (6,) and np.isnan(b.joint_state_arm.q).all()
    assert b.joint_state_arm.error.dtype.kind == "i" and np.all(b.joint_state_arm.error == -1)
    assert b.joint_state_end.q.shape == (1,)
    assert b.joint_state_arm.t == 0.0
    assert b._joint_mode_arm == [None] * 6
    assert b._joint_mode_end == [None]
    # cfg 属性实时引用
    assert b.cfg is b._cfg and b.cfg["name"] == "backend_dm"
    # 发送默认值与 POS_VEL 增益成员自 cfg 提取
    assert np.allclose(b._kp_mit_default_arm, [j["MIT"]["kp"] for j in arm_j])
    assert np.allclose(b._kd_mit_default_arm, [j["MIT"]["kd"] for j in arm_j])
    assert np.allclose(b._vlim_default_arm, [j["POS_VEL"]["vlim"] for j in arm_j])
    assert np.allclose(b._flim_default_arm, [j["POS_VEL"]["flim"] for j in arm_j])
    assert np.allclose(b._pos_kp_arm, [j["POS_VEL"]["pos_kp"] for j in arm_j])
    assert np.allclose(b._pos_ki_arm, [j["POS_VEL"]["pos_ki"] for j in arm_j])
    assert np.allclose(b._vel_kp_arm, [j["POS_VEL"]["vel_kp"] for j in arm_j])
    assert np.allclose(b._vel_ki_arm, [j["POS_VEL"]["vel_ki"] for j in arm_j])
    assert np.allclose(b._kp_mit_default_end, [cfg["end"]["joints"][0]["MIT"]["kp"]])


# ---- T1b 预提取空值（cfg 缺键 → init warn + NaN 占位） ----
def test_02_extract_holes(cfg):
    cfg_hole = copy.deepcopy(cfg)
    del cfg_hole["arm"]["joints"][0]["MIT"]["kp"]
    del cfg_hole["arm"]["joints"][1]["POS_VEL"]["pos_ki"]
    cap.msgs.clear()
    b_hole = DummyBackend(cfg_hole)
    try:
        assert len(warns("MIT.kp")) >= 1 and len(warns("POS_VEL.pos_ki")) >= 1
        assert np.isnan(b_hole._kp_mit_default_arm[0]) and np.isfinite(b_hole._kd_mit_default_arm).all()
        assert np.isnan(b_hole._pos_ki_arm[1]) and np.isfinite(b_hole._pos_kp_arm).all()
    finally:
        b_hole._refresh_stop.set()


# ---- T2 维度校验（先于连接门禁） ----
def test_03_dims(b):
    b._warn_last = {}
    cap.msgs.clear()
    b.send_vel_arm([1, 2, 3, 4, 5])
    assert warns("dq=5") and not any(c[0] == "send_joint_vel_arm" for c in b.calls)
    b._warn_last = {}  # 复位节流配额，独立验证 end 族
    cap.msgs.clear()
    b.send_vel_end([1, 2])
    assert warns("dq=2")
    cap.msgs.clear()
    assert b.write_param_arm("pos_kp", [1, 2]) == 0 and len(warns("值列表维度 2")) >= 1


# ---- T2b 空族（n=0）视为成功但 warn ----
def test_04_empty_family(cfg):
    cfg_ne = copy.deepcopy(cfg)
    cfg_ne.pop("end", None)
    bne = DummyBackend(cfg_ne)
    try:
        cap.msgs.clear()
        assert bne.enable_end() == 1 and len(warns("未配置任何 joint")) >= 1
        bne._info_last = {}  # 复位节流时钟（空族提示为 info 通道），独立验证发送路径
        cap.msgs.clear()
        bne.send_vel_end(np.zeros(1))
        assert len(warns("未配置任何 joint")) >= 1
        bne._info_last = {}
        cap.msgs.clear()
        bne.send_action_end("open")  # 空族判定先于连接门禁（统一顺序）
        assert len(warns("未配置任何 joint")) >= 1
        assert not any(c[0] == "send_action_end" for c in bne.calls)
        bne._info_last = {}
        cap.msgs.clear()
        assert bne.get_error_end() == [] and len(warns("未配置任何 joint")) >= 1
        bne._info_last = {}
        cap.msgs.clear()
        assert bne.check_error_end() is True and len(warns("未配置任何 joint")) >= 1
        bne._info_last = {}
        cap.msgs.clear()
        assert bne.clear_error_end() is True and len(warns("未配置任何 joint")) >= 1
    finally:
        bne._refresh_stop.set()


# ---- T3 未连接门禁 ----
def test_05_unconnected_gates(b):
    cap.msgs.clear()
    assert b.get_state_arm() is None
    assert len(warns("未连接")) >= 1
    b._warn_last = {}  # 复位通道时钟，逐操作独立验证门禁告警（连接门禁同通道限频）
    cap.msgs.clear()
    assert b.send_vel_arm(np.zeros(6)) is None
    assert not any(c[0] == "send_joint_vel_arm" for c in b.calls)
    assert len(warns("未连接")) >= 1
    b._warn_last = {}
    cap.msgs.clear()
    assert b.enable_arm() == 0
    assert len(warns("未连接")) >= 1


# ---- T4 连接 ----
def test_06_connect(b):
    b.calls.clear()
    b.connect_arm()
    assert b.is_connected_arm and ("connect_arm", "/dev/ttyACM0", "serial_can") in b.calls
    b.connect_arm(channel="/dev/ttyACM9")
    assert b._channel_arm == "/dev/ttyACM9"
    assert ("connect_arm", "/dev/ttyACM9", "serial_can") in b.calls
    b.connect_end()
    assert b.is_connected_end
    # connect 自动设 MIT 默认模式（含读回写缓存）
    assert b._joint_mode_arm == [ControlMode.MIT] * 6
    assert b._joint_mode_end == [ControlMode.MIT]
    assert b.mode_arm == ControlMode.MIT and b.mode_end == ControlMode.MIT
    assert any("连接成功" in m for m in cap.msgs)  # 成功信息默认可见
    b._refresh_stop.set()  # 停保活线程，避免后台读取干扰后续调用计数


def test_06b_connect_failure(cfg):
    bf = DummyBackend(cfg)
    try:
        bf.ret_false["connect_end"] = {None}
        cap.msgs.clear()
        bf.connect_end()
        assert bf.is_connected_end is None and len(warns("连接失败")) >= 1
    finally:
        bf._refresh_stop.set()


# ---- T5 使能/失能（三值标志） ----
def test_07_enable_disable(b):
    assert b.enable_arm() == 1 and b.is_abled_arm is True
    b.ret_false["enable_joint_arm"] = {2}
    assert b.enable_arm() == 0 and b.is_abled_arm is None
    del b.ret_false["enable_joint_arm"]
    b.enable_arm()
    b.ret_false["disable_joint_arm"] = {0}
    assert b.disable_arm() == 0 and b.is_abled_arm is None
    del b.ret_false["disable_joint_arm"]
    b.enable_arm()
    assert b.disable_arm() == 1 and b.is_abled_arm is False
    b.enable_arm()


# ---- T6 get_mode（一致→模式+成员；不一致→False；失败→None） ----
def test_08_get_mode(b):
    P, M = ControlMode.POSITION, ControlMode.MIT
    b.modes["arm"] = [P] * 6  # 复位模式（connect 已自动设 MIT）
    b.modes["end"] = [P]
    assert b.get_mode_arm() == P
    assert b._joint_mode_arm == [P] * 6 and b.mode_arm == P
    b.modes["arm"][3] = M
    assert b.get_mode_arm() is None  # 不一致 → None（与读取失败同值）+ 逐 joint 模式保留
    assert b._joint_mode_arm == [P, P, P, M, P, P]
    assert b.mode_arm == P  # 族模式成员不更新
    b.modes["arm"] = [M] * 6
    assert b.get_mode_arm() == M and b.mode_arm == M
    b.fail_at["read_joint_mode_arm"] = {1}
    assert b.get_mode_arm() is None  # 失败 → None 不部分赋值
    assert b._joint_mode_arm == [M] * 6 and b.mode_arm == M
    del b.fail_at["read_joint_mode_arm"]
    assert b.get_mode_end() == P and b.mode_end == P


# ---- T7 get_state ----
def test_09_get_state(b):
    st = b.get_state_arm()
    assert isinstance(st, JointState) and st.q.shape == (6,)
    assert np.allclose(st.q, [0.1 * (i + 1) for i in range(6)])
    assert st.error.dtype.kind == "i" and np.all(st.error == 0) and st.t > 0
    b.jstates["arm"][3].pop("temp_mos")
    st = b.get_state_arm()
    # 缺键=子类不含该量 → 该字段 None，其余照常装配
    assert st is not None and st.temp_mos is None
    assert np.allclose(st.q, [0.1 * (i + 1) for i in range(6)])
    b.jstates["arm"][3]["temp_mos"] = None
    q_prev, t_prev = st.q.copy(), st.t
    b._warn_last = {}  # 复位节流配额，独立验证数据异常节流 warn
    cap.msgs.clear()
    assert b.get_state_arm() is None  # 键在值空=数据异常 → 跳过本轮并节流 warn
    assert np.array_equal(st.q, q_prev) and st.t == t_prev
    assert len(warns("跳过更新")) >= 1
    b.jstates["arm"][3]["temp_mos"] = 33.0
    st = b.get_state_arm()
    q_cm, t_cm = st.q.copy(), st.t
    b._joint_mode_arm[1] = None  # 模式未知（解耦：不阻塞状态装配）
    st2 = b.get_state_arm()
    assert st2 is not None and np.allclose(st2.q, q_cm) and st2.t >= t_cm
    b._joint_mode_arm[1] = ControlMode.MIT
    b.fail_at["read_joint_state_arm"] = {2}
    q_before, t_before = st.q.copy(), st.t
    time.sleep(0.002)
    assert b.get_state_arm() is None  # 失败 → None 且不部分赋值
    assert np.array_equal(st.q, q_before) and st.t == t_before
    del b.fail_at["read_joint_state_arm"]
    assert b.get_state_arm() is b.joint_state_arm  # 返回本次原子换入后的当前对象


# ---- T8 read_param ----
def test_10_read_param(b):
    for i in range(6):
        b.registers[("arm", i, "pos_kp")] = 100 + i
    assert b.read_param_arm("pos_kp") == [100 + i for i in range(6)]
    b.fail_at["read_joint_param_arm"] = {4}
    assert b.read_param_arm("pos_kp") is None
    del b.fail_at["read_joint_param_arm"]


# ---- T9 write_param（写后静置 0.1s 读回逐元素核对） ----
def test_11_write_param(b):
    cap.msgs.clear()
    t0 = time.perf_counter()
    assert b.write_param_arm("pos_kp", [1, 2, 3, 4, 5, 6]) == 1
    assert time.perf_counter() - t0 >= 0.095  # write_settle 默认 0.1 s（cfg 可覆盖）
    assert all(b.registers[("arm", i, "pos_kp")] == i + 1 for i in range(6))
    b.readback_scale = 0.5
    assert b.write_param_arm("pos_kp", [1, 2, 3, 4, 5, 6]) == 0
    assert len(warns("写入验证失败")) >= 1
    b.readback_scale = 1.0
    b.fail_at["write_joint_param_arm"] = {3}
    assert b.write_param_arm("pos_kp", [1, 2, 3, 4, 5, 6]) == 0
    del b.fail_at["write_joint_param_arm"]


# ---- T10 set_mode（读回核对 + 非 MIT 增益校验） ----
def test_12_set_mode(b):
    assert b.set_mode_arm(ControlMode.MIT) == 1
    assert b._joint_mode_arm == [ControlMode.MIT] * 6
    b.fail_at["read_joint_mode_arm"] = {0}
    assert b.set_mode_arm(ControlMode.VELOCITY) == 0  # 读回失败 → 0
    del b.fail_at["read_joint_mode_arm"]
    b.modes["arm"] = [ControlMode.POSITION] * 6
    assert b.set_mode_arm() == 1  # 省参默认 MIT
    assert b._joint_mode_arm == [ControlMode.MIT] * 6


def test_12b_set_mode_gains_warn(cfg):
    cfg_g = copy.deepcopy(cfg)
    del cfg_g["arm"]["joints"][0]["POS_VEL"]["vel_kp"]  # 增益空值
    bg = DummyBackend(cfg_g)
    try:
        cap.msgs.clear()
        assert np.isnan(bg._vel_kp_arm[0])
        bg.connect_arm()  # connect 自动设 MIT 不涉及增益校验
        assert bg.is_connected_arm
        bg._refresh_stop.set()
        cap.msgs.clear()
        bg.calls.clear()
        # 非 MIT 缺增益 → 仅 warn（采用电机内部增益），不拦截切换
        assert bg.set_mode_arm(ControlMode.POSITION) == 1
        assert len(warns("采用电机内部增益")) >= 1
        assert any(c[0] == "set_joint_mode_arm" for c in bg.calls)
        assert bg.set_mode_arm(ControlMode.MIT) == 1
    finally:
        bg._refresh_stop.set()


# ---- T11 set_zero ----
def test_13_set_zero(b):
    assert b.set_zero_arm() == 1
    b.ret_false["set_joint_zero_arm"] = {1}
    assert b.set_zero_arm() == 0
    del b.ret_false["set_joint_zero_arm"]


# ---- T12 错误码获取 / 检查 / 清错 ----
def test_14_get_check_clear_error(b):
    # get_error：直接读状态快照 error 段（不发帧、不触发读取）；0=失能 / 1=使能 均正常
    b.get_state_arm()  # 显式填槽（get_error 自身不读取，槽由刷新线程 / get_state 维护）
    assert b.get_error_arm() == [0] * 6
    b.jstates["arm"][2]["error"] = 1  # 使能态码
    b.jstates["arm"][4]["error"] = 8  # 故障码
    b.get_state_arm()
    assert b.get_error_arm() == [0, 0, 1, 0, 8, 0]
    # check_error：基于 get_error 判断，0/1 均正常，出现其他码即故障
    b.jstates["arm"][4]["error"] = 0
    b.get_state_arm()
    assert b.check_error_arm() is True
    b.jstates["arm"][4]["error"] = 8  # 故障码
    b.get_state_arm()
    cap.msgs.clear()
    assert b.check_error_arm() is False and len(warns("存在故障状态码")) >= 1
    b.jstates["arm"][4]["error"] = 0
    b.jstates["arm"][0]["q"] = None  # 数据异常 → get_state None、槽保持上次快照
    cap.msgs.clear()
    assert b.get_state_arm() is None
    assert b.get_error_arm() == [0, 0, 1, 0, 8, 0]  # 不触发读取 → 返回保持的快照（含故障码 8）
    b.jstates["arm"][0]["q"] = 0.1
    b.get_state_arm()
    assert b.get_error_arm() == [0, 0, 1, 0, 0, 0]
    # clear_error：失能门禁 → 逐 joint 发清错帧（无静置、无读回验证；不自动失能/使能）
    b._is_abled_arm = True  # 已使能 → 拒绝
    cap.msgs.clear()
    b.calls.clear()
    assert b.clear_error_arm() is False and len(warns("清错须在失能状态下执行")) >= 1
    assert not any(c[0] == "clear_joint_error_arm" for c in b.calls)
    assert b.is_abled_arm is True  # 门禁拦截，使能标志不动
    b._is_abled_arm = None  # 未知 → 同样拒绝
    b._warn_last = {}  # 清错通道刚发过门禁告警，复位后再验
    cap.msgs.clear()
    assert b.clear_error_arm() is False and len(warns("清错须在失能状态下执行")) >= 1
    b._is_abled_arm = False  # 失能 → 放行
    b.calls.clear()
    assert b.clear_error_arm() is True
    assert [c[1] for c in b.calls if c[0] == "clear_joint_error_arm"] == list(range(6))
    assert not any(c[0] == "disable_joint_arm" for c in b.calls)
    assert b.is_abled_arm is False  # 使能标志不被清错改动
    b.fail_at["clear_joint_error_arm"] = {2}
    b._warn_last = {}  # 清错通道同上，复位后再验异常告警
    cap.msgs.clear()
    assert b.clear_error_arm() is False and len(warns("清错异常")) >= 1
    del b.fail_at["clear_joint_error_arm"]
    b.jstates["arm"][2]["error"] = 1  # 清错后为使能码 1 亦正常（Dummy 不模拟硬件清码，码表由 get/check 读）
    assert b.clear_error_arm() is True
    b.jstates["arm"][2]["error"] = 0


# ---- T13 守卫与发送：速度 ----
def test_15_send_vel_guard(b):
    b.modes["arm"] = [ControlMode.VELOCITY] * 6
    b.get_mode_arm()
    cap.msgs.clear()
    b._warn_last = {}
    b.calls.clear()
    b.send_vel_arm(np.full(6, 100.0))
    c13 = [c for c in b.calls if c[0] == "send_joint_vel_arm"]
    exp13 = np.clip(np.full(6, 100.0), -b.joint_limits_arm.dq_max, b.joint_limits_arm.dq_max)
    assert len(c13) == 6 and [c[1] for c in c13] == list(range(6))
    assert np.allclose([c[2] for c in c13], exp13)
    for _ in range(9):
        b.send_vel_arm(np.full(6, 100.0))
    assert 1 <= len(warns("越限")) <= 2  # 越限 warn 节流 2Hz


# ---- T13 守卫与发送：位置 ----
def test_16_send_position(b, cfg):
    arm_j = cfg["arm"]["joints"]
    b.modes["arm"] = [ControlMode.POSITION] * 6
    b.get_mode_arm()
    b.calls.clear()
    b.send_position_arm(np.full(6, 99.0))
    c13b = [c for c in b.calls if c[0] == "send_joint_position_arm"]
    assert np.allclose([c[2] for c in c13b], np.clip(np.full(6, 99.0),
                          b.joint_limits_arm.q_min, b.joint_limits_arm.q_max))
    exp_vlim = np.array([j["POS_VEL"]["vlim"] for j in arm_j])
    exp_flim = np.array([j["POS_VEL"]["flim"] for j in arm_j])
    assert np.allclose([c[3] for c in c13b], exp_vlim)  # vlim/flim 缺省取 cfg 成员
    assert np.allclose([c[4] for c in c13b], exp_flim)
    b.calls.clear()
    b.send_position_arm(np.zeros(6), np.full(6, 2.0), np.full(6, 0.2))
    c13b2 = [c for c in b.calls if c[0] == "send_joint_position_arm"]
    assert np.allclose([c[3] for c in c13b2], 2.0)  # 显式覆盖
    assert np.allclose([c[4] for c in c13b2], 0.2)
    b.calls.clear()
    b.send_position_arm(np.zeros(6), np.full(6, 99.0), np.full(6, 1.5))
    c13b3 = [c for c in b.calls if c[0] == "send_joint_position_arm"]
    assert np.allclose([c[3] for c in c13b3], b.joint_limits_arm.dq_max)  # vlim 裁到 [0,dq_max]
    assert np.allclose([c[4] for c in c13b3], 1.0)  # flim 裁到 [0,1]


# ---- T13 守卫与发送：MIT ----
def test_17_send_mit(b, cfg):
    arm_j = cfg["arm"]["joints"]
    b.modes["arm"] = [ControlMode.MIT] * 6
    b.get_mode_arm()
    b.calls.clear()
    b.send_mit_arm(np.full(6, 3.0), np.zeros(6), np.full(6, 2.0))  # tau=3, q=0, dq=2（q=0 在行程内不裁剪）
    c13c = [c for c in b.calls if c[0] == "send_joint_mit_arm"]
    assert np.allclose([c[2] for c in c13c], 3.0)  # 参数顺序 tau, q, dq
    assert np.allclose([c[3] for c in c13c], 0.0) and np.allclose([c[4] for c in c13c], 2.0)
    assert np.allclose([c[5] for c in c13c], [j["MIT"]["kp"] for j in arm_j])  # kp/kd 缺省取 cfg 成员
    assert np.allclose([c[6] for c in c13c], [j["MIT"]["kd"] for j in arm_j])
    b.calls.clear()
    b.send_mit_arm(np.full(6, 3.0), np.zeros(6), np.full(6, 2.0),
                   kp=np.full(6, 7.0), kd=np.full(6, 0.7))  # 显式覆盖增益
    c13e = [c for c in b.calls if c[0] == "send_joint_mit_arm"]
    assert np.allclose([c[5] for c in c13e], 7.0) and np.allclose([c[6] for c in c13e], 0.7)
    cap.msgs.clear()
    b._warn_last = {}
    b.calls.clear()
    b.send_mit_arm(np.full(6, 3.0), np.zeros(6), np.full(6, 2.0), kp=[1, 2])
    assert warns("kp=2") and not [c for c in b.calls if c[0] == "send_joint_mit_arm"]


# ---- T13d 发送空值门禁与模式门禁 ----
def test_18_send_gates(b):
    cap.msgs.clear()
    b._warn_last = {}
    b.calls.clear()
    saved_kp = b._kp_mit_default_arm.copy()
    b._kp_mit_default_arm = saved_kp.copy()
    b._kp_mit_default_arm[0] = np.nan
    b.send_mit_arm(np.full(6, 3.0), np.zeros(6), np.full(6, 2.0))
    assert not [c for c in b.calls if c[0] == "send_joint_mit_arm"]  # 增益成员空值 → 不发送
    assert warns("空值")
    b._kp_mit_default_arm = saved_kp
    cap.msgs.clear()
    b._warn_last = {}
    b.calls.clear()
    b.send_mit_arm(np.full(6, np.nan), np.zeros(6), np.full(6, 2.0))
    assert not [c for c in b.calls if c[0] == "send_joint_mit_arm"]  # 指令含 NaN → 不发送
    assert warns("参数『tau』")
    cap.msgs.clear()
    b._warn_last = {}
    b.calls.clear()
    b.send_vel_arm(np.zeros(6))  # 当前 MIT ≠ 所需 VELOCITY → 模式门禁拦下
    assert not any(c[0] == "send_joint_vel_arm" for c in b.calls)
    assert len(warns("≠ 所需")) >= 1


def test_18b_send_default_hole(cfg):
    cfg_nd = copy.deepcopy(cfg)
    del cfg_nd["arm"]["joints"][3]["POS_VEL"]["vlim"]  # joint4 缺 vlim → 提取 NaN 空值
    bnd = DummyBackend(cfg_nd)
    try:
        assert np.isnan(bnd._vlim_default_arm[3]) and warns("POS_VEL.vlim")
        bnd.connect_arm()
        bnd._refresh_stop.set()
        assert bnd.set_mode_arm(ControlMode.POSITION) == 1  # 放行模式门禁
        cap.msgs.clear()
        bnd.calls.clear()
        bnd.send_position_arm(np.zeros(6))
        assert not any(c[0] == "send_joint_position_arm" for c in bnd.calls)  # 空值拦截
        assert len(warns("空值")) >= 1
    finally:
        bnd._refresh_stop.set()


# ---- T13e 单 joint 发送内核异常：仅 warn 不阻断其余 joint ----
def test_19_send_single_joint_fail(b):
    b.calls.clear()
    cap.msgs.clear()
    b.fail_at["send_mit_arm"] = {2}
    b.send_mit_arm(np.full(6, 3.0), np.zeros(6), np.full(6, 2.0))
    del b.fail_at["send_mit_arm"]
    sent = [c for c in b.calls if c[0] == "send_joint_mit_arm"]
    assert [c[1] for c in sent] == list(range(6))  # 全部 joint 均尝试下发（失败不阻断其余）
    assert any("send_mit_arm（joint3）：内核异常" in m for m in cap.msgs)  # 失败 joint 指名 warn


# ---- T13f 末端发送与离散动作 ----
def test_20_send_end_action(b):
    b.send_position_end(-4.0)
    c13d = [c for c in b.calls if c[0] == "send_joint_position_end"]
    assert len(c13d) == 1 and c13d[0][1] == 0 and c13d[0][2] == -4.0
    assert np.allclose(c13d[0][3], 3.0) and np.allclose(c13d[0][4], 1.0)  # 标量升维 + vlim/flim 回退 cfg
    b.send_action_end("open", speed=2.0)
    assert any(c[0] == "send_action_end" and c[1] == "open"
               and c[2] == {"speed": 2.0} for c in b.calls)


# ---- T14 低频刷新线程（独立 dummy，50Hz） ----
def test_21_refresh_thread(cfg):
    cfg2 = copy.deepcopy(cfg)
    cfg2["state_refresh_hz"] = 50
    r = DummyBackend(cfg2)
    try:
        r.calls.clear()
        time.sleep(0.15)
        assert not [c for c in r.calls if c[0].startswith("read_joint")]  # 未连接跳过
        r.connect_arm()
        r.connect_end()
        r.calls.clear()
        stop_t = threading.Event()

        def toucher():
            while not stop_t.is_set():
                r.joint_state_arm.t = time.time()
                r.joint_state_end.t = time.time()
                time.sleep(0.005)

        th = threading.Thread(target=toucher, daemon=True)
        th.start()
        time.sleep(0.2)
        assert not [c for c in r.calls if c[0].startswith("read_joint")]  # 新鲜（高频已更新）跳过
        stop_t.set()
        th.join()
        time.sleep(0.05)
        r.calls.clear()
        time.sleep(0.2)
        reads = [c for c in r.calls if c[0].startswith("read_joint")]
        assert reads  # 陈旧触发读取
        # mode 已知（connect 自动设 MIT 已读齐）→ 刷新只读状态、不重读模式
        assert not [c for c in reads if c[0].startswith("read_joint_mode")]
        # 模式缓存出现未读取 joint → 刷新先补读模式
        r._joint_mode_arm[0] = None
        r.calls.clear()
        time.sleep(0.2)
        assert [c for c in r.calls if c[0] == "read_joint_mode_arm"]
    finally:
        r._refresh_stop.set()


# ---- T14b 刷新 0.8 陈旧度阈值（独立 dummy，20Hz：period=50ms，阈值=40ms） ----
def test_21b_refresh_threshold(cfg):
    cfg2 = copy.deepcopy(cfg)
    cfg2["state_refresh_hz"] = 20
    r = DummyBackend(cfg2)
    try:
        r.connect_arm()
        r.connect_end()
        stop_t = threading.Event()

        def backdater(lag):
            while not stop_t.is_set():
                r.joint_state_arm.t = time.time() - lag
                r.joint_state_end.t = time.time() - lag
                time.sleep(0.003)

        # 回拨 0.6×period（30ms < 40ms 阈值）→ 视为新鲜（更高频读取在更新），不触发读取
        th = threading.Thread(target=backdater, args=(0.6 * 0.05,), daemon=True)
        th.start()
        time.sleep(0.3)
        assert not [c for c in r.calls if c[0].startswith("read_joint")]
        stop_t.set()
        th.join()

        # 回拨 0.95×period（47.5ms > 40ms 阈值）→ 陈旧，触发读取
        r.calls.clear()
        stop_t.clear()
        th = threading.Thread(target=backdater, args=(0.95 * 0.05,), daemon=True)
        th.start()
        time.sleep(0.3)
        assert [c for c in r.calls if c[0] == "read_joint_state_arm"]
        stop_t.set()
        th.join()
    finally:
        r._refresh_stop.set()


# ---- T14c 刷新按 error 码推导 is_abled（独立 dummy，50Hz） ----
def test_21c_refresh_abled(cfg):
    cfg2 = copy.deepcopy(cfg)
    cfg2["state_refresh_hz"] = 50
    r = DummyBackend(cfg2)
    try:
        r.connect_arm()
        r.connect_end()

        def wait_ab(val, timeout=1.0):
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout:
                if r.is_abled_arm is val and r.is_abled_end is val:
                    return True
                time.sleep(0.01)
            return False

        for jd in r.jstates["arm"] + r.jstates["end"]:
            jd["error"] = 0
        assert wait_ab(False)           # 码全 0（失能）→ False
        for jd in r.jstates["arm"] + r.jstates["end"]:
            jd["error"] = 1
        assert wait_ab(True)            # 码全 1（使能）→ True
        r.jstates["arm"][0]["error"] = 8
        r.jstates["end"][0]["error"] = 8
        assert wait_ab(None)            # 含故障码（混合）→ None
    finally:
        r._refresh_stop.set()


# ---- T14d close() 显式销毁（幂等；只停定时器不动总线） ----
def test_21d_close(cfg):
    r = DummyBackend(cfg)
    th = r._refresh_thread
    r.connect_arm()
    r.close()
    assert not th.is_alive()            # close 返回即已停止（join 带超时）
    r.close()                           # 幂等，二次调用不抛
    assert r.is_connected_arm is True   # close 不断总线（与 disconnect_* 正交）
    assert r.get_state_arm() is not None  # 显式同步调用不依赖定时器
    r._refresh_stop.set()


# ---- T14e 随对象销毁自动退出（弱引用：线程不钉住 backend） ----
def test_21e_destroy_with_object(cfg):
    cfg2 = copy.deepcopy(cfg)
    cfg2["state_refresh_hz"] = 50
    r = DummyBackend(cfg2)
    th = r._refresh_thread
    wr = weakref.ref(r)
    del r
    gc.collect()
    assert wr() is None                 # 线程只持弱引用：对象可正常销毁（不被钉住）
    t0 = time.monotonic()
    while th.is_alive() and time.monotonic() - t0 < 1.0:
        time.sleep(0.01)
    assert not th.is_alive()            # 弱引用失效后 ≤1 周期退出


# ---- T15 types 联动 ----
def test_22_types():
    assert JointState().t == 0.0
    assert JointState().q.shape == (0,) and ArmState().joint.q.shape == (0,)


# ---- T16 断连 ----
def test_23_disconnect(b):
    b._is_abled_arm = True  # 前置：未失能 → 断连前应先失能
    b._info_last = {}  # 复位通道时钟，保证失能/断连成功 info 均可见（顺序断言）
    b.disconnect_arm()
    assert b.is_connected_arm is False and b.is_abled_arm is None
    idx_disc = next((i for i, m in enumerate(cap.msgs) if "断连成功" in m), -1)
    idx_disb = max((i for i, m in enumerate(cap.msgs[:idx_disc]) if "失能成功" in m), default=-1)
    assert idx_disc >= 0 and 0 <= idx_disb < idx_disc  # 先失能后断连（info 顺序）
    b.calls.clear()
    b.disconnect_arm()
    assert not any(c[0] == "disconnect_arm" for c in b.calls)  # 确知未连接时空操作


def test_23b_disconnect_edge(cfg):
    bf2 = DummyBackend(cfg)
    try:
        bf2.connect_arm()
        bf2.fail_at["disconnect_arm"] = {None}
        bf2.disconnect_arm()
        assert bf2.is_connected_arm is None  # 内核异常 → 停留 None
    finally:
        bf2._refresh_stop.set()
    bf3 = DummyBackend(cfg)
    try:
        bf3.calls.clear()
        bf3.disconnect_arm()
        assert any(c[0] == "disconnect_arm" for c in bf3.calls)  # 未知态（None）也尝试断连
        assert bf3.is_connected_arm is False
    finally:
        bf3._refresh_stop.set()


# ============================================================
# 本轮审阅新增用例（文件尾部，独立实例，不依赖主 Dummy ``b`` 的演化状态）
# ============================================================


# ---- T17 修复回归：write_param 读回容差（float32 寄存器） ----
def test_24_write_param_float32(cfg):
    bf32 = DummyBackend(cfg)
    try:
        bf32.connect_arm()
        bf32._refresh_stop.set()
        orig = bf32._read_joint_param_arm
        # 模拟 float32 寄存器：写入值经 float32 截断后读回（0.0125 等 float32 ≠ float64）
        bf32._read_joint_param_arm = lambda i, key: float(np.float32(orig(i, key)))
        vals = [0.0125, 0.004, 150.0, 0.5, 5.0, 1.0]
        assert bf32.write_param_arm("vel_kp", vals) == 1  # 容差比较：低位精度差异不误判失败
    finally:
        bf32._refresh_stop.set()


# ---- T17b 修复回归：非数值输入节流 warn，不向用户抛 ----
def test_25_none_input_no_raise(b):
    cap.msgs.clear()
    b._warn_last = {}
    b.calls.clear()
    b.send_vel_arm(["a"] * 6)  # 非数值 → 节流 warn，不抛
    assert not any(c[0] == "send_joint_vel_arm" for c in b.calls)
    assert len(warns("非数值")) >= 1
    b._warn_last = {}
    cap.msgs.clear()
    b.send_mit_arm([None] * 6, np.zeros(6), np.zeros(6))  # 含 None 同样不抛、不触内核
    assert not any(c[0] == "send_joint_mit_arm" for c in b.calls)
    b._warn_last = {}
    cap.msgs.clear()
    assert b.write_param_end("pos_kp", ["a"]) == 0
    assert len(warns("非数值")) >= 1
    assert not any(c[0] == "write_joint_param_end" for c in b.calls)


# ---- T17c 修复回归：disconnect 清空模式缓存（防重连后陈旧模式放行发送） ----
def test_26_disconnect_clears_mode(b):
    # 主 Dummy 现状：arm 已于 test_23 断连（模式应已清空）
    assert b.is_connected_arm is False
    assert b.mode_arm is None and b._joint_mode_arm == [None] * 6
    # end 仍连接且模式已知 → 断连清空 → 重连自动恢复默认 MIT
    assert b.is_connected_end and b.mode_end == ControlMode.POSITION
    b.disconnect_end()
    assert b.is_connected_end is False
    assert b.mode_end is None and b._joint_mode_end == [None]
    b.connect_end()
    assert b.mode_end == ControlMode.MIT


# ---- end 族对称用例（每用例独立 Dummy：连接 end 即自动设 MIT） ----
@pytest.fixture()
def e(cfg):
    be = DummyBackend(cfg)
    be.connect_end()
    be._refresh_stop.set()
    yield be
    be._refresh_stop.set()


def test_30_end_enable_disable(e):
    assert e.enable_end() == 1 and e.is_abled_end is True
    e.ret_false["enable_joint_end"] = {0}
    assert e.enable_end() == 0 and e.is_abled_end is None  # 任一失败 → 0 且停留未知态
    del e.ret_false["enable_joint_end"]
    assert e.enable_end() == 1 and e.is_abled_end is True
    e.ret_false["disable_joint_end"] = {0}
    assert e.disable_end() == 0 and e.is_abled_end is None
    del e.ret_false["disable_joint_end"]
    assert e.disable_end() == 1 and e.is_abled_end is False


def test_31_end_get_state(e):
    st = e.get_state_end()
    assert isinstance(st, JointState) and st.q.shape == (1,)
    assert np.allclose(st.q, [0.1]) and st.t > 0
    assert st.error.dtype.kind == "i" and st.error[0] == 0
    e.jstates["end"][0].pop("temp_rotor")
    st = e.get_state_end()
    assert st is not None and st.temp_rotor is None  # 缺键=结构性缺席，其余照常装配
    e.jstates["end"][0]["temp_rotor"] = None
    e._warn_last = {}
    cap.msgs.clear()
    q_prev, t_prev = st.q.copy(), st.t
    assert e.get_state_end() is None  # 键在值空=数据异常 → 跳过本轮，保留上次快照
    assert np.array_equal(st.q, q_prev) and st.t == t_prev
    assert len(warns("跳过更新")) >= 1
    e.jstates["end"][0]["temp_rotor"] = 31.0
    e.fail_at["read_joint_state_end"] = {0}
    assert e.get_state_end() is None  # 内核异常 → None
    del e.fail_at["read_joint_state_end"]
    assert e.get_state_end() is e.joint_state_end  # 返回本次原子换入后的当前对象


def test_32_end_mode_params(e):
    assert e.get_mode_end() == ControlMode.MIT  # connect 已自动设定并读回
    e.fail_at["read_joint_mode_end"] = {0}
    assert e.get_mode_end() is None  # 读失败 → None 不部分赋值
    del e.fail_at["read_joint_mode_end"]
    e.registers[("end", 0, "vel_kp")] = 0.0125
    assert e.read_param_end("vel_kp") == [0.0125]
    e.fail_at["read_joint_param_end"] = {0}
    assert e.read_param_end("vel_kp") is None  # 任一失败整族作废
    del e.fail_at["read_joint_param_end"]
    assert e.write_param_end("vel_kp", [0.02]) == 1  # 写后静置读回验证
    e.readback_scale = 0.5
    assert e.write_param_end("vel_kp", [0.02]) == 0  # 读回不一致 → 0
    e.readback_scale = 1.0


def test_33_end_set_mode_zero(e):
    e.fail_at["read_joint_mode_end"] = {0}
    assert e.set_mode_end(ControlMode.VELOCITY) == 0  # 写入成功但读回失败 → 0
    del e.fail_at["read_joint_mode_end"]
    assert e.set_mode_end(ControlMode.VELOCITY) == 1
    assert e.mode_end == ControlMode.VELOCITY
    e.ret_false["set_joint_zero_end"] = {0}
    assert e.set_zero_end() == 0
    del e.ret_false["set_joint_zero_end"]
    assert e.set_zero_end() == 1


def test_34_end_check_clear_error(e):
    # get_error：快照过期（占位 t=0 距今远超 2 个刷新周期）→ None（e 的刷新线程已停，槽不会自行填充）
    cap.msgs.clear()
    assert e.get_error_end() is None and len(warns("状态快照过期")) >= 1
    e.get_state_end()  # 填槽
    assert e.get_error_end() == [0]
    # get_error 不发帧、不触发读取：读前后 read_joint_state_end 调用数不变
    e.calls.clear()
    e.get_state_end()
    assert len([c for c in e.calls if c[0] == "read_joint_state_end"]) == 1
    assert e.get_error_end() == [0]
    assert len([c for c in e.calls if c[0] == "read_joint_state_end"]) == 1
    e._joint_state_end.t -= 0.5  # 回拨快照时刻至 2 个刷新周期之前（10Hz → 阈值 0.2s）
    e._warn_last = {}  # 状态检查通道刚发过过期告警，复位后再验
    cap.msgs.clear()
    assert e.get_error_end() is None and len(warns("状态快照过期")) >= 1
    e.get_state_end()  # 重新填槽恢复新鲜
    assert e.get_error_end() == [0]
    e.jstates["end"][0]["error"] = 9  # 欠压故障码
    assert e.get_error_end() == [0]  # 槽不跟手（仍是旧快照）
    e.get_state_end()
    assert e.get_error_end() == [9]
    e._warn_last = {}  # 状态检查通道同上，复位后验故障告警
    cap.msgs.clear()
    assert e.check_error_end() is False and len(warns("存在故障状态码")) >= 1
    e.jstates["end"][0]["error"] = 0
    e.get_state_end()
    assert e.check_error_end() is True
    # clear_error：失能门禁 → 逐电机发清错帧
    e._is_abled_end = True  # 已使能 → 拒绝
    cap.msgs.clear()
    e.calls.clear()
    assert e.clear_error_end() is False and len(warns("清错须在失能状态下执行")) >= 1
    assert not any(c[0] == "clear_joint_error_end" for c in e.calls)
    e._is_abled_end = False
    e.calls.clear()
    assert e.clear_error_end() is True
    assert [c[1] for c in e.calls if c[0] == "clear_joint_error_end"] == [0]
    e.fail_at["clear_joint_error_end"] = {0}
    e._warn_last = {}  # 清错通道刚发过门禁告警，复位后再验
    cap.msgs.clear()
    assert e.clear_error_end() is False and len(warns("清错异常")) >= 1
    del e.fail_at["clear_joint_error_end"]


def test_35_end_send_mit(e, cfg):
    end_j = cfg["end"]["joints"]
    e.calls.clear()
    e.send_mit_end([1.0], [-1.0], [0.0])
    c = [x for x in e.calls if x[0] == "send_joint_mit_end"]
    assert len(c) == 1
    assert np.allclose([x[2] for x in c], 1.0) and np.allclose([x[3] for x in c], -1.0)
    assert np.allclose([x[5] for x in c], end_j[0]["MIT"]["kp"])  # kp/kd 缺省取 cfg 成员
    assert np.allclose([x[6] for x in c], end_j[0]["MIT"]["kd"])
    e.calls.clear()
    e.send_mit_end([1.0], [-1.0], [0.0], kp=[3.0], kd=[0.3])
    c = [x for x in e.calls if x[0] == "send_joint_mit_end"]
    assert np.allclose([x[5] for x in c], 3.0) and np.allclose([x[6] for x in c], 0.3)
    e._warn_last = {}
    cap.msgs.clear()
    e.calls.clear()
    e.send_mit_end([99.0], [-1.0], [0.0])
    c = [x for x in e.calls if x[0] == "send_joint_mit_end"]
    assert np.allclose([x[2] for x in c], e.joint_limits_end.tau_max)  # tau 裁到 ±tau_max
    assert len(warns("越限")) >= 1


def test_36_end_send_vel(e):
    assert e.set_mode_end(ControlMode.VELOCITY) == 1
    e._warn_last = {}
    cap.msgs.clear()
    e.calls.clear()
    e.send_vel_end([99.0])
    c = [x for x in e.calls if x[0] == "send_joint_vel_end"]
    assert len(c) == 1 and np.allclose(c[0][2], e.joint_limits_end.dq_max)  # 裁到 dq_max
    assert len(warns("越限")) >= 1


def test_38_end_disconnect(e):
    e.enable_end()
    e.disconnect_end()
    assert e.is_connected_end is False and e.is_abled_end is None
    assert e.mode_end is None and e._joint_mode_end == [None]  # 模式缓存随断连清空
    e.calls.clear()
    e.disconnect_end()
    assert not any(c[0] == "disconnect_end" for c in e.calls)  # 确知未连接时空操作


# ---- T19 属性与空段边界 ----
def test_40_properties(b, cfg):
    assert b.name == "backend_dm" and b.n_joints_arm == 6 and b.n_joints_end == 1
    assert b.n_joints_arm == b._n_joints_arm and b.n_joints_end == b._n_joints_end
    assert b.joint_state_arm is b._joint_state_arm and b.joint_state_end is b._joint_state_end
    endj = cfg["end"]["joints"][0]
    assert b.joint_limits_end.q_min[0] == endj["q_min"] and b.joint_limits_end.q_max[0] == endj["q_max"]
    assert b.joint_limits_end.dq_max[0] == endj["dq_max"] and b.joint_limits_end.tau_max[0] == endj["tau_max"]
    assert b.cfg is b._cfg


def test_41_no_arm_section(cfg):
    cfg_na = copy.deepcopy(cfg)
    cfg_na.pop("arm", None)
    bna = DummyBackend(cfg_na)
    try:
        assert bna._n_joints_arm == 0 and bna.n_joints_arm == 0 and bna.joint_limits_arm is None
        assert bna.joint_state_arm.q.shape == (0,)
        bna._warn_last = {}
        cap.msgs.clear()
        assert bna.enable_arm() == 1 and len(warns("未配置任何 joint")) >= 1  # 空族视为成功
    finally:
        bna._refresh_stop.set()
