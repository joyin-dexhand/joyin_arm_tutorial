"""Backend 基类测试：上层公开调用 + 单 joint 内核调用面（Dummy 全内核模拟）。

覆盖：cfg 预解析（限位/发送默认/POS_VEL 增益/baud/运行参数鲁棒回退）、三值状态标志、族内同步读取、
写入无静置读回（验证上移 JoyArm）、set_mode 写入即返 + 乐观更新模式缓存（门禁立即放行，一致性经 get_mode 读回纠正）、
三模式发送流水线（维度/一维性→空值/裁剪→连接/模式门禁→逐 joint 内核）、
状态数据异常拦截（空值/非有限值/非标量）、错误码获取/检查/清错（读快照不发帧、无失能门禁）、set_zero 无失能门禁、
空族跳过、低频保活线程（mode 已知不重读）、断连编排、刷新 0.8 阈值与 abled 推导、定时器销毁双路径
（close() 显式 / 随对象销毁自动）。

注意：test_04 起的多数用例共享模块级主 Dummy ``b`` 的演化状态（与被移植的冒烟脚本一致），
依赖 pytest 同文件按定义顺序执行。
test_24 起为本轮审阅新增（修复回归 / end 族对称 / 属性与边界），均用独立实例，不依赖 ``b`` 的演化状态；
test_42 起为五轴审查新增（指令一维性 / 读参数标量校验 / cfg 参数鲁棒回退 / 状态非有限值 / 设模式类型门禁 / 空族写入一致性）；
test_48 起为第九轮审查新增（写参数非有限值拦截 / 状态非字典拦截）；
test_50 起为第十轮审查新增（内核假值返回统一拦截 / error 非有限值解析异常拦截）；
test_52 为第十一轮审查新增（状态子值非标量拦截，防装配出 (n,1) 字段毒化上层）；
test_53 起为第十二轮审查新增（模式读非枚举拦截防缓存毒化 / 限位 NaN 回退与 q_min>q_max 启动 warn / cfg 段非字典容错）；
test_59 起为第十四轮基类审阅新增（未连接门禁全覆盖 / 内核异常路径 / MIT 与末端位置裁剪对称 / 空族读写断连 /
连接编排细节 / close 慢退出与单拍重试）；test_56 因与 test_50 场景重复已删除。
test_69 为模板冒烟新增（BackendTemplate 空 cfg 可实例化 = 29 桩满足抽象方法强制；29 桩全量按约定格式抛
NotImplementedError，桩消息前缀自动取 cfg name → ``_name``（空回退模板文件名）+ 基类内核集漂移守护；
函数内导入模板，模板问题不拖垮整文件收集）。
test_70 为 backend_dm 重建新增（无硬件协议层冒烟：注册恢复、空 cfg 构造、yaml cfg 电机表/帧前缀、
MIT 位域打包位级校验、反馈帧解码含双温度与残尾拼接、参数/指令事务自环验证、非阻塞状态读三态；
函数内导入 backend_dm，硬件依赖问题不拖垮整文件收集）。
test_71 为 backend_dm 审阅轮新增（总线生命周期：RX 存活判定/僵尸重建/无主清扫，防真机重连失败与泄漏）。
test_72 为 backend_dm 末端离散动作新增（open/home/close/position：自动切 POSITION+增益序列、cfg vlim/flim
默认、按需位置透传与越限裁硬限位、未知动作/缺参/切模式失败异常路径）。
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


# ---- 日志捕获（joyarm_core.backend + joyarm_core.utils.limits logger） ----
class _Cap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


cap = _Cap()
for _name in ("joyarm_core.backend", "joyarm_core.utils.limits"):
    _lg = logging.getLogger(_name)
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
        out = yaml.safe_load(f)["backend"]
    # 默认模式相关断言与 yaml 当前值解耦（单测控制变量；专门用例自改副本不受影响）
    out["arm"]["default_mode"] = "mit"
    out["end"]["default_mode"] = "mit"
    return out


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
    assert b._refresh_hz == cfg.get("state_refresh_hz", 10.0)  # 跟随 yaml 配置（默认 10）
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
    cap.msgs.clear()
    b.send_vel_arm([1, 2, 3, 4, 5])
    assert warns("dq=5") and not any(c[0] == "send_joint_vel_arm" for c in b.calls)
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
        cap.msgs.clear()
        bne.send_vel_end(np.zeros(1))
        assert len(warns("未配置任何 joint")) >= 1
        cap.msgs.clear()
        bne.send_action_end("open")  # 空族判定先于连接门禁（统一顺序）
        assert len(warns("未配置任何 joint")) >= 1
        assert not any(c[0] == "send_action_end" for c in bne.calls)
        cap.msgs.clear()
        assert bne.get_error_end() == [] and len(warns("未配置任何 joint")) >= 1
        cap.msgs.clear()
        assert bne.check_error_end() is True and len(warns("未配置任何 joint")) >= 1
        cap.msgs.clear()
        assert bne.clear_error_end() is True and len(warns("未配置任何 joint")) >= 1
    finally:
        bne._refresh_stop.set()


# ---- T3 未连接门禁 ----
def test_05_unconnected_gates(b):
    cap.msgs.clear()
    assert b.get_state_arm() is None
    assert len(warns("未连接")) >= 1
    cap.msgs.clear()
    assert b.send_vel_arm(np.zeros(6)) is None
    assert not any(c[0] == "send_joint_vel_arm" for c in b.calls)
    assert len(warns("未连接")) >= 1
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
    # connect 自动设 MIT 默认模式（乐观填模式缓存；get_mode 读回核实）
    assert b.get_mode_arm() == ControlMode.MIT
    assert b._joint_mode_arm == [ControlMode.MIT] * 6
    assert b.get_mode_end() == ControlMode.MIT
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
    cap.msgs.clear()
    assert b.get_state_arm() is None  # 键在值空=数据异常 → 跳过本轮并 warn
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


# ---- T9 write_param（写入全组即成功：无静置、无读回验证——验证上移 JoyArm） ----
def test_11_write_param(b):
    cap.msgs.clear()
    assert b.write_param_arm("pos_kp", [1, 2, 3, 4, 5, 6]) == 1
    assert all(b.registers[("arm", i, "pos_kp")] == i + 1 for i in range(6))
    b.fail_at["write_joint_param_arm"] = {3}
    assert b.write_param_arm("pos_kp", [1, 2, 3, 4, 5, 6]) == 0
    del b.fail_at["write_joint_param_arm"]


# ---- T10 set_mode（写入成功即返回 + 乐观更新模式缓存：门禁立即放行；读回核实经 get_mode） ----
def test_12_set_mode(b):
    assert b.set_mode_arm(ControlMode.MIT) == 1
    assert b.mode_arm == ControlMode.MIT and b._joint_mode_arm == [ControlMode.MIT] * 6  # set_mode 乐观更新为目标模式
    assert b.set_mode_arm(ControlMode.VELOCITY) == 1
    assert b.mode_arm == ControlMode.VELOCITY  # set_mode 成功即乐观更新缓存
    assert b.get_mode_arm() == ControlMode.VELOCITY  # 读回硬件真值（内核已改写 dummy）
    assert b.mode_arm == ControlMode.VELOCITY
    b.fail_at["set_joint_mode_arm"] = {0}
    assert b.set_mode_arm(ControlMode.POSITION) == 0  # 内核失败 → 0（joint0 未写入，其余未尝试）
    del b.fail_at["set_joint_mode_arm"]
    assert b.mode_arm == ControlMode.VELOCITY  # 失败同样不动缓存
    assert b.set_mode_arm() == 1  # 省参默认 MIT
    assert b.get_mode_arm() == ControlMode.MIT


def test_12b_set_mode_gains_hole(cfg):
    cfg_g = copy.deepcopy(cfg)
    del cfg_g["arm"]["joints"][0]["POS_VEL"]["vel_kp"]  # 增益空值
    bg = DummyBackend(cfg_g)
    try:
        cap.msgs.clear()
        assert np.isnan(bg._vel_kp_arm[0])
        bg.connect_arm()  # connect 自动设 MIT
        assert bg.is_connected_arm
        bg._refresh_stop.set()
        cap.msgs.clear()
        bg.calls.clear()
        # 缺增益不拦截切换：增益寄存器写入由子类内核按基类成员自行处理（NaN 跳过）
        assert bg.set_mode_arm(ControlMode.POSITION) == 1
        assert any(c[0] == "set_joint_mode_arm" for c in bg.calls)
    finally:
        bg._refresh_stop.set()


# ---- T11 set_zero（无失能门禁：任意使能态直接透传内核，时序由上层保证） ----
def test_13_set_zero(b):
    for abled in (True, None, False):  # 已使能 / 未知 / 已失能：均不拦截，直接逐 joint 设零
        b._is_abled_arm = abled
        b.calls.clear()
        assert b.set_zero_arm() == 1
        assert [c[1] for c in b.calls if c[0] == "set_joint_zero_arm"] == list(range(6))
    b.ret_false["set_joint_zero_arm"] = {1}  # 任一 joint 失败 → 0
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
    # clear_error：无失能门禁 → 任意使能态直接逐 joint 发清错帧（无静置、无读回验证；不自动失能/使能）
    for abled in (True, None):  # 已使能 / 未知态：同样透传（失能时序由上层保证）
        b._is_abled_arm = abled
        b.calls.clear()
        assert b.clear_error_arm() is True
        assert [c[1] for c in b.calls if c[0] == "clear_joint_error_arm"] == list(range(6))
    b._is_abled_arm = False
    b.calls.clear()
    assert b.clear_error_arm() is True
    assert [c[1] for c in b.calls if c[0] == "clear_joint_error_arm"] == list(range(6))
    assert not any(c[0] == "disable_joint_arm" for c in b.calls)
    assert b.is_abled_arm is False  # 使能标志不被清错改动
    b.fail_at["clear_joint_error_arm"] = {2}
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
    b.calls.clear()
    b.send_vel_arm(np.full(6, 100.0))
    c13 = [c for c in b.calls if c[0] == "send_joint_vel_arm"]
    exp13 = np.clip(np.full(6, 100.0), -b.joint_limits_arm.dq_max, b.joint_limits_arm.dq_max)
    assert len(c13) == 6 and [c[1] for c in c13] == list(range(6))
    assert np.allclose([c[2] for c in c13], exp13)
    for _ in range(9):
        b.send_vel_arm(np.full(6, 100.0))
    assert len(warns("越限")) == 10  # 10 次越限发送，每次均 warn


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
    b.calls.clear()
    b.send_mit_arm(np.full(6, 3.0), np.zeros(6), np.full(6, 2.0), kp=[1, 2])
    assert warns("kp=2") and not [c for c in b.calls if c[0] == "send_joint_mit_arm"]


# ---- T13d 发送空值门禁与模式门禁 ----
def test_18_send_gates(b):
    cap.msgs.clear()
    b.calls.clear()
    saved_kp = b._kp_mit_default_arm.copy()
    b._kp_mit_default_arm = saved_kp.copy()
    b._kp_mit_default_arm[0] = np.nan
    b.send_mit_arm(np.full(6, 3.0), np.zeros(6), np.full(6, 2.0))
    assert not [c for c in b.calls if c[0] == "send_joint_mit_arm"]  # 增益成员空值 → 不发送
    assert warns("空值")
    b._kp_mit_default_arm = saved_kp
    cap.msgs.clear()
    b.calls.clear()
    b.send_mit_arm(np.full(6, np.nan), np.zeros(6), np.full(6, 2.0))
    assert not [c for c in b.calls if c[0] == "send_joint_mit_arm"]  # 指令含 NaN → 不发送
    assert warns("参数『tau』")
    cap.msgs.clear()
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
        r.calls.clear()  # 清掉 connect 静置期（write_settle）刷新 tick 的初读（同 test_21），观察窗口从干净状态开始
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


# ---- T17 修复回归：write_param 无读回验证（float32 寄存器低位差异不再误判失败） ----
def test_24_write_param_float32(cfg):
    bf32 = DummyBackend(cfg)
    try:
        bf32.connect_arm()
        bf32._refresh_stop.set()
        orig = bf32._read_joint_param_arm
        # 模拟 float32 寄存器：写入值经 float32 截断后读回（0.0125 等 float32 ≠ float64）
        bf32._read_joint_param_arm = lambda i, key: float(np.float32(orig(i, key)))
        vals = [0.0125, 0.004, 150.0, 0.5, 5.0, 1.0]
        assert bf32.write_param_arm("vel_kp", vals) == 1  # 无读回验证：低位精度差异不影响写入结果
        back = bf32.read_param_arm("vel_kp")
        assert np.allclose(back, vals, rtol=1e-6)  # 读回经 float32 截断，容差内一致
    finally:
        bf32._refresh_stop.set()


# ---- T17b 修复回归：非数值输入 warn，不向用户抛 ----
def test_25_none_input_no_raise(b):
    cap.msgs.clear()
    b.calls.clear()
    b.send_vel_arm(["a"] * 6)  # 非数值 → warn，不抛
    assert not any(c[0] == "send_joint_vel_arm" for c in b.calls)
    assert len(warns("非数值")) >= 1
    cap.msgs.clear()
    b.send_mit_arm([None] * 6, np.zeros(6), np.zeros(6))  # 含 None 同样不抛、不触内核
    assert not any(c[0] == "send_joint_mit_arm" for c in b.calls)
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
    assert b.get_mode_end() == ControlMode.MIT  # 重连自动设默认模式并乐观填缓存，get_mode 核实
    assert b.mode_end == ControlMode.MIT


# ---- end 族对称用例（每用例独立 Dummy：连接 end 即自动设 MIT） ----
@pytest.fixture()
def e(cfg):
    be = DummyBackend(cfg)
    be._refresh_stop.set()  # 先冻结刷新线程再连接：connect 静置期（write_settle）tick 会抢读状态/模式，破坏快照过期等断言的确定性
    be.connect_end()
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
    assert e.write_param_end("vel_kp", [0.02]) == 1  # 无静置、无读回验证
    assert e.registers[("end", 0, "vel_kp")] == 0.02


def test_33_end_set_mode_zero(e):
    e.fail_at["set_joint_mode_end"] = {0}
    assert e.set_mode_end(ControlMode.VELOCITY) == 0  # 内核失败 → 0
    del e.fail_at["set_joint_mode_end"]
    assert e.set_mode_end(ControlMode.VELOCITY) == 1  # 写入成功即返回 + 乐观更新缓存
    assert e.get_mode_end() == ControlMode.VELOCITY and e.mode_end == ControlMode.VELOCITY
    e._is_abled_end = False  # 设零须失能（门禁放行）
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
    cap.msgs.clear()
    assert e.get_error_end() is None and len(warns("状态快照过期")) >= 1
    e.get_state_end()  # 重新填槽恢复新鲜
    assert e.get_error_end() == [0]
    e.jstates["end"][0]["error"] = 9  # 欠压故障码
    assert e.get_error_end() == [0]  # 槽不跟手（仍是旧快照）
    e.get_state_end()
    assert e.get_error_end() == [9]
    cap.msgs.clear()
    assert e.check_error_end() is False and len(warns("存在故障状态码")) >= 1
    e.jstates["end"][0]["error"] = 0
    e.get_state_end()
    assert e.check_error_end() is True
    # clear_error：无失能门禁 → 任意使能态直接逐电机发清错帧
    e._is_abled_end = True  # 已使能：同样直接透传（失能时序由上层保证）
    e.calls.clear()
    assert e.clear_error_end() is True
    assert [c[1] for c in e.calls if c[0] == "clear_joint_error_end"] == [0]
    e._is_abled_end = False
    e.calls.clear()
    assert e.clear_error_end() is True
    assert [c[1] for c in e.calls if c[0] == "clear_joint_error_end"] == [0]
    e.fail_at["clear_joint_error_end"] = {0}
    cap.msgs.clear()
    assert e.clear_error_end() is False and len(warns("清错异常")) >= 1
    del e.fail_at["clear_joint_error_end"]


def test_35_end_send_mit(e, cfg):
    end_j = cfg["end"]["joints"]
    e.get_mode_end()  # 连接已乐观填缓存，get_mode 再核实
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
    cap.msgs.clear()
    e.calls.clear()
    e.send_mit_end([99.0], [-1.0], [0.0])
    c = [x for x in e.calls if x[0] == "send_joint_mit_end"]
    assert np.allclose([x[2] for x in c], e.joint_limits_end.tau_max)  # tau 裁到 ±tau_max
    assert len(warns("越限")) >= 1


def test_36_end_send_vel(e):
    assert e.set_mode_end(ControlMode.VELOCITY) == 1
    e.get_mode_end()  # set_mode 已乐观填缓存，get_mode 核实
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
        cap.msgs.clear()
        assert bna.enable_arm() == 1 and len(warns("未配置任何 joint")) >= 1  # 空族视为成功
    finally:
        bna._refresh_stop.set()


# ============================================================
# 五轴审查新增用例（独立实例：维度一维性 / 读参数标量校验 / cfg 鲁棒 / 非有限值 / 类型门禁 / 空族一致性）
# ============================================================

# ---- 指令二维输入（(n,1) 列向量）被一维性校验拦下，不触内核 ----
def test_42_cmd_2d_blocked(cfg):
    b2 = DummyBackend(cfg)
    try:
        b2.connect_arm()
        b2._refresh_stop.set()
        cap.msgs.clear()
        b2.calls.clear()
        b2.send_vel_arm(np.arange(6).reshape(6, 1))  # len==6 溜过长度检查，须被一维性拦下
        assert not any(c[0] == "send_joint_vel_arm" for c in b2.calls)
        assert len(warns("需为 (n,) 一维序列")) >= 1
        cap.msgs.clear()
        assert b2.write_param_arm("pos_kp", np.arange(6).reshape(6, 1)) == 0  # 写参数路径同样拦截
        assert len(warns("需为 (n,) 一维序列")) >= 1
        assert not any(c[0] == "write_joint_param_arm" for c in b2.calls)
    finally:
        b2._refresh_stop.set()


# ---- read_param 子类误返回序列（非标量）→ 指名 joint warn + 整族作废 ----
def test_43_read_param_nonscalar(cfg):
    b3 = DummyBackend(cfg)
    try:
        b3.connect_arm()
        b3._refresh_stop.set()
        orig = b3._read_joint_param_arm
        b3._read_joint_param_arm = lambda i, key: [orig(i, key)] if i == 2 else orig(i, key)
        cap.msgs.clear()
        assert b3.read_param_arm("pos_kp") is None
        assert len(warns("返回非标量")) >= 1
        assert b3._jname("arm", 2) in warns("返回非标量")[0]  # 指名 joint
    finally:
        b3._refresh_stop.set()


# ---- cfg 运行参数非法（非数值 / NaN / inf / 非正）→ warn + 沿用默认 ----
def test_44_cfg_pos_float_fallback(cfg):
    cfg_bad = copy.deepcopy(cfg)
    cfg_bad["state_refresh_hz"] = "abc"
    cfg_bad["write_settle"] = -1
    cap.msgs.clear()
    b4 = DummyBackend(cfg_bad)
    try:
        assert b4._refresh_hz == 20.0 and b4._write_settle == 0.1
        for k in ("state_refresh_hz", "write_settle"):
            assert any(k in m and "非法" in m for m in cap.msgs), k
    finally:
        b4._refresh_stop.set()


# ---- 状态含 NaN/inf → 按数据异常跳轮（不毒化 JointState 快照） ----
def test_45_state_nonfinite(cfg):
    b5 = DummyBackend(cfg)
    try:
        b5.connect_arm()
        b5._refresh_stop.set()
        b5.get_state_arm()
        q_prev = b5.joint_state_arm.q.copy()
        b5.jstates["arm"][1]["q"] = float("nan")
        b5.jstates["arm"][4]["tau"] = float("inf")
        cap.msgs.clear()
        assert b5.get_state_arm() is None
        assert len(warns("非有限值")) >= 1
        assert np.array_equal(b5.joint_state_arm.q, q_prev)  # 快照保持
        b5.jstates["arm"][1]["q"] = 0.2
        b5.jstates["arm"][4]["tau"] = 0.0
        assert b5.get_state_arm() is not None  # 恢复有限值后正常装配
    finally:
        b5._refresh_stop.set()


# ---- set_mode 非枚举入参 → 拦截且缓存不动 ----
def test_46_set_mode_type_gate(cfg):
    b6 = DummyBackend(cfg)
    try:
        b6._refresh_stop.set()  # 先冻结刷新线程再连接，避免后台读取干扰下方调用计数断言
        b6.connect_arm()
        cap.msgs.clear()
        b6.calls.clear()
        assert b6.set_mode_arm("position") == 0
        assert len(warns("非 ControlMode 枚举")) >= 1
        assert not any(c[0] == "set_joint_mode_arm" for c in b6.calls)
        assert b6.mode_arm == ControlMode.MIT  # 缓存不动（保持 connect 乐观值，类型拦截未触碰）
    finally:
        b6._refresh_stop.set()


# ---- 空族 write_param：值列表任意长度均视为成功跳过（空族语义优先于维度判定） ----
def test_47_write_param_empty_family(cfg):
    cfg_ne = copy.deepcopy(cfg)
    cfg_ne.pop("end", None)
    bne = DummyBackend(cfg_ne)
    try:
        cap.msgs.clear()
        assert bne.write_param_end("pos_kp", [1, 2, 3]) == 1
        assert len(warns("未配置任何 joint")) >= 1
        assert not any(c[0] == "write_joint_param_end" for c in bne.calls)
    finally:
        bne._refresh_stop.set()


# ============================================================
# 第九轮审查新增用例（独立实例：写参数非有限值 / 状态非字典拦截）
# ============================================================

# ---- 写参数值列表含 NaN/inf → 拦截不触内核（寄存器无限位源可裁，拦截而非裁剪） ----
def test_48_write_param_nonfinite(cfg):
    b7 = DummyBackend(cfg)
    try:
        b7.connect_arm()
        b7._refresh_stop.set()
        cap.msgs.clear()
        b7.calls.clear()
        assert b7.write_param_arm("pos_kp", [1.0, np.nan, 3, 4, 5, 6]) == 0
        assert len(warns("NaN/inf")) >= 1
        assert not any(c[0] == "write_joint_param_arm" for c in b7.calls)
        cap.msgs.clear()
        assert b7.write_param_arm("pos_kp", np.full(6, np.inf)) == 0
        assert len(warns("NaN/inf")) >= 1
        assert not any(c[0] == "write_joint_param_arm" for c in b7.calls)
        assert b7.write_param_arm("pos_kp", np.arange(1, 7)) == 1  # 有限值照常写入
    finally:
        b7._refresh_stop.set()


# ---- 内核误返回非字典（list）→ 指名 joint warn + 跳轮（防 in 退化成员测试产生全 None 假成功并毒化 t） ----
def test_49_state_nondict(cfg):
    b8 = DummyBackend(cfg)
    try:
        b8.connect_arm()
        b8._refresh_stop.set()
        b8.get_state_arm()
        q_prev, t_prev = b8.joint_state_arm.q.copy(), b8.joint_state_arm.t
        orig = b8._read_joint_state_arm
        b8._read_joint_state_arm = lambda i: [0.1, 0.0, 0] if i == 2 else orig(i)  # joint3 误返回 list
        cap.msgs.clear()
        assert b8.get_state_arm() is None
        assert len(warns("返回非字典")) >= 1
        assert b8._jname("arm", 2) in warns("返回非字典")[0]  # 指名 joint
        assert "list" in warns("返回非字典")[0]               # 附实际类型
        assert np.array_equal(b8.joint_state_arm.q, q_prev) and b8.joint_state_arm.t == t_prev  # 快照保持
        b8._read_joint_state_arm = orig
        assert b8.get_state_arm() is not None                 # 恢复字典后正常装配
    finally:
        b8._refresh_stop.set()


# ============================================================
# 第十轮审查新增用例（独立实例：内核假值返回统一拦截 / error 非有限值解析异常拦截）
# ============================================================

# ---- 内核误返回 None（假值）→ 与非字典统一：跳轮 + 指名 warn + 快照保持（不再 or {} 归一化为空字典假成功） ----
def test_50_state_none_return(cfg):
    b9 = DummyBackend(cfg)
    try:
        b9.connect_arm()
        b9._refresh_stop.set()
        b9.get_state_arm()
        q_prev, t_prev = b9.joint_state_arm.q.copy(), b9.joint_state_arm.t
        orig = b9._read_joint_state_arm
        b9._read_joint_state_arm = lambda i: None if i == 4 else orig(i)  # joint5 误返回 None
        cap.msgs.clear()
        assert b9.get_state_arm() is None
        assert len(warns("返回非字典")) >= 1
        assert b9._jname("arm", 4) in warns("返回非字典")[0]   # 指名 joint
        assert "NoneType" in warns("返回非字典")[0]            # 附实际类型
        assert np.array_equal(b9.joint_state_arm.q, q_prev) and b9.joint_state_arm.t == t_prev  # 快照保持
        b9._read_joint_state_arm = orig
        assert b9.get_state_arm() is not None                 # 恢复字典后正常装配
    finally:
        b9._refresh_stop.set()


# ---- error 值为 inf → 转整型解析失败按数据异常跳轮（不向调用者抛 OverflowError） ----
def test_51_error_inf_no_raise(cfg):
    b10 = DummyBackend(cfg)
    try:
        b10.connect_arm()
        b10._refresh_stop.set()
        b10.get_state_arm()
        q_prev, t_prev = b10.joint_state_arm.q.copy(), b10.joint_state_arm.t
        b10.jstates["arm"][1]["error"] = float("inf")
        cap.msgs.clear()
        st = b10.get_state_arm()  # 不得上抛（NumPy ≥1.24 的 OverflowError 须被拦截转 warn）
        assert st is None
        assert len(warns("解析失败")) >= 1
        assert np.array_equal(b10.joint_state_arm.q, q_prev) and b10.joint_state_arm.t == t_prev
        b10.jstates["arm"][1]["error"] = 0
        assert b10.get_state_arm() is not None                # 恢复有限码后正常装配
    finally:
        b10._refresh_stop.set()


# ============================================================
# 第十一轮审查新增用例（独立实例：状态子值非标量拦截）
# ============================================================

# ---- 状态子值误返回序列（如 [0.1]）→ 拦截防装配出 (n,1) 字段毒化上层 + 快照保持 ----
def test_52_state_value_nonscalar(cfg):
    b11 = DummyBackend(cfg)
    try:
        b11.connect_arm()
        b11._refresh_stop.set()
        b11.get_state_arm()
        q_prev, t_prev = b11.joint_state_arm.q.copy(), b11.joint_state_arm.t
        orig = b11._read_joint_state_arm
        b11._read_joint_state_arm = lambda i: {**orig(i), "q": [0.1]} if i == 3 else orig(i)  # joint4 子值误返回 list
        cap.msgs.clear()
        assert b11.get_state_arm() is None
        assert len(warns("非标量")) >= 1
        assert b11._jname("arm", 3) in warns("非标量")[0]       # 指名 joint
        assert "『q』" in warns("非标量")[0] and "list" in warns("非标量")[0]  # 指名字段 + 实际类型
        assert b11.joint_state_arm.q.shape == (6,)             # 快照保持 (n,) 不被 (n,1) 毒化
        assert np.array_equal(b11.joint_state_arm.q, q_prev) and b11.joint_state_arm.t == t_prev
        b11._read_joint_state_arm = orig
        assert b11.get_state_arm() is not None                 # 恢复标量后正常装配
    finally:
        b11._refresh_stop.set()


# ============================================================
# 第十二轮审查新增用例（独立实例：模式读非枚举拦截 / 限位 NaN 回退 / cfg 段非字典容错）
# ============================================================

# ---- 内核误返回非 ControlMode（int）→ 整族不更新 + 指名 warn + 缓存不毒化（防刷新线程永不自愈） ----
def test_53_mode_nonenum(cfg):
    b12 = DummyBackend(cfg)
    try:
        b12.connect_arm()
        b12._refresh_stop.set()
        assert b12.get_mode_arm() == ControlMode.MIT  # 连接后正常读回填充缓存
        orig = b12._read_joint_mode_arm
        b12._read_joint_mode_arm = lambda i: 2 if i == 3 else orig(i)  # joint4 误返回 int
        cap.msgs.clear()
        assert b12.get_mode_arm() is None
        assert len(warns("非 ControlMode 枚举")) >= 1
        assert b12._jname("arm", 3) in warns("非 ControlMode 枚举")[0]  # 指名 joint
        assert "int" in warns("非 ControlMode 枚举")[0]                 # 附实际类型
        assert b12._joint_mode_arm == [ControlMode.MIT] * 6  # 逐 joint 缓存不被毒化
        assert b12.mode_arm == ControlMode.MIT               # 族模式成员不动
        b12._read_joint_mode_arm = orig
        assert b12.get_mode_arm() == ControlMode.MIT         # 恢复枚举后正常读取
    finally:
        b12._refresh_stop.set()


# ---- cfg 限位键为 NaN → 回退未限位 + warn（防 np.clip 产出 NaN 指令静默下发）；q_min > q_max → 启动 warn ----
def test_54_limits_nan_and_reversed(cfg):
    cfg_ln = copy.deepcopy(cfg)
    cfg_ln["arm"]["joints"][0]["q_min"] = float("nan")
    cap.msgs.clear()
    b13 = DummyBackend(cfg_ln)
    try:
        assert b13.joint_limits_arm.q_min[0] == -np.inf  # NaN 回退为未限位
        assert any("为 NaN" in m and "q_min" in m for m in cap.msgs)
        assert np.isfinite(b13.joint_limits_arm.q_max).all()  # 其余键不受影响
    finally:
        b13._refresh_stop.set()
    cfg_rv = copy.deepcopy(cfg)
    cfg_rv["arm"]["joints"][1]["q_min"] = 99.0  # > q_max（配置错误）
    cap.msgs.clear()
    b14 = DummyBackend(cfg_rv)
    try:
        assert b14.joint_limits_arm.q_min[1] == 99.0  # 值保留（不阻断），仅启动 warn 提示
        assert any("q_min > q_max" in m for m in cap.msgs)
    finally:
        b14._refresh_stop.set()


# ---- cfg 段误配为非字典（list）→ init 不崩，该关节 NaN 占位 + warn（兑现 docstring 容错承诺） ----
def test_55_section_nondict(cfg):
    cfg_nd2 = copy.deepcopy(cfg)
    cfg_nd2["arm"]["joints"][0]["MIT"] = [0.1, 0.02]
    cap.msgs.clear()
    b15 = DummyBackend(cfg_nd2)
    try:
        assert np.isnan(b15._kp_mit_default_arm[0]) and np.isnan(b15._kd_mit_default_arm[0])
        assert any("MIT.kp" in m for m in cap.msgs) and any("MIT.kd" in m for m in cap.msgs)
        assert np.isfinite(b15._kp_mit_default_arm[1:]).all()  # 其余关节不受影响
        assert np.isfinite(b15._pos_kp_arm).all()              # POS_VEL 段正常解析
    finally:
        b15._refresh_stop.set()


# ---- init 段缺省静默：整段缺失（如舵机无 MIT 段）不告警；段在键缺仍告警 ----
def test_57_extract_section_absent_silent(cfg):
    cfg_sv = copy.deepcopy(cfg)
    for j in cfg_sv["arm"]["joints"]:
        del j["MIT"]  # 整段缺省：模拟无 MIT 能力的舵机配置
    cap.msgs.clear()
    bsv = DummyBackend(cfg_sv)
    try:
        assert np.isnan(bsv._kp_mit_default_arm).all() and np.isnan(bsv._kd_mit_default_arm).all()
        assert not any("MIT.kp" in m or "MIT.kd" in m for m in cap.msgs)  # 整段缺省 → 静默
        assert np.isfinite(bsv._vlim_default_arm).all()                    # 其余段正常解析
    finally:
        bsv._refresh_stop.set()
    cfg_key = copy.deepcopy(cfg)
    del cfg_key["arm"]["joints"][0]["MIT"]["kp"]  # 段在键缺：仍要告警
    cap.msgs.clear()
    bk = DummyBackend(cfg_key)
    try:
        assert np.isnan(bk._kp_mit_default_arm[0]) and np.isfinite(bk._kp_mit_default_arm[1:]).all()
        assert any("MIT.kp" in m for m in cap.msgs)
    finally:
        bk._refresh_stop.set()


# ---- default_mode：cfg 可配连接后自动设置的模式（无 MIT 型号消噪）；非法值回退 MIT 并 warn ----
def test_58_default_mode(cfg):
    cfg_dm = copy.deepcopy(cfg)
    cfg_dm["arm"]["default_mode"] = "position"  # 无 MIT 型号（如舵机）配 position
    bdm = DummyBackend(cfg_dm)
    try:
        assert bdm._default_mode_arm == ControlMode.POSITION
        bdm.connect_arm()
        bdm._refresh_stop.set()
        assert bdm.get_mode_arm() == ControlMode.POSITION  # 连接自动设置的是 position
    finally:
        bdm._refresh_stop.set()
    cfg_bad = copy.deepcopy(cfg)
    cfg_bad["end"]["default_mode"] = "torque"  # 非法值 → 回退 MIT
    cap.msgs.clear()
    bbad = DummyBackend(cfg_bad)
    try:
        assert bbad._default_mode_end == ControlMode.MIT
        assert any("default_mode" in m for m in cap.msgs)
    finally:
        bbad._refresh_stop.set()


# ============================================================
# 第十四轮审查新增用例（基类逐功能审阅补盲：门禁全覆盖 / 内核异常路径 / 裁剪对称 /
# 空族读写断连 / 连接编排细节 / close 慢退出与单拍重试 / 失能失败不阻断断连）
# ============================================================

# ---- 未连接门禁全覆盖：全部公开操作拦截且不触任何内核 ----
def test_59_unconnected_gate_sweep(cfg):
    r = DummyBackend(cfg)
    try:
        r._refresh_stop.set()
        cap.msgs.clear()
        assert r.read_param_arm("pos_kp") is None
        assert r.write_param_arm("pos_kp", np.arange(6)) == 0
        assert r.set_mode_arm(ControlMode.MIT) == 0
        assert r.set_zero_arm() == 0
        assert r.get_mode_arm() is None
        assert r.get_error_arm() is None
        assert r.check_error_arm() is False
        assert r.clear_error_arm() is False
        r.send_mit_arm(np.zeros(6), np.zeros(6), np.zeros(6))
        r.send_position_arm(np.zeros(6))
        r.send_action_end("open")
        assert len(warns("未连接或状态未知")) >= 1
        assert not r.calls  # 任何内核都未被触达
    finally:
        r._refresh_stop.set()


# ---- 内核异常路径（区别于返回 False）：使能/失能/设零异常均转 warn，状态停留 None ----
def test_60_kernel_exception_paths(cfg):
    r = DummyBackend(cfg)
    try:
        r._refresh_stop.set()
        r.connect_arm()
        r.fail_at["enable_joint_arm"] = {2}
        cap.msgs.clear()
        assert r.enable_arm() == 0 and r.is_abled_arm is None
        assert len(warns("使能异常")) >= 1
        del r.fail_at["enable_joint_arm"]
        r.fail_at["disable_joint_arm"] = {3}
        cap.msgs.clear()
        assert r.disable_arm() == 0
        assert len(warns("失能异常")) >= 1
        del r.fail_at["disable_joint_arm"]
        r.fail_at["set_joint_zero_arm"] = {1}
        cap.msgs.clear()
        assert r.set_zero_arm() == 0
        assert len(warns("设零异常")) >= 1
    finally:
        r._refresh_stop.set()


# ---- 清错内核返回 False（区别于异常）→ False + 指名 warn；connect 内核异常 → 停留 None ----
def test_61_clear_retfalse_and_connect_exc(cfg):
    r = DummyBackend(cfg)
    try:
        r._refresh_stop.set()
        r.connect_arm()
        r.ret_false["clear_joint_error_arm"] = {4}
        cap.msgs.clear()
        assert r.clear_error_arm() is False
        assert len(warns("清错失败")) >= 1 and r._jname("arm", 4) in warns("清错失败")[0]
    finally:
        r._refresh_stop.set()
    r2 = DummyBackend(cfg)
    try:
        r2._refresh_stop.set()
        r2.fail_at["connect_arm"] = {0}
        cap.msgs.clear()
        r2.connect_arm()
        assert r2.is_connected_arm is None  # 异常与返回 False 同待遇：停留未知态
        assert len(warns("连接内核异常")) >= 1
    finally:
        r2._refresh_stop.set()


# ---- arm MIT 裁剪对称：tau/q/dq 越限 → 就近裁剪后进内核（kp/kd 不裁） ----
def test_62_mit_clip_arm(cfg):
    r = DummyBackend(cfg)
    try:
        r._refresh_stop.set()
        r.connect_arm()
        assert r.get_mode_arm() == ControlMode.MIT  # 连接已乐观填缓存，get_mode 核实
        r.calls.clear()
        cap.msgs.clear()
        lim = r.joint_limits_arm
        tau = np.array([99.0, -99.0, 0, 0, 0, 0])
        q = np.array([99.0, -99.0, 0, 0, 0, 0])
        dq = np.array([99.0, 0, 0, 0, 0, 0])
        r.send_mit_arm(tau, q, dq)
        calls = [c for c in r.calls if c[0] == "send_joint_mit_arm"]
        assert len(calls) == 6
        assert np.allclose([c[2] for c in calls], np.clip(tau, -lim.tau_max, lim.tau_max))
        assert np.allclose([c[3] for c in calls], np.clip(q, lim.q_min, lim.q_max))
        assert np.allclose([c[4] for c in calls], np.clip(dq, -lim.dq_max, lim.dq_max))
        assert len(warns("指令越限")) >= 1
    finally:
        r._refresh_stop.set()


# ---- end 位置裁剪 + vlim/flim 显式覆盖：三者越限各自就近裁剪（各自独立 warn） ----
def test_63_end_position_clip(cfg):
    r = DummyBackend(cfg)
    try:
        r._refresh_stop.set()
        r.connect_end()
        r.set_mode_end(ControlMode.POSITION)
        assert r.get_mode_end() == ControlMode.POSITION  # set_mode 已乐观填缓存，get_mode 核实
        lim = r.joint_limits_end
        q_ok = float(lim.q_max[0]) - 0.1
        r.calls.clear()
        cap.msgs.clear()
        r.send_position_end([99.0])  # q 越限（vlim/flim 用 cfg 缺省）
        c = next(c for c in r.calls if c[0] == "send_joint_position_end")
        assert np.isclose(c[2], lim.q_max[0]) and 0.0 <= c[3] <= lim.dq_max[0] and 0.0 <= c[4] <= 1.0
        assert len(warns("q 指令越限")) >= 1
        r.calls.clear()
        cap.msgs.clear()
        v_ok = float(lim.dq_max[0]) * 0.5
        r.send_position_end([q_ok], vlim=[99.0], flim=[0.5])  # 仅 vlim 越限
        c = next(c for c in r.calls if c[0] == "send_joint_position_end")
        assert np.isclose(c[3], lim.dq_max[0]) and np.isclose(c[2], q_ok) and np.isclose(c[4], 0.5)  # 显式 flim 原样透传
        assert len(warns("vlim 超出")) >= 1
        r.calls.clear()
        cap.msgs.clear()
        r.send_position_end([q_ok], vlim=[v_ok], flim=[5.0])  # 仅 flim 越限
        c = next(c for c in r.calls if c[0] == "send_joint_position_end")
        assert np.isclose(c[4], 1.0) and np.isclose(c[3], v_ok)
        assert len(warns("flim 超出")) >= 1
    finally:
        r._refresh_stop.set()


# ---- 空族补齐：读/设/断连的空族早退（get_state→None、get_mode→None、read_param→[]、写类→1、断连不发帧） ----
def test_64_empty_family_read_write_disconnect(cfg):
    cfg_ne = copy.deepcopy(cfg)
    cfg_ne.pop("end", None)
    r = DummyBackend(cfg_ne)
    try:
        r._refresh_stop.set()
        assert r.get_state_end() is None
        assert r.get_mode_end() is None
        assert r.read_param_end("pos_kp") == []
        assert r.set_mode_end(ControlMode.MIT) == 1
        assert r.set_zero_end() == 1
        r.disconnect_end()  # 空族断连：空操作
        assert not any(c[0].endswith("_end") for c in r.calls)  # 全程不触任何 end 内核
    finally:
        r._refresh_stop.set()


# ---- 连接编排细节：默认模式设置失败仅 warn 不回滚连接；protocol 覆盖参数同步成员并透传内核 ----
def test_65_connect_default_mode_fail_and_protocol(cfg):
    r = DummyBackend(cfg)
    try:
        r._refresh_stop.set()
        r.fail_at["set_joint_mode_arm"] = {0}
        cap.msgs.clear()
        r.connect_arm()
        assert r.is_connected_arm is True  # 默认模式失败不影响连接成功
        assert len(warns("默认模式（MIT）设置失败")) >= 1
        r.connect_arm(protocol="can_raw")  # 显式 protocol 覆盖
        assert r._protocol_arm == "can_raw"
        assert any(c[0] == "connect_arm" and c[2] == "can_raw" for c in r.calls)
    finally:
        r._refresh_stop.set()


# ---- close() 慢退出：单拍阻塞超过 join 超时 → warn（不抛、不强制杀线程） ----
def test_66_close_slow_exit_warn(cfg):
    r = DummyBackend(cfg)
    release = threading.Event()
    tick_entered = threading.Event()
    orig = r._refresh_tick

    def slow_tick(period):
        tick_entered.set()
        release.wait(5.0)
        orig(period)

    r._refresh_tick = slow_tick
    try:
        assert tick_entered.wait(1.0)  # 等线程进入单拍（阻塞在总线读取的场景模拟）
        cap.msgs.clear()
        r.close()
        assert len(warns("1 秒内未退出")) >= 1
        release.set()
        r._refresh_thread.join(2.0)
        assert not r._refresh_thread.is_alive()  # 单拍结束后自行退出
    finally:
        release.set()
        r._refresh_stop.set()


# ---- 刷新单拍异常 → warn 且下周期重试（线程不死、恢复后状态照常装配） ----
def test_67_refresh_tick_exception_retry(cfg):
    r = DummyBackend(cfg)
    boom = {"on": False}
    orig = r._refresh_tick

    def flaky_tick(period):
        if boom["on"]:
            raise RuntimeError("tick boom")
        orig(period)

    try:
        r.connect_arm()
        r._refresh_tick = flaky_tick
        r.joint_state_arm.t = 0.0  # 人为陈旧，保证恢复后的拍会真正读状态
        boom["on"] = True
        cap.msgs.clear()
        time.sleep(0.25)  # 10Hz：至少 2 拍异常
        assert len(warns("单拍异常")) >= 1
        boom["on"] = False
        t0 = time.monotonic()
        while r.joint_state_arm.t <= 0.0 and time.monotonic() - t0 < 1.0:
            time.sleep(0.02)  # 等一个成功拍重新装配（t > 0）
        assert r.joint_state_arm.t > 0.0
    finally:
        boom["on"] = False
        r._refresh_stop.set()


# ---- 断连前尽力失能：失能部分失败仅提示，不阻断断连 ----
def test_68_disconnect_despite_disable_failure(cfg):
    r = DummyBackend(cfg)
    try:
        r._refresh_stop.set()
        r.connect_arm()
        r.enable_arm()
        r.ret_false["disable_joint_arm"] = {2}
        cap.msgs.clear()
        r.disconnect_arm()
        assert r.is_connected_arm is False  # 断连仍完成
        assert len(warns("失能失败")) >= 1
        assert any(c[0] == "disconnect_arm" for c in r.calls)
    finally:
        r._refresh_stop.set()


# ---- 模板冒烟：BackendTemplate 空 cfg 可实例化（29 桩满足抽象方法强制）、29 桩全量按约定格式上抛 ----
def test_69_template_smoke():
    from joyarm_core.backend.backend_template import BackendTemplate  # 函数内导入，模板问题不拖垮整文件收集

    # 29 个抽象内核的示例入参（与 backend.py 抽象方法签名一一对应）
    kernel_args = {
        "_connect_arm": ("/dev/null", "demo"), "_connect_end": ("/dev/null", "demo"),
        "_disconnect_arm": (), "_disconnect_end": (),
        "_enable_joint_arm": (0,), "_enable_joint_end": (0,),
        "_disable_joint_arm": (0,), "_disable_joint_end": (0,),
        "_read_joint_param_arm": (0, "pos_kp"), "_read_joint_param_end": (0, "pos_kp"),
        "_read_joint_mode_arm": (0,), "_read_joint_mode_end": (0,),
        "_read_joint_state_arm": (0,), "_read_joint_state_end": (0,),
        "_write_joint_param_arm": (0, "pos_kp", 1.0), "_write_joint_param_end": (0, "pos_kp", 1.0),
        "_set_joint_mode_arm": (0, ControlMode.MIT), "_set_joint_mode_end": (0, ControlMode.MIT),
        "_set_joint_zero_arm": (0,), "_set_joint_zero_end": (0,),
        "_send_joint_mit_arm": (0, 0.0, 0.0, 0.0, 0.0, 0.0), "_send_joint_mit_end": (0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "_send_joint_position_arm": (0, 0.0, 0.0, 0.0), "_send_joint_position_end": (0, 0.0, 0.0, 0.0),
        "_send_joint_vel_arm": (0, 0.0), "_send_joint_vel_end": (0, 0.0),
        "_send_action_end": ("open",),
        "_clear_joint_error_arm": (0,), "_clear_joint_error_end": (0,),
    }
    assert set(kernel_args) == Backend.__abstractmethods__  # 基类增删/改名内核即红，提醒同步模板与本表
    t = BackendTemplate({})
    try:
        assert t.n_joints_arm == 0 and t.n_joints_end == 0  # 空 cfg 容错构造，不做任何总线 I/O
        assert t.is_connected_arm is None and t.is_abled_end is None
        for name, args in kernel_args.items():  # 29 桩全量：『文件名 - 方法：』格式上抛 NotImplementedError
            with pytest.raises(NotImplementedError, match=rf"backend_<型号>\.py - {name}"):
                getattr(t, name)(*args)
    finally:
        t._refresh_stop.set()
    t2 = BackendTemplate({"name": "backend_demo"})  # 带注册名构造：桩消息前缀自动切换为 _name
    try:
        assert t2.name == "backend_demo"
        with pytest.raises(NotImplementedError, match=r"backend_demo - _connect_arm"):
            t2._connect_arm("/dev/null", "demo")
    finally:
        t2._refresh_stop.set()


# ---- backend_dm 冒烟：无硬件校验协议层（组帧/解码/分发/事务自环）与注册（函数内导入，硬件问题不拖垮收集） ----
def test_70_backend_dm(cfg):
    import struct as _struct

    from joyarm_core.backend import REGISTRY, get_backend
    from joyarm_core.backend.backend_dm import (
        _CODE_MODE, _DMMotor, _DMBus, _MODEL_LIMITS, _MODE_CODE, _PARAM_RIDS,
        BackendDM, _f2u, _pack_mit, _pack_pos, _pack_vel, _u2can_prefix)

    # 1) 注册表恢复
    assert REGISTRY.get("backend_dm") is BackendDM and get_backend("backend_dm") is BackendDM

    # 2) 空 cfg 可构造（无总线 I/O）
    b = BackendDM({})
    try:
        assert b.n_joints_arm == 0 and b.n_joints_end == 0
    finally:
        b._refresh_stop.set()

    # 3) yaml cfg 构造：电机表 / 型号量程 / U2CAN 帧前缀（CAN id 小端嵌 13-14、DLC=8 嵌 18）
    d = BackendDM(cfg)
    try:
        assert d.name == "backend_dm" and d.n_joints_arm == 6 and d.n_joints_end == 1
        m0, m3 = d._motors_arm[0], d._motors_arm[3]
        assert (m0.slave_id, m0.master_id) == (0x01, 0x11)
        assert (m0.pmax, m0.vmax, m0.tmax) == _MODEL_LIMITS["4340p"]
        assert (m3.pmax, m3.vmax, m3.tmax) == _MODEL_LIMITS["4310"]
        assert len(_u2can_prefix(0)) == 21 and m0.prefix[13] == 0x01 and m0.prefix[18] == 0x08
        assert m0.prefix_pos[13:15] == bytes((0x01, 0x03))  # pos_force 帧 CAN id = 0x300+SlaveID
        assert m0.prefix_vel[13:15] == bytes((0x01, 0x02))  # vel 帧 CAN id = 0x200+SlaveID
        assert m0.query_frame[21:24] == bytes((0x01, 0x00, 0xCC)) and len(m0.query_frame) == 30
    finally:
        d._refresh_stop.set()

    # 4) MIT 位域打包位级校验（对照厂商 float_to_uint 公式独立重推）+ 越界就近钳位
    def ref_f2u(x, lo, hi, bits):
        x = lo if x <= lo else hi if x > hi else x
        return int((x - lo) / (hi - lo) * ((1 << bits) - 1))

    m = _DMMotor(0x01, 0x11, 12.5, 30.0, 10.0)
    q_u, dq_u = ref_f2u(1.23, -12.5, 12.5, 16), ref_f2u(-4.5, -30.0, 30.0, 12)
    kp_u, kd_u, tau_u = ref_f2u(120.0, 0, 500, 12), ref_f2u(4.9, 0, 5, 12), ref_f2u(3.3, -10.0, 10.0, 12)
    ref = bytes(((q_u >> 8) & 0xFF, q_u & 0xFF, dq_u >> 4, ((dq_u & 0xF) << 4) | ((kp_u >> 8) & 0xF),
                 kp_u & 0xFF, kd_u >> 4, ((kd_u & 0xF) << 4) | ((tau_u >> 8) & 0xF), tau_u & 0xFF))
    assert _pack_mit(m, 3.3, 1.23, -4.5, 120.0, 4.9) == ref
    assert _pack_mit(m, 999.0, 999.0, 999.0, 999.0, 999.0) == _pack_mit(m, 10.0, 12.5, 30.0, 500.0, 5.0)
    # 负增益必须钳到 0（防位域编码成大整数 → 真机异常力矩）；上界同理钳满量程
    assert _pack_mit(m, 3.3, 1.23, -4.5, -10.0, -0.5) == _pack_mit(m, 3.3, 1.23, -4.5, 0.0, 0.0)
    assert _f2u(-1.0, 0.0, 500.0, 12) == 0 and _f2u(1e9, 0.0, 5.0, 12) == 4095

    # 5) pos_force / vel 帧数据布局（float32 + uint16 放大 / float32 + 零填充）
    assert _pack_pos(1.0, 5.0, 0.5) == _struct.pack("<fHH", 1.0, 500, 5000)
    assert _pack_vel(-3.25) == _struct.pack("<f", -3.25) + b"\x00" * 4

    # 6) 帧分发：状态帧 → 原子快照（err 高 4 位 + data[6]/[7] 双温度）；前置垃圾跳过 + 残尾跨读拼接
    bus = _DMBus("/dev/null", 921600, 1000.0)  # 构造不开串口（open() 才开），直接喂字节
    bus.register([m])
    q_u, dq_u, tau_u = 0xBFFF, 0x800, 0x400
    # 收帧 16B：[0]=AA [1]=CMD [3..6]=CAN id（低字节在前，见厂商公式 p[3]|p[4]<<8|…） [7..14]=数据 [15]=55
    st_frame = bytes((0xAA, 0x11, 0x00, 0x11, 0, 0, 0,
                      0x11, q_u >> 8, q_u & 0xFF, dq_u >> 4,
                      ((dq_u & 0xF) << 4) | (tau_u >> 8), tau_u & 0xFF, 42, 55, 0x55))
    bus._feed(b"\x00" + st_frame)  # 前置垃圾字节被逐个跳过
    t, q, dq, tau, err, tmos, trot = m.state
    assert t > 0.0 and err == 1 and tmos == 42 and trot == 55
    assert q == pytest.approx(-12.5 + 0xBFFF * 25.0 / 65535)
    assert dq == pytest.approx(-30.0 + 0x800 * 60.0 / 4095)
    assert tau == pytest.approx(-10.0 + 0x400 * 20.0 / 4095)
    assert m.state_event.is_set()
    bus._feed(st_frame[:9])  # 半帧 → 残尾保留
    assert len(bus._rx_buf) == 9
    bus._feed(st_frame[9:])  # 后半帧 → 拼接还原完整帧再分发
    assert bus._rx_buf == b"" and m.state[4] == 1

    # 6b) 解码边界：q_u=0/满量程 → ∓PMAX；err=0xF；温度 0/255
    bus._feed(bytes((0xAA, 0x11, 0x00, 0x11, 0, 0, 0, 0xF0, 0, 0, 0, 0, 0, 0, 255, 0x55)))
    _, q0, _, _, err0, tm0, tr0 = m.state
    assert q0 == pytest.approx(-12.5) and err0 == 15 and tm0 == 0 and tr0 == 255
    bus._feed(bytes((0xAA, 0x11, 0x00, 0x11, 0, 0, 0, 0x11, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0x55)))
    _, q1, dq1, tau1, err1, _, _ = m.state
    assert (q1, dq1, tau1, err1) == (pytest.approx(12.5), pytest.approx(30.0), pytest.approx(10.0), 1)

    # 6c) 分发隔离：data[2]∈{0x33,0x55} 的帧只进参数应答槽，绝不误写状态缓存（反向零污染）
    st_ref = m.state
    bus._feed(bytes((0xAA, 0x11, 0x00, 0x11, 0, 0, 0, 0x01, 0x10, 0x55, 0, 0, 0, 0, 0, 0x55)))  # dq 高字节撞 0x55
    assert m.state is st_ref
    bus._feed(bytes((0xAA, 0x11, 0x00, 0x11, 0, 0, 0, 0x01, 0x00, 0x33, 27)) + _struct.pack("<f", 9.9)
              + bytes((0x55,)))
    assert m.state is st_ref  # 参数应答同样不落状态缓存

    # 7) 参数事务自环（write 打桩：发送即同步喂回应答帧，应答时刻 ≥ 发送前 t0，防洗白检查通过）
    frames = []

    def echo_write(frame):
        frames.append(frame)
        dd = frame[21:29]
        val = dd[4:8] if dd[2] == 0x55 else (  # 写回显原值；读按 RID 造值（10=uint32 模式码 / 27=float32 增益）
            _struct.pack("<I", 4) if dd[3] == 10 else _struct.pack("<f", 150.0))
        bus._feed(bytes((0xAA, 0x11, 0x00, 0x11, 0, 0, 0)) + bytes((dd[0], dd[1], dd[2], dd[3])) + val
                  + bytes((0x55,)))

    bus.write = echo_write
    assert bus.param_read(m, 10) == 4                      # uint32 模式寄存器读
    assert frames[-1][:21] == _u2can_prefix(0x7FF) and frames[-1][29] == 0    # U2CAN 封装 = 前缀+数据+尾
    assert frames[-1][21:29] == bytes((0x01, 0x00, 0x33, 10, 0, 0, 0, 0))     # 0x33 读帧字节（RID10）
    assert bus.param_read(m, 27) == pytest.approx(150.0)   # float32 增益读
    bus.param_write(m, 27, 150.0)                          # 回显一致 → 通过
    assert frames[-1][21:29] == bytes((0x01, 0x00, 0x55, 27)) + _struct.pack("<f", 150.0)  # float 写帧
    bus.param_write(m, 10, 4)
    assert frames[-1][21:29] == bytes((0x01, 0x00, 0x55, 10, 4, 0, 0, 0))     # uint32 写帧（模式码小端）

    def bad_echo(frame):  # 回显值被篡改 → 重试耗尽上抛
        dd = frame[21:29]
        bus._feed(bytes((0xAA, 0x11, 0x00, 0x11, 0, 0, 0)) + bytes((dd[0], dd[1], dd[2], dd[3]))
                  + _struct.pack("<f", -1.0) + bytes((0x55,)))

    bus.write = bad_echo
    with pytest.raises(TimeoutError):
        bus.param_write(m, 27, 150.0)

    # 8) 使能/失能/设零/清错指令验证（任何帧都回 err=0 状态帧：失能/清错/设零通过，使能核对不过）
    def ack_state(frame):
        frames.append(frame)
        bus._feed(bytes((0xAA, 0x11, 0x00, 0x11, 0, 0, 0, 0x01, 0, 0, 0, 0, 0, 0, 0, 0x55)))

    bus.write = ack_state
    assert bus.cmd_verified(m, 0xFD, lambda e: e != 1) is True
    assert frames[-1] == m.prefix + b"\xff" * 7 + b"\xfd" + b"\x00"  # 失能帧整帧字节（FF×7+cmd）
    assert bus.cmd_verified(m, 0xFB, lambda e: e < 2) is True
    assert frames[-1][28] == 0xFB                                     # 清错帧命令字节
    assert bus.cmd_verified(m, 0xFE, None) is True
    assert frames[-1][28] == 0xFE                                     # 设零帧命令字节
    with pytest.raises(TimeoutError):
        bus.cmd_verified(m, 0xFC, lambda e: e == 1)
    assert frames[-1][28] == 0xFC                                     # 使能帧命令字节

    # 8b) 指令帧无应答 → 自动补发 0xCC 查询兜底成功
    def ack_query_only(frame):
        frames.append(frame)
        if frame[23] == 0xCC:  # 只应答 0xCC 查询帧（指令帧本身不回，触发兜底路径）
            bus._feed(bytes((0xAA, 0x11, 0x00, 0x11, 0, 0, 0, 0x11, 0, 0, 0, 0, 0, 0, 0, 0x55)))

    bus.write = ack_query_only
    assert bus.cmd_verified(m, 0xFC, lambda e: e == 1) is True
    assert frames[-2][28] == 0xFC and frames[-1] == m.query_frame  # 先指令帧（超时）后 0xCC 查询帧

    # 9) 状态读取非阻塞三态：硬陈旧先发刷新帧再上抛（恢复路径不锁死）/ 新鲜直返零帧 / 软陈旧发一帧 0xCC 即返当前快照
    dm0 = d._motors_arm[0]
    bus2 = _DMBus("/dev/null", 921600, 1000.0)
    d._bus_arm = bus2
    sent = []
    bus2.write = sent.append
    with pytest.raises(TimeoutError):  # 初值快照（t=0）远超硬陈旧阈值 → 上抛
        d._read_state("arm", 0)
    assert sent == [dm0.query_frame]  # 上抛前已发刷新帧：电机在岸则下一拍自愈
    dm0.state = (time.monotonic(), 1.0, 2.0, 3.0, 1, 40, 41)  # 模拟刷新应答已到（新鲜快照）
    bus2.write = lambda frame: pytest.fail("新鲜缓存读取不应发总线帧")
    assert d._read_state("arm", 0) == {"q": 1.0, "dq": 2.0, "tau": 3.0, "error": 1,
                                       "temp_mos": 40, "temp_rotor": 41}
    dm0.state = (time.monotonic() - 0.06, 9.0, 8.0, 7.0, 0, 30, 31)  # 陈旧（>50ms 且 <1s）
    sent = []
    bus2.write = sent.append
    snap = d._read_state("arm", 0)  # 发一帧 0CCR 后立即返回当前（陈旧）缓存，不等待
    assert len(sent) == 1 and sent[0] == dm0.query_frame
    assert snap["q"] == 9.0 and snap["error"] == 0

    # 10) 模式映射与参数键表
    assert _MODE_CODE[ControlMode.POSITION] == 4 and _MODE_CODE[ControlMode.VELOCITY] == 3
    assert _CODE_MODE.get(2) is None  # DM POS_VEL 码在全库枚举外 → None
    assert {"pos_kp", "pos_ki", "vel_kp", "vel_ki"} <= set(_PARAM_RIDS)


# ---- backend_dm 总线生命周期：RX 存活判定 / 僵尸重建 / 无主清扫（审阅轮新增，防真机重连失败与资源泄漏） ----
def test_71_backend_dm_bus_lifecycle():
    from joyarm_core.backend.backend_dm import _DMBus, BackendDM

    # 未 open 的总线不可用（串口/RX 均无）；close 幂等
    bus = _DMBus("/dev/null", 921600, 1000.0)
    assert bus.is_open is False
    bus.close()
    assert bus.is_open is False

    d = BackendDM({})
    try:
        # 连接中途失败遗留的孤儿总线：断连时被清扫（不泄漏 RX 线程/串口）
        orphan = _DMBus("/dev/null", 921600, 1000.0)
        d._buses["/dev/null"] = orphan
        d._disconnect_family("arm")  # 未连接（_bus_arm=None）也执行清扫
        assert "/dev/null" not in d._buses

        # 僵尸总线（串口或 RX 死亡）：重连时先关再重建（不残留、不双开）；此处新建走真串口必失败
        zombie = _DMBus("/dev/__not_exist__", 921600, 1000.0)
        d._buses["/dev/__not_exist__"] = zombie
        with pytest.raises(Exception):
            d._connect_family("arm", "/dev/__not_exist__")
        assert "/dev/__not_exist__" not in d._buses and d._bus_arm is None

        # 共享总线：对端仍持有时断连保留；最后一族断开才关停移除
        shared = _DMBus("/dev/ttyUSBX", 921600, 1000.0)
        d._buses["/dev/ttyUSBX"] = shared
        d._bus_arm = d._bus_end = shared
        d._disconnect_family("arm")
        assert d._buses["/dev/ttyUSBX"] is shared
        d._disconnect_family("end")
        assert "/dev/ttyUSBX" not in d._buses
    finally:
        d._refresh_stop.set()


# ---- backend_dm 末端离散动作：open/home/close/position（自动切 POSITION + cfg vlim/flim 默认 + 基类限位管道） ----
def test_72_backend_dm_action_end(cfg):
    import struct as _struct

    from joyarm_core.backend.backend_dm import _END_ACTION_POS, _DMBus, BackendDM, _pack_pos

    d = BackendDM(cfg)
    try:
        me = d._motors_end[0]
        bus = _DMBus("/dev/null", 921600, 1000.0)
        bus.register([me])
        d._bus_end = bus
        d._is_connected_end = True
        frames = []

        def echo_write(frame):  # 0x55 参数写即时回显（模式/增益切换事务自环），其余帧仅记录
            frames.append(frame)
            dd = frame[21:29]
            if dd[2] == 0x55:
                bus._feed(bytes((0xAA, 0x11, 0x00, me.master_id, 0, 0, 0))
                          + bytes((dd[0], dd[1], dd[2], dd[3])) + dd[4:8] + bytes((0x55,)))

        bus.write = echo_write

        def pos_frames():  # 已发出的 pos_force 帧子集（CAN id = 0x300+SlaveID 唯一标识）
            return [f for f in frames if f[13:15] == me.prefix_pos[13:15]]

        # open：自动切 POSITION（RID10=4 + 4 增益共 5 个参数帧）→ 1 帧 pos_force（cfg vlim=3.0 / flim=1.0）
        d.send_action_end("open")
        assert d.mode_end is ControlMode.POSITION
        assert frames[-1] == me.prefix_pos + _pack_pos(_END_ACTION_POS["open"], 3.0, 1.0) + b"\x00"
        assert frames[-2][21:29] == bytes((0x07, 0x00, 0x55, 26)) + _struct.pack("<f", 0.002)  # 末帧增益=vel_ki
        n = len(frames)

        # 已在位模式：close/home 不再切模式，各仅 1 帧
        d.send_action_end("close")
        assert len(frames) == n + 1 and frames[-1] == me.prefix_pos + _pack_pos(0.0, 3.0, 1.0) + b"\x00"
        d.send_action_end("home")
        assert len(frames) == n + 2

        # position 按需透传；越限（q_min=-5.49）被基类管道裁硬限位 + warn
        d.send_action_end("position", position=-4.2)
        assert frames[-1] == me.prefix_pos + _pack_pos(-4.2, 3.0, 1.0) + b"\x00"
        cap.msgs.clear()
        d.send_action_end("position", position=-9.0)
        assert any("q 指令越限" in m for m in cap.msgs)
        assert frames[-1] == me.prefix_pos + _pack_pos(-5.49, 3.0, 1.0) + b"\x00"

        # 异常路径（_call 统一转 warn 不抛）：未知动作 / position 缺参 / 切模式失败
        cap.msgs.clear()
        d.send_action_end("foo")
        assert any("未知动作" in m for m in cap.msgs)
        cap.msgs.clear()
        d.send_action_end("position")
        assert any("需提供 position 参数" in m for m in cap.msgs)
        d._mode_end = None
        n_pos = len(pos_frames())
        bus.write = frames.append  # 静默写（无参数回显）→ 切模式验证重试耗尽失败
        cap.msgs.clear()
        d.send_action_end("home")
        assert any("切 POSITION 模式失败" in m for m in cap.msgs)
        assert len(pos_frames()) == n_pos  # 模式切换失败绝不发位置帧
    finally:
        d._refresh_stop.set()
