"""JoyArm 工厂与单类离线测试（不依赖硬件）。

覆盖架构约束的单类行为：①工厂软失败语义（未知型号 → ``None`` + 失败信息）；
②六域成员字典机制（各域 REGISTRY 默认空表——教学各章实现注册后接入）；③配置
读取/运行期设置/自检（``get_config``/``set_config``/``check_config``）；④离线
语义（执行类抛错、未加载域门面显性报错）。

运行：``python test/test_joyarm_factory.py`` 或 pytest。
"""
from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import (  # noqa: E402
    joyarm_factory, JoyArmFactory, JoyArm, Pose, Wrench,
    IKResult, IkineSolver, TrajFrame, TrajPlanner,
    Controller, ControlMode, ArmState, JointState, Backend, cubic_traj,
)
from joyarm_core.joyarm import load_config  # noqa: E402
from joyarm_core.joyarm.joyarm import _build_domain  # noqa: E402

_ARM = None


def _arm():
    global _ARM
    if _ARM is None:
        _ARM = joyarm_factory("joyarm_dm")
        assert _ARM is not None, "joyarm_dm 创建失败"
    return _ARM


def test_factory_creates_dm():
    arm = _arm()
    assert arm.n == 6
    assert arm.model == "joyarm_dm"
    assert JoyArmFactory().list_models() == ["joyarm_dm"]


def test_factory_unknown_model_returns_none():
    """软失败语义（架构约束）：未知型号返回 None + 输出失败信息（不抛异常）。"""
    for bad in ("joyarm_nx", "", "backend_dm"):
        assert joyarm_factory(bad) is None, f"{bad!r} 应返回 None"


def test_load_config_strict_and_lenient():
    assert load_config("joyarm_dm") is not None
    assert load_config("nonexistent") is None          # 容错：None
    try:
        load_config("nonexistent", strict=True)
        raise AssertionError("strict 模式应抛 ValueError")
    except ValueError:
        pass


def test_domain_dicts_default_empty():
    """教学过渡态：各域 REGISTRY 默认空表、config 未配置即无成员；门面调用显性报错。"""
    arm = _arm()
    for d in ("fkine", "ikine", "jacobian", "dynamics", "traj", "control"):
        assert arm._active_name[d] is None
        assert arm.list_solvers(d) == []
    try:
        arm.fkine(arm.rand_q(rng=np.random.default_rng(0)), "ee")
        raise AssertionError("未加载 fkine 应抛 RuntimeError")
    except RuntimeError:
        pass


class _DummyA:
    pass


class _DummyB:
    def __init__(self, x=0):
        self.x = x


def test_set_solver_switch_and_alias():
    """机制测试：注入哑成员验证切换/报错（各域注册表默认空，章节实现后接入）。"""
    arm = _arm()
    arm._controllers["dummy_a"] = _DummyA()
    arm._controllers["dummy_b"] = _DummyB()
    arm._active_name["control"] = "dummy_a"
    inst = arm.set_solver("control", "dummy_b")
    assert isinstance(inst, _DummyB)
    assert arm._active_name["control"] == "dummy_b"
    try:
        arm.set_solver("control", "nonexistent")
        raise AssertionError("未加载注册名应抛 ValueError")
    except ValueError:
        pass
    try:
        arm.set_solver("bogus_domain", "x")
        raise AssertionError("未知域应抛 ValueError")
    except ValueError:
        pass
    # 清理注入成员
    arm._controllers.clear()
    arm._active_name["control"] = None


def test_build_domain_soft_fail_and_multiload():
    """软失败（架构约束）：无效注册名跳过 + 警告；列表多载全部加载；未配置静默跳过。"""
    reg = {"a": _DummyA, "b": _DummyB}
    members = _build_domain("control", reg, [{"name": "bogus"}, {"name": "b", "x": 123}])
    assert sorted(members) == ["b"] and members["b"].x == 123
    m2 = _build_domain("control", reg, ["a", "b"])
    assert sorted(m2) == ["a", "b"]
    assert _build_domain("control", reg, None) == {}   # 域未配置：静默跳过


def test_get_config_deepcopy():
    arm = _arm()
    cfg = arm.get_config()
    assert cfg["basic"]["name"] == "joyarm_dm"
    assert "arm_mdh_and_limits" in cfg["joyarm"]     # 教学数据经类内 config 读取
    cfg["basic"]["name"] = "hacked"
    fresh = arm.get_config()
    assert fresh["basic"]["name"] == "joyarm_dm"


def test_set_config_whitelist():
    arm = _arm()
    # 本体软限位即时生效（四键绝对余量；缺省键不内缩）
    arm.set_config("joyarm.joint_soft_margins", {"q_upper": 0.2, "q_lower": 0.2})
    soft = arm.joint_limits_soft
    assert np.allclose(soft.q_min, arm.joint_limits.q_min + 0.2)
    assert np.allclose(soft.q_max, arm.joint_limits.q_max - 0.2)
    assert np.allclose(soft.dq_max, arm.joint_limits.dq_max)
    # 后端守卫同步换用新软限位
    assert arm._backend is not None and \
        np.allclose(arm._backend.arm_limits_soft.q_min, soft.q_min)
    # 末端软限位同理
    arm.set_config("joyarm.end_soft_margins", {"q_upper": 0.1})
    assert np.allclose(arm.end_limits_soft.q_max, arm.end_limits.q_max - 0.1)
    assert np.allclose(arm._backend.end_limits_soft.q_max, arm.end_limits.q_max - 0.1)
    j = load_config("joyarm_dm")["joyarm"]                  # 还原
    arm.set_config("joyarm.joint_soft_margins", j["joint_soft_margins"])
    arm.set_config("joyarm.end_soft_margins", j["end_soft_margins"])
    # 白名单外拒绝（含 basic.utils 两代旧键路径）
    for path in ("backend.name", "basic.utils.joint_limits_soft_margin",
                 "basic.utils.joint_soft_margins"):
        try:
            arm.set_config(path, "x")
            raise AssertionError(f"白名单外路径 {path!r} 应抛 ValueError")
        except ValueError:
            pass


def test_joint_soft_margins_from_config():
    """config joyarm 段四键 margin → 本体/末端软限位（URDF / end.joints 硬限位内缩）。"""
    arm = _arm()
    cfg = load_config("joyarm_dm")
    m = cfg["joyarm"]["joint_soft_margins"]
    assert np.allclose(arm.joint_limits_soft.q_min,
                       arm.joint_limits.q_min + np.asarray(m["q_lower"], float))
    assert np.allclose(arm.joint_limits_soft.q_max,
                       arm.joint_limits.q_max - np.asarray(m["q_upper"], float))
    assert np.allclose(arm.joint_limits_soft.dq_max,
                       arm.joint_limits.dq_max - np.asarray(m["dq"], float))
    assert np.allclose(arm.joint_limits_soft.tau_max,
                       arm.joint_limits.tau_max - np.asarray(m["tau"], float))
    # 末端：end_limits 自 backend.end.joints 逐电机解析；margin 全 0 → 软=硬
    ej = cfg["backend"]["end"]["joints"][0]
    assert arm.end_limits is not None and np.allclose(arm.end_limits.q_min, ej["q_min"])
    assert np.allclose(arm.end_limits.q_max, ej["q_max"])
    assert np.array_equal(arm.end_limits_soft.q_min, arm.end_limits.q_min)
    # 后端基类守卫副本一致（构造参数传递 → 后端 __init__ 自建）
    assert arm._backend is not None and \
        np.allclose(arm._backend.arm_limits_soft.q_max, arm.joint_limits_soft.q_max) and \
        np.allclose(arm._backend.end_limits_soft.tau_max, arm.end_limits_soft.tau_max)


def test_old_margin_key_deprecated():
    """basic.utils 两代旧键均告警且不生效；margin 一律以 joyarm 段为准。"""
    cfg = copy.deepcopy(load_config("joyarm_dm"))
    cfg.setdefault("basic", {})["utils"] = {
        "joint_limits_soft_margin": 0.05, "joint_soft_margins": {"q_upper": 0.5}}

    class _Cap(logging.Handler):
        def __init__(self):
            super().__init__()
            self.msgs = []

        def emit(self, record):
            self.msgs.append(record.getMessage())

    cap = _Cap()
    log = logging.getLogger("joyarm_core.joyarm")
    log.addHandler(cap)
    try:
        arm = JoyArm(model="joyarm_dm", config=cfg)
    finally:
        log.removeHandler(cap)
    assert len([m for m in cap.msgs if "迁移" in m]) == 2
    m = load_config("joyarm_dm")["joyarm"]["joint_soft_margins"]
    assert np.allclose(arm.joint_limits_soft.q_max,
                       arm.joint_limits.q_max - np.asarray(m["q_upper"], float))


def test_check_config_clean_passes():
    arm = _arm()
    arm.check_config()          # 正常 → 静默通过（异常则 ValueError）


def test_check_config_detects_problems():
    cfg = load_config("joyarm_dm")
    # q_home 长度错 + backend 关节数不符（bogus 注册名走软失败架构，由门面调用报错）
    cfg = copy.deepcopy(cfg)
    cfg["joyarm"]["q_home"] = [0.0, 0.0]
    cfg["robotics"]["ikine"] = "bogus"
    cfg["backend"]["arm"]["joints"] = cfg["backend"]["arm"]["joints"][:4]
    arm = JoyArm(model="joyarm_dm", config=cfg)
    try:
        arm.check_config()
        raise AssertionError("坏配置应抛 ValueError")
    except ValueError as e:
        assert "q_home" in str(e) and "backend.arm.joints" in str(e)
    # 未加载域的门面调用显性报错
    try:
        arm.ikine(Pose(), "ee", np.zeros(6))
        raise AssertionError("未加载 ikine 应抛 RuntimeError")
    except RuntimeError:
        pass


def test_offline_execution_raises():
    """离线语义：限位采样等纯计算可用、执行类 RuntimeError。"""
    arm = _arm()
    q = arm.rand_q(rng=np.random.default_rng(1))
    assert q.shape == (arm.n,)
    for fn in (arm.get_arm_state, arm.enable_arm, arm.end_open):
        try:
            fn()
            raise AssertionError(f"{fn.__name__} 离线应抛 RuntimeError")
        except RuntimeError:
            pass


def test_ikine_contracts():
    """ikine 选解契约：±2π 归位 / 限位剔除+最近解 / dummy 解析求解器端到端。"""
    q_lo = np.array([-np.pi] * 3 + [-2.0] * 3)
    q_hi = np.array([np.pi] * 3 + [2.0] * 3)
    # _shift_2pi：宽区间逐关节平移入界（等价角）
    s = IkineSolver._shift_2pi(np.array([7.5, -8.0, 0.3, 5.0, 0.0, 0.0]), q_lo, q_hi)
    assert np.isclose(s[0], 7.5 - 2 * np.pi) and np.isclose(s[1], -8.0 + 2 * np.pi)
    assert np.isclose(s[2], 0.3) and np.isclose(s[3], 5.0 - 2 * np.pi)
    # 窄区间（< 2π）：平移后更近则平移；无等价角可入界则留在原值
    assert np.isclose(IkineSolver._shift_2pi(np.array([5.9]), [0.0], [1.0])[0], 5.9 - 2 * np.pi)
    assert np.isclose(IkineSolver._shift_2pi(np.array([3.0]), [0.0], [0.1])[0], 3.0)
    # _select_nearest：剔除越限行 + 取与 q0 偏差平方和最小者；全越限 → 失败
    sols = np.array([[0.1, 0, 0, 0, 0, 0],
                     [0.2, 0, 0, 0, 0, 0],
                     [9.9, 0, 0, 0, 0, 0]])
    r = IkineSolver._select_nearest(sols, np.full(6, 0.18), q_lo, q_hi)
    assert r.success and np.allclose(r.q, sols[1])
    r2 = IkineSolver._select_nearest(sols[2:], np.zeros(6), q_lo, q_hi)
    assert (not r2.success) and r2.q.size == 0

    # dummy 解析求解器：solve_all 全解（含 ±2π 归位）；solve 限位内选 q0 最近
    class DummyIk(IkineSolver):
        def solve(self, arm, target, frame, q0, *, tol=1e-4, iters=200, **kw):
            return self._select_nearest(
                self.solve_all(arm, target, frame).q, q0, arm.qlow, arm.qhigh)

        def solve_all(self, arm, target, frame, **kw):
            raw = np.array([[0.1, 0, 0, 0, 0, 0],
                            [7.5, 0, 0, 0, 0, 0]])          # 第二行需 -2π 归位
            shifted = np.array([IkineSolver._shift_2pi(row, arm.qlow, arm.qhigh)
                                for row in raw])
            return IKResult(q=shifted, success=True, err=0.0, n_iter=0)

    arm = SimpleNamespace(qlow=q_lo, qhigh=q_hi)             # 鸭子类型 arm
    d = DummyIk()
    ra = d.solve_all(arm, Pose(), "ee")
    assert ra.q.shape == (2, 6) and q_lo[0] <= ra.q[1, 0] <= q_hi[0]
    rs = d.solve(arm, Pose(), "ee", np.zeros(6))
    assert rs.success and np.allclose(rs.q, np.array([0.1, 0, 0, 0, 0, 0]))


def test_traj_frame_and_planner():
    """TrajFrame 纯数据 + TrajPlanner 目标判别模板 + JoyArm 轨迹桥访问器。"""
    # TrajFrame：纯数据（公开名仅为七个字段，无任何方法）
    public = [n for n in dir(TrajFrame) if not n.startswith("_")]
    assert public == ["dq", "pose", "q", "tau", "time", "twist", "wrench"], public
    assert TrajFrame(time=1.0, q=np.zeros(6)) == TrajFrame(time=1.0, q=np.zeros(6))
    assert TrajFrame(time=1.0, q=np.zeros(6)) != TrajFrame(time=2.0, q=np.zeros(6))

    # _check_targets：合法分支（关节型 / 位姿型含 wrench）
    TrajPlanner._check_targets([TrajFrame(time=10.0, q=np.zeros(6))])
    TrajPlanner._check_targets(
        [TrajFrame(time=10.0, pose=Pose()),
         TrajFrame(time=11.0, pose=Pose(), wrench=Wrench())])
    # 非法分支：time 缺失 / pose、q 双空 / 双非空 / dq 非空
    bad = [TrajFrame(q=np.zeros(6)),
           TrajFrame(time=10.0),
           TrajFrame(time=10.0, pose=Pose(), q=np.zeros(6)),
           TrajFrame(time=10.0, q=np.zeros(6), dq=np.zeros(6))]
    try:
        TrajPlanner._check_targets(bad)
        raise AssertionError("无效目标序列应抛 ValueError")
    except ValueError as e:
        assert "time" in str(e) and "dq" in str(e)

    # plan 模板：单帧归一 + 判别 + 委托内核
    calls = []

    class DummyPlanner(TrajPlanner):
        def _plan(self, arm, targets, **kw):
            calls.append(list(targets))

        def sample_frame(self, t_abs):
            return TrajFrame(time=t_abs, q=np.zeros(6))

    dp = DummyPlanner(plan_hz=2.0, sample_hz=100.0)
    assert dp.plan_hz == 2.0 and dp.sample_hz == 100.0
    dp.plan(None, TrajFrame(time=10.0, q=np.zeros(6)))   # 单帧归一为列表
    assert len(calls) == 1 and len(calls[0]) == 1
    try:
        dp.plan(None, [TrajFrame(time=10.0)])            # 无效 → 不触内核
        raise AssertionError("无效目标应抛 ValueError")
    except ValueError:
        assert len(calls) == 1

    # JoyArm 轨迹桥：默认 None 初始态；set/get 对应；连续 set 取最新
    arm = _arm()
    assert arm.get_target_traj() is None and arm.get_current_frame() is None
    arm.set_target_traj([TrajFrame(time=10.0, q=np.zeros(6))])
    assert len(arm.get_target_traj()) == 1
    arm.set_target_traj(TrajFrame(time=11.0, q=np.zeros(6)))   # 单帧归一
    assert len(arm.get_target_traj()) == 1 and arm.get_target_traj()[0].time == 11.0
    fr1 = TrajFrame(time=20.0, q=np.ones(6))
    arm.set_current_frame(fr1)
    assert arm.get_current_frame() is fr1
    fr2 = TrajFrame(time=20.1, q=np.ones(6))
    arm.set_current_frame(fr2)
    assert arm.get_current_frame() is fr2

    # Trajectory / TrajectorySpace 已删除，不可再导入
    import joyarm_core
    assert not hasattr(joyarm_core, "Trajectory")
    assert not hasattr(joyarm_core, "TrajectorySpace")


def test_controller_step():
    """Controller.step 模板：状态缺省现读、指令原样下发（限位守卫统一在后端基类）。"""
    class FakeArm:                       # 鸭子契约放宽：无需 joint_limits_soft
        def __init__(self):
            self.state = ArmState()
            self.reads = 0
            self.sent = []

        def get_arm_state(self):
            self.reads += 1
            return self.state

        def set_arm_command(self, mode, **cmd):
            self.sent.append((mode, cmd))

    class DummyCtrl(Controller):
        def _compute(self, arm, frame, state, **kw):
            return ControlMode.MIT, {"q": np.full(6, 5.0),   # 越限值原样下发
                                     "dq": np.full(6, 9.0),  # （裁剪归后端基类模板）
                                     "tau": np.zeros(6)}

    arm = FakeArm()
    c = DummyCtrl(ctrl_hz=100.0)
    assert c.ctrl_hz == 100.0
    c.step(arm, TrajFrame(time=1.0, q=np.zeros(6)))          # state 缺省 → 现读一次
    assert arm.reads == 1 and len(arm.sent) == 1
    mode, cmd = arm.sent[0]
    assert mode is ControlMode.MIT
    assert np.allclose(cmd["q"], 5.0) and np.allclose(cmd["dq"], 9.0)   # 未裁剪
    assert np.allclose(cmd["tau"], 0.0)
    c.step(arm, TrajFrame(time=1.0, q=np.zeros(6)), state=arm.state)   # 传 state → 不再读
    assert arm.reads == 1 and len(arm.sent) == 2


def test_repr_contains_state():
    arm = _arm()
    r = repr(arm)
    assert "JoyArm" in r and "model='joyarm_dm'" in r and "offline" in r


# ----------------------------------------------------------
# 运动便利与安全层（FakeBackend 注入离线测；_arm 单例须还原）
# ----------------------------------------------------------
def _fake_backend(q0=None, converge: float = 0.5, n: int = 6) -> Backend:
    """离线哑后端：记录全部调用；q 状态机每次读状态向最后位置目标靠近 converge 比例。"""

    def _init(self):
        Backend.__init__(self, {})
        self.calls = []                      # (方法名, 参数...)
        self._q = np.zeros(n) if q0 is None else np.array(q0, dtype=float)
        self._target = self._q.copy()

    def _rec(name):
        def _m(self, *a, **k):
            self.calls.append((name,) + a)
        return _m

    def _pos(self, q, joint=None):
        self._target = np.asarray(q, dtype=float).copy()
        self.calls.append(("send_position_arm", self._target.copy(), joint))

    def _mit(self, q, dq, tau_ff, kp=None, kd=None, joint=None):
        self._target = np.asarray(q, dtype=float).copy()
        self.calls.append(("send_mit_arm", self._target.copy(),
                           np.asarray(dq, dtype=float), np.asarray(tau_ff, dtype=float),
                           kp, kd, joint))

    def _read_state(self, joint=None):
        self._q = self._q + converge * (self._target - self._q)   # 模真机跟随
        return ArmState(joint=JointState(q=self._q.copy(), dq=np.zeros(n),
                                         tau=np.zeros(n)))

    def _read_state_end(self, joint=None):
        self.calls.append(("read_state_end", joint))
        return {"q": [0.0], "comm_ok": [True], "error": [False]}

    ns = {"__init__": _init, "connected": property(lambda self: True),
          "_send_position_arm": _pos, "_send_mit_arm": _mit,
          "read_state_arm": _read_state, "read_state_end": _read_state_end}
    for name in ("connect", "disconnect",
                 "enable_arm", "disable_arm", "set_zero_arm", "clear_fault_arm",
                 "set_mode_arm", "read_mode_arm", "read_param_arm", "write_param_arm",
                 "_send_velocity_arm",
                 "enable_end", "disable_end", "set_zero_end", "clear_fault_end",
                 "set_mode_end", "read_mode_end",
                 "_send_position_end", "_send_force_end", "_send_mit_end",
                 "send_action_end", "read_param_end", "write_param_end"):
        ns[name] = _rec(name)
    return type("FakeBackend", (Backend,), ns)()


class _FakeSession:
    """单例 arm 换装哑后端并置为已连接；用毕 restore 还原（单例跨测试共享）。"""

    def __init__(self, q0=None, converge: float = 0.5):
        self.arm = _arm()
        self._saved = (self.arm._backend, self.arm.connected)
        self.be = _fake_backend(q0, converge, n=self.arm.n)
        self.arm._backend = self.be
        self.arm.connected = True

    def restore(self):
        self.arm._backend, self.arm.connected = self._saved


def test_deleted_redundant_apis():
    arm = _arm()
    for name in ("state", "qlow", "qhigh", "set_controller", "clamp_q", "is_q_valid"):
        assert not hasattr(arm, name), f"{name} 应已删除"
    q = arm.rand_q(rng=np.random.default_rng(0))
    assert q.shape == (arm.n,)                      # rand_q 保留，按软限位采样


def test_joint_names_and_index():
    arm = _arm()
    assert arm.joint_names == [f"joint{i+1}" for i in range(arm.n)]
    assert arm.joint_index("joint3") == 2
    try:
        arm.joint_index("bogus")
        raise AssertionError("未知名应抛 ValueError")
    except ValueError as e:
        assert "joint" in str(e)


def test_is_in_position_joint_branch():
    s = _FakeSession(q0=np.zeros(6))
    try:
        arm = s.arm
        assert arm.is_in_position(q=np.zeros(6))
        assert not arm.is_in_position(q=np.full(6, 0.5))
        assert arm.is_in_position(q=np.full(6, 0.04))            # tol_q=0.05
        for kw in ({}, {"q": np.zeros(6), "pose": Pose()}, {"q": np.zeros(3)}):
            try:
                arm.is_in_position(**kw)
                raise AssertionError(f"参数 {kw} 应抛 ValueError")
            except ValueError:
                pass
    finally:
        s.restore()
    try:
        _arm().is_in_position(q=np.zeros(6))
        raise AssertionError("离线应抛 RuntimeError")
    except RuntimeError:
        pass
    s2 = _FakeSession()                                          # pose 分支：fkine 未注册
    try:
        s2.arm.is_in_position(pose=Pose())
        raise AssertionError("fkine 未注册应抛 RuntimeError")
    except RuntimeError:
        pass
    finally:
        s2.restore()


def test_lock_and_hold_position():
    s = _FakeSession(q0=np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0]))
    try:
        arm, be = s.arm, s.be
        arm.lock_position()                                      # 急停锁定
        assert be.calls[-2] == ("set_mode_arm", ControlMode.POSITION, None)
        assert be.calls[-1][0] == "send_position_arm" and \
            np.allclose(be.calls[-1][1], [0.1, -0.2, 0.3, 0, 0, 0])
        arm.hold_position()                                      # 原位保持（tau 前馈缺省 0）
        mit = [c for c in be.calls if c[0] == "send_mit_arm"][-1]
        assert np.allclose(mit[1], [0.1, -0.2, 0.3, 0, 0, 0])    # q=当前
        assert np.allclose(mit[2], 0) and np.allclose(mit[3], 0)
        # 注入伪 dynamics → tau 前馈 = gravity(q)
        class _Dyn:
            def gravity(self, arm, q):
                return np.full(arm.n, 2.5)
        arm._dynamics_solvers["fake"] = _Dyn()
        arm._active_name["dynamics"] = "fake"
        try:
            arm.hold_position()
            mit2 = [c for c in be.calls if c[0] == "send_mit_arm"][-1]
            assert np.allclose(mit2[3], 2.5)
        finally:
            arm._dynamics_solvers.clear()
            arm._active_name["dynamics"] = None
    finally:
        s.restore()


def test_move_j():
    s = _FakeSession(q0=np.zeros(6))
    try:
        arm, be = s.arm, s.be
        target = np.array([0.06, 0.05, 0.04, 0.03, 0.02, 0.01])
        arm.move_j(target, t=0.12, rate=200, wait_timeout=2.0)
        frames = [c for c in be.calls if c[0] == "send_position_arm"]
        ts, qs, _ = cubic_traj(np.zeros(6), target, 0.12, 200)
        assert len(frames) == len(ts)                            # 帧数 = 采样数
        assert np.allclose(np.array([f[1] for f in frames]), qs)  # 帧落在三次曲线上
        assert ("set_mode_arm", ControlMode.POSITION, None) in be.calls[:2]
        assert arm.is_in_position(q=target)                      # 到位收尾
    finally:
        s.restore()
    s2 = _FakeSession(q0=np.zeros(6), converge=0.0)              # 永不收敛
    try:
        try:
            s2.arm.move_j(np.zeros(3))
            raise AssertionError("维度错误应抛 ValueError")
        except ValueError:
            pass
        try:
            s2.arm.move_j(np.full(6, 0.2), t=0.02, wait_timeout=0.15)
            raise AssertionError("到位超时应抛 RuntimeError")
        except RuntimeError as e:
            assert "到位超时" in str(e)
    finally:
        s2.restore()


def test_safe_home_zero_and_composition():
    s = _FakeSession(q0=np.full(6, 0.3))
    try:
        arm, be = s.arm, s.be
        try:                                                      # home 前置校验
            arm.home_to_zero()
            raise AssertionError("不在 home 应抛 RuntimeError")
        except RuntimeError as e:
            assert "home" in str(e)
        arm.safe_zero()                                           # 组合：lock → home → zero
        frames = [c for c in be.calls if c[0] == "send_position_arm"]
        assert np.allclose(frames[0][1], 0.3)                    # 首帧 = 急停锁定当前 q
        assert np.allclose(frames[-1][1], np.zeros(6))           # 末帧 = zero
        assert arm.is_in_position(q=np.zeros(6))
        be.calls.clear()                                          # safe_home 单独
        be._q = np.full(6, 0.2)
        be._target = be._q.copy()
        arm.safe_home(t=0.1)
        assert np.allclose(be.calls[-1][1], np.zeros(6))         # 末帧 = home
        assert arm.is_in_position(q=np.zeros(6))
    finally:
        s.restore()


def test_move_l_stub_and_clear_fault_facade():
    s = _FakeSession()
    try:
        arm, be = s.arm, s.be
        try:
            arm.move_l(Pose(), 1.0)
            raise AssertionError("move_l 占位应抛 NotImplementedError")
        except NotImplementedError:
            pass
        arm.clear_fault_arm()
        arm.clear_fault_end(0)
        assert ("clear_fault_arm", None) in be.calls
        assert ("clear_fault_end", 0) in be.calls
    finally:
        s.restore()


def test_context_manager():
    s = _FakeSession()
    try:
        arm, be = s.arm, s.be
        arm.connected = False
        with arm as a:                                            # __enter__ 自动 connect
            assert a is arm and arm.connected
            arm.get_arm_state()
        names = [c[0] for c in be.calls]
        assert "connect" in names and "disable_arm" in names and "disconnect" in names
        assert names.index("disable_arm") < names.index("disconnect")
        assert not arm.connected
    finally:
        s.restore()


# ----------------------------------------------------------
# 直跑
# ----------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"{len(fns)} 项全部通过")
