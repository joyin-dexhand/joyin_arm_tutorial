"""JoyArm 工厂与单类离线测试（不依赖硬件）。

覆盖架构约束的单类行为：①工厂软化语义（未知型号 → ``None`` + 失败信息；
JoyArm 直用为硬失败——坏注册名 / 缺 backend / 缺 config 构造即抛）；②六域
成员字典机制（各域 REGISTRY 默认空表，域未配置即无成员、门面调用显性报错；
配置的成员必须全部创建成功）；③配置读取/静态自检（``get_config``/
``check_config``）与限位加载（硬限位 config 四键、软限位直配仅加载）；④离线
语义（执行类抛错）与拷贝隔离（config / 轨迹桥深拷贝）；⑤前置校验族
（connected / enabled / mode）。

运行：``python test/test_joyarm_factory.py`` 或 pytest。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import (  # noqa: E402
    joyarm_factory, JoyArmFactory, JoyArm, Pose, Wrench,
    IKResult, IkineSolver, TrajFrame, TrajPlanner,
    Controller, ControlMode, ArmState, JointState, Backend,
    clamp_to_limits,
)
from joyarm_core.joyarm import load_config  # noqa: E402
from joyarm_core.joyarm.joyarm import _build_domain, _cubic_traj  # noqa: E402

_ARM = None


def _arm():
    global _ARM
    if _ARM is None:
        _ARM = joyarm_factory("joyarm_dm")
        assert _ARM is not None, "joyarm_dm 创建失败"
    return _ARM


def test_factory_creates_dm():
    arm = _arm()
    assert arm.n_arm == 6
    assert arm.n_end == 1
    assert arm.model == "joyarm_dm"
    assert JoyArmFactory().list_models() == ["joyarm_dm"]


def test_factory_unknown_model_returns_none():
    """软化语义（仅工厂入口）：未知型号返回 None + 输出失败信息（不抛异常）。"""
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


def test_hardfail_construction():
    """硬失败语义：坏注册名 / 缺 backend / 缺 config 构造即抛 ValueError。"""
    cfg = copy.deepcopy(load_config("joyarm_dm"))
    cfg["robotics"] = {"fkine": "bogus"}               # 未实现注册名
    try:
        JoyArm(model="joyarm_dm", config=cfg)
        raise AssertionError("未注册求解器应抛 ValueError")
    except ValueError as e:
        assert "bogus" in str(e)
    cfg2 = copy.deepcopy(load_config("joyarm_dm"))
    del cfg2["backend"]                                # backend 必配
    try:
        JoyArm(model="joyarm_dm", config=cfg2)
        raise AssertionError("缺 backend 段应抛 ValueError")
    except ValueError as e:
        assert "backend" in str(e)
    try:
        JoyArm("nonexistent_model")                    # config 缺失（自动加载失败）
        raise AssertionError("缺 config 应抛 ValueError")
    except ValueError:
        pass


def test_domain_dicts_default_empty():
    """过渡态：各域 REGISTRY 默认空表、config 未配置即无成员；门面调用显性报错。"""
    arm = _arm()
    for d in ("fkine", "ikine", "jacobian", "dynamics", "traj", "control"):
        assert arm._active_name[d] is None
        assert arm.list_solvers(d) == []
    try:
        arm.fkine(arm.rand_q_arm(rng=np.random.default_rng(0)), "ee")
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


def test_build_domain_hard_fail_and_multiload():
    """硬失败（架构约束）：任一成员规格非法/注册名不存在即抛；列表多载全部加载；未配置跳过。"""
    reg = {"a": _DummyA, "b": _DummyB}
    assert _build_domain("control", reg, None) == {}   # 域未配置：跳过
    members = _build_domain("control", reg, [{"name": "b", "x": 123}, "a"])
    assert sorted(members) == ["a", "b"] and members["b"].x == 123
    for bad in ({"name": "bogus"}, "bogus", {"x": 1}, 123):
        try:
            _build_domain("control", reg, bad)
            raise AssertionError(f"{bad!r} 应抛 ValueError（硬失败）")
        except ValueError:
            pass


def test_get_config_deepcopy():
    arm = _arm()
    cfg = arm.get_config()
    assert cfg["basic"]["name"] == "joyarm_dm"
    assert "arm_mdh_and_limits" in cfg["joyarm"]     # 教学数据经类内 config 读取
    cfg["basic"]["name"] = "hacked"
    fresh = arm.get_config()
    assert fresh["basic"]["name"] == "joyarm_dm"


def test_config_and_traj_deepcopy_isolation():
    """深拷贝隔离：外部 config 改动、发布后帧修改均不影响类内数据。"""
    cfg = load_config("joyarm_dm")
    arm = JoyArm(model="joyarm_dm", config=cfg)
    cfg["basic"]["name"] = "hacked"                  # 构造后外部改动
    assert arm.get_config()["basic"]["name"] == "joyarm_dm"
    fr = TrajFrame(time=10.0, q=np.zeros(6))
    arm.set_current_frame(fr)
    fr.time = 99.0                                   # 发布后修改原对象
    assert arm.get_current_frame().time == 10.0
    seq = [TrajFrame(time=1.0, q=np.zeros(6))]
    arm.set_target_traj(seq)
    seq[0].time = 55.0
    assert arm.get_target_traj()[0].time == 1.0


def test_arm_limits_from_config():
    """硬限位自 config backend.*.joints 四键解析；软限位直配加载（仅加载备用）。"""
    arm = _arm()
    cfg = load_config("joyarm_dm")
    aj = cfg["backend"]["arm"]["joints"]
    assert arm.n_arm == len(aj) == 6
    assert np.allclose(arm.arm_limits.q_min, [j["q_min"] for j in aj])
    assert np.allclose(arm.arm_limits.q_max, [j["q_max"] for j in aj])
    assert np.allclose(arm.arm_limits.dq_max, [j["dq_max"] for j in aj])
    assert np.allclose(arm.arm_limits.tau_max, [j["tau_max"] for j in aj])
    # 软限位直配（joyarm.arm_soft_limits 四键直值，不经 margin 换算）
    m = cfg["joyarm"]["arm_soft_limits"]
    assert np.allclose(arm.arm_limits_soft.q_min, np.asarray(m["q_min"], float))
    assert np.allclose(arm.arm_limits_soft.q_max, np.asarray(m["q_max"], float))
    assert np.allclose(arm.arm_limits_soft.dq_max, np.asarray(m["dq_max"], float))
    assert np.allclose(arm.arm_limits_soft.tau_max, np.asarray(m["tau_max"], float))
    # 软限位 ⊆ 硬限位；末端直配 + 电机空间行程
    ej = cfg["backend"]["end"]["joints"][0]
    assert arm.n_end == 1
    assert arm.end_limits is not None and np.allclose(arm.end_limits.q_min, ej["q_min"])
    assert np.allclose(arm.arm_limits_soft.q_min, np.maximum(
        arm.arm_limits_soft.q_min, arm.arm_limits.q_min))          # 软 ⊆ 硬（下界）
    assert np.all(arm.arm_limits_soft.q_max <= arm.arm_limits.q_max + 1e-9)
    assert np.allclose(arm.end_limits_soft.q_min, ej["q_min"])    # end_soft_limits = 硬限位
    # backend 只存硬限位（下发只裁硬限位；无软限位属性）
    assert np.allclose(arm._backend.arm_limits.q_max, arm.arm_limits.q_max)
    for gone in ("arm_limits_soft", "end_limits_soft"):
        assert not hasattr(arm._backend, gone)


def test_soft_limits_default_and_validation():
    """软限位缺省软=硬；越硬限位即构造失败（硬失败）；结构错误经 check_config 报告。"""
    cfg = copy.deepcopy(load_config("joyarm_dm"))
    del cfg["joyarm"]["arm_soft_limits"]                  # 缺省 → 软=硬
    del cfg["joyarm"]["end_soft_limits"]
    arm = JoyArm(model="joyarm_dm", config=cfg)
    assert np.allclose(arm.arm_limits_soft.q_min, arm.arm_limits.q_min)
    assert np.allclose(arm.end_limits_soft.q_max, arm.end_limits.q_max)
    cfg2 = copy.deepcopy(load_config("joyarm_dm"))
    cfg2["joyarm"]["arm_soft_limits"] = {"q_min": -9.9}   # 越硬限位下界
    try:
        JoyArm(model="joyarm_dm", config=cfg2)
        raise AssertionError("软限位越硬限位应抛 ValueError")
    except ValueError as e:
        assert "soft_limits" in str(e)
    cfg3 = copy.deepcopy(load_config("joyarm_dm"))
    cfg3["joyarm"]["arm_soft_limits"] = {"q_min": [0.1, 0.2]}   # 长度错（静态自检捕获）
    try:
        JoyArm.check_config("joyarm_dm", cfg3)
        raise AssertionError("软限位长度错应抛 ValueError")
    except ValueError as e:
        assert "arm_soft_limits" in str(e)


def test_feature_poses():
    """特征位形：zero/neutral 按自由度全零（不经 config）；home 自 config 并可裁剪。"""
    arm = _arm()
    assert np.array_equal(arm.arm_zero, np.zeros(6)) and arm.arm_zero.shape == (6,)
    assert np.array_equal(arm.arm_neutral, np.zeros(6))
    assert np.array_equal(arm.end_zero, np.zeros(1)) and arm.end_zero.shape == (1,)
    assert np.array_equal(arm.end_neutral, np.zeros(1))
    assert np.array_equal(arm.arm_home, np.zeros(6))     # config arm_home 全零
    assert np.array_equal(arm.end_home, np.zeros(1))     # config end_home 全零
    # home 越限自动裁剪并告警
    cfg = copy.deepcopy(load_config("joyarm_dm"))
    cfg["joyarm"]["arm_home"] = [9.0] * 6
    arm2 = JoyArm(model="joyarm_dm", config=cfg)
    assert np.allclose(arm2.arm_home, arm2.arm_limits.q_max)   # 裁到硬限位


def test_check_config_static():
    """check_config 静态自检：通过则静默；坏配置一次列出全部问题。"""
    JoyArm.check_config("joyarm_dm", load_config("joyarm_dm"))   # 正常 → 静默通过
    cfg = copy.deepcopy(load_config("joyarm_dm"))
    cfg["joyarm"]["arm_home"] = [0.0, 0.0]                       # 长度错
    cfg["backend"]["arm"]["joints"][0].pop("q_min")              # 限位键缺失
    try:
        JoyArm.check_config("joyarm_dm", cfg)
        raise AssertionError("坏配置应抛 ValueError")
    except ValueError as e:
        assert "arm_home" in str(e) and "q_min" in str(e)


def test_offline_execution_raises():
    """离线语义：限位采样等纯计算可用、执行类 RuntimeError。"""
    arm = _arm()
    q = arm.rand_q_arm(rng=np.random.default_rng(1))
    assert q.shape == (arm.n_arm,)
    for fn in (arm.get_arm_state, arm.enable_arm, arm.set_end_open):
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
    """TrajFrame 纯数据 + TrajPlanner 内核（规则/超时剔除 + q_home 回退 + 按
    绝对时间采样）+ JoyArm 轨迹桥访问器（周期调度与发布门控归运动管线，
    见 test_review_fixes.py）。"""
    import time as _time

    # TrajFrame：纯数据（公开名仅为七个字段，无任何方法）
    public = [n for n in dir(TrajFrame) if not n.startswith("_")]
    assert public == ["dq", "pose", "q", "tau", "time", "twist", "wrench"], public
    assert TrajFrame(time=1.0, q=np.zeros(6)) == TrajFrame(time=1.0, q=np.zeros(6))
    assert TrajFrame(time=1.0, q=np.zeros(6)) != TrajFrame(time=2.0, q=np.zeros(6))

    # _check_frame：合法（关节型 / 位姿型含 wrench）返回 None；非法返回问题描述
    assert TrajPlanner._check_frame(TrajFrame(time=10.0, q=np.zeros(6))) is None
    assert TrajPlanner._check_frame(TrajFrame(time=10.0, pose=Pose())) is None
    assert TrajPlanner._check_frame(
        TrajFrame(time=10.0, pose=Pose(), wrench=Wrench())) is None
    assert "time" in TrajPlanner._check_frame(TrajFrame(q=np.zeros(6)))
    assert "pose/q" in TrajPlanner._check_frame(TrajFrame(time=10.0))
    assert "pose/q" in TrajPlanner._check_frame(
        TrajFrame(time=10.0, pose=Pose(), q=np.zeros(6)))
    assert "dq" in TrajPlanner._check_frame(
        TrajFrame(time=10.0, q=np.zeros(6), dq=np.zeros(6)))

    # plan_once 管线（FakeArm 鸭子契约：桥目标 / 状态 / arm_home / 当前帧写口）
    class FakeArm:
        def __init__(self, targets=None, q=None, arm_home=None):
            self._target = targets
            self.state = ArmState(joint=JointState(
                q=np.zeros(6) if q is None else np.asarray(q, dtype=float)))
            self.arm_home = np.zeros(6) if arm_home is None else np.asarray(arm_home)
            self.frames = []

        def get_target_traj(self):
            return self._target

        def get_arm_state(self):
            return self.state

        def set_current_frame(self, frame):
            self.frames.append(frame)

    calls = []

    class DummyPlanner(TrajPlanner):
        def _plan(self, arm, targets):
            calls.append(list(targets))

        def sample_frame(self, t_abs):
            return TrajFrame(time=t_abs, q=np.full(6, 7.0))

    # ① 无效/超时帧剔除：_plan 只收到有效且提前量足够的目标
    now = _time.time()
    fa = FakeArm(targets=[
        TrajFrame(q=np.zeros(6)),                          # time 缺失 → 规则剔除
        TrajFrame(time=now + 0.5, q=np.zeros(6)),          # 早于 now+1.0 → 超时剔除
        TrajFrame(time=10.0, pose=Pose(), q=np.zeros(6)),  # pose/q 双非空 → 规则剔除
        TrajFrame(time=now + 2.0, q=np.full(6, 0.3)),      # 有效且未超时 → 保留
    ])
    dp = DummyPlanner(plan_hz=2.0, sample_hz=100.0)
    assert dp.plan_hz == 2.0 and dp.sample_hz == 100.0 and dp.dt_min_required == 1.0
    dp.plan_once(fa)
    assert len(calls) == 1 and len(calls[0]) == 1
    assert calls[0][0].time == now + 2.0 and np.allclose(calls[0][0].q, 0.3)

    # ② 空目标回退：q=arm_home，time = now + max|q_home−q0|（此处 max=0.2）
    q0 = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
    q_home = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    dp.plan_once(FakeArm(targets=None, q=q0, arm_home=q_home))
    assert len(calls) == 2 and len(calls[1]) == 1
    fb = calls[1][0]
    assert np.allclose(fb.q, q_home)
    assert abs(fb.time - (_time.time() + 0.2)) < 0.5      # 容忍执行期 now 漂移

    # ③ 内核直调：规划后按绝对时间采样产出帧（周期调度与发布门控归 JoyArm 运动管线）
    dp2 = DummyPlanner()
    dp2.plan_once(FakeArm(targets=None, q=q0, arm_home=q_home))
    fr = dp2.sample_frame(_time.time())
    assert np.allclose(fr.q, 7.0)

    # JoyArm 轨迹桥：默认 None 初始态；set/get 对应；连续 set 取最新（深拷贝隔离）
    arm = _arm()
    assert arm.get_target_traj() is None and arm.get_current_frame() is None
    arm.set_target_traj([TrajFrame(time=10.0, q=np.zeros(6))])
    assert len(arm.get_target_traj()) == 1
    arm.set_target_traj(TrajFrame(time=11.0, q=np.zeros(6)))   # 单帧归一
    assert len(arm.get_target_traj()) == 1 and arm.get_target_traj()[0].time == 11.0
    arm.set_current_frame(TrajFrame(time=20.0, q=np.ones(6)))
    assert arm.get_current_frame().time == 20.0
    arm.set_current_frame(TrajFrame(time=20.1, q=np.ones(6)))
    assert arm.get_current_frame().time == 20.1

    # Trajectory / TrajectorySpace 已删除，不可再导入
    import joyarm_core
    assert not hasattr(joyarm_core, "Trajectory")
    assert not hasattr(joyarm_core, "TrajectorySpace")


def test_controller_kernel():
    """Controller.compute 内核（纯计算、无线程）：当前帧+状态 → 指令原样透传
    （限位守卫归后端基类）；MODE 声明默认 MIT、子类可覆写。"""
    class DummyCtrl(Controller):
        def compute(self, arm, frame, state):
            return ControlMode.MIT, {"q": np.full(6, 5.0),   # 越限值原样下发
                                     "dq": np.full(6, 9.0),  # （裁剪归后端基类模板）
                                     "tau": np.zeros(6)}

    c = DummyCtrl(ctrl_hz=100.0)
    assert c.ctrl_hz == 100.0 and c.MODE == ControlMode.MIT   # 频率供管线起节拍
    mode, cmd = c.compute(None, TrajFrame(time=1.0, q=np.zeros(6)), ArmState())
    assert mode is ControlMode.MIT
    assert np.allclose(cmd["q"], 5.0) and np.allclose(cmd["dq"], 9.0)   # 未裁剪
    assert np.allclose(cmd["tau"], 0.0)

    class PositionCtrl(Controller):
        MODE = ControlMode.POSITION            # 管线激活/热切换时自动 set_mode_arm

        def compute(self, arm, frame, state):
            return self.MODE, {"q": frame.q}

    p = PositionCtrl()
    assert p.MODE == ControlMode.POSITION
    mode, cmd = p.compute(None, TrajFrame(time=1.0, q=np.ones(6)), ArmState())
    assert mode is ControlMode.POSITION and np.allclose(cmd["q"], 1.0)


def test_state_cache_and_keepalive():
    """get_arm_state/get_end_state 新鲜度感知两级读取 + 保活线程起停与触发。"""
    import time as _time

    arm = _arm()
    saved = (arm._backend, arm.connected)
    try:
        be = _fake_backend()
        age = {"arm": 0.0, "end": 0.0}               # 可控陈旧度（0=新鲜）
        sync = {"arm": 0, "end": 0}
        orig_arm, orig_end = be.read_state_arm, be.read_state_end

        def _sync_arm(joint=None):
            sync["arm"] += 1
            age["arm"] = 0.0                          # 同步刷新后变新鲜
            return orig_arm(joint)

        def _sync_end(joint=None):
            sync["end"] += 1
            age["end"] = 0.0
            return orig_end(joint)

        be.read_state_cache_arm = lambda joint=None: ArmState(
            joint=JointState(q=np.full(6, 1.5)))
        be.read_state_cache_end = lambda joint=None: {"q": [0.25]}
        be.state_age_arm = lambda joint=None: age["arm"]
        be.state_age_end = lambda joint=None: age["end"]
        be.read_state_arm = _sync_arm
        be.read_state_end = _sync_end
        arm._backend = be
        arm.connected = True
        # ① 数据新鲜 → 零总线帧缓存读（不触发同步刷新）
        st = arm.get_arm_state()
        assert sync["arm"] == 0 and np.allclose(st.joint.q, 1.5)
        assert arm.get_end_state()["q"] == [0.25] and sync["end"] == 0
        # ② 数据陈旧 → get_arm_state 内联同步刷新（如 check_hardware 刚连接场景）
        age["arm"] = 1.0
        st = arm.get_arm_state()
        assert sync["arm"] == 1                       # 同步刷新一次
        # ③ connect 启动保活线程；新鲜期不刷新、陈旧后 10Hz 巡检触发保活
        arm.connected = False
        arm.connect()                                 # fake connect + 启动保活
        assert arm._state_thread is not None and arm._state_thread.is_alive()
        sync["arm"] = sync["end"] = 0
        age["arm"] = age["end"] = 0.0
        _time.sleep(0.3)
        assert sync["arm"] == 0 and sync["end"] == 0  # 新鲜 → 保活不动作
        age["arm"] = age["end"] = 1.0                 # 置陈旧 → 保活刷新
        _time.sleep(0.3)
        assert sync["arm"] >= 1 and sync["end"] >= 1
        # ④ disconnect 停止线程
        arm.disconnect()
        assert arm._state_thread is None
        # ⑤ 后端无缓存读/陈旧度（基类默认 NotImplementedError）→ 回退同步路径
        be2 = _fake_backend()
        arm._backend = be2
        arm.connected = True
        st2 = arm.get_arm_state()
        assert st2.joint.q.shape == (6,)
    finally:
        arm._backend, arm.connected = saved


def test_repr_contains_state():
    arm = _arm()
    r = repr(arm)
    assert "JoyArm" in r and "model='joyarm_dm'" in r and "offline" in r


# ----------------------------------------------------------
# 运动便利与安全层（FakeBackend 注入离线测；_arm 单例须还原）
# ----------------------------------------------------------
def _fake_backend(q0=None, converge: float = 0.5, n: int = 6, enabled: bool = True) -> Backend:
    """离线哑后端：记录全部调用；q 状态机每次读状态向最后位置目标靠近 converge 比例。

    模式缓存（set/read_mode_*）与使能位语义同 backend_dm：read_mode_* 离线可查、
    各电机模式唯一才返回；state.enabled 由 ``enabled`` 参数控制。
    """

    def _init(self):
        Backend.__init__(self, {})
        self.calls = []                      # (方法名, 参数...)
        self._q = np.zeros(n) if q0 is None else np.array(q0, dtype=float)
        self._target = self._q.copy()
        self._mode_arm: dict = {}            # 电机名 → 模式（空=未设置）
        self._mode_end: dict = {}

    def _rec(name):
        def _m(self, *a, **k):
            self.calls.append((name,) + a)
        return _m

    def _set_mode_arm(self, mode, joint=None):
        self.calls.append(("set_mode_arm", mode, joint))
        for i in range(n):
            self._mode_arm[f"joint{i + 1}"] = mode

    def _read_mode_arm(self, joint=None):
        vals = list(self._mode_arm.values())
        return vals[0] if len(self._mode_arm) == n and len(set(vals)) == 1 else None

    def _set_mode_end(self, mode, joint=None):
        self.calls.append(("set_mode_end", mode, joint))
        self._mode_end["gripper"] = mode

    def _read_mode_end(self, joint=None):
        vals = list(self._mode_end.values())
        return vals[0] if len(self._mode_end) == 1 else None

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
        ok = np.full(n, enabled, dtype=bool)
        return ArmState(joint=JointState(q=self._q.copy(), dq=np.zeros(n),
                                         tau=np.zeros(n), enabled=ok,
                                         comm_ok=np.ones(n, dtype=bool),
                                         error=np.zeros(n, dtype=bool),
                                         angle_ok=np.ones(n, dtype=bool)))

    def _read_state_end(self, joint=None):
        self.calls.append(("read_state_end", joint))
        return {"q": [0.0], "comm_ok": [True], "error": [False], "enabled": [True]}

    def _pos_end(self, position, joint=None):
        self.calls.append(("send_position_end", np.asarray(position, dtype=float).copy(), joint))

    ns = {"__init__": _init, "connected": property(lambda self: True),
          "_send_position_arm": _pos, "_send_mit_arm": _mit,
          "_send_position_end": _pos_end,
          "read_state_arm": _read_state, "read_state_end": _read_state_end,
          "set_mode_arm": _set_mode_arm, "read_mode_arm": _read_mode_arm,
          "set_mode_end": _set_mode_end, "read_mode_end": _read_mode_end}
    for name in ("connect", "disconnect",
                 "enable_arm", "disable_arm", "set_zero_arm", "clear_fault_arm",
                 "read_param_arm", "write_param_arm",
                 "_send_velocity_arm",
                 "enable_end", "disable_end", "set_zero_end", "clear_fault_end",
                 "_send_tau_end", "_send_mit_end",
                 "send_action_end", "read_param_end", "write_param_end"):
        ns[name] = _rec(name)
    return type("FakeBackend", (Backend,), ns)()


class _FakeSession:
    """单例 arm 换装哑后端并置为已连接；用毕 restore 还原（单例跨测试共享）。"""

    def __init__(self, q0=None, converge: float = 0.5, enabled: bool = True):
        self.arm = _arm()
        self._saved = (self.arm._backend, self.arm.connected)
        self.be = _fake_backend(q0, converge, n=self.arm.n_arm, enabled=enabled)
        self.arm._backend = self.be
        self.arm.connected = True

    def restore(self):
        self.arm._backend, self.arm.connected = self._saved


def test_deleted_redundant_apis():
    arm = _arm()
    for name in ("state", "qlow", "qhigh", "set_controller", "clamp_q", "is_q_valid",
                 "set_config", "set_arm_soft_margins", "set_end_soft_margins"):
        assert not hasattr(arm, name), f"{name} 应已删除"
    q = arm.rand_q_arm(rng=np.random.default_rng(0))
    assert q.shape == (arm.n_arm,)                # rand_q_arm：按本体硬限位采样


def test_joint_names_and_index():
    arm = _arm()
    assert arm.joint_names_arm == [f"joint{i+1}" for i in range(arm.n_arm)]
    assert arm.joint_names_end == ["gripper"]
    assert arm.joint_index_arm("joint3") == 2
    assert arm.joint_index_end("gripper") == 0
    try:
        arm.joint_index_arm("bogus")
        raise AssertionError("未知名应抛 ValueError")
    except ValueError as e:
        assert "joint" in str(e)


def test_require_guards():
    """前置校验族：未切模式 / 模式不符 / 未使能均显性报错。"""
    s = _FakeSession(q0=np.zeros(6))
    try:
        arm = s.arm
        try:
            arm.set_arm_command(ControlMode.POSITION, q=np.zeros(6))
            raise AssertionError("未切模式应抛 RuntimeError")
        except RuntimeError as e:
            assert "set_mode_arm" in str(e)
        arm.set_mode_arm(ControlMode.POSITION)
        arm.set_arm_command(ControlMode.POSITION, q=np.zeros(6))   # 已切模式 → 通过
        try:
            arm.set_arm_command(ControlMode.MIT, q=np.zeros(6),
                                dq=np.zeros(6), tau=np.zeros(6))
            raise AssertionError("模式不符应抛 RuntimeError")
        except RuntimeError:
            pass
    finally:
        s.restore()
    s2 = _FakeSession(q0=np.zeros(6), enabled=False)               # 未使能
    try:
        try:
            s2.arm.move_j(np.zeros(6), t=0.01, wait_timeout=0.3)
            raise AssertionError("未使能应抛 RuntimeError")
        except RuntimeError as e:
            assert "使能" in str(e)
    finally:
        s2.restore()


def test_check_hardware_connects_and_disconnects():
    """硬件自检自包含：临时连接（失能状态）→ 全部自检 → 断开；已连接则沿用。"""
    s = _FakeSession()
    try:
        arm, be = s.arm, s.be
        arm.connected = False
        arm.check_hardware()                     # 通过则静默
        names = [c[0] for c in be.calls]
        assert "connect" in names and "disconnect" in names
        assert names.index("connect") < names.index("disconnect")
        assert not arm.connected
        be.calls.clear()
        arm.connect()                            # 已连接：沿用连接、检完不断开
        arm.check_hardware()
        names = [c[0] for c in be.calls]
        assert "connect" in names and "disconnect" not in names
        assert arm.connected
    finally:
        s.restore()


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
    s2 = _FakeSession()                                          # pose 分支：fkine 未配置
    try:
        s2.arm.is_in_position(pose=Pose())
        raise AssertionError("fkine 未配置应抛 RuntimeError")
    except RuntimeError:
        pass
    finally:
        s2.restore()


def test_lock_and_hold_position():
    s = _FakeSession(q0=np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0]))
    try:
        arm, be = s.arm, s.be
        arm.lock_position()                                      # 急停锁定
        assert ("set_mode_arm", ControlMode.POSITION, None) in be.calls
        pos = [c for c in be.calls if c[0] == "send_position_arm"][-1]
        assert np.allclose(pos[1], [0.1, -0.2, 0.3, 0, 0, 0])
        arm.hold_position()                                      # 原位保持（tau 前馈缺省 0）
        mit = [c for c in be.calls if c[0] == "send_mit_arm"][-1]
        assert np.allclose(mit[1], [0.1, -0.2, 0.3, 0, 0, 0])    # q=当前
        assert np.allclose(mit[2], 0) and np.allclose(mit[3], 0)
        # 注入伪 dynamics → tau 前馈 = gravity(q)
        class _Dyn:
            def gravity(self, arm, q):
                return np.full(arm.n_arm, 2.5)
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
        target = np.array([0.06, -0.25, -0.35, 0.03, 0.02, 0.01])  # 软限位内目标
        arm.move_j(target, t=0.12, rate=200, wait_timeout=2.0)
        frames = [c for c in be.calls if c[0] == "send_position_arm"]
        ts, qs, _ = _cubic_traj(np.zeros(6), target, 0.12, 200)
        assert len(frames) == len(ts)                            # 帧数 = 采样数
        assert np.allclose(np.array([f[1] for f in frames]), qs)  # 帧落在三次曲线上
        assert ("set_mode_arm", ControlMode.POSITION, None) in be.calls[:3]
        assert arm.is_in_position(q=target)                      # 到位收尾
    finally:
        s.restore()
    s3 = _FakeSession(q0=np.zeros(6))                            # 目标越软限位：入口裁剪
    try:
        arm3, be3 = s3.arm, s3.be
        tgt = np.full(6, 5.0)                                    # 全关节越硬上限
        exp = clamp_to_limits(tgt, arm3.arm_limits)             # 判定以裁剪后为准
        arm3.move_j(tgt, t=0.02, wait_timeout=2.0)
        frames3 = [c for c in be3.calls if c[0] == "send_position_arm"]
        assert np.allclose(frames3[-1][1], exp)                  # 末帧 = 裁剪后目标
        assert arm3.is_in_position(q=exp)
    finally:
        s3.restore()
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
    """安全起停（arm+end）：本体 MIT 阻抗流 + 末端位置模式；safe_zero = home → zero。"""
    s = _FakeSession(q0=np.full(6, 0.3))
    try:
        arm, be = s.arm, s.be
        qz = clamp_to_limits(arm.arm_zero, arm.arm_limits)  # 硬限位投影后 zero/home
        try:                                                      # home 前置校验
            arm.home_to_zero()
            raise AssertionError("不在 home 应抛 RuntimeError")
        except RuntimeError as e:
            assert "home" in str(e)
        arm.safe_zero()                                           # 组合：home → home 检查 → zero
        assert ("set_mode_arm", ControlMode.MIT, None) in be.calls   # MIT 阻抗模式
        mits = [c for c in be.calls if c[0] == "send_mit_arm"]
        assert mits and np.allclose(mits[-1][1], qz)             # 末帧 = 裁剪后 zero
        end_pos = [c for c in be.calls if c[0] == "send_position_end"]
        assert end_pos and np.allclose(end_pos[-1][1], arm.end_home)   # 末端随动（位置模式）
        assert arm.is_in_position(q=qz)
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
