"""核心库审阅修复的回归测试（离线，不依赖硬件）。

覆盖本轮修复的缺陷与新增机制：①``utils/loops``（周期循环节拍 / PeriodicThread
生命周期 / stop 超时防双循环 / 异常日志节流）；②数值防护（``damped_pinv``
σ=0 且 λ=0 不产生 NaN）；③协议层（接收残余缓冲上限截断）；④``TrajPlanner``
（NaN time 剔除 / stop 重置采样门控 / 回退帧构造失败跳过本周期不刷异常）；
⑤``JoyArm``（``get_arm_state`` 不污染后端缓存对象 / ``ikine_all`` 统一
RuntimeError / ``start_motion``/``stop_motion`` 管线启停与回滚）；⑥``BackendDM``
（connect 失败回滚 / disconnect 失能失败不阻断关闭）；⑦``@register`` 注册
装饰器。

运行：``python test/test_review_fixes.py`` 或 pytest。
"""
from __future__ import annotations

import copy
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import (  # noqa: E402
    joyarm_factory, JoyArm, Pose, IKResult, TrajFrame, TrajPlanner, Controller,
    JacobianSolver, IkineSolver, FkineSolver, ControlMode,
    ArmState, JointState, Backend,
)
from joyarm_core.utils.loops import run_periodic, PeriodicThread  # noqa: E402
from joyarm_core.robotics._registry import register, _snake  # noqa: E402
from joyarm_core.joyarm import load_config  # noqa: E402

_ARM = None


def _arm():
    global _ARM
    if _ARM is None:
        _ARM = joyarm_factory("joyarm_dm")
        assert _ARM is not None, "joyarm_dm 创建失败"
    return _ARM


# ============================================================
# ① utils/loops：周期循环与线程生命周期
# ============================================================
def test_periodic_thread_lifecycle():
    """节拍近似目标频率；stop 干净退出；可重启；运行中重复 start 拒绝。"""
    import time as _t
    n = {"tick": 0}

    def step():
        n["tick"] += 1

    w = PeriodicThread(step, hz=100.0, name="t-life")
    assert not w.is_alive()                     # 未启动视为不在运行
    w.start()
    assert w.is_alive()
    try:
        w.start()
        raise AssertionError("重复 start 应抛 RuntimeError")
    except RuntimeError:
        pass
    _t.sleep(0.3)
    w.stop()
    assert not w.is_alive()
    assert 15 <= n["tick"] <= 60                # 100Hz×0.3s，容忍调度抖动
    w.start()                                   # 停止后可重启
    _t.sleep(0.05)
    w.stop()


def test_periodic_thread_stuck_stop():
    """stop 超时（单步阻塞）：抛 RuntimeError、引用保留、拒绝重启；解锁后可停。"""
    release = threading.Event()

    def step():
        release.wait(timeout=10.0)              # 模拟阻塞在总线 IO

    w = PeriodicThread(step, hz=10.0, name="t-stuck")
    w.start()
    time.sleep(0.05)                            # 让线程进入 step
    try:
        w.stop()
        raise AssertionError("线程未退出 stop 应抛 RuntimeError")
    except RuntimeError as e:
        assert "未退出" in str(e)
    assert w.is_alive()                         # 引用保留 → 状态可查
    try:
        w.start()                               # 拒绝重启（防双循环）
        raise AssertionError("未退出的线程再次 start 应被拒绝")
    except RuntimeError:
        pass
    release.set()                               # 解除阻塞
    w.stop()                                    # 此时可正常停止
    assert not w.is_alive()


def test_run_periodic_exception_throttled():
    """单步持续抛异常：日志按时间节流（0.5s 一条），不会每个周期刷一条。"""
    class _Counting(logging.Handler):
        def __init__(self):
            super().__init__()
            self.n = 0

        def emit(self, record):
            self.n += 1

    handler = _Counting()
    log = logging.getLogger("joyarm_core.loops")
    log.addHandler(handler)
    try:
        def bad():
            raise RuntimeError("boom")

        stop = threading.Event()
        th = threading.Thread(target=run_periodic,
                              args=(stop, 500.0, bad, "t-throttle"), daemon=True)
        th.start()
        time.sleep(1.2)
        stop.set()
        th.join(timeout=2.0)
    finally:
        log.removeHandler(handler)
    # 1.2s 内至多 3 条（0s/0.5s/1.0s 附近各一条），远小于 500Hz×1.2s=600 次
    assert handler.n <= 3, handler.n


# ============================================================
# ② 数值防护
# ============================================================
class _DummyJac(JacobianSolver):
    """固定 J 的哑实现（离线测试 damped_pinv 防护）。"""

    def __init__(self, J):
        self.J = np.asarray(J, dtype=float)

    def jac(self, arm, q, frame, ref="base"):
        return self.J


def test_damped_pinv_singular_no_nan():
    """σ=0 且 λ=0：该方向系数取 0，不产生 NaN；正常情形与 pinv 一致。"""
    Js = np.diag([1.0, 2.0, 0.0, 0.5, 0.3, 0.1])            # 含两个零奇异值
    s = _DummyJac(Js)
    out = s.damped_pinv(None, None, None, damping=0.0)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, np.linalg.pinv(Js))             # 截断后等价 pinv
    out_d = s.damped_pinv(None, None, None)                 # 默认 λ=1e-3 亦有限
    assert np.all(np.isfinite(out_d))


# ============================================================
# ③ 协议层：接收残余缓冲上限
# ============================================================
def test_extract_frames_residual_cap():
    """持续乱码：残余被截断到上限且保尾部；跨读半帧仍可在下一轮成帧。"""
    from joyarm_core.backend.backend_dm import _extract_frames, _RX_RESIDUAL_MAX

    garbage = bytes([0x00]) * (_RX_RESIDUAL_MAX + 6000)     # 10KB 无帧乱码
    frames, rest = _extract_frames(garbage)
    assert frames == [] and len(rest) == _RX_RESIDUAL_MAX   # 截断保尾

    head = bytes([0xAA]) + bytes(range(1, 8))               # 帧头+7字节（跨读半帧）
    frames, rest = _extract_frames(garbage + head)
    assert frames == [] and len(rest) == _RX_RESIDUAL_MAX
    completion = bytes(range(8, 15)) + bytes([0x55])        # 剩余 9 字节（尾 0x55）
    frames2, _ = _extract_frames(bytes(rest) + completion)  # 残余拼接后续数据
    assert len(frames2) == 1                                 # 半帧跨读不丢


# ============================================================
# ④ TrajPlanner 防护
# ============================================================
class _DummyPlanner(TrajPlanner):
    """记录 _plan 调用的哑规划器。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.planned = []

    def _plan(self, arm, targets):
        self.planned.append(list(targets))

    def sample_frame(self, t_abs):
        return TrajFrame(time=t_abs, q=np.zeros(6))


class _FakeTrajArm:
    """TrajPlanner 测试用哑臂（get_arm_state 可控抛错）。"""

    def __init__(self, targets=None, q=None, arm_home=None, state_raises=False):
        self._target = targets
        self.state = ArmState(joint=JointState(
            q=np.zeros(6) if q is None else np.asarray(q, dtype=float)))
        self.arm_home = np.zeros(6) if arm_home is None else np.asarray(arm_home)
        self.state_raises = state_raises

    def get_target_traj(self):
        return self._target

    def get_arm_state(self):
        if self.state_raises:
            raise RuntimeError("未连接真机")
        return self.state

    def set_current_frame(self, frame):
        pass


def test_check_frame_nan_time():
    """NaN/inf 的 time 与缺失同样被剔除（NaN 比较恒 False，旧实现会穿透）。"""
    assert "非有限" in TrajPlanner._check_frame(TrajFrame(time=float("nan"), q=np.zeros(6)))
    assert "非有限" in TrajPlanner._check_frame(TrajFrame(time=float("inf"), q=np.zeros(6)))


def test_home_fallback_offline_tolerant():
    """断连（get_arm_state 抛错）时 plan_once 跳过本周期：不抛、不调用 _plan。"""
    fa = _FakeTrajArm(targets=None, state_raises=True)
    dp = _DummyPlanner()
    dp.plan_once(fa)                                    # 旧实现：每周期异常+堆栈日志
    assert dp.planned == []                             # 未规划，沿用旧系数


def test_traj_stop_resets_planned():
    """stop() 重置采样发布门控：停止后手动 sample_once 不再发布旧系数帧。"""
    fa = _FakeTrajArm(q=np.zeros(6))
    dp = _DummyPlanner()
    dp.plan_once(fa)
    assert dp._planned is True
    dp.stop()
    assert dp._planned is False                         # 门控复位


# ============================================================
# ⑤ JoyArm 防护与运动管线
# ============================================================
class _SharedCacheBackend:
    """read_state_cache_arm 恒返回同一共享对象（模拟坏后端契约）。"""

    connected = True
    state = None

    def state_age_arm(self, joint=None):
        return 0.0                                      # 恒新鲜 → 走缓存读

    def read_state_cache_arm(self, joint=None):
        return self.state

    def read_state_end(self, joint=None):
        return {"q": [0.0]}


class _DummyFkine(FkineSolver):
    """固定位姿的哑正运动学。"""

    def frame_pose(self, arm, q, frame):
        return Pose(position=np.array([1.0, 2.0, 3.0]))


class _NoSolveAllIkine(IkineSolver):
    """不实现 solve_all 的哑逆运动学（默认 NotImplementedError）。"""

    def solve(self, arm, target, frame, q0, **kw):
        return IKResult(q=np.zeros(6), success=True, info="")


class _DomainSession:
    """单例 arm 换哑后端 + 注入指定域成员；用毕还原（单例跨测试共享）。"""

    def __init__(self, members: dict):
        self.arm = _arm()
        self._saved = (self.arm._backend, self.arm.connected,
                       {d: (dict(self._dict_for(d)), self.arm._active_name.get(d))
                        for d in ("fkine", "ikine", "jacobian", "dynamics",
                                  "traj", "control")})
        self.arm._backend = _SharedCacheBackend()
        self.arm.connected = True
        for domain, (name, obj) in members.items():
            self._dict_for(domain)[name] = obj
            self.arm._active_name[domain] = name

    def _dict_for(self, domain: str) -> dict:
        return {"fkine": self.arm._fkine_solvers,
                "ikine": self.arm._ikine_solvers,
                "jacobian": self.arm._jacobian_solvers,
                "dynamics": self.arm._dynamics_solvers,
                "traj": self.arm._traj_planners,
                "control": self.arm._controllers}[domain]

    def restore(self):
        self.arm._backend, self.arm.connected = self._saved[:2]
        for domain, (members, name) in self._saved[2].items():
            d = self._dict_for(domain)
            d.clear()
            d.update(members)
            if name is None:
                self.arm._active_name.pop(domain, None)
            else:
                self.arm._active_name[domain] = name


def test_get_arm_state_does_not_mutate_cache():
    """get_arm_state 填 tcp.pose 前复制外壳：后端共享缓存对象不被污染。"""
    sess = _DomainSession({"fkine": ("fkine_dummy", _DummyFkine())})
    try:
        be = sess.arm._backend
        be.state = ArmState(joint=JointState(q=np.full(6, 0.5)))
        orig_pose = be.state.tcp.pose
        st = sess.arm.get_arm_state()
        assert np.allclose(st.tcp.pose.position, [1.0, 2.0, 3.0])   # 已填充
        assert st.tcp.pose is not orig_pose                         # 外壳已复制
        assert be.state.tcp.pose is orig_pose                       # 原对象未被改写
    finally:
        sess.restore()


def test_ikine_all_not_implemented_becomes_runtime_error():
    """ikine 成员未实现 solve_all：门面统一转 RuntimeError（与其他域一致）。"""
    sess = _DomainSession({"ikine": ("ik_dummy", _NoSolveAllIkine())})
    try:
        try:
            sess.arm.ikine_all(Pose(), "link_end")
            raise AssertionError("未实现 solve_all 应转 RuntimeError")
        except RuntimeError as e:
            assert "不支持" in str(e)
    finally:
        sess.restore()


class _Handle:
    """记录 start/stop 调用的哑成员。"""

    def __init__(self, fail_start=False):
        self.started = self.stopped = False
        self.fail_start = fail_start

    def start(self, arm):
        if self.fail_start:
            raise RuntimeError("电机无应答")
        self.started = True

    def stop(self):
        self.stopped = True


def test_start_stop_motion_pipeline():
    """start_motion/stop_motion：离线未连接先抛；已连接无成员抛；成员齐备则启停。"""
    arm = _arm()
    # ① 离线：未连接
    try:
        arm.start_motion()
        raise AssertionError("未连接应抛 RuntimeError")
    except RuntimeError as e:
        assert "connect" in str(e)
    assert not arm._motion_running

    # ② 已连接但域未配置：成员未加载
    sess = _DomainSession({})
    try:
        try:
            arm.start_motion()
            raise AssertionError("域未配置应抛 RuntimeError")
        except RuntimeError as e:
            assert "成员未加载" in str(e)
        assert not arm._motion_running

        # ③ 注入哑成员：启停生效、幂等
        planner = _Handle()
        ctrl = _Handle()
        arm._traj_planners["pipe_p"] = planner
        arm._controllers["pipe_c"] = ctrl
        arm._active_name["traj"] = "pipe_p"
        arm._active_name["control"] = "pipe_c"
        try:
            arm.start_motion()
            assert planner.started and ctrl.started and arm._motion_running
            arm.start_motion()                        # 幂等
            arm.stop_motion()
            assert planner.stopped and ctrl.stopped and not arm._motion_running
            arm.stop_motion()                         # 幂等
        finally:
            arm._traj_planners.pop("pipe_p", None)
            arm._controllers.pop("pipe_c", None)
            arm._active_name.pop("traj", None)
            arm._active_name.pop("control", None)
    finally:
        sess.restore()


def test_start_motion_controller_fail_rolls_back_planner():
    """控制器启动失败：规划器被回滚停止，标志不置位。"""
    arm = _arm()
    sess = _DomainSession({})
    try:
        planner = _Handle()
        bad_ctrl = _Handle(fail_start=True)
        arm._traj_planners["pipe_p"] = planner
        arm._controllers["bad_c"] = bad_ctrl
        arm._active_name["traj"] = "pipe_p"
        arm._active_name["control"] = "bad_c"
        try:
            try:
                arm.start_motion()
                raise AssertionError("控制器启动失败应向上抛")
            except RuntimeError as e:
                assert "电机无应答" in str(e)
            assert planner.stopped                     # 规划器已回滚
            assert not arm._motion_running
        finally:
            arm._traj_planners.pop("pipe_p", None)
            arm._controllers.pop("bad_c", None)
            arm._active_name.pop("traj", None)
            arm._active_name.pop("control", None)
    finally:
        sess.restore()


def test_stop_motion_stops_captured_members_after_set_solver():
    """运行期 set_solver 切换激活成员：stop_motion 仍停启动时捕获的原成员。"""
    arm = _arm()
    sess = _DomainSession({})
    try:
        p1, c1 = _Handle(), _Handle()
        p2, c2 = _Handle(), _Handle()
        arm._traj_planners["p1"], arm._traj_planners["p2"] = p1, p2
        arm._controllers["c1"], arm._controllers["c2"] = c1, c2
        arm._active_name["traj"], arm._active_name["control"] = "p1", "c1"
        try:
            arm.start_motion()
            # 运行期切换激活成员（模拟用户换算法）
            arm._active_name["traj"], arm._active_name["control"] = "p2", "c2"
            arm.stop_motion()
            assert p1.stopped and c1.stopped          # 停的是启动时捕获的原成员
            assert not p2.stopped and not c2.stopped  # 新成员未被误停
            assert arm._motion_members is None
        finally:
            for d, names in (("traj", ("p1", "p2")), ("control", ("c1", "c2"))):
                dd = arm._traj_planners if d == "traj" else arm._controllers
                for n in names:
                    dd.pop(n, None)
                arm._active_name.pop(d, None)
    finally:
        sess.restore()


def test_plan_once_dedupes_drop_warning():
    """同一批无效/超时目标：剔除告警只发一次（防死目标源 5Hz 刷屏）。"""
    class _Counting(logging.Handler):
        def __init__(self):
            super().__init__()
            self.n = 0

        def emit(self, record):
            self.n += 1

    handler = _Counting()
    log = logging.getLogger("joyarm_core.traj_planner")
    log.addHandler(handler)
    try:
        fa = _FakeTrajArm(targets=[TrajFrame(q=np.zeros(6))])   # time 缺失 → 剔除
        dp = _DummyPlanner()
        for _ in range(5):                       # 同批目标连续 5 个周期
            dp.plan_once(fa)
        assert handler.n == 1                    # 只告警一次
        fa2 = _FakeTrajArm(targets=None, q=np.zeros(6))          # 恢复正常 → 复位
        dp.plan_once(fa2)
        assert handler.n == 1
        dp.plan_once(fa)                         # 同批问题再现 → 重新告警
        assert handler.n == 2
    finally:
        log.removeHandler(handler)


# ============================================================
# ⑥ BackendDM：connect 回滚 / disconnect 保护
# ============================================================
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
        if self.channel in type(self).fail_open_channels:
            raise OSError(f"cannot open {self.channel}")
        type(self).opened.append(self.channel)

    def close(self):
        type(self).closed.append(self.channel)

    def add_motor(self, motor):
        motor.bus = self

    def disable(self, motor):
        if type(self).fail_disable:
            raise OSError("device unplugged")
        type(self).disabled.append((self.channel, motor.name))


def _make_dm_backend():
    """从 joyarm_dm 真实 config 构建 BackendDM（离线：不碰串口）。"""
    from joyarm_core.backend.backend_dm import BackendDM
    cfg = copy.deepcopy(load_config("joyarm_dm")["backend"])
    cfg.pop("name", None)
    return BackendDM(cfg)


def _with_fake_bus(fn):
    """装饰器：测试期间把 backend_dm.DmCanBus 换成替身，用毕还原。"""
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
    # joyarm_dm 的 arm/end 共享同一 channel——改 end 为独立通道以触发第二段失败
    be._end_cfg["channel"] = "/dev/ttyFAKE_END"
    _FakeDmBus.opened.clear(), _FakeDmBus.closed.clear()
    _FakeDmBus.fail_open_channels = {"/dev/ttyFAKE_END"}
    try:
        be.connect()
        raise AssertionError("end 总线打开失败应抛")
    except OSError:
        pass
    assert len(_FakeDmBus.opened) == 1                  # arm 总线曾打开
    assert _FakeDmBus.closed == _FakeDmBus.opened       # 已被回滚关闭
    assert not be.connected                             # 无残留总线


@_with_fake_bus
def test_dm_disconnect_disable_failure_still_closes():
    """拔线场景：disable 抛错不断开收尾，串口仍被关闭、状态被清。"""
    be = _make_dm_backend()
    _FakeDmBus.opened.clear(), _FakeDmBus.closed.clear(), _FakeDmBus.disabled.clear()
    _FakeDmBus.fail_open_channels = set()
    _FakeDmBus.fail_disable = True
    be.connect()
    assert be.connected
    be.disconnect()                                    # 不抛：失能失败仅告警
    assert _FakeDmBus.disabled == []                   # 全部 disable 失败
    assert len(_FakeDmBus.closed) == len(_FakeDmBus.opened)   # 总线仍全被关闭
    assert not be.connected
    assert all(m.bus is None for m in be._arm_motors + be._end_motors)


# ============================================================
# ⑦ @register 注册装饰器
# ============================================================
def test_register_decorator():
    """命名转换 + 显式注册表注册 + 包外裸用给清晰错误。"""
    assert _snake("PinFkineSolver") == "pin_fkine_solver"     # 前缀式：算法前缀+域基类名
    assert _snake("QuinticTrajPlanner") == "quintic_traj_planner"
    assert _snake("Controller") == "controller"
    assert _snake("DLSController") == "dls_controller"

    reg: dict = {}

    @register(registry=reg)
    class MyFkineSolver(FkineSolver):
        def frame_pose(self, arm, q, frame):
            return None

    assert reg["my_fkine_solver"] is MyFkineSolver

    try:
        @register                                       # 顶层模块裸用 → 清晰报错
        class BadFk(FkineSolver):
            def frame_pose(self, arm, q, frame):
                return None
        raise AssertionError("包外裸用应抛 RuntimeError")
    except RuntimeError as e:
        assert "顶层模块" in str(e)


if __name__ == "__main__":
    failed = 0
    for name in sorted(globals()):
        fn = globals()[name]
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {name}: {e}")
    sys.exit(1 if failed else 0)
