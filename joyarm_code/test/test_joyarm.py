"""JoyArm 状态自愈、模式归属与真机验收判定的离线回归。

使用真实配置/模型和验收脚本的 DummyBackend，绝不连接硬件。
运行：cd joyarm_code && .venv/bin/python -m pytest test/test_joyarm.py -q
"""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from joyarm_core.joyarm import joyarm as jm
from joyarm_core.joyarm import joyarm_verify as verify
from joyarm_core.utils.types import ControlMode, JointState, MotionState, SafetyEvent


@pytest.fixture
def rig():
    verify.install_injection()
    arm, backend = verify.make_arm(lambda cfg: cfg['backend'].update(write_settle=0.001))
    arm.connect_arm()
    backend.close()  # 关闭后台刷新：故障与使能标志由本用例确定性控制。
    backend.set_q('arm', arm.home_arm)
    arm.get_state_arm()
    try:
        yield arm, backend
    finally:
        arm.set_event(SafetyEvent.DISABLE)
        arm.close()
        verify.restore_injection()


@pytest.mark.parametrize('state,event,mode', [
    (MotionState.HOME, SafetyEvent.HOMING, ControlMode.POSITION),
    (MotionState.HOLD, SafetyEvent.HOLDING, ControlMode.POSITION),
    (MotionState.DAMPING, SafetyEvent.DAMPING, ControlMode.MIT),
    (MotionState.MOTION, SafetyEvent.NORMAL, ControlMode.POSITION),
    (MotionState.ZERO, SafetyEvent.ZEROING, ControlMode.POSITION),
])
def test_state_repairs_mode_and_enable_only_when_needed(rig, state, event, mode):
    arm, bk = rig
    arm._current_motion_state = state
    arm.set_event(event)
    arm._hold_q = arm.home_arm.copy()
    arm._zero_moving = True
    bk.set_mode_arm(ControlMode.VELOCITY)
    bk.disable_arm()
    begin = len(bk.calls)
    assert arm._maintain_motion_state(state, event)
    assert bk.get_mode_arm() is mode
    assert bk.is_enabled_arm is True
    assert len(bk.calls_since(begin, 'set_joint_mode_arm')) == arm.n_arm
    assert len(bk.calls_since(begin, 'enable_joint_arm')) == arm.n_arm
    begin = len(bk.calls)
    assert arm._maintain_motion_state(state, event)
    assert not bk.calls_since(begin, 'set_joint_mode_arm')
    assert not bk.calls_since(begin, 'enable_joint_arm')


def test_zero_arrival_stays_disabled_and_hold_keeps_original_position(rig):
    arm, bk = rig
    arm._current_motion_state = MotionState.ZERO
    arm._zero_moving = True
    arm.set_event(SafetyEvent.ZEROING)
    bk.set_q('arm', arm.zero_arm)
    arm.get_state_arm()
    assert arm._maintain_motion_state(MotionState.ZERO, SafetyEvent.ZEROING)
    assert not arm._zero_moving and bk.is_enabled_arm is False
    begin = len(bk.calls)
    for _ in range(3):
        assert arm._maintain_motion_state(MotionState.ZERO, SafetyEvent.ZEROING)
    assert not bk.calls_since(begin, 'enable_joint_arm')
    assert not bk.calls_since(begin, 'set_joint_mode_arm')
    arm._current_motion_state = MotionState.HOLD
    arm.set_event(SafetyEvent.HOLDING)
    assert arm._enter_motion_state(MotionState.HOLD, SafetyEvent.HOLDING)
    hold = arm._hold_q.copy()
    bk.set_q('arm', arm.home_arm)
    bk.disable_arm()
    begin = len(bk.calls)
    assert arm._maintain_motion_state(MotionState.HOLD, SafetyEvent.HOLDING)
    sent = bk.calls_since(begin, 'send_joint_position_arm')
    assert all(c[2] == hold[c[1]] for c in sent)


@pytest.mark.parametrize('failed', ['set_joint_mode_arm', 'enable_joint_arm'])
def test_failed_self_repair_blocks_commands_without_changing_event(rig, failed):
    arm, bk = rig
    arm._current_motion_state = MotionState.DAMPING
    arm.set_event(SafetyEvent.DAMPING)
    bk.set_mode_arm(ControlMode.POSITION)
    bk.disable_arm()
    bk.fail_at[failed] = None
    begin = len(bk.calls)
    for _ in range(55):
        arm._fsm_tick()
        assert arm.get_event() is SafetyEvent.DAMPING
        assert arm.get_current_motion_state() is MotionState.DAMPING
        assert arm.get_target_motion_state() is MotionState.DAMPING
        assert not bk.calls_since(begin, 'send_joint_mit_arm')
    arm.set_event(SafetyEvent.DISABLE)  # 由外部/monitor 发布事件后才迁移。
    arm._fsm_tick()
    assert arm.get_current_motion_state() is MotionState.DISABLE
    assert bk.is_enabled_arm is False


def test_temporary_self_repair_failure_recovers_on_next_tick(rig):
    arm, bk = rig
    arm._current_motion_state = MotionState.DAMPING
    arm.set_event(SafetyEvent.DAMPING)
    bk.set_mode_arm(ControlMode.POSITION)
    bk.fail_at['set_joint_mode_arm'] = None
    arm._fsm_tick()
    bk.fail_at.clear()
    arm._fsm_tick()
    assert bk.get_mode_arm() is ControlMode.MIT
    assert bk.calls_since(0, 'send_joint_mit_arm')


def test_failed_enter_does_not_commit_state_or_clear_hold(rig):
    arm, bk = rig
    arm._current_motion_state = MotionState.HOLD
    arm._hold_q = arm.home_arm.copy()
    arm.set_event(SafetyEvent.HOMING)
    bk.set_mode_arm(ControlMode.MIT)
    bk.fail_at['set_joint_mode_arm'] = None
    arm._fsm_tick()
    assert arm.get_current_motion_state() is MotionState.HOLD
    np.testing.assert_array_equal(arm._hold_q, arm.home_arm)


@pytest.mark.parametrize('throws', [False, True])
def test_first_plan_failure_does_not_commit_motion_or_change_event(rig, monkeypatch, throws):
    arm, bk = rig
    arm._current_motion_state = MotionState.HOME
    arm._home_arrived = True
    arm.set_event(SafetyEvent.NORMAL)

    def fail(*args):
        if throws:
            raise RuntimeError('scripted first-plan failure')
        return False

    monkeypatch.setattr(arm._activated_traj_solver, 'plan', fail)
    begin = len(bk.calls)
    for _ in range(3):
        if throws:
            with pytest.raises(RuntimeError, match='first-plan failure'):
                arm._fsm_tick()
        else:
            arm._fsm_tick()
        assert arm.get_current_motion_state() is MotionState.HOME
        assert arm._ctrl_periodic is None and arm._plan_periodic is None
    assert not bk.calls_since(begin, 'enable_joint_arm')  # 规划应在硬件操作之前。
    assert arm.get_event() is SafetyEvent.NORMAL
    assert arm.get_target_motion_state() is MotionState.MOTION


def test_ctrl_selection_preserves_damping_and_waits_for_motion_check(rig):
    arm, bk = rig
    arm._current_motion_state = MotionState.DAMPING
    arm.set_event(SafetyEvent.DAMPING)
    bk.set_mode_arm(ControlMode.MIT)
    begin = len(bk.calls)
    assert arm.set_ctrl_solver(verify.VERIFY_CTRL)
    assert not bk.calls_since(begin, 'set_joint_mode_arm')
    assert arm._maintain_motion_state(MotionState.DAMPING, SafetyEvent.DAMPING)
    assert bk.get_mode_arm() is ControlMode.MIT
    calls = []
    selected = arm._activated_ctrl_solver
    selected.step = lambda *args: calls.append('step')
    arm._current_motion_state = MotionState.MOTION
    arm.set_event(SafetyEvent.NORMAL)
    arm._tick_ctrl()
    assert not calls
    assert arm._maintain_motion_state(MotionState.MOTION, SafetyEvent.NORMAL)
    assert bk.get_mode_arm() is ControlMode.VELOCITY
    arm._activated_traj_solver.plan(arm, None)
    arm._tick_ctrl()
    assert calls == ['step']


def test_damping_maintain_sends_gravity_feedforward(rig):
    arm, bk = rig
    arm._current_motion_state = MotionState.DAMPING
    arm.set_event(SafetyEvent.DAMPING)
    begin = len(bk.calls)
    assert arm._maintain_motion_state(MotionState.DAMPING, SafetyEvent.DAMPING)
    got = bk.calls_since(begin, 'send_joint_mit_arm')
    assert len(got) == arm.n_arm
    kd_cfg = np.asarray(arm.get_config()['joyarm']['MotionFSM']['damping_kd'], dtype=float).reshape(-1)
    kg_cfg = np.asarray(arm.get_config()['joyarm']['MotionFSM']['damping_k_gravity'], dtype=float).reshape(-1)
    tau_exp = kg_cfg * arm.get_gravity()                    # 被动快照 q（=home）下的补偿期望
    for c in got:
        assert c[5] == 0.0                                  # kp=0（无刚度项）
        assert c[6] == pytest.approx(kd_cfg[c[1]])          # kd=cfg 逐关节
        assert c[2] == pytest.approx(tau_exp[c[1]])         # tau=k_gravity·G(q) 重力补偿前馈


def test_damping_maintain_degrades_to_pure_damping_without_dyn_solver(rig, monkeypatch):
    arm, bk = rig
    arm._current_motion_state = MotionState.DAMPING
    arm.set_event(SafetyEvent.DAMPING)
    monkeypatch.setattr(arm, '_activated_dyn_solver', None)
    begin = len(bk.calls)
    assert arm._maintain_motion_state(MotionState.DAMPING, SafetyEvent.DAMPING)
    got = bk.calls_since(begin, 'send_joint_mit_arm')
    assert got and all(c[2] == 0.0 for c in got)            # dyn 域缺失：tau 全零退化纯阻尼


@pytest.mark.parametrize('bad', ['missing', 'over', 'negative'])
def test_damping_k_gravity_cfg_bounds_rejected(bad):
    def mutate(cfg):
        seg = cfg['joyarm']['MotionFSM']
        if bad == 'missing':
            del seg['damping_k_gravity']
        elif bad == 'over':
            seg['damping_k_gravity'] = [1.5] * 6
        else:
            seg['damping_k_gravity'] = [-0.1] * 6

    verify.install_injection()
    try:
        with pytest.raises(ValueError, match='damping_k_gravity'):
            verify.make_arm(mutate)
    finally:
        verify.restore_injection()


def test_partial_motion_start_is_cleaned_up_and_next_tick_recovers(rig, monkeypatch):
    arm, _ = rig
    arm._current_motion_state = MotionState.HOME
    arm._home_arrived = True
    arm.set_event(SafetyEvent.NORMAL)
    original = jm._PeriodicThread.start

    def start(worker):
        if worker.name == 'joyarm-ctrl':
            raise RuntimeError('scripted ctrl start failure')
        return original(worker)

    monkeypatch.setattr(jm._PeriodicThread, 'start', start)
    with pytest.raises(RuntimeError, match='ctrl start failure'):
        arm._fsm_tick()
    assert arm.get_current_motion_state() is MotionState.HOME
    assert arm._ctrl_periodic is None and arm._plan_periodic is None
    monkeypatch.setattr(jm._PeriodicThread, 'start', original)
    arm._fsm_tick()
    assert arm.get_current_motion_state() is MotionState.MOTION
    assert arm._ctrl_periodic.is_running() and arm._plan_periodic.is_running()


def test_normal_after_failed_hold_transition_restarts_stopped_motion_pipeline(rig):
    arm, bk = rig
    arm._current_motion_state = MotionState.HOME
    arm._home_arrived = True
    arm.set_event(SafetyEvent.NORMAL)
    arm._fsm_tick()
    assert arm.get_current_motion_state() is MotionState.MOTION
    bk.fail_at['read_joint_state_arm'] = None
    arm.set_event(SafetyEvent.HOLDING)
    arm._fsm_tick()
    assert arm.get_current_motion_state() is MotionState.MOTION
    assert arm._ctrl_periodic is None and arm._plan_periodic is None
    bk.fail_at.clear()
    arm.set_event(SafetyEvent.NORMAL)
    arm._fsm_tick()
    assert arm._ctrl_ready()
    assert arm._ctrl_periodic.is_running() and arm._plan_periodic.is_running()


def test_ctrl_selection_waits_for_inflight_step_without_switching_hardware(rig):
    arm, bk = rig
    entered, release, switched = threading.Event(), threading.Event(), threading.Event()

    class BlockingController:
        MODE = ControlMode.POSITION

        def step(self, owner, frame, state):
            entered.set()
            assert release.wait(2.0)
            owner.send_position_arm(frame.q)

    controller = BlockingController()
    arm._activated_ctrl_solver = controller
    arm._motion_ready = controller
    arm._current_motion_state = MotionState.MOTION
    arm.set_event(SafetyEvent.NORMAL)
    bk.set_mode_arm(ControlMode.POSITION)
    arm._activated_traj_solver.plan(arm, None)
    worker = threading.Thread(target=arm._tick_ctrl)
    arm._ctrl_thread = worker
    setter = threading.Thread(target=lambda: (arm.set_ctrl_solver(verify.VERIFY_CTRL), switched.set()))
    begin = len(bk.calls)
    worker.start()
    try:
        assert entered.wait(1.0)
        setter.start()
        assert not switched.wait(0.05)
    finally:
        release.set()
        worker.join(2.0)
        if setter.ident is not None:
            setter.join(2.0)
    assert not worker.is_alive() and not setter.is_alive()
    assert switched.is_set() and arm._motion_ready is None
    assert bk.get_mode_arm() is ControlMode.POSITION
    assert not bk.calls_since(begin, 'set_joint_mode_arm')


def test_protection_event_during_mode_read_cancels_old_self_repair(rig, monkeypatch):
    arm, bk = rig
    arm.set_event(SafetyEvent.HOMING)
    bk.disable_arm()

    def read_mode():
        arm.set_event(SafetyEvent.DISABLE)
        return ControlMode.MIT

    monkeypatch.setattr(bk, 'get_mode_arm', read_mode)
    begin = len(bk.calls)
    assert not arm._maintain_motion_state(MotionState.HOME, SafetyEvent.HOMING)
    assert not bk.calls_since(begin, 'enable_joint_arm')
    assert not bk.calls_since(begin, 'set_joint_mode_arm')
    assert not bk.calls_since(begin, 'send_joint_position_arm')


def test_stop_during_self_repair_cannot_reenable_after_shutdown(rig, monkeypatch):
    arm, bk = rig
    entered, release = threading.Event(), threading.Event()
    original = bk.get_mode_arm

    def blocked_read():
        entered.set()
        assert release.wait(2.0)
        return original()

    monkeypatch.setattr(bk, 'get_mode_arm', blocked_read)
    arm._current_motion_state = MotionState.DAMPING
    arm.set_event(SafetyEvent.DAMPING)
    bk.disable_arm()
    arm._move_wait_timeout = 0.01
    arm._fsm_periodic = jm._PeriodicThread(arm._fsm_worker, 100.0, 'test-fsm')
    begin = len(bk.calls)
    arm._fsm_periodic.start()
    stopper = threading.Thread(target=arm.stop_motion_fsm)
    try:
        assert entered.wait(1.0)
        stopper.start()
        assert arm._fsm_shutdown.wait(1.0)
    finally:
        release.set()
        if stopper.ident is not None:
            stopper.join(2.0)
    assert not stopper.is_alive()
    assert not arm._fsm_running()
    assert bk.is_enabled_arm is False
    assert not bk.calls_since(begin, 'enable_joint_arm')
    assert arm.get_current_motion_state() is MotionState.DISABLE


def test_periodic_stop_waking_wait_does_not_execute_another_step():
    stop = threading.Event()
    invoked = []

    class StopOnWait:
        def is_set(self):
            return stop.is_set()

        def wait(self, timeout):
            stop.set()
            return True

    jm._run_periodic(StopOnWait(), 1.0, lambda: invoked.append(True))
    assert not invoked
    worker = jm._PeriodicThread(lambda: None, 10.0)
    worker.prepare()
    worker.stop()  # 部分启动失败时，未起跑的线程句柄也能回收。
    assert worker.thread is None


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FeedbackArm:
    n_arm = 6

    def __init__(self, clock, follows=False):
        self.clock, self.follows = clock, follows
        self.q = np.zeros(self.n_arm)
        self.target = None
        self.joint_limits_soft_arm = SimpleNamespace(q_min=np.full(6, -1.0), q_max=np.ones(6))

    @property
    def joint_state_arm(self):
        return JointState(q=self.q.copy(), t=self.clock.time())

    def get_state_arm(self):
        return self.joint_state_arm

    def get_config(self):
        return {'backend': {'state_refresh_hz': 20.0}}

    def start_motion_fsm(self):
        pass

    def set_target_joint(self, t, q):
        self.target = np.array(q)
        if self.follows:
            self.q = self.target.copy()


@pytest.mark.parametrize('bad', ['stale', 'future', 'shape', 'nan', 'before_send'])
def test_real_feedback_rejects_invalid_nonempty_read_and_cache(monkeypatch, bad):
    clock = FakeClock()
    monkeypatch.setattr(verify, 'time', clock)
    arm = FeedbackArm(clock)
    q, t = arm.q.copy(), clock.time()
    after = 0.0
    if bad == 'stale':
        t -= 1.0
    elif bad == 'future':
        t += 1.0
    elif bad == 'shape':
        q = np.zeros((6, 1))
    elif bad == 'nan':
        q[0] = np.nan
    else:
        after = t + 0.001
    state = JointState(q=q, t=t)
    arm = SimpleNamespace(n_arm=6, get_state_arm=lambda: state, joint_state_arm=state,
                          get_config=arm.get_config)
    assert verify._real_current_q(arm, after=after)[0] is None


def test_real_feedback_allows_fresh_cached_fallback(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(verify, 'time', clock)
    arm = FeedbackArm(clock)
    arm.get_state_arm = lambda: None
    q, source = verify._real_current_q(arm)
    np.testing.assert_array_equal(q, np.zeros(6))
    assert '快照' in source


@pytest.mark.parametrize('follows,verdict', [(False, 'FAIL'), (True, 'PASS')])
def test_real_motion_verdict_requires_actual_displacement(monkeypatch, follows, verdict):
    clock = FakeClock()
    monkeypatch.setattr(verify, 'time', clock)
    monkeypatch.setattr(verify, '_wait_state', lambda *args, **kw: True)
    arm = FeedbackArm(clock, follows=follows)
    session = SimpleNamespace(arm=lambda: arm)
    result, _ = verify._rt131(session)
    assert result == verdict
    assert arm.target is not None
    assert float(np.max(np.abs(arm.target))) <= 0.05


def test_real_motion_skips_when_safe_margin_cannot_exceed_tolerance(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(verify, 'time', clock)
    monkeypatch.setattr(verify, '_wait_state', lambda *args, **kw: True)
    arm = FeedbackArm(clock)
    arm.joint_limits_soft_arm = SimpleNamespace(q_min=np.full(6, -0.01), q_max=np.full(6, 0.01))
    result, _ = verify._rt131(SimpleNamespace(arm=lambda: arm))
    assert result == 'SKIP'
    assert arm.target is None


def test_velocity_verification_stops_even_when_feedback_raises(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(verify, 'time', clock)
    arm = FeedbackArm(clock)
    commands = []
    arm.send_vel_arm = lambda dq: commands.append(np.array(dq))
    initial = arm.get_state_arm

    def read():
        if commands:
            raise RuntimeError('scripted feedback loss')
        return initial()

    arm.get_state_arm = read
    with pytest.raises(RuntimeError, match='feedback loss'):
        verify._rt122(SimpleNamespace(arm=lambda: arm))
    assert len(commands) == 2
    assert np.any(commands[0] != 0)
    np.testing.assert_array_equal(commands[-1], np.zeros(6))


def test_verification_selection_orders_dependencies_before_integration():
    nums = verify.resolve_steps('31,65,60,30,40,0')
    assert nums == [0, 30, 40, 65, 60, 31]
    assert len(verify.ITEMS) == len(verify.ALL_NUMS)
    legacy = {0, 1, 2, 3, *range(10, 18), *range(20, 24), 30, 31, 32,
              40, 41, 42, 50, 51, 52, 53, *range(60, 68), 70, 71, 72, 80, 81, 82, 90, 91, 92}
    assert legacy <= verify.ALL_NUMS


def test_verification_single_level_and_cumulative_selection():
    single = verify._selected_steps(verify.parse_args(['--level', 'L3']))
    through = verify._selected_steps(verify.parse_args(['--through', 'L3']))
    assert single and all(verify._BY_NUM[num]['level'] == 'L3' for num in single)
    assert set(through) == {it['num'] for it in verify.ITEMS if int(it['level'][1:]) <= 3}
    assert set(single) < set(through)
    with pytest.raises(SystemExit) as exc:
        verify.parse_args(['--level', 'L3', '--step', '60'])
    assert exc.value.code == 2


def test_real_level_selection_never_includes_zero_calibration():
    for level in range(6):
        args = verify.parse_args(['--real', '--through', f'r{level}'])
        assert 151 not in verify._selected_steps(args)
    assert verify.resolve_steps('151', real=True) == [151]
    assert verify._selected_steps(verify.parse_args(['--real', '--level', 'r5'])) == [150]
    with pytest.raises(verify._UsageError):
        verify._selected_steps(verify.parse_args(['--through', 'L9']))


def _suite_probe_items():
    return [dict(num=num, name=str(num), level=level, group=group, purpose='probe', risk='低')
            for num, level, group in [(0, 'L0', 'init'), (1, 'L0', 'init'),
                                       (30, 'L1', 'delegate'), (40, 'L2', 'solvers'),
                                       (90, 'L6', 'close')]]


def test_verification_failure_blocks_higher_levels_but_runs_peers_and_cleanup(capsys):
    items = _suite_probe_items()
    invoked = []

    def execute(it):
        invoked.append(it['num'])
        return ('FAIL' if it['num'] == 0 else 'PASS'), 'probe evidence'

    results = verify._run_suite(list(reversed(items)), execute)
    assert invoked == [0, 1, 90]
    assert [verdict for _, verdict, _ in results] == ['FAIL', 'PASS', 'SKIP', 'SKIP', 'PASS']
    assert verify._print_summary(results, ('init', 'delegate', 'solvers', 'close'), '离线') == 1
    output = capsys.readouterr().out
    assert 'LEVEL L0:' in output and 'GROUP close:' in output
    invoked.clear()
    verify._run_suite(items, execute, keep_going=True)
    assert invoked == [0, 1, 30, 40, 90]


def test_verification_framework_exception_and_user_quit():
    items = _suite_probe_items()

    def execute(it):
        if it['num'] == 0:
            raise RuntimeError('prefill failed')
        return 'PASS', 'probe'

    results = verify._run_suite(items, execute)
    assert results[0][1] == 'FAIL' and 'prefill failed' in results[0][2]
    assert results[-1][1] == 'PASS'
    assert len(verify._run_suite(items, lambda it: ('QUIT', 'user exit'))) == 1


def test_verification_list_does_not_construct_or_connect(monkeypatch, capsys):
    def forbidden(*args, **kw):
        pytest.fail('listing must not construct hardware or install injection')

    monkeypatch.setattr(verify, 'install_injection', forbidden)
    monkeypatch.setattr(verify._RealSession, 'arm', forbidden)
    assert verify.main(['--real', '--through', 'r2', '--list']) == 0
    output = capsys.readouterr().out
    assert '[120]' in output and '[130]' not in output and '[151]' not in output


def test_verification_item_failure_stops_created_dummy_refresh(rig):
    created = []

    def fail():
        arm, backend = verify.make_arm()
        created.append((arm, backend))
        raise RuntimeError('scripted item failure')

    item = dict(num=999, name='cleanup probe', group='init', level='L0', risk='低', purpose='probe', fn=fail)
    try:
        verdict, evidence = verify.run_item(item)
        assert verdict == 'FAIL' and 'scripted item failure' in evidence
        assert not created[0][1]._refresh_thread.is_alive()
        assert all(backend is not created[0][1] for backend in verify._CASE_BACKENDS)
    finally:
        for arm, _ in created:
            arm.close()


def test_disconnected_fsm_preserves_state_warns_once_and_recovers_after_reconnect(rig, caplog):
    arm, backend = rig
    arm._current_motion_state = MotionState.HOME
    arm.set_event(SafetyEvent.DISABLE)
    backend.disconnect_arm()
    mark = len(backend.calls)
    for _ in range(3):
        arm._fsm_tick()
    assert arm.get_current_motion_state() is MotionState.HOME
    assert sum('FSM 挂起' in record.message for record in caplog.records) == 1
    assert not backend.calls[mark:]
    assert arm.connect_arm() is True
    arm._fsm_tick()
    assert arm.get_current_motion_state() is MotionState.DISABLE
    assert arm.get_event() is SafetyEvent.DISABLE
    assert not arm._fsm_disc_warned


@pytest.mark.parametrize('stamp', [0.0, float('nan'), float('inf'), -1.0])
def test_arrival_rejects_invalid_timestamp(rig, stamp):
    arm, bk = rig
    bk._joint_state_arm.t = stamp
    assert not arm.is_in_q_arm(arm.home_arm)


def test_home_gate_rechecks_position_after_previous_arrival(rig):
    arm, bk = rig
    arm._current_motion_state = MotionState.HOME
    arm._home_arrived = True
    bk.set_q('arm', arm.home_arm + 0.2)
    arm.get_state_arm()
    arm.set_event(SafetyEvent.NORMAL)
    arm._fsm_tick()
    assert arm.get_current_motion_state() is MotionState.HOME
    assert arm._ctrl_periodic is None


@pytest.mark.parametrize('stage', ['plan', 'sample', 'step'])
@pytest.mark.parametrize('throws', [False, True])
def test_runtime_pipeline_failure_does_not_change_event(rig, monkeypatch, stage, throws):
    arm, bk = rig
    arm._current_motion_state = MotionState.MOTION
    arm.set_event(SafetyEvent.NORMAL)
    arm._motion_ready = arm._activated_ctrl_solver
    arm._activated_traj_solver.plan(arm, None)

    def fail(*args):
        if throws:
            raise RuntimeError('pipeline failure')
        return None if stage == 'sample' else False

    owner = arm._activated_ctrl_solver if stage == 'step' else arm._activated_traj_solver
    monkeypatch.setattr(owner, stage, fail)
    tick = arm._tick_plan if stage == 'plan' else arm._tick_ctrl
    if throws:
        with pytest.raises(RuntimeError, match='pipeline failure'):
            tick()
    else:
        tick()
    assert arm.get_event() is SafetyEvent.NORMAL
    assert arm.get_current_motion_state() is MotionState.MOTION
    assert arm.get_target_motion_state() is MotionState.MOTION
    assert arm._ctrl_ready()


def test_stop_escalation_during_zero_wait_finishes_without_full_timeout(rig, monkeypatch):
    arm, _ = rig
    arm._current_motion_state = MotionState.HOME
    arm.set_event(SafetyEvent.NORMAL)
    clock = FakeClock()
    monkeypatch.setattr(jm, 'time', clock)
    monkeypatch.setattr(arm, '_fsm_running', lambda: True)
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock.now += seconds
        arm.set_event(SafetyEvent.DISABLE)
        arm._current_motion_state = MotionState.DISABLE

    monkeypatch.setattr(clock, 'sleep', sleep)
    arm.stop_motion_fsm()
    assert len(slept) <= 2


def test_exception_in_context_disables_without_return_to_zero(rig, monkeypatch):
    arm, _ = rig
    seen = []
    monkeypatch.setattr(arm, 'close', lambda: seen.append(arm.get_event()))
    arm.set_event(SafetyEvent.NORMAL)
    with pytest.raises(RuntimeError):
        with arm:
            raise RuntimeError('operator abort')
    assert seen == [SafetyEvent.DISABLE]


def test_close_reclaims_backend_refresh_thread():
    verify.install_injection()
    arm, bk = verify.make_arm()
    try:
        arm.close()
        assert not bk._refresh_thread.is_alive()
    finally:
        bk.close()
        verify.restore_injection()


@pytest.mark.parametrize('offset', [-1.0, 1.0])
def test_arrival_rejects_old_or_future_snapshot(rig, offset):
    arm, bk = rig
    pose = arm.solve_fkine(arm.home_arm)
    bk._joint_state_arm.t = time.time() + offset
    assert not arm.is_in_q_arm(arm.home_arm)
    assert not arm.is_in_pose_arm(pose)


def test_set_target_rejects_nonfinite(rig):
    arm, _ = rig
    n = arm.n_arm
    with pytest.raises(ValueError, match='非有限'):
        arm.set_target_joint(float('nan'), np.zeros(n))
    with pytest.raises(ValueError, match='非有限'):
        arm.set_target_joint(time.time() + 1.0, np.full(n, np.nan))
    with pytest.raises(ValueError, match='非有限'):
        arm.set_target_joint(time.time() + 1.0, np.full(n, np.inf))
    with pytest.raises(ValueError, match='非有限'):
        arm.set_target_cart(time.time() + 1.0, np.full(7, np.inf))
    with pytest.raises(ValueError, match='非有限'):
        arm.set_target_cart(time.time() + 1.0, np.zeros(7), twist=np.full(6, np.nan))
    assert arm.get_target_trajframe() is None


@pytest.mark.parametrize('state,event', [(MotionState.HOME, SafetyEvent.HOMING),
                                        (MotionState.ZERO, SafetyEvent.ZEROING)])
def test_arrival_wait_does_not_set_safety_event(rig, state, event):
    arm, bk = rig
    arm._move_wait_timeout = 0.0
    bk.set_q('arm', arm.home_arm + 0.5)
    arm.get_state_arm()
    arm._current_motion_state = state
    arm.set_event(event)
    arm._home_arrived = False
    arm._zero_moving = True
    for _ in range(3):
        arm._fsm_tick()
    assert arm.get_event() is event
    assert arm.get_current_motion_state() is state
    assert arm.get_target_motion_state() is state


@pytest.mark.parametrize('bad', ['disconnected', 'missing', 'stale', 'future', 'zero_time',
                                 'nan_time', 'inf_time', 'shape', 'nan_q',
                                 'fault', 'missing_error', 'bad_error'])
def test_start_rejects_invalid_preconditions_without_changing_state(rig, monkeypatch, bad):
    arm, bk = rig
    state = arm.get_state_arm()
    if bad == 'disconnected':
        bk.disconnect_arm()
    elif bad == 'missing':
        state = None
    elif bad in ('stale', 'future', 'zero_time', 'nan_time', 'inf_time'):
        state.t = {'stale': time.time() - 1, 'future': time.time() + 1,
                   'zero_time': 0.0, 'nan_time': np.nan, 'inf_time': np.inf}[bad]
    elif bad == 'shape':
        state.q = np.zeros((arm.n_arm, 1))
    elif bad == 'nan_q':
        state.q[0] = np.nan
    elif bad == 'missing_error':
        state.error = None
    else:
        state.error[0] = 2 if bad == 'fault' else -1
    monkeypatch.setattr(bk, 'get_state_arm', lambda: state)
    arm.set_event(SafetyEvent.HOLDING)
    before = (arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event())
    mark = len(bk.calls)
    with pytest.raises(RuntimeError, match='拒绝启动'):
        arm.start_motion_fsm()
    assert (arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event()) == before
    assert arm._fsm_periodic is None
    assert not bk.calls[mark:]  # 前置检查失败不使能、不切模式、不下发运动。


def test_start_initial_state_and_monitor_rules_remain_on_monitor_thread(rig, monkeypatch):
    arm, _ = rig
    entered, release = threading.Event(), threading.Event()
    callers, initial = [], []
    original_start = jm._PeriodicThread.start

    def rule(*args):
        callers.append(threading.current_thread())
        entered.set()
        assert release.wait(2.0)
        return SafetyEvent.DAMPING

    def start(worker):
        if worker.name == 'joyarm-fsm':
            initial.append((arm.get_current_motion_state(), arm.get_target_motion_state(), arm.get_event()))
        else:
            original_start(worker)

    monkeypatch.setattr(jm._PeriodicThread, 'start', start)
    arm.set_monitor_rules(rule)
    arm.start_monitor()
    try:
        assert entered.wait(1.0)
        arm._monitor_none_since = 123.0
        arm.start_motion_fsm()
        assert initial == [(MotionState.DISABLE, MotionState.MOTION, SafetyEvent.NORMAL)]
        assert arm._monitor_none_since == 123.0
    finally:
        release.set()
        arm.stop_monitor()
    assert callers and all(t.name == 'joyarm-monitor' for t in callers)
    assert arm.get_event() is SafetyEvent.DAMPING
    assert arm.get_target_motion_state() is MotionState.DAMPING


def test_missing_control_feedback_skips_step_without_changing_event(rig):
    arm, bk = rig
    arm._current_motion_state = MotionState.MOTION
    arm.set_event(SafetyEvent.NORMAL)
    arm._motion_ready = arm._activated_ctrl_solver
    bk.fail_at['read_joint_state_arm'] = None
    mark = len(bk.calls)
    arm._tick_ctrl()
    assert not bk.calls_since(mark, 'send_joint_position_arm')
    assert arm.get_event() is SafetyEvent.NORMAL
    assert arm._ctrl_ready()


def test_periodic_retries_algorithm_exception_without_setting_event(rig, monkeypatch, caplog):
    arm, _ = rig
    arm.set_event(SafetyEvent.NORMAL)
    stop = threading.Event()
    calls = []

    def plan(*args):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError('scripted transient plan failure')
        stop.set()
        return True

    monkeypatch.setattr(arm._activated_traj_solver, 'plan', plan)
    jm._run_periodic(stop, 100.0, arm._tick_plan)
    assert len(calls) == 2
    assert arm.get_event() is SafetyEvent.NORMAL
    assert '下周期重试' in caplog.text
