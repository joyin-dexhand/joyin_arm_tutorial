"""ToJointTrajPlanner 离线深度测试（哑臂驱动，不依赖硬件与真实模型）。

覆盖：①五次多项式六端点边界条件（钳位采样精确断言：起点 τ=0 / 末端
τ=T）与目标 dq/ddq 缺省 0；②采样帧导数与数值中心差分交叉验证；③超末端
钳位保持末态；④多帧目标拒绝（单帧策略：告警去重 + 回退 q_home）；⑤短时长
防除零；⑥pose 目标拒绝、未规划先采样 RuntimeError；⑦_check_frame 全分支
判别；⑧plan_once 剔除无效/超时目标 + 同批告警去重 + 空目标回退 q_home
（回退帧时刻语义）；⑨回退失败返 False；⑩构造参数注入与频率校验；⑪重规划
系数整体替换（原子发布）。

运行：``python test/test_traj_planner_to_joint.py`` 或 pytest。
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import (  # noqa: E402
    ArmState, JointState, Pose, ToJointTrajPlanner, TrajFrame, TrajPlanner,
    Twist, Wrench,
)


class _QArm:
    """规划器测试哑臂：状态可控、目标可注入。"""

    def __init__(self, q, dq, arm_home=None):
        self.state = ArmState(joint=JointState(q=np.asarray(q, float),
                                               dq=np.asarray(dq, float)))
        self.arm_home = np.zeros_like(self.state.joint.q) \
            if arm_home is None else np.asarray(arm_home, float)
        self._target = None

    def get_target_traj(self):
        return self._target

    def get_arm_state(self):
        return self.state


class _DeadArm:
    """断连哑臂：get_arm_state 抛错、无目标（回退帧构造失败路径）。"""

    def get_target_traj(self):
        return None

    def get_arm_state(self):
        raise RuntimeError("未连接真机")


# ============================================================
# 五次多项式内核：边界条件 / 导数 / 钳位 / 多目标 / 短时长
# ============================================================
_Q0 = np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1])
_DQ0 = np.array([0.05, 0.0, -0.05, 0.1, 0.0, 0.0])
_Q1 = np.array([0.5, 0.1, 0.6, 0.2, 0.3, 0.2])
_DQ1 = np.array([0.0, 0.02, 0.0, 0.0, 0.0, -0.01])


def _plan_default(T=2.0):
    """标准场景规划：q0/dq0 起点 → (t_end) 目标 q1/dq1（ddq 缺省 0）。"""
    fa = _QArm(_Q0, _DQ0)
    t_end = time.time() + T
    fa._target = [TrajFrame(time=t_end, q=_Q1, dq=_DQ1)]
    qp = ToJointTrajPlanner()
    assert qp.plan_once(fa) is True
    return qp, t_end


def test_quintic_boundary_conditions():
    """六端点边界条件：起点 (q0, dq0, ddq0=0) / 末端 (q1, dq1, ddq1=0) 精确。

    钳位采样将 τ 钉死在 0 / T，断言不受执行期时间漂移影响。
    """
    qp, t_end = _plan_default(T=2.0)
    fr0 = qp.sample_frame(time.time() - 10.0)               # τ 钳位 0（起点）
    assert np.allclose(fr0.q, _Q0, atol=1e-9)
    assert np.allclose(fr0.dq, _DQ0, atol=1e-9)
    assert np.allclose(fr0.ddq, 0.0, atol=1e-9)             # ddq0 恒 0（无反馈）
    fr1 = qp.sample_frame(t_end + 10.0)                     # τ 钳位 T（末端）
    assert np.allclose(fr1.q, _Q1, atol=1e-9)
    assert np.allclose(fr1.dq, _DQ1, atol=1e-9)
    assert np.allclose(fr1.ddq, 0.0, atol=1e-9)             # 目标 ddq 缺省 0


def test_sample_derivatives_central_diff():
    """导数正确性交叉验证：dq ≈ q 的中心差分、ddq ≈ dq 的中心差分。"""
    qp, _ = _plan_default(T=2.0)
    h = 1e-3
    now = time.time()
    for dt in (0.5, 1.0, 1.5):                              # 轨迹中段
        t = now + dt
        fr_m, fr_0, fr_p = (qp.sample_frame(t + s * h) for s in (-1, 0, 1))
        assert np.allclose(fr_0.dq,
                           (fr_p.q - fr_m.q) / (2 * h), atol=5e-4)
        assert np.allclose(fr_0.ddq,
                           (fr_p.dq - fr_m.dq) / (2 * h), atol=5e-3)
    fr = qp.sample_frame(now + 1.0)
    assert fr.time == now + 1.0                             # 帧时间 = 采样时刻


def test_default_target_derivatives_zero():
    """目标只给 q（不给 dq/ddq）：末端速度/加速度为 0（缺省语义）。"""
    fa = _QArm(_Q0, _DQ0 * 0)
    t_end = time.time() + 1.5
    fa._target = [TrajFrame(time=t_end, q=_Q1)]
    qp = ToJointTrajPlanner()
    qp.plan_once(fa)
    fr1 = qp.sample_frame(t_end + 5.0)
    assert np.allclose(fr1.q, _Q1, atol=1e-9)
    assert np.allclose(fr1.dq, 0.0, atol=1e-9)
    assert np.allclose(fr1.ddq, 0.0, atol=1e-9)


def test_multi_target_rejected_fallback_home():
    """多帧目标：告警（仅支持单帧）并回退规划到 q_home；重复同批不刷屏。"""
    class _Counting(logging.Handler):
        n = 0

        def emit(self, record):
            self.n += 1

    handler = _Counting()
    log = logging.getLogger("joyarm_core.traj_planner_to_joint")
    log.addHandler(handler)
    try:
        fa = _QArm(_Q0, np.zeros(6), arm_home=_Q1)
        t_end = time.time() + 1.0
        fa._target = [TrajFrame(time=t_end, q=_Q1),
                      TrajFrame(time=t_end + 1.0, q=_Q1 * 0.5)]
        qp = ToJointTrajPlanner()
        assert qp.plan_once(fa) is True                      # 回退路径完成规划
        assert handler.n == 1                                # 多帧告警
        qp.plan_once(fa)
        assert handler.n == 1                                # 同时刻序列去重
        fr = qp.sample_frame(t_end + 6.0)
        assert np.allclose(fr.q, _Q1, atol=1e-9)             # 回家，非任一目标
    finally:
        log.removeHandler(handler)


def test_short_duration_guard():
    """目标时刻极近（T → 0）：时长下限 1e-3 防除零，边界条件仍精确满足。"""
    fa = _QArm(_Q0, np.zeros(6))
    t_end = time.time() + 0.0005
    fa._target = [TrajFrame(time=t_end, q=_Q1)]
    qp = ToJointTrajPlanner(dt_min_required=0.0)            # 放行超近目标
    qp.plan_once(fa)
    fr = qp.sample_frame(t_end + 1.0)
    assert np.all(np.isfinite(fr.q)) and np.allclose(fr.q, _Q1, atol=1e-9)


def test_pose_target_rejected():
    """纯 pose 目标：入口校验放行（基类契约）→ _plan 明确拒绝（ValueError）。"""
    fa = _QArm(_Q0, np.zeros(6))
    fa._target = [TrajFrame(time=time.time() + 2.0,
                            pose=Pose(position=np.array([0.3, 0, 0.2])))]
    qp = ToJointTrajPlanner()
    try:
        qp.plan_once(fa)
        raise AssertionError("pose 目标应被 _plan 拒绝")
    except ValueError as e:
        assert "关节目标" in str(e)
    assert qp._coeffs is None                                # 异常前未发布系数


def test_sample_before_plan_runtime_error():
    """未规划先采样：显性 RuntimeError（运行期保护）。"""
    qp = ToJointTrajPlanner()
    try:
        qp.sample_frame(time.time())
        raise AssertionError("未规划采样应抛 RuntimeError")
    except RuntimeError as e:
        assert "尚未规划" in str(e)


# ============================================================
# _check_frame 全分支（基类校验管线）
# ============================================================
def test_check_frame_valid_cases():
    """合法帧：纯 q / q+dq+ddq / pose+wrench。"""
    assert TrajPlanner._check_frame(TrajFrame(time=1.0, q=np.zeros(6))) is None
    assert TrajPlanner._check_frame(
        TrajFrame(time=1.0, q=np.zeros(6), dq=np.zeros(6),
                  ddq=np.zeros(6))) is None
    assert TrajPlanner._check_frame(
        TrajFrame(time=1.0, pose=Pose(),
                  wrench=Wrench())) is None


def test_check_frame_time_problems():
    """time 缺失（0.0）/ NaN / inf：一律剔除。"""
    for t in (0.0, float("nan"), float("inf")):
        problem = TrajPlanner._check_frame(TrajFrame(time=t, q=np.zeros(6)))
        assert problem is not None and "time" in problem


def test_check_frame_pose_q_exclusivity():
    """pose/q 恰一非空：双空 / 双满均拒绝。"""
    assert "双双为空" in TrajPlanner._check_frame(TrajFrame(time=1.0))
    assert "双双非空" in TrajPlanner._check_frame(
        TrajFrame(time=1.0, pose=Pose(), q=np.zeros(6)))


def test_check_frame_joint_target_field_rules():
    """关节目标：禁 twist/tau/wrench；q/dq/ddq 禁 NaN/inf。"""
    zero = np.zeros(6)
    cases = [
        dict(q=zero, twist=Twist()),                        # 关节目标禁 twist
        dict(q=zero, tau=zero),                             # 禁 tau
        dict(q=zero, wrench=Wrench()),                      # 禁 wrench
        dict(q=np.full(6, np.nan)),                         # q 含 NaN
        dict(q=zero, dq=np.full(6, np.inf)),                # dq 含 inf
        dict(q=zero, ddq=np.full(6, np.nan)),               # ddq 含 NaN
    ]
    for kw in cases:
        problem = TrajPlanner._check_frame(TrajFrame(time=1.0, **kw))
        assert problem is not None, kw
        assert ("非有限" in problem) or ("应为空" in problem), (kw, problem)


def test_check_frame_pose_target_field_rules():
    """pose 目标：禁 twist/dq/ddq/tau，仅可带 wrench。"""
    zero = np.zeros(6)
    for extra in (dict(twist=Twist()), dict(dq=zero), dict(ddq=zero),
                  dict(tau=zero)):
        problem = TrajPlanner._check_frame(
            TrajFrame(time=1.0, pose=Pose(), **extra))
        assert problem is not None and "应为空" in problem, (extra, problem)


# ============================================================
# plan_once 管线：剔除 / 告警去重 / 回退 / 返回值
# ============================================================
def test_plan_once_drops_stale_target():
    """超时目标（time 早于 now+dt_min_required）剔除 → 回退 q_home。"""
    fa = _QArm(_Q0, np.zeros(6), arm_home=_Q1)
    fa._target = [TrajFrame(time=time.time() - 5.0, q=_Q1 * 0.3)]  # 过期目标
    qp = ToJointTrajPlanner(dt_min_required=1.0)
    assert qp.plan_once(fa) is True                          # 回退路径也算完成
    fr = qp.sample_frame(time.time() + 30.0)                 # 回退轨迹已走完
    assert np.allclose(fr.q, _Q1, atol=1e-9)                 # 回家而非过期目标


def test_plan_once_drops_nan_target_and_falls_back():
    """NaN q 目标入口剔除 → 空目标回退 q_home。"""
    fa = _QArm(_Q0, np.zeros(6), arm_home=_Q1)
    fa._target = [TrajFrame(time=time.time() + 2.0,
                            q=np.full(6, np.nan))]
    qp = ToJointTrajPlanner()
    assert qp.plan_once(fa) is True
    fr = qp.sample_frame(time.time() + 30.0)
    assert np.all(np.isfinite(fr.q)) and np.allclose(fr.q, _Q1, atol=1e-9)


def test_drop_warning_deduplicated():
    """同批无效目标告警去重：同内容只 1 条；恢复后再现重新告警。"""
    class _Counting(logging.Handler):
        n = 0

        def emit(self, record):
            self.n += 1

    handler = _Counting()
    log = logging.getLogger("joyarm_core.traj_planner")
    log.addHandler(handler)
    try:
        fa = _QArm(_Q0, np.zeros(6))
        bad = [TrajFrame(time=0.0, q=_Q1)]                   # time 缺失 → 无效
        good = [TrajFrame(time=time.time() + 2.0, q=_Q1)]
        qp = ToJointTrajPlanner()
        fa._target = bad
        qp.plan_once(fa)
        n1 = handler.n
        assert n1 == 1                                       # 首次告警
        qp.plan_once(fa)
        assert handler.n == n1                               # 同批去重
        fa._target = good
        qp.plan_once(fa)                                     # 恢复正常：复位
        fa._target = bad
        qp.plan_once(fa)
        assert handler.n == n1 + 1                           # 再现重新告警
    finally:
        log.removeHandler(handler)


def test_home_fallback_frame_semantics():
    """回退帧语义（静态直测）：q=q_home、时刻 = now + max|q_home−q0|。"""
    q0 = np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1])
    home = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
    fa = _QArm(q0, np.zeros(6), arm_home=home)
    fr = TrajPlanner._home_fallback(fa, now=100.0)
    assert fr.time == 100.0 + float(np.max(np.abs(home - q0)))
    assert np.allclose(fr.q, home)
    assert fr.pose is None and fr.dq is None                # 纯关节帧


def test_plan_once_fallback_failure_returns_false():
    """回退帧构造失败（断连读不到状态）：返回 False、沿用旧系数。"""
    qp = ToJointTrajPlanner()
    assert qp.plan_once(_DeadArm()) is False


def test_replan_replaces_coeffs_atomically():
    """重规划：系数包整体替换（新对象），非原地修改（并发契约）。"""
    fa = _QArm(_Q0, np.zeros(6))
    fa._target = [TrajFrame(time=time.time() + 1.0, q=_Q1)]
    qp = ToJointTrajPlanner()
    qp.plan_once(fa)
    coeffs1 = qp._coeffs
    fa._target = [TrajFrame(time=time.time() + 1.0, q=_Q1 * 0.5)]
    qp.plan_once(fa)
    assert qp._coeffs is not coeffs1                         # 一次性原子赋值
    assert coeffs1[1] > 0                                    # 旧包未被破坏


# ============================================================
# 构造参数与频率校验
# ============================================================
def test_ctor_params_and_hz_validation():
    """三构造参数注入生效；plan_hz/sample_hz 非正在构造时暴露。"""
    qp = ToJointTrajPlanner(plan_hz=10.0, sample_hz=250.0, dt_min_required=0.5)
    assert (qp.plan_hz, qp.sample_hz, qp.dt_min_required) == (10.0, 250.0, 0.5)
    for bad in (0, -5):
        for ctor in (lambda v: ToJointTrajPlanner(plan_hz=v),
                     lambda v: ToJointTrajPlanner(sample_hz=v)):
            try:
                ctor(bad)
                raise AssertionError(f"hz={bad} 应抛 ValueError")
            except ValueError:
                pass
    ToJointTrajPlanner(plan_hz=5.0, sample_hz=200.0)         # 正常值不受影响


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
