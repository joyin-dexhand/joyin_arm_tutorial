"""JointPositionController 离线深度测试（哑臂驱动，不依赖硬件）。

覆盖：①MODE=POSITION 声明与 ctrl_hz 属性；②小步原样直通；③大步逐关节
独立裁剪到 q_cur ± dq_max/ctrl_hz（含混合步长与负方向）；④自定义 ctrl_hz
换算；⑤越限告警 0.5s 节流；⑥frame 缺 q / q 含 NaN 显性拒绝；⑦arm 无限位
声明（arm_limits=None）直通不裁剪；⑧返回契约 (模式, 指令字典)。

运行：``python test/test_controller_joint_position.py`` 或 pytest。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import (  # noqa: E402
    ArmState, ControlMode, JointLimits, JointPositionController, JointState,
    Pose, TrajFrame,
)


class _LimArm:
    """控制器测试哑臂：提供可控硬限位。"""

    def __init__(self, dq_max, q_min=-3.0, q_max=3.0):
        self.arm_limits = JointLimits(q_min=np.full(6, q_min),
                                      q_max=np.full(6, q_max),
                                      dq_max=np.full(6, dq_max),
                                      tau_max=np.full(6, 20.0))


class _NoLimArm:
    """无限位哑臂（arm_limits=None）：控制器直通分支。"""

    arm_limits = None


def _state(q):
    return ArmState(joint=JointState(q=np.asarray(q, dtype=float)))


def test_mode_declaration_and_attrs():
    """MODE 类属性声明 POSITION；ctrl_hz 注入为属性。"""
    ctrl = JointPositionController(ctrl_hz=200.0)
    assert ctrl.MODE == ControlMode.POSITION
    assert JointPositionController.MODE == ControlMode.POSITION   # 类级声明
    assert ctrl.ctrl_hz == 200.0
    ctrl100 = JointPositionController(ctrl_hz=100.0)
    assert ctrl100.ctrl_hz == 100.0


def test_small_step_passthrough():
    """小步（|Δq| ≤ dq_max/ctrl_hz）：目标原样直通，无裁剪。"""
    ctrl = JointPositionController(ctrl_hz=200.0)
    arm = _LimArm(dq_max=2.0)                               # step_max = 0.01
    st = _state(np.full(6, 0.5))
    fr = TrajFrame(time=1.0, q=np.full(6, 0.505))
    mode, cmd = ctrl.compute(arm, fr, st)
    assert mode == ControlMode.POSITION
    assert np.allclose(cmd["q"], 0.505)                     # 直通


def test_large_step_clipped_per_joint():
    """大步：逐关节独立裁剪到 q_cur ± step_max（混合步长 + 负方向）。"""
    ctrl = JointPositionController(ctrl_hz=200.0)
    arm = _LimArm(dq_max=2.0)                               # step_max = 0.01
    st = _state(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    q_ref = np.array([0.505, 0.9, 0.505, 0.4, 0.505, 0.505])
    # 关节 1/3 超限（+0.4/−0.1），其余小步直通
    _, cmd = ctrl.compute(arm, TrajFrame(time=1.0, q=q_ref), st)
    expect = np.array([0.505, 0.51, 0.505, 0.49, 0.505, 0.505])
    assert np.allclose(cmd["q"], expect)
    # 负向大步：q_cur=0.5、q_ref=-0.9 → 钳到 0.5-0.01
    _, cmd2 = ctrl.compute(arm, TrajFrame(time=1.0, q=np.full(6, -0.9)), st)
    assert np.allclose(cmd2["q"], 0.49)


def test_step_bound_follows_custom_hz():
    """步长上限 = dq_max / ctrl_hz：ctrl_hz=100 时步长上限翻倍。"""
    ctrl = JointPositionController(ctrl_hz=100.0)
    arm = _LimArm(dq_max=2.0)                               # step_max = 0.02
    st = _state(np.full(6, 0.5))
    _, cmd = ctrl.compute(arm, TrajFrame(time=1.0, q=np.full(6, 0.9)), st)
    assert np.allclose(cmd["q"], 0.52)                      # 0.5 + 0.02


def test_warning_throttled():
    """越限告警节流：连续越限 0.5s 至多一条；恢复后再越限重新告警。"""
    class _Counting(logging.Handler):
        n = 0

        def emit(self, record):
            self.n += 1

    handler = _Counting()
    log = logging.getLogger("joyarm_core.controller")
    log.addHandler(handler)
    try:
        ctrl = JointPositionController(ctrl_hz=200.0)
        arm = _LimArm(dq_max=2.0)
        st = _state(np.full(6, 0.5))
        big = TrajFrame(time=1.0, q=np.full(6, 0.9))
        small = TrajFrame(time=1.0, q=np.full(6, 0.505))
        ctrl.compute(arm, big, st)
        n1 = handler.n
        assert n1 == 1                                       # 首次告警
        for _ in range(5):
            ctrl.compute(arm, big, st)
        assert handler.n in (n1, n1 + 1)                     # 0.5s 内至多再 1 条
        ctrl.compute(arm, small, st)                         # 恢复（不告警）
        assert handler.n == 1 or handler.n == 2
    finally:
        log.removeHandler(handler)


def test_missing_q_rejected():
    """当前帧缺 q（如 pose 型帧）：显性 ValueError。"""
    ctrl = JointPositionController()
    arm = _LimArm(dq_max=2.0)
    fr = TrajFrame(time=1.0, pose=Pose())
    try:
        ctrl.compute(arm, fr, _state(np.zeros(6)))
        raise AssertionError("缺 q 应抛 ValueError")
    except ValueError as e:
        assert "缺 q" in str(e)


def test_nan_q_rejected():
    """q 含 NaN/inf：拒绝下发（显性 ValueError，防毒化指令）。"""
    ctrl = JointPositionController()
    arm = _LimArm(dq_max=2.0)
    st = _state(np.zeros(6))
    for bad in (np.full(6, np.nan), np.full(6, np.inf)):
        try:
            ctrl.compute(arm, TrajFrame(time=1.0, q=bad), st)
            raise AssertionError("NaN/inf 应抛 ValueError")
        except ValueError as e:
            assert "非有限" in str(e)


def test_no_limits_passthrough():
    """arm 未声明硬限位（arm_limits=None）：不裁剪直通。"""
    ctrl = JointPositionController(ctrl_hz=200.0)
    st = _state(np.full(6, 0.5))
    _, cmd = ctrl.compute(_NoLimArm(),
                          TrajFrame(time=1.0, q=np.full(6, 2.0)), st)
    assert np.allclose(cmd["q"], 2.0)                       # 大步也直通


def test_return_contract():
    """返回契约：(ControlMode.POSITION, {"q": (n,)})，仅含 q 键。"""
    ctrl = JointPositionController(ctrl_hz=200.0)
    arm = _LimArm(dq_max=2.0)
    mode, cmd = ctrl.compute(arm, TrajFrame(time=1.0, q=np.full(6, 0.505)),
                             _state(np.full(6, 0.5)))
    assert isinstance(mode, ControlMode) and mode is ControlMode.POSITION
    assert set(cmd) == {"q"}
    assert np.asarray(cmd["q"]).shape == (6,)
    assert np.all(np.isfinite(cmd["q"]))


def test_ctor_rejects_nonpositive_hz():
    """ctrl_hz 非正在构造时暴露（防线程静默死亡）。"""
    for bad in (0, -5):
        try:
            JointPositionController(ctrl_hz=bad)
            raise AssertionError(f"ctrl_hz={bad} 应抛 ValueError")
        except ValueError:
            pass


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
