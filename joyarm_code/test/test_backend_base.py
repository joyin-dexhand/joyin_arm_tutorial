"""Backend 基类限位守卫离线单测（硬限位裁剪唯一执行点：send_* 模板前置，arm/end 两族）。

覆盖：①限位构建（构造参数：本体硬限位；cfg 自解析：末端逐电机硬限位，
缺键量纲 ±∞，无 end 段不启用；只存硬限位、无软限位属性）；②本体守卫
（q 硬限位 / dq、tau 幅值 / kp、kd 透传 / 单关节切片 / 维度不符放行）；
③末端守卫（逐电机裁剪 / 标量广播裁剪（多执行器）/ force 按 ±tau_max 数值
守卫 / MIT 末端）；④越限告警节流。

运行：``python test/test_backend_base.py`` 或 pytest。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import Backend, JointLimits  # noqa: E402

# 与指令内核无关的其余抽象方法：哑化（type() 创建时注入以满足 ABC 检查；
# send_position/force/mit_end 已为基类模板，不在哑化列表）
_STUB_NAMES = (
    "connect", "disconnect",
    "enable_arm", "disable_arm", "set_zero_arm", "clear_fault_arm", "set_mode_arm",
    "read_mode_arm", "read_state_arm", "read_param_arm", "write_param_arm",
    "enable_end", "disable_end", "set_zero_end", "clear_fault_end", "set_mode_end",
    "read_mode_end", "read_state_end", "send_action_end",
    "read_param_end", "write_param_end",
)


def _dummy_backend(cfg: dict | None = None, **ctor_kw) -> Backend:
    """离线哑后端：仅记录六个指令内核收到的参数；构造参数原样透传基类。"""

    def _init(self, cfg=None, arm_limits=None):
        Backend.__init__(self, cfg or {}, arm_limits=arm_limits)
        self.kernel = []          # [(方法名, 收到的参数元组)]

    def _noop(self, *a, **k):
        return None

    def _rec(tag):
        def _kernel(self, *a, **k):
            self.kernel.append((tag,) + tuple(np.asarray(x, float) if np.ndim(x) else x
                                              for x in a) + (k.get("joint"),))
        return _kernel

    ns = {
        "__init__": _init,
        "connected": property(lambda self: True),
        "_send_position_arm": _rec("pos_arm"), "_send_velocity_arm": _rec("vel_arm"),
        "_send_position_end": _rec("pos_end"), "_send_tau_end": _rec("tau_end"),
    }

    def _mit_arm(self, q, dq, tau_ff, kp=None, kd=None, joint=None):
        self.kernel.append(("mit_arm", np.asarray(q, float), np.asarray(dq, float),
                            np.asarray(tau_ff, float), kp, kd, joint))

    def _mit_end(self, q, dq, tau_ff, kp=None, kd=None, joint=None):
        self.kernel.append(("mit_end", np.asarray(q, float), np.asarray(dq, float),
                            np.asarray(tau_ff, float), kp, kd, joint))

    ns["_send_mit_arm"] = _mit_arm
    ns["_send_mit_end"] = _mit_end
    ns.update({name: _noop for name in _STUB_NAMES})
    return type("DummyBackend", (Backend,), ns)(cfg, **ctor_kw)


def _hard(n: int = 3) -> JointLimits:
    """本体硬限位：q ∈ [-1,1]，dq ≤ 2，tau ≤ 10。"""
    return JointLimits(q_min=np.full(n, -1.0), q_max=np.full(n, 1.0),
                       dq_max=np.full(n, 2.0), tau_max=np.full(n, 10.0))


# 双电机末端：电机 2 缺 tau_max（→ ∞ 不参与守卫）、行程更紧
_END_CFG = {"end": {"joints": [
    {"name": "g1", "q_min": -1.0, "q_max": 1.0, "dq_max": 2.0, "tau_max": 10.0},
    {"name": "g2", "q_min": -0.5, "q_max": 0.5, "dq_max": 3.0},
]}}


# ----------------------------------------------------------
# 限位构建（构造参数 + cfg 自解析）
# ----------------------------------------------------------
def test_limits_build_from_constructor():
    """本体硬限位随构造传入原样保存；backend 只存硬限位（无软限位属性）。"""
    be = _dummy_backend(arm_limits=_hard(3))
    s = be.arm_limits
    assert np.allclose(s.q_min, -1.0) and np.allclose(s.q_max, 1.0)
    assert np.allclose(s.dq_max, 2.0) and np.allclose(s.tau_max, 10.0)
    for gone in ("arm_limits_soft", "end_limits_soft",
                 "set_arm_limits_soft", "set_end_limits_soft"):
        assert not hasattr(be, gone), f"{gone} 应已随软限位机制移除"


def test_end_limits_from_cfg():
    """末端硬限位自 cfg 逐电机解析（缺键 ±∞）；无 end 段 → None。"""
    be = _dummy_backend(cfg=_END_CFG)
    e = be.end_limits
    assert np.allclose(e.q_min, [-1.0, -0.5]) and np.allclose(e.q_max, [1.0, 0.5])
    assert np.allclose(e.dq_max, [2.0, 3.0])
    assert np.isfinite(e.tau_max[0]) and np.isinf(e.tau_max[1])
    be2 = _dummy_backend(cfg={"arm": {}})
    assert be2.end_limits is None


# ----------------------------------------------------------
# 本体守卫（模板前置）
# ----------------------------------------------------------
def test_guard_passthrough_without_limits():
    """未传本体硬限位（独立使用）：本体守卫放行。"""
    be = _dummy_backend()
    assert be.arm_limits is None
    be.send_position_arm(np.full(3, 99.0))
    assert np.allclose(be.kernel[0][1], 99.0)


def test_guard_clips_q_dq_tau():
    """本体：q 硬限位裁剪、dq/tau 幅值裁剪、kp/kd 透传。"""
    be = _dummy_backend(arm_limits=_hard(3))
    be.send_position_arm(np.full(3, 5.0))                       # q → 1.0（硬上限）
    assert be.kernel[-1][0] == "pos_arm" and np.allclose(be.kernel[-1][1], 1.0)
    be.send_velocity_arm(np.full(3, -9.0))                      # dq → -2.0
    assert np.allclose(be.kernel[-1][1], -2.0)
    be.send_mit_arm(np.full(3, -5.0), np.full(3, 9.0), np.full(3, 50.0),
                    kp=np.full(3, 7.0), kd=np.full(3, 8.0))
    _, q, dq, tau, kp, kd, _ = be.kernel[-1]
    assert np.allclose(q, -1.0) and np.allclose(dq, 2.0) and np.allclose(tau, 10.0)
    assert np.allclose(kp, 7.0) and np.allclose(kd, 8.0)


def test_guard_single_joint_slice():
    """单关节指令：取该关节限位切片（其余关节限位不参与）。"""
    be = _dummy_backend(arm_limits=_hard(3))
    be._arm_limits.q_max[1] = 0.5                       # joint2 上限更紧
    be.send_position_arm(np.array([99.0]), joint=1)
    assert be.kernel[-1][0] == "pos_arm" and np.allclose(be.kernel[-1][1], 0.5)
    assert be.kernel[-1][2] == 1


def test_guard_dimension_mismatch_passthrough():
    """指令长度 ≠ n：守卫放行，由内核（真实后端）报清晰的维度错误。"""
    be = _dummy_backend(arm_limits=_hard(3))
    be.send_position_arm(np.zeros(4))
    assert be.kernel[-1][1].shape == (4,)
    be2 = _dummy_backend(cfg=_END_CFG)
    be2.send_position_end(np.zeros(3))                    # 末端同理
    assert be2.kernel[-1][1].shape == (3,)


def test_guard_scalar_promotion():
    """标量指令升维为 (1,)（单关节用法），守卫按长度放行交内核校验。"""
    be = _dummy_backend(arm_limits=_hard(3))
    be.send_position_arm(1.5, joint=0)
    assert be.kernel[-1][1].shape == (1,) and np.allclose(be.kernel[-1][1], 1.0)


# ----------------------------------------------------------
# 末端守卫（逐电机 + 标量广播）
# ----------------------------------------------------------
def test_end_guard_clips_position():
    """末端位置：逐电机裁剪到各自行程。"""
    be = _dummy_backend(cfg=_END_CFG)
    be.send_position_end(np.array([5.0, -5.0]))
    assert be.kernel[-1][0] == "pos_end"
    assert np.allclose(be.kernel[-1][1], [1.0, -0.5])


def test_end_guard_scalar_broadcast():
    """末端标量广播（多执行器）：按逐电机限位各自就近裁剪 → (n_end,)。"""
    be = _dummy_backend(cfg=_END_CFG)
    be.send_position_end(99.0)                             # g1→1.0，g2→0.5
    assert be.kernel[-1][1].shape == (2,)
    assert np.allclose(be.kernel[-1][1], [1.0, 0.5])


def test_end_guard_single_motor_slice():
    """末端单电机指令：取该电机限位切片。"""
    be = _dummy_backend(cfg=_END_CFG)
    be.send_position_end(np.array([9.0]), joint=1)
    assert be.kernel[-1][0] == "pos_end" and np.allclose(be.kernel[-1][1], 0.5)
    assert be.kernel[-1][2] == 1


def test_end_guard_tau_clip():
    """末端力矩：tau 按逐电机 ±tau_max 裁剪（缺 tau_max 不裁）。"""
    be = _dummy_backend(cfg=_END_CFG)
    be.send_tau_end(50.0)                                  # 标量广播：g1→10，g2→∞不裁
    assert be.kernel[-1][0] == "tau_end"
    assert np.allclose(be.kernel[-1][1], [10.0, 50.0])


def test_end_guard_mit():
    """末端 MIT：q/dq/tau 守卫，kp/kd 透传。"""
    be = _dummy_backend(cfg=_END_CFG)
    be.send_mit_end(np.array([5.0, -5.0]), np.array([9.0, -9.0]),
                    np.array([50.0, -50.0]), kp=np.ones(2), kd=np.ones(2))
    _, q, dq, tau, kp, kd, _ = be.kernel[-1]
    assert np.allclose(q, [1.0, -0.5])
    assert np.allclose(dq, [2.0, -3.0])
    assert np.allclose(tau, [10.0, -50.0])                 # g2 tau_max=∞ 不裁
    assert np.allclose(kp, 1.0) and np.allclose(kd, 1.0)


def test_end_guard_without_end_section():
    """无 end 段：末端守卫放行（透传内核）。"""
    be = _dummy_backend(arm_limits=_hard(3))
    be.send_position_end(np.array([99.0]))
    assert np.allclose(be.kernel[-1][1], 99.0)


# ----------------------------------------------------------
# 越限告警
# ----------------------------------------------------------


class _Cap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def test_guard_warns_on_clip():
    """越限告警节流：首条即告警，0.5s 内重复越限静默（裁剪照常），过后再告警。"""
    import joyarm_core.backend.backend as be_mod
    be = _dummy_backend(arm_limits=_hard(3))
    cap = _Cap()
    log = logging.getLogger("joyarm_core.backend")
    log.addHandler(cap)
    clock = {"t": 100.0}                               # 伪时钟（替换模块 time，finally 恢复）
    orig_time = be_mod.time
    be_mod.time = SimpleNamespace(monotonic=lambda: clock["t"])
    try:
        be.send_position_arm(np.array([0.5, 9.0, -0.5]))    # 首次越限 → 告警
        assert len(cap.msgs) == 1 and "越限" in cap.msgs[0]
        assert np.allclose(be.kernel[-1][1], np.array([0.5, 1.0, -0.5]))
        clock["t"] += 0.2
        be.send_position_arm(np.array([0.5, 9.0, -0.5]))    # 0.2s 内 → 节流静默
        assert len(cap.msgs) == 1
        assert np.allclose(be.kernel[-1][1], np.array([0.5, 1.0, -0.5]))   # 裁剪不受节流影响
        clock["t"] += 0.4
        be.send_position_arm(np.array([0.5, 9.0, -0.5]))    # 累计 0.6s → 再告警
        assert len(cap.msgs) == 2
    finally:
        be_mod.time = orig_time
        log.removeHandler(cap)


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
