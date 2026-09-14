"""Backend 基类限位守卫离线单测（硬限位裁剪唯一执行点：send_* 模板前置，arm/end 两族）。

覆盖：①限位构建（cfg 自解析：本体/末端逐关节硬限位，缺键量纲 ±∞，无对应
段不启用；只存硬限位、无软限位属性）；②本体守卫
（q 硬限位 / dq、tau 幅值 / kp、kd 透传 / 单关节切片 / 维度不符放行）；
③末端守卫（逐电机裁剪 / 标量广播裁剪（多执行器）/ force 按 ±tau_max 数值
守卫 / MIT 末端）；④越限告警节流；⑤缓存契约默认实现（read_state_cache_* /
state_age_* 未覆写时 NotImplementedError）；⑥指令升维与透传细节（_vec /
列表输入 / MIT kp/kd=None 透传 / 末端标量+单电机 / 末端维度不符放行 /
限内不告警 / 末端缺 q 键 ±∞ 不裁）。

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

from joyarm_core import Backend, ControlMode  # noqa: E402

class _DummyBackend(Backend):
    """离线哑后端：仅记录六个指令内核收到的参数；与指令无关的其余抽象
    方法哑化（空实现满足 ABC 契约；send_position/force/mit_end 已为基类
    模板，不在哑化之列）。"""

    def __init__(self, cfg: dict | None = None):
        Backend.__init__(self, cfg or {})
        self.kernel = []          # [(方法名, 收到的参数元组)]

    connected = True

    # ---- 指令内核：记录收到的参数（数组升维后记录便于断言）----
    def _send_position_arm(self, q, joint=None):
        self.kernel.append(("pos_arm", np.asarray(q, float), joint))

    def _send_velocity_arm(self, dq, joint=None):
        self.kernel.append(("vel_arm", np.asarray(dq, float), joint))

    def _send_position_end(self, position, joint=None):
        self.kernel.append(("pos_end", np.asarray(position, float), joint))

    def _send_tau_end(self, tau, joint=None):
        self.kernel.append(("tau_end", np.asarray(tau, float), joint))

    def _send_mit_arm(self, q, dq, tau_ff, kp=None, kd=None, joint=None):
        self.kernel.append(("mit_arm", np.asarray(q, float), np.asarray(dq, float),
                            np.asarray(tau_ff, float), kp, kd, joint))

    def _send_mit_end(self, q, dq, tau_ff, kp=None, kd=None, joint=None):
        self.kernel.append(("mit_end", np.asarray(q, float), np.asarray(dq, float),
                            np.asarray(tau_ff, float), kp, kd, joint))

    # ---- 其余接口：哑化（空实现）----
    def connect(self):
        pass

    def disconnect(self):
        pass

    def enable_arm(self, joint=None):
        pass

    def disable_arm(self, joint=None):
        pass

    def set_zero_arm(self, joint=None):
        pass

    def clear_fault_arm(self, joint=None):
        pass

    def set_mode_arm(self, mode=ControlMode.POSITION, joint=None):
        pass

    def read_mode_arm(self, joint=None):
        return None

    def read_state_arm(self, joint=None):
        return None

    def read_param_arm(self, key, joint=None):
        return None

    def write_param_arm(self, key, value, joint=None, persist=False):
        pass

    def enable_end(self, joint=None):
        pass

    def disable_end(self, joint=None):
        pass

    def set_zero_end(self, joint=None):
        pass

    def clear_fault_end(self, joint=None):
        pass

    def set_mode_end(self, mode=ControlMode.POSITION, joint=None):
        pass

    def read_mode_end(self, joint=None):
        return None

    def read_state_end(self, joint=None):
        return None

    def send_action_end(self, action, joint=None):
        pass

    def read_param_end(self, key, joint=None):
        return None

    def write_param_end(self, key, value, joint=None, persist=False):
        pass


# 本体三关节硬限位（cfg 四键）：q ∈ [-1,1]，dq ≤ 2，tau ≤ 10
_ARM_CFG = {"arm": {"joints": [
    {"name": f"j{i + 1}", "q_min": -1.0, "q_max": 1.0, "dq_max": 2.0, "tau_max": 10.0}
    for i in range(3)
]}}


# 双电机末端：电机 2 缺 tau_max（→ ∞ 不参与守卫）、行程更紧
_END_CFG = {"end": {"joints": [
    {"name": "g1", "q_min": -1.0, "q_max": 1.0, "dq_max": 2.0, "tau_max": 10.0},
    {"name": "g2", "q_min": -0.5, "q_max": 0.5, "dq_max": 3.0},
]}}


# ----------------------------------------------------------
# 限位构建（cfg 自解析：arm/end 同构）
# ----------------------------------------------------------
def test_arm_limits_from_cfg():
    """本体硬限位自 cfg 逐关节解析；backend 只存硬限位（无软限位属性）。"""
    be = _DummyBackend(cfg=_ARM_CFG)
    s = be.arm_limits
    assert np.allclose(s.q_min, -1.0) and np.allclose(s.q_max, 1.0)
    assert np.allclose(s.dq_max, 2.0) and np.allclose(s.tau_max, 10.0)
    be0 = _DummyBackend(cfg={"end": {}})
    assert be0.arm_limits is None                      # 无 arm 段 → 守卫放行
    for gone in ("arm_limits_soft", "end_limits_soft",
                 "set_arm_limits_soft", "set_end_limits_soft"):
        assert not hasattr(be, gone), f"{gone} 应已随软限位机制移除"


def test_end_limits_from_cfg():
    """末端硬限位自 cfg 逐电机解析（缺键 ±∞）；无 end 段 → None。"""
    be = _DummyBackend(cfg=_END_CFG)
    e = be.end_limits
    assert np.allclose(e.q_min, [-1.0, -0.5]) and np.allclose(e.q_max, [1.0, 0.5])
    assert np.allclose(e.dq_max, [2.0, 3.0])
    assert np.isfinite(e.tau_max[0]) and np.isinf(e.tau_max[1])
    be2 = _DummyBackend(cfg={"arm": {}})
    assert be2.end_limits is None


# ----------------------------------------------------------
# 本体守卫（模板前置）
# ----------------------------------------------------------
def test_guard_passthrough_without_limits():
    """cfg 无 arm 段（独立使用无本体限位）：本体守卫放行。"""
    be = _DummyBackend()
    assert be.arm_limits is None
    be.send_position_arm(np.full(3, 99.0))
    assert np.allclose(be.kernel[0][1], 99.0)


def test_guard_clips_q_dq_tau():
    """本体：q 硬限位裁剪、dq/tau 幅值裁剪、kp/kd 透传。"""
    be = _DummyBackend(cfg=_ARM_CFG)
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
    be = _DummyBackend(cfg=_ARM_CFG)
    be._arm_limits.q_max[1] = 0.5                       # joint2 上限更紧
    be.send_position_arm(np.array([99.0]), joint=1)
    assert be.kernel[-1][0] == "pos_arm" and np.allclose(be.kernel[-1][1], 0.5)
    assert be.kernel[-1][2] == 1


def test_guard_dimension_mismatch_passthrough():
    """指令长度 ≠ n：守卫放行，由内核（真实后端）报清晰的维度错误。"""
    be = _DummyBackend(cfg=_ARM_CFG)
    be.send_position_arm(np.zeros(4))
    assert be.kernel[-1][1].shape == (4,)
    be2 = _DummyBackend(cfg=_END_CFG)
    be2.send_position_end(np.zeros(3))                    # 末端同理
    assert be2.kernel[-1][1].shape == (3,)


def test_guard_scalar_promotion():
    """标量指令升维为 (1,)（单关节用法），守卫按长度放行交内核校验。"""
    be = _DummyBackend(cfg=_ARM_CFG)
    be.send_position_arm(1.5, joint=0)
    assert be.kernel[-1][1].shape == (1,) and np.allclose(be.kernel[-1][1], 1.0)


# ----------------------------------------------------------
# 末端守卫（逐电机 + 标量广播）
# ----------------------------------------------------------
def test_end_guard_clips_position():
    """末端位置：逐电机裁剪到各自行程。"""
    be = _DummyBackend(cfg=_END_CFG)
    be.send_position_end(np.array([5.0, -5.0]))
    assert be.kernel[-1][0] == "pos_end"
    assert np.allclose(be.kernel[-1][1], [1.0, -0.5])


def test_end_guard_scalar_broadcast():
    """末端标量广播（多执行器）：按逐电机限位各自就近裁剪 → (n_end,)。"""
    be = _DummyBackend(cfg=_END_CFG)
    be.send_position_end(99.0)                             # g1→1.0，g2→0.5
    assert be.kernel[-1][1].shape == (2,)
    assert np.allclose(be.kernel[-1][1], [1.0, 0.5])


def test_end_guard_single_motor_slice():
    """末端单电机指令：取该电机限位切片。"""
    be = _DummyBackend(cfg=_END_CFG)
    be.send_position_end(np.array([9.0]), joint=1)
    assert be.kernel[-1][0] == "pos_end" and np.allclose(be.kernel[-1][1], 0.5)
    assert be.kernel[-1][2] == 1


def test_end_guard_tau_clip():
    """末端力矩：tau 按逐电机 ±tau_max 裁剪（缺 tau_max 不裁）。"""
    be = _DummyBackend(cfg=_END_CFG)
    be.send_tau_end(50.0)                                  # 标量广播：g1→10，g2→∞不裁
    assert be.kernel[-1][0] == "tau_end"
    assert np.allclose(be.kernel[-1][1], [10.0, 50.0])


def test_end_guard_mit():
    """末端 MIT：q/dq/tau 守卫，kp/kd 透传。"""
    be = _DummyBackend(cfg=_END_CFG)
    be.send_mit_end(np.array([5.0, -5.0]), np.array([9.0, -9.0]),
                    np.array([50.0, -50.0]), kp=np.ones(2), kd=np.ones(2))
    _, q, dq, tau, kp, kd, _ = be.kernel[-1]
    assert np.allclose(q, [1.0, -0.5])
    assert np.allclose(dq, [2.0, -3.0])
    assert np.allclose(tau, [10.0, -50.0])                 # g2 tau_max=∞ 不裁
    assert np.allclose(kp, 1.0) and np.allclose(kd, 1.0)


def test_end_guard_without_end_section():
    """无 end 段：末端守卫放行（透传内核）。"""
    be = _DummyBackend(cfg=_ARM_CFG)
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
    be = _DummyBackend(cfg=_ARM_CFG)
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
# 缓存契约默认实现（未覆写后端显性 NotImplementedError）
# ----------------------------------------------------------
def test_cache_contract_defaults():
    """read_state_cache_* / state_age_*：未覆写的后端抛 NotImplementedError
    （消息提示回退 read_state_* 请求-应答路径）。"""
    be = _DummyBackend(cfg=_ARM_CFG)
    for call in (lambda: be.read_state_cache_arm(),
                 lambda: be.read_state_cache_arm(0),
                 lambda: be.read_state_cache_end(),
                 lambda: be.state_age_arm(),
                 lambda: be.state_age_end()):
        try:
            call()
            raise AssertionError("应抛 NotImplementedError")
        except NotImplementedError as e:
            assert "未实现" in str(e)


# ----------------------------------------------------------
# 指令升维与透传细节
# ----------------------------------------------------------
def test_vec_promotion():
    """_vec 升维：None 透传、标量 → (1,)、列表 → (k,)、统一 float dtype。"""
    assert Backend._vec(None) is None
    v = Backend._vec(1.5)
    assert v.shape == (1,) and v.dtype == np.float64 and v[0] == 1.5
    v = Backend._vec([1, 2, 3])
    assert v.shape == (3,) and v.dtype == np.float64
    assert np.allclose(v, [1.0, 2.0, 3.0])


def test_list_input_promotion():
    """列表输入指令：升维后正常裁剪（越限元素就近裁到硬限位）。"""
    be = _DummyBackend(cfg=_ARM_CFG)
    be.send_position_arm([5.0, 0.0, -0.5])
    assert np.allclose(be.kernel[-1][1], [1.0, 0.0, -0.5])


def test_mit_kp_kd_none_passthrough():
    """MIT kp/kd=None 原样透传内核（增益回退 config 由子类内核负责，守卫不动）。"""
    be = _DummyBackend(cfg=_ARM_CFG)
    be.send_mit_arm(np.zeros(3), np.zeros(3), np.ones(3))   # kp/kd 缺省 None
    _, _, _, _, kp, kd, _ = be.kernel[-1]
    assert kp is None and kd is None


def test_end_guard_scalar_with_single_joint():
    """末端标量 + 单电机索引：取该电机限位切片裁剪（不触发广播分支）。"""
    be = _DummyBackend(cfg=_END_CFG)
    be.send_position_end(99.0, joint=1)                # g2 上限 0.5
    assert be.kernel[-1][0] == "pos_end"
    assert be.kernel[-1][1].shape == (1,)
    assert np.allclose(be.kernel[-1][1], 0.5)


def test_end_tau_dimension_mismatch_passthrough():
    """末端 tau 长度 ≠ n_end：守卫放行，由内核校验维度。"""
    be = _DummyBackend(cfg=_END_CFG)
    be.send_tau_end(np.zeros(3))
    assert be.kernel[-1][1].shape == (3,)


def test_in_range_no_warning_and_missing_q_keys_inf():
    """限内指令不告警；末端缺 q_min/q_max 键 → ±∞ 边界不裁（对称于缺 tau_max）。"""
    cap = _Cap()
    log = logging.getLogger("joyarm_core.backend")
    log.addHandler(cap)
    try:
        cfg = {"end": {"joints": [
            {"name": "g1", "q_min": -1.0, "q_max": 1.0,
             "dq_max": 2.0, "tau_max": 10.0},
            {"name": "g2", "dq_max": 3.0},             # 缺 q/tau 键 → ±∞
        ]}}
        be = _DummyBackend(cfg=cfg)
        be.send_position_arm(np.array([0.5, -0.5, 0.0]))   # 限内：不告警
        assert cap.msgs == []
        be.send_position_end(np.array([5.0, 5.0]))         # 仅 g1 越限告警
        assert len(cap.msgs) == 1 and "end" in cap.msgs[0]
        assert np.allclose(be.kernel[-1][1], [1.0, 5.0])   # g2 不裁
    finally:
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
