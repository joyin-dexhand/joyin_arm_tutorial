"""PinDynamicsSolver 离线深度测试（joyarm_dm 真实 pin 模型，不依赖硬件）。

覆盖：①四内核逐项 pin 对拍（M=crba 对称化 / G=rnea(q,0,0) / C=nle−G /
idyn=rnea，多随机位形）；②M 对称 + 正定；③运动方程自洽闭环（τ=idyn →
fdyn 反推 ddq 闭合）与 fdyn 手算一致；④cartesian_inertia（对称/正定/
双参考系/与 J⁺ᵀMJ⁺ 手算一致、未知 ref 与未知帧拒绝）；⑤f_ext 拒绝；
⑥无跨调用状态泄漏（不同 q 产生不同 M）。

运行：``python test/test_dynamics_solver_pin.py`` 或 pytest。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import PinDynamicsSolver, joyarm_factory  # noqa: E402

_ARM = None


def _arm():
    global _ARM
    if _ARM is None:
        _ARM = joyarm_factory("joyarm_dm")
        assert _ARM is not None, "joyarm_dm 创建失败"
    return _ARM


def test_mass_matrix_vs_crba():
    """M：与 pin.crba 上三角 + 对称化逐位一致；对称 + 特征值全正（正定）。"""
    arm, dyn = _arm(), PinDynamicsSolver()
    rng = np.random.default_rng(5)
    m = arm.pin_model
    for _ in range(3):
        q = arm.rand_q_arm(rng=rng)
        M = dyn.mass_matrix(arm, q)
        Mc = np.array(pin.crba(m, pin.Data(m), q), dtype=float)
        Mc = np.triu(Mc) + np.triu(Mc, 1).T                # 复刻实现的对称化
        assert M.shape == (arm.n_arm, arm.n_arm)
        assert np.allclose(M, Mc, atol=1e-12)
        assert np.allclose(M, M.T, atol=1e-12)             # 对称
        assert np.all(np.linalg.eigvalsh(M) > 0)           # 正定


def test_gravity_vs_rnea():
    """G：= rnea(q, 0, 0)（零速度零加速度特例）；home 位形数值有限。"""
    arm, dyn = _arm(), PinDynamicsSolver()
    rng = np.random.default_rng(6)
    m, d = arm.pin_model, arm.pin_data
    z = np.zeros(arm.n_arm)
    for _ in range(3):
        q = arm.rand_q_arm(rng=rng)
        assert np.allclose(dyn.gravity(arm, q), pin.rnea(m, d, q, z, z),
                           atol=1e-12)
    G_home = dyn.gravity(arm, arm.arm_home)
    assert G_home.shape == (arm.n_arm,) and np.all(np.isfinite(G_home))


def test_coriolis_vs_nle():
    """C 项向量：= nonLinearEffects − G；零速度时恒为 0（无向心/科氏项）。"""
    arm, dyn = _arm(), PinDynamicsSolver()
    rng = np.random.default_rng(7)
    m, d = arm.pin_model, arm.pin_data
    z = np.zeros(arm.n_arm)
    for _ in range(3):
        q = arm.rand_q_arm(rng=rng)
        dq = rng.uniform(-1.0, 1.0, arm.n_arm)
        expect = pin.nonLinearEffects(m, d, q, dq) - pin.rnea(m, d, q, z, z)
        assert np.allclose(dyn.coriolis(arm, q, dq), expect, atol=1e-12)
        assert np.allclose(dyn.coriolis(arm, q, z), z, atol=1e-12)  # dq=0 → 0


def test_idyn_vs_rnea():
    """idyn：与 rnea(q, dq, ddq) 逐位一致；零状态退化 = G。"""
    arm, dyn = _arm(), PinDynamicsSolver()
    rng = np.random.default_rng(8)
    m, d = arm.pin_model, arm.pin_data
    z = np.zeros(arm.n_arm)
    for _ in range(3):
        q = arm.rand_q_arm(rng=rng)
        dq = rng.uniform(-1.0, 1.0, arm.n_arm)
        ddq = rng.uniform(-2.0, 2.0, arm.n_arm)
        tau = dyn.idyn(arm, q, dq, ddq)
        assert tau.shape == (arm.n_arm,)
        assert np.allclose(tau, pin.rnea(m, d, q, dq, ddq), atol=1e-12)
    assert np.allclose(dyn.idyn(arm, arm.arm_home, z, z), dyn.gravity(
        arm, arm.arm_home), atol=1e-12)


def test_equation_consistency_idyn_fdyn():
    """运动方程自洽闭环：τ = idyn(q,dq,ddq) → fdyn(q,dq,τ) 反推 ≈ ddq。"""
    arm, dyn = _arm(), PinDynamicsSolver()
    rng = np.random.default_rng(9)
    for _ in range(3):
        q = arm.rand_q_arm(rng=rng)
        dq = rng.uniform(-1.0, 1.0, arm.n_arm)
        ddq = rng.uniform(-2.0, 2.0, arm.n_arm)
        tau = dyn.idyn(arm, q, dq, ddq)
        ddq_rec = dyn.fdyn(arm, q, dq, tau)
        assert ddq_rec.shape == (arm.n_arm,)
        assert np.allclose(ddq_rec, ddq, atol=1e-9)


def test_fdyn_manual_formula():
    """fdyn：= solve(M, τ − (C+G)) 与手算一致；零力矩 → 自由下落加速度。"""
    arm, dyn = _arm(), PinDynamicsSolver()
    rng = np.random.default_rng(10)
    q = arm.rand_q_arm(rng=rng)
    dq = rng.uniform(-1.0, 1.0, arm.n_arm)
    tau = rng.uniform(-5.0, 5.0, arm.n_arm)
    h = dyn.coriolis(arm, q, dq) + dyn.gravity(arm, q)
    expect = np.linalg.solve(dyn.mass_matrix(arm, q), tau - h)
    assert np.allclose(dyn.fdyn(arm, q, dq, tau), expect, atol=1e-10)
    # 零重力补偿零力矩：home 位形（非零重力项）→ 加速度非零（下坠方向）
    free = dyn.fdyn(arm, arm.arm_home, np.zeros(arm.n_arm),
                    np.zeros(arm.n_arm))
    assert np.linalg.norm(free) > 1e-6


def test_cartesian_inertia_properties_and_manual():
    """Λ：对称、正定、(6,6)；与 J⁺ᵀ M J⁺ 手算一致；base/local 双参考系。"""
    arm, dyn = _arm(), PinDynamicsSolver()
    rng = np.random.default_rng(11)
    m = arm.pin_model
    fid = int(m.getFrameId(arm.ee_frame_name))
    for _ in range(2):
        q = arm.rand_q_arm(rng=rng)
        for ref, rf in (("base", pin.LOCAL_WORLD_ALIGNED),
                        ("local", pin.LOCAL)):
            Lam = dyn.cartesian_inertia(arm, q, arm.ee_frame_name, ref=ref)
            assert Lam.shape == (6, 6)
            assert np.allclose(Lam, Lam.T, atol=1e-9)      # 对称
            assert np.all(np.linalg.eigvalsh(Lam) > 0)     # 正定
            J = pin.computeFrameJacobian(m, pin.Data(m), q, fid, rf)
            Jp = np.linalg.pinv(J)
            expect = Jp.T @ dyn.mass_matrix(arm, q) @ Jp
            assert np.allclose(Lam, expect, atol=1e-9)


def test_cartesian_inertia_rejects_bad_ref_and_frame():
    """Λ：未知 ref / 未知帧名 / 负索引：ValueError。"""
    arm, dyn = _arm(), PinDynamicsSolver()
    q = arm.rand_q_arm(rng=np.random.default_rng(12))
    try:
        dyn.cartesian_inertia(arm, q, arm.ee_frame_name, ref="world")
        raise AssertionError("未知 ref 应抛 ValueError")
    except ValueError as e:
        assert "ref" in str(e)
    for bad in ("no_such_frame", -1):
        try:
            dyn.cartesian_inertia(arm, q, bad)
            raise AssertionError(f"{bad!r} 应抛 ValueError")
        except ValueError as e:
            assert "未找到" in str(e)


def test_f_ext_not_implemented():
    """idyn 外力项：暂不支持（待力控章节），非 None 抛 NotImplementedError。"""
    arm, dyn = _arm(), PinDynamicsSolver()
    q = arm.rand_q_arm(rng=np.random.default_rng(13))
    z = np.zeros(arm.n_arm)
    assert np.all(np.isfinite(dyn.idyn(arm, q, z, z)))     # f_ext=None 正常
    try:
        dyn.idyn(arm, q, z, z, f_ext=np.zeros(6))
        raise AssertionError("f_ext 应抛 NotImplementedError")
    except NotImplementedError as e:
        assert "力控" in str(e)


def test_no_state_leak_between_calls():
    """无跨调用状态泄漏：不同位形产生不同 M/G（每次私有 pin.Data）。"""
    arm, dyn = _arm(), PinDynamicsSolver()
    rng = np.random.default_rng(14)
    q1, q2 = arm.rand_q_arm(rng=rng), arm.rand_q_arm(rng=rng)
    M1a = dyn.mass_matrix(arm, q1)
    dyn.mass_matrix(arm, q2)                               # 中途调用不得污染
    M1b = dyn.mass_matrix(arm, q1)
    assert np.array_equal(M1a, M1b)
    assert not np.allclose(M1a, dyn.mass_matrix(arm, q2), atol=1e-6)
    assert not np.allclose(dyn.gravity(arm, q1), dyn.gravity(arm, q2),
                           atol=1e-6)


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
