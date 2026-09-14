"""PinJacobianSolver 离线深度测试（joyarm_dm 真实 pin 模型，不依赖硬件）。

覆盖：①jac 内核 base/local 双参考系 vs pin 直算、形状、帧名/帧索引等价、
链中部帧；②非法 ref / 未知帧 / 负索引拒绝；③基类 9 个派生量逐一——
fkine_vel / ikine_vel 往返 / statics / singular_values / is_singular /
manipulability（=Πσ 且 =sqrt(det(JJᵀ))）/ manipulability_gradient /
damped_pinv / nullspace_projector；④哑 J 矩阵语义（σ=0 无 NaN、冗余臂
零空间投影）。

运行：``python test/test_jacobian_solver_pin.py`` 或 pytest。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import (  # noqa: E402
    JacobianSolver, PinJacobianSolver, joyarm_factory,
)

_ARM = None


def _arm():
    global _ARM
    if _ARM is None:
        _ARM = joyarm_factory("joyarm_dm")
        assert _ARM is not None, "joyarm_dm 创建失败"
    return _ARM


def test_jac_base_local_vs_pin():
    """jac 内核：base（LWA）/ local（LOCAL）双参考系与 pin 直算逐位一致。"""
    arm, js = _arm(), PinJacobianSolver()
    rng = np.random.default_rng(3)
    m, d = arm.pin_model, arm.pin_data
    fid = int(m.getFrameId(arm.ee_frame_name))
    for _ in range(5):
        q = arm.rand_q_arm(rng=rng)
        for ref, rf in (("base", pin.LOCAL_WORLD_ALIGNED), ("local", pin.LOCAL)):
            J = js.jac(arm, q, arm.ee_frame_name, ref=ref)
            assert J.shape == (6, arm.n_arm)
            assert np.allclose(J, pin.computeFrameJacobian(m, d, q, fid, rf),
                               atol=1e-12)


def test_frame_index_name_equiv_mid_chain():
    """帧名 str 与帧索引 int 等价；链中部帧同样可解。"""
    arm, js = _arm(), PinJacobianSolver()
    m = arm.pin_model
    fid = int(m.getFrameId(arm.ee_frame_name)) // 2
    q = arm.rand_q_arm(rng=np.random.default_rng(1))
    J_name = js.jac(arm, q, m.frames[fid].name)
    J_idx = js.jac(arm, q, fid)
    assert np.allclose(J_name, J_idx, atol=1e-15)


def test_invalid_ref_and_frame_rejected():
    """未知 ref / 未知帧名 / 负索引：ValueError。"""
    arm, js = _arm(), PinJacobianSolver()
    q = arm.rand_q_arm(rng=np.random.default_rng(2))
    for ref in ("world", "", "BASE"):
        try:
            js.jac(arm, q, arm.ee_frame_name, ref=ref)
            raise AssertionError(f"ref={ref!r} 应抛 ValueError")
        except ValueError as e:
            assert "ref" in str(e)
    for bad in ("no_such_frame", -1, 10 ** 6):
        try:
            js.jac(arm, q, bad)
            raise AssertionError(f"{bad!r} 应抛 ValueError")
        except ValueError as e:
            assert "未找到" in str(e)


def test_fkine_vel_matches_jacobian_product():
    """速度正解：V = J·q̇，与手算矩阵-向量积一致（双参考系）。"""
    arm, js = _arm(), PinJacobianSolver()
    rng = np.random.default_rng(4)
    q = arm.rand_q_arm(rng=rng)
    dq = rng.uniform(-0.5, 0.5, arm.n_arm)
    for ref in ("base", "local"):
        V = js.fkine_vel(arm, q, dq, arm.ee_frame_name, ref=ref)
        assert V.shape == (6,)
        assert np.allclose(V, js.jac(arm, q, arm.ee_frame_name, ref=ref) @ dq,
                           atol=1e-12)


def test_ikine_vel_roundtrip():
    """微分逆解往返：良态位形（σ_min 较大）dq → V → dq* ≈ dq。

    近奇异位形阻尼伪逆会**有意收缩**最弱方向（DLS 设计行为，误差 ~λ²/σ³），
    不做紧断言；仅要求结果有限（见 test_damped_pinv_singular_no_nan）。
    """
    arm, js = _arm(), PinJacobianSolver()
    rng = np.random.default_rng(5)
    q, s = None, 0.0
    for _ in range(200):                                    # 搜索良态位形 σ_min>0.05
        cand = arm.rand_q_arm(rng=rng)
        s = js.singular_values(arm, cand, arm.ee_frame_name)[-1]
        if s > 0.05:
            q = cand
            break
    assert q is not None, "限位内未搜到 σ_min>0.05 的良态位形"
    dq = rng.uniform(-0.5, 0.5, arm.n_arm)
    V = js.fkine_vel(arm, q, dq, arm.ee_frame_name)
    dq2 = js.ikine_vel(arm, q, V, arm.ee_frame_name)
    assert dq2.shape == (arm.n_arm,)
    assert np.allclose(dq2, dq, atol=1e-3)
    # 近奇异位形：往返偏差大但结果有限（阻尼防 1/σ 爆炸）
    qs = arm.rand_q_arm(rng=rng)
    Vs = js.fkine_vel(arm, qs, dq, arm.ee_frame_name)
    dqs = js.ikine_vel(arm, qs, Vs, arm.ee_frame_name)
    assert np.all(np.isfinite(dqs))


def test_statics_matches_transpose():
    """静力学：τ = JᵀF 与手算一致（双参考系）。"""
    arm, js = _arm(), PinJacobianSolver()
    rng = np.random.default_rng(6)
    q = arm.rand_q_arm(rng=rng)
    F = rng.uniform(-10.0, 10.0, 6)                         # 六维力旋量
    for ref in ("base", "local"):
        tau = js.statics(arm, q, F, arm.ee_frame_name, ref=ref)
        assert tau.shape == (arm.n_arm,)
        assert np.allclose(tau,
                           js.jac(arm, q, arm.ee_frame_name, ref=ref).T @ F,
                           atol=1e-10)


def test_singular_values_descending():
    """奇异值谱：降序，与 np.linalg.svd(J) 一致。"""
    arm, js = _arm(), PinJacobianSolver()
    rng = np.random.default_rng(7)
    for _ in range(3):
        q = arm.rand_q_arm(rng=rng)
        sv = js.singular_values(arm, q, arm.ee_frame_name)
        expect = np.linalg.svd(js.jac(arm, q, arm.ee_frame_name),
                               compute_uv=False)
        assert sv.shape == (min(6, arm.n_arm),)
        assert np.all(np.diff(sv) <= 1e-15)                 # 降序
        assert np.allclose(sv, expect, atol=1e-12)


def test_is_singular_semantics():
    """奇异性判别 σ_min < tol：近奇异位形（数值搜索）两侧验证。"""
    arm, js = _arm(), PinJacobianSolver()
    rng = np.random.default_rng(8)
    # 限位内随机搜索 σ_min 最小的位形（近似奇异）
    best_q, best_s = None, np.inf
    for _ in range(300):
        q = arm.rand_q_arm(rng=rng)
        s = js.singular_values(arm, q, arm.ee_frame_name)[-1]
        if s < best_s:
            best_q, best_s = q, s
    assert js.is_singular(arm, best_q, arm.ee_frame_name,
                          tol=best_s * 1.001 + 1e-12)       # tol 略高于 σ_min
    assert not js.is_singular(arm, best_q, arm.ee_frame_name,
                              tol=best_s * 0.999)           # tol 略低于 σ_min
    # 常规位形（σ_min 远大于默认容差）：非奇异
    q = arm.rand_q_arm(rng=rng)
    s = js.singular_values(arm, q, arm.ee_frame_name)[-1]
    assert not js.is_singular(arm, q, arm.ee_frame_name)    # tol=1e-6
    assert s > 1e-3                                         # 该位形确属常规


def test_manipulability_product_and_det():
    """可操作度：= Πσᵢ；n=6 非奇异处 = sqrt(det(JJᵀ))（docstring 契约）。"""
    arm, js = _arm(), PinJacobianSolver()
    rng = np.random.default_rng(9)
    for _ in range(3):
        q = arm.rand_q_arm(rng=rng)
        J = js.jac(arm, q, arm.ee_frame_name)
        w = js.manipulability(arm, q, arm.ee_frame_name)
        sv = np.linalg.svd(J, compute_uv=False)
        assert np.isclose(w, float(np.prod(sv)), rtol=1e-12)
        assert np.isclose(w, float(np.sqrt(np.linalg.det(J @ J.T))), rtol=1e-6)
        assert w > 0


def test_manipulability_gradient_central_diff():
    """可操作度梯度：与手动中心差分（不同步长）数值一致。"""
    arm, js = _arm(), PinJacobianSolver()
    q = arm.rand_q_arm(rng=np.random.default_rng(10))
    grad = js.manipulability_gradient(arm, q, arm.ee_frame_name)
    assert grad.shape == (arm.n_arm,)
    eps = 1e-4                                              # 与实现的 1e-6 不同步长
    manual = np.zeros_like(q)
    for i in range(q.size):
        qp, qm = q.copy(), q.copy()
        qp[i] += eps
        qm[i] -= eps
        manual[i] = (js.manipulability(arm, qp, arm.ee_frame_name)
                     - js.manipulability(arm, qm, arm.ee_frame_name)) / (2 * eps)
    assert np.allclose(grad, manual, rtol=1e-3, atol=1e-8)


def test_damped_pinv_limits_and_reconstruction():
    """DLS 广义逆：λ→0 退化为 pinv；J·J*·V ≈ V（非奇异全空间重构）。"""
    arm, js = _arm(), PinJacobianSolver()
    rng = np.random.default_rng(11)
    q = arm.rand_q_arm(rng=rng)
    J = js.jac(arm, q, arm.ee_frame_name)
    Jp = js.damped_pinv(arm, q, arm.ee_frame_name, damping=1e-12)
    assert Jp.shape == (arm.n_arm, 6)
    assert np.allclose(Jp, np.linalg.pinv(J), atol=1e-8)
    V = rng.uniform(-1.0, 1.0, 6)
    assert np.allclose(J @ (Jp @ V), V, atol=1e-6)          # 非奇异：可全重构


class _DummyJac(JacobianSolver):
    """固定 J 的哑实现（绕开真实模型，测基类派生量的纯矩阵语义）。"""

    def __init__(self, J):
        self.J = np.asarray(J, dtype=float)

    def jac(self, arm, q, frame, ref="base"):
        return self.J


def test_damped_pinv_singular_no_nan():
    """σ=0 且 λ=0：该方向系数取 0，无 NaN，与截断 pinv 等价。"""
    Js = np.diag([1.0, 2.0, 0.0, 0.5, 0.3, 0.1])            # 含两个零奇异值
    solver = _DummyJac(Js)
    out = solver.damped_pinv(None, None, None, damping=0.0)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, np.linalg.pinv(Js), atol=1e-12)


def test_nullspace_projector_semantics():
    """零空间投影：真实臂（n=6 满秩）N≈0；哑宽 J（6×8 冗余）投影到零空间。"""
    arm, js = _arm(), PinJacobianSolver()
    rng = np.random.default_rng(12)
    q = arm.rand_q_arm(rng=rng)
    N = js.nullspace_projector(arm, q, arm.ee_frame_name, damping=1e-9)
    assert N.shape == (arm.n_arm, arm.n_arm)
    assert np.max(np.abs(N)) < 1e-6                          # 非冗余臂：N ≈ 0
    # 冗余臂：J = [I₆ | 0]（6×8），零空间 = 后 2 维
    dummy = _DummyJac(np.hstack([np.eye(6), np.zeros((6, 2))]))
    N8 = dummy.nullspace_projector(None, None, None, damping=1e-9)
    assert N8.shape == (8, 8)
    assert np.max(np.abs(dummy.J @ N8)) < 1e-8               # J·N ≈ 0
    assert np.allclose(N8[:6, :6], np.zeros((6, 6)), atol=1e-8)   # 任务空间零投影
    assert np.allclose(N8[6:, 6:], np.eye(2), atol=1e-8)     # 零空间全保留
    assert np.isclose(np.trace(N8), 2.0, atol=1e-8)          # 零空间维数 = 2


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
