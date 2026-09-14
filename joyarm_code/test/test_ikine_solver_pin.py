"""PinIkineSolver 离线深度测试（joyarm_dm 真实 pin 模型，不依赖硬件）。

覆盖：①可达位姿收敛三起点策略——随机起点+充分重启 / 平凡起点（q0=q*，
机器精度）/ 邻近起点（默认配置）；②不可达目标 success=False 不抛、err 大、
q 有限；③iters 耗尽不抛；④构造参数注入（damping/step_max/restarts）与
restarts=0 单起点；⑤重启采样确定性（同输入同结果）；⑥solve_all 默认
NotImplementedError；⑦q0 取 list；⑧基类解析法助手 _shift_2pi /
_select_nearest 单测。

运行：``python test/test_ikine_solver_pin.py`` 或 pytest。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import (  # noqa: E402
    IkineSolver, PinFkineSolver, PinIkineSolver, Pose, joyarm_factory,
)

_ARM = None


def _arm():
    global _ARM
    if _ARM is None:
        _ARM = joyarm_factory("joyarm_dm")
        assert _ARM is not None, "joyarm_dm 创建失败"
    return _ARM


def _quat_close(q1, q2, atol: float) -> bool:
    """四元数比较（q 与 −q 同姿态，符号对齐后比较）。"""
    q1, q2 = np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)
    if np.linalg.norm(q1 - q2) > np.linalg.norm(q1 + q2):
        q2 = -q2
    return bool(np.allclose(q1, q2, atol=atol))


def test_reachable_converges_with_restarts():
    """随机起点 + 充分重启（10）：可达位姿全部收敛，FK 回算闭合、解在硬限位内。

    注：随机起点下 LM 存在局部极小（多起点重启缓解而非消除）——默认
    ``restarts=3`` 对个别难目标可能 ``success=False``（重启起点全部落入同类
    极小），``restarts=20`` 实测可全部救回；见 test_unreachable_no_throw 的
    尽力而为语义。
    """
    arm, fk = _arm(), PinFkineSolver()
    ik = PinIkineSolver(restarts=10)
    rng = np.random.default_rng(11)
    lim = arm.arm_limits
    for _ in range(5):
        q_star = arm.rand_q_arm(rng=rng)                    # 限位内可达位姿
        target = fk.frame_pose(arm, q_star, arm.ee_frame_name)
        q0 = arm.rand_q_arm(rng=rng)                        # 随机起点
        res = ik.solve(arm, target, arm.ee_frame_name, q0=q0, tol=1e-4)
        assert res.success and res.err < 1e-4, (res.success, res.err)
        got = fk.frame_pose(arm, res.q, arm.ee_frame_name)
        assert np.allclose(got.position, target.position, atol=1e-3)
        assert _quat_close(got.orientation, target.orientation, atol=1e-3)
        assert np.all(res.q >= lim.q_min - 1e-9) and np.all(res.q <= lim.q_max + 1e-9)
        assert np.all(np.isfinite(res.q))


def test_from_q_star_exact():
    """平凡起点（q0 = 生成目标的关节角）：收敛到机器精度。"""
    arm, ik, fk = _arm(), PinIkineSolver(), PinFkineSolver()
    rng = np.random.default_rng(18)
    q_star = arm.rand_q_arm(rng=rng)
    target = fk.frame_pose(arm, q_star, arm.ee_frame_name)
    res = ik.solve(arm, target, arm.ee_frame_name, q0=q_star, tol=1e-8)
    assert res.success and res.err < 1e-10
    assert np.allclose(res.q, q_star, atol=1e-6)


def test_nearby_start_default_config():
    """邻近起点（±0.1 rad 扰动）：默认配置（restarts=3）收敛且精度达 tol=1e-5。"""
    arm, ik, fk = _arm(), PinIkineSolver(), PinFkineSolver()
    rng = np.random.default_rng(12)
    for _ in range(5):
        q_star = arm.rand_q_arm(rng=rng)
        target = fk.frame_pose(arm, q_star, arm.ee_frame_name)
        q0 = np.clip(q_star + rng.normal(0.0, 0.1, q_star.size),
                     arm.arm_limits.q_min, arm.arm_limits.q_max)
        res = ik.solve(arm, target, arm.ee_frame_name, q0=q0, tol=1e-5)
        assert res.success and res.err < 1e-5, (res.success, res.err)


def test_unreachable_no_throw():
    """不可达远点（远超臂长）：不抛、success=False、err 大、q 仍有限。"""
    arm, ik = _arm(), PinIkineSolver()
    far = Pose(position=np.array([10.0, 0.5, 0.3]))
    res = ik.solve(arm, far, arm.ee_frame_name,
                   q0=arm.rand_q_arm(rng=np.random.default_rng(13)), iters=100)
    assert (not res.success) and res.err > 1e-2
    assert np.all(np.isfinite(res.q))                       # 尽力解仍可用


def test_iters_exhausted_no_throw():
    """iters=1 极限：难目标下不抛，返回有限值（尽力而为语义）。"""
    arm, ik = _arm(), PinIkineSolver(restarts=0)
    far = Pose(position=np.array([10.0, 0.0, 0.0]))
    res = ik.solve(arm, far, arm.ee_frame_name,
                   q0=arm.rand_q_arm(rng=np.random.default_rng(14)), iters=1)
    assert np.all(np.isfinite(res.q))


def test_ctor_params_injected():
    """构造参数注入：damping/step_max/restarts 自定义可用；restarts=0 单起点。

    （取邻近起点验证参数机制本身；随机起点下的重启鲁棒性见
    test_reachable_converges_with_restarts。）
    """
    arm, fk = _arm(), PinFkineSolver()
    rng = np.random.default_rng(15)
    ik = PinIkineSolver(damping=1e-2, step_max=0.3, restarts=1)
    assert (ik.damping, ik.step_max, ik.restarts) == (1e-2, 0.3, 1)
    q_star = arm.rand_q_arm(rng=rng)
    target = fk.frame_pose(arm, q_star, arm.ee_frame_name)
    q0 = np.clip(q_star + 0.05, arm.arm_limits.q_min, arm.arm_limits.q_max)
    res = ik.solve(arm, target, arm.ee_frame_name, q0=q0, tol=1e-4)
    assert res.success
    ik0 = PinIkineSolver(restarts=0)                        # 不重启（纯单起点）
    res0 = ik0.solve(arm, target, arm.ee_frame_name, q0=q0, tol=1e-4)
    assert res0.success


def test_deterministic():
    """确定性：同输入两次求解结果逐位一致（重启采样固定种子）。"""
    arm, ik, fk = _arm(), PinIkineSolver(), PinFkineSolver()
    rng = np.random.default_rng(16)
    q_star = arm.rand_q_arm(rng=rng)
    target = fk.frame_pose(arm, q_star, arm.ee_frame_name)
    q0 = arm.rand_q_arm(rng=rng)
    r1 = ik.solve(arm, target, arm.ee_frame_name, q0=q0)
    r2 = ik.solve(arm, target, arm.ee_frame_name, q0=q0)
    assert np.array_equal(r1.q, r2.q) and r1.err == r2.err
    assert r1.success == r2.success and r1.n_iter == r2.n_iter


def test_solve_all_not_implemented():
    """数值实现不提供全解：基类默认 NotImplementedError。"""
    arm, ik, fk = _arm(), PinIkineSolver(), PinFkineSolver()
    target = fk.frame_pose(arm, np.zeros(arm.n_arm), arm.ee_frame_name)
    try:
        ik.solve_all(arm, target, arm.ee_frame_name)
        raise AssertionError("solve_all 应抛 NotImplementedError")
    except NotImplementedError:
        pass


def test_q0_list_input():
    """q0 接受 python list（内核内统一 asarray）。"""
    arm, ik, fk = _arm(), PinIkineSolver(), PinFkineSolver()
    rng = np.random.default_rng(17)
    target = fk.frame_pose(arm, arm.rand_q_arm(rng=rng), arm.ee_frame_name)
    res = ik.solve(arm, target, arm.ee_frame_name, q0=[0.0] * arm.n_arm,
                   tol=1e-4)
    assert res.success


# ============================================================
# 基类解析法助手（供 Ch3 解析 IK 子类复用，先于子类落测）
# ============================================================
def test_shift_2pi():
    """±2π 归位：限位内原样；限位外移入；无整数 k 可入界取离区间最近等价角。"""
    q_min = np.full(3, -2.8)
    q_max = np.full(3, 2.8)
    q = np.array([1.0, 1.5 + 2.0 * np.pi, -3.0])
    out = IkineSolver._shift_2pi(q, q_min, q_max)
    assert out[0] == 1.0                                    # 限位内原样保留
    assert abs(out[1] - 1.5) < 1e-12                        # −2π 移入限位
    assert out[2] == -3.0    # 无等价角可入界（−3.0 距区间 0.2，最近）


def test_shift_2pi_narrow_limits():
    """限位区间窄于 2π 时：整体平移后仍在界外，取离区间最近的等价角。"""
    q_min = np.array([0.0])
    q_max = np.array([1.0])                                 # 宽 1 < 2π
    out = IkineSolver._shift_2pi(np.array([5.0]), q_min, q_max)
    # 5.0−2π≈−1.28（距区间 1.28）、5.0（距 4.0）：最优为 −1.28
    assert abs(out[0] - (5.0 - 2.0 * np.pi)) < 1e-12


def test_select_nearest():
    """限位剔除 + q0 最近：越限行剔除、合法行取与 q0 偏差平方和最小者。"""
    q_min, q_max = np.full(2, -1.0), np.full(2, 1.0)
    sols = np.array([[0.5, 0.5],                            # 距 q0 最近
                     [0.9, -0.9],
                     [2.0, 0.0]])                           # 任一关节越限 → 剔除
    res = IkineSolver._select_nearest(sols, q0=np.array([0.4, 0.4]),
                                      q_min=q_min, q_max=q_max)
    assert res.success
    assert np.allclose(res.q, [0.5, 0.5])
    assert res.n_iter == 0 and res.err == 0.0               # 解析法约定


def test_select_nearest_all_infeasible():
    """全部候选越限：success=False、q 为空、err=inf。"""
    res = IkineSolver._select_nearest(np.array([[2.0, 0.0], [-2.0, 0.0]]),
                                      q0=np.zeros(2),
                                      q_min=np.full(2, -1.0), q_max=np.full(2, 1.0))
    assert (not res.success) and res.q.size == 0
    assert res.err == float("inf")


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
