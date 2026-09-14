"""PinFkineSolver 离线深度测试（joyarm_dm 真实 pin 模型，不依赖硬件）。

覆盖：①单点/零位/链中部帧位姿 vs pin 直算（位置 1e-12 级、四元数、T 互转）；
②帧名 str 与帧索引 int 等价；③批量模板（(N,n) → list[Pose] / (N,4,4)，
rep="se3" → pin.SE3），批量与单点逐一一致；④输入形态与非法参数拒绝
（q.ndim=3 / 非法 rep / 未知帧名 / 负与越界索引）；⑤两线程并发 vs 串行
（每次计算私有 pin.Data、无共享可变状态的回归）。

运行：``python test/test_fkine_solver_pin.py`` 或 pytest。
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pinocchio as pin

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import Pose, PinFkineSolver, R_to_quat, joyarm_factory  # noqa: E402

_ARM = None


def _arm():
    global _ARM
    if _ARM is None:
        _ARM = joyarm_factory("joyarm_dm")
        assert _ARM is not None, "joyarm_dm 创建失败"
    return _ARM


def _pin_T(arm, q, frame) -> np.ndarray:
    """pin 直算参照：指定帧在基座系下的 4×4 齐次矩阵。"""
    m, d = arm.pin_model, arm.pin_data
    pin.forwardKinematics(m, d, np.asarray(q, dtype=float))
    fid = frame if isinstance(frame, (int, np.integer)) else m.getFrameId(str(frame))
    return pin.updateFramePlacement(m, d, int(fid)).homogeneous


def _quat_close(q1, q2, atol: float = 1e-12) -> bool:
    """四元数比较（q 与 −q 同姿态，符号对齐后比较）。"""
    q1, q2 = np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)
    if np.linalg.norm(q1 - q2) > np.linalg.norm(q1 + q2):
        q2 = -q2
    return bool(np.allclose(q1, q2, atol=atol))


def test_frame_pose_matches_pin():
    """随机位形单点 FK：位置 1e-12 吻合、单位四元数、Pose↔T 互转往返一致。"""
    arm, fk = _arm(), PinFkineSolver()
    rng = np.random.default_rng(7)
    for _ in range(8):
        q = arm.rand_q_arm(rng=rng)
        pose = fk.frame_pose(arm, q, arm.ee_frame_name)
        T = _pin_T(arm, q, arm.ee_frame_name)
        assert np.allclose(pose.position, T[:3, 3], atol=1e-12)
        assert _quat_close(pose.orientation, R_to_quat(T[:3, :3]))
        assert abs(np.linalg.norm(pose.orientation) - 1.0) < 1e-12   # 单位四元数
        assert np.allclose(pose.T, T, atol=1e-12)                    # Pose → T 往返


def test_zero_configuration():
    """零位形（全零关节角）与 pin 直算一致。"""
    arm, fk = _arm(), PinFkineSolver()
    q = np.zeros(arm.n_arm)
    pose = fk.frame_pose(arm, q, arm.ee_frame_name)
    assert np.allclose(pose.T, _pin_T(arm, q, arm.ee_frame_name), atol=1e-12)


def test_mid_chain_frame_and_index_name_equiv():
    """链中部帧同样可解；同帧「帧名 str」与「帧索引 int」结果逐位一致。"""
    arm, fk = _arm(), PinFkineSolver()
    m = arm.pin_model
    fid = int(m.getFrameId(arm.ee_frame_name)) // 2        # 链中部帧索引
    name = m.frames[fid].name
    q = arm.rand_q_arm(rng=np.random.default_rng(1))
    by_name = fk.frame_pose(arm, q, name)
    by_idx = fk.frame_pose(arm, q, fid)
    assert np.allclose(by_name.T, by_idx.T, atol=1e-15)
    assert np.allclose(by_idx.T, _pin_T(arm, q, fid), atol=1e-12)


def test_solve_batch_pose_and_T():
    """批量模板：(N,n) → list[Pose]；rep="T" → (N,4,4)；与单点逐一一致。"""
    arm, fk = _arm(), PinFkineSolver()
    rng = np.random.default_rng(2)
    Q = np.stack([arm.rand_q_arm(rng=rng) for _ in range(5)])
    poses = fk.solve(arm, Q, arm.ee_frame_name)
    assert isinstance(poses, list) and len(poses) == 5
    assert all(isinstance(p, Pose) for p in poses)
    Ts = fk.solve(arm, Q, arm.ee_frame_name, rep="T")
    assert isinstance(Ts, np.ndarray) and Ts.shape == (5, 4, 4)
    for i in range(5):
        single = fk.solve(arm, Q[i], arm.ee_frame_name, rep="T")
        assert np.allclose(Ts[i], single, atol=1e-12)
        assert np.allclose(poses[i].T, single, atol=1e-12)  # 两种表示互证


def test_solve_rep_se3():
    """rep="se3"：单点 pin.SE3、批量 list[pin.SE3]，homogeneous 与直算一致。"""
    arm, fk = _arm(), PinFkineSolver()
    rng = np.random.default_rng(3)
    q = arm.rand_q_arm(rng=rng)
    M = fk.solve(arm, q, arm.ee_frame_name, rep="se3")
    assert isinstance(M, pin.SE3)
    assert np.allclose(M.homogeneous, _pin_T(arm, q, arm.ee_frame_name), atol=1e-12)
    Q = np.stack([arm.rand_q_arm(rng=rng) for _ in range(3)])
    Ms = fk.solve(arm, Q, arm.ee_frame_name, rep="se3")
    assert isinstance(Ms, list) and len(Ms) == 3
    assert all(isinstance(x, pin.SE3) for x in Ms)
    assert np.allclose(Ms[1].homogeneous, _pin_T(arm, Q[1], arm.ee_frame_name),
                       atol=1e-12)


def test_input_shapes():
    """q 取 python list 亦可用；ndim=3 显性拒绝并说明合法维度。"""
    arm, fk = _arm(), PinFkineSolver()
    q = [0.1, -0.2, 0.3, 0.0, 0.1, -0.1]
    pose = fk.solve(arm, q, arm.ee_frame_name)
    assert np.allclose(pose.T, _pin_T(arm, np.asarray(q), arm.ee_frame_name),
                       atol=1e-12)
    try:
        fk.solve(arm, np.zeros((2, 6, 1)), arm.ee_frame_name)
        raise AssertionError("ndim=3 应抛 ValueError")
    except ValueError as e:
        assert "维度" in str(e)


def test_invalid_rep_rejected():
    """非法 rep 显性拒绝并列出可选值（大小写敏感）。"""
    arm, fk = _arm(), PinFkineSolver()
    q = np.zeros(arm.n_arm)
    for bad in ("world", "", "POSE", None):
        try:
            fk.solve(arm, q, arm.ee_frame_name, rep=bad)
            raise AssertionError(f"rep={bad!r} 应抛 ValueError")
        except ValueError as e:
            assert "rep" in str(e) and "pose" in str(e)


def test_unknown_frame_and_bad_index_rejected():
    """未知帧名 / 负索引 / 越界索引：ValueError，消息含可用帧名助定位。"""
    arm, fk = _arm(), PinFkineSolver()
    q = arm.rand_q_arm(rng=np.random.default_rng(4))
    for bad in ("no_such_frame", -1, 10 ** 6):
        try:
            fk.frame_pose(arm, q, bad)
            raise AssertionError(f"{bad!r} 应抛 ValueError")
        except ValueError as e:
            assert "未找到" in str(e)
            assert arm.ee_frame_name in str(e)             # 可用帧名列表


def test_concurrent_thread_safety():
    """并发回归：两线程交替求解与串行结果逐位一致（私有 pin.Data）。"""
    arm, fk = _arm(), PinFkineSolver()
    rng = np.random.default_rng(5)
    Q = np.stack([arm.rand_q_arm(rng=rng) for _ in range(40)])
    expect = np.stack([fk.solve(arm, q, arm.ee_frame_name, rep="T") for q in Q])
    got = np.empty_like(expect)

    def worker(idx):
        for i in idx:
            got[i] = fk.solve(arm, Q[i], arm.ee_frame_name, rep="T")

    halves = list(range(0, 20)), list(range(20, 40))
    threads = [threading.Thread(target=worker, args=(h,)) for h in halves]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert np.allclose(got, expect, atol=1e-12)


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
