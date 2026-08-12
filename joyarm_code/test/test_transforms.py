"""``joyarm.transforms`` 的纯 numpy 单元测试。

覆盖全部 23 个公共函数，并包含三条 bug 回归用例：

- Bug 1: ``R_to_rpy`` 万向锁符号错误（pitch=+π/2）
- Bug 2: ``R_to_axis_angle`` θ≈π 时无法恢复混合符号轴
- Bug 3: ``quat_to_axis_angle`` 对 w<0 返回 θ>π
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from joyarm_code.joyarm.utils import transforms as T


# ============================================================
# 辅助：判断一个矩阵是否为合法旋转矩阵
# ============================================================
def _assert_rotation_matrix(R, atol=1e-9):
    """断言 R 是 (3,3) 正交、行列式为 +1 的旋转矩阵。"""
    assert R.shape == (3, 3)
    assert_allclose(R @ R.T, np.eye(3), atol=atol)  # 正交
    assert np.linalg.det(R) == pytest.approx(1.0, abs=atol)


# ============================================================
# 基本旋转矩阵 rot_x / rot_y / rot_z
# ============================================================
class TestBasicRotations:
    @pytest.mark.parametrize("fn", [T.rot_x, T.rot_y, T.rot_z])
    def test_zero_angle_is_identity(self, fn):
        assert_allclose(fn(0.0), np.eye(3), atol=1e-12)

    @pytest.mark.parametrize("fn", [T.rot_x, T.rot_y, T.rot_z])
    def test_is_rotation_matrix(self, fn):
        _assert_rotation_matrix(fn(0.7))

    @pytest.mark.parametrize("fn", [T.rot_x, T.rot_y, T.rot_z])
    def test_inverse_is_transpose(self, fn):
        a = 1.234
        assert_allclose(fn(a) @ fn(-a), np.eye(3), atol=1e-12)

    def test_rot_x_hand_computed(self):
        # 绕 X 转 90°：Y→Z
        R = T.rot_x(np.pi / 2)
        expected = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float
        )
        assert_allclose(R, expected, atol=1e-12)

    def test_rot_y_hand_computed(self):
        # 绕 Y 转 90°：Z→X
        R = T.rot_y(np.pi / 2)
        expected = np.array(
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float
        )
        assert_allclose(R, expected, atol=1e-12)

    def test_rot_z_hand_computed(self):
        # 绕 Z 转 90°：X→Y
        R = T.rot_z(np.pi / 2)
        expected = np.array(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float
        )
        assert_allclose(R, expected, atol=1e-12)


# ============================================================
# RPY ↔ 旋转矩阵
# ============================================================
class TestRpy:
    def test_rpy_to_R_convention(self):
        """约定 R = Rz(y)·Ry(p)·Rx(r)。"""
        r, p, y = 0.3, 0.4, 0.5
        R_expected = T.rot_z(y) @ T.rot_y(p) @ T.rot_x(r)
        assert_allclose(T.rpy_to_R([r, p, y]), R_expected, atol=1e-12)

    def test_rpy_to_R_zero(self):
        assert_allclose(T.rpy_to_R([0, 0, 0]), np.eye(3), atol=1e-12)

    def test_round_trip_random(self, random_rpy):
        R = T.rpy_to_R(random_rpy)
        rpy_back = T.R_to_rpy(R)
        assert_allclose(T.rpy_to_R(rpy_back), R, atol=1e-9)

    def test_pitch_zero(self):
        # pitch=0 时 roll/yaw 各自独立
        rpy = [0.5, 0.0, -0.3]
        R = T.rpy_to_R(rpy)
        assert_allclose(T.R_to_rpy(R), rpy, atol=1e-9)

    # ---- Bug 1 回归：万向锁符号 ----
    @pytest.mark.parametrize(
        "pitch", [np.pi / 2, -np.pi / 2], ids=["pitch=+pi/2", "pitch=-pi/2"]
    )
    def test_R_to_rpy_gimbal_lock(self, pitch):
        """pitch=±90° 进入万向锁；约定 y=0，必须仍能精确重建 R。"""
        for r0, y0 in [(0.2, 0.5), (-0.3, 0.4), (0.7, -0.6), (0.0, 0.0)]:
            R = T.rpy_to_R([r0, pitch, y0])
            r, p, y = T.R_to_rpy(R)
            # 约定：y=0
            assert y == pytest.approx(0.0, abs=1e-12)
            # 关键断言：重建后必须回到原 R
            R_recon = T.rpy_to_R([r, p, y])
            assert_allclose(R_recon, R, atol=1e-9, err_msg=f"failed at ({r0},{pitch},{y0})")


# ============================================================
# 轴角 ↔ 旋转矩阵
# ============================================================
class TestAxisAngle:
    def test_rodrigues_axis_known(self):
        # 绕 Z 转 90° == rot_z(90°)
        R = T.rodrigues([0, 0, 1], np.pi / 2)
        assert_allclose(R, T.rot_z(np.pi / 2), atol=1e-12)

    def test_axis_angle_to_R_normalizes_axis(self):
        # 非单位轴应与单位轴结果一致
        k = np.array([2.0, 0.0, 0.0])
        assert_allclose(T.axis_angle_to_R(k, 0.5), T.rot_x(0.5), atol=1e-12)

    def test_rodrigues_rotation_vector_mode(self):
        """theta=None 时 k 视为旋转向量（方向为轴、模长为角）。"""
        k = np.array([0.0, 0.0, np.pi / 2])  # 模长=π/2，轴=Z
        assert_allclose(T.rodrigues(k), T.rot_z(np.pi / 2), atol=1e-12)

    def test_rodrigues_zero_angle_identity(self):
        assert_allclose(T.rodrigues([1, 0, 0], 0.0), np.eye(3), atol=1e-12)

    def test_rodrigues_zero_vector_identity(self):
        # 零旋转向量 → 单位阵
        assert_allclose(T.rodrigues([0, 0, 0]), np.eye(3), atol=1e-12)

    def test_R_to_axis_angle_zero(self):
        k, th = T.R_to_axis_angle(np.eye(3))
        assert th == pytest.approx(0.0, abs=1e-12)

    def test_round_trip_random(self, random_R):
        k, th = T.R_to_axis_angle(random_R)
        R_back = T.rodrigues(k, th)
        assert_allclose(R_back, random_R, atol=1e-9)
        # 轴应为单位向量
        assert np.linalg.norm(k) == pytest.approx(1.0, abs=1e-9)

    def test_round_trip_general_angle(self, rng):
        """从轴角出发，经 R→轴角 应回到同一个轴角（一般情况）。"""
        for _ in range(20):
            k = rng.normal(size=3)
            k = k / np.linalg.norm(k)
            theta = rng.uniform(0.1, np.pi - 0.1)  # 避开 0 和 π
            R = T.rodrigues(k, theta)
            k2, theta2 = T.R_to_axis_angle(R)
            R2 = T.rodrigues(k2, theta2)
            assert_allclose(R2, R, atol=1e-9)

    # ---- Bug 2 回归：θ≈π 混合符号轴 ----
    @pytest.mark.parametrize(
        "axis",
        [
            [1, -1, 1],
            [1, 1, -1],
            [-1, 1, 1],
            [1, -2, 3],
            [2, 3, -1],
            [-3, 1, 2],
        ],
    )
    def test_R_to_axis_angle_near_pi_mixed_sign(self, axis):
        """θ≈π 且轴含混合符号时，恢复出的轴必须能重建原 R。

        Bug 2: 原实现只试整体 ±1 两个全局符号，对 (1,-1,1) 之类的轴恢复失败。
        """
        k0 = np.array(axis, dtype=float)
        k0 = k0 / np.linalg.norm(k0)
        R = T.rodrigues(k0, np.pi - 1e-9)
        k, th = T.R_to_axis_angle(R)
        # 关键：恢复出的 (k, θ) 必须能重建 R（不要求 k 与 k0 同号，θ=π 时 ±k 等价）
        R_recon = T.rodrigues(k, th)
        assert_allclose(R_recon, R, atol=1e-6, err_msg=f"failed at axis={axis}")


# ============================================================
# 四元数
# ============================================================
class TestQuaternion:
    def test_quat_to_R_identity(self):
        R = T.quat_to_R([1, 0, 0, 0])
        assert_allclose(R, np.eye(3), atol=1e-12)

    def test_R_to_quat_identity(self):
        q = T.R_to_quat(np.eye(3))
        assert_allclose(q, [1, 0, 0, 0], atol=1e-12)

    def test_quat_to_R_normalizes(self):
        # 非单位四元数应与归一化后一致
        q = np.array([2.0, 0.0, 0.0, 0.0])
        assert_allclose(T.quat_to_R(q), np.eye(3), atol=1e-12)

    def test_quat_R_round_trip(self, random_R):
        q = T.R_to_quat(random_R)
        # 单位四元数
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)
        R_back = T.quat_to_R(q)
        assert_allclose(R_back, random_R, atol=1e-9)

    def test_R_to_quat_shepperd_branches(self):
        """显式覆盖 Shepperd 四分支：迹>0 / R00 最大 / R11 最大 / R22 最大。"""
        # 旋转使迹最大（小角度旋转）
        Rs = [
            T.rot_x(0.1),                       # trace > 0 分支
            T.rot_x(np.pi - 0.1),               # 大角度，落到 R00/R11/R22 最大分支之一
            T.rot_y(np.pi - 0.1),
            T.rot_z(np.pi - 0.1),
        ]
        for R in Rs:
            q = T.R_to_quat(R)
            assert_allclose(T.quat_to_R(q), R, atol=1e-9)

    def test_quat_norm_unit_unchanged(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        assert_allclose(T.quat_norm(q), q, atol=1e-12)

    def test_quat_norm_nonunit(self):
        q = np.array([3.0, 0.0, 0.0, 0.0])
        assert_allclose(T.quat_norm(q), [1, 0, 0, 0], atol=1e-12)

    def test_quat_norm_zero_returns_identity(self):
        assert_allclose(T.quat_norm([0, 0, 0, 0]), [1, 0, 0, 0], atol=1e-12)

    def test_quat_mul_identity(self):
        q = np.array([0.7, 0.1, 0.2, 0.3])
        q = q / np.linalg.norm(q)
        assert_allclose(T.quat_mul(q, [1, 0, 0, 0]), q, atol=1e-12)
        assert_allclose(T.quat_mul([1, 0, 0, 0], q), q, atol=1e-12)

    def test_quat_mul_matches_R_composition(self, random_R, rng):
        """quat_mul(q1,q2) 的旋转 == R1 @ R2。"""
        v = rng.normal(size=3)
        k2 = v / np.linalg.norm(v)
        th2 = rng.uniform(0.1, 3.0)
        R2 = T.rodrigues(k2, th2)
        q1 = T.R_to_quat(random_R)
        q2 = T.R_to_quat(R2)
        q_mul = T.quat_mul(q1, q2)
        R_mul = T.quat_to_R(q_mul)
        assert_allclose(R_mul, random_R @ R2, atol=1e-9)

    def test_quat_mul_noncommutative(self):
        q1 = T.R_to_quat(T.rot_x(0.5))
        q2 = T.R_to_quat(T.rot_y(0.7))
        # 一般情况下顺序不同结果不同
        assert not np.allclose(T.quat_mul(q1, q2), T.quat_mul(q2, q1))

    def test_quat_conj_inverse(self):
        q = np.array([0.6, 0.2, 0.4, 0.65])
        q = q / np.linalg.norm(q)
        prod = T.quat_mul(q, T.quat_conj(q))
        assert_allclose(prod, [1, 0, 0, 0], atol=1e-9)

    # ---- Bug 3 回归：quat_to_axis_angle 对 w<0 ----
    def test_quat_to_axis_angle_canonical_range(self):
        """对任意输入四元数，返回的 θ 必须 ∈ [0, π] 且能重建 q 的旋转。

        Bug 3: w<0 时 θ=2·arccos(w)>π，未利用 q≡-q 规范化。
        """
        # w<0 的四元数：表示一个绕 y 轴约 π 的旋转，但写成"长弧"形式
        for q_bad in [
            np.array([-0.70710678, 0.0, 0.70710678, 0.0]),  # w<0
            np.array([-0.8660254, 0.5, 0.0, 0.0]),
            np.array([-0.95, 0.1, 0.2, 0.2]),
        ]:
            k, th = T.quat_to_axis_angle(q_bad)
            assert th <= np.pi + 1e-9, f"θ={th} 超过 π"
            assert th >= -1e-9, f"θ={th} 为负"
            # 必须能重建原旋转
            q_back = T.axis_angle_to_quat(k, th)
            R_back = T.quat_to_R(q_back)
            R_orig = T.quat_to_R(q_bad)
            assert_allclose(R_back, R_orig, atol=1e-9)

    def test_axis_angle_quat_round_trip(self, rng):
        for _ in range(20):
            v = rng.normal(size=3)
            k = v / np.linalg.norm(v)
            theta = rng.uniform(0.1, np.pi - 0.05)
            q = T.axis_angle_to_quat(k, theta)
            k2, theta2 = T.quat_to_axis_angle(q)
            assert_allclose(theta2, theta, atol=1e-9)
            # 轴方向应一致（θ∈(0,π) 内确定）
            assert_allclose(k2, k, atol=1e-9)

    def test_rpy_quat_round_trip(self, random_rpy):
        q = T.rpy_to_quat(random_rpy)
        rpy_back = T.quat_to_rpy(q)
        assert_allclose(T.rpy_to_R(rpy_back), T.rpy_to_R(random_rpy), atol=1e-9)


# ============================================================
# 齐次变换矩阵
# ============================================================
class TestHomogeneous:
    def test_make_T_defaults(self):
        T4 = T.make_T()
        assert_allclose(T4, np.eye(4), atol=1e-12)

    def test_make_T_rotation_only(self):
        R = T.rot_x(0.3)
        T4 = T.make_T(R=R)
        expected = np.eye(4)
        expected[:3, :3] = R
        assert_allclose(T4, expected, atol=1e-12)

    def test_make_T_translation_only(self):
        p = np.array([1.0, 2.0, 3.0])
        T4 = T.make_T(p=p)
        expected = np.eye(4)
        expected[:3, 3] = p
        assert_allclose(T4, expected, atol=1e-12)

    def test_make_T_both(self):
        R = T.rot_y(0.5)
        p = np.array([1.0, 2.0, 3.0])
        T4 = T.make_T(R, p)
        expected = np.eye(4)
        expected[:3, :3] = R
        expected[:3, 3] = p
        assert_allclose(T4, expected, atol=1e-12)

    def test_T_to_Rp_round_trip(self):
        R = T.rot_z(0.7)
        p = np.array([4.0, 5.0, 6.0])
        T4 = T.make_T(R, p)
        R2, p2 = T.T_to_Rp(T4)
        assert_allclose(R2, R, atol=1e-12)
        assert_allclose(p2, p, atol=1e-12)
        # 返回的应是副本，修改不影响外部
        R2[0, 0] = 999.0
        assert T4[0, 0] != 999.0

    def test_T_inv_is_inverse(self, random_R, rng):
        p = rng.normal(size=3)
        T4 = T.make_T(random_R, p)
        T_inv = T.T_inv(T4)
        assert_allclose(T4 @ T_inv, np.eye(4), atol=1e-9)
        assert_allclose(T_inv @ T4, np.eye(4), atol=1e-9)

    def test_T_inv_pure_translation(self):
        # 纯平移的逆 = 反向平移
        T4 = T.make_T(p=np.array([1.0, 2.0, 3.0]))
        T_inv = T.T_inv(T4)
        expected = T.make_T(p=np.array([-1.0, -2.0, -3.0]))
        assert_allclose(T_inv, expected, atol=1e-12)

    def test_T_mul(self, random_R, rng):
        v = rng.normal(size=3)
        k = v / np.linalg.norm(v)
        R2 = T.rodrigues(k, rng.uniform(0.1, 3.0))
        T1 = T.make_T(random_R, rng.normal(size=3))
        T2 = T.make_T(R2, rng.normal(size=3))
        assert_allclose(T.T_mul(T1, T2), T1 @ T2, atol=1e-9)

    def test_adT_shape_and_block_structure(self):
        T4 = T.make_T(np.eye(3), np.array([1.0, 2.0, 3.0]))
        Ad = T.adT(T4)
        assert Ad.shape == (6, 6)
        # 左下角应为 0
        assert_allclose(Ad[3:, :3], np.zeros((3, 3)), atol=1e-12)

    def test_adT_known_value(self):
        """纯绕 Z 转 90°、不平移时，Ad_T 应为分块对角且上下都是 R。"""
        T4 = T.make_T(T.rot_z(np.pi / 2), np.zeros(3))
        Ad = T.adT(T4)
        R = T.rot_z(np.pi / 2)
        expected = np.zeros((6, 6))
        expected[:3, :3] = R
        expected[3:, 3:] = R
        assert_allclose(Ad, expected, atol=1e-12)

    def test_adT_identity(self):
        Ad = T.adT(np.eye(4))
        assert_allclose(Ad, np.eye(6), atol=1e-12)


# ============================================================
# 球面线性插值 slerp
# ============================================================
class TestSlerp:
    def test_endpoints(self, random_R, rng):
        v = rng.normal(size=3)
        k = v / np.linalg.norm(v)
        R1 = T.rodrigues(k, rng.uniform(0.5, 2.0))
        assert_allclose(T.slerp(random_R, R1, 0.0), random_R, atol=1e-9)
        assert_allclose(T.slerp(random_R, R1, 1.0), R1, atol=1e-9)

    def test_same_rotation_identity(self):
        R = T.rot_x(0.5)
        # 两个相同旋转之间插值应处处等于该旋转
        for s in [0.0, 0.3, 0.7, 1.0]:
            assert_allclose(T.slerp(R, R, s), R, atol=1e-9)

    def test_result_is_rotation_matrix(self, random_R, rng):
        v = rng.normal(size=3)
        k = v / np.linalg.norm(v)
        R1 = T.rodrigues(k, rng.uniform(0.5, 2.0))
        for s in [0.25, 0.5, 0.75]:
            _assert_rotation_matrix(T.slerp(random_R, R1, s))

    def test_takes_shortest_arc(self):
        """两个相隔近 π 的旋转应走短弧（结果插值应 < 半圈）。"""
        R0 = np.eye(3)
        R1 = T.rot_x(np.pi - 0.01)
        # 中点角度应约为 (π-0.01)/2，不应走到 π
        R_mid = T.slerp(R0, R1, 0.5)
        _, theta_mid = T.R_to_axis_angle(R_mid)
        assert theta_mid < np.pi - 0.3
