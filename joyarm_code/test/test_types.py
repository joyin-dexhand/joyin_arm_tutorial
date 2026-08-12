"""``joyarm.types`` 的单元测试：枚举 / 数据类 / clamp_to_limits。"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from joyarm_code.joyarm.utils import types
from joyarm_code.joyarm.utils import transforms as T


# ============================================================
# 枚举
# ============================================================
class TestEnums:
    def test_control_mode_values(self):
        assert types.ControlMode.POSITION.value == "position"
        assert types.ControlMode.VELOCITY.value == "velocity"
        assert types.ControlMode.TORQUE.value == "torque"
        assert types.ControlMode.MIT.value == "mit"
        # 共 4 个成员
        assert len(list(types.ControlMode)) == 4

    def test_severity_values(self):
        assert [m.value for m in types.Severity] == [
            "info",
            "warning",
            "error",
            "critical",
        ]

    def test_safety_action_values(self):
        assert types.SafetyAction.NONE.value == "none"
        assert types.SafetyAction.CLAMP.value == "clamp"
        assert types.SafetyAction.DAMPING_HOLD.value == "damping_hold"
        assert types.SafetyAction.FREEZE.value == "freeze"
        assert types.SafetyAction.ESTOP.value == "estop"

    def test_trajectory_space_values(self):
        assert types.TrajectorySpace.JOINT.value == "joint"
        assert types.TrajectorySpace.CARTESIAN.value == "cartesian"


# ============================================================
# Pose / Transform
# ============================================================
class TestPoseTransform:
    def test_pose_defaults(self):
        pose = types.Pose()
        assert_allclose(pose.position, np.zeros(3))
        assert_allclose(pose.orientation, [1, 0, 0, 0])

    def test_pose_from_T_round_trip(self, random_R, rng):
        p = rng.normal(size=3)
        T4 = T.make_T(random_R, p)
        pose = types.Pose.from_T(T4)
        assert_allclose(pose.position, p, atol=1e-9)
        assert_allclose(pose.T, T4, atol=1e-9)

    def test_pose_T_property_reconstructs(self):
        R = T.rot_x(0.4)
        p = np.array([1.0, 2.0, 3.0])
        pose = types.Pose(position=p, orientation=T.R_to_quat(R))
        T4 = pose.T
        assert_allclose(T4[:3, :3], R, atol=1e-9)
        assert_allclose(T4[:3, 3], p, atol=1e-9)

    def test_transform_defaults(self):
        tf = types.Transform()
        assert_allclose(tf.translation, np.zeros(3))
        assert_allclose(tf.rotation, [1, 0, 0, 0])

    def test_transform_from_T_round_trip(self, random_R, rng):
        p = rng.normal(size=3)
        T4 = T.make_T(random_R, p)
        tf = types.Transform.from_T(T4)
        assert_allclose(tf.translation, p, atol=1e-9)
        assert_allclose(tf.T, T4, atol=1e-9)

    def test_transform_to_pose(self):
        tf = types.Transform(
            translation=np.array([1.0, 2.0, 3.0]),
            rotation=np.array([0.5, 0.5, 0.5, 0.5]),
        )
        pose = tf.to_pose()
        assert_allclose(pose.position, tf.translation)
        assert_allclose(pose.orientation, tf.rotation)


# ============================================================
# 状态快照
# ============================================================
class TestStateSnapshots:
    def test_joint_state_defaults_shape(self):
        js = types.JointState()
        assert js.q.shape == (0,)
        assert js.dq.shape == (0,)
        assert js.tau.shape == (0,)
        assert js.status.dtype == int
        assert js.voltage == 0.0
        assert js.current == 0.0

    def test_tcp_state_defaults(self):
        ts = types.TcpState()
        assert isinstance(ts.pose, types.Pose)
        assert isinstance(ts.twist, types.Twist)
        assert isinstance(ts.wrench, types.Wrench)

    def test_arm_state_defaults(self):
        s = types.ArmState()
        assert isinstance(s.joint, types.JointState)
        assert isinstance(s.tcp, types.TcpState)
        assert s.mode == types.ControlMode.POSITION
        assert s.timestamp == 0.0
        assert s.errors == []

    def test_arm_state_errors_mutable_per_instance(self):
        # 默认 list 不应在实例间共享
        a = types.ArmState()
        b = types.ArmState()
        a.errors.append("x")
        assert b.errors == []

    def test_wrench_defaults(self):
        w = types.Wrench()
        assert_allclose(w.force, np.zeros(3))
        assert_allclose(w.torque, np.zeros(3))

    def test_twist_defaults(self):
        tw = types.Twist()
        assert_allclose(tw.linear, np.zeros(3))
        assert_allclose(tw.angular, np.zeros(3))


# ============================================================
# 限位声明
# ============================================================
class TestLimits:
    def test_joint_limits_defaults(self):
        jl = types.JointLimits()
        # 所有数组字段默认空
        assert jl.q_min.shape == (0,)
        assert jl.q_max.shape == (0,)
        assert jl.dq_max.shape == (0,)
        assert jl.tau_max.shape == (0,)

    def test_tcp_limits_defaults(self):
        tl = types.TcpLimits()
        assert tl.workspace_box.shape == (3, 2)
        assert tl.v_lin_max == 0.0

    def test_violation_defaults(self):
        v = types.Violation()
        assert v.joint_idx == -1
        assert v.severity == types.Severity.WARNING


# ============================================================
# 求解结果 / 柔顺参数
# ============================================================
class TestResultsAndCompliance:
    def test_ikresult_defaults(self):
        r = types.IKResult()
        assert r.success is False
        assert r.err == float("inf")
        assert r.n_iter == 0

    def test_compliance_params_defaults_are_identity(self):
        c = types.ComplianceParams()
        assert_allclose(c.K, np.eye(6))
        assert_allclose(c.D, np.eye(6))
        assert_allclose(c.M, np.eye(6))


# ============================================================
# clamp_to_limits
# ============================================================
class TestClampToLimits:
    def _make_limits(self, n=3, lo=-1.0, hi=1.0):
        return types.JointLimits(
            q_min=np.full(n, lo), q_max=np.full(n, hi)
        )

    def test_clamp_within_limits_unchanged(self):
        limits = self._make_limits(n=3, lo=-1.0, hi=1.0)
        targets = np.array([0.5, -0.5, 0.0])
        out = types.clamp_to_limits(targets, limits)
        assert_allclose(out, targets)

    def test_clamp_above_max(self):
        limits = self._make_limits(n=2, lo=-1.0, hi=1.0)
        targets = np.array([1.5, 0.5])
        out = types.clamp_to_limits(targets, limits)
        assert_allclose(out, [1.0, 0.5])

    def test_clamp_below_min(self):
        limits = self._make_limits(n=2, lo=-1.0, hi=1.0)
        targets = np.array([-2.0, 0.0])
        out = types.clamp_to_limits(targets, limits)
        assert_allclose(out, [-1.0, 0.0])

    def test_clamp_batch_N_n(self):
        # (N,n) 批量裁剪
        limits = self._make_limits(n=3, lo=-0.5, hi=0.5)
        targets = np.array(
            [[1.0, 0.0, -1.0], [0.2, 0.9, -0.3]]
        )
        out = types.clamp_to_limits(targets, limits)
        assert_allclose(
            out, [[0.5, 0.0, -0.5], [0.2, 0.5, -0.3]]
        )

    def test_clamp_dimension_mismatch_raises(self):
        limits = self._make_limits(n=3, lo=-1.0, hi=1.0)
        targets = np.array([0.5, 0.5])  # 2 维，限位 3 维
        with pytest.raises(ValueError, match="不匹配"):
            types.clamp_to_limits(targets, limits)

    def test_clamp_per_joint_different_limits(self):
        # 不同关节不同限位
        limits = types.JointLimits(
            q_min=np.array([-1.0, -0.5, 0.0]),
            q_max=np.array([1.0, 0.5, 2.0]),
        )
        targets = np.array([-2.0, 0.9, -0.5])
        out = types.clamp_to_limits(targets, limits)
        assert_allclose(out, [-1.0, 0.5, 0.0])
