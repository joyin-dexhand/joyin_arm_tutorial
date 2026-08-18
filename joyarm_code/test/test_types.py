"""``joyarm.utils.types`` 的单元测试：模块契约 / 枚举 / 数据类读写 / 序列化 /
相等性。"""
from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose

from joyarm.utils import types
from joyarm.utils import transforms as T

# JointLimits 的 10 个数组字段名（读/写与默认值独立性测试共用）
JOINT_LIMITS_ARRAY_FIELDS = [
    "q_min",
    "q_max",
    "dq_max",
    "ddq_max",
    "tau_max",
    "temp_motor_max",
    "temp_driver_max",
    "voltage_min",
    "voltage_max",
    "current_max",
]


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
# Pose
# ============================================================
class TestPose:
    def test_pose_defaults(self):
        pose = types.Pose()
        assert_allclose(pose.position, np.zeros(3))
        assert_allclose(pose.orientation, [1, 0, 0, 0])

    def test_pose_from_T_round_trip(self, random_R, rng):
        p = rng.normal(size=3)
        T4 = T.Rp_to_T(random_R, p)
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


# ============================================================
# 状态快照
# ============================================================
class TestStateSnapshots:
    def test_joint_state_defaults_shape(self):
        js = types.JointState()
        assert js.q.shape == (0,)
        assert js.dq.shape == (0,)
        assert js.tau.shape == (0,)
        assert js.enabled.dtype == bool
        assert js.error.dtype == bool
        assert js.comm_ok.dtype == bool
        assert js.angle_ok.dtype == bool
        assert js.temp_motor.shape == (0,)
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
# 模块契约（__all__ / 三层导入）
# ============================================================
class TestModuleContract:
    """``types`` 模块导出契约与三层导入路径一致性。"""

    # 模块内导入的依赖名（不属于本模块定义，不要求列入 __all__）
    IMPORTED = {
        "annotations",
        "np",
        "dataclass",
        "field",
        "fields",
        "Enum",
        "T_to_Rp",
        "R_to_quat",
        "quat_to_R",
        "Rp_to_T",
    }

    def test_all_names_gettable(self):
        missing = [n for n in types.__all__ if not hasattr(types, n)]
        assert missing == [], f"__all__ 中有未定义的名字: {missing}"

    def test_all_no_duplicates(self):
        names = types.__all__
        dups = [n for n in set(names) if names.count(n) > 1]
        assert dups == [], f"__all__ 中有重复: {dups}"

    def test_all_covers_public_definitions(self):
        # 模块中每个公共名（排除导入依赖）都应列入 __all__
        missing = sorted(
            n
            for n in dir(types)
            if not n.startswith("_")
            and n not in self.IMPORTED
            and n not in types.__all__
        )
        assert missing == [], f"存在未列入 __all__ 的公共名: {missing}"

    def test_reexported_identically_at_all_levels(self):
        # 三条导入路径（types / joyarm.utils / joyarm）指向同一对象，
        # 且均列入两个上层包的 __all__
        import joyarm
        import joyarm.utils

        for name in types.__all__:
            assert (
                getattr(types, name)
                is getattr(joyarm.utils, name)
                is getattr(joyarm, name)
            ), f"{name} 三层导入不是同一对象"
            assert name in joyarm.__all__
            assert name in joyarm.utils.__all__


# ============================================================
# 枚举语义
# ============================================================
class TestEnumSemantics:
    """按值构造语义。"""

    ENUMS = [
        types.ControlMode,
        types.Severity,
        types.SafetyAction,
        types.TrajectorySpace,
    ]

    @pytest.mark.parametrize("enum_cls", ENUMS, ids=lambda c: c.__name__)
    def test_by_value_construction(self, enum_cls):
        # 每个成员按其 value 构造回同一对象（亦 pin 值唯一、无别名成员）
        for m in enum_cls:
            assert enum_cls(m.value) is m

    def test_by_value_invalid_raises(self):
        with pytest.raises(ValueError):
            types.ControlMode("nonexistent")


# ============================================================
# 字段读/写
# ============================================================
class TestFieldReadWrite:
    """全字段「写→读」往返：构造传值后逐字段读回。"""

    def test_fields_mutable_after_construction(self):
        # 数据类为可变快照：构造后字段可整体替换 / 列表可追加
        js = types.JointState()
        js.q = np.array([1.0, 2.0])
        assert_allclose(js.q, [1.0, 2.0])
        st = types.ArmState()
        st.mode = types.ControlMode.MIT
        st.errors.append("e")
        assert st.mode is types.ControlMode.MIT
        assert st.errors == ["e"]

    def test_wrench_read_write(self, rng):
        f = rng.normal(size=3)
        t = rng.normal(size=3)
        w = types.Wrench(force=f, torque=t)
        assert_allclose(w.force, f)
        assert_allclose(w.torque, t)

    def test_twist_read_write(self, rng):
        lin = rng.normal(size=3)
        ang = rng.normal(size=3)
        tw = types.Twist(linear=lin, angular=ang)
        assert_allclose(tw.linear, lin)
        assert_allclose(tw.angular, ang)

    def test_pose_read_write(self, rng):
        p3 = rng.normal(size=3)
        q4 = rng.normal(size=4)
        q4 /= np.linalg.norm(q4)
        p = types.Pose(position=p3, orientation=q4)
        assert_allclose(p.position, p3)
        assert_allclose(p.orientation, q4)

    def test_pose_T_with_non_unit_quaternion(self):
        # quat_to_R 内部归一化：非单位四元数不影响 T 的正交性
        R = types.Pose(orientation=np.array([2.0, 0.0, 0.0, 0.0])).T[:3, :3]
        assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-9)

    def test_joint_state_read_write_n7(self, rng):
        n = 7
        vals = {
            "q": rng.normal(size=n),
            "dq": rng.normal(size=n),
            "ddq": rng.normal(size=n),
            "tau": rng.normal(size=n),
            "enabled": rng.integers(0, 2, size=n).astype(bool),
            "error": rng.integers(0, 2, size=n).astype(bool),
            "comm_ok": rng.integers(0, 2, size=n).astype(bool),
            "angle_ok": rng.integers(0, 2, size=n).astype(bool),
            "temp_motor": 30.0 + rng.normal(size=n),
            "temp_driver": 40.0 + rng.normal(size=n),
        }
        js = types.JointState(voltage=48.0, current=3.5, **vals)
        for fname, v in vals.items():
            assert_allclose(getattr(js, fname), v)
        assert js.error.dtype == bool  # bool dtype 保持
        assert (js.voltage, js.current) == (48.0, 3.5)

    def test_tcp_state_read_write(self):
        pose = types.Pose(position=np.array([0.1, 0.2, 0.3]))
        twist = types.Twist(linear=np.array([0.01, 0.0, 0.0]))
        wrench = types.Wrench(force=np.array([1.0, 0.0, 0.0]))
        ts = types.TcpState(pose=pose, twist=twist, wrench=wrench)
        assert ts.pose is pose
        assert ts.twist is twist
        assert ts.wrench is wrench

    def test_arm_state_read_write(self):
        js = types.JointState(q=np.array([0.1, -0.2]))
        st = types.ArmState(
            joint=js,
            mode=types.ControlMode.VELOCITY,
            timestamp=123.456,
            errors=["e1", "e2"],
        )
        assert st.joint is js
        assert st.mode is types.ControlMode.VELOCITY
        assert st.timestamp == 123.456
        assert st.errors == ["e1", "e2"]

    @pytest.mark.parametrize("fname", JOINT_LIMITS_ARRAY_FIELDS)
    def test_joint_limits_each_field(self, fname, rng):
        vals = rng.normal(size=5)
        jl = types.JointLimits(**{fname: vals})
        assert_allclose(getattr(jl, fname), vals)

    def test_tcp_limits_read_write(self, rng):
        box = rng.normal(size=(3, 2))
        tl = types.TcpLimits(
            workspace_box=box,
            v_lin_max=0.5,
            v_ang_max=2.0,
            f_max=20.0,
            t_max=1.5,
        )
        assert_allclose(tl.workspace_box, box)
        assert (
            tl.v_lin_max,
            tl.v_ang_max,
            tl.f_max,
            tl.t_max,
        ) == (0.5, 2.0, 20.0, 1.5)

    def test_violation_read_write(self):
        v = types.Violation(
            layer="joint",
            joint_idx=2,
            metric="dq",
            value=3.2,
            limit=2.0,
            severity=types.Severity.ERROR,
        )
        assert (v.layer, v.joint_idx, v.metric) == ("joint", 2, "dq")
        assert (v.value, v.limit) == (3.2, 2.0)
        assert v.severity is types.Severity.ERROR

    def test_ikresult_read_write(self, rng):
        q = rng.normal(size=6)
        r = types.IKResult(q=q, success=True, err=1e-7, n_iter=42)
        assert_allclose(r.q, q)
        assert (r.success, r.err, r.n_iter) == (True, 1e-7, 42)

    def test_compliance_params_read_write(self, rng):
        K_raw = rng.normal(size=(6, 6))
        K = K_raw + K_raw.T  # 对称刚度阵
        D = np.eye(6) * 2.0
        M = np.eye(6) * 0.5
        c = types.ComplianceParams(K=K, D=D, M=M)
        assert_allclose(c.K, K)
        assert_allclose(c.D, D)
        assert_allclose(c.M, M)


# ============================================================
# 默认值实例独立性
# ============================================================
class TestDefaultsIndependence:
    """default_factory 独立性：任意两实例的 ndarray 默认字段互不共享。"""

    def test_array_defaults_not_shared(self):
        # 遍历全部数据类的全部 ndarray 字段（新增类型/字段自动纳入）
        for name in types.__all__:
            cls = getattr(types, name)
            if not dataclasses.is_dataclass(cls):
                continue
            a, b = cls(), cls()
            for f in dataclasses.fields(a):
                va, vb = getattr(a, f.name), getattr(b, f.name)
                if not isinstance(va, np.ndarray):
                    continue
                assert va is not vb, f"{name}.{f.name} 默认值在实例间共享"
                if va.size > 0:  # 非空默认数组：原地修改 a 不影响 b
                    before = vb.copy()
                    va[:] = 123
                    assert_allclose(vb, before, err_msg=f"{name}.{f.name}")

    def test_nested_defaults_not_shared(self):
        # 嵌套默认（ArmState → joint/tcp）亦逐实例独立；
        # errors 列表的独立性由 test_arm_state_errors_mutable_per_instance 覆盖
        a = types.ArmState()
        b = types.ArmState()
        a.joint.q = np.array([1.0, 2.0])
        a.tcp.pose.position = np.array([9.0, 9.0, 9.0])
        assert b.joint.q.shape == (0,)
        assert_allclose(b.tcp.pose.position, 0.0)


# ============================================================
# 序列化 / dataclasses 工具往返
# ============================================================
class TestSerialization:
    """pickle / asdict / astuple / replace 往返保持值与类型。"""

    @staticmethod
    def _full_arm_state(rng):
        n = 3
        js = types.JointState(
            q=rng.normal(size=n),
            dq=rng.normal(size=n),
            enabled=rng.integers(0, 2, size=n).astype(bool),
            error=rng.integers(0, 2, size=n).astype(bool),
            voltage=48.0,
            current=2.0,
        )
        tcp = types.TcpState(pose=types.Pose(position=rng.normal(size=3)))
        return types.ArmState(
            joint=js,
            tcp=tcp,
            mode=types.ControlMode.TORQUE,
            timestamp=99.5,
            errors=["overtemp"],
        )

    def test_pickle_round_trip(self, rng):
        s = self._full_arm_state(rng)
        s2 = pickle.loads(pickle.dumps(s))
        assert s2 == s  # 深相等（numpy 安全 __eq__）
        assert isinstance(s2.joint, types.JointState)
        assert isinstance(s2.tcp.pose, types.Pose)
        assert s2.mode is types.ControlMode.TORQUE
        assert s2.errors == ["overtemp"]

    def test_asdict_nested(self, rng):
        s = self._full_arm_state(rng)
        d = dataclasses.asdict(s)
        assert d["timestamp"] == 99.5
        assert d["mode"] is types.ControlMode.TORQUE
        assert_allclose(d["joint"]["q"], s.joint.q)
        # 嵌套 dataclass 递归展开为 dict
        assert isinstance(d["tcp"], dict)
        assert isinstance(d["tcp"]["pose"], dict)
        assert_allclose(d["tcp"]["pose"]["position"], s.tcp.pose.position)

    def test_replace_keeps_unreplaced_fields(self, rng):
        s = self._full_arm_state(rng)
        r = dataclasses.replace(s, timestamp=1.0, errors=[])
        assert (r.timestamp, r.errors) == (1.0, [])
        assert r.joint == s.joint  # 未替换字段保持


# ============================================================
# 相等性语义（numpy 安全 __eq__）
# ============================================================
class TestEqualitySemantics:
    """含 ndarray 字段的 dataclass 相等性：等值 / 不等 / 异类 / NaN。"""

    DATA_CLASSES = [
        types.Wrench,
        types.Twist,
        types.Pose,
        types.JointState,
        types.TcpState,
        types.ArmState,
        types.JointLimits,
        types.TcpLimits,
        types.Violation,
        types.IKResult,
        types.ComplianceParams,
    ]

    @pytest.mark.parametrize(
        "cls", DATA_CLASSES, ids=lambda c: c.__name__
    )
    def test_default_instances_equal(self, cls):
        # 等值实例 == 为 True（dataclass 默认 __eq__ 会抛 ValueError）
        assert cls() == cls()

    def test_different_values_not_equal(self):
        p = types.Pose(position=np.array([1.0, 2.0, 3.0]))
        q = types.Pose(position=np.array([1.0, 2.0, 3.1]))
        assert p != q

    def test_shape_mismatch_returns_false_no_raise(self):
        a = types.JointState(q=np.zeros(3))
        b = types.JointState(q=np.zeros(4))
        assert a != b

    def test_cross_type_comparison_false(self):
        assert (types.Pose() == 42) is False
        assert types.Pose() != "pose"

    def test_nan_fields_equal_positions(self):
        # NaN 位于相同位置视为相等（equal_nan=True）
        a = types.IKResult(q=np.array([np.nan, 1.0]))
        b = types.IKResult(q=np.array([np.nan, 1.0]))
        c = types.IKResult(q=np.array([1.0, np.nan]))
        assert a == b
        assert a != c

    def test_enum_and_nested_fields_participate(self):
        # 枚举 / 列表 / 嵌套 dataclass 字段均参与相等性
        s1 = types.ArmState(
            mode=types.ControlMode.MIT, errors=["x"]
        )
        s2 = types.ArmState(
            mode=types.ControlMode.MIT, errors=["x"]
        )
        s3 = types.ArmState(
            mode=types.ControlMode.POSITION, errors=["x"]
        )
        assert s1 == s2
        assert s1 != s3
        t1 = types.TcpState(pose=types.Pose(position=np.array([1.0, 0.0, 0.0])))
        t2 = types.TcpState(pose=types.Pose(position=np.array([1.0, 0.0, 0.0])))
        t3 = types.TcpState(pose=types.Pose(position=np.array([2.0, 0.0, 0.0])))
        assert t1 == t2
        assert t1 != t3

    def test_unhashable(self):
        # 与可变快照语义一致：不可哈希（pin 现状）
        with pytest.raises(TypeError):
            hash(types.Pose())


# ============================================================
# 深层嵌套相等性边缘用例
# ============================================================
class TestEqualityDeepEdgeCases:
    """多层嵌套 dataclass（ArmState→TcpState→Pose）的相等性递归正确性。"""

    def test_armstate_deep_equal_with_nested_data(self, rng):
        """两个值相同的 ArmState（含 JointState + TcpState + Pose）应 ==。"""
        n = 3
        js = types.JointState(q=rng.normal(size=n), dq=rng.normal(size=n))
        tcp = types.TcpState(
            pose=types.Pose(
                position=rng.normal(size=3),
                orientation=T.R_to_quat(T.rot_z(0.5)),
            )
        )

        def make():
            return types.ArmState(
                joint=types.JointState(q=js.q.copy(), dq=js.dq.copy()),
                tcp=types.TcpState(
                    pose=types.Pose(
                        position=tcp.pose.position.copy(),
                        orientation=tcp.pose.orientation.copy(),
                    )
                ),
                mode=types.ControlMode.TORQUE,
                timestamp=42.0,
                errors=["e1"],
            )

        s1, s2 = make(), make()
        assert s1 == s2  # 深相等：嵌套 dataclass 逐层比较

        # 改任一嵌套字段 → 不等
        s2.tcp.pose.position = s2.tcp.pose.position + 0.001
        assert s1 != s2

    def test_equal_nan_consistency_nested(self):
        """嵌套 Pose 的 NaN position：相同位置 NaN 视为相等，不同位置则不等。"""
        nan_pose_a = types.Pose(position=np.array([np.nan, 0.0, 0.0]))
        nan_pose_b = types.Pose(position=np.array([np.nan, 0.0, 0.0]))
        clean_pose = types.Pose(position=np.array([0.0, 0.0, 0.0]))

        tcp_a = types.TcpState(pose=nan_pose_a)
        tcp_b = types.TcpState(pose=nan_pose_b)
        tcp_c = types.TcpState(pose=clean_pose)

        assert tcp_a == tcp_b      # NaN 在相同位置 → 相等
        assert tcp_a != tcp_c      # NaN vs 非 NaN → 不等
