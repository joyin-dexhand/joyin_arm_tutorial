"""``ArmProtocol`` 契约测试：成员钉扎 / 实现一致性 / 鸭子类型充分性 / 负控 / 依赖方向。

验证 ``joyarm_core/utils/interfaces.py`` 定义的协议**充分、不冗余、有效**：真实
``JoyArmDM`` 与仅实现协议成员的 ``FakeArm``（2 关节平面臂手写 FK）均满足
契约；缺任一成员即判定失败；``robotics`` / ``safe_monitors`` 不 import ``joyarms``。
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from joyarm_core.joyarms.joyarm import JoyArm
from joyarm_core.robotics.fkine import fkine
from joyarm_core.utils.interfaces import ArmProtocol
from joyarm_core.utils.transforms import Rp_to_T
from joyarm_core.utils.types import JointLimits, Pose, TcpLimits

# 协议契约冻结清单（与 interfaces.py 逐字对应；有意改动时同步这里）
PROTOCOL_ATTRS = frozenset({
    "model", "data", "n", "nv", "ee_frame_name", "ee_frame_id", "T_base",
    "joint_limits", "joint_limits_soft", "tcp_limits", "q_neutral",
})
PROTOCOL_METHODS = frozenset({"frame_placement"})
PROTOCOL_MEMBERS = PROTOCOL_ATTRS | PROTOCOL_METHODS

JOYARM_ROOT = Path(importlib.import_module("joyarm_core").__file__).resolve().parent


class FakeArm:
    """仅实现协议成员的最小假本体：2 关节平面臂（l1=l2=1），手写 FK。"""

    def __init__(self):
        self.model = None
        self.data = None
        self.n = 2
        self.nv = 2
        self.ee_frame_name = "ee"
        self.ee_frame_id = 0
        self.T_base = np.eye(4)
        self.joint_limits = JointLimits(
            q_min=np.full(2, -np.pi), q_max=np.full(2, np.pi))
        self.joint_limits_soft = JointLimits(
            q_min=np.full(2, -np.pi + 0.05), q_max=np.full(2, np.pi - 0.05))
        self.tcp_limits = TcpLimits()
        self.q_neutral = np.zeros(2)

    def frame_placement(self, q, frame=None):
        """末端姿态角 = q0+q1；位置 = (cos q0 + cos(q0+q1), sin q0 + sin(q0+q1), 0)。"""
        q0, q1 = float(q[0]), float(q[1])
        c, s = np.cos(q0 + q1), np.sin(q0 + q1)
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        T[0, 3] = np.cos(q0) + c
        T[1, 3] = np.sin(q0) + s
        return self.T_base @ T


@pytest.fixture(scope="module")
def arm():
    """离线 JoyArmDM（默认未连接，纯计算可用）。"""
    return importlib.import_module("joyarm_core").JoyArmDM()


@pytest.fixture
def fake_arm():
    # 函数级：个别用例会修改 T_base，不可复用
    return FakeArm()


# ============================================================
# 契约钉扎：成员集合与冻结清单一致（防无意增删）
# ============================================================
class TestContractPinning:
    def test_member_set_frozen(self):
        """11 个数据属性 + 唯一方法 frame_placement，无多余成员。"""
        assert set(ArmProtocol.__annotations__) == PROTOCOL_ATTRS
        public_callables = {
            name for name in dir(ArmProtocol)
            if not name.startswith("_") and callable(getattr(ArmProtocol, name))
        }
        assert public_callables == PROTOCOL_METHODS

    def test_module_defines_exactly_one_object(self):
        """模块 __all__ 恰为 ["ArmProtocol"]，无其他本模块定义（无冗余）。"""
        from joyarm_core.utils import interfaces

        assert interfaces.__all__ == ["ArmProtocol"]
        defined = {
            name for name, obj in vars(interfaces).items()
            if getattr(obj, "__module__", None) == "joyarm_core.utils.interfaces"
        }
        assert defined == {"ArmProtocol"}


# ============================================================
# Protocol 语义
# ============================================================
class TestProtocolSemantics:
    def test_cannot_instantiate(self):
        """契约只能被结构化满足，不能实例化。"""
        with pytest.raises(TypeError):
            ArmProtocol()

    def test_isinstance_executable_on_plain_object(self):
        """@runtime_checkable 生效：isinstance 可执行，对裸对象返回 False。"""
        assert isinstance(object(), ArmProtocol) is False

    def test_top_level_reexport_identity(self):
        """顶层再导出与源定义同源且列入 __all__。"""
        joyarm_core = importlib.import_module("joyarm_core")
        assert joyarm_core.ArmProtocol is ArmProtocol
        assert "ArmProtocol" in joyarm_core.__all__


# ============================================================
# 真实实现一致性（JoyArmDM 经 JoyArm 满足协议）
# ============================================================
class TestArmConformance:
    def test_arm_satisfies_protocol(self, arm):
        assert isinstance(arm, ArmProtocol) is True

    @pytest.mark.parametrize("attr, expected", [
        ("n", int), ("nv", int), ("ee_frame_id", int),
        ("ee_frame_name", str),
        ("T_base", np.ndarray), ("q_neutral", np.ndarray),
        ("joint_limits", JointLimits), ("joint_limits_soft", JointLimits),
        ("tcp_limits", TcpLimits),
    ])
    def test_attribute_runtime_types(self, arm, attr, expected):
        """属性运行时类型与协议注解一致（model/data 标 Any）。"""
        assert isinstance(getattr(arm, attr), expected)

    def test_model_data_loaded(self, arm):
        assert arm.model is not None
        assert arm.data is not None

    def test_attribute_shapes(self, arm):
        n = arm.n
        assert arm.T_base.shape == (4, 4)
        assert arm.q_neutral.shape == (n,)
        for jl in (arm.joint_limits, arm.joint_limits_soft):
            assert jl.q_min.shape == (n,)
            assert jl.q_max.shape == (n,)

    def test_frame_placement_signature_matches_protocol(self):
        """JoyArm 实现与协议声明的 frame_placement 签名逐字一致。"""

        def shape_of(fn):
            return tuple(
                (p.name, p.kind, p.default)
                for p in inspect.signature(fn).parameters.values()
            )

        assert shape_of(JoyArm.frame_placement) == shape_of(ArmProtocol.frame_placement)

    def test_frame_placement_returns_valid_T(self, arm):
        """返回 (4,4) 齐次变换：旋转块正交、底行 [0,0,0,1]。"""
        T = arm.frame_placement(arm.q_neutral)
        assert T.shape == (4, 4)
        assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
        assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_protocol_consumer_path_matches_facade(self, arm, rng):
        """协议消费路径 robotics.fkine(arm, q) 与门面 arm.fkine(q) 一致。"""
        q = arm.rand_q(rng=rng)
        assert_allclose(fkine(arm, q, rep="pos"), arm.fkine(q, rep="pos"), atol=1e-12)


# ============================================================
# 鸭子类型充分性：仅实现协议成员即可驱动算法层
# ============================================================
class TestDuckTyping:
    def test_fake_satisfies_protocol(self, fake_arm):
        """纯鸭子类型（不继承任何基类）即满足协议。"""
        assert isinstance(fake_arm, ArmProtocol) is True

    @pytest.mark.parametrize("q, expected", [
        ([0.0, 0.0], [2.0, 0.0, 0.0]),          # 完全伸展
        ([np.pi / 2, 0.0], [0.0, 2.0, 0.0]),    # 整体转 90°
        ([0.0, np.pi / 2], [1.0, 1.0, 0.0]),    # 肘部折 90°
        ([0.0, np.pi], [0.0, 0.0, 0.0]),        # 完全折叠
    ], ids=["stretch", "rot90", "elbow90", "fold"])
    def test_fkine_pos_matches_hand_computed(self, fake_arm, q, expected):
        """rep="pos" 与平面 2R 臂手工算例一致。"""
        assert_allclose(fkine(fake_arm, np.array(q), rep="pos"), expected, atol=1e-12)

    def test_fkine_T_pose_roundtrip(self, fake_arm):
        """rep="T" 返回 Pose，position / 重建的 T 与手写 FK 一致（任意位形）。"""
        q = np.array([0.3, -0.8])
        pose = fkine(fake_arm, q, rep="T")
        assert isinstance(pose, Pose)
        T_ref = fake_arm.frame_placement(q)
        assert_allclose(pose.position, T_ref[:3, 3], atol=1e-9)
        assert_allclose(pose.T, T_ref, atol=1e-9)

    def test_fkine_batch_consistent_with_single(self, fake_arm):
        """批量 (N,2) → (N,3)，逐行与单点结果一致。"""
        Q = np.array([[0.0, 0.0], [np.pi / 2, 0.0], [0.0, np.pi / 2]])
        P = fkine(fake_arm, Q, rep="pos")
        assert P.shape == (3, 3)
        for i in range(3):
            assert_allclose(P[i], fkine(fake_arm, Q[i], rep="pos"), atol=1e-12)

    def test_fkine_se3(self, fake_arm):
        """rep="se3" 返回 pinocchio.SE3，齐次矩阵与手写 FK 一致。"""
        import pinocchio as pin

        q = np.array([0.0, np.pi / 2])
        M = fkine(fake_arm, q, rep="se3")
        assert isinstance(M, pin.SE3)
        assert_allclose(M.homogeneous, fake_arm.frame_placement(q), atol=1e-12)

    def test_fkine_T_base_offset_applied(self, fake_arm):
        """T_base 基坐标系偏移生效：基座抬升 5 m 后末端随之偏移。"""
        fake_arm.T_base = Rp_to_T(np.eye(3), np.array([0.0, 0.0, 5.0]))
        pos = fkine(fake_arm, np.array([0.0, 0.0]), rep="pos")
        assert_allclose(pos, [2.0, 0.0, 5.0], atol=1e-12)


# ============================================================
# 负控：缺任一成员 → isinstance 失败（每个成员都不冗余）
# ============================================================
def _protocol_members() -> dict:
    """构造完整满足协议的成员字典（值取自 FakeArm）。"""
    fake = FakeArm()
    members = {name: getattr(fake, name) for name in PROTOCOL_ATTRS}
    members["frame_placement"] = FakeArm.frame_placement
    return members


class TestNegativeControls:
    def test_complete_namespace_passes(self):
        """对照组：具备全部成员的对象通过判定。"""
        assert isinstance(SimpleNamespace(**_protocol_members()), ArmProtocol) is True

    @pytest.mark.parametrize("missing", sorted(PROTOCOL_MEMBERS))
    def test_missing_each_member_fails(self, missing):
        """逐一删成员均判定失败——每个成员都参与运行时契约。"""
        members = _protocol_members()
        del members[missing]
        assert isinstance(SimpleNamespace(**members), ArmProtocol) is False


# ============================================================
# 依赖方向不变量：robotics / safe_monitors / interfaces 不 import joyarms
# ============================================================
def _absolute_imports(path: Path) -> list:
    """AST 解析文件的全部 import，相对导入解析为绝对模块名。"""
    rel = path.relative_to(JOYARM_ROOT).with_suffix("")
    package = ("joyarm_core",) + rel.parts[:-1]
    mods = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                mods.append(node.module or "")
            else:  # level=1 → 当前包，level=2 → 父包
                base = package[: len(package) - (node.level - 1)]
                extra = tuple(node.module.split(".")) if node.module else ()
                mods.append(".".join(base + extra))
    return mods


class TestDependencyDirection:
    def test_algorithm_layer_never_imports_arms(self):
        """依赖倒置：robotics/、safe_monitors/、interfaces.py 禁止 import joyarm_core.joyarms。"""
        files = [*(JOYARM_ROOT / "robotics").glob("*.py"),
                 *(JOYARM_ROOT / "safe_monitors").glob("*.py"),
                 JOYARM_ROOT / "utils" / "interfaces.py"]
        violations = [
            f"{f.relative_to(JOYARM_ROOT)} -> {m}"
            for f in files
            for m in _absolute_imports(f)
            if m == "joyarm_core.joyarms" or m.startswith("joyarm_core.joyarms.")
        ]
        assert violations == []

    def test_interfaces_only_depends_on_utils_types(self):
        """interfaces.py 处最底层：joyarm_core 内依赖仅 utils.types。"""
        mods = [m for m in _absolute_imports(JOYARM_ROOT / "utils" / "interfaces.py")
                if m.startswith("joyarm_core")]
        assert mods == ["joyarm_core.utils.types"]
