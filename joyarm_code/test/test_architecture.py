"""架构测试：依赖方向不变量 / 求解器策略族契约 / 门面与注入链路。

- 依赖方向：robotics / utils 不 import joyarms（组合根单向向下）；
- ``FkineSolver`` 模板方法（形状重载 / rep 转换 / 批量循环）用手写 2R 平面臂内核做
  数学对照——策略族只依赖机械臂的最小公开属性（鸭子类型，无协议类）；
- JoyArm 门面 → ``_fkine_solver`` 成员链路：注入假求解器即换实现（动态组合）。
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from joyarm_core.joyarms.joyarm import _build_component
from joyarm_core.robotics.fkine import (
    REGISTRY as FKINE_REGISTRY,
    FkineSolver,
    MdhFkineSolver,
    PinFkineSolver,
)
from joyarm_core.utils.transforms import Rp_to_T
from joyarm_core.utils.types import Pose

JOYARM_ROOT = Path(importlib.import_module("joyarm_core").__file__).resolve().parent


# ============================================================
# 依赖方向不变量：robotics / utils 不 import joyarms
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
        """分层：robotics/、utils/ 禁止 import joyarm_core.joyarms。"""
        files = [*(JOYARM_ROOT / "robotics").rglob("*.py"),
                 *(JOYARM_ROOT / "utils").rglob("*.py")]
        assert files, "扫描范围意外为空"
        violations = [
            f"{f.relative_to(JOYARM_ROOT)} -> {m}"
            for f in files
            for m in _absolute_imports(f)
            if m == "joyarm_core.joyarms" or m.startswith("joyarm_core.joyarms.")
        ]
        assert violations == []


# ============================================================
# FkineSolver 模板契约：手写 2R 平面臂内核做数学对照
# ============================================================
class Hand2RFkineSolver(FkineSolver):
    """2 关节平面臂手写内核（l1=l2=1）：位置=(cos q0+cos(q0+q1), sin q0+sin(q0+q1), 0)。"""

    def frame_T(self, arm, q, frame=None):
        q0, q1 = float(q[0]), float(q[1])
        c, s = np.cos(q0 + q1), np.sin(q0 + q1)
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        T[0, 3] = np.cos(q0) + c
        T[1, 3] = np.sin(q0) + s
        return arm.T_base @ T


@pytest.fixture
def solver():
    return Hand2RFkineSolver()


@pytest.fixture
def fake_arm():
    # 函数级：个别用例会修改 T_base，不可复用
    return SimpleNamespace(n=2, T_base=np.eye(4))


class TestFkineSolverTemplate:
    @pytest.mark.parametrize("q, expected", [
        ([0.0, 0.0], [2.0, 0.0, 0.0]),          # 完全伸展
        ([np.pi / 2, 0.0], [0.0, 2.0, 0.0]),    # 整体转 90°
        ([0.0, np.pi / 2], [1.0, 1.0, 0.0]),    # 肘部折 90°
        ([0.0, np.pi], [0.0, 0.0, 0.0]),        # 完全折叠
    ], ids=["stretch", "rot90", "elbow90", "fold"])
    def test_pos_matches_hand_computed(self, solver, fake_arm, q, expected):
        """rep="pos" 与平面 2R 臂手工算例一致。"""
        assert_allclose(solver.solve(fake_arm, np.array(q), rep="pos"), expected, atol=1e-12)

    def test_T_pose_roundtrip(self, solver, fake_arm):
        """rep="T" 返回 Pose，position / 重建的 T 与手写内核一致。"""
        q = np.array([0.3, -0.8])
        pose = solver.solve(fake_arm, q, rep="T")
        assert isinstance(pose, Pose)
        T_ref = solver.frame_T(fake_arm, q)
        assert_allclose(pose.position, T_ref[:3, 3], atol=1e-9)
        assert_allclose(pose.T, T_ref, atol=1e-9)

    def test_batch_consistent_with_single(self, solver, fake_arm):
        """批量 (N,2) → (N,3)，逐行与单点结果一致。"""
        Q = np.array([[0.0, 0.0], [np.pi / 2, 0.0], [0.0, np.pi / 2]])
        P = solver.solve(fake_arm, Q, rep="pos")
        assert P.shape == (3, 3)
        for i in range(3):
            assert_allclose(P[i], solver.solve(fake_arm, Q[i], rep="pos"), atol=1e-12)

    def test_se3(self, solver, fake_arm):
        """rep="se3" 返回 pinocchio.SE3，齐次矩阵与手写内核一致。"""
        import pinocchio as pin

        q = np.array([0.0, np.pi / 2])
        M = solver.solve(fake_arm, q, rep="se3")
        assert isinstance(M, pin.SE3)
        assert_allclose(M.homogeneous, solver.frame_T(fake_arm, q), atol=1e-12)

    def test_T_base_offset_applied(self, solver, fake_arm):
        """T_base 基坐标系偏移生效：基座抬升 5 m 后末端随之偏移。"""
        fake_arm.T_base = Rp_to_T(np.eye(3), np.array([0.0, 0.0, 5.0]))
        pos = solver.solve(fake_arm, np.array([0.0, 0.0]), rep="pos")
        assert_allclose(pos, [2.0, 0.0, 5.0], atol=1e-12)

    def test_bad_rep_and_dim_raise(self, solver, fake_arm):
        """rep 非法 / q 维度非法均抛 ValueError。"""
        with pytest.raises(ValueError):
            solver.solve(fake_arm, np.zeros(2), rep="xyz")
        with pytest.raises(ValueError):
            solver.solve(fake_arm, np.zeros((2, 2, 2)))


# ============================================================
# JoyArm 门面 → _fkine_solver 链路（动态组合）
# ============================================================
class FixedSolver(FkineSolver):
    """注入用假内核：返回固定位姿，验证门面确实走了成员。"""

    def frame_T(self, arm, q, frame=None):
        T = np.eye(4)
        T[:3, 3] = [7.0, 8.0, 9.0]
        return T


@pytest.fixture(scope="module")
def arm():
    """离线 JoyArmDM（默认未连接，纯计算可用）。"""
    return importlib.import_module("joyarm_core").JoyArmDM()


class TestSolverComposition:
    def test_default_member_is_pin(self, arm):
        """config 缺省时组装 PinFkineSolver。"""
        assert isinstance(arm._fkine_solver, PinFkineSolver)

    def test_facade_delegates_to_member(self, arm):
        """注入假内核后，门面结果即假内核结果——成员可换。"""
        original = arm._fkine_solver
        try:
            arm._fkine_solver = FixedSolver()
            assert_allclose(arm.fkine(arm.q_neutral, rep="pos"), [7.0, 8.0, 9.0], atol=1e-12)
            assert_allclose(arm.fkine(arm.rand_q(size=3), rep="pos"),
                            np.tile([7.0, 8.0, 9.0], (3, 1)), atol=1e-12)
        finally:
            arm._fkine_solver = original

    def test_facade_math_smoke(self, arm):
        """默认门面：返回 (4,4) 齐次变换（旋转块正交、底行 [0,0,0,1]）。"""
        T = arm.fkine(arm.q_neutral, rep="T").T
        assert T.shape == (4, 4)
        assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
        assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_build_component_registry(self):
        """注册表按名选型；未知名报错并列出可选项。"""
        assert type(_build_component(FKINE_REGISTRY, "pin", "fkine")) is PinFkineSolver
        assert type(_build_component(FKINE_REGISTRY, "mdh", "fkine")) is MdhFkineSolver
        with pytest.raises(ValueError, match="可用"):
            _build_component(FKINE_REGISTRY, "nope", "fkine")
