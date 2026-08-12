"""``joyarm`` 包入口测试：导入冒烟、``__all__`` 一致性、``load_arm()`` 离线实例化。

依赖：pinocchio（环境已装 4.1.0）。
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.fixture(scope="module")
def joyarm_pkg():
    """导入 joyarm 包（触发全部子模块导入，验证分层导入链无错）。"""
    return importlib.import_module("joyarm")


# ============================================================
# 导入冒烟 + 元信息
# ============================================================
class TestImportSmoke:
    def test_package_imports_successfully(self, joyarm_pkg):
        # 导入成功即通过
        assert joyarm_pkg is not None

    def test_version(self, joyarm_pkg):
        assert joyarm_pkg.__version__ == "0.1.0"

    def test_all_names_are_gettable(self, joyarm_pkg):
        """``__all__`` 中每个名字都能从包顶层 getattr。"""
        missing = [n for n in joyarm_pkg.__all__ if not hasattr(joyarm_pkg, n)]
        assert missing == [], f"__all__ 中有未定义的名字: {missing}"

    def test_all_has_no_duplicates(self, joyarm_pkg):
        names = joyarm_pkg.__all__
        dups = [n for n in set(names) if names.count(n) > 1]
        assert dups == [], f"__all__ 中有重复: {dups}"

    def test_core_layers_exported(self, joyarm_pkg):
        # 抽查各层至少有代表性符号
        for name in [
            # 基础层
            "Pose",
            "JointState",
            "rpy_to_R",
            "slerp",
            "clamp_to_limits",
            # 运动学模型层
            "Arm",
            "JoyArmRebotDM",
            "fkine",
            "ikine",
            # 轨迹 / 动力学
            "Trajectory",
            "fdyn",
            # 控制 / 安全 / 应用支撑
            "ControlLoop",
            "SafetySupervisor",
            "Gripper",
            # 可视化 / 通信
            "viz",
            "backends",
            "JoyArmRebotDMBackend",
            # 工厂 / 版本
            "load_arm",
            "__version__",
        ]:
            assert name in joyarm_pkg.__all__, f"{name} 未列入 __all__"
            assert hasattr(joyarm_pkg, name), f"{name} 在 __all__ 但 getattr 失败"


# ============================================================
# load_arm() 工厂
# ============================================================
class TestLoadArm:
    def test_default_offline(self, joyarm_pkg):
        """默认参数 → 离线 Arm 实例。"""
        arm = joyarm_pkg.load_arm()
        assert isinstance(arm, joyarm_pkg.Arm)
        assert isinstance(arm, joyarm_pkg.JoyArmRebotDM)
        # 离线模式 backend 为 None
        assert arm.backend is None

    def test_explicit_model(self, joyarm_pkg):
        arm = joyarm_pkg.load_arm(model="joyarm_rebot_dm")
        assert isinstance(arm, joyarm_pkg.JoyArmRebotDM)

    def test_unknown_model_raises(self, joyarm_pkg):
        with pytest.raises(ValueError, match="未知型号"):
            joyarm_pkg.load_arm(model="does_not_exist")

    def test_unknown_backend_raises(self, joyarm_pkg):
        with pytest.raises(ValueError, match="未知 backend"):
            joyarm_pkg.load_arm(backend="can")

    def test_real_backend_offline_instantiation(self, joyarm_pkg):
        """backend='real' 仅构造 backend 实例（不真连），Arm 仍可被实例化。"""
        arm = joyarm_pkg.load_arm(
            backend="real", backend_kwargs={}
        )
        assert arm.backend is not None
        assert isinstance(arm.backend, joyarm_pkg.JoyArmRebotDMBackend)

    def test_backend_kwargs_with_offline_not_swallowed(self, joyarm_pkg):
        """Bug 5 回归：backend=None 时传 backend_kwargs 不应触发 TypeError。

        原实现仅在 backend=='real' 分支 pop backend_kwargs，
        backend=None 时该键被原样透传给 Arm 构造函数，触发意外关键字错误。
        """
        # 应被 load_arm 自身消化（pop 出来），不再透传给 Arm
        arm = joyarm_pkg.load_arm(backend=None, backend_kwargs={})
        assert arm.backend is None

    def test_arm_supports_offline_compute(self, joyarm_pkg):
        """离线模式下计算路径可用：rand_q / fkine。"""
        arm = joyarm_pkg.load_arm()
        q = arm.rand_q(size=4)
        assert q.shape == (4, arm.n) or q.shape[-1] == arm.n

    def test_offline_get_state_raises(self, joyarm_pkg):
        """离线模式：get_state / command 应 raise RuntimeError。"""
        arm = joyarm_pkg.load_arm()
        with pytest.raises(RuntimeError):
            arm.get_state()
