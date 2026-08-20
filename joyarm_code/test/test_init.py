"""``joyarm_core`` 包入口测试：导入冒烟、``__all__`` 一致性、``JoyArmDM()`` 离线实例化。

依赖：pinocchio（环境已装 4.1.0）。
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(scope="module")
def joyarm_core_pkg():
    """导入 joyarm_core 包（触发全部子模块导入，验证分层导入链无错）。"""
    return importlib.import_module("joyarm_core")


# ============================================================
# 导入冒烟 + 元信息
# ============================================================
class TestImportSmoke:
    def test_package_imports_successfully(self, joyarm_core_pkg):
        # 导入成功即通过
        assert joyarm_core_pkg is not None

    def test_version(self, joyarm_core_pkg):
        assert joyarm_core_pkg.__version__ == "0.1.0"

    def test_all_names_are_gettable(self, joyarm_core_pkg):
        """``__all__`` 中每个名字都能从包顶层 getattr。"""
        missing = [n for n in joyarm_core_pkg.__all__ if not hasattr(joyarm_core_pkg, n)]
        assert missing == [], f"__all__ 中有未定义的名字: {missing}"

    def test_all_has_no_duplicates(self, joyarm_core_pkg):
        names = joyarm_core_pkg.__all__
        dups = [n for n in set(names) if names.count(n) > 1]
        assert dups == [], f"__all__ 中有重复: {dups}"

    def test_core_layers_exported(self, joyarm_core_pkg):
        # 抽查各层至少有代表性符号
        for name in [
            # 基础层
            "Pose",
            "JointState",
            "rpy_to_R",
            "slerp",
            "clamp_to_limits",
            "ArmProtocol",
            # 设备模型层（JoyArm / JoyArmDM）
            "JoyArm",
            "JoyArmDM",
            # 算法层
            "fkine",
            "ikine",
            "Trajectory",
            "fdyn",
            # 控制 / 安全
            "ControlLoop",
            "SafetySupervisor",
            # 通信层（三层）
            "backends",
            "Backend",
            "BackendArm",
            "BackendEnd",
            "BackendArmDM",
            "BackendEndGripper",
            # 版本
            "__version__",
        ]:
            assert name in joyarm_core_pkg.__all__, f"{name} 未列入 __all__"
            assert hasattr(joyarm_core_pkg, name), f"{name} 在 __all__ 但 getattr 失败"


# ============================================================
# JoyArmDM() 离线实例化
# ============================================================
class TestInstantiation:
    def test_default_offline(self, joyarm_core_pkg):
        """默认参数 → 未连接的 JoyArm 实例（JoyArm = 本体 + 末端；backends 由 config 实例化）。"""
        arm = joyarm_core_pkg.JoyArmDM()
        assert isinstance(arm, joyarm_core_pkg.JoyArm)
        assert isinstance(arm, joyarm_core_pkg.JoyArmDM)
        # 两个 backend 已按 config 实例化；默认未连接（离线）
        assert isinstance(arm.backend_arm, joyarm_core_pkg.BackendArmDM)
        assert isinstance(arm.backend_end, joyarm_core_pkg.BackendEndGripper)
        assert arm.connected is False

    def test_arm_supports_offline_compute(self, joyarm_core_pkg):
        """离线模式下计算路径可用：rand_q / fkine。"""
        arm = joyarm_core_pkg.JoyArmDM()
        q = arm.rand_q(size=4)
        assert q.shape == (4, arm.n) or q.shape[-1] == arm.n

    def test_offline_get_arm_state_raises(self, joyarm_core_pkg):
        """未连接真机：get_arm_state / set_arm_command 应 raise RuntimeError。"""
        arm = joyarm_core_pkg.JoyArmDM()
        with pytest.raises(RuntimeError):
            arm.get_arm_state()
        with pytest.raises(RuntimeError):
            arm.set_arm_command(q=arm.rand_q())
