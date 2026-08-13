"""``joyarm`` 包入口测试：导入冒烟、``__all__`` 一致性、``JoyArmRebotDM()`` 离线实例化。

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
            "MasProtocol",
            # 设备模型层（Mas / End / Arm）
            "Mas",
            "End",
            "Arm",
            "JoyArmRebotDM",
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
            "BackendMas",
            "BackendEnd",
            "BackendMasRebotDM",
            "BackendEndJoyGripper",
            # 版本
            "__version__",
        ]:
            assert name in joyarm_pkg.__all__, f"{name} 未列入 __all__"
            assert hasattr(joyarm_pkg, name), f"{name} 在 __all__ 但 getattr 失败"


# ============================================================
# JoyArmRebotDM() 离线实例化
# ============================================================
class TestInstantiation:
    def test_default_offline(self, joyarm_pkg):
        """默认参数 → 未连接的 Arm 实例（Arm = Mas + End；backends 由 config 实例化）。"""
        arm = joyarm_pkg.JoyArmRebotDM()
        assert isinstance(arm, joyarm_pkg.Arm)
        assert isinstance(arm, joyarm_pkg.Mas)               # Arm is-a Mas
        assert isinstance(arm, joyarm_pkg.JoyArmRebotDM)
        # 两个 backend 已按 config 实例化；默认未连接（离线）
        assert isinstance(arm.backend_mas, joyarm_pkg.BackendMasRebotDM)
        assert isinstance(arm.end.backend_end, joyarm_pkg.BackendEndJoyGripper)
        assert arm.connected is False

    def test_arm_supports_offline_compute(self, joyarm_pkg):
        """离线模式下计算路径可用：rand_q / fkine。"""
        arm = joyarm_pkg.JoyArmRebotDM()
        q = arm.rand_q(size=4)
        assert q.shape == (4, arm.n) or q.shape[-1] == arm.n

    def test_offline_get_state_raises(self, joyarm_pkg):
        """未连接真机：get_state / command 应 raise RuntimeError。"""
        arm = joyarm_pkg.JoyArmRebotDM()
        with pytest.raises(RuntimeError):
            arm.get_state()
        with pytest.raises(RuntimeError):
            arm.command(q=arm.rand_q())
