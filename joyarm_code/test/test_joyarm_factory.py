"""JoyArm 工厂与单类离线测试（不依赖硬件）。

覆盖架构约束的单类行为：①工厂软失败语义（未知型号 → ``None`` + 失败信息）；
②六域成员字典机制（各域 REGISTRY 默认空表——教学各章实现注册后接入）；③配置
读取/运行期设置/自检（``get_config``/``set_config``/``check_config``）；④离线
语义（执行类抛错、未加载域门面显性报错）。

运行：``python test/test_joyarm_factory.py`` 或 pytest。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import joyarm_factory, JoyArmFactory, JoyArm  # noqa: E402
from joyarm_core.joyarm import load_config  # noqa: E402
from joyarm_core.joyarm.joyarm import _build_domain  # noqa: E402
from joyarm_core.utils.types import Severity  # noqa: E402

_ARM = None


def _arm():
    global _ARM
    if _ARM is None:
        _ARM = joyarm_factory("joyarm_dm")
        assert _ARM is not None, "joyarm_dm 创建失败"
    return _ARM


def test_factory_creates_dm():
    arm = _arm()
    assert arm.n == 6
    assert arm.model == "joyarm_dm"
    assert JoyArmFactory().list_models() == ["joyarm_dm"]


def test_factory_unknown_model_returns_none():
    """软失败语义（架构约束）：未知型号返回 None + 输出失败信息（不抛异常）。"""
    for bad in ("joyarm_nx", "", "backend_dm"):
        assert joyarm_factory(bad) is None, f"{bad!r} 应返回 None"


def test_load_config_strict_and_lenient():
    assert load_config("joyarm_dm") is not None
    assert load_config("nonexistent") is None          # 容错：None
    try:
        load_config("nonexistent", strict=True)
        raise AssertionError("strict 模式应抛 ValueError")
    except ValueError:
        pass


def test_domain_dicts_default_empty():
    """教学过渡态：各域 REGISTRY 默认空表、config 未配置即无成员；门面调用显性报错。"""
    arm = _arm()
    for d in ("fkine", "ikine", "jacobian", "dynamics", "traj", "control"):
        assert arm._active_name[d] is None
        assert arm.list_solvers(d) == []
    try:
        arm.fkine(arm.rand_q(rng=np.random.default_rng(0)))
        raise AssertionError("未加载 fkine 应抛 RuntimeError")
    except RuntimeError:
        pass


class _DummyA:
    pass


class _DummyB:
    def __init__(self, x=0):
        self.x = x


def test_set_solver_switch_and_alias():
    """机制测试：注入哑成员验证切换/别名/报错（各域注册表默认空，章节实现后接入）。"""
    arm = _arm()
    arm._controllers["dummy_a"] = _DummyA()
    arm._controllers["dummy_b"] = _DummyB()
    arm._active_name["control"] = "dummy_a"
    inst = arm.set_solver("control", "dummy_b")
    assert isinstance(inst, _DummyB)
    assert arm._active_name["control"] == "dummy_b"
    assert arm.set_controller("dummy_a") is not None  # 惯用别名
    try:
        arm.set_solver("control", "nonexistent")
        raise AssertionError("未加载注册名应抛 ValueError")
    except ValueError:
        pass
    try:
        arm.set_solver("bogus_domain", "x")
        raise AssertionError("未知域应抛 ValueError")
    except ValueError:
        pass
    # 清理注入成员
    arm._controllers.clear()
    arm._active_name["control"] = None


def test_build_domain_soft_fail_and_multiload():
    """软失败（架构约束）：无效注册名跳过 + 警告；列表多载全部加载；未配置静默跳过。"""
    reg = {"a": _DummyA, "b": _DummyB}
    members = _build_domain("control", reg, [{"name": "bogus"}, {"name": "b", "x": 123}])
    assert sorted(members) == ["b"] and members["b"].x == 123
    m2 = _build_domain("control", reg, ["a", "b"])
    assert sorted(m2) == ["a", "b"]
    assert _build_domain("control", reg, None) == {}   # 域未配置：静默跳过


def test_get_config_deepcopy():
    arm = _arm()
    cfg = arm.get_config()
    assert cfg["basic"]["name"] == "joyarm_dm"
    assert "arm_mdh_and_limits" in cfg["joyarm"]     # 教学数据经类内 config 读取
    cfg["basic"]["name"] = "hacked"
    fresh = arm.get_config()
    assert fresh["basic"]["name"] == "joyarm_dm"


def test_set_config_whitelist():
    arm = _arm()
    # 软限位即时生效
    arm.set_config("basic.utils.joint_limits_soft_margin", 0.1)
    span = arm.joint_limits.q_max - arm.joint_limits.q_min
    assert np.allclose(arm.qlow, arm.joint_limits.q_min + 0.1 * span)
    arm.set_config("basic.utils.joint_limits_soft_margin", 0.05)  # 还原
    # 白名单外拒绝
    try:
        arm.set_config("backend.name", "x")
        raise AssertionError("白名单外路径应抛 ValueError")
    except ValueError:
        pass


def test_check_config_clean_passes():
    arm = _arm()
    issues = arm.check_config()
    assert issues == [], f"干净配置不应有违规：{issues}"


def test_check_config_detects_problems():
    cfg = load_config("joyarm_dm")
    # q_home 长度错 + 无效注册名 + backend 关节数不符
    cfg = copy.deepcopy(cfg)
    cfg["joyarm"]["q_home"] = [0.0, 0.0]
    cfg["robotics"]["ikine"] = "bogus"
    cfg["backend"]["arm"]["joints"] = cfg["backend"]["arm"]["joints"][:4]
    arm = JoyArm(model="joyarm_dm", config=cfg)
    issues = arm.check_config()
    metrics = [v.metric for v in issues]
    assert "q_home:len" in metrics
    assert "robotics.ikine:not_loaded" in metrics
    assert "backend.arm.joints:len" in metrics
    assert any(v.severity is Severity.ERROR for v in issues)
    # 未加载域的门面调用显性报错
    try:
        arm.ikine(np.eye(4))
        raise AssertionError("未加载 ikine 应抛 RuntimeError")
    except RuntimeError:
        pass


def test_offline_execution_raises():
    """离线语义：限位采样等纯计算可用、执行类 RuntimeError。"""
    arm = _arm()
    q = arm.rand_q(rng=np.random.default_rng(1))
    assert q.shape == (arm.n,)
    for fn in (arm.get_arm_state, arm.enable_arm, arm.end_open):
        try:
            fn()
            raise AssertionError(f"{fn.__name__} 离线应抛 RuntimeError")
        except RuntimeError:
            pass


def test_repr_contains_state():
    arm = _arm()
    r = repr(arm)
    assert "JoyArm" in r and "model='joyarm_dm'" in r and "offline" in r


# ----------------------------------------------------------
# 直跑
# ----------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"{len(fns)} 项全部通过")
