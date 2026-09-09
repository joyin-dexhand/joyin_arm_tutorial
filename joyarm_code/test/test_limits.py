"""utils/limits.py 离线单测：限位构建（模型/config 直配软限位）+ 指令裁剪。

覆盖：①``joint_limits_from_model`` 硬限位解析（velocity/effort 缺失或 ``nv != n``
时上限退化 ∞）；②``soft_limits_from_cfg`` 四键直值语义（逐关节列表 / 标量广播 /
缺省键 = ±∞）与非法输入报错（非字典、未知键、负幅值、长度错、限位交叉）；
③``clamp_to_limits`` 裁剪与结构错误。

运行：``python test/test_limits.py`` 或 pytest。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import JointLimits, clamp_to_limits, joint_limits_from_model, soft_limits_from_cfg  # noqa: E402


def _hard(n: int = 3) -> JointLimits:
    """确定值硬限位（q_min/q_max/dq_max/tau_max 均逐关节不同）。"""
    return JointLimits(
        q_min=np.array([-1.0, -2.0, -0.5])[:n],
        q_max=np.array([1.0, 0.0, 1.5])[:n],
        dq_max=np.array([10.0, 20.0, 30.0])[:n],
        tau_max=np.array([5.0, 6.0, 7.0])[:n],
    )


def _model(n: int = 3, with_dq_tau: bool = True, nv: int | None = None) -> SimpleNamespace:
    """鸭子类型运动学模型（pinocchio model 等价结构；上限数组长度 = nv）。"""
    nv = n if nv is None else nv
    hard = _hard(n)
    ns = dict(
        nq=n, nv=nv,
        lowerPositionLimit=hard.q_min, upperPositionLimit=hard.q_max,
    )
    if with_dq_tau:
        ns.update(velocityLimit=np.full(nv, 25.0), effortLimit=np.full(nv, 6.0))
    return SimpleNamespace(**ns)


def test_joint_limits_from_model():
    hard = joint_limits_from_model(_model(3))
    ref = _hard(3)
    assert np.allclose(hard.q_min, ref.q_min) and np.allclose(hard.q_max, ref.q_max)
    assert np.allclose(hard.dq_max, 25.0) and np.allclose(hard.tau_max, 6.0)  # 模型均匀上限
    # velocity/effort 属性缺失 → 上限置 ∞
    hard2 = joint_limits_from_model(_model(3, with_dq_tau=False))
    assert np.all(np.isinf(hard2.dq_max)) and np.all(np.isinf(hard2.tau_max))
    # nv != n（含非旋转关节）→ 逐关节上限退化 ∞（由 config 精确覆盖）
    hard3 = joint_limits_from_model(_model(3, nv=4))
    assert np.all(np.isinf(hard3.dq_max)) and np.all(np.isinf(hard3.tau_max))


def test_soft_limits_from_cfg_direct_values():
    """四键直值（不经 margin 换算）：逐关节列表原样生效。"""
    soft = soft_limits_from_cfg({
        "q_min": [-1.3, -2.2, -0.6], "q_max": [0.9, -0.1, 1.4],
        "dq_max": [9.0, 18.0, 27.0], "tau_max": [4.5, 5.0, 5.5],
    }, n=3)
    assert np.allclose(soft.q_min, [-1.3, -2.2, -0.6])
    assert np.allclose(soft.q_max, [0.9, -0.1, 1.4])
    assert np.allclose(soft.dq_max, [9.0, 18.0, 27.0])
    assert np.allclose(soft.tau_max, [4.5, 5.0, 5.5])


def test_soft_limits_from_cfg_scalar_broadcast_and_defaults():
    # 标量：全关节统一
    soft = soft_limits_from_cfg({"q_max": 0.5}, n=3)
    assert np.allclose(soft.q_max, 0.5)
    assert np.all(np.isneginf(soft.q_min))              # 缺省键 = ±∞（该量不设软限）
    assert np.all(np.isinf(soft.dq_max)) and np.all(np.isinf(soft.tau_max))


def test_soft_limits_from_cfg_invalid_inputs():
    def _expect_error(cfg, n, keyword):
        try:
            soft_limits_from_cfg(cfg, n)
            raise AssertionError(f"cfg={cfg!r} 应抛 ValueError（{keyword}）")
        except ValueError as e:
            assert keyword in str(e)

    _expect_error(0.05, 3, "四键字典")                          # 非字典
    _expect_error({"bogus": 0.1}, 3, "未知键")                  # 未知键
    _expect_error({"q_max": [0.1, 0.2]}, 3, "长度")             # 列表长度 ≠ n
    _expect_error({"q_min": 1.0, "q_max": -1.0}, 3, "q_min > q_max")   # 限位交叉
    _expect_error({"dq_max": -1.0}, 3, "负值")                  # 负的幅值上限


def test_clamp_to_limits():
    limits = _hard(3)
    # (n,)：越下限抬回、越上限压回、界内不动
    out = clamp_to_limits(np.array([-5.0, -1.0, 0.7]), limits)
    assert np.allclose(out, np.array([-1.0, -1.0, 0.7]))
    # (N,n) 批量广播
    out2 = clamp_to_limits(np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]), limits)
    assert np.allclose(out2, np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))

    def _expect_error(targets, keyword):
        try:
            clamp_to_limits(targets, limits)
            raise AssertionError(f"targets={targets!r} 应抛 ValueError（{keyword}）")
        except ValueError as e:
            assert keyword in str(e)

    _expect_error(0.5, "标量")
    _expect_error(np.zeros(4), "不匹配")
    bad = JointLimits(q_min=np.array([1.0, -2.0, -0.5]), q_max=np.array([-1.0, 0.0, 1.5]),
                      dq_max=np.zeros(3), tau_max=np.zeros(3))
    try:
        clamp_to_limits(np.zeros(3), bad)
        raise AssertionError("q_min > q_max 应抛 ValueError")
    except ValueError as e:
        assert "配置错误" in str(e)


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
