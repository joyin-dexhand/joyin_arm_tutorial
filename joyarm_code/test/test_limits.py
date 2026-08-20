"""``joyarm_core.utils.limits.clamp_to_limits`` 的单元测试（指令路径守卫）。"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from joyarm_core.utils import types
from joyarm_core.utils.limits import clamp_to_limits


# ============================================================
# clamp_to_limits（关节层，已实现）
# ============================================================
class TestClampToLimits:
    def _make_limits(self, n=3, lo=-1.0, hi=1.0):
        return types.JointLimits(
            q_min=np.full(n, lo), q_max=np.full(n, hi)
        )

    def test_clamp_within_limits_unchanged(self):
        limits = self._make_limits(n=3, lo=-1.0, hi=1.0)
        targets = np.array([0.5, -0.5, 0.0])
        out = clamp_to_limits(targets, limits)
        assert_allclose(out, targets)

    def test_clamp_above_max(self):
        limits = self._make_limits(n=2, lo=-1.0, hi=1.0)
        targets = np.array([1.5, 0.5])
        out = clamp_to_limits(targets, limits)
        assert_allclose(out, [1.0, 0.5])

    def test_clamp_below_min(self):
        limits = self._make_limits(n=2, lo=-1.0, hi=1.0)
        targets = np.array([-2.0, 0.0])
        out = clamp_to_limits(targets, limits)
        assert_allclose(out, [-1.0, 0.0])

    def test_clamp_batch_N_n(self):
        # (N,n) 批量裁剪
        limits = self._make_limits(n=3, lo=-0.5, hi=0.5)
        targets = np.array(
            [[1.0, 0.0, -1.0], [0.2, 0.9, -0.3]]
        )
        out = clamp_to_limits(targets, limits)
        assert_allclose(
            out, [[0.5, 0.0, -0.5], [0.2, 0.5, -0.3]]
        )

    def test_clamp_dimension_mismatch_raises(self):
        limits = self._make_limits(n=3, lo=-1.0, hi=1.0)
        targets = np.array([0.5, 0.5])  # 2 维，限位 3 维
        with pytest.raises(ValueError, match="不匹配"):
            clamp_to_limits(targets, limits)

    def test_clamp_per_joint_different_limits(self):
        # 不同关节不同限位
        limits = types.JointLimits(
            q_min=np.array([-1.0, -0.5, 0.0]),
            q_max=np.array([1.0, 0.5, 2.0]),
        )
        targets = np.array([-2.0, 0.9, -0.5])
        out = clamp_to_limits(targets, limits)
        assert_allclose(out, [-1.0, 0.5, 0.0])

    def test_clamp_accepts_list_input(self):
        # list / 嵌套 list 输入（内部 asarray 强转 float）
        limits = self._make_limits(n=2, lo=-1.0, hi=1.0)
        assert_allclose(
            clamp_to_limits([2.0, -2.0], limits), [1.0, -1.0]
        )
        assert_allclose(
            clamp_to_limits([[2.0, -2.0], [0.0, 0.5]], limits),
            [[1.0, -1.0], [0.0, 0.5]],
        )

    def test_clamp_int_targets_promoted_to_float(self):
        # int 目标数组 → float 输出
        limits = self._make_limits(n=3, lo=-1.0, hi=1.0)
        out = clamp_to_limits(np.array([0, 2, -2]), limits)
        assert out.dtype == np.float64
        assert_allclose(out, [0.0, 1.0, -1.0])

    def test_clamp_does_not_mutate_input(self):
        # 裁剪返回新数组，不原地修改输入
        limits = self._make_limits(n=3, lo=-1.0, hi=1.0)
        targets = np.array([5.0, -5.0, 0.0])
        orig = targets.copy()
        clamp_to_limits(targets, limits)
        assert_allclose(targets, orig)

    def test_clamp_min_greater_than_max_raises(self):
        # 限位配置错误（q_min > q_max）应显式报错而非静默裁剪
        limits = types.JointLimits(
            q_min=np.array([1.0, -1.0]), q_max=np.array([-1.0, 1.0])
        )
        with pytest.raises(ValueError, match="限位配置错误"):
            clamp_to_limits(np.zeros(2), limits)

    def test_clamp_empty_targets_with_empty_limits(self):
        # 空目标 + 空限位：维度匹配（0 == 0），返回空数组
        out = clamp_to_limits(np.zeros(0), types.JointLimits())
        assert out.shape == (0,)

    def test_clamp_degenerate_min_eq_max(self):
        # 退化限位 min == max：目标无论高低均夹到该值
        limits = types.JointLimits(
            q_min=np.array([0.5]), q_max=np.array([0.5])
        )
        for target in (-1.0, 2.0):
            assert_allclose(
                clamp_to_limits(np.array([target]), limits), [0.5]
            )

    # ---- Bug 回归：标量输入崩溃 ----
    def test_clamp_scalar_input_raises_valueerror(self):
        """标量（0-d）目标应抛 ValueError，而非 IndexError。

        Bug: ``clamp_to_limits(5.0, limits)`` 中 ``np.asarray(5.0).shape``
        为 ``()``，``targets.shape[-1]`` 越界抛 ``IndexError``。
        修复后应在维度校验前拦截 0-d 输入。
        """
        limits = self._make_limits(n=3, lo=-1.0, hi=1.0)
        with pytest.raises(ValueError, match="标量|scalar"):
            clamp_to_limits(5.0, limits)

    def test_clamp_nan_targets_propagates(self):
        """NaN 目标不被裁剪修正，原样传播（clip 语义）。"""
        limits = self._make_limits(n=3, lo=-1.0, hi=1.0)
        targets = np.array([0.5, np.nan, -0.5])
        out = clamp_to_limits(targets, limits)
        assert np.isnan(out[1])
        assert_allclose(out[[0, 2]], [0.5, -0.5])

    def test_clamp_inf_targets_clipped(self):
        """±inf 目标被夹到 q_max / q_min。"""
        limits = self._make_limits(n=2, lo=-1.0, hi=1.0)
        targets = np.array([np.inf, -np.inf])
        out = clamp_to_limits(targets, limits)
        assert_allclose(out, [1.0, -1.0])

    def test_clamp_large_batch(self):
        """大批量 (1000, 6) 裁剪正确性。"""
        n = 6
        limits = types.JointLimits(
            q_min=np.full(n, -0.5), q_max=np.full(n, 0.5)
        )
        rng = np.random.default_rng(99)
        targets = rng.uniform(-2.0, 2.0, size=(1000, n))
        out = clamp_to_limits(targets, limits)
        assert out.shape == (1000, n)
        assert np.all(out >= -0.5 - 1e-12)
        assert np.all(out <= 0.5 + 1e-12)
