"""pytest 共享配置：路径注入 + 可复现随机 fixture。

约定
----
- 数值断言统一用 ``numpy.testing.assert_allclose``，旋转矩阵 atol=1e-9，
  涉及 θ≈π 奇异分支 atol=1e-6。
- 所有随机量来自固定种子的 fixture，保证 CI 与本地结果一致。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# 把 joyarm_code/ 父目录加进 sys.path，使 ``import joyarm_core`` 可达
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_THIS_DIR)  # .../joyarm_code
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """固定种子的随机数发生器，保证随机用例可复现。"""
    return np.random.default_rng(seed=0)


@pytest.fixture
def random_rpy(rng):
    """返回非奇异（避开 ±π/2）的随机 RPY 三元组。"""
    while True:
        rpy = rng.uniform(-np.pi, np.pi, size=3)
        # 避开 pitch=±π/2 附近的万向锁区域
        if abs(abs(rpy[1]) - np.pi / 2) > 0.05:
            return rpy


@pytest.fixture
def random_R(rng):
    """返回一个随机旋转矩阵（由随机轴角构造，保证正交、det=+1）。"""
    # 在单位球面上随机取一根轴
    v = rng.normal(size=3)
    k = v / np.linalg.norm(v)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    # 用 axis_angle_to_R 构造（已归一化轴）
    from joyarm_core.utils.transforms import axis_angle_to_R

    return axis_angle_to_R(k, theta)
