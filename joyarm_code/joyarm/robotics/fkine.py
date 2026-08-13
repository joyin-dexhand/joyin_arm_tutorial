"""正运动学（运动学模型层）。

形状重载（无单独 ``*_batch`` 函数）：

- ``q`` 为 ``(n,)`` → 返回单个结果；
- ``q`` 为 ``(N,n)`` → 返回批量结果。

默认走 **pinocchio 黑盒**（``urdf + pin``，应用版）。若需手写 MDH 递归（白盒原理版），
请在 ``Mas`` 子类中覆盖 ``fkine`` 方法（不在本函数加 ``method`` 参数）。

``rep``（返回表示）：

- ``"T"``：``Pose`` 对象（position + orientation 四元数，默认）。
- ``"pos"``：``(3,)`` 末端位置。
- ``"se3"``：``pinocchio.SE3`` 对象（进阶用）。

对应章节：Ch2 即用。
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..utils.types import Pose

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover
    pin = None

__all__ = ["fkine"]


def fkine(
    mas,
    q: np.ndarray,
    frame: Optional[Union[str, int]] = None,
    rep: str = "T",
):
    """正运动学：关节角 → 末端（或任意帧）位姿。

    :param mas: :class:`joyarm.arms.mas.Mas` 实例（满足 :class:`~joyarm.utils.interfaces.MasProtocol`）。
    :param q: ``(n,)`` 或 ``(N,n)`` 关节角，弧度。
    :param frame: 帧名（``str``）或帧索引（``int``）；缺省为末端帧。
    :param rep: 返回表示，``"T"``/``"pos"``/``"se3"``。
    :return:
        - ``q`` 为 ``(n,)``：
            - ``rep="T"``  → :class:`Pose`
            - ``rep="pos"`` → ``p(3,)``
            - ``rep="se3"`` → ``pinocchio.SE3``
        - ``q`` 为 ``(N,n)``：对应 ``list[Pose]`` / ``(N,3)`` / ``list[pinocchio.SE3]``

    :raises ValueError: ``rep`` 非法时抛出。
    """
    if rep not in ("T", "pos", "se3"):
        raise ValueError(f"未知 rep={rep!r}（仅支持 'T'/'pos'/'se3'）")

    q_arr = np.asarray(q, dtype=float)
    # 形状重载：1D = 单点 FK，2D = 批量 FK
    if q_arr.ndim == 1:
        return _fkine_single(mas, q_arr, frame=frame, rep=rep)
    elif q_arr.ndim == 2:
        return _fkine_batch(mas, q_arr, frame=frame, rep=rep)
    else:
        raise ValueError(f"q 维度需为 1 或 2，收到 q.shape={q_arr.shape}")


# ============================================================
# 内部实现
# ============================================================
def _fkine_single(mas, q: np.ndarray, frame, rep: str):
    """单点 FK：委托 ``mas.frame_placement``（已含 ``T_base`` 基坐标系偏移），再按 rep 转表示。"""
    T_world = mas.frame_placement(q, frame)
    if rep == "T":
        return Pose.from_T(T_world)
    elif rep == "pos":
        return T_world[:3, 3]
    else:  # se3
        return pin.SE3(T_world[:3, :3], T_world[:3, 3])


def _fkine_batch(mas, Q: np.ndarray, frame, rep: str):
    """批量 FK：循环复用 data 缓冲区。"""
    N = Q.shape[0]
    if rep == "T":
        return [_fkine_single(mas, Q[i], frame=frame, rep="T") for i in range(N)]
    elif rep == "pos":
        out = np.empty((N, 3))
        for i in range(N):
            out[i] = _fkine_single(mas, Q[i], frame=frame, rep="pos")
        return out
    else:  # se3
        return [_fkine_single(mas, Q[i], frame=frame, rep="se3") for i in range(N)]
