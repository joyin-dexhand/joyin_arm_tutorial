"""正运动学（运动学模型层）。

形状重载（无单独 ``*_batch`` 函数）：

- ``q`` 为 ``(n,)`` → 返回单个结果；
- ``q`` 为 ``(N,n)`` → 返回批量结果。

``method`` 双版本（与 ikine/jacobian/dyn 统一）：

- ``"auto"``：pinocchio 黑盒（应用版，本模块实现）。
- ``"manual"``：手写 MDH 递归（白盒，原理版）——
  当前 ``raise NotImplementedError("原理版待 Ch2.2 实现")`` 占位。

``rep``（返回表示）：

- ``"T"``：``Pose`` 对象（position + orientation 四元数，默认）。
- ``"pos"``：``(3,)`` 末端位置。
- ``"se3"``：``pinocchio.SE3`` 对象（进阶用）。

对应章节：Ch2.2 即用；``chapter2_2.md`` §5.2.1 表格直接调用。
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
    arm,
    q: np.ndarray,
    frame: Optional[Union[str, int]] = None,
    rep: str = "T",
    method: str = "auto",
):
    """正运动学：关节角 → 末端（或任意帧）位姿。

    :param arm: :class:`joyarm.arm.Arm` 实例。
    :param q: ``(n,)`` 或 ``(N,n)`` 关节角，弧度。
    :param frame: 帧名（``str``）或帧索引（``int``）；缺省为末端帧。
    :param rep: 返回表示，``"T"``/``"pos"``/``"se3"``。
    :param method: ``"auto"``（pinocchio 黑盒）/ ``"manual"``（白盒，占位）。
    :return:
        - ``q`` 为 ``(n,)``：
            - ``rep="T"``  → :class:`Pose`（position + orientation 四元数）
            - ``rep="pos"`` → ``p(3,)``
            - ``rep="se3"`` → ``pinocchio.SE3``
        - ``q`` 为 ``(N,n)``：
            - ``rep="T"``  → ``list[Pose]``
            - ``rep="pos"`` → ``(N,3)``
            - ``rep="se3"`` → ``list[pinocchio.SE3]``

    :raises ValueError: ``method``/``rep`` 非法时抛出。
    :raises NotImplementedError: ``method="manual"`` 时抛出（待 Ch2.2 实现）。
    """
    if method == "manual":
        # 白盒原理版（手写 MDH 递归）尚未实现，留给 Ch2.2 教学推导
        raise NotImplementedError(
            "fkine method='manual'（手写 MDH 递归原理版）待 Ch2.2 实现；"
            "当前请用 method='auto'（pinocchio 黑盒）。"
        )
    if method != "auto":
        raise ValueError(f"未知 method={method!r}（仅支持 'auto'/'manual'）")
    if rep not in ("T", "pos", "se3"):
        raise ValueError(f"未知 rep={rep!r}（仅支持 'T'/'pos'/'se3'）")

    q_arr = np.asarray(q, dtype=float)
    # 形状重载：1D = 单点 FK，2D = 批量 FK（一次性算多组关节角）
    if q_arr.ndim == 1:
        return _fkine_single(arm, q_arr, frame=frame, rep=rep)
    elif q_arr.ndim == 2:
        return _fkine_batch(arm, q_arr, frame=frame, rep=rep)
    else:
        raise ValueError(f"q 维度需为 1 或 2，收到 q.shape={q_arr.shape}")


# ============================================================
# 内部实现
# ============================================================
def _fkine_single(arm, q: np.ndarray, frame, rep: str):
    """单点 FK。始终应用 ``arm.T_base`` 基坐标系偏移，与
    :meth:`joyarm.arm.Arm.frame_placement` 语义一致（默认单位阵 → 无影响）。"""
    fid = arm._resolve_frame(frame)            # 把帧名/索引/None 解析成 pinocchio 帧 ID
    # pinocchio 正运动学两步走：先算所有关节位姿，再更新目标帧
    pin.forwardKinematics(arm.model, arm.data, q)
    pin.updateFramePlacement(arm.model, arm.data, fid)
    se3 = arm.data.oMf[fid]                    # 该帧的 SE3 位姿（pinocchio 对象）
    # 基坐标系偏移：T_world = T_base @ T_local（默认 T_base=I，透明）
    T_world = arm.T_base @ se3.homogeneous
    # 按 rep 选返回形式
    if rep == "T":                             # Pose（position+orientation 四元数）
        return Pose.from_T(T_world)
    elif rep == "pos":                         # 仅取平移部分（末端位置）
        return T_world[:3, 3]
    else:  # se3                              # 返回 pinocchio.SE3 对象（进阶用）
        return pin.SE3(T_world[:3, :3], T_world[:3, 3])


def _fkine_batch(arm, Q: np.ndarray, frame, rep: str):
    """批量 FK：循环复用 data 缓冲区。"""
    N = Q.shape[0]                            # 批量大小（多少组关节角）
    # 循环里复用同一个 arm.data 缓冲区（避免重复创建）
    if rep == "T":                            # Pose 不是 ndarray，用列表收集
        return [_fkine_single(arm, Q[i], frame=frame, rep="T") for i in range(N)]
    elif rep == "pos":
        out = np.empty((N, 3))
        for i in range(N):
            out[i] = _fkine_single(arm, Q[i], frame=frame, rep="pos")
        return out
    else:  # se3                              # se3 不是 ndarray，用列表收集
        return [_fkine_single(arm, Q[i], frame=frame, rep="se3") for i in range(N)]
