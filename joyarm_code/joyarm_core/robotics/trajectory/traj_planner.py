"""TrajPlanner —— 轨迹规划器接口（ABC）+ Trajectory 轨迹载体

规划器子类需实现四个入口（空间 joint/cart × 粒度 p2p/waypoints）
- :meth:`plan_joint_p2p` / :meth:`plan_joint_waypoints` /
  :meth:`plan_cart_p2p` / :meth:`plan_cart_waypoints`。

``method`` 选同族内方式（如直线/圆弧），config 选整个规划器——两层选型不混淆。
config ``robotics.traj`` 段写注册名，即按名实例化装入 ``_traj_planners`` 成员字典。
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ...utils.types import Pose, TrajectorySpace

__all__ = ["TrajPlanner", "Trajectory", "JOINT_METHODS", "CART_METHODS"]

# 公有参数 method 的合法取值表（plan_joint_p2p / plan_cart_p2p 契约，各实现类共享）
JOINT_METHODS = ("cubic", "quintic", "lspb")
CART_METHODS = ("line", "arc")


class Trajectory:
    """轨迹主体：时间戳 + 采样点序列 + 元数据。

    :ivar space: :class:`~joyarm_core.utils.types.TrajectorySpace`，关节 / 笛卡尔。
    :ivar t: ``(K,)`` 时间戳序列，秒。
    :ivar q: ``(K,n)`` 关节空间采样点（``space=JOINT`` 时使用）。
    :ivar dq: ``(K,n)`` 关节速度采样（关节空间可选，p2p 族自动生成）。
    :ivar ddq: ``(K,n)`` 关节加速度采样（同上）。
    :ivar poses: ``list[Pose]`` 笛卡尔空间采样点（``space=CARTESIAN`` 时使用）。
    :ivar metadata: 元数据字典（如生成参数、源记录等）。
    """

    def __init__(self,
        space: TrajectorySpace = TrajectorySpace.JOINT,
        t: Optional[np.ndarray] = None,
        q: Optional[np.ndarray] = None,
        poses: Optional[List[Pose]] = None,
        metadata: Optional[dict] = None,
        dq: Optional[np.ndarray] = None,
        ddq: Optional[np.ndarray] = None,
    ):
        self.space: TrajectorySpace = space
        self.t: np.ndarray = np.asarray(t, dtype=float) if t is not None else np.zeros(0)
        self.q: Optional[np.ndarray] = q
        self.dq: Optional[np.ndarray] = dq
        self.ddq: Optional[np.ndarray] = ddq
        self.poses: Optional[List[Pose]] = poses
        self.metadata: dict = metadata or {}

    @property
    def duration(self) -> float:
        """轨迹总时长（末时间戳），秒。"""
        return float(self.t[-1]) if self.t.size else 0.0

    def sample(self, t: float) -> dict:
        """按时间取**最近采样点**（越界取端点；供规划/控制线程按 wall-time 求参考）。

        :return: 关节空间 ``{"t", "q", "dq", "ddq"}``（dq/ddq 可能缺省 ``None``）；
            笛卡尔空间 ``{"t", "pose"}``（``pose`` 为 :class:`Pose`）。
        """
        t = float(t)
        if self.t.size == 0:
            raise ValueError("空轨迹无法采样")
        k = int(np.clip(np.searchsorted(self.t, t), 0, self.t.size - 1))
        out = {"t": self.t[k]}
        if self.space is TrajectorySpace.JOINT:
            out["q"] = self.q[k]
            out["dq"] = self.dq[k] if self.dq is not None else None
            out["ddq"] = self.ddq[k] if self.ddq is not None else None
        else:
            out["pose"] = self.poses[k]
        return out

    def save(self, path: str) -> None:
        """轨迹持久化（npz）：时间戳、采样点与元数据一并存盘。

        :param path: 输出文件路径（``.npz``）。
        """
        data = {"space": self.space.value, "t": self.t,
                "metadata": json.dumps(self.metadata, ensure_ascii=False, default=str)}
        if self.q is not None:
            data["q"] = self.q
        if self.dq is not None:
            data["dq"] = self.dq
        if self.ddq is not None:
            data["ddq"] = self.ddq
        if self.poses is not None:
            data["poses"] = np.asarray(
                [np.concatenate([p.position, p.orientation]) for p in self.poses]
            )
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> "Trajectory":
        """从 npz 文件加载轨迹（:meth:`save` 的逆过程）。

        :param path: ``.npz`` 文件路径。
        :return: :class:`Trajectory` 实例。
        """
        with np.load(path, allow_pickle=False) as z:
            space = TrajectorySpace(str(z["space"]))
            poses = None
            if "poses" in z:
                arr = z["poses"]
                poses = [Pose(position=r[:3], orientation=r[3:]) for r in arr]
            return cls(
                space=space,
                t=z["t"],
                q=z["q"] if "q" in z else None,
                dq=z["dq"] if "dq" in z else None,
                ddq=z["ddq"] if "ddq" in z else None,
                poses=poses,
                metadata=json.loads(str(z["metadata"])) if "metadata" in z else {},
            )


class TrajPlanner(ABC):
    """轨迹规划策略接口：关节 / 笛卡尔空间的平滑轨迹序列。"""

    @abstractmethod
    def plan_joint_p2p(self,
        arm,
        q0: np.ndarray,
        qf: np.ndarray,
        *,
        method: str = "quintic",
        T: float = 2.0,
        hz: int = 200,
        **kw,
    ) -> Trajectory:
        """关节空间点到点：``method`` ∈ cubic/quintic/lspb；``T`` 总时长 s；``hz`` 采样 Hz。"""

    @abstractmethod
    def plan_joint_waypoints(self,
        arm,
        qs: List[np.ndarray],
        Ts: List[float],
        *,
        smooth: bool = True,
    ) -> Trajectory:
        """关节空间多点途经（段间五次拼接，接缝位置/速度/加速度连续）。"""

    @abstractmethod
    def plan_cart_p2p(self,
        arm,
        *,
        method: str = "line",
        T: float = 2.0,
        hz: int = 200,
        **kw,
    ) -> Trajectory:
        """笛卡尔点到点：``line``（需 ``pose0``/``posef``，:class:`Pose`）/ ``arc``
        （需 ``pose_start``/``center``/``radius``/``angle``/``plane``）；位置沿几何
        路径、姿态 slerp，均配五次时间律（C2）。"""

    @abstractmethod
    def plan_cart_waypoints(self,
        arm,
        poses: List[Pose],
        Ts: List[float],
        *,
        smooth: bool = True,
    ) -> Trajectory:
        """笛卡尔多点途经：``poses`` 为 :class:`Pose` 列表（含起止）；段间平滑拼接（C2）。"""
