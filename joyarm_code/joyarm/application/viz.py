"""meshcat 运动学可视化（横切层）。

职责：meshcat 浏览器可视化（点云 / 坐标系 / 机器人模型 / 轨迹预演）。
**纯运动学层面，不含物理仿真**（架构设计第 2 条）——仿真/预演/教学走
``Arm(backend=None)`` 离线模式 + 本模块。

依赖
----

- :mod:`meshcat`（浏览器 3D 可视化，重依赖）：惰性导入；缺失时调用
  meshcat 相关函数会抛出带安装提示的 :class:`ImportError`。
- :mod:`matplotlib`（无浏览器环境备用的 3D 点云）。

"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "make_viewer",
    "load_robot",
    "plot_points",
    "plot_frame",
    "plot_workspace",
    "plot_trajectory",
    "plot_points_mpl",
]


def _require_meshcat():
    """惰性导入 meshcat；缺失时抛带提示的 ImportError。"""
    try:
        import meshcat

        return meshcat
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "meshcat 未安装。请运行：pip install meshcat\n"
            "（首次使用会自动启动本地 zerorpc 服务并打开浏览器）"
        ) from e


# ============================================================
# viewer / 机器人模型
# ============================================================
def make_viewer(url: Optional[str] = None):
    """创建 MeshCat 查看器。

    :param url: 已有 meshcat-server 的 URL；缺省时 meshcat 自动启服务。
    :return: :class:`meshcat.Visualizer` 实例。
    """
    meshcat = _require_meshcat()
    # 若给定 url（已有 meshcat-server），连过去；否则 meshcat 自动起本地服务并开浏览器
    v = meshcat.Visualizer(zmq_url=url) if url else meshcat.Visualizer()
    return v


def load_robot(viewer, arm) -> None:
    """在 MeshCat 中加载机器人模型（URDF/mesh）。

    需要 ``arm`` 已加载几何（``load_geometry=True`` 构造，即
    ``arm.visual_model`` / ``arm.collision_model`` 非空）。若未加载，仅清空
    场景并提示——运动学可视化（``plot_frame``/``plot_workspace``）仍可用，
    但看不到机器人本体 mesh。

    :param viewer: :func:`make_viewer` 返回的查看器。
    :param arm: :class:`joyarm.arm.Arm` 实例（建议用 ``load_geometry=True`` 构造）。
    """
    try:
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer

        # MeshcatVisualizer 需要 GeometryModel（visual/collision），不是 Data
        visual_model = getattr(arm, "visual_model", None)
        collision_model = getattr(arm, "collision_model", None)
        if visual_model is None:
            raise RuntimeError(
                "arm 未加载几何模型；请用 load_geometry=True 构造 Arm（如 "
                "JoyArm(load_geometry=True)）。本次仅清空场景。"
            )
        viz = MeshcatVisualizer(
            arm.model,
            collision_model if collision_model is not None else pin.GeometryModel(),   # 无碰撞模型时用空模型
            visual_model,
        )
        viz.viewer = viewer                  # 绑定外部传入的 meshcat 查看器
        viz.loadViewerModel()                # 把 URDF/mesh 真正载入场景
    except Exception as e:
        # 回退：不做模型加载，仅清空场景（plot_points/plot_frame 仍可用）
        viewer["robot"].delete()
        import warnings

        warnings.warn(f"load_robot 未能加载 mesh 模型：{e}")


# ============================================================
# 点云 / 坐标系
# ============================================================
def plot_points(
    viewer,
    P: np.ndarray,
    name: str = "points",
    color=None,
    size: float = 0.01,
) -> None:
    """绘制 / 增量添加 3D 点云 ``(N,3)``（工作空间包络）。

    :param viewer: :func:`make_viewer` 返回的查看器。
    :param P: ``(N,3)`` 点坐标，单位 m。
    :param name: meshcat 场景节点名（同名重复写会覆盖）。
    :param color: ``(r,g,b,a)``，``r/g/b/a`` ∈ [0,1]；缺省为灰色。
    :param size: 点的大小，单位 m。
    """
    meshcat = _require_meshcat()
    P = np.asarray(P, dtype=float).reshape(-1, 3)   # 统一成 (N,3) 形状
    # meshcat 的 PointCloud 需要 (3,N) float32（转置 + 类型转换）
    pts = P.T.astype(np.float32)
    color = color if color is not None else (0.5, 0.5, 0.5, 1.0)   # 默认灰色、不透明
    color_arr = np.array(color, dtype=np.float32)
    pc = meshcat.PointCloud(pts, color_arr, size=size)
    viewer[name].set_object(pc)               # 写入场景节点（同名会覆盖）


def plot_frame(
    viewer,
    T: np.ndarray,
    name: str = "frame",
    scale: float = 0.2,
) -> None:
    """绘制 RGB 坐标系三轴（红=X / 绿=Y / 蓝=Z）。

    :param viewer: :func:`make_viewer` 返回的查看器。
    :param T: ``(4,4)`` 坐标系位姿。
    :param name: 节点名。
    :param scale: 三轴长度，单位 m。
    """
    meshcat = _require_meshcat()
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]               # 拆出旋转 R 和原点 p
    # 三段线段：原点 → 各轴端点（X 红 / Y 绿 / Z 蓝）
    for axis_idx, color in enumerate(["red", "green", "blue"]):
        end = p + scale * R[:, axis_idx]     # 轴端点 = 原点 + 轴长·该轴方向
        # stack 成 (2,3) 再转置成 (3,2)：meshcat.Line 要 (3,N) float32
        line_pts = np.stack([p, end], axis=0).T.astype(np.float32)
        line = meshcat.Line(line_pts, line_width=3.0)
        viewer[f"{name}/{axis_idx}"].set_object(line)   # 三条线分三个子节点


# ============================================================
# 一站式工作空间
# ============================================================
def plot_workspace(
    arm,
    N: int = 100_000,
    viewer=None,
    rng: Optional[np.random.Generator] = None,
):
    """一站式工作空间绘制（封装 chapter2_2.md §5.2 全流程）。

    流程：在软限位内采样 → 批量 FK 取末端位置 → meshcat 点云。

    :param arm: :class:`joyarm.arm.Arm` 实例（离线模式即可）。
    :param N: 采样点数，默认 10 万。
    :param viewer: 复用已有查看器；缺省则新建。
    :param rng: ``numpy.random.Generator``；缺省新建默认生成器。
    :return: ``(viewer, P)``，``P`` 为 ``(N,3)`` 末端位置点云。
    """
    from ..robotics.fkine import fkine

    if viewer is None:
        viewer = make_viewer()
    # 三步走：随机采样关节角 → 批量正运动学取末端位置 → 画点云
    Q = arm.rand_q(size=N, rng=rng)           # (N,n) 在软限位内均匀采样
    P = fkine(arm, Q, rep="pos")  # (N,3)     # 批量 FK，只取末端位置
    plot_points(viewer, P, name="workspace", size=0.008)
    return viewer, P


# ============================================================
# 轨迹预演
# ============================================================
def plot_trajectory(viewer, traj, arm=None) -> None:
    """轨迹预演：按轨迹对象逐帧更新机器人模型 / 末端坐标系。

    .. note::

        当前为占位骨架；``traj`` 由 :mod:`joyarm.robotics.trajectory` 提供
        （Ch5 实现）。在 trajectory 模块未落地前调用本函数会因
        ``traj`` 缺少必要字段而抛错，签名已就位。

    :param viewer: :func:`make_viewer` 返回的查看器。
    :param traj: :class:`joyarm.robotics.trajectory.Trajectory` 对象。
    :param arm: 可选，用于在 meshcat 中驱动机器人模型。
    """
    # 用 getattr 容错取轨迹对象的字段（trajectory 模块 Ch5 才落地，字段可能不全）
    q_seq = getattr(traj, "q", None)          # 关节空间轨迹：每帧关节角序列
    poses = getattr(traj, "poses", None)      # 笛卡尔空间轨迹：每帧末端位姿

    if q_seq is not None and arm is not None:
        # 关节空间轨迹：逐帧 FK → 末端坐标系
        from ..robotics.fkine import fkine

        Ts = fkine(arm, np.asarray(q_seq), rep="T")   # 批量算出每帧的末端 T
        for i, T in enumerate(Ts):
            plot_frame(viewer, T, name=f"traj/frame", scale=0.15)
    elif poses is not None:
        # 笛卡尔空间轨迹：位姿已直接给出，无需 FK
        for i, pose in enumerate(poses):
            T = getattr(pose, "T", pose)      # 兼容 Pose 对象或裸数组
            plot_frame(viewer, np.asarray(T), name=f"traj/frame", scale=0.15)
    else:
        raise ValueError(
            "plot_trajectory 需要 traj.q（配 arm）或 traj.poses；"
            "当前均缺失（trajectory 模块 Ch5 实现）。"
        )


# ============================================================
# matplotlib 备用
# ============================================================
def plot_points_mpl(P: np.ndarray, ax=None):
    """Matplotlib 3D 点云（无浏览器环境备用）。

    :param P: ``(N,3)`` 点坐标。
    :param ax: 已有 3D axes；缺省新建。
    :return: matplotlib axes。
    """
    import matplotlib.pyplot as plt

    P = np.asarray(P, dtype=float).reshape(-1, 3)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.5, c="steelblue", alpha=0.6)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Workspace point cloud")
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    return ax
