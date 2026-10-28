"""
==============================================================================
JoyArm 工作空间蒙特卡洛采样与可视化  (第二章 §5)
==============================================================================
【功能概要】
    用蒙特卡洛采样近似 JoyArm 的可达工作空间，并动态可视化：
      1. 离线创建 JoyArm（加载型号 URDF，无需连接硬件）
      2. 在关节硬限位内均匀随机采样 N 组关节角
      3. 对每组关节角求末端位置（Pinocchio 正运动学，同第二章 §4.3.2）
      4. MeshCat 3D 点云逐批动态刷新，点色按到基座的距离渐变
    采样完成后在浏览器打开提示的 URL，旋转/缩放观察点云包络。

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 2. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（在 joyarm_code/ 目录下）
    python chapt/chapt2_workspace.py                     # 默认 1 万采样点
    python chapt/chapt2_workspace.py --n 100000          # 更精细的包络
    python chapt/chapt2_workspace.py --headless          # 仅采样不保持可视化（冒烟测试用）

【命令行参数】
    --n        采样点数（默认 10000；建议先 1e4 跑通，再 1e5~1e6 取精细包络）
    --batch    每批添加的点数（默认 2000，决定刷新粒度）
    --seed     随机种子（默认 0，结果可复现）
    --headless 采样完成后直接退出，不启动 MeshCat 等待查看
==============================================================================
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装时也可直接运行
from joyarm_core import joyArm, rand_within_limits  # noqa: E402

POINT_SIZE = 0.003   # MeshCat 点大小（m）


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JoyArm 工作空间蒙特卡洛采样与可视化")
    p.add_argument("--n", type=int, default=10_000, help="采样点数")
    p.add_argument("--batch", type=int, default=2_000, help="每批添加的点数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--headless", action="store_true", help="仅采样，不保持可视化")
    return p.parse_args()


def fk_positions(model, data, fid, Q: np.ndarray) -> np.ndarray:
    """逐组求末端位置：Pinocchio 正运动学（用 JoyArm 加载的 URDF 缩减模型）。

    :param model: pin.Model（arm.pin_model，nq = 本体关节数）。
    :param data: pin.Data（由调用方自建，复用同一对象逐组更新）。
    :param fid: 末端帧 id（model.getFrameId(config joyarm.frame_end)）。
    :param Q: (N, n) 关节角批量。
    :return: (N, 3) 末端位置（基座系）。
    """
    out = np.empty((len(Q), 3))
    for i, q in enumerate(Q):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, fid)
        out[i] = data.oMf[fid].translation
    return out


def main() -> None:
    args = parse_args()

    try:
        arm = joyArm("rebot_arm_b601_dm")
    except Exception as e:   # 工厂校验失败抛异常
        raise SystemExit(f"JoyArm 创建失败：{e}")
    print(f"模型就绪：{arm}\n")

    # Pinocchio 模型与末端帧：data 自建复用、帧名读 config（joyarm.frame_end）
    model = arm.pin_model
    data = model.createData()
    fid = model.getFrameId(arm.get_config()["joyarm"]["frame_end"])

    # 硬限位内均匀采样（与库一致的口径：rand_q_arm 即委托 rand_within_limits；
    # 此处直接带 size/rng 一次采 N 组，保证 --seed 可复现）
    Q = rand_within_limits(arm.joint_limits_hard_arm, size=args.n, rng=np.random.default_rng(args.seed))
    print(f"硬限位内采样 {args.n} 组关节角，每批 {args.batch} 组……")

    vis = None
    if not args.headless:
        import meshcat
        from matplotlib import colormaps
        vis = meshcat.Visualizer()
        cmap = colormaps["viridis"]

    positions = np.empty((0, 3))
    t0 = time.time()
    for start in range(0, args.n, args.batch):
        positions = np.vstack([positions, fk_positions(model, data, fid, Q[start:start + args.batch])])
        if vis is not None:
            r = np.linalg.norm(positions, axis=1)            # 点色 = 到基座距离
            colors = (cmap((r - r.min()) / max(np.ptp(r), 1e-9))[..., :3] * 255).astype(np.uint8)
            vis["workspace"].set_object(meshcat.geometry.PointCloud(
                positions=positions.T.astype(np.float32),
                colors=colors.T, size=POINT_SIZE))
        print(f"\r已求解 {len(positions)}/{args.n} 组（{time.time() - t0:.1f}s）", end="")
    print()

    radius = np.linalg.norm(positions, axis=1)
    print(f"完成：最大臂展 {radius.max():.3f} m，高度范围 "
          f"[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] m")

    if vis is not None:
        print(f"在浏览器打开查看工作空间点云：{vis.url()}")
        print("Ctrl+C 退出")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
