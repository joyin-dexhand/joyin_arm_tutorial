"""
==============================================================================
JoyArm 工作空间绘制与分析
==============================================================================
【功能概要】
    用蒙特卡洛法求机械臂的可达工作空间并绘图分析（离线计算，无需连接硬件）：
      1. 在关节硬限位内均匀随机采样 N 组关节角
      2. 逐组调用 arm.solve_fkine(q) 求末端位置，点云逐批刷新
      3. 三幅子图：3D 视图（点云 + home 位形机械臂骨架 + 末端坐标系）、俯视图 X-Y、侧视图 X-Z
      4. 终端打印分析数据：xyz 范围、最大臂展、最近距离、体积、安装面以下占比、求解耗时
      5. 改顶部常量的数字即可调参：采样数、批数、随机种子、体素边长、求解器、显示点数上限

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate
    # 若 Linux 缺少 libxcb-cursor.so.0 ，则先安装：
    sudo apt-get install -y libxcb-cursor0

    # 2. 运行本脚本
    python ./chapt/chapt2_workspace.py <型号>  # 如：rebot_arm_b601_dm

    # 注：点色表示末端到基座原点的距离（viridis 色图：紫色近、黄色远）；关闭绘图窗口后程序结束。
    #     体积由体素法估算，是随采样数增多仍缓慢变大的近似值，详见第二章 §5.3.3 与 §5.5。

==============================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装时也可直接运行
from joyarm_core import JointLimits, joyArm, listArm, rand_within_limits

# ----------------------------------------------------------------------------
# 可调参数（改数字即可）
# ----------------------------------------------------------------------------
N_SAMPLES = 200_000          # 采样组数
N_BATCHES = 10               # 分批数：每求完一批刷新一次画面
SEED = 0                     # 随机种子（结果可复现）
VOXEL = 0.05                 # 体素边长 (m)
SOLVER = "pin_fkine_solver"  # 正运动学求解器注册名（可改为 "mdh_fkine_solver"）
SHOW_MAX = 20_000            # 画面最多显示的点数（统计仍用全部点；点太多重画很慢）
CJK_FONTS = ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC")  # 中文字体候选（按优先级）


def parse_model() -> str:
    """读取命令行的型号参数；个数不对时打印用法与可用型号，以退出码 1 结束。"""
    if len(sys.argv) != 2:                                  # 必须恰好带 1 个型号参数
        print("用法：python ./chapt/chapt2_workspace.py <型号>")
        listArm()                                           # 列出全部可用型号
        sys.exit(1)
    return sys.argv[1].strip()


def main() -> None:
    """创建对象 → 采样 → 画布与机械臂 → 逐批求解与刷新 → 分析数据。"""
    model = parse_model()

    # ① 创建对象：只加载该型号的配置与模型，不连接硬件
    print(f"① 创建 JoyArm 对象：{model}（离线，不连接硬件）")
    arm = joyArm(model)

    # ② 采样：硬限位内均匀采样；连续旋转关节的限位是 ±inf，截到 ±π（转满一圈已覆盖全部位置）
    lim = arm.joint_limits_hard_arm
    q_min = np.where(np.isinf(lim.q_min), -np.pi, lim.q_min)
    q_max = np.where(np.isinf(lim.q_max), np.pi, lim.q_max)
    Q = rand_within_limits(JointLimits(q_min=q_min, q_max=q_max), size=N_SAMPLES, rng=np.random.default_rng(SEED))
    print(f"② 硬限位内采样 {N_SAMPLES} 组关节角：q_min = {q_min.round(2)}，q_max = {q_max.round(2)} (rad)")

    # ③ 画布与机械臂：一行三列子图（3D | 俯视 X-Y | 侧视 X-Z），三轴等比例
    print("③ 绘制画布与 home 位形下的机械臂")
    # 中文字体：只用本机已安装的候选（避免缺字告警与豆腐块），DejaVu Sans 兜底
    installed = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [f for f in CJK_FONTS if f in installed] + ["DejaVu Sans"]
    fig = plt.figure(figsize=(16, 5.5))
    fig.suptitle(f"JoyArm 工作空间（型号 {model}，求解器 {SOLVER}，采样 {N_SAMPLES} 组）")
    ax3d = fig.add_subplot(1, 3, 1, projection="3d", computed_zorder=False)  # 关闭自动深度排序，按 zorder 叠放
    ax_xy = fig.add_subplot(1, 3, 2)
    ax_xz = fig.add_subplot(1, 3, 3)
    ax3d.set(title="工作空间点云（点色：到基座距离）", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax_xy.set(title="俯视图（X-Y）", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax_xz.set(title="侧视图（X-Z）", xlabel="x (m)", ylabel="z (m)", aspect="equal")

    # 骨架：基座原点 → 各关节帧 → 末端帧连成折线；MDH 版只支持末端帧，所以骨架固定用 pin 版求
    arm.set_fkine_solver("pin_fkine_solver")
    frames = list(arm.get_joint_names_arm()) + [arm.get_config()["joyarm"]["frame_end"]]
    skeleton = np.array([[0.0, 0.0, 0.0]] + [arm.solve_fkine(arm.home_arm, f).position for f in frames])
    x, y, z = skeleton.T
    arm_style = dict(color="k", marker="o", markersize=3, linewidth=2, zorder=3)  # zorder 比点云大，画在上层
    ax3d.plot(x, y, z, label="机械臂（home 位形）", **arm_style)
    ax_xy.plot(x, y, **arm_style)
    ax_xz.plot(x, z, **arm_style)

    if not arm.set_fkine_solver(SOLVER):                    # 切换到要使用的求解器
        print(f"求解器 {SOLVER} 不可用，可选：{arm.get_fkine_solvers()}")
        sys.exit(1)
    T = arm.solve_fkine(arm.home_arm).T                     # home 位形下末端的位姿矩阵 (4×4)
    o, R = T[:3, 3], T[:3, :3]                              # 末端原点位置、旋转矩阵
    for i, (color, name) in enumerate(zip("rgb", "XYZ")):   # R 的第 i 列 = 末端第 i 根轴在基座系下的方向
        ax3d.quiver(*o, *(0.1 * R[:, i]), color=color, linewidth=2, zorder=3, label=f"末端 {name} 轴")  # 长 0.1 m
    ax3d.legend(loc="upper left", fontsize=8)

    # ④ 逐批求解与刷新：每求完一批，重画一次点云并估算一次体积
    print(f"④ 分 {N_BATCHES} 批求解末端位置（求解器 {SOLVER}）")
    plt.ion()                                               # 交互模式：显示窗口后不阻塞，边算边画
    plt.show()
    P = np.empty((0, 3))                                    # 点云：已求出的全部末端位置
    dots = []                                               # 上一批画的散点
    t_solve = 0.0                                           # 累计求解耗时 (s)
    for batch in np.array_split(Q, N_BATCHES):              # 把 N 组关节角均分成 N_BATCHES 批
        t0 = time.perf_counter()
        pts = [arm.solve_fkine(q).position for q in batch]  # 正运动学：关节角 → 末端位置
        t_solve += time.perf_counter() - t0                 # 只计求解，不计绘图
        P = np.vstack([P, pts])

        show = P[::max(1, len(P) // SHOW_MAX)]              # 等间隔抽点显示（统计仍用全部点）
        for dot in dots:
            dot.remove()
        x, y, z = show.T
        dot_style = dict(c=np.linalg.norm(show, axis=1), cmap="viridis", s=1, alpha=0.3)  # 点色 = 到基座原点的距离
        dots = [ax3d.scatter(x, y, z, **dot_style), ax_xy.scatter(x, y, **dot_style), ax_xz.scatter(x, z, **dot_style)]
        ax3d.set_aspect("equal")                            # 点云范围变了，重设 3D 三轴等比例

        # 体素法估体积：把空间切成边长 VOXEL 的小方格，体积 ≈ 含点的方格数 × 单格体积
        vol = len(np.unique(np.floor(P / VOXEL).astype(int), axis=0)) * VOXEL**3
        print(f"已求解 {len(P)}/{N_SAMPLES} 组，体积约 {vol:.3f} m³")
        plt.pause(0.01)                                     # 让窗口刷新画面

    # ⑤ 分析数据
    r = np.linalg.norm(P, axis=1)                           # 各点到基座原点的距离
    print(f"\n⑤ 工作空间分析：型号 {model}｜求解器 {SOLVER}｜硬限位内采样 N = {N_SAMPLES} 组")
    print("-" * 72)
    print("  项目              数值")
    print("-" * 72)
    for i, axis in enumerate("xyz"):
        print(f"  {axis} 范围 (m)        [{P[:, i].min():+.3f}, {P[:, i].max():+.3f}]")
    print(f"  最大臂展 (m)      {r.max():.3f}（末端到基座原点的最远距离）")
    print(f"  最近距离 (m)      {r.min():.3f}（末端到基座原点的最近距离）")
    print(f"  体积 (m³)         {vol:.3f}（体素法，边长 {VOXEL} m，近似值）")
    print(f"  安装面以下点占比  {np.mean(P[:, 2] < 0):.1%}（z < 0：低于安装面，台面安装时不可达）")
    print(f"  求解耗时          {t_solve:.2f} s（单次 {t_solve / N_SAMPLES * 1e6:.1f} µs）")
    print("-" * 72)

    plt.ioff()                                              # 关闭交互模式：show 阻塞，窗口保持到手动关闭
    plt.show()
    arm.close()


if __name__ == "__main__":
    main()
