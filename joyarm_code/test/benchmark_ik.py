"""比较解析 IK 与 Pin 数值 IK 的端到端单次求解耗时。

运行：
    .venv/bin/python test/benchmark_ik.py
    .venv/bin/python test/benchmark_ik.py --count 10 --starts 10

随机生成可达目标，并为每个目标在 0～2 rad 邻域内生成多组 q0。
计时循环只包含 solve()，不包含目标生成、打印、保存和绘图。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from joyarm_core import AnalyticIkineSolver, FakeArm, PinIkineSolver


def make_cases(arm, count: int, seed: int, starts: int = 100):
    """生成可达目标及每个目标在 0～2 rad 邻域内的多个 q0。"""
    rng = np.random.default_rng(seed)
    q_min = arm.arm_limits_soft.q_min
    q_max = arm.arm_limits_soft.q_max

    for _ in range(count):
        q_known = rng.uniform(q_min, q_max)
        target = arm.fkine(q_known, "frame_end_tcp")
        for _ in range(starts):
            radius = rng.uniform(0.0, 2.0)
            lower = np.maximum(q_min, q_known - radius)
            upper = np.minimum(q_max, q_known + radius)
            q0 = rng.uniform(lower, upper)
            distance = float(np.max(np.abs(q_known - q0)))
            yield target, q0, distance


def benchmark(solver, arm, cases, *, name: str, **kwargs):
    """计时 solve()，统计成功率、耗时和累计外层迭代次数。"""
    success = 0
    elapsed = 0.0
    success_elapsed = 0.0
    distances = []
    times_us = []
    successes = []
    iterations = []

    for target, q0, distance in cases:
        t_start = perf_counter()
        result = solver.solve(arm, target, "frame_end_tcp", q0, **kwargs)
        solve_elapsed = perf_counter() - t_start

        elapsed += solve_elapsed
        success += int(result.success)
        if result.success:
            success_elapsed += solve_elapsed

        distances.append(distance)
        times_us.append(solve_elapsed * 1e6)
        successes.append(result.success)
        iterations.append(result.n_iter)

        if len(distances) % 10_000 == 0:
            print(f"已完成 {len(distances)} 次求解", flush=True)

    count = len(distances)
    if count == 0:
        raise ValueError("benchmark_ik.py - benchmark：没有测试样本")

    print(f"{name}：")
    print(f"  成功数：{success}/{count}")
    print(f"  总耗时：{elapsed:.6f} s")
    print(f"  平均单次：{elapsed / count * 1e6:.2f} us")
    if success:
        print(f"  仅成功解平均单次：{success_elapsed / success * 1e6:.2f} us")

    total_iterations = sum(iterations)
    if total_iterations > 0:
        print(f"  平均每次迭代耗时：{elapsed * 1e6 / total_iterations:.2f} us")

    return (
        np.asarray(distances),
        np.asarray(times_us),
        np.asarray(successes, dtype=bool),
        np.asarray(iterations, dtype=int),
    )


def add_success_statistics(ax, distances, times_us, successes):
    """按 0.05 rad 分段绘制成功样本的平均值和中位数。"""
    edges = np.linspace(0.0, 2.0, 41)
    centers = (edges[:-1] + edges[1:]) / 2
    mean_times = np.full(len(centers), np.nan)
    median_times = np.full(len(centers), np.nan)

    for i in range(len(centers)):
        right = (
            distances <= edges[i + 1]
            if i == len(centers) - 1
            else distances < edges[i + 1]
        )
        mask = (distances >= edges[i]) & right & successes
        if np.any(mask):
            mean_times[i] = times_us[mask].mean()
            median_times[i] = np.median(times_us[mask])

    ax.plot(
        centers,
        mean_times,
        color="darkorange",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Mean (successful solves only)",
        zorder=6,
    )
    ax.plot(
        centers,
        median_times,
        color="green",
        linewidth=2,
        marker="s",
        markersize=3,
        label="Median (successful solves only)",
        zorder=7,
    )


def set_log_time_axis(ax, times_us):
    """设置以 10 为底的耗时对数轴，并标出 450 us。"""
    ticks = [
        20, 30, 50, 100, 200, 300, 450, 700,
        1000, 2000, 3000, 5000, 10000, 15000,
        20000, 30000, 40000, 60000, 100000,
    ]
    y_min = min(float(times_us.min()), 450.0) * 0.8
    y_max = max(float(times_us.max()), 450.0) * 1.2
    visible_ticks = [value for value in ticks if y_min <= value <= y_max]

    ax.set_yscale("log", base=10)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(FixedLocator(visible_ticks))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    ax.yaxis.set_minor_formatter(NullFormatter())


def main():
    parser = argparse.ArgumentParser(
        description="比较解析 IK 与 Pin IK，并分析初值距离、耗时和迭代次数。",
    )
    parser.add_argument("--count", type=int, default=1000, help="目标数量")
    parser.add_argument("--starts", type=int, default=100, help="每个目标的初值数量")
    parser.add_argument("--seed", type=int, default=20261009, help="随机种子")
    parser.add_argument("--tol", type=float, default=1e-4, help="Pin IK 收敛容差")
    parser.add_argument("--iters", type=int, default=200, help="Pin IK 最大迭代次数")
    args = parser.parse_args()

    if min(args.count, args.starts, args.iters) <= 0:
        parser.error("count、starts、iters 必须为正整数")
    if not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("tol 必须为有限正数")

    arm = FakeArm("joyarm_dm")
    print(
        f"目标数量：{args.count}；每目标初值：{args.starts}；"
        f"随机种子：{args.seed}"
    )
    print("计时范围：各求解器的 solve()；不含采样和目标生成。")

    _, analytic_times, analytic_successes, _ = benchmark(
        AnalyticIkineSolver(),
        arm,
        make_cases(arm, args.count, args.seed, args.starts),
        name="解析 IK",
    )
    distances, pin_times, pin_successes, pin_iterations = benchmark(
        PinIkineSolver(),
        arm,
        make_cases(arm, args.count, args.seed, args.starts),
        name="Pin 数值 IK",
        tol=args.tol,
        iters=args.iters,
    )

    print(
        f"数值 IK / 解析 IK 平均耗时比："
        f"{pin_times.mean() / analytic_times.mean():.2f}x"
    )
    if not analytic_successes.all() or not pin_successes.all():
        print("注意：存在求解失败；平均耗时包含失败尝试，需结合成功率解读。")

    output = Path(__file__).resolve().parents[2] / "output" / "ik_distance"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "samples.npz",
        distances=distances,
        pin_times=pin_times,
        pin_successes=pin_successes,
        pin_iterations=pin_iterations,
        analytic_times=analytic_times,
        analytic_successes=analytic_successes,
        count=args.count,
        starts=args.starts,
        seed=args.seed,
        tol=args.tol,
        iters=args.iters,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        distances[pin_successes],
        pin_times[pin_successes],
        s=3,
        alpha=0.3,
        color="tab:blue",
        label="Pin Success",
    )
    ax.scatter(
        distances[~pin_successes],
        pin_times[~pin_successes],
        s=3,
        alpha=0.3,
        color="tab:red",
        label="Pin Failure",
    )
    add_success_statistics(ax, distances, pin_times, pin_successes)
    ax.axhline(
        y=450,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="450 us reference",
        zorder=5,
    )
    ax.set_xlabel("Max joint angle difference |q - q0| (rad)")
    ax.set_ylabel("Pin IK solve time (us, log scale)")
    ax.set_title(
        f"Pin IK: {args.count} targets x {args.starts} starts\n"
        f"tol={args.tol:g}, success rate={pin_successes.mean():.1%}"
    )
    ax.set_xlim(0, 2)
    set_log_time_axis(ax, pin_times)
    ax.grid(alpha=0.3, which="major")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(output / "ik_distance.png", dpi=200)

    fig_iter, ax_iter = plt.subplots(figsize=(10, 6))
    scatter = ax_iter.scatter(
        distances,
        pin_times,
        c=pin_iterations,
        s=3,
        alpha=0.3,
        cmap="viridis",
    )
    ax_iter.axhline(
        y=450,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="450 us reference",
    )
    ax_iter.set_xlabel("Max joint angle difference |q - q0| (rad)")
    ax_iter.set_ylabel("Pin IK solve time (us, log scale)")
    ax_iter.set_title(
        f"Pin IK iteration count: {args.count} targets x {args.starts} starts"
    )
    ax_iter.set_xlim(0, 2)
    set_log_time_axis(ax_iter, pin_times)
    ax_iter.grid(alpha=0.3, which="major")
    ax_iter.legend()
    fig_iter.colorbar(
        scatter,
        ax=ax_iter,
        label="Accumulated iteration count",
    )
    fig_iter.tight_layout()
    fig_iter.savefig(output / "ik_distance_iterations.png", dpi=200)

    print(f"数据和图片已保存到：{output}")
    plt.show()


if __name__ == "__main__":
    main()
