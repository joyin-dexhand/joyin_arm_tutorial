"""比较解析 IK 与 Pin 数值 IK 的端到端单次求解耗时。

运行：
    .venv/bin/python test/benchmark_ik.py
    .venv/bin/python test/benchmark_ik.py --count 1000

计时前用 Pin FK 生成同一批可达目标，并为两种求解器准备同一批 q0。
计时循环只包含 solve()，不包含目标生成、打印和 FK 回代验证。
"""
from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from joyarm_core import AnalyticIkineSolver, FakeArm, PinIkineSolver


def _far_q0(rng, q_known: np.ndarray, q_min: np.ndarray,
            q_max: np.ndarray) -> np.ndarray:
    """采样与 q_known 至少相距 1.5 rad 的参考角，模拟首次规划。"""
    for _ in range(100):
        q0 = rng.uniform(q_min, q_max)
        if np.linalg.norm(q0 - q_known) >= 1.5:
            return q0
    raise RuntimeError("benchmark_ik.py - _far_q0：无法采样到足够远的 q0")


def make_cases(arm, count: int, seed: int, scenario: str):
    """生成指定场景的可达目标及两种 IK 共用的 q0。"""
    rng = np.random.default_rng(seed)
    q_min = arm.arm_limits_soft.q_min
    q_max = arm.arm_limits_soft.q_max
    mdh = np.asarray(arm.get_config()["joyarm"]["arm_mdh"], dtype=float)

    # theta3 = 0 是肘部两分支合并的位置；q3 = theta3 - theta30。
    q3_near_elbow = np.clip(-mdh[2, 3] + 1e-4, q_min[2] + 1e-4, q_max[2] - 1e-4)
    # theta5 = pi 是腕部奇异位置。当前软限位无法精确到达时，取限位内最近点。
    q5_near_wrist = np.clip(
        np.pi - mdh[4, 3],
        q_min[4] + 1e-4,
        q_max[4] - 1e-4,
    )

    cases = []
    for _ in range(count):
        q_known = rng.uniform(q_min, q_max)
        if scenario == "singular":
            q_known[2] = q3_near_elbow
            q_known[4] = q5_near_wrist
        target = arm.fkine(q_known, "frame_end_tcp")
        if scenario in ("near", "singular"):
            q0 = np.clip(
                q_known + rng.uniform(-0.15, 0.15, arm.n_arm),
                q_min,
                q_max,
            )
        else:
            q0 = _far_q0(rng, q_known, q_min, q_max)
        cases.append((target, q0))
    return cases


def benchmark(solver, arm, cases, *, name: str, **kwargs):
    """计时 solver.solve，并分别统计全部尝试和成功解的耗时。"""
    success = 0
    elapsed = 0.0
    success_elapsed = 0.0
    for target, q0 in cases:
        t_start = perf_counter()
        result = solver.solve(arm, target, "frame_end_tcp", q0, **kwargs)
        solve_elapsed = perf_counter() - t_start
        elapsed += solve_elapsed
        success += int(result.success)
        if result.success:
            success_elapsed += solve_elapsed

    count = len(cases)
    print(f"{name}：")
    print(f"  成功数：{success}/{count}")
    print(f"  总耗时：{elapsed:.6f} s")
    print(f"  平均单次：{elapsed / count * 1e6:.2f} us")
    if success:
        print(f"  仅成功解平均单次：{success_elapsed / success * 1e6:.2f} us")
    return elapsed, success


def main():
    parser = argparse.ArgumentParser(
        description="同一批可达目标下，比较解析 IK 与 Pin 数值 IK 的求解耗时。",
    )
    parser.add_argument("--count", type=int, default=10_000, help="目标数量")
    parser.add_argument("--seed", type=int, default=20260930, help="随机种子")
    parser.add_argument("--tol", type=float, default=1e-3, help="Pin IK 收敛容差")
    parser.add_argument("--iters", type=int, default=200, help="Pin IK 最大迭代次数")
    parser.add_argument(
        "--scenario",
        choices=("near", "far", "singular", "all"),
        default="all",
        help="near=q0 接近目标；far=q0 远离目标；singular=近肘/腕奇异；all=依次运行三者",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count 必须为正整数")

    arm = FakeArm("joyarm_dm")
    scenarios = ("near", "far", "singular") if args.scenario == "all" else (args.scenario,)
    labels = {
        "near": "q0 接近目标（连续轨迹）",
        "far": "q0 远离目标（首次规划）",
        "singular": "近肘部／腕部奇异",
    }
    print(f"每个场景目标数量：{args.count}；随机种子：{args.seed}")
    print("计时范围：各求解器的 solve()；不含目标生成和 FK 回代。")
    for index, scenario in enumerate(scenarios):
        # 每种场景偏移种子，既可复现，也不会跨场景复用同一批目标。
        cases = make_cases(arm, args.count, args.seed + index, scenario)
        print(f"\n场景：{labels[scenario]}")
        analytic_time, analytic_success = benchmark(
            AnalyticIkineSolver(),
            arm,
            cases,
            name="解析 IK",
        )
        pin_time, pin_success = benchmark(
            PinIkineSolver(),
            arm,
            cases,
            name="Pin 数值 IK",
            tol=args.tol,
            iters=args.iters,
        )
        print(f"数值 IK / 解析 IK 平均耗时比：{pin_time / analytic_time:.2f}x")
        if analytic_success != args.count or pin_success != args.count:
            print("注意：存在求解失败；平均耗时包含失败尝试，需结合成功率解读。")


if __name__ == "__main__":
    main()
