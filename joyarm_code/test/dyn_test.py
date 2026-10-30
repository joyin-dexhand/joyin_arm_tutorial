"""
==============================================================================
dynamics 动力学测试分析  (test 离线测试)
==============================================================================
【功能概要】
    离线测试脚本（无需硬件）：围绕 DynamicsSolver 基类的四个接口，测试分析动力学求解器：
      1. 质量矩阵 M(q)：在 ≥5 组随机位形上验证对称性（max|M−Mᵀ|）与正定性（λ_min > 0）
      2. RNEA 分解一致性：solve_idyn(q,dq,ddq) 与 M·ddq + C·dq + G 分项和的最大误差表
      3. 静止特例：dq=ddq=0 时逆动力学退化为重力项（idyn == G），附重力符号口径说明
      4. home / zero / rand 三种位形的重力力矩对比表
      5. 末尾汇总上述检查的分析结论表
    全程经 joyArm 数学门面调用当前激活的 dynamics 求解器（pin_dynamics_solver）。

【原理简介】
    刚体动力学把"机械臂怎么动"与"关节需要输出多大力矩"联系起来，核心是运动方程：
        τ = M(q)·ddq + C(q,dq)·dq + G(q)
    其中 M(q) 是质量（惯量）矩阵，刻画关节惯性，理论上恒对称正定；C·dq 是科氏/向心力
    矩项；G(q) 是重力力矩项。RNEA（递推牛顿-欧拉法）不分项、直接由 (q,dq,ddq) 递推出
    总力矩 τ——"分项求和"与"RNEA 直算"两路结果一致，即可互相印证求解器实现正确。
    静止特例 dq=ddq=0 时方程退化为 τ = G；重力口径取 G = +∂U/∂q（U 为重力势能），
    即机械臂静态悬停时电机必须输出的补偿力矩。

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 2. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（离线脚本，无需连接机械臂硬件）
    python test/dyn_test.py

【预期输出】
    ①~⑥ 六节分析：M 对称/正定表、RNEA 分解误差表、静止特例表、重力对比表、结论表；
    收尾时「arm 未连接……豁免跳过」类日志属正常提示，非报错。
==============================================================================
"""

from pathlib import Path                                    # 路径处理：定位 joyarm_code/ 目录
import sys                                                  # 解释器操作：临时插入模块搜索路径
import unicodedata                                          # 判定字符终端显示宽度（中文占 2 列），用于表格对齐

import numpy as np                                          # 数值计算：向量 / 矩阵 / 特征值

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装 joyarm_core 时，把 joyarm_code/ 加入搜索路径
from joyarm_core import joyArm                              # noqa: E402  机械臂主类（离线构造即可做动力学计算）

N_RAND = 6                                                  # 随机位形组数（任务要求 ≥5，取 6 更充分）
SYM_TOL = 1e-12                                             # 对称性容差：max|M−Mᵀ| 上限（实现显式对称化，应精确为 0）
PD_TOL = 0.0                                                # 正定性阈值：最小特征值须严格大于 0
RNEA_TOL = 1e-8                                             # RNEA 分解一致性容差（N·m，机器精度量级）
STATIC_TOL = 1e-12                                          # 静止特例容差（N·m，同一路 RNEA 算法，应精确相等）


def disp_width(text: str) -> int:
    """计算字符串在终端的显示宽度：全角字符（中文等）按 2 列、半角按 1 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)  # 逐字符累计显示列数


def pad(text: str, width: int) -> str:
    """按显示宽度右侧补空格到指定列宽：保证含中文表头的表格列对齐。"""
    return text + " " * max(0, width - disp_width(text))    # 不足列宽补空格；已超宽则原样返回


def print_table(headers: list, rows: list) -> None:
    """打印列对齐表格：列宽取表头与所有单元格显示宽度的最大值，列间留两个空格。

    :param headers: 表头字符串列表（每列一个）。
    :param rows: 数据行列表（每行为与表头等长的字符串列表）。
    """
    widths = [max(disp_width(h), *(disp_width(r[i]) for r in rows)) for i, h in enumerate(headers)]  # 逐列求最大显示宽度
    print("  ".join(pad(h, w) for h, w in zip(headers, widths)).rstrip())  # 表头行（去掉行尾多余空格）
    print("  ".join("-" * w for w in widths))               # 表头下的分隔线行
    for row in rows:                                        # 逐行打印数据
        print("  ".join(pad(c, w) for c, w in zip(row, widths)).rstrip())  # 单元格按列宽对齐（去行尾空格）


def section_setup(arm) -> None:
    """① 打印模型与 dynamics 域求解器信息：确认本脚本测试的是哪一个求解器实现。"""
    print("=" * 78)                                        # 大节分隔线
    print("① 模型与 dynamics 域求解器（离线创建，不连接硬件）")   # 大节标题
    print(f"   关节数 n_arm = {arm.n_arm}；关节名（顺序 = q 向量顺序）：{arm.get_joint_names_arm('all')}")   # 关节数与关节名
    print(f"   dynamics 域已加载求解器：{arm.get_dyn_solvers()}；当前激活：{arm.get_activated_dyn_solver()}")   # 域成员与激活
    print("   数学门面 → DynamicsSolver 基类接口：solve_idyn→idyn、get_mass_matrix→mass_matrix、")   # 门面转发关系一
    print("   get_coriolis→coriolis、get_gravity→gravity（门面把调用转发给激活的求解器）")   # 门面转发关系二


def section_mass_matrix(arm, qs: list) -> dict:
    """② 质量矩阵检查：对每组随机位形 q 验证 M(q) 的对称性与正定性。

    :param arm: 离线 joyArm 对象（数学门面入口）。
    :param qs: 随机关节角列表，每组 `(n_arm,)`。
    :return: 统计字典：最差对称偏差、最小特征值、两项检查的通过组数与总结论。
    """
    print("=" * 78)                                        # 大节分隔线
    print(f"② 质量矩阵 M(q)：对称性（max|M−Mᵀ| ≤ {SYM_TOL:.0e}）与正定性（λ_min > {PD_TOL:.0f}），共 {len(qs)} 组随机位形")
    print("-" * 78)                                        # 小节分隔线
    headers = ["组号", "max|M−Mᵀ|", "λ_min(M)", "λ_max(M)", "对称", "正定"]  # 表头（中文按 2 列宽参与对齐）
    rows = []                                              # 数据行收集器
    max_asym, min_eig = 0.0, float("inf")                  # 跨组统计量初值（取最差值）
    n_sym = n_pd = 0                                       # 对称 / 正定通过组数计数器
    for i, q in enumerate(qs, start=1):                    # 逐组随机位形检查
        M = arm.get_mass_matrix(q)                         # 门面 → mass_matrix：CRBA 算 M(q)
        asym = float(np.max(np.abs(M - M.T)))              # 对称性偏差：M 与自身转置逐元素差的最大绝对值
        eigs = np.linalg.eigvalsh(M)                       # 对称矩阵特征值（升序返回），供正定性判定
        sym_ok = asym <= SYM_TOL                           # 对称判定：偏差不超过容差
        pd_ok = eigs[0] > PD_TOL                           # 正定判定：最小特征值严格为正
        rows.append([str(i), f"{asym:.2e}", f"{eigs[0]:.4f}", f"{eigs[-1]:.4f}",
                     "✓" if sym_ok else "✗", "✓" if pd_ok else "✗"])  # 本组结果入表
        max_asym = max(max_asym, asym)                     # 更新跨组最差对称偏差
        min_eig = min(min_eig, float(eigs[0]))             # 更新跨组最小特征值
        n_sym += int(sym_ok)                               # 对称通过组数累加
        n_pd += int(pd_ok)                                 # 正定通过组数累加
    print_table(headers, rows)                             # 打印结果表
    print("-" * 78)                                        # 小节分隔线
    print(f"小结：{n_sym}/{len(qs)} 组对称（最差 max|M−Mᵀ| = {max_asym:.2e}）；"
          f"{n_pd}/{len(qs)} 组正定（最小 λ_min = {min_eig:.4f}）")
    return {"max_asym": max_asym, "min_eig": min_eig,      # 统计值供结论表引用
            "pass_sym": n_sym == len(qs), "pass_pd": n_pd == len(qs)}  # 两项检查各自总结论


def section_rnea(arm, qs: list, rng) -> dict:
    """③ RNEA 分解一致性：逆动力学直算 τ 与分项和 M·ddq + C·dq + G 的最大误差。

    :param arm: 离线 joyArm 对象（数学门面入口）。
    :param qs: 随机关节角列表，每组 `(n_arm,)`（与 ② 同一批位形）。
    :param rng: 已定种子的随机数发生器（采样 dq/ddq，保证结果可复现）。
    :return: 统计字典：最大绝对 / 相对误差、通过组数与总结论。
    """
    print("=" * 78)                                        # 大节分隔线
    print("③ RNEA 分解一致性：solve_idyn(q,dq,ddq) 对比 M(q)·ddq + C(q,dq)·dq + G(q)")
    print("-" * 78)                                        # 小节分隔线
    print("   口径说明：get_coriolis 返回科氏/向心力矩向量 C(q,dq)·dq（形状 (n,)、单位 N·m），")   # 口径第一句
    print("   而非 C 矩阵本身（依据 joyarm.py get_coriolis docstring 与 PinDynamicsSolver")   # 口径依据
    print("   实现：nonLinearEffects 结果减重力项），故分项和直接三项相加即可。")   # 口径结论
    print("-" * 78)                                        # 小节分隔线
    headers = ["组号", "max|τ_RNEA| (N·m)", "max|τ_分项和| (N·m)", "max|Δτ| (N·m)", "相对误差", "判定"]
    rows = []                                              # 数据行收集器
    max_err = max_rel = 0.0                                # 跨组最大绝对 / 相对误差初值
    n_ok = 0                                               # 容差内一致组数计数器
    for i, q in enumerate(qs, start=1):                    # 逐组随机位形检查
        dq = rng.uniform(-2.0, 2.0, arm.n_arm)             # 随机关节速度 (rad/s)：幅度取常见中速，结论对任意值均应成立
        ddq = rng.uniform(-5.0, 5.0, arm.n_arm)            # 随机关节加速度 (rad/s²)：同上
        tau_rnea = arm.solve_idyn(q, dq, ddq)              # RNEA 直算：一次递推得总力矩 τ
        tau_sum = (arm.get_mass_matrix(q) @ ddq            # 分项一：惯性项 M(q)·ddq
                   + arm.get_coriolis(q, dq)               # 分项二：科氏/向心项 C·dq（向量口径，直接相加）
                   + arm.get_gravity(q))                   # 分项三：重力项 G(q)
        err = float(np.max(np.abs(tau_rnea - tau_sum)))    # 两路结果的最大绝对误差
        rel = err / max(float(np.max(np.abs(tau_rnea))), 1e-12)  # 相对误差（对最大力矩幅值，防零除）
        ok = err <= RNEA_TOL                               # 一致性判定：误差不超过容差
        rows.append([str(i), f"{np.max(np.abs(tau_rnea)):.3f}", f"{np.max(np.abs(tau_sum)):.3f}",
                     f"{err:.2e}", f"{rel:.2e}", "✓" if ok else "✗"])  # 本组结果入表
        max_err = max(max_err, err)                        # 更新跨组最大绝对误差
        max_rel = max(max_rel, rel)                        # 更新跨组最大相对误差
        n_ok += int(ok)                                    # 通过组数累加
    print_table(headers, rows)                             # 打印结果表
    print("-" * 78)                                        # 小节分隔线
    print(f"小结：{n_ok}/{len(qs)} 组在容差 {RNEA_TOL:.0e} N·m 内一致"
          f"（最大绝对误差 {max_err:.2e} N·m，最大相对误差 {max_rel:.2e}）")
    return {"max_err": max_err, "max_rel": max_rel, "n_ok": n_ok, "pass": n_ok == len(qs)}  # 供结论表引用


def section_static(arm, poses: dict) -> dict:
    """④ 静止特例：dq=ddq=0 时逆动力学应退化为重力项，并说明重力符号口径。

    :param arm: 离线 joyArm 对象（数学门面入口）。
    :param poses: 位形字典 {名称: 关节角向量 `(n_arm,)`}。
    :return: 统计字典：最大误差、通过数与总结论。
    """
    print("=" * 78)                                        # 大节分隔线
    print("④ 静止特例：dq = ddq = 0 时，solve_idyn(q, 0, 0) 应等于 get_gravity(q)")
    print("-" * 78)                                        # 小节分隔线
    print("   重力符号口径：G = +∂U/∂q（U 为重力势能）——运动方程 τ = M·ddq + C·dq + G 中，")
    print("   G 是静态悬停时电机为抵消重力必须输出的补偿力矩（dq=ddq=0 ⇒ τ = G）。")
    print("-" * 78)                                        # 小节分隔线
    headers = ["位形", "max|idyn(q,0,0) − G(q)| (N·m)", "判定"]
    rows = []                                              # 数据行收集器
    max_err, n_ok = 0.0, 0                                 # 跨组最大误差与通过数初值
    dq0 = np.zeros(arm.n_arm)                              # 零速度向量
    ddq0 = np.zeros(arm.n_arm)                             # 零加速度向量
    for name, q in poses.items():                          # 逐位形检查静止特例
        tau_static = arm.solve_idyn(q, dq0, ddq0)          # 静止状态逆动力学（RNEA 全量调用）
        g = arm.get_gravity(q)                             # 重力项（实现上即 RNEA 零速零加速特例）
        err = float(np.max(np.abs(tau_static - g)))        # 两者的最大绝对误差
        ok = err <= STATIC_TOL                             # 特例判定：误差不超过容差
        rows.append([name, f"{err:.2e}", "✓" if ok else "✗"])  # 本位形结果入表
        max_err = max(max_err, err)                        # 更新跨组最大误差
        n_ok += int(ok)                                    # 通过数累加
    print_table(headers, rows)                             # 打印结果表
    print("-" * 78)                                        # 小节分隔线
    print(f"小结：{n_ok}/{len(poses)} 个位形在容差 {STATIC_TOL:.0e} N·m 内 idyn == G"
          f"（最大误差 {max_err:.2e} N·m）")
    return {"max_err": max_err, "n_ok": n_ok, "pass": n_ok == len(poses)}  # 供结论表引用


def section_gravity(arm, poses: dict) -> dict:
    """⑤ 重力力矩对比表：home / zero / rand 位形下 G(q) 的分布差异。

    :param arm: 离线 joyArm 对象（数学门面入口）。
    :param poses: 位形字典 {名称: (关节角向量, 说明文字)}。
    :return: 各位形的最大关节重力矩 {名称: 数值}（供结论表观察项引用）。
    """
    print("=" * 78)                                        # 大节分隔线
    print("⑤ 重力力矩对比：home / zero / rand 位形下的 G(q)")
    headers = ["位形", "说明", "关节角 q (rad)", "重力矩 G(q) (N·m)", "max|G| (N·m)"]
    rows, peak = [], {}                                    # 数据行收集器 / 各位形最大重力矩记录
    for name, (q, note) in poses.items():                  # 逐位形计算重力并填表
        g = arm.get_gravity(q)                             # 该位形的重力力矩向量
        peak[name] = float(np.max(np.abs(g)))              # 最大关节重力矩（负载最重的关节）
        q_str = " ".join(f"{x:+.2f}" for x in q)           # 关节角紧凑显示（带符号 2 位小数）
        g_str = " ".join(f"{x:+.2f}" for x in g)           # 重力矩紧凑显示（同上）
        rows.append([name, note, q_str, g_str, f"{peak[name]:.3f}"])  # 本位形结果入表
    print_table(headers, rows)                             # 打印结果表
    print("-" * 78)                                        # 小节分隔线
    print("观察：不同位形连杆伸展方向不同，重力对各关节的负载差异显著——重力补偿前馈")
    print("      是真机控制中力矩的主要来源之一（对应本表 G(q)）。")
    return peak                                            # 返回供结论表引用


def section_conclusion(res_m: dict, res_r: dict, res_s: dict, peak: dict) -> None:
    """⑥ 汇总分析结论表：逐项给出检查结果与关键数值。

    :param res_m: ② 质量矩阵统计字典。
    :param res_r: ③ RNEA 分解一致性统计字典。
    :param res_s: ④ 静止特例统计字典。
    :param peak: ⑤ 各位形最大关节重力矩字典。
    """
    print("=" * 78)                                        # 大节分隔线
    print("⑥ 分析结论")                                    # 大节标题
    headers = ["检查项", "关键数值", "结论"]               # 结论表三列表头
    rows = [                                               # 逐项汇总（布尔结论转中文，观察项单列）
        ["质量矩阵对称性", f"最差 max|M−Mᵀ| = {res_m['max_asym']:.2e}",   # 结论行 1：对称性
         "通过" if res_m["pass_sym"] else "未通过"],
        ["质量矩阵正定性", f"最小 λ_min = {res_m['min_eig']:.4f} > 0",    # 结论行 2：正定性
         "通过" if res_m["pass_pd"] else "未通过"],
        ["RNEA 分解一致性", f"max|idyn − (M·ddq + C·dq + G)| = {res_r['max_err']:.2e} N·m",   # 结论行 3：一致性
         "通过" if res_r["pass"] else "未通过"],
        ["静止特例 idyn==G", f"max|idyn(q,0,0) − G| = {res_s['max_err']:.2e} N·m",   # 结论行 4：静止特例
         "通过" if res_s["pass"] else "未通过"],
        ["重力位形对比", " / ".join(f"{k}: {v:.2f}" for k, v in peak.items()) + " (N·m)", "观察项"],   # 结论行 5：观察项
    ]
    print_table(headers, rows)                             # 打印结论表
    print("=" * 78)                                        # 收尾大节分隔线


def main() -> None:
    """主流程：离线创建机械臂 → 四类动力学检查 → 重力对比 → 汇总结论。"""
    rng = np.random.default_rng(2026)                      # 定种子：③ 的 dq/ddq 采样可复现
    with joyArm("rebot_arm_b601_dm") as arm:               # 离线创建（不连硬件），with 保证退出安全收尾
        section_setup(arm)                                 # ① 模型与 dynamics 域求解器信息
        qs = [arm.rand_q_arm() for _ in range(N_RAND)]     # 软限位内均匀随机采 6 组位形（≥5），贯穿 ②③④⑤ 复用
        res_m = section_mass_matrix(arm, qs)               # ② 质量矩阵对称 + 正定检查
        res_r = section_rnea(arm, qs, rng)                 # ③ RNEA 分解一致性检查
        poses_static = {                                   # ④ 静止特例所用位形
            "home": arm.home_arm,                          # cfg 归位位形
            "zero": arm.zero_arm,                          # 全零位形
            "rand": qs[0],                                 # 复用第一组随机位形
        }
        res_s = section_static(arm, poses_static)          # ④ dq=ddq=0 特例检查
        poses_g = {                                        # ⑤ 重力对比所用位形（附说明文字）
            "home": (arm.home_arm, "cfg 归位位形"),        # home 位形
            "zero": (arm.zero_arm, "全零位形"),            # zero 位形
            "rand": (qs[0], "软限位内随机位形"),           # 随机位形（复用②第一组）
        }
        peak = section_gravity(arm, poses_g)               # ⑤ 三位形重力对比表
        section_conclusion(res_m, res_r, res_s, peak)      # ⑥ 汇总结论表
    print("离线动力学测试分析完毕（收尾「未连接……豁免跳过」类日志为正常提示）")


if __name__ == "__main__":
    main()                                                 # 脚本直接运行时执行主流程
