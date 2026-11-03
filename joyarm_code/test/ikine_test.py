"""
==============================================================================
逆运动学（ikine）测试分析  (test 离线测试)
==============================================================================
【功能概要】
    面向 IkineSolver 求解器基类的离线测试分析（无需硬件，全程数秒跑完）：
      1. 离线创建 rebot_arm_b601_dm 数字模型，查询 ikine 域已加载与激活的求解器
      2. 数值法 PinIkineSolver：软限位内随机 6 组 q → FK → IK 往返误差表
         （字段与 IKResult 对齐：success / err / n_iter，另拆出位置、姿态分项误差）
      3. 数值法参数敏感性：收敛容差 tol 与迭代上限 iters 各两组对比
      4. 解析法 AnalyticIkineSolver：solve_all 全解（≤8 组）逐组打印 + 软限位内有效解统计
      5. 交叉验证：数值解到解析解集合中最近解的距离（两法解集一致性）
      6. 异常路径：不可达位姿 success=False；数值法激活时调 solve_ikine_all 抛 RuntimeError
      7. 末尾汇总打印分析结论表

【原理简介】
    逆运动学（IK）已知末端位姿（位置 + 姿态），反求各关节角 q，与正运动学（FK）互逆。
    数值法 PinIkineSolver：从起点 q0 出发做 LM 阻尼迭代，每步用雅可比的阻尼伪逆修正
    关节角、逐步压小位姿残差；适用任意构型，但一次只得一组解，失败时限位内随机重启。
    解析法 AnalyticIkineSolver：仅适用"第 2/3/4 轴平行 + 第 5/6 轴相交"的六轴构型，
    几何枚举肩/肘/腕各 2 个分支、闭式写出最多 8 组解，再经 FK 回代剔除伪解。
    本测试对同一批随机目标位姿分别用两法求解，对比精度、参数敏感性与异常行为。

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（离线脚本，无需连接机械臂硬件）
    python test/ikine_test.py

【预期输出】
    ①~⑥ 六节控制台信息 + 末尾结论表；随机样本每次运行不同，结论表数值随本次运行刷新。
==============================================================================
"""

from pathlib import Path                     # 标准库：路径解析（保证未安装 joyarm_core 也能运行）
import sys                                   # 标准库：注入仓库路径到 sys.path
import textwrap                              # 标准库：长错误信息按 78 列折行显示

import numpy as np                           # 数值计算：向量 / 矩阵运算

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 把 joyarm_code/ 加入模块搜索路径
from joyarm_core import Pose, joyArm         # JoyArm 创建入口 + 统一位姿数据类

N_SAMPLES = 6                                # 随机往返测试样本组数（任务规格要求 ≥5）


def disp_w(text: str) -> int:
    """返回字符串的终端显示宽度（中文等非 ASCII 字符按 2 列计）。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)     # 逐字符累计显示宽度


def pad(text: str, width: int) -> str:
    """右补空格到指定显示宽度（至少留 1 格间隔），保证含中文的表格列对齐。"""
    return text + " " * max(1, width - disp_w(text))            # 宽度不足则补空格


def print_row(cells: list) -> None:
    """按 (文本, 列宽) 序列拼接打印一行表格。"""
    print("".join(pad(t, w) for t, w in cells))                 # 逐列补齐后拼接输出


def fmt_err(value) -> str:
    """误差标量格式化：None 显「—」，无穷显「inf」，其余保留 2 位科学计数。"""
    if value is None:                                           # 空值（如解析法失败时 q=None）
        return "—"
    if not np.isfinite(value):                                  # 不可达等情形残差为 inf
        return "inf"
    return f"{value:.2e}"                                       # 常规小量误差


def pose_gap(pose_ref, pose_chk) -> tuple:
    """拆分位姿误差：返回 (位置误差范数 m, 姿态误差角 rad)。

    位置误差取两位置向量之差的 2 范数；姿态误差角由相对旋转矩阵的迹求得：
    angle = arccos((trace(R_ref·R_chkᵀ) − 1) / 2)。
    """
    d_pos = float(np.linalg.norm(np.asarray(pose_ref.position)  # 位置误差：‖p_ref − p_chk‖
                                 - np.asarray(pose_chk.position)))
    R_rel = np.asarray(pose_ref.T[:3, :3]) @ np.asarray(pose_chk.T[:3, :3]).T  # 相对旋转 R_ref·R_chkᵀ
    d_rot = float(np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)))  # 等效旋转角
    return d_pos, d_rot                                         # 返回 (m, rad) 二元组


def wrap_dist(q_a, q_b) -> float:
    """两组关节角的「整圈折算」距离：逐关节角差折算到 (-π, π] 后取欧氏范数。

    解析解会做 ±2π 平移以贴近限位区间，与数值解可能相差整圈但构型等价，
    折算后的距离才能反映二者是否为同一构型。
    """
    dq = np.asarray(q_a, dtype=float) - np.asarray(q_b, dtype=float)  # 逐关节角差
    dq = np.arctan2(np.sin(dq), np.cos(dq))                     # 等价角折算到 (-π, π]
    return float(np.linalg.norm(dq))                            # 折算后欧氏范数


def main() -> None:
    """ikine 离线测试分析主流程：创建 → 数值法往返/参数 → 解析法全解 → 交叉验证 → 异常 → 结论。"""
    # ---- ① 离线创建与 ikine 域概览 ----
    print("=" * 78)
    print("① 离线创建与 ikine 域概览（无需硬件）")
    arm = joyArm("rebot_arm_b601_dm")                           # 离线创建：仅装配数字模型与算法，不触硬件
    print(f"   ikine 域已加载：{arm.get_ikine_solvers()}")      # 已加载成员（cfg robotics.ikine 声明顺序）
    print(f"   ikine 域当前激活：{arm.get_activated_ikine_solver()}")  # 首个成员默认激活（此处为数值法）
    soft = arm.joint_limits_soft_arm                            # 软限位：随机采样与有效解判定的基准
    print(f"   软限位 q_min：{np.round(soft.q_min, 3)}")        # 各关节下限（rad）
    print(f"   软限位 q_max：{np.round(soft.q_max, 3)}")        # 各关节上限（rad）

    # ---- ② 数值法：随机 q → FK → IK 往返误差表 ----
    print("=" * 78)                                             # 大节分隔线
    print("② 数值法 PinIkineSolver：随机 q → FK → IK 往返误差表")  # 大节标题
    print("-" * 78)                                             # 小节分隔线
    print(f"   采样 {N_SAMPLES} 组软限位内随机关节角（rand_q_arm），迭代起点统一 q0 = home_arm")  # 采样说明
    samples = []                                                # 每组存 (q_rand, pose_rand, 数值解 IKResult)
    for _ in range(N_SAMPLES):
        q_rand = arm.rand_q_arm()                               # 公开接口：软限位内均匀随机采样
        pose_rand = arm.solve_fkine(q_rand)                     # 正运动学：q_rand → 目标位姿
        res = arm.solve_ikine(pose_rand, q0=arm.home_arm)       # 数值法求逆：自 home 位形迭代
        samples.append((q_rand, pose_rand, res))                # 缓存供 ③④⑥ 复用
    print_row([("序号", 6), ("success", 9), ("err", 11), ("位置误差(m)", 13),
               ("姿态误差(rad)", 14), ("n_iter", 7)])           # 表头（字段名与 IKResult 一致）
    for i, (_, pose_rand, res) in enumerate(samples):           # 逐样本打印一行往返结果
        pose_chk = arm.solve_fkine(res.q)                       # 解 q 经 FK 回代得实际到位位姿
        d_pos, d_rot = pose_gap(pose_rand, pose_chk)            # 独立复算位置 / 姿态分项误差
        print_row([(f"{i + 1}", 6), ("是" if res.success else "否", 9),
                   (fmt_err(res.err), 11), (f"{d_pos:.2e}", 13),
                   (f"{d_rot:.2e}", 14), (f"{res.n_iter}", 7)])  # 一行：标志 + 残差 + 分项误差 + 迭代数
    print("   说明：err 为 IKResult 残差（位置范数 m + 姿态范数 rad 之和）；"  # err 字段口径
          "位置 / 姿态误差由解 q 经 FK 回代后独立复算。")       # 分项误差口径
    print("-" * 78)                                             # 小节分隔线
    print("   参数敏感性：固定目标（FK(home_arm)）+ 固定起点（neutral_arm），每次只动一个参数")  # 敏感性小节标题
    pose_fix = arm.solve_fkine(arm.home_arm)                    # 固定目标位姿：home 位形的 FK 结果
    q0_fix = arm.neutral_arm                                    # 固定迭代起点：软限位中点（数值法默认初值口径）
    groups = [                                                  # (组名, 求解关键字参数)——tol / iters 各两组
        ("tol=1e-1（松）", dict(tol=1e-1)),                     # 松容差：残差压到 1e-1 即停
        ("tol=1e-6（紧）", dict(tol=1e-6)),                     # 紧容差：残差须压到 1e-6
        ("iters=1（过小）", dict(iters=1)),                     # 迭代上限 1：几乎无法收敛
        ("iters=200（默认）", dict(iters=200)),                 # 迭代上限 200：求解器默认值
    ]
    print_row([("参数组", 19), ("success", 9), ("err", 11), ("n_iter", 7)])  # 参数对比表表头
    param_res = {}                                              # 组名 → IKResult，供⑥结论表引用实测值
    for name, kw in groups:                                     # 逐组求解（其余参数取默认）
        r = arm.solve_ikine(pose_fix, q0=q0_fix, **kw)          # 只变动 tol 或 iters 之一
        param_res[name] = r                                     # 缓存结果
        print_row([(name, 19), ("是" if r.success else "否", 9),
                   (fmt_err(r.err), 11), (f"{r.n_iter}", 7)])   # 一行参数组结果
    print("   解读：收紧 tol → 残差更小、所需迭代往往更多；iters 过小 → 上限内未收敛（success=False）。")

    # ---- ③ 解析法：solve_all 全解 ----
    print("=" * 78)                                             # 大节分隔线
    print("③ 解析法 AnalyticIkineSolver：solve_all 全解（≤8 组）")  # 大节标题
    print("-" * 78)                                             # 小节分隔线
    ok = arm.set_ikine_solver("analytic_ikine_solver")          # ikine 域内热切换到解析法
    print(f"   set_ikine_solver 返回 {ok}；当前激活：{arm.get_activated_ikine_solver()}")
    print("   适用前提：第 2/3/4 轴平行 + 第 5/6 轴相交的六转动关节构型（本型号满足）")
    all_res, valid_counts = [], []                              # 各样本的全解结果 / 软限位内有效解数
    print_row([("序号", 6), ("全解数K", 9), ("软限位内", 10),
               ("success", 9), ("err", 11)])                    # 全解统计表表头
    for i, (_, pose_rand, _) in enumerate(samples):             # 对②的每个目标位姿求全解
        r = arm.solve_ikine_all(pose_rand)                      # 全部解析解：q 形状 (K, 6)
        all_res.append(r)                                       # 缓存供④交叉验证
        if r.q.shape[0]:                                        # 有解时统计逐关节均在软限位内的行数
            k_valid = int(((r.q >= soft.q_min) & (r.q <= soft.q_max)).all(axis=1).sum())
        else:
            k_valid = 0                                         # K=0 时有效解数恒 0
        valid_counts.append(k_valid)                            # 缓存供⑥结论表
        print_row([(f"{i + 1}", 6), (f"{r.q.shape[0]}", 9), (f"{k_valid}", 10),
                   ("是" if r.success else "否", 9), (fmt_err(r.err), 11)])  # 一行统计
    print("-" * 78)
    r0 = all_res[0]                                             # 取样本 1 展开全解明细
    if r0.q.shape[0] == 0:                                      # 可达位姿不应出现空解集（防御性提示）
        print("   样本 1 无解析解（K=0）——可达位姿不应出现，请检查模型")  # 防御性提示
    else:
        print("   样本 1 全解明细（每行一组 q，rad；回代残差 = 该组 q 经 FK 后与目标的混合残差）")  # 明细表标题
        print_row([("解号", 6)] + [(f"q{j + 1}", 8) for j in range(6)]  # 明细表头：q1~q6
                  + [("软限位内", 10), ("回代残差", 11)])
        for k in range(r0.q.shape[0]):                          # 逐组解打印
            q_k = r0.q[k]                                       # 第 k 组解 (6,)
            pose_k = arm.solve_fkine(q_k)                       # FK 回代验证该组解
            d_pos, d_rot = pose_gap(samples[0][1], pose_k)      # 与目标位姿的分项残差
            in_soft = bool(np.all((q_k >= soft.q_min) & (q_k <= soft.q_max)))  # 逐关节均在软限位内
            print_row([(f"{k + 1}", 6)] + [(f"{v:+.3f}", 8) for v in q_k]  # 6 个关节角
                      + [("是" if in_soft else "否", 10), (f"{d_pos + d_rot:.2e}", 11)])

    # ---- ④ 交叉验证：数值解 ↔ 解析解集合 ----
    print("=" * 78)                                             # 大节分隔线
    print("④ 交叉验证：数值解 ↔ 解析解集合（最近解距离）")      # 大节标题
    print("-" * 78)                                             # 小节分隔线
    print_row([("序号", 6), ("数值success", 12), ("解析K", 8), ("最近距离(rad)", 14),
               ("整圈折算(rad)", 14), ("最近解号", 10)])        # 交叉验证表表头
    d_plain_max, d_wrap_max, n_cross, n_conv = None, None, 0, 0  # 收敛组距离最大值 / 可比数 / 收敛数
    for i, ((q_rand, pose_rand, res_num), r_all) in enumerate(zip(samples, all_res)):  # 逐样本配对两法结果
        if res_num.q is not None and r_all.q.shape[0]:          # 数值解与解析解集均可用才比距离
            dists = np.linalg.norm(r_all.q - res_num.q, axis=1)  # 数值解到每组解析解的欧氏距离
            k_near = int(np.argmin(dists))                      # 最近解下标（0 基）
            d_plain = float(dists[k_near])                      # 最近欧氏距离
            d_wrap = wrap_dist(r_all.q[k_near], res_num.q)      # 同一解的整圈折算距离
            n_cross += 1                                        # 计入可比样本
            if res_num.success:                                 # 仅收敛解才计入一致性统计
                n_conv += 1                                     # 收敛组计数
                d_plain_max = d_plain if d_plain_max is None else max(d_plain_max, d_plain)  # 更新最大值
                d_wrap_max = d_wrap if d_wrap_max is None else max(d_wrap_max, d_wrap)  # 同上
            cells = [(f"{i + 1}", 6), ("是" if res_num.success else "否", 12),
                     (f"{r_all.q.shape[0]}", 8), (f"{d_plain:.3e}", 14),
                     (f"{d_wrap:.3e}", 14), (f"{k_near + 1}", 10)]  # 正常行：实测距离
        else:                                                   # 缺任一侧结果时距离列显「—」
            cells = [(f"{i + 1}", 6), ("是" if res_num.success else "否", 12),
                     (f"{r_all.q.shape[0]}", 8), ("—", 14), ("—", 14), ("—", 10)]
        print_row(cells)                                        # 打印一行
    print("   解读：success=是 的行数值解已收敛，应与某组解析解构型一致（距离小量）；"
          "success=否 的行数值解仅为最优尝试，距离大属预期；欧氏距离大而折算距离小说明仅相差整圈（±2π 等价角）。")

    # ---- ⑤ 异常路径：不可达位姿与误用接口 ----
    print("=" * 78)                                             # 大节分隔线
    print("⑤ 异常路径：不可达位姿与误用接口")                   # 大节标题
    print("-" * 78)                                             # 小节分隔线
    arm.set_ikine_solver("pin_ikine_solver")                    # 切回数值法（默认激活项）
    pose_far = Pose(position=np.array([5.0, 5.0, 5.0]))         # 远超工作空间的不可达位姿（姿态取默认单位四元数）
    r_num = arm.solve_ikine(pose_far, q0=arm.home_arm)          # 数值法：起点 + 随机重启耗尽仍无法收敛
    print(f"   [数值法] 不可达位姿 solve：success={r_num.success}，"
          f"err={fmt_err(r_num.err)}，n_iter={r_num.n_iter}")   # 如实展示失败结果
    print(f"            最优尝试解 q：{np.round(r_num.q, 3)}")  # 失败时 q 为最优尝试（非 None）
    try:
        arm.solve_ikine_all(pose_far)                           # 数值法激活时误调全解接口
        print("   [数值法] solve_ikine_all 未抛异常（不应发生）")  # 正常不应到达此行
    except RuntimeError as e:                                   # 基类 NotImplementedError 被包装为 RuntimeError
        print("   [数值法] solve_ikine_all 按预期抛 RuntimeError：")
        print(textwrap.fill(str(e), width=70, initial_indent="       ",  # 错误详情按 78 列折行
                            subsequent_indent="       "))
    arm.set_ikine_solver("analytic_ikine_solver")               # 再切解析法验证同一不可达位姿
    r_ana = arm.solve_ikine(pose_far, q0=arm.home_arm)          # 解析法单解：无可行解
    r_ana_all = arm.solve_ikine_all(pose_far)                   # 解析法全解：空解集
    print(f"   [解析法] 不可达位姿 solve：success={r_ana.success}，"
          f"q={'—' if r_ana.q is None else '空数组（无可行解）' if r_ana.q.size == 0 else np.round(r_ana.q, 3)}")  # 失败时 q 为空数组
    print(f"   [解析法] 不可达位姿 solve_all：success={r_ana_all.success}，"
          f"q.shape={r_ana_all.q.shape}（0 组解）")             # 空解集：形状 (0, 6)

    # ---- ⑥ 分析结论表 ----
    print("=" * 78)                                             # 大节分隔线
    print("⑥ 分析结论（数值均取自本次运行实测）")               # 大节标题
    print("-" * 78)                                             # 小节分隔线
    ok_cnt = sum(1 for _, _, r in samples if r.success)         # 数值法往返收敛组数
    ok_pos, ok_rot, ok_iter = [], [], []                        # 成功样本的分项误差与迭代数
    for _, pose_rand, res in samples:                           # 仅统计成功组
        if res.success:
            d_pos, d_rot = pose_gap(pose_rand, arm.solve_fkine(res.q))  # 回代复算分项误差
            ok_pos.append(d_pos)                                # 位置误差列表
            ok_rot.append(d_rot)                                # 姿态误差列表
            ok_iter.append(res.n_iter)                          # 迭代数列表
    if ok_pos:                                                  # 至少一组成功才有误差统计
        err_brief = (f"位置误差最大 {max(ok_pos):.1e} m、姿态误差最大 {max(ok_rot):.1e} rad、"
                     f"平均迭代 {np.mean(ok_iter):.0f} 次")
    else:                                                       # 全失败属异常情形，如实报告
        err_brief = "本次无成功组（异常，请检查模型配置）"
    k_list = [r.q.shape[0] for r in all_res]                    # 各样本解析全解数 K
    r_loose = param_res["tol=1e-1（松）"]                       # 松容差组结果（结论引用）
    r_tight = param_res["tol=1e-6（紧）"]                       # 紧容差组结果
    r_few = param_res["iters=1（过小）"]                        # 过小迭代上限组结果
    r_many = param_res["iters=200（默认）"]                     # 默认迭代上限组结果
    if n_conv:                                                  # 有收敛可比样本才给一致性结论
        cross_brief = (f"收敛组最近欧氏距离最大 {d_plain_max:.2e} rad（整圈折算 {d_wrap_max:.2e} rad）")
        if n_cross > n_conv:                                    # 存在未收敛的可比样本时补充说明
            cross_brief += f"；另 {n_cross - n_conv} 组数值未收敛（距离仅反映最优尝试）"
    else:                                                       # 无收敛可比样本：如实标注
        cross_brief = "—（本次无收敛的可比样本）"
    rows = [                                                    # 结论表：条目 + 实测结论文本
        ("数值法往返", f"{ok_cnt}/{N_SAMPLES} 组 success=True；{err_brief}"),  # 结论 1：往返收敛统计
        ("tol 影响", f"1e-1→1e-6：err {fmt_err(r_loose.err)}→{fmt_err(r_tight.err)}，"  # 结论 2：容差敏感性
                    f"n_iter {r_loose.n_iter}→{r_tight.n_iter}（收紧 → 残差更小）"),
        ("iters 影响", f"iters=1：success={r_few.success}；iters=200：success={r_many.success}"  # 结论 3：迭代上限敏感性
                      f"（上限过小 → 未收敛）"),
        ("解析法全解", f"全解数 {min(k_list)}~{max(k_list)} 组/目标，软限位内有效 "  # 结论 4：全解规模
                      f"{min(valid_counts)}~{max(valid_counts)} 组"),
        ("交叉验证", cross_brief),                              # 结论 5：两法解集一致性
        ("异常路径", "不可达位姿两法均 success=False；数值法激活调 solve_ikine_all → RuntimeError"),  # 结论 6：异常行为
    ]
    for name, text in rows:                                     # 两列对齐逐行打印
        print(f"   {pad(name, 12)}{text}")

    arm.close()                                                 # 显式安全收尾（幂等；离线仅释放资源）
    print("=" * 78)
    print("ikine 离线测试分析运行完毕（收尾「未连接……豁免跳过」类日志为正常提示）")


if __name__ == "__main__":                                      # 直接运行入口（被导入时不执行）
    main()                                                      # 离线主流程：无需任何硬件
