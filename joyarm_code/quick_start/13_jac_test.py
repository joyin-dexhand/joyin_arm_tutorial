"""
==============================================================================
雅可比测试分析（JacobianSolver 基类）  (快速上手 例程 13)
==============================================================================
【功能概要】
    无需连接硬件，离线测试 jacobian 域求解器（基类 JacobianSolver → pin_jacobian_solver）：
      1. 求解器域信息与形状契约：已注册 / 当前激活求解器，J 与 J* 的形状检查
      2. 解析 vs 数值差分：5 组 rand_q 用中心差分（步长 1e-6）构造数值雅可比，
         与 get_jac 逐元素对比，表格打印最大误差
      3. 奇异值分析：home 与 4 个典型位形上的 σ_min / σ_max / 条件数 / 可操作度
      4. 阻尼伪逆：get_jac_pinv_damped 在 λ = 0（参照）/ 1e-3 / 0.1 三档下验证
         J·J* 逼近单位阵的程度与阻尼增大的偏差（附理论上界 λ²/(σ_min²+λ²)）
      5. 末尾打印分析结论表

【原理简介】
    雅可比 J(q) 是基座系下的 6×n 微分映射矩阵：上 3 行把关节速度映射为末端线速度、
    下 3 行映射为角速度，即 v_末端 = J·dq。get_jac 走 pinocchio 解析式；数值差分则对
    每个关节做 ±h 扰动各算一次正运动学：位置列用中心差分 (p⁺-p⁻)/(2h)，角速度列用
    相对旋转的对数映射 log₃(R⁺R⁻ᵀ)/(2h)，两者一致即互为印证。J 的奇异值 σ 是各方向
    的伸缩率：σ_min→0 即奇异（某方向近乎失控），条件数 σ_max/σ_min 与可操作度
    √det(JJᵀ) 度量离奇异多远。奇异附近伪逆会放大指令，阻尼伪逆 J* = Jᵀ(JJᵀ+λ²I)⁻¹
    用 λ 压制小 σ 方向的增益，代价是 J·J* 偏离 I，逐方向偏差恰为 λ²/(σ²+λ²)。
    代码架构：arm.get_jac / get_jac_pinv_damped 门面转发给激活求解器；基类
    JacobianSolver 定义抽象方法 jac、并内置通用的 jac_pinv_damped（SVD 实现）。

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 2. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（在 joyarm_code/ 目录下；离线脚本，无需机械臂硬件）
    python quick_start/13_jac_test.py

【预期输出】
    ①~⑤ 五节控制台信息（差分对比表 / 奇异值分析表 / 阻尼伪逆偏差表 /
    分析结论表），末尾打印「离线例程运行完毕」；rand_q 每次运行取值不同，但各项
    误差与指标量级不变；收尾时「未连接……豁免跳过」类日志属正常提示，非报错。
==============================================================================
"""

from pathlib import Path                       # 组装「未安装也可运行」的搜索路径
import sys                                     # 同上：改 sys.path 需要它

import numpy as np                             # 数值计算（矩阵 / SVD / 行列式）
import pinocchio as pin                        # joyarm_core 底层运动学库：log₃ 旋转对数映射

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装时也可直接运行
from joyarm_core import joyArm  # noqa: E402   # JoyArm 主门面（依赖上方 sys.path 生效）
from joyarm_core.robotics.jacobian import JacobianSolver, PinJacobianSolver  # noqa: E402

H_DIFF = 1e-6                                  # 中心差分步长（rad）：再小会被浮点噪声淹没，再大截断误差增大


def disp_w(text: str) -> int:
    """字符串的终端显示宽度：CJK 全角字符按 2 列、其余按 1 列计。

    f-string 的 :<n 只按字符个数补空格，中文表头会错位，故自算显示宽度。
    """
    return sum(2 if ord(c) > 0x2E80 else 1 for c in text)   # 0x2E80 起为 CJK 区段的粗略分界


def pad(text: str, width: int, align: str = ">") -> str:
    """按显示宽度补空格对齐：align='>' 右对齐（数值列），'<' 左对齐（文本列）。"""
    gap = max(width - disp_w(text), 0)         # 防负数：超宽时不截断、只不补
    return " " * gap + text if align == ">" else text + " " * gap   # 按对齐方向补空格


def row(cells: list, widths: list, aligns: list) -> str:
    """拼一行表格文本：cells 逐列按 widths / aligns 对齐后以两空格连接。"""
    return "  ".join(pad(c, w, a) for c, w, a in zip(cells, widths, aligns))   # 逐列对齐拼接


def fmt(v, nd: int = 4) -> str:
    """指标格式化：None 显「—」、inf 显「∞」、其余保留 nd 位小数。"""
    if v is None:                              # 无值（本例程不会出现，助手函数兜底）
        return "—"
    if np.isinf(v):                            # σ_min=0 时条件数无穷大
        return "∞"
    return f"{v:.{nd}f}"                       # 常规定点小数


def jac_numeric(arm, q: np.ndarray, frame: str, h: float = H_DIFF) -> np.ndarray:
    """中心差分数值雅可比：逐关节 ±h 扰动正运动学，差分出 J 的每一列。

    :param arm: 离线 JoyArm 对象（提供 solve_fkine 正运动学）。
    :param q: `(n,)` 关节角 rad。
    :param frame: 目标帧名（必须与 get_jac 用同一帧才可比）。
    :param h: 差分半步长 rad，缺省 H_DIFF。
    :return: `(6,n)` 基座系数值雅可比（上 3 行线速度、下 3 行角速度）。
    """
    J = np.zeros((6, arm.n_arm))               # 预分配结果：6 个速度分量 × n 个关节列
    for i in range(arm.n_arm):                 # 逐关节列差分
        qp, qm = q.copy(), q.copy()            # 各拷贝一份，避免污染原 q
        qp[i] += h                             # 第 i 关节 +h 扰动
        qm[i] -= h                             # 第 i 关节 -h 扰动
        Tp = arm.solve_fkine(qp, frame).T      # +h 位形末端齐次矩阵（基座系）
        Tm = arm.solve_fkine(qm, frame).T      # -h 位形末端齐次矩阵（基座系）
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)          # 位置中心差分 → 线速度列
        J[3:, i] = pin.log3(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * h)   # 相对旋转 log₃ → 基座系角速度列
    return J                                   # 形状与 get_jac 相同，可逐元素比对


def singularity_metrics(J: np.ndarray) -> tuple:
    """由雅可比计算奇异值分析四指标。

    :param J: `(6,n)` 雅可比矩阵。
    :return: (σ_min, σ_max, 条件数 σ_max/σ_min, 可操作度 √det(JJᵀ))。
    """
    s = np.linalg.svd(J, compute_uv=False)     # 奇异值降序排列：s[0]=σ_max，s[-1]=σ_min
    s_min, s_max = float(s[-1]), float(s[0])   # 转内置 float，便于格式化
    cond = s_max / s_min if s_min > 0.0 else float("inf")    # σ_min=0（奇异）时条件数无穷大
    det = float(np.linalg.det(J @ J.T))        # 可操作度定义式中的 det(JJᵀ)
    mani = float(np.sqrt(det)) if det > 0.0 else 0.0         # det≤0（浮点误差/降秩）时取 0
    return s_min, s_max, cond, mani            # 四元组一并返回


def main() -> None:
    """离线主流程：创建对象 → 差分对比 → 奇异值分析 → 阻尼伪逆 → 结论表 → 收尾。"""
    # ---- ① 对象与求解器：离线创建 JoyArm，查看 jacobian 域与形状契约 ----
    print("=" * 78)
    print("① 对象与求解器：离线创建 JoyArm，查看 jacobian 域与形状契约")
    arm = joyArm("rebot_arm_b601_dm")          # 离线创建：装配 URDF/pin 模型与算法层，不触硬件
    frame_end = arm.get_config()["joyarm"]["frame_end"]       # cfg 指定的末端帧名，后续统一用它
    print(f"   型号 rebot_arm_b601_dm：n_arm = {arm.n_arm}，末端帧 frame_end = {frame_end}")  # 模型基本参数
    print(f"   jacobian 域已注册求解器：{arm.get_jac_solvers()}")   # 域内注册名（字母序）
    print(f"   jacobian 域当前激活求解器：{arm.get_activated_jac_solver()}")  # 缺省激活（cfg 首位）
    solver = PinJacobianSolver()               # 激活求解器对应的类：实例化做基类契约说明
    print(f"   PinJacobianSolver 是 JacobianSolver 基类的子类：{isinstance(solver, JacobianSolver)}"
          "（基类定义抽象方法 jac、内置通用 jac_pinv_damped，门面 get_jac* 转发到激活实例）")
    J0 = arm.get_jac(arm.home_arm, frame_end)              # 门面 get_jac → 激活求解器 .jac(解析式)
    J0s = arm.get_jac_pinv_damped(arm.home_arm, frame_end)  # 门面 → 基类通用 SVD 阻尼伪逆
    ok_shape = J0.shape == (6, arm.n_arm) and J0s.shape == (arm.n_arm, 6)   # 契约：J 为 (6,n)、J* 为 (n,6)
    print(f"   形状契约：J {J0.shape} = (6, n)、J* {J0s.shape} = (n, 6) → {'满足' if ok_shape else '不满足'}")
    print("   home 位形的 J(q)（行 = 末端速度分量 [vx,vy,vz,wx,wy,wz]，列 = 关节 1~6）：")
    with np.printoptions(precision=3, suppress=True):      # 紧凑打印：3 位小数、去多余零
        print(J0)                              # 上 3 行单位 m/rad、下 3 行 rad/rad

    # ---- ② 解析 vs 数值差分：5 组 rand_q，中心差分步长 h = 1e-6 ----
    print("=" * 78)                                      # 大节分隔线
    print("② 解析雅可比 vs 数值差分：5 组 rand_q，中心差分步长 h = 1e-6")  # 大节标题
    print("-" * 78)                                      # 小节分隔线
    widths = [6, 20, 20, 18]                   # 四列显示宽度：组号 / 线速度行 / 角速度行 / 整体
    aligns = [">", ">", ">", ">"]              # 全数值列，统一右对齐
    print(row(["组#", "max|ΔJ| 线速度行", "max|ΔJ| 角速度行", "max|ΔJ| 整体"], widths, aligns))
    print("-" * 78)
    worst_diff = 0.0                           # 记录 5 组中的整体最大误差（供结论表引用）
    for k in range(5):                         # 5 组软限位内均匀随机关节角
        q = arm.rand_q_arm()                   # 每次调用独立随机（量级结论不受取值影响）
        J_ana = arm.get_jac(q, frame_end)      # 解析雅可比（被测对象）
        J_num = jac_numeric(arm, q, frame_end)  # 数值差分雅可比（独立参照）
        d = np.abs(J_ana - J_num)              # 逐元素误差
        lin = float(d[:3, :].max())            # 上 3 行（线速度行）最大误差
        ang = float(d[3:, :].max())            # 下 3 行（角速度行）最大误差
        tot = float(d.max())                   # 整体最大误差
        worst_diff = max(worst_diff, tot)      # 更新 5 组最差值
        print(row([str(k + 1), f"{lin:.2e}", f"{ang:.2e}", f"{tot:.2e}"], widths, aligns))
    verdict_diff = "一致" if worst_diff < 1e-6 else "不一致"   # 1e-6 为该步长下宽松合格线
    print(f"   结论：最大误差 {worst_diff:.2e}（截断 + 浮点噪声量级）→ 解析与数值雅可比{verdict_diff}")

    # ---- ③ 奇异值分析：home 与 4 个典型位形对比 ----
    print("=" * 78)
    print("③ 奇异值分析：σ_min / σ_max / 条件数 / 可操作度（home + 4 个典型位形）")
    configs = [                                # (名称, 说明, q)：均取软限位内的可达位形
        ("home", "归位位形（cfg home_arm）", np.asarray(arm.home_arm, dtype=float)),      # cfg 归位构型
        ("zero", "全零位形（伸直边界）", np.asarray(arm.zero_arm, dtype=float)),         # 全零伸直构型
        ("neutral", "软限位中点（数值法初值）", np.asarray(arm.neutral_arm, dtype=float)),  # 数值求解默认初值
        ("folded", "折叠位形（肘部大幅弯折）", np.array([0.0, -2.9, -2.9, 0.0, 1.3, 0.0])),  # 人为折叠
        ("side", "侧伸位形（肩转 90°）", np.array([1.57, -1.2, -0.5, 0.0, 0.8, 0.0])),  # 人为侧伸
    ]
    for name, desc, q in configs:              # 先逐行交代位形含义与关节角取值
        print(f"   {name:<8s} {desc}：q = {np.round(q, 2).tolist()}")
    print("-" * 78)
    widths = [10, 10, 10, 10, 10, 12]          # 位形 / σ_min / σ_max / 条件数 / 可操作度 / 判读
    aligns = ["<", ">", ">", ">", ">", "<"]    # 文本列左对齐、数值列右对齐
    print(row(["位形", "σ_min", "σ_max", "条件数", "可操作度", "判读"], widths, aligns))
    print("-" * 78)
    metrics = {}                               # 位形名 → 四指标（供 ④⑤ 节复用）
    for name, _, q in configs:                 # 逐位形算指标并打印
        s_min, s_max, cond, mani = singularity_metrics(arm.get_jac(q, frame_end))   # J(q) → 四指标
        metrics[name] = (s_min, s_max, cond, mani)          # 缓存备用
        judge = "良态" if s_min >= 0.05 else ("条件较差" if s_min >= 0.01 else "接近奇异")   # 按 σ_min 裕度分档
        print(row([name, fmt(s_min), fmt(s_max), fmt(cond, 1), fmt(mani), judge], widths, aligns))
    best_cfg = min(metrics, key=lambda n: metrics[n][2])    # 条件数最小 = 最远离奇异
    worst_cfg = max(metrics, key=lambda n: metrics[n][2])   # 条件数最大 = 最接近奇异
    worst_mani_cfg = min(metrics, key=lambda n: metrics[n][3])   # 可操作度最小 = 各向灵巧度最差
    print(f"   结论：{best_cfg} 条件数最小（{metrics[best_cfg][2]:.1f}，灵活性最好）；"
          f"{worst_cfg} 条件数最大（{metrics[worst_cfg][2]:.1f}，奇异风险最高）")

    # ---- ④ 阻尼伪逆：λ = 0（参照）/ 1e-3（默认档）/ 0.1（强阻尼档） ----
    print("=" * 78)                                      # 大节分隔线
    print("④ 阻尼伪逆：get_jac_pinv_damped 三档 λ 下 max|J·J*-I| 与理论上界对比")  # 大节标题
    print("   J·J* = I 表示末端速度指令被精确还原；偏差 = 阻尼引入的『执行损失』")  # 偏差的物理含义
    print("-" * 78)
    widths = [10, 22, 20, 26]                  # 位形 / λ / 实测偏差 / 理论上界
    aligns = ["<", "<", ">", ">"]              # 前两列文本左对齐、后两列数值右对齐
    print(row(["位形", "阻尼 λ", "实测 max|J·J*-I|", "理论上界 λ²/(σ_min²+λ²)"], widths, aligns))
    print("-" * 78)
    lams = [(0.0, "0（精确伪逆参照）"), (1e-3, "1e-3（默认档）"), (0.1, "0.1（强阻尼档）")]   # 三档阻尼
    dev_by_cfg = {}                            # 位形名 → {λ: 实测偏差}（供趋势判定与结论表）
    for name, _, q in configs:                 # 与 ③ 相同的 5 个位形逐一测试
        J = arm.get_jac(q, frame_end)          # 该位形的解析雅可比
        dev_by_cfg[name] = {}                  # 本位形的偏差记录字典
        for lam, label in lams:                # 三档阻尼逐一求 J* 并核对 J·J*
            Js = arm.get_jac_pinv_damped(q, frame_end, damping=lam)   # 阻尼伪逆 J*
            dev = float(np.abs(J @ Js - np.eye(6)).max())             # 偏离单位阵的最大元素
            bound = lam ** 2 / (metrics[name][0] ** 2 + lam ** 2) if lam > 0.0 else 0.0   # 最劣奇异方向的理论偏差
            dev_by_cfg[name][lam] = dev        # 记录实测偏差
            print(row([name, label, f"{dev:.2e}", f"{bound:.2e}" if lam > 0.0 else "0"], widths, aligns))
    mono_ok = all(d[0.1] > d[1e-3] > d[0.0] for d in dev_by_cfg.values())   # 全部位形：λ↑ ⇒ 偏差单调增大
    exact_ok = all(d[0.0] < 1e-8 for d in dev_by_cfg.values())             # λ=0 满秩处应精确还原（数值精度内）
    bound_ok = all(dev_by_cfg[n][lam] <= lam ** 2 / (metrics[n][0] ** 2 + lam ** 2) + 1e-12
                   for n in dev_by_cfg for lam in (1e-3, 0.1))             # λ>0 各行实测偏差 ≤ 理论上界
    print(f"   结论：λ 增大偏差单调增大（5/5 位形）{'✓' if mono_ok else '✗'}；"  # 趋势一：λ↑ ⇒ 偏差↑
          f"λ=0 精确伪逆（<1e-8）{'✓' if exact_ok else '✗'}；"                 # 趋势二：λ=0 精确还原
          f"实测偏差 ≤ 理论上界 {'✓' if bound_ok else '✗'}；{worst_cfg} 位形上即便小阻尼也现大偏差"  # 趋势三：上界验证
          f"（1e-3 档偏差 {dev_by_cfg[worst_cfg][1e-3]:.2e}）——阻尼的意义正是压制奇异方向的增益爆炸")  # 压制增益爆炸

    # ---- ⑤ 分析结论表 ----
    print("=" * 78)                                      # 大节分隔线
    print("⑤ 分析结论表")                                 # 大节标题
    print("-" * 78)                                      # 小节分隔线
    widths = [6, 22, 44]                       # 序号 / 测试项 / 判定与结论
    aligns = [">", "<", "<"]                   # 序号右对齐，其余文本左对齐
    print(row(["序号", "测试项", "判定 / 结论"], widths, aligns))
    print("-" * 78)
    concl = [                                  # 各测试项一行结论（数值均取自上方实测）
        ("1", "形状契约", f"J 为 (6, n)、J* 为 (n, 6)：{'满足' if ok_shape else '不满足'}"  # 结论 1：形状契约
                        f"（激活求解器 {arm.get_activated_jac_solver()}）"),
        ("2", "解析 vs 数值差分", f"5 组 rand_q 整体最大误差 {worst_diff:.2e}（< 1e-6）→ {verdict_diff}"),  # 结论 2
        ("3", "奇异值分析", f"{best_cfg} 条件数最小（{metrics[best_cfg][2]:.1f}）、{worst_cfg} 最大"  # 结论 3：条件数对比
                          f"（{metrics[worst_cfg][2]:.1f}）；可操作度最低为 {worst_mani_cfg}"
                          f"（{metrics[worst_mani_cfg][3]:.4f}，奇异裕度最小）"),
        ("4", "阻尼伪逆", f"λ↑ ⇒ 偏差↑ {'✓' if mono_ok else '✗'}、λ=0 精确 {'✓' if exact_ok else '✗'}、"  # 结论 4：三档验证
                        f"实测 ≤ 理论上界 {'✓' if bound_ok else '✗'}（上界 λ²/(σ_min²+λ²)）"),
    ]
    for no, item, res in concl:                # 逐行打印结论表
        print(row([no, item, res], widths, aligns))

    # ---- 收尾 ----
    print("=" * 78)
    arm.close()                                # 显式安全收尾（幂等；离线未连接仅停线程清缓存）
    print("离线例程运行完毕（收尾时「未连接……豁免跳过」类日志为正常提示）")


if __name__ == "__main__":
    main()                                     # 直接运行入口：执行离线测试主流程
