"""
==============================================================================
fkine 正运动学测试分析  (test 离线测试)
==============================================================================
【功能概要】
    无需硬件，离线测试分析 FkineSolver 基类的两个实现：
      1. 双实现对比：PinFkineSolver 与 MdhFkineSolver 对 zero/home/5 组软限位内
         随机位形求末端位姿，表格对比位置差与姿态差（四元数角距）；
      2. 帧选择：末端帧 / 中间连杆帧 / 基座系（frame=0）位姿对比；
      3. 两种用法：joyArm 门面 solve_fkine 与直接实例化求解器 .solve(arm, q, frame)
         （以仅含两个成员的鸭子 DuckArm 演示"arm 鸭子类型"）；
      4. 异常路径：未知帧名、mdh 非末端帧的 ValueError 捕获演示；
      5. 末尾输出全项分析结论表。

【原理简介】
    正运动学（FK）回答"关节角 q 已知时末端在哪"：从基座起逐关节连乘 4×4 齐次
    变换矩阵，得到目标帧在基座系下的位置与姿态四元数。
    库内有两个实现——PinFkineSolver 调 pinocchio 原生算法（黑盒、支持任意帧）；
    MdhFkineSolver 用 config 的 arm_mdh 教学参数手工连乘（白盒、仅末端帧）。
    两实现算法独立而结果应数值一致，互为正确性对照。
    frame 参数指定求哪个坐标系：整数 0 即基座系（universe 帧），任意 q 下位姿恒为恒等。

【环境与运行】
    ⓪ 安装 uv（仅首次）：curl -LsSf https://astral.sh/uv/install.sh | sh
      （Windows PowerShell：irm https://astral.sh/uv/install.ps1 | iex）
    ① cd <仓库路径>/joyarm_code && uv sync && source .venv/bin/activate
    ② python test/fkine_test.py（离线脚本，无需硬件）

【预期输出】
    ①~⑦ 七节控制台信息：两实现位置差在 1e-12 m 量级（数值一致），末尾打印分析
    结论表；收尾时「未连接……豁免跳过」类日志属正常提示，非报错。
==============================================================================
"""

import sys                # 标准库：解释器路径操作（未安装库时把上级目录加入搜索路径）
import unicodedata        # 标准库：字符宽度判定（表格中中文按 2 列对齐用）
from pathlib import Path  # 标准库：跨平台路径处理

import numpy as np  # 数值计算：向量/矩阵运算与格式化

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))        # 未安装 joyarm_core 时也可直接运行
from joyarm_core import joyArm                                         # oyArm 唯一创建入口
from joyarm_core.robotics.fkine import MdhFkineSolver, PinFkineSolver  # 受测双实现
from joyarm_core.utils.limits import rand_within_limits                # 软限位内随机采样（固定种子）


# ============================================================
# 表格与几何辅助（全脚本共用）
# ============================================================
def _disp_width(text: str) -> int:
    """字符串的终端显示宽度（中日韩全角字符按 2 列计，其余按 1 列）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "FW" else 1 for ch in text)  # 逐字符累计显示列宽


def _pad(text: str, width: int) -> str:
    """右侧补空格至指定显示宽度（表格列对齐；超宽原样返回不截断）。"""
    return text + " " * max(0, width - _disp_width(text))  # 差多少列补多少空格


def _fmt_vec(vec, nd: int = 3) -> str:
    """向量格式化为 [+0.123 -0.456 +0.789]（定宽正负号数字，表格列对齐用）。"""
    return "[" + " ".join(f"{float(v):+.{nd}f}" for v in vec) + "]"  # 每个元素统一 +/− 号与小数位数


def _show(value, fmt: str = "{:.1e}") -> str:
    """表格单元格显示规则：None 显「—」（不适用/不可得），其余按给定格式化。"""
    return "—" if value is None else fmt.format(value)  # 空值与数值统一走一处格式化


def quat_angle(q1, q2) -> float:
    """两单位四元数的姿态角距（rad）：θ = 2·arccos(|q1·q2|)，姿态相同则为 0。"""
    dot = abs(float(np.dot(q1, q2)))              # 取点积绝对值：四元数 q 与 -q 表示同一姿态，消除双号歧义
    return 2.0 * float(np.arccos(min(1.0, dot)))  # 裁剪进 arccos 定义域 [0,1]，防浮点误差越界


# ============================================================
# 鸭子机械臂：演示求解器只按"属性约定"消费 arm
# ============================================================
class DuckArm:
    """最小鸭子机械臂：仅暴露 fkine 求解器实际消费的两个成员。

    求解器接口 .solve(arm, q, frame) 不检查 arm 的类型，只访问其成员——
    pin 实现读 arm.pin_model，mdh 实现另读 arm.get_config()["joyarm"]。
    任何具备这两个成员的对象都能充当 arm，即"鸭子类型"。
    """

    def __init__(self, pin_model, config):
        """记录 pin 模型与配置字典（分别供 pin / mdh 两实现消费）。"""
        self.pin_model = pin_model  # pinocchio 模型：两实现求帧索引均需要
        self._config = config       # 配置字典：仅 mdh 实现读 arm_mdh / frame_end 等键

    def get_config(self):
        """返回配置字典（与 JoyArm.get_config 同名同义的鸭子约定方法）。"""
        return self._config  # 直接返回持有引用，无额外逻辑


# ============================================================
# 各节功能函数
# ============================================================
def build_q_cases(arm) -> list:
    """构造测试位形组：zero + home + 5 组软限位内随机位形（固定种子可复现）。

    :param arm: 离线 JoyArm 对象。
    :return: [(名称, q), ...] 共 7 组，q 均为 `(n_arm,)` 弧度。
    """
    rng = np.random.default_rng(2026)                                         # 固定随机种子：每次运行采样相同，输出可复现
    qs_rand = rand_within_limits(arm.joint_limits_soft_arm, size=5, rng=rng)  # 软限位内均匀采 5 组
    cases = [("zero", np.asarray(arm.zero_arm, dtype=float)),                 # 零位形：全 0 关节角
             ("home", np.asarray(arm.home_arm, dtype=float))]                 # home 位形：cfg 归位构型
    cases += [(f"rand_q{i}", np.asarray(q, dtype=float))                      # 随机位形逐组命名并入表
              for i, q in enumerate(qs_rand, start=1)]
    return cases  # 返回 [(名称, q), ...]


def section_compare_impls(arm, frame_end: str, cases: list):
    """③ 双实现对比：末端帧上 pin 与 mdh 逐位形求位姿，打印位置差/角距表。

    :param arm: 离线 JoyArm 对象。
    :param frame_end: 末端帧名（mdh 实现唯一支持的帧）。
    :param cases: build_q_cases 产出的位形组。
    :return: (位置差最大值 m, 姿态角距最大值 rad)，供结论表汇总。
    """
    pin_fk = PinFkineSolver()  # 直接实例化 pin 黑盒实现（走 pinocchio 原生算法）
    mdh_fk = MdhFkineSolver()  # 直接实例化 mdh 白盒实现（走 cfg arm_mdh 教学数据）
    print("对比表（末端帧 %r，同一 q 喂给两实现，差值越小说明越一致）：" % frame_end)  # 表标题
    header = (_pad("位形", 9) + " " + _pad("pin 位置 (m)", 22) + " " + _pad("mdh 位置 (m)", 22)  # 表头行
              + " " + _pad("位置差 (m)", 10) + " " + _pad("角距 (rad)", 10))
    print(header)                                                       # 打印表头
    print("-" * 78)                                                     # 表头下分隔线
    mx_pos = 0.0                                                        # 位置差滚动最大值
    mx_rot = 0.0                                                        # 姿态角距滚动最大值
    for name, q in cases:                                               # 逐位形求解并填表
        p_pin = pin_fk.solve(arm, q, frame_end)                         # pin 实现求末端位姿
        p_mdh = mdh_fk.solve(arm, q, frame_end)                         # mdh 实现求末端位姿（仅末端帧可解）
        d_pos = float(np.linalg.norm(p_pin.position - p_mdh.position))  # 位置差：两位置向量欧氏距离
        d_rot = quat_angle(p_pin.orientation, p_mdh.orientation)        # 姿态差：两四元数角距
        mx_pos = max(mx_pos, d_pos)                                     # 更新位置差最大值
        mx_rot = max(mx_rot, d_rot)                                     # 更新角距最大值
        print(_pad(name, 9) + " " + _pad(_fmt_vec(p_pin.position), 22) + " "  # 本行：名称 + 两实现位置
              + _pad(_fmt_vec(p_mdh.position), 22) + " "                # mdh 位置列
              + _pad(f"{d_pos:.1e}", 10) + " " + _pad(f"{d_rot:.1e}", 10))  # 两差值列
    return mx_pos, mx_rot                                               # 返回最大差值供结论表引用


def section_frames(arm, q, frame_end: str) -> float:
    """④ 帧选择与两种用法：末端/中间/基座系位姿对比 + 门面 vs 直接实例化。

    :param arm: 离线 JoyArm 对象。
    :param q: 示例关节角（取 home 位形）。
    :param frame_end: 末端帧名。
    :return: 门面与直接实例化结果的位置差（m），供结论表汇总。
    """
    mdh_fk = MdhFkineSolver()                                                 # mdh 实例：仅末端帧可解，其余帧用于演示「—」
    frames = [("基座系(0)", 0),                                               # 整数 0 = pin 模型 0 号帧 universe（基座系）
              ("link3", "link3"),                                             # 中间连杆帧 3
              ("link6", "link6"),                                             # 中间连杆帧 6
              (frame_end, frame_end)]                                         # cfg 指定末端帧
    print("同一 q 下不同帧的位姿（mdh 位置差列显「—」= 该帧 mdh 不可解）：")  # 表标题
    header = (_pad("帧名", 14) + " " + _pad("位置 (m)", 22) + " "             # 表头行
              + _pad("姿态四元数 (w,x,y,z)", 29) + " " + _pad("mdh 位置差 (m)", 14))
    print(header)                          # 打印表头
    print("-" * 78)                        # 表头下分隔线
    for label, frame in frames:            # 逐帧求解填表
        pose = arm.solve_fkine(q, frame)   # 门面用法：激活的 pin 实现支持任意 pin 模型帧
        d_mdh = None                       # mdh 位置差缺省 None（表格显「—」）
        if label == frame_end:             # mdh 实现仅末端帧可解
            d_mdh = float(np.linalg.norm(  # 末端帧上补算 mdh 位置差（应≈0）
                pose.position - mdh_fk.solve(arm, q, frame_end).position))
        print(_pad(label, 14) + " " + _pad(_fmt_vec(pose.position), 22) + " "  # 帧名 + 位置列
              + _pad(_fmt_vec(pose.orientation), 29) + " " + _pad(_show(d_mdh), 14))  # 四元数 + mdh 差列
    p_base = arm.solve_fkine(arm.rand_q_arm(), 0)                              # 换随机 q 再求基座系位姿：验证恒等与 q 无关
    print(f"基座系位姿与 q 无关：随机 q 下位置仍为 {_fmt_vec(p_base.position, nd=1)}（恒等变换）")  # 恒等性说明

    print("-" * 78)                                                         # 小节分隔线
    print("两种调用用法（应得到完全一致的结果）：")                         # 小节标题
    p_facade = arm.solve_fkine(q, frame_end)                                # 用法一：joyArm 门面（自动路由到激活求解器）
    duck = DuckArm(arm.pin_model, arm.get_config())                         # 鸭子 arm：仅 pin_model + get_config 两成员
    p_duck_pin = PinFkineSolver().solve(duck, q, frame_end)                 # 用法二：直接实例化求解器（arm 参数鸭子类型）
    p_duck_mdh = MdhFkineSolver().solve(duck, q, frame_end)                 # mdh 同样可消费鸭子 arm（读 cfg 键）
    d_pin = float(np.linalg.norm(p_facade.position - p_duck_pin.position))  # 门面 vs 鸭子 pin 的位置差
    d_mdh = float(np.linalg.norm(p_facade.position - p_duck_mdh.position))  # 门面 vs 鸭子 mdh 的位置差
    print(f"   门面 solve_fkine 位置：{_fmt_vec(p_facade.position)}")       # 打印门面结果
    print(f"   DuckArm 上 pin / mdh 位置：{_fmt_vec(p_duck_pin.position)} / {_fmt_vec(p_duck_mdh.position)}")  # 打印鸭子结果
    print(f"   与门面结果的位置差：pin {d_pin:.1e} m，mdh {d_mdh:.1e} m（求解器只按属性约定消费 arm）")  # 一致性结论
    return d_pin                                                            # 返回门面 vs 直接求解差值


def section_exceptions(arm) -> None:
    """⑥ 异常路径：未知帧名 / mdh 非末端帧均按设计抛 ValueError，此处捕获打印。"""
    q = np.zeros(arm.n_arm)                                                      # 零向量仅作触发异常的示例输入（位形与异常无关）
    cases = [("pin 未知帧名", lambda: arm.solve_fkine(q, "no_such_frame")),      # 门面路径传入模型中不存在的帧名
             ("mdh 非末端帧", lambda: MdhFkineSolver().solve(arm, q, "link6"))]  # mdh 实现请求中间帧
    for label, trigger in cases:                                                 # 逐异常场景演示
        try:
            trigger()  # 触发异常路径
            print(f"   {label}：未抛异常（非预期行为）")  # 设计上不可达：到达即提示异常
        except ValueError as e:
            print(f"   {label} → 捕获 {type(e).__name__}：{e}")  # 捕获设计内异常并整段打印


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    print("=" * 78)                                               # 大节分隔线
    print("① 离线创建 JoyArm 并查看 fkine 域求解器（无需硬件）")  # 大节标题
    arm = joyArm("rebot_arm_b601_dm")                             # 离线构造：装配 URDF/pin 模型与算法层，不触硬件
    cfg = arm.get_config()["joyarm"]                              # 取配置深拷贝快照：末端帧名与 arm_mdh 的数据来源
    frame_end = str(cfg["frame_end"])                             # cfg 指定的末端帧名（pin 模型帧）
    print(f"   fkine 域已加载求解器：{arm.get_fkine_solvers()}")  # 打印域内注册名（字母序）
    print(f"   当前激活：{arm.get_activated_fkine_solver()}")     # 打印缺省激活实现
    arm.set_fkine_solver("pin_fkine_solver")                      # 显式固定激活为 pin 实现（幂等；门面调用走它）
    print(f"   末端帧名 cfg joyarm.frame_end = {frame_end!r}；"
          f"pin 模型共 {len(arm.pin_model.frames)} 个帧")              # 末端帧与模型帧总数
    mdh_arr = np.asarray(cfg["arm_mdh"], dtype=float)                  # mdh 教学参数：6 行 × [α, a, d, θ_offset]
    print(f"   cfg joyarm.arm_mdh（{mdh_arr.shape[0]} 行，mdh 白盒实现的数据源，逐关节打印）：")  # 参数矩阵标题
    for i, row in enumerate(mdh_arr, start=1):                         # 逐关节打印 MDH 参数行
        print(f"      关节{i} [α, a, d, θ_offset] = {_fmt_vec(row)}")  # 打印一行 MDH 参数
    print("   说明：PinFkineSolver 调 pinocchio（任意帧可用）；"
          "MdhFkineSolver 调上表教学数据且仅支持末端帧")  # 双实现定位一句话

    print("=" * 78)             # 大节分隔线
    print("② 测试位形组：zero + home + 5 组软限位内随机位形（固定种子 2026）")  # 大节标题
    cases = build_q_cases(arm)  # 构造 7 组位形 [(名称, q), ...]
    for name, q in cases:       # 逐组打印关节角（保留 3 位小数便于阅读）
        print(f"   {_pad(name, 9)}：{_fmt_vec(q)} rad")  # 打印一组位形

    print("=" * 78)                                                # 大节分隔线
    print("③ 双实现对比：PinFkineSolver vs MdhFkineSolver")        # 大节标题
    mx_pos, mx_rot = section_compare_impls(arm, frame_end, cases)  # 跑对比表并取回最大差值

    print("=" * 78)                                                     # 大节分隔线
    print("④ 帧选择与两种用法：末端帧 / 中间帧 / 基座系（home 位形）")  # 大节标题
    d_facade = section_frames(arm, arm.home_arm, frame_end)             # 跑帧对比与两种用法演示

    print("=" * 78)                                                # 大节分隔线
    print("⑤ 往返准备：FK(q)→pose 的思路留给 test/ikine_test.py")  # 大节标题
    print("   本例的『q → 末端位姿』采样即逆运动学的目标位姿来源：test/ikine_test.py 将做 pose → q̂ 往返验证")  # 承上启下一句话

    print("=" * 78)          # 大节分隔线
    print("⑥ 异常路径：未知帧名 / mdh 非末端帧")  # 大节标题
    section_exceptions(arm)  # 跑异常捕获演示

    print("=" * 78)                                                                           # 大节分隔线
    print("⑦ 分析结论表")                                                                     # 大节标题
    rows = [                                                                                  # 结论行列表（项目, 结论），数值来自上文实测
        ("测试位形", f"{len(cases)} 组（zero + home + 5×软限位内 rand_q，种子 2026）"),       # 位形组规模
        ("pin↔mdh 位置差最大", f"{mx_pos:.1e} m（两实现数值一致）"),                          # 位置一致性结论
        ("pin↔mdh 姿态角距最大", f"{mx_rot:.1e} rad（同姿态）"),                              # 姿态一致性结论
        ("pin 支持帧", "全部 pin 模型帧 + 整数 0（基座系 universe，恒等位姿）"),              # pin 帧能力
        ("mdh 支持帧", f"仅末端帧 {frame_end}（依赖 cfg arm_mdh 教学数据）"),                 # mdh 帧限制
        ("门面 vs 直接求解", f"位置差 {d_facade:.1e} m（鸭子 arm 结果一致）"),                # 两种用法一致性
        ("异常路径", "未知帧名 / mdh 非末端帧均按预期抛 ValueError"),                         # 异常路径结论
        ("下一步（test/ikine_test.py）", "用本例 FK(q)→pose 的位姿作 ikine 目标做往返验证"),  # 承接 ikine 测试
    ]
    print(_pad("项目", 22) + "结论")  # 结论表表头
    print("-" * 78)                   # 表头下分隔线
    for key, val in rows:             # 逐行打印结论
        print(_pad(key, 22) + val)    # 左列项目名对齐 + 右列结论

    print("=" * 78)  # 大节分隔线
    arm.close()      # 安全收尾（幂等；未连接时仅停线程清缓存并打提示日志）
    print("离线测试运行完毕（收尾「未连接……豁免跳过」类日志为正常提示）")  # 结束语


if __name__ == "__main__":  # 直接运行本文件时执行主流程
    main()                  # 进入主流程
