"""
==============================================================================
trajectory 轨迹规划测试分析  (test 离线测试)
==============================================================================
【功能概要】
    面向 robotics 六域之一的 trajectory 轨迹规划域做离线测试分析（无需硬件，数秒跑完）：
      1. 查看轨迹域注册表 REGISTRY 与 joyArm 已加载/默认激活的规划器，打印 TrajFrame 轨迹帧字段表；
      2. 测试 single_joint_direct_traj_planner（单帧关节域直传）：单帧目标四域透传（缺失为
         None）、无效目标回退 home 且三域置 None、多帧告警去重、未规划先采样抛错；
      3. 测试 single_joint_cubic_traj_planner（三次多项式）：[t0,t0+T] 均匀采样序列表、四边界
         约束校验、超末保持、过期目标恒定发布、多帧告警、未规划采样抛错；
      4. 同一目标下双规划器采样序列并排对比，说明 direct 忽略目标时刻 t——这正是
         quick_start/06_fsm_motion.py 演示回放前切 cubic 的原因；
      5. 末尾打印分析结论表（两规划器行为逐项对照 + 断言统计；None 一律显「—」）。

【原理简介】
    轨迹规划器把"带到达时刻的目标"加工成随时间平滑的轨迹：plan(arm, target) 结合机械臂当前
    状态算出轨迹并缓存于规划器内部，sample(t) 再按绝对时刻 t 取回该时刻的轨迹帧 TrajFrame。
    - direct（单帧关节域直传）：plan 只取单帧目标（q 锚定有效性），任意时刻采样恒透传该帧
      q/dq/ddq/tau（缺失域为 None，不补 0），平滑与限速完全交给电机内部与 backend 承担；
    - cubic（三次多项式）：由起点 (q0,dq0) 与终点 (q1,dq1) 的四个边界约束解析解
      q(τ)=c0+c1τ+c2τ²+c3τ³，其中 T=目标时刻−规划时刻、采样时 τ 钳位 [0,T]（超出保持末端
      状态），目标已过期则不规划多项式、直接恒定发布目标；
    - TrajFrame 为纯数据轨迹帧：time + 末端 pose/twist/accel/wrench + 关节 q/dq/ddq/tau 共
      九个可选字段，本测试脚本只填 time/q/dq/tau 四个（完整字段表见运行输出①）。
    cubic 的规划起点取自 arm.get_state_arm()；离线未连接的真机该接口返回 None，故本测试脚本用
    接口同名的鸭子替身 StubArm 提供确定性起点。

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（在 joyarm_code/ 目录下；离线脚本，无需连接机械臂硬件）
    python test/traj_test.py

【预期输出】
    ①~⑤ 五节控制台信息：轨迹域注册表与 TrajFrame 字段表、两规划器逐项测试（含三次多项式
    采样序列表与四边界校验）、双规划器并排对比表、末尾分析结论表与断言统计；离线收尾时
    「arm 未连接……豁免跳过」类日志属正常提示，非报错。
==============================================================================
"""

import logging          # 捕获规划器的 WARNING 告警（验证多帧目标的告警去重行为）
import sys              # 把 joyarm_code/ 插入模块搜索路径（未 pip 安装也能直接运行）
import time             # 构造目标帧的绝对到达时刻与各采样时刻
import unicodedata      # 计算字符串终端显示宽度（中文等全角字符占 2 列），保证表格列对齐
from pathlib import Path  # 定位本文件所在目录，进而定位上一级的 joyarm_code/

import numpy as np      # 向量构造与数值断言（allclose / 最大误差等）

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 保证未安装 joyarm_core 时可直接运行
from joyarm_core import JointState, TrajFrame, joyArm  # 数据类 + 离线机械臂创建入口
from joyarm_core.robotics.trajectory import (    # 轨迹域：注册表 + 基类 + 双规划器
    REGISTRY,                                    # 轨迹域注册表（注册名 → 规划器类）
    SingleJointCubicTrajPlanner,                 # cubic 三次多项式规划器（被测对象之一）
    SingleJointDirectTrajPlanner,                # direct 单帧关节域直传规划器（被测对象之二）
    TrajPlanner,                                 # 规划器基类（继承关系核对用）
)

T_DEMO = 2.0     # cubic 演示段时长 T（秒）：目标到达时刻 = 规划时刻 + T_DEMO
N_SAMPLE = 10    # [t0, t0+T] 均匀采样的分段数（序列表共 N_SAMPLE+1 行，另加 1 行越末钳位行）
DQ0_DEMO = 0.05  # 规划起点关节速度 dq0（rad/s，全关节同值）：验证非零初速边界约束
DQ1_DEMO = 0.02  # 目标终点关节速度 dq1（rad/s，全关节同值）：验证非零末速边界约束

RESULTS = []     # 全部断言结果 [(名称, 是否通过)]，供 ⑤ 的统计汇总


# ============ 输出格式化辅助（大节/小节/表格/断言打印） ============
def section(title: str) -> None:
    """打印大节标题：上下各一条 78 字符 '=' 分隔线。"""
    print("=" * 78)                                              # 上边框
    print(title)                                                 # 节标题
    print("=" * 78)                                              # 下边框


def sub(title: str) -> None:
    """打印小节标题：上方一条 78 字符 '-' 分隔线。"""
    print("-" * 78)                                              # 分隔线
    print(title)                                                 # 小节标题


def fmt(v, nd: int = 3) -> str:
    """把标量/向量格式化为紧凑字符串；None 一律显「—」（终端输出约定）。"""
    if v is None:                                                # None → 「—」
        return "—"
    return np.array2string(np.round(np.asarray(v, dtype=float), nd), separator=" ")  # 如 [ 0.3 -0.75 …]


def _disp_width(s: str) -> int:
    """字符串的终端显示宽度：全角/宽字符（如中文）计 2 列，其余计 1 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)  # W=宽字符 F=全角


def _pad(s: str, width: int) -> str:
    """右侧补空格至给定显示宽度（供表格列对齐；不足才补，超宽不截断）。"""
    return s + " " * max(0, width - _disp_width(s))              # 按显示宽度补齐


def print_table(headers: list, rows: list) -> None:
    """打印列对齐表格：首行列名、次行横线、随后逐行数据（列宽取该列最大显示宽度）。"""
    widths = [max(_disp_width(h), *(_disp_width(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("   " + "  ".join(_pad(h, w) for h, w in zip(headers, widths)))   # 表头行
    print("   " + "  ".join("-" * w for w in widths))                       # 表头下的分隔线
    for r in rows:                                                          # 数据行逐行打印
        print("   " + "  ".join(_pad(c, w) for c, w in zip(r, widths)))


def check(name: str, ok: bool) -> bool:
    """登记并打印一条断言（✅ 通过 / ❌ 失败），返回 ok 供需要分支的调用方使用。"""
    RESULTS.append((name, ok))                                   # 汇总进 ⑤ 的断言统计
    print(f"   [{'✅' if ok else '❌'}] {name}")
    return ok


def expect_raise(name: str, exc_type, fn) -> bool:
    """断言调用 fn() 会抛出 exc_type 异常：抛了判通过并展示消息首行，没抛判失败。"""
    try:
        fn()                                                     # 受检调用（预期抛错）
    except exc_type as e:                                        # 捕到预期类型的异常
        first = str(e).splitlines()[0][:52]                      # 只取消息首行前 52 字符（防刷屏）
        return check(f"{name}（消息首行：{first}…）", True)
    return check(f"{name}：竟未抛出 {exc_type.__name__}", False)


def invalid_targets(n: int) -> list:
    """构造五类无效目标（direct 与 cubic 同一判定口径）。

    :param n: 机械臂自由度（用于构造"维度不符"的 q）。
    :return: [(目标值, 说明文字), ...]，依次为 q 缺失 / q 空数组 / q 维度不符 / None / 空列表。
    """
    return [(TrajFrame(time=time.time() + 1.0), "TrajFrame()：q 为 None"),  # 无效 1：q 缺失
            (TrajFrame(time=time.time() + 1.0, q=np.zeros(0)), "q 为空数组（size=0）"),  # 无效 2：q 空数组
            (TrajFrame(time=time.time() + 1.0, q=np.zeros(n - 1)), f"q 维度不符（≠ n={n}）"),  # 无效 3：维度不符
            (None, "目标为 None"),                               # 无效 4：目标 None
            ([], "目标为空列表 []")]                             # 无效 5：空列表


# ============ 鸭子替身与告警计数器 ============
class StubArm:
    """鸭子替身机械臂：只实现两规划器 plan 所需的最小接口，名字与 JoyArm 公开接口同名。

    direct 的 plan 只读 home_arm（定维与回退基准）；cubic 的 plan 另调 get_state_arm()
    取规划起点 (q, dq)。离线未连接的真机 get_state_arm() 返回 None（cubic 将因取不到
    起点而抛 RuntimeError），故用本替身提供确定性的起点状态。
    """

    def __init__(self, home_arm) -> None:
        """以真实 home 位形构造替身：当前状态置于 home 位形、零速度。"""
        self.home_arm = np.asarray(home_arm, dtype=float).copy()  # home 位形（无效目标的回退基准）
        self._st = JointState(q=self.home_arm.copy(),             # 关节状态快照：q = home
                              dq=np.zeros_like(self.home_arm))    # dq = 0（可经 set_state 修改）

    def get_state_arm(self):
        """返回内部维护的关节状态快照（cubic 的 plan 以其 q/dq 为规划起点）。"""
        return self._st                                           # 恒返回同一可变快照（接口语义同真机）

    def set_state(self, q, dq) -> None:
        """设定替身当前状态：模拟机械臂正处在位形 q、以关节速度 dq 运动。"""
        self._st.q = np.asarray(q, dtype=float).copy()            # 更新关节位置
        self._st.dq = np.asarray(dq, dtype=float).copy()          # 更新关节速度


class _WarnCounter(logging.Handler):
    """挂在规划器模块 logger 上的告警计数器：捕获 WARNING 消息，验证"同一多帧目标只告警一次"。"""

    def __init__(self) -> None:
        """构造：告警消息列表置空，处理级别设为 WARNING。"""
        super().__init__(level=logging.WARNING)                  # 只关心 WARNING 及以上
        self.messages: list = []                                 # 按到达顺序缓存的告警文本

    def emit(self, record: logging.LogRecord) -> None:
        """Handler 钩子：把一条日志的消息文本追加进列表（自身不做终端输出）。"""
        self.messages.append(record.getMessage())                # 只留消息文本，便于计数与展示


def main() -> None:
    """离线主流程：① 测试环境 → ② direct 测试 → ③ cubic 测试 → ④ 双规划器对比 → ⑤ 分析结论。"""
    counter = _WarnCounter()                                     # 告警捕获/计数器（两规划器共用）
    for name in ("joyarm_core.traj_planner_single_joint_direct",  # direct 规划器源码的模块 logger 名
                 "joyarm_core.traj_planner_single_joint_cubic"):  # cubic 规划器源码的模块 logger 名
        lg = logging.getLogger(name)                             # 取该模块 logger
        lg.setLevel(logging.WARNING)                             # 确保告警不被日志级别过滤
        lg.addHandler(counter)                                   # 之后 plan 产生的告警都进计数器

    # ================= ① 测试环境 =================
    section("① 测试环境：trajectory 域注册表、TrajFrame 结构与鸭子替身 StubArm")
    arm = joyArm("rebot_arm_b601_dm")                            # 离线创建机械臂（不连硬件）
    print(f"   traj 域已加载规划器：{arm.get_traj_solvers()}")   # cfg 中 trajectory 列表的注册名
    print(f"   traj 域默认激活：{arm.get_activated_traj_solver()}（cfg 列表首位 = 默认激活）")
    n = arm.n_arm                                                # 本体自由度（本型号为 6）
    print(f"   本体自由度 n_arm = {n}，home_arm（rad）= {fmt(arm.home_arm)}")  # 回退基准位形
    sub("1.1 注册表 REGISTRY：注册名 → 规划器类（应均继承基类 TrajPlanner）")
    print_table(["注册名", "规划器类", "继承 TrajPlanner"],
                [[name, cls.__name__,                            # 注册名与类名
                  "是" if issubclass(cls, TrajPlanner) else "否"]  # 逐项验证继承关系
                 for name, cls in sorted(REGISTRY.items())])     # 按注册名排序，输出顺序稳定
    sub("1.2 TrajFrame 轨迹帧字段（joyarm_core/utils/types.py；九字段均可缺省 None → 显「—」）")  # 小节标题
    print_table(["字段", "类型", "含义"],                             # 字段表：三列
                [["time", "float", "绝对时间戳（Unix 秒）"],          # 字段 1：time
                 ["pose", "Pose", "末端位姿参考（笛卡尔任务）"],      # 字段 2：pose
                 ["twist", "Twist", "末端速度参考"],                  # 字段 3：twist
                 ["accel", "Accel", "末端加速度参考"],                # 字段 4：accel
                 ["wrench", "Wrench", "末端力/力矩参考"],             # 字段 5：wrench
                 ["q", "ndarray (n,)", "关节位置参考（rad）"],        # 字段 6：q
                 ["dq", "ndarray (n,)", "关节速度参考（rad/s）"],     # 字段 7：dq
                 ["ddq", "ndarray (n,)", "关节加速度参考（rad/s²）"],  # 字段 8：ddq
                 ["tau", "ndarray (n,)", "前馈力矩参考（N·m）"]])     # 字段 9：tau
    print("   本测试脚本只填 time/q/dq/tau 关节域字段：direct 透传 q/dq/tau（time 仅回填采样时刻），cubic 用 time/q/dq（见②③）。")  # 本脚本用法
    sub("1.3 鸭子替身 StubArm：给 cubic 提供确定的规划起点")     # 小节标题
    print("   direct 的 plan 只读 arm.home_arm；cubic 的 plan 另调 arm.get_state_arm() 取起点 (q,dq)。")  # 两规划器差异
    print(f"   离线未连接的真机 get_state_arm() = {fmt(arm.get_state_arm())}（无快照，cubic 取不到起点），")  # 离线真机行为
    print("   故以接口同名的替身 StubArm 提供确定起点（替身定义见本文件下方 StubArm 类）。")  # 替身动机
    stub = StubArm(arm.home_arm)                                 # 替身：home 与真机一致、状态可控
    q0 = stub.home_arm.copy()                                    # 规划起点位形：home
    dq0 = np.full(n, DQ0_DEMO)                                   # 规划起点速度：全关节 0.05 rad/s
    q_goal = stub.home_arm + np.round(np.linspace(0.30, 0.05, n), 2)  # 目标位形：home 上叠单调小偏移
    dq1 = np.full(n, DQ1_DEMO)                                   # 目标终点速度：全关节 0.02 rad/s
    stub.set_state(q0, dq0)                                      # 设定替身当前状态（即 cubic 的起点）
    print(f"   设定起点 q0（rad）= {fmt(q0)}，dq0（rad/s）= {fmt(dq0)}")
    print(f"   设定目标 q_goal（rad）= {fmt(q_goal)}，dq1（rad/s）= {fmt(dq1)}")
    frames_multi = [TrajFrame(time=time.time() + 1.0, q=stub.home_arm + 0.1),  # 多帧目标第一帧
                    TrajFrame(time=time.time() + 2.0, q=stub.home_arm + 0.2)]  # 第二帧（两帧即"多帧"）

    # ================= ② direct 测试 =================
    section("② single_joint_direct_traj_planner（单帧关节域直传）测试")
    sub("2.1 未规划先采样 → RuntimeError")
    p_fresh = SingleJointDirectTrajPlanner()                     # 新规划器：尚未 plan 过
    expect_raise("direct：未 plan 先 sample 抛 RuntimeError", RuntimeError,
                 lambda: p_fresh.sample(time.time()))            # 期望抛错而非返回默认帧
    sub("2.2 单帧目标：plan 后任意时刻逐域透传（q/dq/tau 透传、ddq 缺失为 None）")
    p_dir = SingleJointDirectTrajPlanner()                       # direct 主测试实例
    frame_goal = TrajFrame(time=time.time() + 5.0, q=q_goal,     # 故意填 time/q/dq/tau——q 锚定有效性
                           dq=np.ones(n), tau=np.ones(n))        # ddq 缺失（验证缺失域原样 None）
    ret = p_dir.plan(stub, frame_goal)                           # 规划：四域原子发布
    fr_a = p_dir.sample(time.time())                             # 立即采样
    fr_b = p_dir.sample(time.time() + 7.0)                       # 7 秒后采样（任意时刻）
    print(f"   plan 返回 = {ret}；帧 time 字段仅回填各次采样时刻，q/dq/ddq/tau 如下：")  # plan 结果说明
    print_table(["采样时刻", "q（rad）", "dq（rad/s）", "ddq（rad/s²）", "tau（N·m）"],  # 采样对比表五列
                [["now", fmt(fr_a.q), fmt(fr_a.dq), fmt(fr_a.ddq), fmt(fr_a.tau)],  # 立即采样行
                 ["now+7s", fmt(fr_b.q), fmt(fr_b.dq), fmt(fr_b.ddq), fmt(fr_b.tau)]])  # 7 秒后采样行
    check("direct：任意时刻逐域透传（q/dq/tau 恒为目标值、ddq 缺失为 None）",  # 断言：恒定透传
          ret is True and np.allclose(fr_a.q, q_goal) and np.allclose(fr_b.q, q_goal)  # 条件一：两次采样 q 均为目标
          and np.allclose(fr_a.dq, 1.0) and np.allclose(fr_b.dq, 1.0)  # 条件二：dq 原样透传
          and np.allclose(fr_a.tau, 1.0) and fr_a.ddq is None and fr_b.ddq is None)  # 条件三：tau 透传、ddq 缺失
    keep0 = q_goal[0]                                            # 记录目标首关节原值
    q_goal[0] = keep0 + 9.9                                      # 外部原地修改已发布的目标（试发布隔离性）
    check("direct：目标发布为拷贝，外部原地修改不影响采样",
          np.allclose(p_dir.sample(time.time()).q[0], keep0))    # 采样应仍是原值
    q_goal[0] = keep0                                            # 恢复目标值（③④ 继续用同一目标）
    sub("2.3 无效目标（q 缺失/空/维度不符/目标空）：回退恒定 home 且三域置 None，plan 仍返回 True")
    rows, all_ok = [], True                                      # 表格行积累器 / 总判定
    for bad, why in invalid_targets(n):                          # 五类无效目标（与 ③ 同口径）
        ok = p_dir.plan(stub, bad) is True                       # plan 恒成功（直传无离线失败路径）
        fr = p_dir.sample(time.time())                           # 采样（应恒为 home）
        none3 = fr.dq is None and fr.ddq is None and fr.tau is None  # 三域应一并置 None
        hit = ok and np.allclose(fr.q, stub.home_arm) and none3  # 回退 home 且三域 None 才算通过
        all_ok = all_ok and hit                                  # 汇总本小节判定
        rows.append([why, "True" if ok else "—", fmt(fr.q), "是" if none3 else "否",  # 本行结果入表
                     "✅" if hit else "❌"])                      # 行判定
    print_table(["无效目标", "plan 返回", "采样 q（rad）", "dq/ddq/tau 均 None", "判定"], rows)  # 无效目标结果表
    check("direct：五类无效目标均回退恒定 home 且 dq/ddq/tau 置 None", all_ok)  # 断言：五类全回退
    sub("2.4 多帧目标：告警一次并回退 home；单帧规划后告警去重复位")  # 小节标题
    p_m = SingleJointDirectTrajPlanner()                         # 新实例：告警去重标志为初始 False
    counter.messages.clear()                                     # 清零计数，只观察本小节的告警
    for _ in range(3):                                           # 同一多帧目标重复规划 3 次
        p_m.plan(stub, frames_multi)                             # 每次都因不支持多帧而回退 home
    check("direct：同一多帧目标重复 plan 3 次仅告警 1 次（去重）", len(counter.messages) == 1)
    fr_m = p_m.sample(time.time())                               # 多帧回退后的采样
    check("direct：多帧目标回退恒定 home 且三域置 None",
          np.allclose(fr_m.q, stub.home_arm) and fr_m.dq is None and fr_m.tau is None)  # 采样恒 home
    counter.messages.clear()                                     # 再清零，验证复位语义
    p_m.plan(stub, TrajFrame(q=q_goal))                          # 有效单帧规划 → 复位去重标志
    mid = len(counter.messages)                                  # 应为 0（单帧规划不告警）
    p_m.plan(stub, frames_multi)                                 # 再遇多帧 → 复位后应再次告警
    check("direct：单帧规划复位告警去重（再遇多帧会再次告警）", mid == 0 and len(counter.messages) == 1)
    print(f"   告警消息样例：{counter.messages[-1][:56]}…")      # 展示源码内置告警文案的开头

    # ================= ③ cubic 测试 =================
    section("③ single_joint_cubic_traj_planner（三次多项式）测试")
    sub("3.1 未规划先采样 → RuntimeError（与 direct 同）")
    p_fresh2 = SingleJointCubicTrajPlanner()                     # 新规划器：尚未 plan 过
    expect_raise("cubic：未 plan 先 sample 抛 RuntimeError", RuntimeError,
                 lambda: p_fresh2.sample(time.time()))           # 期望抛错
    sub(f"3.2 单帧目标：[t0, t0+T] 均匀采样序列（此处 T = {T_DEMO:.1f}s = 目标时刻−规划时刻）")
    stub.set_state(q0, dq0)                                      # 起点：home 位形 + 非零初速 dq0
    p_seq = SingleJointCubicTrajPlanner()                        # 采样序列/边界校验共用的实例
    t_pre = time.time()                                          # plan 调用前的时刻：内部 t0 ≥ t_pre
    ret = p_seq.plan(stub, TrajFrame(time=t_pre + T_DEMO, q=q_goal, dq=dq1))  # T_DEMO 秒后到达目标
    print(f"   plan 返回 = {ret}；标称 τ = 0 ~ {T_DEMO:.1f}s 共 {N_SAMPLE + 1} 行，另加 1 行越末钳位行")
    q_rows, dq_rows = [], []                                     # 位置表/速度表的行积累器
    for tau in np.linspace(0.0, T_DEMO, N_SAMPLE + 1):           # 均匀 τ 网格（首行被钳位到 τ=0）
        fr = p_seq.sample(t_pre + float(tau))                    # 采样时刻锚定 t_pre（与 t0 同基准）
        q_rows.append([f"{tau:.2f}"] + [f"{v:+.3f}" for v in fr.q])  # 该时刻各关节位置（rad）
        dq_rows.append([f"{tau:.2f}"] + [f"{v:+.3f}" for v in fr.dq])  # 该时刻各关节速度（rad/s）
    fr_over = p_seq.sample(time.time() + T_DEMO + 1.0)           # 越末时刻采样（τ 被钳位到 T）
    q_rows.append(["钳位→T"] + [f"{v:+.3f}" for v in fr_over.q])  # 越末行：位置应为终点 q1
    dq_rows.append(["钳位→T"] + [f"{v:+.3f}" for v in fr_over.dq])  # 越末行：速度应为终点 dq1
    print(f"   位置 q（rad），各列 = 关节 J1~J{n}：")            # 位置表标题
    print_table(["τ(s)"] + [f"J{i + 1}" for i in range(n)], q_rows)  # 位置采样序列表
    print(f"   速度 dq（rad/s），各列 = 关节 J1~J{n}：")         # 速度表标题
    print_table(["τ(s)"] + [f"J{i + 1}" for i in range(n)], dq_rows)  # 速度采样序列表
    sub("3.3 四边界约束校验：起点 (q0,dq0)、终点 (q1,dq1)（钳位采样得精确边界）")
    fr_s = p_seq.sample(t_pre)                                   # 时刻 ≤ t0 → τ 精确为 0（起点）
    fr_e = p_seq.sample(t_pre + T_DEMO + 5.0)                    # 时刻 ≥ t0+T → τ 精确为 T（终点）
    b_rows, b_all = [], True                                     # 边界表行积累器 / 总判定
    for name_b, expect, actual in (("q(t0)=q0", q0, fr_s.q),     # 起点位置约束
                                   ("dq(t0)=dq0", dq0, fr_s.dq),  # 起点速度约束
                                   ("q(t0+T)=q1", q_goal, fr_e.q),  # 终点位置约束
                                   ("dq(t0+T)=dq1", dq1, fr_e.dq)):  # 终点速度约束
        err = float(np.max(np.abs(np.asarray(actual) - np.asarray(expect))))  # 全关节最大绝对误差
        ok = np.allclose(actual, expect, atol=1e-9)              # 边界约束应精确成立（解析解）
        b_all = b_all and ok                                     # 汇总四项约束
        b_rows.append([name_b, f"{expect[0]:+.3f}", f"{actual[0]:+.3f}", f"{err:.2e}",  # 约束名 + 期望 + 实测
                       "✅" if ok else "❌"])                    # 行判定
    print_table(["约束", "期望(J1)", "实测(J1)", "全关节最大误差", "判定"], b_rows)
    check("cubic：四边界约束（起终点位置与速度）全部精确成立", b_all)
    p_nd = SingleJointCubicTrajPlanner()                         # 附带验证：目标 dq 缺省的默认行为
    p_nd.plan(stub, TrajFrame(time=time.time() + 0.5, q=q_goal))  # 不填 dq（即 dq=None）
    check("cubic：目标 dq 缺省（None）时终点速度取 0",
          np.allclose(p_nd.sample(time.time() + 1.0).dq, 0.0))   # 越末采样 → 末端 dq 应为 0
    sub("3.4 超末保持：采样时刻远超 t0+T，恒保持末端状态 (q1, dq1)")
    fr_x1 = p_seq.sample(time.time() + T_DEMO + 10.0)            # 超末 10 秒采样
    fr_x2 = p_seq.sample(time.time() + T_DEMO + 60.0)            # 超末 60 秒采样（更远）
    check("cubic：远超末端的任意采样恒为末端状态",               # 断言：超末保持
          np.allclose(fr_x1.q, q_goal) and np.allclose(fr_x2.q, q_goal)
          and np.allclose(fr_x1.dq, dq1) and np.allclose(fr_x2.dq, dq1))
    sub("3.5 过期目标（到达时刻 < 当前时刻）：不规划多项式，恒定发布目标 q1")  # 小节标题
    p_exp = SingleJointCubicTrajPlanner()                        # 新实例：干净规划状态
    ret_e = p_exp.plan(stub, TrajFrame(time=time.time() - 1.0,   # 到达时刻已过 1 秒（过期目标）
                                       q=q_goal, dq=dq1))
    fr_e0 = p_exp.sample(time.time())                            # 立即采样
    fr_e9 = p_exp.sample(time.time() + 10.0)                     # 远期采样
    print(f"   plan 返回 = {ret_e}；立即/远期采样 q 恒为目标、dq 恒 0（维持终点状态而非解多项式）")  # 过期行为说明
    check("cubic：过期目标恒定发布目标 q1（dq 恒 0）",           # 断言：过期目标恒定发布
          ret_e is True and np.allclose(fr_e0.q, q_goal) and np.allclose(fr_e9.q, q_goal)  # 条件一：立即/远期 q 均为目标
          and np.allclose(fr_e9.dq, 0.0))                        # 条件二：dq 恒 0
    sub("3.6 无效目标：与 direct 同口径的五类无效 → 回退恒定 home（零系数退化段）")
    p_bad = SingleJointCubicTrajPlanner()                        # 新实例：干净规划状态
    ok_all = True                                                # 五类无效目标的总判定
    for bad, why in invalid_targets(n):                          # 复用②的同一口径
        ret_b = p_bad.plan(stub, bad)                            # 无效 → 零系数退化段（恒 home）
        fr_b = p_bad.sample(time.time() + 10.0)                  # 越末采样（退化段恒定不变）
        hit = ret_b is True and np.allclose(fr_b.q, stub.home_arm) and np.allclose(fr_b.dq, 0.0)
        ok_all = ok_all and hit                                  # 汇总本小节判定
        print(f"   {why}：plan={ret_b}，采样 q=home、dq=0 → {'✅' if hit else '❌'}")  # 单类无效结果
    check("cubic：五类无效目标均回退恒定 home（dq 恒 0）", ok_all)
    sub("3.7 多帧目标：告警一次并回退恒定 home（与 direct 同语义）")
    p_m2 = SingleJointCubicTrajPlanner()                         # 新实例：干净告警标志
    counter.messages.clear()                                     # 清零计数，只观察本小节的告警
    ret_m = p_m2.plan(stub, frames_multi)                        # 两帧目标 → 告警 + 退化段回退
    fr_m = p_m2.sample(time.time() + 10.0)                       # 越末采样（退化段恒 home）
    check("cubic：多帧目标告警一次并回退恒定 home",              # 断言：告警一次 + 回退 home
          ret_m is True and len(counter.messages) == 1           # 条件一：plan 成功且恰告警一次
          and np.allclose(fr_m.q, stub.home_arm) and np.allclose(fr_m.dq, 0.0))  # 条件二：采样恒为 home
    sub("3.8 绕过封装且 time=None：q 有效但缺到达时刻 → ValueError（显性报错）")
    p_v = SingleJointCubicTrajPlanner()                          # 新实例
    expect_raise("cubic：q 有效但 time=None 抛 ValueError", ValueError,
                 lambda: p_v.plan(stub, TrajFrame(q=q_goal)))    # 手写 TrajFrame 漏填 time 的典型误用

    # ================= ④ 双规划器对比 =================
    section("④ 同一目标双规划器并排对比：direct 忽略目标时刻 t")
    stub.set_state(q0, dq0)                                      # 与 ③ 相同的起点状态（指代无歧义）
    p_d2 = SingleJointDirectTrajPlanner()                        # 对比用 direct 实例
    p_c2 = SingleJointCubicTrajPlanner()                         # 对比用 cubic 实例
    t_cmp = time.time()                                          # 对比锚定时刻（在两个 plan 之前捕获）
    tgt = TrajFrame(time=t_cmp + T_DEMO, q=q_goal, dq=dq1)       # 同一目标帧：T_DEMO 秒后到达 q_goal
    p_d2.plan(stub, tgt)                                         # direct 规划：透传 q/dq、忽略 time
    p_c2.plan(stub, tgt)                                         # cubic 规划：解三次多项式（T 生效）
    rows_c, direct_const = [], True                              # 对比表行积累器 / direct 恒定性判定
    for tau in np.linspace(0.0, T_DEMO, N_SAMPLE + 1):           # 同一 τ 网格并排采样
        fh = p_d2.sample(t_cmp + float(tau))                     # direct 采样（恒为目标帧）
        fc = p_c2.sample(t_cmp + float(tau))                     # cubic 采样（沿多项式平滑过渡）
        direct_const = direct_const and np.allclose(fh.q, q_goal) \
            and np.allclose(fh.dq, dq1)                          # 逐行验证 direct 恒透传 q/dq
        rows_c.append([f"{tau:.2f}", f"{fh.q[0]:+.3f}", f"{fh.dq[0]:+.3f}",
                       f"{fc.q[0]:+.3f}", f"{fc.dq[0]:+.3f}"])   # 以关节 J1 为代表列（其余同理）
    fh = p_d2.sample(t_cmp + T_DEMO + 1.0)                       # 越末一行：direct 仍恒定
    fc = p_c2.sample(t_cmp + T_DEMO + 1.0)                       # 越末一行：cubic 钳位保持末端
    rows_c.append(["钳位→T", f"{fh.q[0]:+.3f}", f"{fh.dq[0]:+.3f}",
                   f"{fc.q[0]:+.3f}", f"{fc.dq[0]:+.3f}"])       # 越末行（τ 钳位到 T）
    print("   同一目标帧、同一 τ 网格并排采样（以关节 J1 为代表，其余关节同理）：")  # 对比表标题
    print_table(["τ(s)", "direct q(J1)", "direct dq(J1)", "cubic q(J1)", "cubic dq(J1)"], rows_c)  # 对比表
    check("对比：direct 全程恒透传目标帧 q/dq（目标时刻 t 被忽略），cubic 于 τ=T 精确到达",  # 断言：恒定 vs 到达
          direct_const and np.allclose(fc.q, q_goal) and np.allclose(fc.dq, dq1))
    print("   - direct 的 plan 只取单帧关节域：目标到达时刻 t 不参与计算，采样一步到位、q/dq 原样透传；")  # direct 行为一
    print("   - cubic 的 plan 以 T = t − 规划时刻 解三次多项式：采样沿 [0,T] 平滑过渡，τ=T 精确到达。")  # cubic 行为
    print("   - quick_start/06_fsm_motion.py 的回放以 10 Hz 逐点下发 set_target_joint(t=当前时刻+0.15, q=路点)：")  # 与 06 例程的关联
    print("     默认激活 direct 时 t 被忽略，回放退化为位置直传流（dq 缺省 None，速度仅由电机 vlim 兜底）；")  # direct 回放行为
    print("     切到 single_joint_cubic_traj_planner 后 T 才生效、路点按到达时刻逐段三次平滑——")  # cubic 回放行为
    print("     这正是 quick_start/06_fsm_motion.py 演示回放前将 traj 域切到 cubic 规划器的原因。")  # 关联结论

    # ================= ⑤ 分析结论 =================
    section("⑤ 分析结论")                                        # 大节标题
    print_table(["测试项", "direct（单帧关节域直传）", "cubic（三次多项式）"],  # 结论表三列
                [["单帧目标", "任意时刻逐域透传 q/dq/ddq/tau（缺失 None）", "[t0,t0+T] 三次多项式平滑过渡，τ=T 精确到达"],  # 行 1
                 ["目标时刻 t", "忽略（不参与 plan 计算）", "决定 T = t − 规划时刻 与过渡节奏"],  # 行 2
                 ["无效目标", "回退恒定 home 且三域置 None，plan 仍返回 True", "零系数退化段恒保持 home，plan 仍返回 True"],  # 行 3
                 ["过期目标", "无时间概念，行为与单帧相同", "不规划多项式，恒定发布目标 q1（dq=0）"],  # 行 4
                 ["超末采样", "恒为目标帧（四域不变）", "τ 钳位到 T，保持末端状态 (q1, dq1)"],  # 行 5
                 ["多帧目标", "告警一次 + 回退 home，单帧规划后复位", "同左（同一语义）"],  # 行 6
                 ["未规划先采样", "RuntimeError", "RuntimeError"],  # 行 7
                 ["适用场景", "高频目标帧直传流（限速交电机与 backend）", "带到达时刻的单段平滑运动"]])  # 行 8
    ok_n = sum(1 for _, ok in RESULTS if ok)                     # 通过的断言数
    total = len(RESULTS)                                         # 断言总数
    print(f"\n   断言统计：{ok_n}/{total} 项通过" + (" ✅ 全部通过" if ok_n == total else " ❌ 存在未通过项"))  # 统计行
    print('   真机切换方法：arm.set_traj_solver("single_joint_cubic_traj_planner")'  # 切换命令
          "（新规划器 plan 成功后才替换激活）。")
    arm.close()                                                  # 显式安全收尾（幂等；离线仅停线程清缓存）
    print("离线测试运行完毕（收尾时「arm 未连接……豁免跳过」类日志为正常提示）")


if __name__ == "__main__":
    main()
