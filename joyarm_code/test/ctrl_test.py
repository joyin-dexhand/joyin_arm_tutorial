"""
==============================================================================
控制律测试分析（Controller / JointPositionDirectController）  (test 离线测试)
==============================================================================
【功能概要】
    离线测试脚本（无需硬件），对 robotics 六域之一的 control（控制律）做测试分析：
      1. 契约测试：step() 收有效轨迹帧（含 q）→ 返回 True 且经 send_position_arm
         下发，StubArm 捕获下发序列并打印；
      2. 异常路径：缺 q 帧、q 含 NaN 帧两条路径均抛 ValueError 且零下发；
      3. MODE 机制：Controller 基类缺省 MIT、子类声明所需电机模式、FSM 每拍自愈；
      4. 离线闭环小仿真：cubic 规划器 + 位置控制器 + 一阶跟随 StubArm 串联 50 拍，
         打印目标/实际轨迹对比表与到位判断，末尾汇总分析结论表。

【原理简介】
    control 域采用「基类定契约、子类实现」架构：Controller（ABC）只规定唯一接口
    step(arm, target, joint_state)——每控制拍调用一次，基于当前轨迹帧与关节状态
    算出单步指令并经 arm 下发；JointPositionDirectController 是默认实现：先校验轨迹帧
    的 q（缺失或含 NaN/inf 一律抛 ValueError 拒绝下发），再把该 q 经
    send_position_arm 直发——跟踪闭环由电机内部位置环完成，本层不反馈、不裁剪。
    类属性 MODE 声明本控制律所需的电机控制模式；FSM 每拍自愈核对：只读一次实际
    模式，与 MODE 不一致先切模式、再核使能，失败下一拍重试，指令路径始终合法。

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code && uv sync && source .venv/bin/activate

    # 2. 运行本脚本（离线测试脚本，无需连接机械臂硬件）
    python test/ctrl_test.py

【预期输出】
    ①~④ 四节——契约与 MODE 说明、三条契约用例、闭环仿真对比表
    与到位判断、分析结论表；全程数秒跑完。
==============================================================================
"""

from pathlib import Path                                    # 定位 joyarm_code/ 目录（未安装也可运行）
import sys                                                  # 修改模块搜索路径
import time                                                 # 提供规划器所需的绝对时间戳

import numpy as np                                          # 数值计算（向量运算 / NaN 判定）

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 把 joyarm_code/ 加入 sys.path
from joyarm_core import (                                   # 路径注入后再导入核心库
    Controller,                                             # 控制律基类（ABC，仅引用其 MODE 缺省值）
    JointPositionDirectController,                          # 默认实现：关节位置直传控制器（被测对象）
    SingleJointCubicTrajPlanner,                            # cubic 三次多项式规划器（闭环仿真用）
)
from joyarm_core.utils.types import (                       # 共享数据类型与枚举
    ControlMode,                                            # 电机控制模式枚举 MIT / POSITION / VELOCITY
    JointLimits,                                            # 关节限位声明（StubArm 的同名属性）
    JointState,                                             # 关节状态快照（StubArm 内部持有）
    TrajFrame,                                              # 轨迹帧（控制器 step 的目标输入）
)


# ============================================================
# 输出格式化辅助（表格列对齐、None 显「—」）
# ============================================================
def disp_len(s: str) -> int:
    """计算字符串的终端显示宽度（CJK 全角字符按 2 列计，其余按 1 列）。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)    # 逐字符累计显示宽度


def pad(s: str, width: int) -> str:
    """按显示宽度在右侧补空格到 width 列（表格列对齐用，超出则原样返回）。"""
    return s + " " * max(width - disp_len(s), 0)            # 不足 width 列才补空格


def print_table(header: list, rows: list, widths: list) -> None:
    """打印列对齐表格：header 为表头、rows 为数据行、widths 为各列显示宽度。"""
    print("    " + "".join(pad(h, w) for h, w in zip(header, widths)))  # 表头行（统一缩进 4 空格）
    for r in rows:                                          # 逐行按同宽拼接，保证列对齐
        print("    " + "".join(pad(str(c), w) for c, w in zip(r, widths)))


def vec_str(v, nd: int = 3) -> str:
    """向量转字符串：None 显「—」，数组逐元素保留 nd 位小数。"""
    return "—" if v is None else str(np.round(np.asarray(v, dtype=float), nd).tolist())  # numpy → list 更易读


# ============================================================
# 最小鸭子桩（属性/方法名与 JoyArm 公开接口同名，控制器无需修改即可调用）
# ============================================================
class StubArm:
    """离线替代真机的最小桩：记录下发指令 + 一阶跟随模型模拟关节响应。

    - 指令记录：send_position_arm 把每拍收到的位置指令存入 self.cmds；
    - 被控对象：按关节一阶跟随模型（时间常数 tau）推进 q 向指令位置靠近，
      模拟「电机内部位置环驱动关节」的闭环跟踪效果。
    """

    ctrl_hz = 50.0                                          # 控制频率 50 Hz（闭环仿真步长与之同步）
    home_arm = np.zeros(6)                                  # home 位形：六关节全零
    joint_limits_hard_arm = JointLimits(                    # 硬限位声明（与 JoyArm 同名属性）
        q_min=np.full(6, -3.0), q_max=np.full(6, 3.0),      # 位置限位 ±3 rad
        dq_max=np.full(6, 5.0), tau_max=np.full(6, 30.0))   # 速度 / 力矩上限

    def __init__(self, tau: float = 0.08):
        """初始化内部关节状态与指令记录。

        :param tau: 关节一阶跟随时间常数（秒），越小跟随越快。
        """
        self._st = JointState(q=np.zeros(6), dq=np.zeros(6))  # 当前关节状态（规划器/控制器的读数来源）
        self.cmds = []                                      # 每拍 send_position_arm 收到的指令序列
        self._tau = tau                                     # 一阶跟随时间常数

    def get_state_arm(self):
        """返回当前关节状态快照（规划器取规划起点、控制器收状态入参均用它）。"""
        return self._st                                     # 返回内部引用：便于随推进实时更新

    def send_position_arm(self, q, vlim=None, flim=None):
        """记录位置指令，并按一阶模型把关节向指令位置推进一拍。

        :param q: 目标关节位置向量 (n,)，弧度。
        :param vlim: 速度限制（桩不实现，保留签名与 JoyArm 接口一致）。
        :param flim: follow 限制（同上）。
        """
        q_cmd = np.asarray(q, dtype=float).reshape(-1)      # 统一为一维 float 向量
        self.cmds.append(q_cmd.copy())                      # 拷贝存档，防外部原地修改污染记录
        dt = 1.0 / self.ctrl_hz                             # 一拍时长 = 控制周期（20 ms）
        self._st.dq = (q_cmd - self._st.q) / self._tau      # 一阶模型：速度正比位置偏差
        self._st.q = self._st.q + self._st.dq * dt          # 位置按本拍速度推进（模拟电机位置环）


# ============================================================
# ① control 域契约与 MODE 机制
# ============================================================
def show_contract_and_mode() -> bool:
    """① 打印 step 契约与 MODE 机制说明（基类缺省 / 子类声明 / FSM 自愈）。"""
    print("=" * 78)                                         # 大节分隔线
    print("① control 域契约与 MODE 机制（Controller 基类 / JointPositionDirectController 子类）")
    print("-" * 78)                                         # 小节分隔线
    print("1.1 step 契约（Controller 唯一抽象接口，子类必须实现）")  # 契约小节标题
    print("    step(arm, target, joint_state) -> bool")     # 契约签名
    print("    入参：arm 机械臂对象（指令经它下发）；target 当前轨迹帧 TrajFrame；")  # 入参说明
    print("        joint_state 当前关节状态 JointState；返回是否已成功下发本拍指令。")  # 入参与返回值
    print("    JointPositionDirectController 实现：校验 target.q → send_position_arm 直发。")  # 默认实现说明
    print("-" * 78)                                         # 小节分隔线
    mode_ok = JointPositionDirectController.MODE == ControlMode.POSITION  # 核验子类声明的电机模式
    print("1.2 MODE 类属性——控制律声明的电机控制模式")
    print_table(                                            # 两级 MODE 对比表
        ["控制律类", "MODE 类属性", "含义"],                # 表头三列
        [["Controller（基类）", str(Controller.MODE), "缺省：子类未覆写时沿用"],  # 基类行：缺省 MIT
         ["JointPositionDirectController", str(JointPositionDirectController.MODE),
          "覆写：位置跟随所需模式"]],                      # 子类行：POSITION
        [31, 21, 29])
    print(f"    校验 JointPositionDirectController.MODE == ControlMode.POSITION：{mode_ok}")  # 覆写校验结果
    print("    机制：切换控制器时 JoyArm 校验 MODE 合法性；MOTION 态每拍 FSM 自愈——")  # 自愈机制第一句
    print("    只读一次实际模式，与 MODE 不一致先切模式、再核使能，失败下一拍重试。")  # 自愈机制第二句
    return mode_ok                                          # 供 ④ 结论表引用


# ============================================================
# ② 契约测试（有效帧下发 / 缺 q 拒绝 / NaN q 拒绝）
# ============================================================
def run_contract_tests() -> dict:
    """② 三条契约用例：StubArm 捕获下发序列，返回各用例结果供结论表使用。"""
    print("=" * 78)                                         # 大节分隔线
    print("② 契约测试：StubArm 捕获下发序列，逐条核对 step 的三条契约")
    ctrl = JointPositionDirectController()                  # 被测控制律（默认实现）
    q_ok = np.array([0.25, -0.20, 0.15, -0.15, 0.20, -0.25])  # 有效目标：六元有限向量

    print("-" * 78)                                         # 小节分隔线
    print("2.1 有效帧（含 q）→ 返回 True 且经 send_position_arm 下发")
    stub = StubArm()                                        # 全新桩：指令记录从零开始
    ret = ctrl.step(stub, TrajFrame(q=q_ok), stub.get_state_arm())  # 正常一步：应下发 q_ok
    sent_ok = len(stub.cmds) == 1 and np.allclose(stub.cmds[0], q_ok)  # 恰一条指令且等于参考
    print(f"    step 返回：{ret}；下发条数：{len(stub.cmds)}；指令 q：{vec_str(stub.cmds[0])}")
    print(f"    校验（返回 True 且指令 == 参考 q）：{'通过' if ret is True and sent_ok else '未通过'}")

    print("-" * 78)                                         # 小节分隔线
    print("2.2 缺 q 帧（TrajFrame() 的 q 默认 None）→ 抛 ValueError 且不下发")
    stub2 = StubArm()                                       # 全新桩：验证拒绝时零下发
    raised = False                                          # 是否按预期抛错
    try:
        ctrl.step(stub2, TrajFrame(), stub2.get_state_arm())  # q 缺失：预期抛 ValueError
        print("    未抛异常（契约失效）")                   # 走到这里说明未按契约拒绝
    except ValueError as e:
        raised = True                                       # 按契约抛错
        print(f"    捕获 ValueError：{e}")
    no_q = raised and len(stub2.cmds) == 0                  # 抛错且零下发才算通过
    print(f"    下发条数：{len(stub2.cmds)}；校验（抛 ValueError 且零下发）：{'通过' if no_q else '未通过'}")

    print("-" * 78)                                         # 小节分隔线
    print("2.3 NaN q 帧（第 3 关节被置为 NaN）→ 抛 ValueError 且不下发")
    stub3 = StubArm()                                       # 全新桩：同上隔离验证
    q_nan = q_ok.copy()                                     # 复制有效目标再注入坏值
    q_nan[2] = np.nan                                       # 单关节 NaN（模拟脏数据）
    raised = False                                          # 是否按预期抛错
    try:
        ctrl.step(stub3, TrajFrame(q=q_nan), stub3.get_state_arm())  # NaN 入参：预期抛错
        print("    未抛异常（契约失效）")                   # 走到这里说明未按契约拒绝
    except ValueError as e:
        raised = True                                       # NaN 兜底生效
        print(f"    捕获 ValueError：{e}")
    nan_q = raised and len(stub3.cmds) == 0                 # 抛错且零下发才算通过
    print(f"    下发条数：{len(stub3.cmds)}；校验（抛 ValueError 且零下发）：{'通过' if nan_q else '未通过'}")
    return {"valid": ret is True and sent_ok, "no_q": no_q, "nan_q": nan_q}   # 结果交 ④ 汇总


# ============================================================
# ③ 离线闭环小仿真（cubic 规划器 → 位置控制器 → 一阶跟随 StubArm）
# ============================================================
def run_closed_loop() -> dict:
    """③ 闭环仿真 50 拍：逐拍「规划器采样 → 控制器下发 → 桩推进」，返回结果。"""
    print("=" * 78)                                         # 大节分隔线
    print("③ 离线闭环小仿真：cubic 规划器 + JointPositionDirectController + 一阶跟随 StubArm")
    print("-" * 78)                                         # 小节分隔线
    q_goal = np.array([0.60, -0.45, 0.30, -0.30, 0.45, -0.60])  # 目标关节角：限位内、各关节幅值不同
    n_tick, t_traj, tol = 50, 0.9, 0.01                     # 拍数 / 轨迹时长 s / 到位容差 rad
    dt = 1.0 / StubArm.ctrl_hz                              # 一拍时长 20 ms（与 ctrl_hz 一致）
    n_hold = sum(1 for k in range(n_tick) if k * dt >= t_traj)  # 采样超出轨迹末端的拍数（保持段）
    print("3.1 仿真参数")                                   # 仿真参数小节标题
    print(f"    目标 q_goal (rad)：{vec_str(q_goal)}；终点速度 dq_goal：{vec_str(None)}（缺省取 0）")  # 目标与末速
    print(f"    轨迹时长 {t_traj:.2f} s；控制频率 {StubArm.ctrl_hz:.0f} Hz；共 {n_tick} 拍"  # 时长/频率/拍数
          f"（其中 {n_hold} 拍超出轨迹末端：规划器保持末端状态）")
    print("    数据通路：规划器 sample 出参考帧 → 控制器 step 直发 q → 桩按一阶模型推进关节")   # 数据通路说明

    stub = StubArm()                                        # 被控对象：一阶跟随（tau=0.08 s）
    planner = SingleJointCubicTrajPlanner()                 # 规划器：三次多项式到单点目标
    ctrl = JointPositionDirectController()                  # 控制器：直发位置参考（与 ② 同类）
    ok_plan = planner.plan(stub, TrajFrame(time=time.time() + t_traj, q=q_goal))  # 规划：now 起 0.9 s 到达
    t0 = time.time()                                        # 仿真零点（sample 按绝对时间求值）
    q_refs, q_acts, oks = [], [], []                        # 每拍参考 / 实际 / step 返回的记录
    for k in range(n_tick):                                 # 50 拍闭环主循环
        frame = planner.sample(t0 + k * dt)                 # 本拍参考帧（超末端自动钳位保持）
        oks.append(ctrl.step(stub, frame, stub.get_state_arm()))  # 控制器下发 frame.q，桩随之推进
        q_refs.append(np.asarray(frame.q, dtype=float).copy())  # 存参考（拷贝防篡改）
        q_acts.append(stub.get_state_arm().q.copy())        # 存本拍指令作用后的实际关节角
    sent_eq_ref = all(np.allclose(stub.cmds[k], q_refs[k]) for k in range(n_tick))  # 逐拍指令==参考

    print("-" * 78)                                         # 小节分隔线
    print("3.2 逐拍记录（每 5 拍一行；关节 0 为代表列，‖e‖ 为六关节全误差范数）")
    print_table(                                            # 目标 / 实际轨迹对比表
        ["拍", "τ(s)", "关节0 q_ref", "关节0 q_act", "‖e‖ (rad)"],
        [[f"{k}", f"{k * dt:.2f}", f"{q_refs[k][0]:+.4f}", f"{q_acts[k][0]:+.4f}",
          f"{np.linalg.norm(q_refs[k] - q_acts[k]):.5f}"]   # 行数据：拍号/时刻/代表关节/误差范数
         for k in range(n_tick) if k % 5 == 0 or k == n_tick - 1],  # 每 5 拍 + 末拍
        [6, 8, 14, 14, 12])
    print("    规律：中段参考速度大 → 一阶跟随滞后使 ‖e‖ 最大；进入保持段后误差收敛")

    print("-" * 78)                                         # 小节分隔线
    print("3.3 首 / 中 / 末拍全关节对比（q_ref 参考 vs q_act 实际，rad）")
    for tag, k in (("首拍", 0), ("中拍", n_tick // 2), ("末拍", n_tick - 1)):   # 三个代表时刻
        print(f"    {pad(tag + f'（k={k}）', 14)}q_ref：{vec_str(q_refs[k], 2)}")
        print(f"    {pad('', 14)}q_act：{vec_str(q_acts[k], 2)}")    # 对齐显示同一拍的实际值

    print("-" * 78)                                         # 小节分隔线
    print("3.4 到位判断（末拍实际 vs 目标 q_goal，全关节进容差才算到位）")  # 到位判断小节标题
    err_final = np.abs(q_acts[-1] - q_goal)                 # 末拍逐关节绝对误差
    arrived = bool(err_final.max() < tol)                   # 到位判定：最大误差 < 容差
    print(f"    逐关节绝对误差 (rad)：{vec_str(err_final, 4)}")  # 误差明细
    print(f"    最大误差 {err_final.max():.5f} rad < 容差 {tol} rad → {'到位' if arrived else '未到位'}")  # 判定结论
    print(f"    附：规划成功 {ok_plan}；{n_tick} 拍 step 全部返回 True：{all(oks)}；"  # 附注一：规划与 step 返回
          f"逐拍指令 == 参考 q：{sent_eq_ref}；下发总数 {len(stub.cmds)}")  # 附注二：指令一致性
    return {"arrived": arrived, "max_err": float(err_final.max()), "tol": tol,  # 结果交 ④ 汇总
            "all_ok": all(oks), "sent_eq_ref": sent_eq_ref, "n_cmds": len(stub.cmds)}


# ============================================================
# ④ 分析结论表
# ============================================================
def print_conclusion(contract: dict, loop: dict, mode_ok: bool) -> None:
    """④ 汇总打印本测试的分析结论表（契约 / MODE / 闭环）。"""
    print("=" * 78)                                         # 大节分隔线
    print("④ 分析结论")
    rows = [                                                # 五行结论：三条契约 + MODE + 闭环
        ("1", "有效帧经 step 下发", "通过" if contract["valid"] else "未通过",  # 结论 1：有效帧
         "返回 True；指令恰 1 条且 == 参考 q"),
        ("2", "缺 q 帧拒绝", "通过" if contract["no_q"] else "未通过",          # 结论 2：缺 q 拒绝
         "抛 ValueError；零下发"),
        ("3", "NaN q 帧拒绝", "通过" if contract["nan_q"] else "未通过",        # 结论 3：NaN 拒绝
         "抛 ValueError；零下发（NaN 兜底）"),
        ("4", "MODE 机制", "通过" if mode_ok else "未通过",                     # 结论 4：MODE 声明
         "基类缺省 MIT；子类覆写为 POSITION"),
        ("5", "闭环到位", "通过" if loop["arrived"] else "未通过",              # 结论 5：闭环到位
         f"最大误差 {loop['max_err']:.4f} rad < 容差 {loop['tol']} rad"),
    ]
    print_table(["序号", "测试项", "结果", "说明"], rows, [6, 20, 8, 36])   # 结论表输出


def main() -> None:
    """离线主流程：① 契约与 MODE 说明 → ② 契约用例 → ③ 闭环仿真 → ④ 结论表。"""
    mode_ok = show_contract_and_mode()                      # ① 类契约与 MODE 机制
    contract = run_contract_tests()                         # ② 三条契约用例（结果供 ④）
    loop = run_closed_loop()                                # ③ 闭环仿真（结果供 ④）
    print_conclusion(contract, loop, mode_ok)               # ④ 分析结论表
    print("=" * 78)                                         # 收尾分隔线
    print("离线测试运行完毕（control 域：契约测试 / MODE 机制 / 闭环仿真 / 结论）")


if __name__ == "__main__":                                  # 直接运行本文件时进入主流程
    main()
