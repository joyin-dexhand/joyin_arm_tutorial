"""
==============================================================================
离线模型与数学  (快速上手 例程 01)
==============================================================================
【功能概要】
    无需连接硬件，纯控制台体验 JoyArm 的离线数字模型与数学能力：
      1. listArm 查可用型号 → joyArm("rebot_arm_b601_dm") 离线创建机械臂对象
      2. 模型信息：关节数 / 关节名 / 软硬限位 / zero-neutral-home 位形 / 末端限位 / 频率
      3. 正运动学、雅可比（含阻尼伪逆）、动力学（惯量 / 科氏 / 重力 / 逆动力学）
      4. 逆运动学：数值法单解（IKResult）与解析法全解（solve_ikine_all）
      5. 求解器六域（fkine/ikine/jacobian/dynamics/trajectory/control）查询与切换
      6. 到位判断接口 is_in_q_arm / is_in_pose_arm 的离线行为
    全程数秒跑完，无 GUI、无硬件，适合零基础新手跑通第一段 JoyArm 代码。

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 2. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（在 joyarm_code/ 目录下）
    python quickstart/01_offline_model.py

    预期输出：①~⑦ 七节控制台信息，末尾打印「离线例程运行完毕」；
    退出收尾时的「arm 未连接……豁免跳过」类日志属正常提示，非报错。
==============================================================================
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装时也可直接运行
from joyarm_core import joyArm, listArm  # noqa: E402


def main() -> None:
    print("=" * 70)
    print("① 创建 JoyArm（离线加载 URDF 数字模型，不连接任何硬件）")
    print(f"listArm() 返回：{listArm()}")        # listArm 自身也会打印一行可用型号
    arm = joyArm("rebot_arm_b601_dm")                    # 离线创建：装配模型与算法，不触硬件
    print(f"print(arm) 展示 __repr__：\n   {arm}")

    print("=" * 70)
    print("② 模型基本信息（只读 property + 关节名查询）")
    print(f"   本体关节数 n_arm = {arm.n_arm}，末端电机数 n_end = {arm.n_end}")
    print(f"   本体关节名（cfg 顺序 = q 向量顺序）：{arm.get_joint_names_arm('all')}")
    hard, soft = arm.joint_limits_hard_arm, arm.joint_limits_soft_arm   # 硬/软限位
    print(f"   硬限位 q_min / q_max：{np.round(hard.q_min, 3)} / {np.round(hard.q_max, 3)}")
    print(f"   软限位 q_min / q_max：{np.round(soft.q_min, 3)} / {np.round(soft.q_max, 3)}")
    print(f"   零位 zero_arm：{np.round(arm.zero_arm, 3)}")
    print(f"   数值法默认初值 neutral_arm：{np.round(arm.neutral_arm, 3)}")
    print(f"   归位位形 home_arm：{np.round(arm.home_arm, 3)}")
    print(f"   末端工作空间包围盒 workspace_box（m，每行为 x/y/z 的 [min, max]）：\n"
          f"{np.round(arm.tcp_limits.workspace_box, 3)}")
    print(f"   规划 / 控制线程频率：plan_hz = {arm.plan_hz}，ctrl_hz = {arm.ctrl_hz}")

    print("=" * 70)
    print("③ 正运动学与雅可比（以 home 位形为例）")
    q = arm.home_arm                             # 取 home 关节角作为示例输入
    pose = arm.solve_fkine(q)                    # 正运动学：q → Pose（位置 + 四元数）
    print(f"   末端位置 (m)：{np.round(pose.position, 4)}")
    print(f"   末端姿态四元数 (w, x, y, z)：{np.round(pose.orientation, 4)}")
    print(f"   雅可比 get_jac 形状：{arm.get_jac(q).shape}")
    print(f"   阻尼伪逆 get_jac_pinv_damped 形状：{arm.get_jac_pinv_damped(q).shape}")

    print("=" * 70)
    print("④ 动力学（速度、加速度均取零向量）")
    dq0 = np.zeros(arm.n_arm)                    # 零速度向量
    print(f"   惯量阵 get_mass_matrix 形状：{arm.get_mass_matrix(q).shape}")
    print(f"   科氏项 get_coriolis 形状：{arm.get_coriolis(q, dq0).shape}")
    print(f"   重力项 get_gravity (N·m)：{np.round(arm.get_gravity(q), 3)}")
    print(f"   逆动力学 solve_idyn (N·m)：{np.round(arm.solve_idyn(q, dq0), 3)}")

    print("=" * 70)
    print("⑤ 逆运动学：先随机采一个可达位姿，再把它解回去")
    q_rand = arm.rand_q_arm()                    # 软限位内均匀随机采样一组关节角
    pose_rand = arm.solve_fkine(q_rand)          # 该随机关节角对应的末端位姿
    res = arm.solve_ikine(pose_rand, q0=arm.home_arm)   # 数值法：自 q0 迭代求单解
    print(f"   数值法 solve_ikine：success={res.success}，err={res.err:.2e}，n_iter={res.n_iter}")
    print(f"   解 q：{np.round(res.q, 3)}")
    arm.set_ikine_solver("analytic_ikine_solver")        # 切换解析法（可求全部解）
    res_all = arm.solve_ikine_all(pose_rand)     # 全部解；数值法激活时本方法抛 RuntimeError
    print(f"   解析法 solve_ikine_all：success={res_all.success}，"
          f"q 形状 {res_all.q.shape}（K 组解 × n_arm）")

    print("=" * 70)
    print("⑥ 求解器六域管理：fkine / ikine / jacobian / dynamics / trajectory / control")
    domains = [                                  # 各域（已加载列表, 当前激活）查询入口
        ("fkine     ", arm.get_fkine_solvers, arm.get_activated_fkine_solver),
        ("ikine     ", arm.get_ikine_solvers, arm.get_activated_ikine_solver),
        ("jacobian  ", arm.get_jac_solvers, arm.get_activated_jac_solver),
        ("dynamics  ", arm.get_dyn_solvers, arm.get_activated_dyn_solver),
        ("trajectory", arm.get_traj_solvers, arm.get_activated_traj_solver),  # 只读不 set：新规划器须 plan 成功才替换激活
        ("control   ", arm.get_ctrl_solvers, arm.get_activated_ctrl_solver),
    ]
    for name, get_all, get_act in domains:       # 六域的已加载与当前激活全部打印
        print(f"   {name}：已加载 {get_all()}，当前激活 {get_act()}")
    arm.set_ikine_solver("pin_ikine_solver")     # set 演示一：ikine 由解析法切回数值法
    arm.set_jac_solver(arm.get_activated_jac_solver())   # set 演示二：jac 域切回原激活（保持不变）
    print(f"   set 后：ikine 激活 = {arm.get_activated_ikine_solver()}，"
          f"jac 激活 = {arm.get_activated_jac_solver()}")

    print("=" * 70)
    print("⑦ 到位判断（读硬件状态帧；离线未连接无快照，返回 False 属预期）")
    print(f"   is_in_q_arm(home_arm) = {arm.is_in_q_arm(arm.home_arm)}")
    print(f"   is_in_pose_arm(fk(home_arm)) = {arm.is_in_pose_arm(arm.solve_fkine(arm.home_arm))}")
    print("   真机在线时：实际关节角 / 位姿进入容差内即返回 True")

    print("=" * 70)
    arm.close()                                  # 显式安全收尾（幂等，可重复调用）
    print("离线例程运行完毕（收尾时「未连接……豁免跳过」类日志为正常提示）")


if __name__ == "__main__":
    main()
