"""
==============================================================================
连接与状态读取  (快速上手 例程 02)
==============================================================================
【功能概要】
    真机例程：连接 JoyArm 机械臂（型号 joyarm_dm），完整走一遍"只读为主"的基础流程：
      1. 连接本体 arm / 末端 end 电机总线，查询连接状态
      2. 使能 / 失能电机，观察使能状态的三态取值（True / False / None）
      3. 两种口径读关节状态：get_state_arm（读时刷新）与 joint_state_arm（被动快照）
      4. 读取控制模式、逐关节状态码、健康检查结果与电机参数（pmax）
    全程不主动运动机械臂；另附三个高危接口的注释示例（默认不执行）。

【安全须知】
    - 清空机械臂工作半径内的人员与障碍物，机械臂稳固放置后再上电
    - 熟悉急停开关位置，运行中保持随手可及，异常立即拍下急停
    - 使能后电机进入锁定状态，切勿强行扳动关节

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 2. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 3. 运行本脚本（在 joyarm_code/ 目录下）
    python quickstart/02_connect_read.py

    预期输出：依次打印连接 / 使能 / 关节状态 / 模式 / 状态码 / 参数读取结果，
    机械臂全程保持不动。
==============================================================================
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装时也可直接运行
from joyarm_core import joyArm  # noqa: E402


def vec_str(v) -> str:
    """数值向量转字符串：None 显「—」，其余保留 3 位小数。"""
    return "—" if v is None else np.round(v, 3).tolist()


def print_state(tag: str, s) -> None:
    """打印一份关节状态快照（q / dq / tau / 双温度 / 状态码）。"""
    if s is None:
        print(f"{tag}：读取失败（返回 None）")
        return
    print(f"{tag}：")
    print(f"    位置 q (rad)     ：{vec_str(s.q)}")
    print(f"    速度 dq (rad/s)  ：{vec_str(s.dq)}")
    print(f"    力矩 tau (N·m)   ：{vec_str(s.tau)}")
    print(f"    驱动板温度 (℃)  ：{vec_str(s.temp_mos)}")
    print(f"    转子温度 (℃)    ：{vec_str(s.temp_rotor)}")
    print(f"    状态码 error     ：{'—' if s.error is None else s.error.tolist()}"
          "（0=失能、1=使能、≥2=故障）")


def main() -> None:
    """真机流程：连接 → 使能 → 状态读取 → 模式/故障检查 → 参数读取 → 失能收尾。"""
    # ---- ① 连接：with 创建机械臂对象（退出自动安全收尾），连接本体与末端总线 ----
    with joyArm("joyarm_dm") as arm:
        ok = arm.connect_arm()                # 连接本体电机总线（成功返回 True）
        print(f"connect_arm 返回：{ok}；is_connected_arm：{arm.is_connected_arm}")
        ok = arm.connect_end()                # 连接末端（夹爪）电机总线
        print(f"connect_end 返回：{ok}；is_connected_end：{arm.is_connected_end}")

        # ---- ② 使能：先读使能前状态，使能后复查 ----
        print(f"使能前 is_enabled_arm：{arm.is_enabled_arm}（None=未知：未读到有效状态、各关节不一致或含故障码）")
        ret = arm.enable_arm()                # 使能本体全部电机：1=全部成功、0=有失败
        print(f"enable_arm 返回：{ret}；使能后 is_enabled_arm：{arm.is_enabled_arm}")
        ret = arm.enable_end()                # 使能末端电机（返回 1/0，含义同上）
        print(f"enable_end 返回：{ret}；使能后 is_enabled_end：{arm.is_enabled_end}")

        # ---- ③ 状态两种口径：get_state_arm 主动发读帧取最新值（要实时数据时用）；
        #      joint_state_arm 只读后台刷新线程的最近快照、不占用总线（高频轮询时用）----
        s = arm.get_state_arm()
        print_state("get_state_arm（读时刷新）", s)
        print_state("joint_state_arm（被动快照）", arm.joint_state_arm)
        print_state("joint_state_end（夹爪被动快照）", arm.joint_state_end)

        # ---- ④ 模式与故障 ----
        print(f"arm 控制模式：{arm.get_mode_arm()}（全组一致才返回模式，否则 None）")
        errs = arm.get_error_arm()            # 各关节状态码缓存（0=失能、1=使能、≥2=故障）
        print(f"arm 状态码缓存 get_error_arm：{errs}")
        print(f"健康检查 check_error_arm：{arm.check_error_arm()}（全 0/1 判正常）")

        # ---- ⑤ 参数读取：pmax = MIT 位域位置量程 (rad)，逐关节返回 ----
        vals = arm.read_param_arm("pmax")     # 可读键还有 vmax / tmax / pos_kp 等
        print(f"read_param_arm('pmax') 返回：{vals}")

        # ---- ⑥ 高危操作（默认注释不执行；理解风险并确认需要时再手动放开）----
        # arm.write_param_arm("pmax", [12.5] * arm.n_arm)  # 写电机参数寄存器（RAM 态、断电丢失；写错可致电机行为异常）
        # arm.set_zero_arm()                                # 把当前位置记为零点（对应教程「零位标定」；执行后零点即被改写）
        # arm.clear_error_arm()                             # 清除各关节硬件故障码（先排查并消除故障原因，再清除）

        # ---- ⑦ 收尾：主动失能两端电机，随后 with 退出自动断连 ----
        print(f"disable_arm 返回：{arm.disable_arm()}；disable_end 返回：{arm.disable_end()}")
    # with 退出（含异常）：__exit__ 调 close()——停状态机与各线程 → 失能 → 断开总线


if __name__ == "__main__":
    main()
