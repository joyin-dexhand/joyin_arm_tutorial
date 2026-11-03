"""
==============================================================================
例程 03 —— 标零与参数设置（需真机）
==============================================================================
【功能概要】
    1. 连接电机总线 → get/check_error_arm/end 检查错误码 → 使能 → clear_error_arm/end；
    2. 零位标定：set_zero_arm/end, 打印说明与当前位置，输入 yes 确认（默认跳过、零点不动）；
    3. 参数写入：read_param_arm/end 读 pos_kp 原值 → 输入 yes 确认 → write_param_* 写入新值 → 读回核对。

【安全须知】
    - 零位标定前须把机械臂摆到零位构型，并清空工作半径内的人员与障碍物
    - 机械臂行为异常则立刻急停或断电；Ctrl+C 随时中断，with 退出自动安全收尾

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux
    irm https://astral.sh/uv/install.ps1 | iex          # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate

    # 2. 运行本脚本（需连接对应的机械臂）
    python quick_start/03_set_zero_and_params.py <型号>  # 如：rebot_arm_b601_dm

【预期输出】
    连接 → 错误码检查 → 使能 → 清错 → 标零 → 参数写入与读回核对 → 失能退出。
==============================================================================
"""

from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from joyarm_core import joyArm, listArm

ZERO_TOL = 0.05  # 标零复核容差（rad）


# ============================================================
# 一、命令行解析：必填型号参数，缺参打印用法、列出可用型号并以退出码 1 结束
# ============================================================
def parse_model() -> str:
    """解析必填的命令行型号参数；缺失时打印用法并列出可用型号（listArm），以退出码 1 结束。"""
    if len(sys.argv) != 2:                       # 必须恰好携带 1 个型号参数
        print(f"用法：python quick_start/03_set_zero_and_params.py <型号>")  # 缺参：先打印用法
        listArm()                                # 顺带列出全部可用型号
        sys.exit(1)                              # 退出码 1，不进入演示
    return sys.argv[1].strip()                   # 型号名


# ============================================================
# 二、打印格式化助手：显示宽度计算与表格列对齐补空格
# ============================================================
def disp_width(s: str) -> int:
    """计算字符串的终端显示宽度（中文字符按 2 列计），用于表格列对齐。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)


def pad(s: str, width: int, align: str = "left") -> str:
    """按显示宽度补空格对齐：align="right" 右对齐（数值列），"left" 左对齐（名称列）。"""
    gap = " " * max(width - disp_width(s), 0)    # 缺口 = 目标宽 − 显示宽（不为负）
    return gap + s if align == "right" else s + gap


# ============================================================
# 三、主流程：连接、错误码检查与使能 → 零位标定 → 参数写入 → 失能收尾
# ============================================================
def main() -> None:
    """真机流程：连接、错误码检查与使能 → 零位标定（默认跳过）→ 参数写入（默认跳过）→ 失能收尾。"""
    model = parse_model()                                 # 必填型号参数

    with joyArm(model) as arm:                            # with 退出自动安全收尾：停线程→失能→断连

        print("=============== ① 连接、错误码检查与使能 ================================= ")
        ok = arm.connect_arm()                            # 连接本体电机总线
        print(f"connect_arm 返回：{ok}；is_connected_arm：{arm.is_connected_arm}")
        ok = arm.connect_end()                            # 连接末端（夹爪）电机总线
        print(f"connect_end 返回：{ok}；is_connected_end：{arm.is_connected_end}")
        print(f"arm get_error_arm：{arm.get_error_arm()}（0=失能、1=使能、≥2=故障）；check_error_arm：{arm.check_error_arm()}")
        print(f"end get_error_end：{arm.get_error_end()}；check_error_end：{arm.check_error_end()}（全 0/1 判正常 True）")
        ret = arm.enable_arm()                            # 使能本体全部关节电机：1=全部成功、0=有失败
        print(f"enable_arm 返回：{ret}；使能后 is_enabled_arm：{arm.is_enabled_arm}")
        ret = arm.enable_end()                            # 使能末端电机（返回 1/0，含义同上）
        print(f"enable_end 返回：{ret}；使能后 is_enabled_end：{arm.is_enabled_end}")
        time.sleep(0.01)                                  # 等待电机使能并状态更新完毕
        yes = input("【确认】输入 yes 执行 clear_error_*，其他输入跳过：").strip() == "yes"
        if yes:                                       # 确认 yes 才执行；clear_error 返回 True=全部清除成功
            print(f"clear_error_arm 返回：{arm.clear_error_arm()}；clear_error_end 返回：{arm.clear_error_end()}")
            time.sleep(0.01)                          # 等待状态更新完毕
            print(f"清错后状态码 arm：{arm.get_error_arm()}；end：{arm.get_error_end()}（清错后电机自动失能属正常）")
        else:
            print("未确认 yes：已跳过清错（有错误码的电机不会响应后续指令）。")
        
        print("=============== ② 标零：默认只打印说明，不执行 ============================== ")
        print("电机标零：把电机「当前时刻的位置」记为零点，永久保存该零点，不可逆。执行前需确认机械臂已摆到零位构型")

        print("—— 本体 arm：set_zero_arm ——")
        s = arm.get_state_arm()                           # 主动读当前关节状态（读取失败为 None）
        print(f"当前位置 q（rad）：{'—' if s is None else np.round(s.q, 3).tolist()}")
        yes = s is not None and input("【确认】当前位置将被记为零点，输入 yes 执行，其他输入跳过：").strip() == "yes"
        if yes:                                           # 确认 yes 才执行
            ret = arm.set_zero_arm()                      # 执行标零：1=全部成功、0=任一失败
            time.sleep(0.1)                               # 等待标零后的新状态数据
            s = arm.get_state_arm()                       # 复核标零后的结果：当前位置应接近全零
            q_max = float("nan") if s is None else float(np.max(np.abs(s.q)))
            print(f"set_zero_arm 返回：{ret}；复核 |q|max = {q_max:.4f} rad（容差 {ZERO_TOL}）→ "
                  f"{'标零生效' if q_max <= ZERO_TOL else '未接近零点（读取失败或状态帧尚未刷新，可重跑）'}")
        else:
            print("未确认 yes（或当前位置读取失败）：已跳过 arm 标零。")

        print("—— 末端 end：set_zero_end ——")
        s = arm.get_state_end()                           # end 电机：流程与 arm 相同
        print(f"当前位置 q（rad）：{'—' if s is None else np.round(s.q, 3).tolist()}")
        yes = s is not None and input("【确认】当前位置将被记为零点，输入 yes 执行，其他输入跳过：").strip() == "yes"
        if yes:
            ret = arm.set_zero_end()                      # 执行标零：1=全部成功、0=任一失败
            time.sleep(0.1)
            s = arm.get_state_end()
            q_max = float("nan") if s is None else float(np.max(np.abs(s.q)))
            print(f"set_zero_end 返回：{ret}；复核 |q|max = {q_max:.4f} rad（容差 {ZERO_TOL}）→ "
                  f"{'标零生效' if q_max <= ZERO_TOL else '未接近零点（读取失败或状态帧尚未刷新，可重跑）'}")
        else:
            print("未确认 yes（或当前位置读取失败）：已跳过 end 标零。")

        print("=============== ③ 参数写入安全演示 ==================================== ")
        print("（以位置环增益 pos_kp 为例；可读写键名依型号由 backend_* 定义；设模式时 backend 自动从 cfg 设置相关控制增益参数")

        print("—— arm：read 原值 → write_param_arm 写新值（原值 + 10）→ read 读回核对 ——")
        before = arm.read_param_arm("pos_kp")             # 读原值：逐关节当前增益列表
        print(f"当前 pos_kp：{'—' if before is None else np.round(before, 3).tolist()}")
        yes = before is not None and input("【确认】将 pos_kp 写为原值 + 10（切模式或断电后恢复），输入 yes 执行，其他输入跳过：").strip() == "yes"
        if yes:                                           # 确认 yes 才执行
            target = [v + 10.0 for v in before]           # 新值 = 原值 + 10
            ret = arm.write_param_arm("pos_kp", target)   # 写入新值：1=成功、0=失败
            after = arm.read_param_arm("pos_kp")          # 读新值：应与写入值一致
            print(f"write_param_arm('pos_kp', 原值 + 10) 返回：{ret}")
            print(pad("关节", 12) + pad("原值", 14, "right") + pad("写入值", 14, "right") + pad("读回值", 14, "right"))
            for i, name in enumerate(arm.get_joint_names_arm()):
                ok = abs(after[i] - target[i]) <= 1e-3    # 生效判据：读回 ≈ 写入
                print(pad(str(name), 12) + pad(f"{before[i]:.3f}", 14, "right") + pad(f"{target[i]:.3f}", 14, "right")
                      + pad(f"{after[i]:.3f}", 14, "right") + ("  已生效" if ok else "  未生效!"))
        else:
            print("未确认 yes（或读取失败）：已跳过 arm 写入。")

        print("—— 末端 end：read 原值 → write_param_end 写新值（原值 + 10）→ read 读回核对 ——")
        before = arm.read_param_end("pos_kp")             # 读原值：逐关节当前增益列表
        print(f"当前 pos_kp：{'—' if before is None else np.round(before, 3).tolist()}")
        yes = before is not None and input("【确认】将 pos_kp 写为原值 + 10（切模式或断电后恢复），输入 yes 执行，其他输入跳过：").strip() == "yes"
        if yes:                                           # 确认 yes 才执行
            target = [v + 10.0 for v in before]           # 新值 = 原值 + 10
            ret = arm.write_param_end("pos_kp", target)
            after = arm.read_param_end("pos_kp")
            print(f"write_param_end('pos_kp', 原值 + 10) 返回：{ret}")
            print(pad("关节", 12) + pad("原值", 14, "right") + pad("写入值", 14, "right") + pad("读回值", 14, "right"))
            for i, name in enumerate(arm.get_joint_names_end()):
                ok = abs(after[i] - target[i]) <= 1e-3    # 生效判据：读回 ≈ 写入
                print(pad(str(name), 12) + pad(f"{before[i]:.3f}", 14, "right") + pad(f"{target[i]:.3f}", 14, "right")
                      + pad(f"{after[i]:.3f}", 14, "right") + ("  已生效" if ok else "  未生效!"))
        else:
            print("未确认 yes（或读取失败）：已跳过 end 写入。")

        print("=============== ④ 收尾 ========================================== ")
        print(f"disable_arm 返回：{arm.disable_arm()}；disable_end 返回：{arm.disable_end()}")  # 主动失能两端电机
    print("结束收尾：机械臂已失能，with 退出自动断开总线。")


if __name__ == "__main__":
    try:
        main()                 # 主流程
    except KeyboardInterrupt:  # Ctrl+C
        print("\n检测到 Ctrl+C：已安全退出。")
