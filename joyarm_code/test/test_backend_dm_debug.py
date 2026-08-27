"""BackendDM 分步调试脚本：逐电机、逐功能、按风险从低到高交互测试（需真机）。

【功能概要】
交互式菜单驱动 backend_dm 单项功能测试，菜单序即推荐执行序（危险度递增）：
连接 → 读状态 → 读参数 → 写参数 → 设零位 → 安全使能 → 失能 → 模式切换 → 发送指令。
每个动作前可选择作用目标：全部本体关节 / 单个关节（输入 1-6 对应 joint1~joint6，
内部转 0 基代码索引）/ 末端夹爪。

安全设计：
- 安全使能：先切 MIT 模式，使能后立刻下发 kp=kd=tau=0 的零阻抗 MIT 指令，
  电机零力矩"松软"，防止使能瞬间机械臂突然运动；此时可手动搬动验证。
- 位置指令默认目标 = 当前 q（发送即保持不动），修改数值才产生运动。
- 末端预设：open=-1 / close=3 / zero=0（电机弧度，7 号电机 gripper）。
- 写参数 / 设零位需二次确认；退出前自动失能并断开。

【安全提示】
上电使能前：清空机械臂工作空间、确认急停可用、扶稳大臂防坠落；
本脚本会真实驱动电机，请务必在有人监护下使用。

【环境与运行】
1. 新建环境并安装依赖：``cd joyarm_code && uv venv && uv sync && source .venv/bin/activate``
2. 接好达妙 CAN 桥（默认 ``/dev/ttyACM0``）与电机电源后运行：
   ``python test/test_backend_dm_debug.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core.backends.backend_dm import BackendDM  # noqa: E402
from joyarm_core.utils.types import ControlMode  # noqa: E402

_PARAM_KEYS = ["ctrl_mode", "vel_kp", "vel_ki", "pos_kp", "pos_ki"]  # DM 可读参数名

# 末端预设位（电机弧度，作用于 7 号电机 gripper）：open 张开 / close 闭合 / zero 归零，
# 位于 config 行程 q_min/q_max 内，经 send_position_end 下发
_END_PRESETS = {"o": ("open", -1.0), "c": ("close", 3.0), "z": ("zero", 0.0)}

_MENU = """--------------------------------------------------
  [1] 连接            [2] 读状态         [3] 读参数
  [4] 写参数(慎)      [5] 设零位(慎)     [6] 安全使能(MIT零阻抗)
  [7] 失能            [8] 切换控制模式   [9] 发送控制指令
  [0] 断开并退出
--------------------------------------------------（菜单序即推荐执行序）"""


def _load_backend() -> tuple[BackendDM, list[str], list[str], str]:
    """读 joyarm_dm.yaml 构造 BackendDM（不连接），返回 (后端, 关节名, 末端名, 通道)。"""
    cfg = yaml.safe_load((_ROOT / "joyarm_core" / "configs" / "joyarm_dm.yaml").read_text(encoding="utf-8"))
    bcfg = dict(cfg["backend"])
    bcfg.pop("name")
    arm = bcfg.get("arm") or {}
    end = bcfg.get("end") or {}
    arm_names = [j["name"] for j in arm.get("joints", [])]
    end_names = [j["name"] for j in end.get("joints", [])]
    return BackendDM(bcfg), arm_names, end_names, arm.get("channel", "?")


def _ask(prompt: str) -> str:
    return input(f"{prompt}: ").strip()


def _ask_float(label: str, default: float) -> float:
    ans = input(f"{label}（回车={default:g}）: ").strip()
    return default if ans == "" else float(ans)


def _pick(arm_names: list[str], end_names: list[str]):
    """选择作用目标 → (group, joint, n)；取消返回 None。

    a=全部本体关节（joint=None）；1-6=joint1~joint6（1 基输入，内部转
    0 基代码索引）；g=末端全部电机。
    """
    hint = f"a=全部{len(arm_names)}关节 | 1-{len(arm_names)}=joint1~joint{len(arm_names)}"
    if end_names:
        hint += " | g=末端"
    ans = input(f"目标 [{hint} | 回车=取消]: ").strip().lower()
    if ans == "a":
        return "arm", None, len(arm_names)
    if ans.isdigit() and 1 <= int(ans) <= len(arm_names):
        return "arm", int(ans) - 1, 1  # joint1~joint6 → 代码索引 0~5
    if ans == "g" and end_names:
        return "end", None, len(end_names)
    if ans == "g":
        print("✗ 本型号未配置末端")
    return None


def _target_names(group: str, joint, arm_names, end_names) -> list[str]:
    if group == "arm":
        return arm_names if joint is None else [arm_names[joint]]
    return list(end_names)


def _selected_mode(backend: BackendDM, group: str, joint):
    """经公开接口查询所选电机当前控制模式（未设置或混合返回 None）。"""
    return backend.read_mode_arm(joint) if group == "arm" else backend.read_mode_end(joint)


def _print_state(backend: BackendDM, group: str, joint, arm_names, end_names) -> None:
    names = _target_names(group, joint, arm_names, end_names)
    if group == "arm":
        s = backend.read_state_arm(joint)
        js = s.joint
        rows = zip(names, js.q, js.dq, js.tau, js.temp_mos, js.temp_rotor,
                   js.enabled, js.error, js.comm_ok)
        errs = s.errors
    else:
        e = backend.read_state_end(joint)
        rows = zip(names, e["q"], e["dq"], e["tau"], e["temp_mos"], e["temp_rotor"],
                   e["enabled"], e["error"], e["comm_ok"])
        errs = [f"{nm}: 通讯无应答" for nm, ok in zip(names, e["comm_ok"]) if not ok]
    print(f"{'关节':<10}{'q(rad)':>10}{'dq(rad/s)':>11}{'tau(N·m)':>10}{'tMOS':>5}{'tRot':>5}  使能 故障 通讯")
    for nm, q, dq, tau, tm, tr, en, fl, ok in rows:
        t2 = lambda t: "—" if t == 0 else f"{t:.0f}"
        print(f"{nm:<10}{q:+10.4f}{dq:+11.4f}{tau:+10.4f}{t2(tm):>5}{t2(tr):>5}"
              f"   {'是' if en else '否'}   {'是' if fl else '否'}   {'是' if ok else '否'}")
    print(f"故障：{'；'.join(errs) if errs else '无'}")


def _act_read_param(backend: BackendDM, group: str, joint, names: list[str]) -> None:
    print(f"可读参数：{_PARAM_KEYS}")
    key = _ask("参数名")
    if key not in _PARAM_KEYS:
        print("✗ 未知参数名")
        return
    val = backend.read_param_arm(key, joint) if group == "arm" else backend.read_param_end(key, joint)
    if joint is None:
        for nm, v in zip(names, val):
            print(f"  {nm:<10}{key} = {v:g}")
    else:
        print(f"  {names[0]} {key} = {val:g}")


def _act_write_param(backend: BackendDM, group: str, joint, names: list[str]) -> None:
    print(f"可写参数：{_PARAM_KEYS}")
    key = _ask("参数名")
    if key not in _PARAM_KEYS:
        print("✗ 未知参数名")
        return
    value = _ask_float("新值", 0.0)
    persist = _ask("持久化到闪存？(y/N)").lower() == "y"
    if persist:
        print("⚠ 存闪存将自动失能电机并保持失能（DM 硬约束）")
    if group == "arm":
        backend.write_param_arm(key, value, joint, persist=persist)
    else:
        backend.write_param_end(key, value, joint, persist=persist)
    print(f"✓ 已写入 {key}={value:g}" + ("（已存闪存）" if persist else "（仅 RAM，掉电丢失）"))


def _act_set_zero(backend: BackendDM, group: str, joint, names: list[str]) -> None:
    print(f"⚠ 将把 {names} 当前位置设为机械零位（改写电机零位，不可撤销）")
    print("  请先将机械臂摆到期望零位姿态并扶稳（流程会先失能）")
    if _ask("确认输入 yes") != "yes":
        print("已取消")
        return
    if group == "arm":
        backend.set_zero_arm(joint)
    else:
        backend.set_zero_end(joint)
    print("✓ 标零完成")


def _act_safe_enable(backend: BackendDM, group: str, joint, n: int) -> None:
    if group == "arm":
        print("步骤 1/3：切换 MIT 模式（电机仍失能）...")
        backend.set_mode_arm(ControlMode.MIT, joint)
        print("步骤 2/3：使能...")
        backend.enable_arm(joint)
        print("步骤 3/3：下发零阻抗 MIT 指令（kp=kd=tau=0，零力矩）...")
        z = np.zeros(n)
        backend.send_mit_arm(z, z, z, z, z, joint)
        print("✓ 已使能：零阻抗零力矩，电机松软，可手动搬动验证")
    else:
        backend.set_mode_end(ControlMode.MIT, joint)
        backend.enable_end(joint)
        print("✓ 末端已使能（MIT 模式；未下发指令前无力矩输出，用 [9] 下发位置/力度）")


def _act_set_mode(backend: BackendDM, group: str, joint) -> None:
    print("若电机当前处于使能态，建议先 [7] 失能再切换")
    ans = _ask("模式 [1=MIT | 2=POSITION(默认) | 3=VELOCITY]") or "2"
    mode = {"1": ControlMode.MIT, "2": ControlMode.POSITION, "3": ControlMode.VELOCITY}.get(ans)
    if mode is None:
        print("✗ 未知模式")
        return
    if group == "arm":
        backend.set_mode_arm(mode, joint)
    else:
        backend.set_mode_end(mode, joint)
    print(f"✓ 已切换到 {mode.value}")


def _act_send(backend: BackendDM, group: str, joint, n: int,
              arm_names: list[str], end_names: list[str]) -> None:
    names = _target_names(group, joint, arm_names, end_names)
    mode = _selected_mode(backend, group, joint)
    if mode is None:
        print("✗ 所选电机模式未设置或不一致，请先 [8] 切换控制模式")
        return
    print(f"当前模式：{mode.value}")
    if group == "arm":
        if mode == ControlMode.MIT:
            print("提示：kp/kd=0 时 q/dq 不起作用（纯前馈力矩）")
            q = _ask_float("q 目标(rad)", 0.0)
            dq = _ask_float("dq 目标(rad/s)", 0.0)
            tau = _ask_float("tau 前馈(N·m)", 0.0)
            kp = _ask_float("kp", 0.0)
            kd = _ask_float("kd", 0.0)
            backend.send_mit_arm(np.full(n, q), np.full(n, dq), np.full(n, tau),
                                 np.full(n, kp), np.full(n, kd), joint)
            print("✓ MIT 指令已下发")
        elif mode == ControlMode.POSITION:
            cur = backend.read_state_arm(joint).joint.q
            ans = input(f"目标 q（回车=保持当前 {np.array2string(cur, precision=3)} 不动；"
                        f"单值广播到所选电机）: ").strip()
            target = cur if ans == "" else np.full(n, float(ans))
            backend.send_position_arm(target, joint)
            print("✓ 位置指令已下发（按 config POS_VEL.vlim 限速）")
        else:
            dq = _ask_float("dq 目标(rad/s)", 0.0)
            backend.send_velocity_arm(np.full(n, dq), joint)
            print("✓ 速度指令已下发")
        return
    # 末端（位置语义：电机弧度，预设 open=-1 / close=3 / zero=0，7 号电机 gripper）
    ans = _ask("末端指令 [p=位置 | f=力度 | o=open(-1) | c=close(3) | z=zero(0)]").lower()
    if ans == "p":
        cur = backend.read_state_end(joint)["q"]
        pos = _ask_float("位置(电机弧度)", cur[0] if len(cur) == 1 else 0.0)
        backend.send_position_end(pos, joint)
        print("✓ 末端位置指令已下发（自动裁剪到 q_min/q_max）")
    elif ans == "f":
        print("⚠ 力度经 MIT 近似：以 config MIT 增益闭合至 q_max，前馈 = 力度×force_to_tau")
        force = _ask_float("力度(N)", 0.0)
        backend.send_force_end(force, joint)
        print("✓ 末端力度指令已下发")
    elif ans in _END_PRESETS:
        label, target = _END_PRESETS[ans]
        backend.send_position_end(target, joint)
        print(f"✓ 末端{label}：gripper → {target:g} rad（限幅内，按 vlim 限速）")
    else:
        print("✗ 未知指令")


def main() -> None:
    backend, arm_names, end_names, channel = _load_backend()
    print(f"BackendDM 分步调试 | 通道 {channel} | 本体 {len(arm_names)} 关节"
          f" + 末端 {len(end_names)} 电机")
    print("⚠ 本脚本会真实驱动电机：清空工作空间、确认急停可用、有人监护")
    try:
        while True:
            conn = "已连接" if backend.connected else "未连接"
            pairs = [(nm, backend.read_mode_arm(i)) for i, nm in enumerate(arm_names)]
            pairs += [(nm, backend.read_mode_end(i)) for i, nm in enumerate(end_names)]
            modes = "，".join(f"{nm}={m.value}" for nm, m in pairs if m is not None) or "未设置"
            print(f"\n[{conn}] 模式缓存：{modes}")
            print(_MENU)
            choice = _ask("选择功能")
            try:
                if choice == "0":
                    break
                if choice == "1":
                    backend.connect()
                    print(f"✓ 已连接 {channel}（电机保持失能）")
                elif choice in "23456789" and choice != "":
                    picked = _pick(arm_names, end_names)
                    if picked is None:
                        continue
                    group, joint, n = picked
                    names = _target_names(group, joint, arm_names, end_names)
                    if choice == "2":
                        _print_state(backend, group, joint, arm_names, end_names)
                    elif choice == "3":
                        _act_read_param(backend, group, joint, names)
                    elif choice == "4":
                        _act_write_param(backend, group, joint, names)
                    elif choice == "5":
                        _act_set_zero(backend, group, joint, names)
                    elif choice == "6":
                        _act_safe_enable(backend, group, joint, n)
                    elif choice == "7":
                        (backend.disable_arm if group == "arm" else backend.disable_end)(joint)
                        print(f"✓ {names} 已失能")
                    elif choice == "8":
                        _act_set_mode(backend, group, joint)
                    elif choice == "9":
                        _act_send(backend, group, joint, n, arm_names, end_names)
            except (RuntimeError, ValueError, TimeoutError) as exc:
                print(f"✗ {exc}")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        backend.disconnect()  # 先广播失能再关串口（幂等，未连接时安全）
        print("已退出：电机失能、连接断开。")


if __name__ == "__main__":
    main()
