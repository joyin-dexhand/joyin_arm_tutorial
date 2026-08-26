"""BackendDM 真机状态监视：每 0.5 s 刷新全部关节电机状态（电机保持失能）。

【功能概要】
按 ``joyarm_dm.yaml`` 连接 backend_dm —— 只打开串口并启动接收线程，全程不调用
``enable_*``，电机保持失能。每 0.5 s 读取一次 6 个本体关节 + 夹爪电机的全部
状态量（q / dq / tau / 使能 / 故障 / 通讯）并在终端原地刷新；手动搬动机械臂
即可观察反馈数据变化。注：DM 状态帧不含 ddq / 电压 / 电流（ArmState 中恒为
0），故表格不显示。

【环境与运行】
1. 新建环境并安装依赖：``cd joyarm_code && uv venv && uv sync``
2. 激活环境：``source .venv/bin/activate``
3. 接好达妙 CAN 桥（默认 ``/dev/ttyACM0``）与电机电源后运行：
   - ``python test/test_backend_dm_monitor.py``     # 无限刷新，Ctrl+C 退出
   - ``python test/test_backend_dm_monitor.py 20``  # 刷新 20 个周期后自动退出
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core.backends.backend_dm import BackendDM  # noqa: E402

REFRESH_PERIOD = 0.5  # 刷新周期（秒）


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


def _print_frame(out, arm_names: list[str], end_names: list[str], s, e: dict) -> None:
    """打印一帧状态表；``out`` 为输出函数（终端模式带行尾清理）。"""
    out(f"{'关节':<8}  {'q(rad)':>9} {'dq(rad/s)':>10} {'tau(N·m)':>9}    使能  故障  通讯")
    out("-" * 64)
    rows = list(zip(arm_names, s.joint.q, s.joint.dq, s.joint.tau,
                    s.joint.enabled, s.joint.error, s.joint.comm_ok))
    rows += list(zip(end_names, e["q"], e["dq"], e["tau"],
                     e["enabled"], e["error"], e["comm_ok"]))
    for name, q, dq, tau, en, flt, ok in rows:
        out(f"{name:<10}{q:+9.4f}{dq:+10.4f}{tau:+9.4f}"
            f"     {'是' if en else '否'}    {'是' if flt else '否'}    {'是' if ok else '否'}")
    out("-" * 64)
    stamp = time.strftime("%H:%M:%S", time.localtime(s.timestamp)) + f".{int(s.timestamp * 1000) % 1000:03d}"
    out(f"时间 {stamp} | 电机保持失能 | 周期 {REFRESH_PERIOD} s | Ctrl+C 退出")
    out(f"故障：{'；'.join(s.errors) if s.errors else '无'}")


def main() -> None:
    cycles: int | None = None  # 刷新周期数（缺省无限，Ctrl+C 退出）
    if len(sys.argv) > 1:
        try:
            cycles = max(1, int(sys.argv[1]))
        except ValueError:
            sys.exit(f"用法：python test/test_backend_dm_monitor.py [刷新周期数]（缺省无限，Ctrl+C 退出）")

    backend, arm_names, end_names, channel = _load_backend()
    try:
        backend.connect()
    except ImportError:
        sys.exit("缺少 pyserial：请先 `uv sync`（或 pip install pyserial）")
    except Exception as exc:
        sys.exit(f"连接失败（检查 {channel} 是否存在 / 权限 / 被占用）：{exc}")

    tty = sys.stdout.isatty()

    def out(line: str) -> None:
        # 终端模式行尾清至行末，避免上一帧残留；输出重定向时退化为逐帧打印
        print(line + "\x1b[K" if tty else line, flush=True)

    if tty:
        sys.stdout.write("\x1b[2J")  # 首帧清屏，此后光标归位原地重绘
    else:
        print(f"已连接 {channel}（电机保持失能，可手动搬动机械臂观察反馈）", flush=True)
    try:
        n = 0
        while cycles is None or n < cycles:
            t0 = time.monotonic()
            s = backend.read_state_arm()
            e = backend.read_state_end()
            if tty:
                sys.stdout.write("\x1b[H")
            _print_frame(out, arm_names, end_names, s, e)
            n += 1
            time.sleep(max(0.0, REFRESH_PERIOD - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，退出监视。")
    except Exception as exc:
        print(f"\n读取异常（串口可能已断开）：{exc}")
    finally:
        backend.disconnect()
        print("已断开连接，电机保持失能。", flush=True)


if __name__ == "__main__":
    main()
