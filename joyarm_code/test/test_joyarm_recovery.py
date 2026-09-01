"""JoyArm 真机复测脚本（事故恢复版）：仅覆盖当前可用功能，全程保护。

【定位】
原复测对象中的 MIT 软件伺服回放 / play_cart / track_cart 依赖 robotics 执行器
与求解器（现为教程 Ch2~6 教学内容，实现注册后随章补回）；本脚本仅复测
backend 级功能：构造连接与基准位形、阻尼使能+位置保持、速度模式短时、
本体标零与断开（与 test_joyarm_full.py 对应步骤同构）。
⚠ 前提：连杆已修复。复测顺序即风险递增序。

【保护功能（全程自动生效）】
- 速度模式为单拍指令（无软件闭环），本脚本以轮询守护覆盖：
  运行期间每 20ms 检查通讯与 |dq|，异常 → damping_mode 全电机紧急阻尼。

【安全设计】（与 test_joyarm_full.py 一致）
config 安全化：margin=0、限速 0.5 rad/s、MIT kp≤20/kd≤2、末端 tau≤0.5。
运动幅度 0.2 rad/s×0.6s。高风险项须输入 yes；try/finally：失能 → 断开 → 报告。

【环境与运行】
1. ``cd joyarm_code && uv venv && uv sync && source .venv/bin/activate``
2. 修复连杆、确认急停可用、清空工作空间后：
   ``python test/test_joyarm_recovery.py``        # 交互逐项执行（4 步）
   ``python test/test_joyarm_recovery.py --list`` # 仅列出，不执行
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import JoyArm  # noqa: E402
from joyarm_core.utils.types import ControlMode  # noqa: E402

# ---- 安全上限（与 test_joyarm_full.py 相同）----
SOFT_MARGIN = 0.0
SAFE_ARM_VLIM = 0.5
SAFE_ARM_MIT_KP = 20.0
SAFE_ARM_MIT_KD = 2.0
SAFE_END_VLIM = 1.0
SAFE_END_DQ = 1.0
SAFE_END_TAU = 0.5
# ---- 动作幅度 / 守护参数 ----
VEL_TEST = 0.2
VEL_DUR = 0.6
VEL_GUARD_DQ = 1.0       # 速度模式脚本级守护阈值 rad/s
DAMP_KD = 1.5

_BANNER = """
╔══════════════════════════════════════════════════════════╗
║   JoyArm 真机复测（事故恢复版，4 步，仅 backend 级功能）     ║
╚══════════════════════════════════════════════════════════╝
⚠ 前提：连杆已修复；清空工作空间、急停可用、全程有人监护。
速度模式脚本守护：通讯异常或 |dq|>1.0 rad/s → 全电机紧急阻尼。
已收紧：限速 0.5 rad/s、MIT kp≤20/kd≤2、margin=0；高风险项须输入 yes。
"""


def _fmt(a, prec: int = 4) -> str:
    return np.array2string(np.asarray(a, dtype=float), precision=prec)


class JoyArmRecoveryTest:
    """复测执行器：4 个步骤风险递增（构造连接 → 使能保持 → 速度 → 标零断开）。"""

    def __init__(self) -> None:
        cfg = yaml.safe_load(
            (_ROOT / "joyarm_core" / "configs" / "joyarm_dm.yaml").read_text(encoding="utf-8")
        )
        self._safeguard(cfg)
        self.arm = JoyArm(model="joyarm_dm", config=cfg)
        self.arm_names = [j["name"] for j in cfg["backend"]["arm"]["joints"]]
        self.q_base: np.ndarray | None = None
        self.results: list[tuple[str, str, str]] = []

    # ----------------------------------------------------------
    # config 安全化（与 full 相同策略）
    # ----------------------------------------------------------
    @staticmethod
    def _safeguard(cfg: dict) -> None:
        b = cfg.setdefault("basic", {}).setdefault("utils", {})
        b["joint_limits_soft_margin"] = SOFT_MARGIN
        for j in cfg["backend"]["arm"]["joints"]:
            j.setdefault("POS_VEL", {})["vlim"] = min(float(j["POS_VEL"].get("vlim", 0.0)), SAFE_ARM_VLIM)
            mit = j.setdefault("MIT", {})
            mit["kp"] = min(float(mit.get("kp", 0.0)), SAFE_ARM_MIT_KP)
            mit["kd"] = min(float(mit.get("kd", 0.0)), SAFE_ARM_MIT_KD)
        for j in cfg["backend"]["end"]["joints"]:
            j.setdefault("POS_VEL", {})["vlim"] = min(float(j["POS_VEL"].get("vlim", 0.0)), SAFE_END_VLIM)
            j["dq_max"] = min(float(j.get("dq_max", SAFE_END_DQ)), SAFE_END_DQ)
            j["tau_max"] = min(float(j.get("tau_max", SAFE_END_TAU)), SAFE_END_TAU)

    # ----------------------------------------------------------
    # 交互与辅助
    # ----------------------------------------------------------
    def _confirm(self, idx: int, total: int, title: str, risk: str, desc: str) -> str:
        print("\n" + "─" * 56)
        print(f"[步骤 {idx}/{total}] 风险:{risk} | {title}")
        print(f"说明：{desc}")
        if risk == "高":
            ans = input("⚠ 高风险：输入 yes 执行（回车/s=跳过 q=退出）: ").strip().lower()
        else:
            ans = input("[回车]=执行  s=跳过  q=退出: ").strip().lower()
        if ans == "q":
            return "quit"
        if ans == "s" or (risk == "高" and ans != "yes"):
            return "skip"
        return "run"

    def _q_now(self) -> np.ndarray:
        return np.asarray(self.arm.get_arm_state().joint.q, dtype=float)

    def _guarded_velocity(self, dq_cmd: np.ndarray, duration: float) -> None:
        """速度模式脚本级守护：运行期间轮询 |dq|，超阈立即全阻尼。"""
        t0 = time.time()
        while time.time() - t0 < duration:
            st = self.arm.get_arm_state()
            if not bool(np.all(st.joint.comm_ok)):
                self.arm.damping_mode()
                raise AssertionError("通讯异常 → 紧急阻尼")
            if np.abs(np.asarray(st.joint.dq)).max() > VEL_GUARD_DQ:
                self.arm.damping_mode()
                raise AssertionError(f"速度模式过速 |dq|>{VEL_GUARD_DQ} → 紧急阻尼")
            time.sleep(0.02)

    # ----------------------------------------------------------
    # 步骤（4 步，风险递增）
    # ----------------------------------------------------------
    def _steps(self) -> list[tuple[str, str, str, object, object]]:
        """(标题, 风险, 一句话说明, 执行函数, 预案函数)。预案函数返回本步的
        测试预案卡（参数/模式/预期/保护/急停），执行前打印供人工审阅。"""
        return [
            ("R0-构造连接与基准位形", "低",
             "安全化构造（margin=0/限速/增益）→ connect → q_base=clamp_q(当前 q)",
             self.step_setup, self.brief_setup),
            ("R1-阻尼使能+位置保持", "中",
             "MIT 阻尼安全使能（kp=0/kd=1.5）→ 切 POSITION 保持 q_base；扶稳大臂",
             self.step_enable, self.brief_enable),
            ("R2-速度模式短时", "高",
             f"VELOCITY 零速使能 → joint1 {VEL_TEST} rad/s×{VEL_DUR}s → 归零；"
             f"脚本守护：|dq|>{VEL_GUARD_DQ} → damping_mode",
             self.step_velocity, self.brief_velocity),
            ("R3-本体标零与断开", "高",
             "选 |q| 最小关节：标零→回原零位→恢复标零→回原位置→验证；失能断开",
             self.step_set_zero_teardown, self.brief_set_zero),
        ]

    # ----------------------------------------------------------
    # 测试预案卡（执行前人工审阅：参数 / 模式 / 预期 / 保护 / 急停）
    # ----------------------------------------------------------
    _ESTOP = ("[急停] 终端按 q+回车（触发清理：失能→断开）| Ctrl+C | 物理急停按钮；"
              "脚本守护触发会先切全电机阻尼(kd=10)")

    @staticmethod
    def _header(name: str) -> str:
        return f"━━━ {name} 测试预案（人工审阅）━━━"

    def _qbase_note(self) -> str:
        if self.q_base is None:
            return "（q_base 待 R0 连接后按实测计算）"
        return ""

    def brief_setup(self) -> str:
        return "\n".join([
            self._header("R0 构造连接"),
            f"[安全化] margin={SOFT_MARGIN}（软=URDF 硬限位）| 本体限速 {SAFE_ARM_VLIM} rad/s"
            f" | MIT 回退增益 kp≤{SAFE_ARM_MIT_KP}/kd≤{SAFE_ARM_MIT_KD} | 末端 tau≤{SAFE_END_TAU} N·m",
            "[内容] 纯只读：建立串口总线连接（电机保持失能），读取当前关节角，",
            "       q_base = clamp_q(当前 q)（joint2/3 越界部分将被夹回，作为后续基准）。",
            "[本步无任何运动/使能指令]",
            self._ESTOP,
        ])

    def brief_enable(self) -> str:
        return "\n".join([
            self._header("R1 阻尼使能+位置保持"),
            f"[使能] 先 MIT 阻尼（kp=0, kd={DAMP_KD}，松软可手搬）再切 POSITION 模式",
            f"[指令] 位置保持 q_base{self._qbase_note()}：下发后臂可能出现的唯一运动 = ",
            f"       joint2/3 从越界位（约 +0.15/+0.02 rad）被限速拉回 0（≤{SAFE_ARM_VLIM} rad/s，很轻）",
            "[预期] 保持不动或轻微回位后稳定；1.5 s 后读偏差（<0.05 rad 为正常）",
            self._ESTOP,
        ])

    def brief_velocity(self) -> str:
        return "\n".join([
            self._header("R2 速度模式短时"),
            f"[指令] 模式 VELOCITY：joint1 恒速 {VEL_TEST} rad/s 持续 {VEL_DUR} s"
            f"（预计位移 {VEL_TEST * VEL_DUR:.2f} rad），先零速使能后发速度",
            f"[注意] 速度模式无位置闭环——脚本守护轮询（20ms）：|dq|>{VEL_GUARD_DQ} rad/s "
            "→ 立即 damping_mode",
            f"[收尾] 归零速度 → 切 POSITION 回 q_base（约 1.5 s）",
            self._ESTOP,
        ])

    def brief_set_zero(self) -> str:
        return "\n".join([
            self._header("R3 本体标零与断开"),
            "[流程] 选 |q| 最小关节 j（运行时定）→ 标零 j → 位置回原零位（运动≈|q_j| rad，",
            "       其余关节保持不动）→ 失能 → 恢复标零 → 回原位置 → 验证读数复原（<0.05 rad）",
            "[影响] 编码器零位两次写入，最终还原；期间仅 j 关节小幅往返",
            "[收尾] 失能全部电机 → 断开连接（本脚本终点）",
            self._ESTOP,
        ])

    def step_setup(self) -> None:
        self.arm.connect()
        q = self._q_now()
        self.q_base = self.arm.clamp_q(q)
        print(f"✓ 已连接；当前 q：{_fmt(q)}")
        print(f"  q_base（夹紧后基准）：{_fmt(self.q_base)}")
        print(f"✓ JoyArm（model={self.arm.model}）：{self.arm!r}")

    def step_enable(self) -> None:
        q = self._q_now()
        self.arm.set_mode_arm(ControlMode.MIT)
        self.arm.enable_arm()
        self.arm.set_arm_command(ControlMode.MIT, q=q, dq=np.zeros(6), tau=np.zeros(6),
                                 kp=np.zeros(6), kd=np.full(6, DAMP_KD))
        self.arm.set_mode_arm(ControlMode.POSITION)
        self.arm.set_arm_command(ControlMode.POSITION, q=self.q_base)
        time.sleep(1.5)
        err = np.abs(self._q_now() - self.q_base).max()
        print(f"✓ 阻尼使能 → POSITION 保持 q_base（偏差 {err:.4f} rad）")

    def step_velocity(self) -> None:
        arm = self.arm
        arm.set_mode_arm(ControlMode.VELOCITY)
        arm.enable_arm()
        arm.set_arm_command(ControlMode.VELOCITY, dq=np.zeros(6))
        q0 = float(self._q_now()[0])
        arm.set_arm_command(ControlMode.VELOCITY, dq=[VEL_TEST], joint=0)
        self._guarded_velocity(np.array([VEL_TEST]), VEL_DUR)
        arm.set_arm_command(ControlMode.VELOCITY, dq=[0.0], joint=0)
        q1 = float(self._q_now()[0])
        arm.set_mode_arm(ControlMode.POSITION)
        arm.set_arm_command(ControlMode.POSITION, q=self.q_base)
        time.sleep(1.5)
        print(f"✓ joint1 {VEL_TEST} rad/s×{VEL_DUR}s：Δ={q1 - q0:+.4f} rad，"
              f"脚本守护（|dq|>{VEL_GUARD_DQ}→阻尼）未触发，已回基准")

    def step_set_zero_teardown(self) -> None:
        arm = self.arm
        qs = self._q_now()
        j = int(np.argmin(np.abs(qs)))
        q0 = float(qs[j])
        print(f"选定 {self.arm_names[j]}（|q| 最小）：当前 {q0:+.4f} rad（自动恢复流程）")
        arm.set_zero_arm(joint=j)
        arm.set_mode_arm(ControlMode.POSITION)
        arm.enable_arm()
        hold = qs.copy()
        hold[j] = -q0
        arm.set_arm_command(ControlMode.POSITION, q=hold)
        t0 = time.time()
        while time.time() - t0 < 6.0:
            if abs(float(self._q_now()[j]) - (-q0)) <= 0.05:
                break
            time.sleep(0.08)
        arm.disable_arm()
        arm.set_zero_arm(joint=j)
        arm.enable_arm()
        arm.set_arm_command(ControlMode.POSITION, q=qs)
        t0 = time.time()
        while time.time() - t0 < 6.0:
            if abs(float(self._q_now()[j]) - q0) <= 0.05:
                break
            time.sleep(0.08)
        arm.disable_arm()
        q_now = float(self._q_now()[j])
        assert abs(q_now - q0) < 0.05, f"零位恢复失败：{q_now:+.4f} vs {q0:+.4f}"
        print(f"✓ 标零→恢复完成（{q_now:+.4f} ≈ {q0:+.4f}）")
        arm.disable_arm()
        arm.disable_end()
        arm.disconnect()
        assert arm.connected is False
        print("✓ 已失能全部并断开")

    # ----------------------------------------------------------
    # 执行 / 清理 / 报告
    # ----------------------------------------------------------
    def run(self) -> None:
        steps = self._steps()
        total = len(steps)
        for idx, (title, risk, desc, fn, brief) in enumerate(steps, 1):
            print("\n" + brief())                          # 测试预案卡（人工审阅）
            act = self._confirm(idx, total, title, risk, desc)
            if act == "quit":
                print("用户退出")
                break
            if act == "skip":
                self.results.append((title, risk, "跳过"))
                continue
            t0 = time.time()
            try:
                fn()
                self.results.append((title, risk, f"✓ ({time.time() - t0:.1f}s)"))
            except Exception as exc:
                print(f"✗ 失败：{exc}")
                self.results.append((title, risk, "✗"))

    def cleanup(self) -> None:
        if not self.arm.connected:
            return
        try:
            self.arm.disable_arm()
            self.arm.disable_end()
        except (RuntimeError, TimeoutError):
            pass
        self.arm.disconnect()
        print("已清理：失能全部 → 断开连接")

    def report(self) -> None:
        n_ok = sum(1 for _, _, s in self.results if s.startswith("✓"))
        n_bad = sum(1 for _, _, s in self.results if s.startswith("✗"))
        n_skip = sum(1 for _, _, s in self.results if s == "跳过")
        print("\n" + "═" * 56)
        print(f"复测汇总：✓ {n_ok} 成功 | ✗ {n_bad} 失败 | 跳过 {n_skip}（共 {len(self._steps())} 项）")
        for title, risk, status in self.results:
            print(f"  [{risk}] {title:<34}{status}")


def main() -> None:
    if "--list" in sys.argv:
        t = JoyArmRecoveryTest()
        steps = t._steps()
        for idx, (title, risk, desc, _, _) in enumerate(steps, 1):
            print(f"[{idx}/{len(steps)}] {risk:<2} {title} —— {desc}")
        return
    if "--plan" in sys.argv:                             # 全量预案预演（不连接、不执行）
        t = JoyArmRecoveryTest()
        for title, risk, _, _, brief in t._steps():
            print(f"\n[{risk}] {title}")
            print(brief())
        print("\n（--plan 仅预演，未触碰硬件；q_base 相关数值在正式运行 R0 后为实测值）")
        return
    if not sys.stdin.isatty():
        print("需交互终端运行（防误触发真机）；查看步骤请用 --list")
        sys.exit(1)
    print(_BANNER)
    if input("确认连杆已修复、工作空间已清空、急停可用、有人监护；输入 yes 开始: ").strip().lower() != "yes":
        print("已取消")
        return
    t = JoyArmRecoveryTest()
    try:
        t.run()
    except (KeyboardInterrupt, EOFError):
        print("\n中断")
    finally:
        t.cleanup()
        t.report()


if __name__ == "__main__":
    main()
