"""BackendDM 完全功能测试：公开接口全覆盖，按风险由低到高逐项执行（需真机）。

【功能概要】
顺序执行 42 个测试项（顺序即风险递增序），每项执行前打印提示（编号/名称/
风险/说明）并等待确认，高风险项须输入 yes。覆盖 BackendDM 全部公开接口：
离线语义与守卫 → 连接 → 读状态（整臂/子集）→ 读参数（只读/可写/单电机）→
写参数（列表/标量广播/单关节/末端/只读拒绝/错误路径）→ 模式切换（三模式/
单关节/混合语义）→ 安全使能（MIT 阻尼）→ MIT 指令（显式增益/回退增益/单关节/
维度校验）→ 位置模式（保持/逐关节小幅往返/模式不匹配守卫）→ 速度模式（零速/
短时小速）→ 末端（位置标量与序列/越行程裁剪/离散动作/非法动作/力度控制）→
存闪存（本体/末端）→ 设零位（本体/末端，均带自动恢复）→ 断开与断开后守卫。

【安全设计】
- 构造前对 config 安全化：本体 POS_VEL.vlim 封顶 0.5 rad/s、MIT 回退增益
  封顶 kp=20/kd=2；末端 vlim/dq_max 封顶 1.0 rad/s、tau_max 封顶 0.5 N·m。
- 运动幅度：本体单关节 ±0.05 rad、末端 ±0.2 rad（动作/裁剪测试为全行程，
  自动限速）；速度测试 0.2 rad/s×0.6 s；力度 0.5 N。
- MIT 使能后立刻下发 kp=0/kd=1.5 阻尼保持（非零阻抗松软），减缓坠落。
- 固件参数改动（max_spd）写后即时读回并恢复；异常退出时兜底恢复。
- 高风险项（速度模式/存闪存×2/设零位×2）须输入 yes；设零位均含自动恢复流程。
- try/finally：任何退出路径先恢复参数 → 失能全部电机 → 断开连接 → 汇总报告。

【安全提示】
上电使能前：清空机械臂工作空间、确认急停可用、扶稳大臂防坠落；
本脚本会真实驱动电机，请务必在有人监护下使用；力度测试时夹爪内勿放手指。

【不可真机覆盖项】（离线单测 test_backend_dm.py 已覆盖）
故障码 8~E 注入（不可安全人为触发）、CANID==0 旧固件回退分发（依赖固件
形态）、persist 掉电重启验证（需人工断电）、多电机末端（本型号为单夹爪）。

【环境与运行】
1. 新建环境并安装依赖：``cd joyarm_code && uv venv && uv sync && source .venv/bin/activate``
2. 接好达妙 CAN 桥（默认 ``/dev/ttyACM0``）与电机电源后运行：
   ``python test/test_backend_dm_full.py``          # 交互逐项执行
   ``python test/test_backend_dm_full.py --list``   # 仅列出测试项，不执行
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core.backend.backend_dm import (  # noqa: E402
    _MOTOR_LIMITS,
    _PARAM_RIDS,
    _READONLY_KEYS,
    BackendDM,
)
from joyarm_core.utils.types import ControlMode  # noqa: E402

_PARAM_KEYS = sorted(_PARAM_RIDS)                          # 全部可读参数名
_WRITABLE_KEYS = sorted(set(_PARAM_RIDS) - _READONLY_KEYS)  # 可写参数名
_VERSION_KEYS = ("hw_ver", "sw_ver", "sn", "sub_ver")       # 版本身份（只读）
_PHYSICS_KEYS = ("kt", "gr", "pmax", "vmax", "tmax")        # 物理特性（只读）

# ---- 安全上限（构造前写进 config，全程生效）----
SAFE_ARM_VLIM = 0.5      # 本体位置指令限速 rad/s（config POS_VEL.vlim 封顶）
SAFE_ARM_MIT_KP = 20.0   # 本体 MIT 回退增益封顶（kp/kd 逐项取 min）
SAFE_ARM_MIT_KD = 2.0
SAFE_END_VLIM = 1.0      # 末端限速 rad/s
SAFE_END_DQ = 1.0        # 末端速度上限封顶 rad/s
SAFE_END_TAU = 0.5       # 末端力矩上限封顶 N·m（即"最大力矩调小"）
# ---- 测试动作幅度（全程只做小幅运动）----
MOVE_AMP = 0.05          # 本体单关节往返幅度 rad
VEL_TEST = 0.2           # 速度模式测试速度 rad/s
VEL_DUR = 0.6            # 速度模式持续时间 s
MIT_KP_TEST = 5.0        # MIT 小阻抗保持增益
MIT_KD_TEST = 1.0
DAMP_KD = 1.5            # 使能后阻尼保持（kp=0，防坠落冲击）
END_AMP = 0.2            # 末端位置往返幅度 rad
END_TAU = 0.5             # 末端力矩测试 N·m

_BANNER = """
╔══════════════════════════════════════════════════════════╗
║       BackendDM 完全功能测试（真机，42 项，风险递增）        ║
╚══════════════════════════════════════════════════════════╝
⚠ 本脚本会真实驱动电机（含小幅运动与末端夹合）：
  - 清空机械臂工作空间，确认急停可用，全程有人监护；
  - 已自动收紧：限速 0.5 rad/s、末端力矩 0.5 N·m、MIT 增益 kp≤20/kd≤2；
  - 高风险项（速度模式/存闪存/设零位）须输入 yes 才执行。
"""


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _fmt(a, prec: int = 4) -> str:
    return np.array2string(np.asarray(a, dtype=float), precision=prec)


class BackendDMFullTest:
    """顺序执行器：持 BackendDM 实例与测试上下文（末端行程、待恢复参数、结果表）。"""

    def __init__(self) -> None:
        cfg = yaml.safe_load(
            (_ROOT / "joyarm_core" / "configs" / "joyarm_dm.yaml").read_text(encoding="utf-8")
        )
        bcfg = dict(cfg["backend"])
        bcfg.pop("name")
        self._safe_diff = self._safeguard(bcfg)          # [(对象, 键, 原值, 新值)]
        self.be = BackendDM(bcfg)
        self.arm_names = [j["name"] for j in bcfg["arm"]["joints"]]
        self.arm_models = [j.get("model", "?") for j in bcfg["arm"]["joints"]]
        end_joints = (bcfg.get("end") or {}).get("joints") or []
        self.end_names = [j["name"] for j in end_joints]
        ej = end_joints[0] if end_joints else {}
        self.end_qmin = float(ej.get("q_min", 0.0))      # 末端行程（到位判断）
        self.end_qmax = float(ej.get("q_max", 0.0))
        self._pending: list[dict] = []                   # 异常退出兜底恢复的固件参数
        self.results: list[tuple[str, str, str]] = []    # (标题, 风险, 结果)

    # ----------------------------------------------------------
    # config 安全化（构造 BackendDM 前）
    # ----------------------------------------------------------
    @staticmethod
    def _safeguard(bcfg: dict) -> list[tuple[str, str, float, float]]:
        diff: list[tuple[str, str, float, float]] = []
        for j in bcfg["arm"]["joints"]:
            pv = j.setdefault("POS_VEL", {})
            old, pv["vlim"] = float(pv.get("vlim", 0.0)), min(float(pv.get("vlim", SAFE_ARM_VLIM)), SAFE_ARM_VLIM)
            diff.append((j["name"], "POS_VEL.vlim", old, pv["vlim"]))
            mit = j.setdefault("MIT", {})
            for key, cap in (("kp", SAFE_ARM_MIT_KP), ("kd", SAFE_ARM_MIT_KD)):
                old, mit[key] = float(mit.get(key, 0.0)), min(float(mit.get(key, cap)), cap)
                diff.append((j["name"], f"MIT.{key}", old, mit[key]))
        for j in (bcfg.get("end") or {}).get("joints") or []:
            pv = j.setdefault("POS_VEL", {})
            old, pv["vlim"] = float(pv.get("vlim", 0.0)), min(float(pv.get("vlim", SAFE_END_VLIM)), SAFE_END_VLIM)
            diff.append((j["name"], "POS_VEL.vlim", old, pv["vlim"]))
            for key, cap in (("dq_max", SAFE_END_DQ), ("tau_max", SAFE_END_TAU)):
                old, j[key] = float(j.get(key, cap)), min(float(j.get(key, cap)), cap)
                diff.append((j["name"], key, old, j[key]))
        return diff

    # ----------------------------------------------------------
    # 交互与通用辅助
    # ----------------------------------------------------------
    def _confirm(self, idx: int, total: int, title: str, risk: str, desc: str) -> str:
        print("\n" + "─" * 56)
        print(f"[步骤 {idx:>2}/{total}] 风险:{risk} | {title}")
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

    def _print_arm_state(self, joint=None) -> None:
        s = self.be.read_state_arm(joint)
        js = s.joint
        names = self.arm_names if joint is None else [self.arm_names[joint]]
        print(f"{'关节':<9}{'q(rad)':>10}{'dq(rad/s)':>10}{'tau(N·m)':>9}{'tMOS':>5}{'tRot':>5}  使能 故障 通讯")
        for nm, q, dq, tau, tm, tr, en, fl, ok in zip(
                names, js.q, js.dq, js.tau, js.temp_mos, js.temp_rotor,
                js.enabled, js.error, js.comm_ok):
            t2 = lambda t: "—" if t == 0 else f"{t:.0f}"
            print(f"{nm:<10}{q:+10.4f}{dq:+10.4f}{tau:+9.3f}{t2(tm):>5}{t2(tr):>5}"
                  f"   {'是' if en else '否'}   {'是' if fl else '否'}   {'是' if ok else '否'}")
        print(f"故障：{'；'.join(s.errors) if s.errors else '无'}")

    def _print_end_state(self, joint=None) -> None:
        e = self.be.read_state_end(joint)
        names = self.end_names if joint is None else [self.end_names[joint]]
        print(f"{'电机':<10}{'q(rad)':>10}{'dq(rad/s)':>10}{'tau(N·m)':>9}{'tMOS':>5}{'tRot':>5}  使能 故障 通讯")
        for nm, q, dq, tau, tm, tr, en, fl, ok in zip(
                names, e["q"], e["dq"], e["tau"], e["temp_mos"], e["temp_rotor"],
                e["enabled"], e["error"], e["comm_ok"]):
            t2 = lambda t: "—" if t == 0 else f"{t:.0f}"
            print(f"{nm:<10}{q:+10.4f}{dq:+10.4f}{tau:+9.3f}{t2(tm):>5}{t2(tr):>5}"
                  f"   {'是' if en else '否'}   {'是' if fl else '否'}   {'是' if ok else '否'}")

    def _wait_arm_q(self, joint: int, target: float, tol: float = 0.03,
                    timeout: float = 4.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if abs(float(self.be.read_state_arm(joint).joint.q[0]) - target) <= tol:
                return True
            time.sleep(0.08)
        return False

    def _wait_end_q(self, target: float, tol: float = 0.15, timeout: float = 10.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if abs(float(self.be.read_state_end()["q"][0]) - target) <= tol:
                return True
            time.sleep(0.08)
        return False

    @staticmethod
    def _expect(exc_type, call, what: str) -> None:
        try:
            call()
            raise AssertionError(f"{what} 应抛 {exc_type.__name__}")
        except exc_type:
            print(f"  ✓ {what} → {exc_type.__name__}")

    # ----------------------------------------------------------
    # 测试步骤（顺序即执行序，风险由低到高；42 项）
    # ----------------------------------------------------------
    def _steps(self) -> list[tuple[str, str, str, object]]:
        return [
            ("安全化配置并构造 BackendDM", "低", "打印限速/限矩/增益收紧对照表（不连接、无运动）", self.step_construct),
            ("离线语义与未连接守卫", "低", "connected=False、read_mode 未设置=None、未连接调用应抛 RuntimeError", self.step_offline),
            ("connect() 建立通信", "低", "打开串口总线与 RX 线程（arm/end 同通道共享总线），电机保持失能", self.step_connect),
            ("read_state_arm（失能态）", "低", "全量读取 6 关节状态表（q/dq/tau/温度/使能/故障/通讯）", self.step_state_arm),
            ("read_state_end（失能态）", "低", "读取末端夹爪状态表（joint=None 整组 + joint=0 子集）", self.step_state_end),
            ("read_param 版本身份（只读）", "低", "本体全部+joint0 子集+末端：hw_ver/sw_ver/sn/sub_ver", self.step_param_version),
            ("read_param 物理特性（只读）", "低", "kt/gr/pmax/vmax/tmax，并与 config 型号限值对照", self.step_param_physics),
            ("read_param 可写参数快照", "低", "全部可写参数当前值（后续写入步骤的回读基准）", self.step_param_snapshot),
            ("错误路径抽查①", "低", "未知参数名、joint 越界 → ValueError（失能态，无总线副作用）", self.step_err_paths),
            ("write_param acc 原值回写", "低", "读→逐电机列表写回原值→验证写确认机制，不改变量", self.step_param_writeback),
            ("write_param acc 标量广播+单关节", "低", "标量广播到全部电机→恢复；joint=0 单独写→验证（仅 RAM）", self.step_param_scalar),
            ("write_param max_spd 调小→恢复", "中", "固件限速临时调小（≤2 rad/s）读回确认，随后恢复原值", self.step_param_maxspd),
            ("write_param_end acc 回写（末端）", "低", "末端电机参数真实写入与写确认（原值回写）", self.step_param_end_write),
            ("write_param 只读守卫", "低", "sw_ver/pmax 为只读参数，写入应抛 ValueError", self.step_param_readonly),
            ("set_mode_arm(MIT)", "中", "写 RID10 切 MIT 模式（失能态无运动），read_mode 整臂/joint 验证", self.step_mode_mit),
            ("read_mode_arm 混合语义", "低", "joint1 单独切 POSITION → 整臂 None、joint1=POSITION、joint2=MIT → 恢复", self.step_mode_mixed),
            ("disable_arm 幂等", "低", "失能态再次失能不报错", self.step_disable_idle),
            ("安全使能（MIT 阻尼保持）", "中", "使能后立刻下发 kp=0/kd=1.5 阻尼指令；⚠ 请扶稳大臂防下垂", self.step_safe_enable),
            ("read_state_arm（使能态+子集）", "低", "enabled 应全为 True；joint4 子集单独读取", self.step_state_enabled),
            ("MIT 小阻抗保持（显式增益）", "中", "kp=5/kd=1 弱刚度保持当前位形 2 s（力矩极小）", self.step_mit_hold),
            ("MIT 回退增益（kp/kd=None）", "中", "缺省增益回退 config MIT 段（已封顶 kp≤20/kd≤2）保持 2 s", self.step_mit_fallback),
            ("MIT 单关节指令（joint6）", "中", "joint 子集 MIT 阻尼保持 1.5 s，其余关节不受影响", self.step_mit_single),
            ("失能验证+MIT 维度校验+模式守卫", "低", "失能后 enabled=False；MIT 下发 3 维向量→ValueError；MIT 下发位置指令→RuntimeError", self.step_disable_mit_checks),
            ("set_mode_arm(POSITION)", "中", "写 POS_VEL 闭环增益（RID25~28）+ 切模式（失能态切换）", self.step_mode_position),
            ("使能+位置保持", "中", "发送当前位形为目标（应保持不动，限速 0.5 rad/s）", self.step_pos_hold),
            ("逐关节小幅往返（±0.05 rad）", "中", "joint1~6 依次单关节 +0.05 rad → 回原位，验证位置通道", self.step_pos_sweep),
            ("disable_arm", "低", "位置测试后失能", self.step_disable_plain),
            ("set_mode_arm(VELOCITY)+零速", "高", "切速度模式并使能，先发 0 速度（速度模式无位置守卫）", self.step_mode_vel),
            ("joint1 短时小速度", "高", "0.2 rad/s 持续 0.6 s（约 0.12 rad）→ 归零；⚠ 确认基座周围无障碍", self.step_vel_move),
            ("disable_arm", "低", "速度测试后失能", self.step_disable_plain),
            ("末端：POSITION+使能+保持", "中", "夹爪切位置模式、使能、发送当前位置（joint=0 子集读状态）", self.step_end_enable),
            ("send_position_end 小幅往返", "中", "±0.2 rad（行程内限幅）：标量形式 + 序列形式 [x] → 回原位", self.step_end_pos),
            ("send_position_end 越行程裁剪", "中", "发送超出行程的目标 99 rad，验证被裁剪到 q_max（自动限速）", self.step_end_clamp),
            ("send_action_end 离散动作", "中", "open→close→zero（行程内限幅、限速）+ 非法动作拒绝", self.step_end_action),
            ("disable_end", "低", "末端失能", self.step_end_disable),
            ("末端：MIT 力矩控制", "中", "0.5 N·m 经 MIT 近似闭合（前馈封顶 0.5 N·m）；⚠ 夹爪内勿放手指", self.step_end_tau),
            ("disable_end", "低", "力度测试后失能", self.step_end_disable),
            ("write_param_arm persist 存闪存", "高", "acc 原值写入并存闪存（自动失能并保持）；掉电不丢，验证读回", self.step_persist_arm),
            ("write_param_end persist 存闪存", "高", "末端 acc 原值存闪存（自动失能并保持），验证读回", self.step_persist_end),
            ("set_zero_arm（自动恢复）", "高", "选 |q| 最小关节：标零→回原零位→恢复标零→回原位置→验证", self.step_set_zero_arm),
            ("set_zero_end（自动恢复）", "高", "末端标零→回原零位→恢复标零→回原位置→验证", self.step_set_zero_end),
            ("disconnect 断开+断开后守卫", "低", "失能→停 RX 线程→关串口；断开后调用应抛 RuntimeError", self.step_disconnect),
        ]

    # ---- 离线与连接 ----
    def step_construct(self) -> None:
        print(f"参数映射：读 {_PARAM_KEYS}")
        print("安全化对照（原值 → 测试值）：")
        for name, key, old, new in self._safe_diff:
            mark = "" if abs(old - new) < 1e-9 else "  ← 已收紧"
            print(f"  {name:<10}{key:<14}{old:>8.3f} → {new:<8.3f}{mark}")
        print(f"✓ BackendDM 已构造（未连接）：本体 {len(self.arm_names)} 关节 + 末端 {len(self.end_names)} 电机")

    def step_offline(self) -> None:
        assert self.be.connected is False
        assert self.be.read_mode_arm() is None and self.be.read_mode_arm(0) is None
        assert self.be.read_mode_end() is None
        guards = [
            lambda: self.be.enable_arm(),
            lambda: self.be.read_state_arm(),
            lambda: self.be.read_param_arm("pos_kp"),
            lambda: self.be.write_param_arm("acc", 1.0),
            lambda: self.be.enable_end(),
            lambda: self.be.set_mode_arm(),
        ]
        for call in guards:
            try:
                call()
                raise AssertionError("未连接应抛 RuntimeError")
            except RuntimeError:
                pass
        print(f"✓ 离线语义正确：connected=False、read_mode=None、{len(guards)} 项未连接守卫均抛 RuntimeError")

    def step_connect(self) -> None:
        self.be.connect()
        assert self.be.connected is True
        print("✓ 已连接（arm/end 同通道共享单总线；电机保持失能）")

    # ---- 状态读取 ----
    def step_state_arm(self) -> None:
        self._print_arm_state()

    def step_state_end(self) -> None:
        print("joint=None 整组：")
        self._print_end_state()
        print("joint=0 子集：")
        self._print_end_state(0)

    # ---- 参数读取 ----
    def step_param_version(self) -> None:
        print("本体版本身份（joint=None 整臂）：")
        for key in _VERSION_KEYS:
            vals = self.be.read_param_arm(key)
            print(f"  {key:<9}: " + " ".join(f"{nm}={v}" for nm, v in zip(self.arm_names, vals)))
        for key in _VERSION_KEYS:
            print(f"  joint1 子集 {key:<9}: {self.be.read_param_arm(key, 0)}")
        print("末端版本身份：")
        for key in _VERSION_KEYS:
            print(f"  {key:<9}: " + " ".join(f"{nm}={v}" for nm, v in zip(self.end_names, self.be.read_param_end(key))))

    def step_param_physics(self) -> None:
        vals = {k: self.be.read_param_arm(k) for k in _PHYSICS_KEYS}
        print(f"{'关节':<9}{'kt':>8}{'gr':>6}{'pmax':>7}{'vmax':>7}{'tmax':>7}   config 限值(pmax,vmax,tmax)")
        for i, nm in enumerate(self.arm_names):
            exp = _MOTOR_LIMITS[self.arm_models[i]]
            print(f"{nm:<10}{vals['kt'][i]:>8.3f}{vals['gr'][i]:>6.2f}{vals['pmax'][i]:>7.2f}"
                  f"{vals['vmax'][i]:>7.1f}{vals['tmax'][i]:>7.1f}   ({exp[0]}, {exp[1]}, {exp[2]})")
        bad = [self.arm_names[i] for i in range(len(self.arm_names))
               if not all(abs(vals[k][i] - _MOTOR_LIMITS[self.arm_models[i]][j]) < 1e-3
                          for j, k in enumerate(("pmax", "vmax", "tmax")))]
        print(f"✓ pmax/vmax/tmax 与 config 型号限值{'一致' if not bad else '不一致：' + str(bad)}")
        print("末端物理特性：")
        for key in _PHYSICS_KEYS:
            print(f"  {key:<9}: {self.be.read_param_end(key)}")

    def step_param_snapshot(self) -> None:
        print(f"{'参数':<10}本体（逐关节）")
        for key in _WRITABLE_KEYS:
            print(f"  {key:<9}{_fmt(self.be.read_param_arm(key), 4)}")
        if self.end_names:
            print("末端：")
            for key in _WRITABLE_KEYS:
                print(f"  {key:<9}{self.be.read_param_end(key)}")

    def step_err_paths(self) -> None:
        self._expect(ValueError, lambda: self.be.read_param_arm("bad_key"), "read_param_arm 未知参数名")
        self._expect(ValueError, lambda: self.be.read_state_arm(9), "read_state_arm joint 越界")
        self._expect(ValueError, lambda: self.be.read_param_end("bad_key"), "read_param_end 未知参数名")
        self._expect(ValueError, lambda: self.be.read_state_end(3), "read_state_end joint 越界")

    # ---- 参数写入 ----
    def step_param_writeback(self) -> None:
        orig = self.be.read_param_arm("acc")
        self.be.write_param_arm("acc", list(orig))       # 逐电机写回原值（写确认内建）
        back = self.be.read_param_arm("acc")
        assert all(abs(a - b) <= max(1e-4, 1e-3 * abs(b)) for a, b in zip(back, orig)), \
            f"回读不一致：{back} vs {orig}"
        print(f"✓ acc 原值回写确认（逐电机列表）：{_fmt(back)}")

    def step_param_scalar(self) -> None:
        orig = self.be.read_param_arm("acc")
        scalar = float(orig[0])
        self._pending.append({"key": "acc", "value": list(orig)})
        self.be.write_param_arm("acc", scalar)           # 标量广播到全部电机
        got = self.be.read_param_arm("acc")
        assert all(abs(g - scalar) <= max(1e-4, 1e-3 * abs(scalar)) for g in got), f"标量广播未生效：{got}"
        print(f"✓ 标量广播 {scalar:g} → 全部电机：{_fmt(got)}")
        self.be.write_param_arm("acc", list(orig))       # 恢复逐电机原值
        self.be.write_param_arm("acc", scalar, joint=0)  # 单关节子集写
        got1 = self.be.read_param_arm("acc", 0)
        assert abs(got1 - scalar) <= max(1e-4, 1e-3 * abs(scalar))
        rest = self.be.read_param_arm("acc")
        assert abs(rest[1] - orig[1]) <= max(1e-4, 1e-3 * abs(orig[1])), "单关节写不应影响其他电机"
        self._pending.pop()
        print(f"✓ joint=0 子集写 {scalar:g}：读回 {got1:g}，其余电机不受影响")

    def step_param_maxspd(self) -> None:
        orig = self.be.read_param_arm("max_spd")
        self._pending.append({"key": "max_spd", "value": list(orig)})
        new = [max(0.5, min(v * 0.5, 2.0)) for v in orig]
        self.be.write_param_arm("max_spd", new)
        got = self.be.read_param_arm("max_spd")
        assert all(abs(g - n) <= max(1e-4, 1e-3 * abs(n)) for g, n in zip(got, new)), \
            f"限速未生效：{got} vs {new}"
        print(f"✓ max_spd 临时调小：{_fmt(orig)} → {_fmt(got)}")
        self.be.write_param_arm("max_spd", list(orig))
        got = self.be.read_param_arm("max_spd")
        assert all(abs(g - o) <= max(1e-4, 1e-3 * abs(o)) for g, o in zip(got, orig))
        self._pending.pop()
        print(f"✓ max_spd 已恢复：{_fmt(got)}")

    def step_param_end_write(self) -> None:
        orig = self.be.read_param_end("acc")
        scalar = float(orig[0]) if isinstance(orig, list) else float(orig)
        self.be.write_param_end("acc", scalar)           # 末端真实写入（原值，写确认内建）
        got = self.be.read_param_end("acc")
        got_v = got[0] if isinstance(got, list) else got
        assert abs(got_v - scalar) <= max(1e-4, 1e-3 * abs(scalar)), f"末端回读不一致：{got} vs {scalar}"
        print(f"✓ 末端 acc 原值回写确认：{got_v:g}")

    def step_param_readonly(self) -> None:
        self._expect(ValueError, lambda: self.be.write_param_arm("sw_ver", 1), "write_param_arm 只读参数 sw_ver")
        self._expect(ValueError, lambda: self.be.write_param_end("pmax", 1.0), "write_param_end 只读参数 pmax")

    # ---- 本体：模式与 MIT ----
    def step_mode_mit(self) -> None:
        self.be.set_mode_arm(ControlMode.MIT)
        assert self.be.read_mode_arm() == ControlMode.MIT
        assert self.be.read_mode_arm(3) == ControlMode.MIT
        print("✓ 已切 MIT 模式（read_mode_arm 整臂/joint4 均验证）")

    def step_mode_mixed(self) -> None:
        self.be.set_mode_arm(ControlMode.POSITION, joint=0)   # 仅 joint1 切 POSITION
        assert self.be.read_mode_arm() is None, "混合模式整臂应返回 None"
        assert self.be.read_mode_arm(0) == ControlMode.POSITION
        assert self.be.read_mode_arm(1) == ControlMode.MIT
        print("✓ 混合语义：joint1=POSITION、joint2=MIT → 整臂 None，子集各自正确")
        self.be.set_mode_arm(ControlMode.MIT, joint=0)        # 恢复整臂一致
        assert self.be.read_mode_arm() == ControlMode.MIT

    def step_disable_idle(self) -> None:
        self.be.disable_arm()
        print("✓ 失能态再次 disable_arm 无异常（幂等）")

    def step_safe_enable(self) -> None:
        q = np.array(self.be.read_state_arm().joint.q, dtype=float)
        self.be.enable_arm()
        self.be.send_mit_arm(q, np.zeros(6), np.zeros(6),
                             np.zeros(6), np.full(6, DAMP_KD))
        print(f"✓ 已使能并下发阻尼保持（kp=0, kd={DAMP_KD}）：目标 q={_fmt(q)}")
        print("  电机近似黏滞阻尼，可缓慢手动搬动；请扶稳大臂")

    def step_state_enabled(self) -> None:
        self._print_arm_state()
        assert bool(np.all(self.be.read_state_arm().joint.enabled)), "使能态 enabled 应全为 True"
        print("joint4 子集：")
        self._print_arm_state(3)

    def step_mit_hold(self) -> None:
        q = np.array(self.be.read_state_arm().joint.q, dtype=float)
        self.be.send_mit_arm(q, np.zeros(6), np.zeros(6),
                             np.full(6, MIT_KP_TEST), np.full(6, MIT_KD_TEST))
        time.sleep(2.0)
        print(f"✓ MIT 小阻抗保持 2 s（kp={MIT_KP_TEST}, kd={MIT_KD_TEST}）后状态：")
        self._print_arm_state()

    def step_mit_fallback(self) -> None:
        q = np.array(self.be.read_state_arm().joint.q, dtype=float)
        self.be.send_mit_arm(q, np.zeros(6), np.zeros(6))   # kp/kd=None → 回退 config（已封顶）
        time.sleep(2.0)
        print("✓ MIT 回退增益保持 2 s（kp/kd=config 封顶值 kp≤20/kd≤2）后状态：")
        self._print_arm_state()

    def step_mit_single(self) -> None:
        q6 = float(self.be.read_state_arm(5).joint.q[0])
        self.be.send_mit_arm(np.array([q6]), np.zeros(1), np.zeros(1),
                             np.zeros(1), np.full(1, MIT_KD_TEST), joint=5)
        time.sleep(1.5)
        q6b = float(self.be.read_state_arm(5).joint.q[0])
        print(f"✓ joint6 单关节 MIT 阻尼保持 1.5 s：q {q6:+.4f} → {q6b:+.4f} rad（其余关节不受影响）")

    def step_disable_mit_checks(self) -> None:
        self.be.disable_arm()
        assert not bool(np.any(self.be.read_state_arm().joint.enabled)), "失能后 enabled 应全为 False"
        print("✓ 已失能（enabled 全为 False）")
        self._expect(ValueError,
                     lambda: self.be.send_mit_arm(np.zeros(3), np.zeros(3), np.zeros(3)),
                     "MIT 指令维度不匹配（3≠6）")
        q = np.array(self.be.read_state_arm().joint.q, dtype=float)
        self._expect(RuntimeError, lambda: self.be.send_position_arm(q),
                     "MIT 模式下发位置指令（模式不匹配）")
        self._expect(RuntimeError, lambda: self.be.send_velocity_arm(np.zeros(6)),
                     "MIT 模式下发速度指令（模式不匹配）")

    # ---- 本体：位置模式 ----
    def step_mode_position(self) -> None:
        self.be.set_mode_arm(ControlMode.POSITION)
        assert self.be.read_mode_arm() == ControlMode.POSITION
        print("✓ 已切 POSITION 模式（含 POS_VEL 闭环增益写入）")

    def step_pos_hold(self) -> None:
        q = np.array(self.be.read_state_arm().joint.q, dtype=float)
        self.be.enable_arm()
        self.be.send_position_arm(q)
        print(f"✓ 已使能并保持当前位形（限速 {SAFE_ARM_VLIM} rad/s）：q={_fmt(q)}")

    def step_pos_sweep(self) -> None:
        q0 = np.array(self.be.read_state_arm().joint.q, dtype=float)
        worst = 0.0
        for j, nm in enumerate(self.arm_names):
            target = float(q0[j]) + MOVE_AMP
            self.be.send_position_arm(np.array([target]), joint=j)
            ok1 = self._wait_arm_q(j, target)
            self.be.send_position_arm(np.array([float(q0[j])]), joint=j)
            ok2 = self._wait_arm_q(j, float(q0[j]))
            err = abs(float(self.be.read_state_arm(j).joint.q[0]) - float(q0[j]))
            worst = max(worst, err)
            print(f"  {nm:<9}+{MOVE_AMP} rad 往返：{'✓' if ok1 and ok2 else '未到位'}（回位误差 {err:.4f} rad）")
        print(f"✓ 逐关节小幅往返完成（最大回位误差 {worst:.4f} rad）")

    def step_disable_plain(self) -> None:
        self.be.disable_arm()
        print("✓ 已失能")

    # ---- 本体：速度模式（高风险）----
    def step_mode_vel(self) -> None:
        self.be.set_mode_arm(ControlMode.VELOCITY)
        assert self.be.read_mode_arm() == ControlMode.VELOCITY
        self.be.enable_arm()
        self.be.send_velocity_arm(np.zeros(6))
        print("✓ 速度模式已使能并保持零速")

    def step_vel_move(self) -> None:
        q0 = float(self.be.read_state_arm(0).joint.q[0])
        self.be.send_velocity_arm(np.array([VEL_TEST]), joint=0)
        time.sleep(VEL_DUR)
        self.be.send_velocity_arm(np.zeros(1), joint=0)
        q1 = float(self.be.read_state_arm(0).joint.q[0])
        print(f"✓ joint1 以 {VEL_TEST} rad/s 运行 {VEL_DUR} s：q {q0:+.4f} → {q1:+.4f} rad（Δ={q1 - q0:+.4f}）")

    # ---- 末端 ----
    def step_end_enable(self) -> None:
        self.be.set_mode_end(ControlMode.POSITION)
        assert self.be.read_mode_end() == ControlMode.POSITION
        q = float(self.be.read_state_end(0)["q"][0])     # joint=0 子集读
        self.be.enable_end()
        self.be.send_position_end(q)
        print(f"✓ 末端已使能并保持当前位置（q={q:+.4f} rad，限速 {SAFE_END_VLIM} rad/s）")

    def step_end_pos(self) -> None:
        q0 = float(self.be.read_state_end()["q"][0])
        t1 = _clamp(q0 + END_AMP, self.end_qmin, self.end_qmax)
        self.be.send_position_end(t1)                     # 标量形式
        ok1 = self._wait_end_q(t1)
        print(f"  标量 → {t1:+.4f} rad：{'✓ 到位' if ok1 else '未到位'}")
        t2 = _clamp(q0 - END_AMP, self.end_qmin, self.end_qmax)
        self.be.send_position_end([t2])                   # 序列形式（单元素列表）
        ok2 = self._wait_end_q(t2)
        print(f"  序列 → {t2:+.4f} rad：{'✓ 到位' if ok2 else '未到位'}")
        self.be.send_position_end(q0)
        ok3 = self._wait_end_q(q0)
        print(f"  回原位 {q0:+.4f} rad：{'✓ 到位' if ok3 else '未到位'}")

    def step_end_clamp(self) -> None:
        self.be.send_position_end(99.0)                   # 超出行程 → 应被裁剪到 q_max
        ok = self._wait_end_q(self.end_qmax)
        q = float(self.be.read_state_end()["q"][0])
        assert q <= self.end_qmax + 0.2, f"越出行程上限：{q} > {self.end_qmax}"
        print(f"✓ 目标 99 rad 被裁剪：实际到达 {q:+.4f} rad（q_max={self.end_qmax}，{'到位' if ok else '趋近中'}）")

    def step_end_action(self) -> None:
        for action, target in (("open", self.end_qmin), ("close", self.end_qmax),
                               ("zero", _clamp(0.0, self.end_qmin, self.end_qmax))):
            self.be.send_action_end(action)
            ok = self._wait_end_q(target)
            print(f"  {action:<6}→ {target:+.4f} rad：{'✓ 到位' if ok else '未到位'}")
        self._expect(ValueError, lambda: self.be.send_action_end("bad"),
                     "send_action_end 非法动作")

    def step_end_disable(self) -> None:
        self.be.disable_end()
        print("✓ 末端已失能")

    def step_end_tau(self) -> None:
        self.be.set_mode_end(ControlMode.MIT)
        assert self.be.read_mode_end() == ControlMode.MIT
        self.be.enable_end()
        self.be.send_tau_end(END_TAU)
        ok = self._wait_end_q(self.end_qmax, tol=0.2, timeout=6.0)
        e = self.be.read_state_end()
        print(f"✓ 力矩 {END_TAU} N·m（前馈封顶 {SAFE_END_TAU}）："
              f"闭合{'到位' if ok else '未完全到位'}，tau={e['tau'][0]:+.3f} N·m")

    # ---- 高风险可选项 ----
    def step_persist_arm(self) -> None:
        orig = self.be.read_param_arm("acc")
        self.be.write_param_arm("acc", list(orig), persist=True)   # 自动失能并保持
        got = self.be.read_param_arm("acc")
        assert all(abs(a - b) <= max(1e-4, 1e-3 * abs(b)) for a, b in zip(got, orig))
        print(f"✓ acc={_fmt(orig)} 已存闪存（电机已自动失能），掉电重启后保持")

    def step_persist_end(self) -> None:
        orig = self.be.read_param_end("acc")
        scalar = float(orig[0]) if isinstance(orig, list) else float(orig)
        self.be.write_param_end("acc", scalar, persist=True)
        got = self.be.read_param_end("acc")
        got_v = got[0] if isinstance(got, list) else got
        assert abs(got_v - scalar) <= max(1e-4, 1e-3 * abs(scalar))
        print(f"✓ 末端 acc={scalar:g} 已存闪存（电机已自动失能）")

    def _zero_restore(self, set_zero, set_mode_on, enable_on, send_q, read_q,
                      disable_on, wait_q, joint, q0: float, label: str) -> None:
        """标零自动恢复流程：标零→回原零位→恢复标零→回原位置→验证（全程打印）。"""
        print(f"  {label} 当前读数 {q0:+.4f} rad（全程运动幅度≈{abs(q0):.4f} rad 后自动恢复）")
        set_zero(joint)                                     # 新零位 = 当前物理位置
        set_mode_on(joint)
        enable_on(joint)
        send_q(-q0, joint)                                  # 物理回到原零位
        if not wait_q(-q0, joint):
            print("  ⚠ 回原零位超时，继续恢复流程")
        disable_on(joint)
        set_zero(joint)                                     # 零位恢复为原零位
        enable_on(joint)
        send_q(q0, joint)                                   # 回到原物理位置
        if not wait_q(q0, joint):
            print("  ⚠ 回原位置超时")
        disable_on(joint)
        q_now = read_q(joint)
        assert abs(q_now - q0) < 0.05, f"零位恢复验证失败：{q_now:+.4f} vs {q0:+.4f}"
        print(f"✓ {label} 标零→恢复全流程完成（当前读数 {q_now:+.4f} ≈ 原 {q0:+.4f} rad，零位已还原）")

    def step_set_zero_arm(self) -> None:
        qs = np.array(self.be.read_state_arm().joint.q, dtype=float)
        j = int(np.argmin(np.abs(qs)))
        print(f"选定 {self.arm_names[j]}（|q| 最小）")
        self._zero_restore(
            self.be.set_zero_arm,
            lambda jt: self.be.set_mode_arm(ControlMode.POSITION, jt),
            self.be.enable_arm,
            lambda v, jt: self.be.send_position_arm(np.array([v]), joint=jt),
            lambda jt: float(self.be.read_state_arm(jt).joint.q[0]),
            self.be.disable_arm,
            lambda tgt, jt: self._wait_arm_q(jt, tgt, tol=0.05, timeout=6.0),
            j, float(qs[j]), self.arm_names[j])

    def step_set_zero_end(self) -> None:
        q0 = float(self.be.read_state_end()["q"][0])
        self._zero_restore(
            self.be.set_zero_end,
            lambda jt: self.be.set_mode_end(ControlMode.POSITION, jt),
            self.be.enable_end,
            lambda v, jt: self.be.send_position_end(v, jt),
            lambda jt: float(self.be.read_state_end(jt)["q"][0]),
            self.be.disable_end,
            lambda tgt, jt: self._wait_end_q(tgt, tol=0.05, timeout=12.0),
            0, q0, self.end_names[0] if self.end_names else "end")

    def step_disconnect(self) -> None:
        self.be.disconnect()
        assert self.be.connected is False
        self._expect(RuntimeError, lambda: self.be.read_state_arm(), "断开后 read_state_arm")
        print("✓ 已断开（失能→停 RX 线程→关串口），断开后守卫生效")

    # ----------------------------------------------------------
    # 执行 / 清理 / 报告
    # ----------------------------------------------------------
    def run(self) -> None:
        steps = self._steps()
        total = len(steps)
        for idx, (title, risk, desc, fn) in enumerate(steps, 1):
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
            except Exception as exc:  # 单项失败不中断后续项
                print(f"✗ 失败：{exc}")
                self.results.append((title, risk, "✗"))

    def cleanup(self) -> None:
        if not self.be.connected:
            return
        for r in self._pending:                                   # 兜底恢复固件参数
            try:
                self.be.write_param_arm(r["key"], r["value"])
                print(f"已兜底恢复 {r['key']} = {_fmt(r['value'])}")
            except Exception as exc:
                print(f"✗ 兜底恢复 {r['key']} 失败：{exc}")
        try:
            self.be.disable_arm()
            self.be.disable_end()
        except (RuntimeError, TimeoutError):
            pass
        self.be.disconnect()
        print("已清理：参数恢复 → 电机失能 → 断开连接")

    def report(self) -> None:
        n_ok = sum(1 for _, _, s in self.results if s.startswith("✓"))
        n_bad = sum(1 for _, _, s in self.results if s.startswith("✗"))
        n_skip = sum(1 for _, _, s in self.results if s == "跳过")
        total = len(self._steps())
        print("\n" + "═" * 56)
        print(f"测试汇总：✓ {n_ok} 成功 | ✗ {n_bad} 失败 | 跳过 {n_skip}（共 {total} 项）")
        for title, risk, status in self.results:
            print(f"  [{risk}] {title:<28}{status}")


def main() -> None:
    if "--list" in sys.argv:
        t = BackendDMFullTest()
        steps = t._steps()
        for idx, (title, risk, desc, _) in enumerate(steps, 1):
            print(f"[{idx:>2}/{len(steps)}] {risk:<2} {title} —— {desc}")
        return
    if not sys.stdin.isatty():
        print("需交互终端运行（防误触发真机）；查看步骤请用 --list")
        sys.exit(1)
    print(_BANNER)
    if input("确认工作空间已清空、急停可用、有人监护；输入 yes 开始: ").strip().lower() != "yes":
        print("已取消")
        return
    t = BackendDMFullTest()
    try:
        t.run()
    except (KeyboardInterrupt, EOFError):
        print("\n中断")
    finally:
        t.cleanup()
        t.report()


if __name__ == "__main__":
    main()
