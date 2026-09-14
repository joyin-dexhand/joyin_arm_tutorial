"""JoyArm 完全功能真机测试：公开接口全覆盖，按风险由低到高逐层执行。

【功能概要】
九层递进（29 项，顺序即风险递增序），每项执行前打印提示（编号/风险/说明）
并等待确认，高风险项须输入 yes。覆盖 JoyArm（单类）当前全部公开接口——
robotics 各域仅 ABC 接口（算法为教程 Ch2~6 教学内容，实现注册后随章补回
FK/IK/规划/回放类测试）：

  L0  离线计算层：工厂与软失败、配置 API（margin 置 0）、六域空表与门面守卫、
      采样限位、配置自检
  L1  连接只读层：connect、get_arm_state/state、get_end_state、read_mode、
      check_hardware、安全基准位形 q_base
  L2  使能层：模式轮切与混合语义、MIT 阻尼安全使能、紧急阻尼 damping_mode、
      失能守卫
  L3  参数层：读（版本/快照）、写（原值回写/标量广播/单关节/只读拒绝/
      max_spd 调小恢复/persist 存闪存）
  L4  末端层：位置往返、open/close/zero、越行程裁剪、力度控制、末端标零（自动恢复）
  L5  本体位置层：保持基准位形、逐关节小幅往返（含 set_arm_command 单关节）
  L6  MIT 层：显式小增益步进
  L7  速度层：零速使能、短时小速
  L8  恢复层：本体标零（自动恢复）、失能断开与断开后守卫

【安全设计】（参照 test_backend_dm_full.py）
- 构造前对 config 安全化：本体 POS_VEL.vlim 封顶 0.5 rad/s、MIT 回退增益封顶
  kp=20/kd=2；末端 vlim/dq_max 封顶 1.0 rad/s、tau_max 封顶 0.5 N·m；
  软限位 margin 置 0.0（本次测试要求，软限位=URDF 硬限位）。
- 运动幅度：本体单关节 ±0.05 rad（方向自动取远离限位边界一侧）；速度测试
  0.2 rad/s×0.6 s；末端 ±0.2 rad、力度 0.5 N。
- MIT 使能后立刻下发 kp=0/kd=1.5 阻尼保持（非零阻抗松软），减缓坠落。
- 高风险项（速度模式/persist/标零×2）须输入 yes；标零均指定 joint
  且含自动恢复流程。
- try/finally：任何退出路径先参数兜底恢复 → 失能全部电机 → 断开 → 汇总报告。

【安全提示】
上电使能前：清空机械臂工作空间、确认急停可用、扶稳大臂防坠落；
本脚本会真实驱动电机，请务必在有人监护下使用；力度测试时夹爪内勿放手指。

【环境与运行】
1. 新建环境并安装依赖：``cd joyarm_code && uv venv && uv sync && source .venv/bin/activate``
2. 接好达妙 CAN 桥（默认 ``/dev/ttyACM0``）与电机电源后运行：
   ``python test/test_joyarm_full.py``          # 交互逐项执行
   ``python test/test_joyarm_full.py --list``   # 仅列出测试项，不执行
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import joyarm_factory, JoyArm, clamp_to_limits  # noqa: E402
from joyarm_core.utils.types import ControlMode  # noqa: E402

# ---- 安全上限（构造前写进 config，全程生效）----
SOFT_MARGIN = 0.0         # 软限位内缩比例（本次测试要求置 0：软=硬）
SAFE_ARM_VLIM = 0.5       # 本体位置指令限速 rad/s（config POS_VEL.vlim 封顶）
SAFE_ARM_MIT_KP = 20.0    # 本体 MIT 回退增益封顶
SAFE_ARM_MIT_KD = 2.0
SAFE_END_VLIM = 1.0       # 末端限速 rad/s
SAFE_END_DQ = 1.0         # 末端速度上限封顶 rad/s
SAFE_END_TAU = 0.5        # 末端力矩上限封顶 N·m
# ---- 测试动作幅度（全程只做小幅运动）----
MOVE_AMP = 0.05           # 本体单关节往返幅度 rad
VEL_TEST = 0.2            # 速度模式测试速度 rad/s
VEL_DUR = 0.6             # 速度模式持续时间 s
MIT_KP_TEST = 5.0         # MIT 小阻抗保持增益
MIT_KD_TEST = 1.0
DAMP_KD = 1.5             # 使能后阻尼保持（kp=0，防坠落冲击）
END_AMP = 0.2             # 末端位置往返幅度 rad
END_TAU = 0.5             # 末端力矩测试 N·m

_BANNER = """
╔══════════════════════════════════════════════════════════╗
║    JoyArm 完全功能测试（真机，29 项，九层递进）            ║
╚══════════════════════════════════════════════════════════╝
⚠ 本脚本会真实驱动电机（含小幅运动与末端夹合）：
  - 清空机械臂工作空间，确认急停可用，全程有人监护；
  - 已自动收紧：限速 0.5 rad/s、末端力矩 0.5 N·m、MIT 增益 kp≤20/kd≤2；
    软限位 margin=0（软=URDF 硬限位）；
  - 高风险项（速度模式/存闪存/标零）须输入 yes 才执行。
"""


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _fmt(a, prec: int = 4) -> str:
    return np.array2string(np.asarray(a, dtype=float), precision=prec)


class JoyArmFullTest:
    """顺序执行器：持 JoyArm 实例与测试上下文（安全基准位形、末端行程、结果表）。"""

    def __init__(self) -> None:
        cfg = yaml.safe_load(
            (_ROOT / "joyarm_core" / "configs" / "joyarm_dm.yaml").read_text(encoding="utf-8")
        )
        self._safe_diff = self._safeguard(cfg)
        # 直用类构造注入安全化 config（model 须与 basic.name 一致以符合命名链）
        self.arm = JoyArm(model="joyarm_dm", config=cfg)
        self.arm_names = [j["name"] for j in cfg["backend"]["arm"]["joints"]]
        ej = cfg["backend"]["end"]["joints"][0]
        self.end_qmin = float(ej["q_min"])       # 末端行程（到位判断）
        self.end_qmax = float(ej["q_max"])
        self.q_base: np.ndarray | None = None    # 连接后的安全基准位形
        self._pending: list[dict] = []           # 异常退出兜底恢复的固件参数
        self.results: list[tuple[str, str, str]] = []

    # ----------------------------------------------------------
    # config 安全化（构造前）
    # ----------------------------------------------------------
    @staticmethod
    def _safeguard(cfg: dict) -> list[tuple[str, str, float, float]]:
        """真机安全化：直接收窄 backend.*.joints 四键硬限位（backend 下发只裁
        硬限位，故收紧硬限位即收紧全部指令守卫）+ 限速/限矩/增益封顶。"""
        diff: list[tuple[str, str, float, float]] = []
        for part in ("arm", "end"):
            for j in cfg["backend"].get(part, {}).get("joints", []):
                q_lo, q_hi = float(j["q_min"]), float(j["q_max"])
                new_lo, new_hi = q_lo + SOFT_MARGIN, q_hi - SOFT_MARGIN
                if new_lo < new_hi:                      # 防过收窄交叉
                    j["q_min"], j["q_max"] = new_lo, new_hi
                    diff.append((j["name"], "q_min", q_lo, new_lo))
                    diff.append((j["name"], "q_max", q_hi, new_hi))
        for j in cfg["backend"]["arm"]["joints"]:
            pv = j.setdefault("POS_VEL", {})
            old, pv["vlim"] = float(pv.get("vlim", 0.0)), min(float(pv.get("vlim", SAFE_ARM_VLIM)), SAFE_ARM_VLIM)
            diff.append((j["name"], "POS_VEL.vlim", old, pv["vlim"]))
            mit = j.setdefault("MIT", {})
            for key, cap in (("kp", SAFE_ARM_MIT_KP), ("kd", SAFE_ARM_MIT_KD)):
                old, mit[key] = float(mit.get(key, 0.0)), min(float(mit.get(key, cap)), cap)
                diff.append((j["name"], f"MIT.{key}", old, mit[key]))
        for j in cfg["backend"]["end"]["joints"]:
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

    def _q_now(self) -> np.ndarray:
        return np.asarray(self.arm.get_arm_state().joint.q, dtype=float)

    def _assert_healthy(self) -> None:
        js = self.arm.get_arm_state().joint
        assert bool(np.all(js.comm_ok)), "通讯异常"
        assert not bool(np.any(js.error)), "电机故障标志置位"

    def _print_arm_state(self) -> None:
        s = self.arm.get_arm_state()
        js = s.joint
        print(f"{'关节':<9}{'q(rad)':>10}{'dq(rad/s)':>10}{'tau(N·m)':>9}{'tMOS':>5}{'tRot':>5}  使能 故障 通讯")
        t2 = lambda t: "—" if t == 0 else f"{t:.0f}"
        for nm, q, dq, tau, tm, tr, en, fl, ok in zip(
                self.arm_names, js.q, js.dq, js.tau, js.temp_mos, js.temp_rotor,
                js.enabled, js.error, js.comm_ok):
            print(f"{nm:<10}{q:+10.4f}{dq:+10.4f}{tau:+9.3f}{t2(tm):>5}{t2(tr):>5}"
                  f"   {'是' if en else '否'}   {'是' if fl else '否'}   {'是' if ok else '否'}")
        tcp_pos = (_fmt(s.tcp.pose.position, 3) if s.tcp.pose is not None
                   else "（fkine 未注册，跳过）")
        print(f"TCP 位置：{tcp_pos} | 故障：{'；'.join(s.errors) if s.errors else '无'}")

    def _print_end_state(self) -> None:
        e = self.arm.get_end_state()
        t2 = lambda t: "—" if t == 0 else f"{t:.0f}"
        print(f"{'末端':<9}{'q(rad)':>10}{'dq(rad/s)':>10}{'tau(N·m)':>9}{'tMOS':>5}{'tRot':>5}  使能 故障 通讯")
        for q, dq, tau, tm, tr, en, fl, ok in zip(
                e["q"], e["dq"], e["tau"], e["temp_mos"], e["temp_rotor"],
                e["enabled"], e["error"], e["comm_ok"]):
            print(f"{'gripper':<10}{q:+10.4f}{dq:+10.4f}{tau:+9.3f}{t2(tm):>5}{t2(tr):>5}"
                  f"   {'是' if en else '否'}   {'是' if fl else '否'}   {'是' if ok else '否'}")

    def _wait_arm_q(self, joint: int, target: float, tol: float = 0.03,
                    timeout: float = 4.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if abs(float(self._q_now()[joint]) - target) <= tol:
                return True
            time.sleep(0.08)
        return False

    def _wait_end_q(self, target: float, tol: float = 0.15, timeout: float = 10.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if abs(float(self.arm.get_end_state()["q"][0]) - target) <= tol:
                return True
            time.sleep(0.08)
        return False

    def _safe_dir(self, i: int) -> int:
        """第 i 关节的安全运动方向：朝限位区间较宽一侧；**该侧余量不足
        MOVE_AMP 时换向**（q 贴边时"宽侧"可能是零余量方向）。"""
        q, lo, hi = (self.q_base[i], self.arm.arm_limits.q_min[i],
                        self.arm.arm_limits.q_max[i])
        d = 1 if (q - lo) >= (hi - q) else -1
        if d > 0 and (hi - q) < MOVE_AMP:
            d = -1
        elif d < 0 and (q - lo) < MOVE_AMP:
            d = 1
        return d

    @staticmethod
    def _expect(exc_type, call, what: str) -> None:
        try:
            call()
            raise AssertionError(f"{what} 应抛 {exc_type.__name__}")
        except exc_type:
            print(f"  ✓ {what} → {exc_type.__name__}")

    # ----------------------------------------------------------
    # 测试步骤（顺序即执行序，九层风险递增；29 项）
    # ----------------------------------------------------------
    def _steps(self) -> list[tuple[str, str, str, object]]:
        return [
            # ---- L0 离线计算层（无连接、无运动）----
            ("L0-构造与安全化对照", "低", "打印硬限位收窄/限速/限矩/增益收紧对照表；repr 与教学数据读取", self.step_construct),
            ("L0-工厂与软失败", "低", "joyarm_factory 正常创建 + 未知型号返回 None（离线第二实例，不连接）", self.step_factory),
            ("L0-配置 API", "低", "get_config 深拷贝快照（set_config 已随 margin 机制移除）", self.step_config_api),
            ("L0-六域空表与门面守卫", "低", "各域默认空表（教学过渡态）、未加载域门面 RuntimeError", self.step_domains),
            ("L0-采样与限位", "低", "rand_q_arm + utils clamp_to_limits 裁剪；硬限位内采样（config 四键）", self.step_sampling),
            ("L0-配置自检", "低", "check_config 正常静默通过（异常则 ValueError）", self.step_check_config),
            # ---- L1 连接只读层（不使能）----
            ("L1-connect", "低", "建立通信（共享总线，电机保持失能）；离线守卫抽查", self.step_connect),
            ("L1-读状态与模式", "低", "get_arm_state / read_mode_arm·end（失能态）", self.step_read_state),
            ("L1-末端状态", "低", "get_end_state 字段族（joint=None/0）", self.step_end_state),
            ("L1-硬件自检", "低", "check_hardware 最小自检（通讯/故障/编码器；正常静默通过，异常 RuntimeError）", self.step_check_hw),
            ("L1-安全基准位形", "低", "q_base = clamp_to_limits(当前 q)（joint2/3 上限 0，越界部分将被夹回）", self.step_qbase),
            # ---- L2 使能层 ----
            ("L2-模式轮切与混合语义", "中", "失能态轮切 MIT/POSITION/VELOCITY + read_mode 混合语义验证（无运动）", self.step_modes),
            ("L2-安全使能（MIT 阻尼）", "中", "使能后立刻下发 kp=0/kd=1.5 阻尼指令；⚠ 请扶稳大臂防下垂", self.step_safe_enable),
            ("L2-紧急阻尼 damping_mode", "中", "任何状态一键全电机（含末端）MIT 纯阻尼（kp=q=dq=tau=0, kd=10）；可手搬体验黏滞", self.step_damping),
            ("L2-失能与守卫", "低", "失能验证 enabled=False + MIT 维度校验 + 参数缺失守卫", self.step_disable_checks),
            # ---- L3 参数层 ----
            ("L3-读参数", "低", "read_param_arm/end：版本身份 + 可写参数快照", self.step_param_read),
            ("L3-写参数（原值回写）", "低", "acc 逐电机回写、标量广播、joint=0 单关节、只读拒绝", self.step_param_write),
            ("L3-max_spd 调小→恢复", "中", "固件限速临时调小（≤2 rad/s）读回确认，随后恢复", self.step_param_maxspd),
            ("L3-persist 存闪存", "高", "acc 原值写入并存闪存（自动失能）；掉电不丢", self.step_persist),
            # ---- L4 末端层 ----
            ("L4-末端位置往返", "中", "POSITION 使能+保持+±0.2 rad 往返（标量/序列两种形式）", self.step_end_pos),
            ("L4-末端动作与裁剪", "中", "set_end_open/close/zero + 越行程裁剪（99 rad→q_max）+ 非法动作拒绝", self.step_end_actions),
            ("L4-末端力矩控制", "中", "set_end_tau 0.5 N·m（MIT 近似闭合，前馈封顶 0.5 N·m）；⚠ 勿放手指", self.step_end_tau),
            ("L4-末端标零（自动恢复）", "高", "set_zero_end → 回原零位 → 恢复标零 → 回原位置 → 验证", self.step_set_zero_end),
            # ---- L5 本体位置层 ----
            ("L5-位置保持", "中", "POSITION 使能并保持 q_base（首次指令会把 joint2/3 夹回限位内）", self.step_pos_hold),
            ("L5-逐关节小幅往返", "中", "joint1~6 依次 ±0.05 rad（安全方向）→ 回基准（含单关节 joint= API）", self.step_pos_sweep),
            ("L5-move_j 小行程", "中", "三次多项式小行程往返（q_base±0.05 安全方向）+ is_in_position 到位判断", self.step_move_j),
            # ---- L6 MIT 层 ----
            ("L6-MIT 小增益步进", "中", "joint1 MIT kp=5/kd=1 步进 +0.05 rad → 回基准（电机侧 PD）", self.step_mit_step),
            # ---- L7 速度层 ----
            ("L7-速度模式（短时小速）", "高", "VELOCITY 使能+零速 → joint1 0.2 rad/s×0.6 s → 归零；⚠ 周围无障碍", self.step_velocity),
            # ---- L8 恢复层 ----
            ("L8-本体标零（自动恢复）", "高", "选 |q| 最小关节：标零→回原零位→恢复标零→回原位置→验证", self.step_set_zero_arm),
            ("L8-急停锁定与原位保持", "中", "lock_position 位置锁定 + hold_position 阻抗保持（tau 前馈退化 0）；⚠ 请扶稳", self.step_lock_hold),
            ("L8-安全回零族", "高", "safe_home → home_to_zero（safe_zero 组合；⚠ 大范围运动，请清场扶稳）", self.step_safe_family),
            ("L8-故障清除（正常态）", "低", "clear_fault_arm/end 验证式复位（无故障应静默通过）", self.step_clear_fault),
            ("L8-失能断开与守卫", "低", "回 q_base → 失能全部 → disconnect → 断开后守卫", self.step_teardown),
        ]

    # ================= L0 离线计算层 =================
    def step_construct(self) -> None:
        def _fv(v):
            if isinstance(v, dict):
                return str({k: _fv(x) for k, x in v.items()})
            return _fmt(v, 1) if isinstance(v, list) else f"{v:.3f}"
        print("安全化对照（原值 → 测试值）：")
        for name, key, old, new in self._safe_diff:
            mark = "" if _fv(old) == _fv(new) else "  ← 已收紧"
            print(f"  {name:<10}{key:<16}{_fv(old):>18} → {_fv(new):<18}{mark}")
        arm = self.arm
        print(f"✓ JoyArm 已构造（未连接）：{arm!r}")
        mdh = arm.get_config()["joyarm"]["arm_mdh_and_limits"]   # 类内 config 读取教学数据
        print(f"✓ 教学数据（config 内读取）：arm_mdh_and_limits {len(mdh)}×{len(mdh[0])}"
              f"（MDH 白盒链路，Ch2 教学算法使用）")
        print(f"✓ 硬限位收窄 {SOFT_MARGIN}："
              f"q_min={_fmt(arm.arm_limits.q_min, 2)} "
              f"q_max={_fmt(arm.arm_limits.q_max, 2)}（backend 下发只裁硬限位）")
        print(f"✓ 软限位（直配仅加载，上层状态判断用）："
              f"q_max={_fmt(arm.arm_limits.q_max, 2)}")
        self._expect(RuntimeError, lambda: arm.damping_mode(), "未连接调用 damping_mode")

    def step_factory(self) -> None:
        a = joyarm_factory("joyarm_dm")
        assert a is not None and a.n_arm == 6
        assert joyarm_factory("joyarm_nx") is None        # 软失败：None + 日志
        print("✓ 工厂：joyarm_dm 创建成功；未知型号返回 None（失败信息见日志）")

    def step_config_api(self) -> None:
        arm = self.arm
        cfg = arm.get_config()
        cfg["basic"]["name"] = "hacked"
        assert arm.get_config()["basic"]["name"] == "joyarm_dm"   # 深拷贝
        print("✓ 配置 API：深拷贝快照（set_config 已随 margin 机制移除）")

    def step_domains(self) -> None:
        arm = self.arm
        for d in ("fkine", "ikine", "jacobian", "dynamics", "traj", "control"):
            assert arm.list_solvers(d) == [], f"{d} 应为空（教学过渡态）"
        self._expect(RuntimeError,
                     lambda: arm.fkine(arm.arm_zero, "ee"), "调用未加载域 fkine 门面")
        print("✓ 六域默认空表；未加载域门面显性 RuntimeError（各章实现注册后接入）")

    def step_sampling(self) -> None:
        arm = self.arm
        rng = np.random.default_rng(0)
        q = arm.rand_q_arm(rng=rng)
        clipped = clamp_to_limits(q + 10.0, arm.arm_limits)
        assert np.allclose(q, clamp_to_limits(q, arm.arm_limits))  # 采样在限位内
        assert not np.allclose(q + 10.0, clipped)                        # 越限被裁
        print(f"✓ 采样/裁剪：rand_q_arm {q.round(2)}（硬限位内，经 utils.clamp_to_limits）")

    def step_check_config(self) -> None:
        JoyArm.check_config(self.arm.model, self.arm.get_config())    # 正常 → 静默通过；异常 → ValueError 列出全部问题
        print("✓ check_config：通过")

    # ================= L1 连接只读层 =================
    def step_connect(self) -> None:
        self.arm.connect()
        assert self.arm.connected is True
        print("✓ 已连接（arm/end 共享总线；电机保持失能）")

    def step_read_state(self) -> None:
        self._print_arm_state()
        st = self.arm.get_arm_state()
        assert st is not None and st.joint.q.size == 6
        m_arm, m_end = self.arm.read_mode_arm(), self.arm.read_mode_end()
        assert m_arm is None and m_end is None
        print("✓ state property 现读 ✓；read_mode_arm/end 失能未设置 → None")

    def step_end_state(self) -> None:
        self._print_end_state()
        e0 = self.arm.get_end_state(0)
        assert float(e0["q"][0]) == float(self.arm.get_end_state()["q"][0])
        print("✓ get_end_state joint=None/0 一致")

    def step_check_hw(self) -> None:
        self.arm.check_hardware()  # 正常 → 静默通过；异常 → RuntimeError 列出全部问题
        print("✓ check_hardware：通过（通讯/故障/编码器；温度等运行期监控归 ROS2 节点）")

    def step_qbase(self) -> None:
        q = self._q_now()
        self.q_base = clamp_to_limits(q, self.arm.arm_limits)
        print(f"  当前 q：{_fmt(q)}")
        print(f"  q_base（夹紧后，后续小幅运动基准）：{_fmt(self.q_base)}")
        assert np.all(self.q_base >= self.arm.arm_limits.q_min - 1e-9)
        assert np.all(self.q_base <= self.arm.arm_limits.q_max + 1e-9)
        print("✓ 安全基准位形已建立")

    # ================= L2 使能层 =================
    def step_modes(self) -> None:
        arm = self.arm
        for m in (ControlMode.MIT, ControlMode.POSITION, ControlMode.VELOCITY):
            arm.set_mode_arm(m)
            assert arm.read_mode_arm() is m
        arm.set_mode_arm(ControlMode.POSITION, joint=0)     # 混合语义
        assert arm.read_mode_arm() is None
        assert arm.read_mode_arm(0) is ControlMode.POSITION
        assert arm.read_mode_arm(1) is ControlMode.VELOCITY
        arm.set_mode_arm(ControlMode.MIT)                   # 恢复整臂一致
        assert arm.read_mode_arm() is ControlMode.MIT
        print("✓ 三模式轮切 + read_mode 混合语义（整臂 None、子集各自正确）")

    def step_safe_enable(self) -> None:
        q = self._q_now()
        self.arm.set_mode_arm(ControlMode.MIT)
        self.arm.enable_arm()
        self.arm.set_arm_command(ControlMode.MIT, q=q, dq=np.zeros(6), tau=np.zeros(6),
                                 kp=np.zeros(6), kd=np.full(6, DAMP_KD))
        self._assert_healthy()
        js = self.arm.get_arm_state().joint
        assert bool(np.all(js.enabled)), "使能后 enabled 应全为 True"
        print(f"✓ 已使能并下发阻尼保持（kp=0, kd={DAMP_KD}）：目标 q={_fmt(q)}")
        print("  电机近似黏滞阻尼，可缓慢手动搬动；请扶稳大臂")

    def step_damping(self) -> None:
        self.arm.damping_mode()                     # 紧急阻尼（kd=10）
        assert self.arm.read_mode_arm() is ControlMode.MIT
        js = self.arm.get_arm_state().joint
        assert bool(np.all(js.enabled)), "阻尼模式应保持使能"
        print(f"✓ damping_mode：全电机（本体+末端）MIT 纯阻尼 kd=10，模式/使能已验证")
        print("  此时可缓慢手动搬动机械臂感受黏滞阻力（仅速度阻尼，无位置保持）")

    def step_disable_checks(self) -> None:
        self.arm.disable_arm()
        js = self.arm.get_arm_state().joint
        assert not bool(np.any(js.enabled)), "失能后 enabled 应全为 False"
        q = js.q
        self._expect(ValueError,
                     lambda: self.arm.set_arm_command(ControlMode.MIT, q=q[:3], dq=q[:3], tau=q[:3]),
                     "MIT 指令维度不匹配（3≠6）")
        self._expect(ValueError,
                     lambda: self.arm.set_arm_command(ControlMode.POSITION),
                     "POSITION 缺 q")
        print("✓ 已失能（enabled 全 False）+ 维度/参数缺失守卫")

    # ================= L3 参数层 =================
    def step_param_read(self) -> None:
        arm = self.arm
        for key in ("hw_ver", "sw_ver", "sn", "kt", "gr"):
            print(f"  {key:<8}: 本体 {arm.read_param_arm(key)}")
        print(f"  acc    : 本体 {_fmt(arm.read_param_arm('acc'))} / 末端 {arm.read_param_end('acc')}")
        print("✓ 参数读取（版本身份 + 快照）")

    def step_param_write(self) -> None:
        arm = self.arm
        orig = arm.read_param_arm("acc")
        arm.write_param_arm("acc", list(orig))              # 原值回写（写确认内建）
        back = arm.read_param_arm("acc")
        assert all(abs(a - b) <= max(1e-4, 1e-3 * abs(b)) for a, b in zip(back, orig))
        scalar = float(orig[0])
        arm.write_param_arm("acc", scalar)                  # 标量广播
        assert all(abs(g - scalar) <= max(1e-4, 1e-3 * abs(scalar))
                   for g in arm.read_param_arm("acc"))
        arm.write_param_arm("acc", list(orig))              # 恢复
        arm.write_param_arm("acc", scalar, joint=0)         # 单关节
        assert abs(arm.read_param_arm("acc", 0) - scalar) <= max(1e-4, 1e-3 * abs(scalar))
        arm.write_param_arm("acc", list(orig))              # 恢复
        self._expect(ValueError, lambda: arm.write_param_arm("sw_ver", 1),
                     "写入只读参数 sw_ver")
        print("✓ 参数写入：原值回写/标量广播/单关节/只读拒绝（均已恢复原值）")

    def step_param_maxspd(self) -> None:
        arm = self.arm
        orig = arm.read_param_arm("max_spd")
        self._pending.append({"key": "max_spd", "value": list(orig)})
        new = [max(0.5, min(v * 0.5, 2.0)) for v in orig]
        arm.write_param_arm("max_spd", new)
        got = arm.read_param_arm("max_spd")
        assert all(abs(g - n) <= max(1e-4, 1e-3 * abs(n)) for g, n in zip(got, new))
        arm.write_param_arm("max_spd", list(orig))
        self._pending.pop()
        print(f"✓ max_spd 调小 {_fmt(new)} → 已恢复 {_fmt(arm.read_param_arm('max_spd'))}")

    def step_persist(self) -> None:
        arm = self.arm
        orig = arm.read_param_arm("acc")
        arm.write_param_arm("acc", list(orig), persist=True)   # 自动失能并保持
        back = arm.read_param_arm("acc")
        assert all(abs(a - b) <= max(1e-4, 1e-3 * abs(b)) for a, b in zip(back, orig))
        print("✓ acc 已存闪存（电机自动失能），掉电保持")

    # ================= L4 末端层 =================
    def step_end_pos(self) -> None:
        arm = self.arm
        arm.set_mode_end(ControlMode.POSITION)
        assert arm.read_mode_end() is ControlMode.POSITION
        q0 = float(arm.get_end_state()["q"][0])
        arm.enable_end()
        arm.set_end_position(q0)                            # 保持
        t1 = _clamp(q0 + END_AMP, self.end_qmin, self.end_qmax)
        arm.set_end_position(t1)                            # 标量形式
        ok1 = self._wait_end_q(t1)
        t2 = _clamp(q0 - END_AMP, self.end_qmin, self.end_qmax)
        arm.set_end_position([t2])                          # 序列形式
        ok2 = self._wait_end_q(t2)
        arm.set_end_position(q0)
        ok3 = self._wait_end_q(q0)
        print(f"✓ 末端位置往返：+{END_AMP} {'✓' if ok1 else '未到位'} / "
              f"−{END_AMP} {'✓' if ok2 else '未到位'} / 回原位 {'✓' if ok3 else '未到位'}")

    def step_end_actions(self) -> None:
        arm = self.arm
        for action, target in (("open", self.end_qmin), ("close", self.end_qmax),
                               ("zero", _clamp(0.0, self.end_qmin, self.end_qmax))):
            getattr(arm, f"set_end_{action}")()
            ok = self._wait_end_q(target)
            print(f"  set_end_{action:<4}→ {target:+.4f} rad：{'✓ 到位' if ok else '未到位'}")
        arm.set_end_position(99.0)                          # 越行程 → 裁剪
        self._wait_end_q(self.end_qmax)
        q = float(arm.get_end_state()["q"][0])
        assert q <= self.end_qmax + 0.2
        # end_* 门面族动作固定，非法动作经 backend 层验证
        self._expect(ValueError, lambda: arm._backend.send_action_end("bad"),
                     "非法末端动作拒绝")
        print(f"✓ 越行程裁剪：99 rad → 实际 {q:+.4f}（q_max={self.end_qmax}）")

    def step_end_tau(self) -> None:
        arm = self.arm
        arm.set_mode_end(ControlMode.MIT)
        assert arm.read_mode_end() is ControlMode.MIT
        arm.enable_end()
        arm.set_end_tau(END_TAU)
        self._wait_end_q(self.end_qmax, tol=0.2, timeout=6.0)
        e = arm.get_end_state()
        arm.disable_end()
        print(f"✓ 力矩 {END_TAU} N·m：闭合 tau={e['tau'][0]:+.3f} N·m（封顶 {SAFE_END_TAU}），已失能")

    def step_set_zero_end(self) -> None:
        arm = self.arm
        q0 = float(arm.get_end_state()["q"][0])
        # 前置保护：回原零位目标 -q0 须在行程内，否则先移到行程中点再走流程
        if not (self.end_qmin <= -q0 <= self.end_qmax):
            mid = 0.5 * (self.end_qmin + self.end_qmax)
            print(f"  -q0={-q0:+.4f} 越行程（[{self.end_qmin}, {self.end_qmax}]），先移到中点 {mid:+.4f}")
            arm.set_mode_end(ControlMode.POSITION)
            arm.enable_end()
            arm.set_end_position(mid)
            self._wait_end_q(mid)
            arm.disable_end()
        q0 = float(arm.get_end_state()["q"][0])
        print(f"  末端当前读数 {q0:+.4f} rad（自动恢复流程）")
        arm.set_zero_end()                                  # 新零位 = 当前位置
        arm.set_mode_end(ControlMode.POSITION)
        arm.enable_end()
        arm.set_end_position(-q0)                           # 物理回原零位
        if not self._wait_end_q(-q0, tol=0.05, timeout=12.0):
            print("  ⚠ 回原零位超时，继续恢复")
        arm.disable_end()
        arm.set_zero_end()                                  # 零位还原
        arm.enable_end()
        arm.set_end_position(q0)                            # 回原位置
        self._wait_end_q(q0, tol=0.05, timeout=12.0)
        arm.disable_end()
        q_now = float(arm.get_end_state()["q"][0])
        assert abs(q_now - q0) < 0.05, f"零位恢复失败：{q_now:+.4f} vs {q0:+.4f}"
        print(f"✓ 末端标零→恢复完成（当前 {q_now:+.4f} ≈ 原 {q0:+.4f}，零位已还原）")

    # ================= L5 本体位置层 =================
    def step_pos_hold(self) -> None:
        self.arm.set_mode_arm(ControlMode.POSITION)
        self.arm.enable_arm()
        self.arm.set_arm_command(ControlMode.POSITION, q=self.q_base)
        time.sleep(1.0)
        err = np.abs(self._q_now() - self.q_base).max()
        print(f"✓ 位置保持 q_base（joint2/3 越界部分已夹回，限速 {SAFE_ARM_VLIM}）：偏差 {err:.4f} rad")

    def step_pos_sweep(self) -> None:
        worst = 0.0
        for j, nm in enumerate(self.arm_names):
            target = float(np.clip(self.q_base[j] + self._safe_dir(j) * MOVE_AMP,
                                   self.arm.arm_limits.q_min[j],
                                   self.arm.arm_limits.q_max[j]))
            self.arm.set_arm_command(ControlMode.POSITION, q=[target], joint=j)
            ok1 = self._wait_arm_q(j, target)
            self.arm.set_arm_command(ControlMode.POSITION, q=[float(self.q_base[j])], joint=j)
            ok2 = self._wait_arm_q(j, float(self.q_base[j]))
            err = abs(float(self._q_now()[j]) - float(self.q_base[j]))
            worst = max(worst, err)
            print(f"  {nm:<9}{self._safe_dir(j) * MOVE_AMP:+.2f} rad 往返："
                  f"{'✓' if ok1 and ok2 else '未到位'}（回位误差 {err:.4f} rad）")
        print(f"✓ 逐关节小幅往返完成（单关节 joint= API；最大回位误差 {worst:.4f} rad）")

    # ================= L6 MIT 层 =================
    def step_mit_step(self) -> None:
        self.arm.set_mode_arm(ControlMode.MIT)
        target = float(self.q_base[0]) + self._safe_dir(0) * MOVE_AMP
        self.arm.set_arm_command(ControlMode.MIT, q=[target], dq=[0.0], tau=[0.0],
                                 kp=[MIT_KP_TEST], kd=[MIT_KD_TEST], joint=0)
        ok = self._wait_arm_q(0, target, tol=0.05)
        self.arm.set_arm_command(ControlMode.MIT, q=[float(self.q_base[0])], dq=[0.0],
                                 tau=[0.0], kp=[MIT_KP_TEST], kd=[MIT_KD_TEST], joint=0)
        self._wait_arm_q(0, float(self.q_base[0]), tol=0.05)
        self.arm.set_mode_arm(ControlMode.POSITION)
        self.arm.set_arm_command(ControlMode.POSITION, q=self.q_base)
        time.sleep(0.5)
        print(f"✓ joint1 MIT 步进（kp={MIT_KP_TEST}/kd={MIT_KD_TEST}）：{'到位' if ok else '未到位'}，已回基准")

    # ================= L7 速度层 =================
    def step_velocity(self) -> None:
        arm = self.arm
        arm.set_mode_arm(ControlMode.VELOCITY)
        assert arm.read_mode_arm() is ControlMode.VELOCITY
        arm.enable_arm()
        arm.set_arm_command(ControlMode.VELOCITY, dq=np.zeros(6))     # 先零速
        q0 = float(self._q_now()[0])
        arm.set_arm_command(ControlMode.VELOCITY, dq=[VEL_TEST], joint=0)
        time.sleep(VEL_DUR)
        arm.set_arm_command(ControlMode.VELOCITY, dq=[0.0], joint=0)
        q1 = float(self._q_now()[0])
        arm.set_mode_arm(ControlMode.POSITION)
        arm.set_arm_command(ControlMode.POSITION, q=self.q_base)
        time.sleep(1.5)
        print(f"✓ joint1 {VEL_TEST} rad/s × {VEL_DUR} s：q {q0:+.4f} → {q1:+.4f}"
              f"（Δ={q1 - q0:+.4f}），已回 POSITION 并回基准")

    # ================= L8 恢复层 =================
    def step_set_zero_arm(self) -> None:
        arm = self.arm
        qs = self._q_now()
        j = int(np.argmin(np.abs(qs)))
        q0 = float(qs[j])
        print(f"选定 {self.arm_names[j]}（|q| 最小）：当前 {q0:+.4f} rad（自动恢复流程）")
        arm.set_zero_arm(joint=j)                           # 新零位 = 当前物理位置
        arm.set_mode_arm(ControlMode.POSITION)
        arm.enable_arm()
        hold = qs.copy()
        hold[j] = -q0
        arm.set_arm_command(ControlMode.POSITION, q=hold)   # 物理回原零位（其余保持）
        self._wait_arm_q(j, -q0, tol=0.05, timeout=6.0)
        arm.disable_arm()
        arm.set_zero_arm(joint=j)                           # 零位还原
        arm.enable_arm()
        arm.set_arm_command(ControlMode.POSITION, q=qs)     # 回原位置
        self._wait_arm_q(j, q0, tol=0.05, timeout=6.0)
        arm.disable_arm()
        q_now = float(self._q_now()[j])
        assert abs(q_now - q0) < 0.05, f"零位恢复失败：{q_now:+.4f} vs {q0:+.4f}"
        print(f"✓ 本体标零→恢复完成（当前 {q_now:+.4f} ≈ 原 {q0:+.4f}，零位已还原）")

    def step_move_j(self) -> None:
        """L5：move_j 三次多项式小行程（与规划器并行的独立功能）+ 到位判断。"""
        arm = self.arm
        j = 0
        target = float(np.clip(self.q_base[j] + self._safe_dir(j) * 0.05,
                               arm.arm_limits.q_min[j],
                               arm.arm_limits.q_max[j]))
        q_t = np.asarray(self.q_base, dtype=float).copy()
        q_t[j] = target
        arm.move_j(q_t, t=0.6, wait_timeout=5.0)
        assert arm.is_in_position(q=q_t)
        arm.move_j(np.asarray(self.q_base, dtype=float), t=0.6, wait_timeout=5.0)
        assert arm.is_in_position(q=self.q_base)
        print("✓ move_j：小行程三次多项式往返 + is_in_position 通过")

    def step_lock_hold(self) -> None:
        """L8：急停锁定 + 原位保持。"""
        self.arm.lock_position()
        self.arm.hold_position()
        print("✓ lock_position / hold_position：位置锁定 + 阻抗保持（tau 前馈退化 0）")

    def step_safe_family(self) -> None:
        """L8：safe_home → home_to_zero（safe_zero 组合的大范围运动，需清场）。"""
        arm = self.arm
        arm.safe_home(t=3.0, wait_timeout=10.0)
        assert arm.is_in_position(q=arm.arm_home)
        arm.home_to_zero(t=3.0, wait_timeout=10.0)
        assert arm.is_in_position(q=arm.arm_zero)
        print("✓ safe_home / home_to_zero：安全回零族通过（当前位于 zero）")

    def step_clear_fault(self) -> None:
        """L8：clear_fault 验证式复位（正常态应静默通过）。"""
        self.arm.clear_fault_arm()
        self.arm.clear_fault_end()
        print("✓ clear_fault_arm/end：正常态静默通过（失能→核对→使能→核对）")

    def step_teardown(self) -> None:
        arm = self.arm
        arm.set_mode_arm(ControlMode.POSITION)
        arm.enable_arm()
        arm.set_arm_command(ControlMode.POSITION, q=self.q_base)   # 回基准
        time.sleep(1.0)
        arm.disable_arm()
        arm.disable_end()
        arm.disconnect()
        assert arm.connected is False
        self._expect(RuntimeError, lambda: arm.get_arm_state(), "断开后 get_arm_state")
        print("✓ 已回基准 → 失能全部 → 断开（断开后守卫生效）")

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
        if not self.arm.connected:
            return
        for r in self._pending:                                   # 兜底恢复固件参数
            try:
                self.arm.write_param_arm(r["key"], r["value"])
                print(f"已兜底恢复 {r['key']} = {_fmt(r['value'])}")
            except Exception as exc:
                print(f"✗ 兜底恢复 {r['key']} 失败：{exc}")
        try:
            self.arm.disable_arm()
            self.arm.disable_end()
        except (RuntimeError, TimeoutError):
            pass
        self.arm.disconnect()
        print("已清理：参数恢复 → 电机失能 → 断开连接")

    def report(self) -> None:
        n_ok = sum(1 for _, _, s in self.results if s.startswith("✓"))
        n_bad = sum(1 for _, _, s in self.results if s.startswith("✗"))
        n_skip = sum(1 for _, _, s in self.results if s == "跳过")
        total = len(self._steps())
        print("\n" + "═" * 56)
        print(f"测试汇总：✓ {n_ok} 成功 | ✗ {n_bad} 失败 | 跳过 {n_skip}（共 {total} 项）")
        for title, risk, status in self.results:
            print(f"  [{risk}] {title:<32}{status}")


def main() -> None:
    if "--list" in sys.argv:
        t = JoyArmFullTest()
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
    t = JoyArmFullTest()
    try:
        t.run()
    except (KeyboardInterrupt, EOFError):
        print("\n中断")
    finally:
        t.cleanup()
        t.report()


if __name__ == "__main__":
    main()
