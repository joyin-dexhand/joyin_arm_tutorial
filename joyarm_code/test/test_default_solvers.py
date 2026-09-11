"""六个默认求解器离线测试（joyarm_dm 真实 pin 模型，不依赖硬件）。

覆盖：①注册冒烟（六注册名）；②PinFkine vs pin 直算（单点/批量/T 表示）；
③PinJacobian 双参考系 vs pin 直算 + 微分逆解往返；④PinIkine 收敛精度 /
不可达失败 / 硬限位钳位；⑤PinDynamics M 对称正定 / G / C / Λ 对称；
⑥ToJoint 规划器六端点边界条件 / 中段连续 / 末端钳位 / 目标缺省 0；⑦关节位置控制器
小步直通 / 大步裁剪 + 节流告警 / MODE；⑧config 端到端（robotics 段选配六默认
→ JoyArm 构造 → 离线门面可用）+ 运动管线冒烟（帧含 q/dq/ddq）。

运行：``python test/test_default_solvers.py`` 或 pytest。
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import (  # noqa: E402
    joyarm_factory, JoyArm, Pose, ControlMode, ArmState, JointState,
    TrajFrame, clamp_to_limits, R_to_quat,
    PinFkineSolver, PinIkineSolver, PinJacobianSolver, PinDynamicsSolver,
    ToJointTrajPlanner, JointPositionController,
)
from joyarm_core.robotics.fkine import REGISTRY as _RF
from joyarm_core.robotics.ikine import REGISTRY as _RI
from joyarm_core.robotics.jacobian import REGISTRY as _RJ
from joyarm_core.robotics.dynamics import REGISTRY as _RD
from joyarm_core.robotics.trajectory import REGISTRY as _RT
from joyarm_core.robotics.control import REGISTRY as _RC
from joyarm_core.joyarm import load_config  # noqa: E402

_ARM = None


def _arm():
    global _ARM
    if _ARM is None:
        _ARM = joyarm_factory("joyarm_dm")
        assert _ARM is not None, "joyarm_dm 创建失败"
    return _ARM


def test_registry_defaults():
    """六域默认实现注册名就位（@register 自动入表）。"""
    assert _RF == {"pin_fkine_solver": PinFkineSolver}
    assert _RI == {"pin_ikine_solver": PinIkineSolver}
    assert _RJ == {"pin_jacobian_solver": PinJacobianSolver}
    assert _RD == {"pin_dynamics_solver": PinDynamicsSolver}
    assert _RT == {"to_joint_traj_planner": ToJointTrajPlanner}
    assert _RC == {"joint_position_controller": JointPositionController}


def test_pin_fkine():
    """PinFkine：单点/批量 pose 与 T 表示均与 pin 直算一致；未知帧名可定位报错。"""
    arm, fk = _arm(), PinFkineSolver()
    rng = np.random.default_rng(7)
    m, d = arm.pin_model, arm.pin_data
    for _ in range(5):
        q = arm.rand_q_arm(rng=rng)
        pose = fk.frame_pose(arm, q, arm.ee_frame_name)
        pin.forwardKinematics(m, d, q)
        T = pin.updateFramePlacement(m, d, m.getFrameId(arm.ee_frame_name)).homogeneous
        assert np.allclose(pose.position, T[:3, 3], atol=1e-12)
        assert np.allclose(pose.orientation, R_to_quat(T[:3, :3]))
        assert np.allclose(pose.T, T, atol=1e-12)            # T 表示互转一致
    # 批量（ABC 模板）：(N,n) → list[Pose]，逐行核对
    Q = np.stack([arm.rand_q_arm(rng=rng) for _ in range(4)])
    poses = fk.solve(arm, Q, arm.ee_frame_name)
    assert isinstance(poses, list) and len(poses) == 4
    assert np.allclose(poses[2].T, fk.solve(arm, Q[2], arm.ee_frame_name, rep="T"))
    try:
        fk.frame_pose(arm, Q[0], "no_such_frame")
        raise AssertionError("未知帧名应抛 ValueError")
    except ValueError as e:
        assert "no_such_frame" in str(e)


def test_pin_jacobian():
    """PinJacobian：base/local 两参考系与 pin 直算一致；微分逆解往返闭合。"""
    arm, js = _arm(), PinJacobianSolver()
    rng = np.random.default_rng(3)
    m, d = arm.pin_model, arm.pin_data
    fid = m.getFrameId(arm.ee_frame_name)
    q = arm.rand_q_arm(rng=rng)
    for ref, rf in (("base", pin.LOCAL_WORLD_ALIGNED), ("local", pin.LOCAL)):
        J = js.jac(arm, q, arm.ee_frame_name, ref=ref)
        assert J.shape == (6, arm.n_arm)
        assert np.allclose(J, pin.computeFrameJacobian(m, d, q, fid, rf))
    # 微分逆解往返：V = J·dq → q̇* = J*·V ≈ dq（非奇异处）
    dq = np.array([0.1, -0.2, 0.15, 0.3, -0.1, 0.05])
    V = js.fkine_vel(arm, q, dq, arm.ee_frame_name)
    dq2 = js.ikine_vel(arm, q, V, arm.ee_frame_name)
    assert np.allclose(dq2, dq, atol=1e-3)
    try:
        js.jac(arm, q, arm.ee_frame_name, ref="world")
        raise AssertionError("未知 ref 应抛 ValueError")
    except ValueError:
        pass


def test_pin_ikine():
    """PinIkine：可达位姿收敛到容差内；不可达目标 success=False 不抛。"""
    arm, ik = _arm(), PinIkineSolver()
    fk = PinFkineSolver()
    rng = np.random.default_rng(11)
    for _ in range(3):
        q_star = arm.rand_q_arm(rng=rng)                    # 限位内可达位姿
        pose = fk.frame_pose(arm, q_star, arm.ee_frame_name)
        # 起点取限位内随机位形（arm_home 全零位于部分关节限位边界，劣质起点）
        q0 = arm.rand_q_arm(rng=rng)
        res = ik.solve(arm, pose, arm.ee_frame_name, q0=q0, tol=1e-4)
        assert res.success and res.err < 1e-4, (res.success, res.err)
        pose2 = fk.frame_pose(arm, res.q, arm.ee_frame_name)
        assert np.allclose(pose2.position, pose.position, atol=1e-3)
    # 不可达（远超臂长）：不抛、success=False
    far = Pose(position=np.array([10.0, 0.0, 0.0]))
    res = ik.solve(arm, far, arm.ee_frame_name, q0=arm.rand_q_arm(rng=rng), iters=100)
    assert (not res.success) and res.err > 1e-2


def test_pin_dynamics():
    """PinDynamics：M 对称正定；G=rnea(q,0,0)；C=nle−G；Λ 对称；f_ext 拒绝。"""
    arm, dyn = _arm(), PinDynamicsSolver()
    rng = np.random.default_rng(5)
    m, d = arm.pin_model, arm.pin_data
    q = arm.rand_q_arm(rng=rng)
    dq = np.array([0.2, -0.3, 0.1, 0.4, -0.2, 0.1])
    M = dyn.mass_matrix(arm, q)
    assert np.allclose(M, M.T)                              # 对称
    assert np.all(np.linalg.eigvalsh(M) > 0)                # 正定
    z = np.zeros(arm.n_arm)
    assert np.allclose(dyn.gravity(arm, q), pin.rnea(m, d, q, z, z))
    nle = pin.nonLinearEffects(m, d, q, dq)
    assert np.allclose(dyn.coriolis(arm, q, dq), nle - pin.rnea(m, d, q, z, z))
    tau = dyn.idyn(arm, q, dq, dq * 2)
    assert np.allclose(tau, pin.rnea(m, d, q, dq, dq * 2))
    Lam = dyn.cartesian_inertia(arm, q, arm.ee_frame_name)
    assert Lam.shape == (6, 6) and np.allclose(Lam, Lam.T, atol=1e-9)
    try:
        dyn.idyn(arm, q, dq, dq, f_ext=dq)
        raise AssertionError("f_ext 应抛 NotImplementedError")
    except NotImplementedError:
        pass


class _QArm:
    """规划器测试哑臂：状态可控、记录发布帧。"""

    def __init__(self, q, dq, arm_home=None):
        self.state = ArmState(joint=JointState(q=np.asarray(q, float),
                                               dq=np.asarray(dq, float)))
        self.arm_home = np.zeros_like(self.state.joint.q) \
            if arm_home is None else np.asarray(arm_home, float)
        self._target = None

    def get_target_traj(self):
        return self._target

    def get_arm_state(self):
        return self.state

    def set_current_frame(self, frame):
        self.frames = getattr(self, "frames", []) + [frame]


def test_quintic_planner():
    """Quintic：六端点边界条件精确满足；中段连续；超末端钳位；缺省 0。"""
    q0 = np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1])
    dq0 = np.array([0.05, 0.0, -0.05, 0.1, 0.0, 0.0])
    q1 = np.array([0.5, 0.1, 0.6, 0.2, 0.3, 0.2])
    dq1 = np.array([0.0, 0.02, 0.0, 0.0, 0.0, -0.01])
    ddq1 = np.zeros(6)
    T = 2.0
    fa = _QArm(q0, dq0)
    t_end = time.time() + T
    fa._target = [TrajFrame(time=t_end, q=q1, dq=dq1)]     # ddq 缺省 0
    qp = ToJointTrajPlanner()
    qp.plan_once(fa)

    def sample(dt):
        fr = qp.sample_frame(time.time() + dt)
        assert fr.dq is not None and fr.ddq is not None
        return fr

    # 边界（容差吸收执行期时间漂移，取小裕度）
    fr0 = sample(-T + 1e-6)
    assert np.allclose(fr0.q, q0, atol=1e-6) and np.allclose(fr0.dq, dq0, atol=1e-5)
    assert np.allclose(fr0.ddq, 0.0, atol=1e-4)            # ddq0 = 0（无反馈）
    fr1 = sample(T + 5.0)
    assert np.allclose(fr1.q, q1, atol=1e-6) and np.allclose(fr1.dq, dq1, atol=1e-5)
    assert np.allclose(fr1.ddq, ddq1, atol=1e-4)           # 超末端钳位保持末态
    # 中段平滑连续（相邻采样步进有界）
    prev = sample(0.5).q
    for dt in np.linspace(0.6, T - 0.1, 12):
        fr = sample(dt)
        assert np.max(np.abs(fr.q - prev)) < 0.2
        prev = fr.q
    # 多目标取终点（最晚时刻）
    fa._target = [TrajFrame(time=t_end, q=q1),
                  TrajFrame(time=t_end + 1.0, q=q1 * 0.5)]
    qp.plan_once(fa)
    fr = qp.sample_frame(time.time() + 5.0)
    assert np.allclose(fr.q, q1 * 0.5, atol=1e-6)


class _LimArm:
    """控制器测试哑臂：提供硬限位与状态。"""

    def __init__(self, dq_max):
        self.arm_limits = type("L", (), {"q_min": np.full(6, -3.0),
                                         "q_max": np.full(6, 3.0),
                                         "dq_max": np.full(6, dq_max),
                                         "tau_max": np.full(6, 20.0)})()


def test_joint_position_controller():
    """关节位置控制器：MODE=POSITION；小步直通；大步裁剪到 dq_max/hz 并告警。"""
    ctrl = JointPositionController(ctrl_hz=200.0)
    assert ctrl.MODE == ControlMode.POSITION
    dq_max = 2.0
    arm = _LimArm(dq_max)
    step_max = dq_max / 200.0                                # 0.01 rad/拍

    class _Counting(logging.Handler):
        n = 0

        def emit(self, record):
            self.n += 1

    handler = _Counting()
    log = logging.getLogger("joyarm_core.controller")
    log.addHandler(handler)
    try:
        # ① 小步：原样直通
        st = ArmState(joint=JointState(q=np.full(6, 0.5)))
        fr = TrajFrame(time=1.0, q=np.full(6, 0.505))
        mode, cmd = ctrl.compute(arm, fr, st)
        assert mode == ControlMode.POSITION and np.allclose(cmd["q"], 0.505)
        # ② 大步：逐关节裁剪到 q_cur ± step_max（帧间不超过最大速度）
        fr2 = TrajFrame(time=1.0, q=np.full(6, 0.9))
        mode, cmd2 = ctrl.compute(arm, fr2, st)
        assert np.allclose(cmd2["q"], 0.51)                 # 0.5 + 0.01
        for _ in range(5):                                   # 连续越限：告警节流
            ctrl.compute(arm, fr2, st)
        assert 1 <= handler.n <= 2                           # 0.5s 节流
        # ③ 帧缺 q：显性报错
        try:
            ctrl.compute(arm, TrajFrame(time=1.0, pose=Pose()), st)
            raise AssertionError("缺 q 应抛 ValueError")
        except ValueError:
            pass
    finally:
        log.removeHandler(handler)


# ============================================================
# config 端到端 + 运动管线冒烟（fake 后端）
# ============================================================
class _FakeMotionBackend:
    """端到端哑后端：新鲜缓存状态 + 记录模式与指令。"""

    connected = True

    def __init__(self, q0):
        self.calls = []
        self._mode = None
        self.state = ArmState(joint=JointState(q=np.asarray(q0, float),
                                               dq=np.zeros(6)))

    def state_age_arm(self, joint=None):
        return 0.0

    def read_state_cache_arm(self, joint=None):
        return self.state

    def read_state_end(self, joint=None):
        return {"q": [0.0]}

    def set_mode_arm(self, mode, joint=None):
        self.calls.append(("set_mode_arm", mode))
        self._mode = mode

    def read_mode_arm(self, joint=None):
        return self._mode

    def send_position_arm(self, q, joint=None):
        self.calls.append(("send_position_arm", np.asarray(q, float).copy()))

    def set_mode_end(self, mode, joint=None):
        self.calls.append(("set_mode_end", mode))

    def send_mit_end(self, *a, **k):
        self.calls.append(("send_mit_end",))

    def enable_arm(self, joint=None):
        pass

    def enable_end(self, joint=None):
        pass


_ROBOTICS_CFG = {
    "fkine": "pin_fkine_solver",
    "ikine": "pin_ikine_solver",
    "jacobian": "pin_jacobian_solver",
    "dynamics": "pin_dynamics_solver",
    "traj": "to_joint_traj_planner",
    "control": "joint_position_controller",
}


def _configured_arm():
    """robotics 段选配六默认实现构造的 JoyArm（离线）。"""
    cfg = load_config("joyarm_dm")
    cfg["robotics"] = dict(_ROBOTICS_CFG)
    return JoyArm(model="joyarm_dm", config=cfg)


def test_config_end_to_end_offline():
    """config → REGISTRY → 成员组装 → 离线门面全链路可用。"""
    arm = _configured_arm()
    for domain in _ROBOTICS_CFG:
        assert arm.list_solvers(domain) == [_ROBOTICS_CFG[domain]]
    rng = np.random.default_rng(2)
    q = arm.rand_q_arm(rng=rng)
    pose = arm.fkine(q, arm.ee_frame_name)                  # 离线计算可用
    assert pose.position.shape == (3,)
    J = arm.jac(q, arm.ee_frame_name)
    assert J.shape == (6, 6)
    assert arm.manipulability(q, arm.ee_frame_name) > 0
    res = arm.ikine(pose, arm.ee_frame_name, q0=arm.arm_home, tol=1e-4)
    assert res.success
    M = arm.mass_matrix(q)
    assert np.allclose(M, M.T)
    assert np.allclose(arm.gravity(q),
                       PinDynamicsSolver().gravity(arm, q))
    # 未连接仍禁执行（执行语义不因域配置改变）
    try:
        arm.get_arm_state()
        raise AssertionError("离线执行应抛 RuntimeError")
    except RuntimeError:
        pass


def test_pipeline_with_default_solvers():
    """运动管线 + 六默认实现冒烟：帧含 q/dq/ddq、指令经裁剪下发、停机阻尼。"""
    arm = _configured_arm()
    saved = (arm._backend, arm.connected)
    try:
        be = _FakeMotionBackend(q0=arm.arm_home)
        arm._backend = be
        arm.connected = True
        arm.start_motion()                                  # MODE 自动 POSITION + 同步首帧
        assert ("set_mode_arm", ControlMode.POSITION) in be.calls
        fr = arm.get_current_frame()
        assert fr is not None and fr.q is not None \
            and fr.dq is not None and fr.ddq is not None    # 五次采样三参考
        # 写入未来目标：滚动重规划 → 采样帧向目标推进
        tgt = clamp_to_limits(arm.arm_home + 0.05, arm.arm_limits)
        arm.set_target_traj([TrajFrame(time=time.time() + 2.0, q=tgt)])
        time.sleep(0.3)
        pos = [c for c in be.calls if c[0] == "send_position_arm"]
        assert len(pos) >= 20                               # 200Hz 控制流
        fr2 = arm.get_current_frame()
        # fake 状态冻结 → 每次重规划都以静态 q0 重锚，位移小但方向正确
        assert np.max(fr2.q - arm.arm_home) > 1e-6          # 帧已朝目标离开起点
        arm.stop_motion()                                   # 默认纯阻尼收尾
        assert ("set_mode_arm", ControlMode.MIT) in be.calls
    finally:
        if arm._motion_running:
            arm.stop_motion(damping=False)
        arm._backend, arm.connected = saved


if __name__ == "__main__":
    failed = 0
    for name in sorted(globals()):
        fn = globals()[name]
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {name}: {e}")
    sys.exit(1 if failed else 0)
