"""robotics 六域离线回归（fkine / ikine / jacobian / dynamics / trajectory / control；无硬件）。

覆盖：每个测试对应一次 bug 修复或域契约核对（片段由各修复员提交、本文件整编合并），
防止修复行为回归；求解器调用只消费离线构造的 `joyArm("rebot_arm_b601_dm")`（不连机，URDF/pin
模型与算法层）或鸭子 `StubArm`，对照独立参考式（pinocchio 原生算法等）逐位/数值校验，
真机行为一概不触。

组织方式（参照 test/test_backend_base.py 的头部说明与用例组织）：
- `_arm()`：模块级缓存离线 joyArm("rebot_arm_b601_dm")（JoyArm 构造后默认未连接，backend 不上总线；
  构造失败抛携带原因的异常，不返回 None），`pin_model` 等只读成员跨用例共享，避免重复构造；
- `StubArm`：traj/control 域用鸭子 stub（ctrl_hz / n_arm / home_arm / joint_limits_hard_arm /
  get_state_arm / send_position_arm / get_gravity / send_mit_arm，与 JoyArm 公开属性同名），
  可按需改 `_st.q/dq` 模拟当前状态、设 `grav_fn` 控制重力桩，`cmds` / `mit_cmds` 记录下发指令；
- 同一 bug 的多个片段合并为一个测试函数；函数命名 `test_<主题>`。

已收录：
- fkine：Pin 求解器与 pinocchio 原生 updateFramePlacement 逐位一致（多 q × 多帧，齐次
  矩阵比对）、MDH 白盒与 pin 原生数值一致（仅末端帧）、未知帧/非末端帧 ValueError；
- ikine：Pin 数值法与解析法对 fk 采样位姿往返复现（位置 + 姿态）、解析法 `n_iter=0`、
  远超工作空间目标 `success=False`；
- jacobian：Pin 雅可比与 pinocchio 原生 getFrameJacobian（LOCAL_WORLD_ALIGNED）逐位一致、
  阻尼伪逆 `J·J*≈I`（满秩；λ=0 退化最小范数伪逆）；
- dynamics-F0：`PinDynamicsSolver.mass_matrix` 输出与 pin.crba 原式逐位一致
  （含上三角对称化，纯整理不改行为）；
- single_joint_direct 单帧关节域直传（2026-10-12 hold，2026-11-04 改名升级）：q 锚定有效性，
  q/dq/ddq/tau 原样透传（缺失 None）、q 空/维度不符/多帧/空目标回退 q_home（三域置 None）；
- single_joint_cubic 三次多项式（2026-10-12 新增）：起终点位置+速度四约束边界、
  超末端保持、无效回退恒定 q_home、过期目标直接恒定发布目标；
- control：JointPositionDirectController 有效帧经 send_position_arm 下发该 q、缺 q / q 含
  NaN 抛 ValueError 且不下发（本层不裁剪，越限与限速归 backend 与电机）；
- control-drag：JointDragTeachController 拖动示教（2026-11-04 新增）：MIT 平衡点保持 +
  阈值/步进率缓释跟随、构造数值校验与首拍广播、q 无效零下发、dyn 缺失退化 tau=0。
"""
import time

import numpy as np
import pinocchio as pin

from joyarm_core import ControlMode, JointLimits, JointState, Pose, TrajFrame, joyArm
from joyarm_core.robotics.control import JointDragTeachController, JointPositionDirectController
from joyarm_core.robotics.dynamics import PinDynamicsSolver
from joyarm_core.robotics.fkine import MdhFkineSolver, PinFkineSolver
from joyarm_core.robotics.ikine import AnalyticIkineSolver, PinIkineSolver
from joyarm_core.robotics.jacobian import PinJacobianSolver
from joyarm_core.robotics.trajectory import SingleJointCubicTrajPlanner, SingleJointDirectTrajPlanner
from joyarm_core.utils.limits import rand_within_limits

# ---- 离线机械臂（模块级缓存：构造不连机，模型只读共享） ----
_ARM = None


def _arm():
    """离线 joyArm("rebot_arm_b601_dm")：仅算法层使用（pin_model 等只读成员），跨用例共享单例。"""
    global _ARM
    if _ARM is None:
        try:
            _ARM = joyArm("rebot_arm_b601_dm")      # 失败抛携带原因的异常（不返回 None）
        except Exception:
            print("离线构造 joyArm('rebot_arm_b601_dm') 失败，异常如下：")
            raise
    return _ARM


# ---- 鸭子 stub（traj/control 离线检查用：属性名与 JoyArm 公开接口同名） ----
class StubArm:
    ctrl_hz = 200.0
    home_arm = np.zeros(6)
    n_arm = 6                                                # 本体自由度（drag teach 首拍广播定维）
    joint_limits_hard_arm = JointLimits(q_min=np.full(6, -3.0), q_max=np.full(6, 3.0),
                                        dq_max=np.full(6, 5.0), tau_max=np.full(6, 30.0))

    def __init__(self):
        self._st = JointState(q=np.zeros(6), dq=np.zeros(6))
        self.cmds = []
        self.mit_cmds = []                                   # send_mit_arm 五参存档 (q, dq, tau, kp, kd)
        self.grav_fn = None                                  # get_gravity 桩（None = dyn 未配置抛错）

    def get_state_arm(self):
        return self._st

    def send_position_arm(self, q, vlim=None, flim=None):
        self.cmds.append(np.array(q))

    def get_gravity(self, q=None):
        if self.grav_fn is None:                             # grav_fn=None 模拟 dyn 域未配置
            raise RuntimeError("StubArm - get_gravity：dyn 域未配置（桩 grav_fn=None）")
        return self.grav_fn(q)

    def send_mit_arm(self, q, dq=None, tau=None, kp=None, kd=None):
        self.mit_cmds.append((np.array(q), np.array(dq), np.array(tau),
                              np.array(kp), np.array(kd)))


# ============================================================
# dynamics 域（bug 修复回归）
# ============================================================

# ---- dynamics-F0：mass_matrix 输出与 pin.crba 原式逐位一致（纯整理不改行为） ----
def test_dynamics_F0_mass_matrix_split_line_bitwise():
    arm = _arm()
    model = arm.pin_model
    solver = PinDynamicsSolver()
    rng = np.random.default_rng(42)
    for i in range(5):
        q = rng.uniform(model.lowerPositionLimit, model.upperPositionLimit)
        M_mod = solver.mass_matrix(arm, q)          # 现行实现
        M_ref = np.array(pin.crba(model, pin.Data(model),
                                  np.asarray(q, dtype=float)), dtype=float)  # 原式参照
        M_ref = np.triu(M_ref) + np.triu(M_ref, 1).T                         # + 上三角对称化
        assert M_mod.tobytes() == M_ref.tobytes(), f"q[{i}] 实现整理改变输出"
    assert np.allclose(M_mod, M_mod.T), "对称化失效"


# ============================================================
# trajectory 域（single_joint_direct 单帧直传 / single_joint_cubic 三次多项式，2026-10-12）
# ============================================================

# ---- q-only 目标：任意时刻采样 q=目标、dq/ddq/tau 均 None（发布为拷贝不受外部修改） ----
def test_traj_single_joint_direct_q_only_passes_through():
    stub = StubArm()
    planner = SingleJointDirectTrajPlanner()
    q = np.full(6, 0.3)
    frame = TrajFrame(time=time.time() + 1.0, q=q)  # 仅 q（time 仅回填采样时刻）
    assert planner.plan(stub, frame) is True
    fr1 = planner.sample(time.time())
    fr2 = planner.sample(time.time() + 5.0)
    assert np.allclose(fr1.q, q) and np.allclose(fr2.q, q)     # 任意时刻恒为目标 q
    for fr in (fr1, fr2):                          # 缺失域原样 None（不补 0）
        assert fr.dq is None and fr.ddq is None and fr.tau is None
    q[0] = 9.9                                      # 外部原地修改不影响已发布目标
    assert np.allclose(planner.sample(time.time()).q[0], 0.3)


# ---- 带 dq 与 tau 的帧直传：q+dq+tau 有效、ddq 缺失 → 采样逐域相等、ddq 为 None ----
def test_traj_single_joint_direct_dq_tau_pass_through():
    stub = StubArm()
    planner = SingleJointDirectTrajPlanner()
    q, dq, tau = np.full(6, 0.2), np.full(6, 0.1), np.full(6, 1.5)
    frame = TrajFrame(time=time.time() + 1.0, q=q, dq=dq, tau=tau)  # ddq 缺失
    assert planner.plan(stub, frame) is True
    fr = planner.sample(time.time())
    assert np.allclose(fr.q, q) and np.allclose(fr.dq, dq) and np.allclose(fr.tau, tau)
    assert fr.ddq is None                          # 缺失域原样 None（不补 0）


# ---- q 为空 / 维度不符 / 空目标：一律回退 q_home，dq/ddq/tau 均 None ----
def test_traj_single_joint_direct_invalid_q_falls_back_home():
    stub = StubArm()
    planner = SingleJointDirectTrajPlanner()
    for bad in (TrajFrame(dq=np.ones(6)),          # q 为 None（q 锚定：dq 有效也不透传）
                TrajFrame(q=np.zeros(0)),          # q 为空数组
                TrajFrame(q=np.zeros(5)),          # 维度不符（≠ n=6）
                None, []):                         # 空目标
        assert planner.plan(stub, bad) is True
        fr = planner.sample(time.time())
        assert np.allclose(fr.q, stub.home_arm)    # 回退 q_home
        assert fr.dq is None and fr.ddq is None and fr.tau is None  # 三域置 None


# ---- 多帧：视为无效，告警一次并回退 q_home（单帧规划后告警复位） ----
def test_traj_single_joint_direct_multi_frame_warns_and_falls_back(caplog):
    stub = StubArm()
    planner = SingleJointDirectTrajPlanner()
    frames = [TrajFrame(q=np.full(6, 0.1)), TrajFrame(q=np.full(6, 0.2))]
    with caplog.at_level("WARNING", logger="joyarm_core.traj_planner_single_joint_direct"):
        for _ in range(3):                          # 同一多帧目标重复规划
            assert planner.plan(stub, frames) is True
        assert len(caplog.records) == 1             # 只告警一次
        fr = planner.sample(time.time())
        assert np.allclose(fr.q, stub.home_arm) and fr.dq is None and fr.tau is None
        caplog.clear()
        assert planner.plan(stub, TrajFrame(q=np.full(6, 0.4))) is True
        assert not caplog.records                   # 单帧规划复位，不告警
        assert np.allclose(planner.sample(time.time()).q, 0.4)


# ---- 未规划先采样：RuntimeError ----
def test_traj_single_joint_direct_sample_before_plan_raises():
    planner = SingleJointDirectTrajPlanner()
    try:
        planner.sample(time.time())
        assert False, "应抛 RuntimeError"
    except RuntimeError:
        pass


# ============================================================
# trajectory 域（single_joint_cubic 三次多项式，2026-10-12 新增）
# ============================================================

# ---- 四边界约束：起点 (q0, dq0)、终点 (q1, dq1)；超末端精确保持末端状态 ----
def test_traj_single_joint_cubic_boundary_conditions():
    stub = StubArm()
    q0 = np.array([0.1, -0.2, -0.1, 0.0, 0.3, 0.0])
    dq0 = np.full(6, 0.05)
    stub._st.q = q0
    stub._st.dq = dq0
    planner = SingleJointCubicTrajPlanner()
    q1 = np.array([0.3, -0.5, -0.4, 0.2, 0.5, 0.1])
    dq1 = np.full(6, 0.02)
    assert planner.plan(stub, TrajFrame(time=time.time() + 0.5, q=q1, dq=dq1)) is True
    far = planner.sample(time.time() + 10.0)          # 超末端：精确保持末端状态
    assert np.allclose(far.q, q1, atol=1e-9)          # q(T) = q1
    assert np.allclose(far.dq, dq1, atol=1e-9)        # q'(T) = dq1（终点速度约束）
    now = planner.sample(time.time())                 # τ≈0：起点位置/速度
    assert np.allclose(now.q, q0, atol=0.02)
    assert np.allclose(now.dq, dq0, atol=0.2)
    assert now.ddq.shape == (6,)


# ---- 无效目标（q 空/维度不符/空目标）：回退恒定 q_home ----
def test_traj_single_joint_cubic_invalid_falls_back_hold_home():
    stub = StubArm()
    planner = SingleJointCubicTrajPlanner()
    for bad in (TrajFrame(time=time.time() + 1.0),                 # q 为 None
                TrajFrame(time=time.time() + 1.0, q=np.zeros(5)),  # 维度不符
                None, []):                                         # 空目标
        assert planner.plan(stub, bad) is True
        fr = planner.sample(time.time() + 10.0)
        assert np.allclose(fr.q, stub.home_arm)
        assert np.allclose(fr.dq, 0.0)


# ---- 目标已过期：不规划多项式，直接恒定发布目标（维持终点状态） ----
def test_traj_single_joint_cubic_expired_target_holds_goal():
    stub = StubArm()
    stub._st.q = np.zeros(6)
    planner = SingleJointCubicTrajPlanner()
    q1 = np.full(6, 0.3)
    assert planner.plan(stub, TrajFrame(time=time.time() - 1.0, q=q1)) is True
    fr = planner.sample(time.time() + 10.0)
    assert np.allclose(fr.q, q1)                       # 恒为目标（终点状态）
    assert np.allclose(fr.dq, 0.0)


# ---- 多帧：告警一次并回退恒定 q_home ----
def test_traj_single_joint_cubic_multi_frame_warns(caplog):
    stub = StubArm()
    planner = SingleJointCubicTrajPlanner()
    frames = [TrajFrame(time=time.time() + 1.0, q=np.full(6, 0.1)),
              TrajFrame(time=time.time() + 2.0, q=np.full(6, 0.2))]
    with caplog.at_level("WARNING", logger="joyarm_core.traj_planner_single_joint_cubic"):
        assert planner.plan(stub, frames) is True
        assert len(caplog.records) == 1
        assert np.allclose(planner.sample(time.time() + 10.0).q, stub.home_arm)


# ---- 未规划先采样：RuntimeError ----
def test_traj_single_joint_cubic_sample_before_plan_raises():
    planner = SingleJointCubicTrajPlanner()
    try:
        planner.sample(time.time())
        assert False, "应抛 RuntimeError"
    except RuntimeError:
        pass


# ============================================================
# fkine 域（Pin 对照 pinocchio 原生 / MDH 白盒对照，2026-10-19）
# ============================================================

def _frames(arm):
    """受测帧名：cfg 末端帧 + 本体 link6 / link3。"""
    return [arm.get_config()["joyarm"]["frame_end"], "link6", "link3"]


def _pose_T_ref(model, q, frame):
    """pinocchio 原生参考：forwardKinematics + updateFramePlacement 齐次矩阵。"""
    data = pin.Data(model)
    pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
    return pin.updateFramePlacement(model, data, model.getFrameId(frame)).homogeneous


# ---- Pin 求解器与 pinocchio 原生逐位一致（多 q × 多帧；齐次矩阵比对绕开四元数约定） ----
def test_fkine_pin_matches_pinocchio_native():
    arm = _arm()
    solver = PinFkineSolver()
    rng = np.random.default_rng(7)
    for i in range(5):
        q = rng.uniform(arm.pin_model.lowerPositionLimit, arm.pin_model.upperPositionLimit)
        for frame in _frames(arm):
            T = solver.solve(arm, q, frame).T
            assert np.allclose(T, _pose_T_ref(arm.pin_model, q, frame), atol=1e-9), \
                f"q[{i}] 帧 {frame} 与 pin 原生不一致"


# ---- MDH 白盒（cfg arm_mdh 解析式）与 pin 原生数值一致（仅支持末端帧）；非末端帧拒绝 ----
def test_fkine_mdh_matches_pin_and_rejects_non_end_frame():
    arm = _arm()
    frame_end = arm.get_config()["joyarm"]["frame_end"]
    solver = MdhFkineSolver()
    rng = np.random.default_rng(11)
    for _ in range(5):
        q = rand_within_limits(arm.joint_limits_soft_arm, rng=rng)
        T = solver.solve(arm, q, frame_end).T
        assert np.allclose(T, _pose_T_ref(arm.pin_model, q, frame_end), atol=1e-6)
    try:
        solver.solve(arm, np.zeros(arm.n_arm), "link6")
        assert False, "非末端帧应抛 ValueError"
    except ValueError:
        pass


# ---- 未知帧名：ValueError（拦截 pin getFrameId 静默哨兵） ----
def test_fkine_unknown_frame_raises():
    arm = _arm()
    try:
        PinFkineSolver().solve(arm, np.zeros(arm.n_arm), "no_such_frame")
        assert False, "未知帧应抛 ValueError"
    except ValueError:
        pass


# ============================================================
# ikine 域（数值/解析对 fk 采样位姿往返复现，2026-10-19）
# ============================================================

# ---- Pin 数值法：fk 采样位姿往返复现（位置 + 姿态），确定性重启 ----
def test_ikine_pin_roundtrip():
    arm = _arm()
    frame_end = arm.get_config()["joyarm"]["frame_end"]
    fk = PinFkineSolver()
    solver = PinIkineSolver()
    rng = np.random.default_rng(23)
    for _ in range(3):
        q_star = rand_within_limits(arm.joint_limits_soft_arm, rng=rng)
        pose = fk.solve(arm, q_star, frame_end)
        res = solver.solve(arm, pose, frame_end, arm.neutral_arm)
        assert res.success, f"可达位姿求解失败：err={res.err}"
        T_back = fk.solve(arm, res.q, frame_end).T
        assert np.allclose(T_back[:3, 3], pose.position, atol=1e-3)
        d_rot = float(np.linalg.norm(pin.log3(pose.T[:3, :3] @ T_back[:3, :3].T)))
        assert d_rot < 1e-3


# ---- 解析法：同样往返复现且 n_iter==0（闭式解不迭代） ----
def test_ikine_analytic_roundtrip_zero_iter():
    arm = _arm()
    frame_end = arm.get_config()["joyarm"]["frame_end"]
    fk = PinFkineSolver()
    solver = AnalyticIkineSolver()
    rng = np.random.default_rng(29)
    for _ in range(3):
        q_star = rand_within_limits(arm.joint_limits_soft_arm, rng=rng)
        pose = fk.solve(arm, q_star, frame_end)
        res = solver.solve(arm, pose, frame_end, arm.neutral_arm)
        assert res.success, f"可达位姿求解失败：err={res.err}"
        assert res.n_iter == 0, "解析法应恒 0 次迭代"
        T_back = fk.solve(arm, res.q, frame_end).T
        assert np.allclose(T_back[:3, 3], pose.position, atol=1e-4)
        d_rot = float(np.linalg.norm(pin.log3(pose.T[:3, :3] @ T_back[:3, :3].T)))
        assert d_rot < 1e-4


# ---- 远超工作空间的目标：success=False（数值法返回最优尝试解与残差） ----
def test_ikine_pin_unreachable_reports_failure():
    arm = _arm()
    frame_end = arm.get_config()["joyarm"]["frame_end"]
    far = Pose(position=np.array([10.0, 0.0, 0.0]))
    res = PinIkineSolver().solve(arm, far, frame_end, arm.neutral_arm)
    assert res.success is False
    assert res.err > 0.1
    assert res.q.shape == (arm.n_arm,) and bool(np.all(np.isfinite(res.q)))


# ============================================================
# jacobian 域（对照 pinocchio 原生 getFrameJacobian，2026-10-19）
# ============================================================

def _jac_ref(model, q, frame):
    """pinocchio 原生参考：computeJointJacobians + updateFramePlacements + getFrameJacobian
    （getFrameJacobian 消费 data.J 与 oMf，两者须先算好）。"""
    data = pin.Data(model)
    pin.computeJointJacobians(model, data, np.asarray(q, dtype=float))
    pin.updateFramePlacements(model, data)
    return pin.getFrameJacobian(model, data, model.getFrameId(frame), pin.LOCAL_WORLD_ALIGNED)


# ---- Pin 雅可比与 pinocchio 原生另一 API 路径逐位一致；未知帧 ValueError ----
def test_jac_pin_matches_pinocchio_native():
    arm = _arm()
    solver = PinJacobianSolver()
    rng = np.random.default_rng(31)
    for frame in _frames(arm):
        q = rng.uniform(arm.pin_model.lowerPositionLimit, arm.pin_model.upperPositionLimit)
        J = solver.jac(arm, q, frame)
        assert J.shape == (6, arm.n_arm), f"帧 {frame} 雅可比形状 {J.shape} ≠ (6, n)"
        assert np.allclose(J, _jac_ref(arm.pin_model, q, frame), atol=1e-9), f"帧 {frame} 不一致"
    try:
        solver.jac(arm, np.zeros(arm.n_arm), "no_such_frame")
        assert False, "未知帧应抛 ValueError"
    except ValueError:
        pass


# ---- 阻尼伪逆数值性质：满秩处 J @ J* ≈ I6（λ=0 精确伪逆、默认 λ 近似） ----
def test_jac_pinv_damped_identity_when_full_rank():
    arm = _arm()
    frame_end = arm.get_config()["joyarm"]["frame_end"]
    solver = PinJacobianSolver()
    rng = np.random.default_rng(37)
    checked = 0
    for _ in range(8):
        q = rand_within_limits(arm.joint_limits_soft_arm, rng=rng)
        J = solver.jac(arm, q, frame_end)
        s_min = float(np.linalg.svd(J, compute_uv=False)[-1])
        if s_min < 0.05:
            continue                       # 接近奇异跳过：λ²/s² 偏差发散，只对良态位形断言
        Jp0 = solver.jac_pinv_damped(arm, q, frame_end, damping=0.0)
        assert Jp0.shape == (arm.n_arm, 6)
        assert np.allclose(J @ Jp0, np.eye(6), atol=1e-8), "λ=0 最小范数伪逆应精确"
        Jp = solver.jac_pinv_damped(arm, q, frame_end)
        # 偏差逐奇异方向为 λ²/(s²+λ²)≤λ²/s_min²，s_min≥0.05 时 <4e-4
        assert np.allclose(J @ Jp, np.eye(6), atol=1e-3), "默认阻尼下近似单位映射"
        checked += 1
    assert checked >= 3, "随机采样竟无良态满秩位形，采样策略可疑"


# ============================================================
# control 域（JointPositionDirectController 位置参考直发，2026-10-19）
# ============================================================

# ---- 有效帧：经 send_position_arm 下发该 q，返回 True ----
def test_control_step_sends_frame_q():
    stub = StubArm()
    q = np.full(6, 0.25)
    assert JointPositionDirectController().step(stub, TrajFrame(q=q), stub._st) is True
    assert len(stub.cmds) == 1 and np.allclose(stub.cmds[0], q)


# ---- 缺 q：抛 ValueError 且不下发 ----
def test_control_step_missing_q_raises():
    stub = StubArm()
    try:
        JointPositionDirectController().step(stub, TrajFrame(), stub._st)
        assert False, "缺 q 应抛 ValueError"
    except ValueError:
        pass
    assert stub.cmds == []


# ---- q 含 NaN：抛 ValueError 且不下发 ----
def test_control_step_nan_q_raises():
    stub = StubArm()
    q = np.zeros(6)
    q[2] = np.nan
    try:
        JointPositionDirectController().step(stub, TrajFrame(q=q), stub._st)
        assert False, "NaN q 应抛 ValueError"
    except ValueError:
        pass
    assert stub.cmds == []


# ============================================================
# control 域（JointDragTeachController 拖动示教，2026-11-04）
# ============================================================

# ---- 首拍：平衡点锚定当前 q，MIT 帧五参 q=q_eq / dq=0 / kp·kd 默认广播 / tau=G(q) ----
def test_control_drag_teach_first_tick_anchors_and_sends_mit():
    stub = StubArm()
    stub.grav_fn = lambda q: np.full(6, 2.0)        # 可控重力桩：G(q) 恒 2.0
    q = np.array([0.1, -0.1, 0.2, 0.0, -0.2, 0.05])
    ctrl = JointDragTeachController()
    assert ctrl.step(stub, TrajFrame(), JointState(t=time.time(), q=q)) is True
    assert len(stub.mit_cmds) == 1                  # 本拍已下发
    q_cmd, dq_cmd, tau_cmd, kp_cmd, kd_cmd = stub.mit_cmds[0]
    assert np.allclose(q_cmd, q)                    # 平衡点 = 当前 q（首拍锚定）
    assert np.allclose(dq_cmd, 0.0)                 # 速度参考为 0（阻尼抗漂移）
    assert np.allclose(kp_cmd, 10.0) and np.allclose(kd_cmd, 1.0)  # 默认增益标量广播
    assert np.allclose(tau_cmd, 2.0)                # tau = G(q)


# ---- 阈值内第二拍：|q − q_eq| ≤ threshold，平衡点不动（仍发首拍锚定值） ----
def test_control_drag_teach_within_threshold_equilibrium_holds():
    stub = StubArm()
    stub.grav_fn = lambda q: np.zeros(6)
    ctrl = JointDragTeachController()
    q0 = np.zeros(6)
    ctrl.step(stub, TrajFrame(), JointState(t=time.time(), q=q0))
    q1 = q0 + np.array([0.04, -0.04, 0.05, -0.05, 0.01, 0.0])  # 全关节 |diff| ≤ 0.05
    assert ctrl.step(stub, TrajFrame(), JointState(t=time.time(), q=q1)) is True
    assert np.allclose(stub.mit_cmds[1][0], q0)     # 平衡点未移动（阈值内不动）


# ---- 单关节超阈值：仅该关节平衡点按 step_rate 前移，其余关节不动 ----
def test_control_drag_teach_over_threshold_moves_single_joint():
    stub = StubArm()
    stub.grav_fn = lambda q: np.zeros(6)
    ctrl = JointDragTeachController()
    q0 = np.zeros(6)
    ctrl.step(stub, TrajFrame(), JointState(t=time.time(), q=q0))
    q1 = q0.copy()
    q1[2] = 0.5                                     # 仅关节 3 超阈值（0.5 > 0.05）
    assert ctrl.step(stub, TrajFrame(), JointState(t=time.time(), q=q1)) is True
    expect = q0.copy()
    expect[2] = 0.1 * 0.5                           # step_rate=0.1：该关节前移 0.05
    assert np.allclose(stub.mit_cmds[1][0], expect)  # 其余关节平衡点不动


# ---- 构造参数非法：kd=0 / step_rate=0 / step_rate=1.5 → ValueError ----
def test_control_drag_teach_invalid_params_raise():
    for kw in ({"kd": 0.0}, {"step_rate": 0.0}, {"step_rate": 1.5}):
        try:
            JointDragTeachController(**kw)
            assert False, f"{kw} 应抛 ValueError"
        except ValueError:
            pass


# ---- 参数列表长度 ≠ n_arm：首拍广播抛 ValueError（构造时未知维数）、零下发 ----
def test_control_drag_teach_wrong_length_raises_on_first_tick():
    stub = StubArm()
    ctrl = JointDragTeachController(kd=np.full(5, 1.0))    # 长度 5 ≠ n_arm 6
    try:
        ctrl.step(stub, TrajFrame(), JointState(t=time.time(), q=np.zeros(6)))
        assert False, "长度不符应抛 ValueError"
    except ValueError:
        pass
    assert stub.mit_cmds == []                     # 首拍即抛错，未下发


# ---- dyn 未配置（get_gravity 抛 RuntimeError）：tau 退化为 0、warn 一次、仍下发 ----
def test_control_drag_teach_gravity_missing_degrades_warns_once(caplog):
    stub = StubArm()                               # grav_fn=None → 模拟 dyn 域未配置
    ctrl = JointDragTeachController()
    with caplog.at_level("WARNING", logger="joyarm_core.controller_joint_drag_teach"):
        for _ in range(2):                         # 两拍均走退化路径
            q = np.full(6, 0.1)
            assert ctrl.step(stub, TrajFrame(), JointState(t=time.time(), q=q)) is True
        assert len(caplog.records) == 1            # 退化告警只一次（去重）
    assert len(stub.mit_cmds) == 2                 # 两拍均仍下发
    assert np.allclose(stub.mit_cmds[0][2], 0.0) and np.allclose(stub.mit_cmds[1][2], 0.0)


# ---- joint_state.q 缺失 / 含 NaN：return False、零下发、warn 去重仅一次 ----
def test_control_drag_teach_bad_state_skips_and_warns_once(caplog):
    stub = StubArm()
    stub.grav_fn = lambda q: np.zeros(6)
    ctrl = JointDragTeachController()
    with caplog.at_level("WARNING", logger="joyarm_core.controller_joint_drag_teach"):
        for bad in (JointState(t=time.time(), q=None),  # q 缺失
                    JointState(t=time.time(), q=np.full(6, np.nan))):  # q 含 NaN
            assert ctrl.step(stub, TrajFrame(), bad) is False
        assert len(caplog.records) == 1            # 无效状态告警去重（仅一次）
    assert stub.mit_cmds == []                     # 零下发


# ---- MODE 声明：MIT（拖动示教需力矩前馈通道下发重力补偿） ----
def test_control_drag_teach_mode_is_mit():
    assert JointDragTeachController.MODE == ControlMode.MIT
