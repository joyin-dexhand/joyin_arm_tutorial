"""robotics 六域回归测试：fkine / ikine / jacobian / dynamics / trajectory / control（离线、无硬件）。

覆盖：每个测试对应一次 bug 修复（片段由各修复员提交、本文件整编合并），防止修复行为回归；
求解器调用只消费离线构造的 ``joyArm("joyarm_dm")``（不连机，URDF/pin 模型与算法层）或鸭子
``StubArm``，对照独立参考式（pinocchio 原生算法等）逐位/数值校验，真机行为一概不触。

组织方式（参照 test/test_backend_base.py 的头部说明与用例组织）：
- ``_arm()``：模块级缓存离线 joyArm("joyarm_dm")（JoyArm 构造后默认未连接，backend 不上总线；
  构造失败抛携带原因的异常，不返回 None），``pin_model`` 等只读成员跨用例共享，避免重复构造；
- ``StubArm``：traj/control 域用鸭子 stub（ctrl_hz / home_arm / joint_limits_hard_arm /
  get_state_arm / send_position_arm，与 JoyArm 公开属性同名），可按需改 ``_st.q/dq`` 模拟当前状态，
  ``cmds`` 记录每拍下发指令；
- 同一 bug 的多个片段合并为一个测试函数；函数命名 ``test_<主题>``。

已收录：
- dynamics-F0：``PinDynamicsSolver.mass_matrix`` 的 crba 拆行写法与修复前单行原式输出
  逐位一致（纯格式整理不改行为），对称化照常生效；
- single_joint_hold 直传保持（2026-10-12）：有效 q 恒输出（其余字段忽略）、q 空/维度
  不符/多帧/空目标回退 q_home、发布为拷贝不受外部原地修改；
- single_joint_cubic 三次多项式（2026-10-12 新增）：起终点位置+速度四约束边界、
  超末端保持、无效回退恒定 q_home、过期目标直接恒定发布目标。
"""
import time

import numpy as np
import pinocchio as pin

from joyarm_core import JointLimits, JointState, TrajFrame, joyArm
from joyarm_core.robotics.dynamics import PinDynamicsSolver
from joyarm_core.robotics.trajectory import SingleJointCubicTrajPlanner, SingleJointHoldTrajPlanner

# ---- 离线机械臂（模块级缓存：构造不连机，模型只读共享） ----
_ARM = None


def _arm():
    """离线 joyArm("joyarm_dm")：仅算法层使用（pin_model 等只读成员），跨用例共享单例。"""
    global _ARM
    if _ARM is None:
        try:
            _ARM = joyArm("joyarm_dm")      # 失败抛携带原因的异常（不返回 None）
        except Exception:
            print("离线构造 joyArm('joyarm_dm') 失败，异常如下：")
            raise
    return _ARM


# ---- 鸭子 stub（traj/control 离线检查用：属性名与 JoyArm 公开接口同名） ----
class StubArm:
    ctrl_hz = 200.0
    home_arm = np.zeros(6)
    joint_limits_hard_arm = JointLimits(q_min=np.full(6, -3.0), q_max=np.full(6, 3.0),
                                        dq_max=np.full(6, 5.0), tau_max=np.full(6, 30.0))

    def __init__(self):
        self._st = JointState(q=np.zeros(6), dq=np.zeros(6))
        self.cmds = []

    def get_state_arm(self):
        return self._st

    def send_position_arm(self, q, vlim=None, flim=None):
        self.cmds.append(np.array(q))


# ============================================================
# dynamics 域（bug 修复回归）
# ============================================================

# ---- dynamics-F0：crba 拆行写法与修复前单行原式输出逐位一致（纯格式不改行为） ----
def test_dynamics_F0_mass_matrix_split_line_bitwise():
    arm = _arm()
    model = arm.pin_model
    solver = PinDynamicsSolver()
    rng = np.random.default_rng(42)
    for i in range(5):
        q = rng.uniform(model.lowerPositionLimit, model.upperPositionLimit)
        M_mod = solver.mass_matrix(arm, q)          # 修复后的拆行实现
        M_ref = np.array(pin.crba(model, pin.Data(model),
                                  np.asarray(q, dtype=float)), dtype=float)  # 修复前原式
        M_ref = np.triu(M_ref) + np.triu(M_ref, 1).T                # + 原对称化
        assert M_mod.tobytes() == M_ref.tobytes(), f"q[{i}] 拆行改写改变输出"
    assert np.allclose(M_mod, M_mod.T), "对称化失效"


# ============================================================
# trajectory 域（single_joint_hold 直传保持 / single_joint_cubic 三次多项式，2026-10-12）
# ============================================================

# ---- 有效 q 恒定直传（time/dq/ddq 等其余字段忽略；发布为拷贝不受外部修改） ----
def test_traj_single_joint_hold_valid_q_held_constant():
    stub = StubArm()
    planner = SingleJointHoldTrajPlanner()
    q = np.full(6, 0.3)
    frame = TrajFrame(time=time.time() + 1.0, q=q, dq=np.ones(6))  # dq/time 应被忽略
    assert planner.plan(stub, frame) is True
    fr1 = planner.sample(time.time())
    fr2 = planner.sample(time.time() + 5.0)
    assert np.allclose(fr1.q, q) and np.allclose(fr2.q, q)     # 任意时刻恒为目标 q
    assert np.allclose(fr1.dq, 0.0) and np.allclose(fr1.ddq, 0.0)
    q[0] = 9.9                                                  # 外部原地修改不影响已发布目标
    assert np.allclose(planner.sample(time.time()).q[0], 0.3)


# ---- q 为空 / 维度不符 / 空目标：一律回退 q_home ----
def test_traj_single_joint_hold_invalid_q_falls_back_home():
    stub = StubArm()
    planner = SingleJointHoldTrajPlanner()
    for bad in (TrajFrame(),                        # q 为 None
                TrajFrame(q=np.zeros(0)),           # q 为空数组
                TrajFrame(q=np.zeros(5)),           # 维度不符（≠ n=6）
                None, []):                          # 空目标
        assert planner.plan(stub, bad) is True
        assert np.allclose(planner.sample(time.time()).q, stub.home_arm)


# ---- 多帧：视为无效，告警一次并回退 q_home（单帧规划后告警复位） ----
def test_traj_single_joint_hold_multi_frame_warns_and_falls_back(caplog):
    stub = StubArm()
    planner = SingleJointHoldTrajPlanner()
    frames = [TrajFrame(q=np.full(6, 0.1)), TrajFrame(q=np.full(6, 0.2))]
    with caplog.at_level("WARNING", logger="joyarm_core.traj_planner_single_joint_hold"):
        for _ in range(3):                          # 同一多帧目标重复规划
            assert planner.plan(stub, frames) is True
        assert len(caplog.records) == 1             # 只告警一次
        assert np.allclose(planner.sample(time.time()).q, stub.home_arm)
        caplog.clear()
        assert planner.plan(stub, TrajFrame(q=np.full(6, 0.4))) is True
        assert not caplog.records                   # 单帧规划复位，不告警
        assert np.allclose(planner.sample(time.time()).q, 0.4)


# ---- 未规划先采样：RuntimeError ----
def test_traj_single_joint_hold_sample_before_plan_raises():
    planner = SingleJointHoldTrajPlanner()
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
    for bad in (TrajFrame(time=time.time() + 1.0),                # q 为 None
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
