"""FakeArm 离线单测（无硬件、无后端、无线程；URDF/config 与 JoyArm 同源）。

覆盖：①构造与属性（关节数/关节名/限位/特征位形/末端帧）；②初始状态快照
（q=home、tcp.pose 由内置 fkine 填充）；③生命周期与指令（connect/enable
标志、三模式指令、越限裁剪、模式守卫、单关节扩展）；④move_j 理想插值
（轨迹数组边界、即时到位）；⑤末端（开/闭/归零/位置/力矩）；⑥目标轨迹
协议（深拷贝语义）；⑦六域算法以 FakeArm 为 arm 的协议符合性（fkine 对拍、
ikine 往返、jacobian/dynamics、ToJointTrajPlanner 规划、JointPositionController
控制）；⑧与真 JoyArm 同 q 同结果对拍；⑨get_config 深拷贝隔离。

运行：``python test/test_fakearm.py`` 或 pytest。
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import (  # noqa: E402
    ControlMode, FakeArm, JoyArm, TrajFrame,
    PinFkineSolver, PinIkineSolver, PinJacobianSolver, PinDynamicsSolver,
    ToJointTrajPlanner, JointPositionController,
)
from joyarm_core.joyarm import load_config  # noqa: E402

_Q_STAR = np.array([0.3, -0.6, -0.9, 0.5, 0.4, 0.8])   # 限位内的确定性测试位形


def test_construct_basic():
    """构造：关节数/关节名/型号/末端帧/表示；默认型号 joyarm_dm。"""
    arm = FakeArm()
    assert arm.model == "joyarm_dm" and arm.n_arm == 6 and arm.n_end == 1
    assert arm.joint_names_arm == [f"joint{i}" for i in range(1, 7)]
    assert arm.joint_names_end == ["gripper"]
    assert arm.pin_model is not None and arm.pin_data is not None
    assert arm.ee_frame_name == "link_end"
    assert 0 <= arm.ee_frame_id < len(arm.pin_model.frames)
    assert "FakeArm" in repr(arm) and "offline" in repr(arm)
    arm2 = FakeArm(q0=_Q_STAR)                      # 指定初始位形
    assert np.allclose(arm2.get_arm_state().joint.q, _Q_STAR)


def test_limits_and_pose_homes():
    """限位与特征位形：硬/软限位、zero/neutral 全零、home 来自 config。"""
    arm = FakeArm()
    assert np.allclose(arm.arm_limits.q_min, [-2.80, -3.14, -3.14, -1.87, -1.57, -3.14])
    assert np.allclose(arm.arm_limits.q_max, [2.80, 0.0, 0.0, 1.57, 1.57, 3.14])
    assert np.allclose(arm.arm_limits.dq_max, [50.0, 50.0, 50.0, 200.0, 200.0, 200.0])
    assert np.allclose(arm.arm_home, np.zeros(6))           # config arm_home 全零
    assert np.allclose(arm.arm_zero, 0) and np.allclose(arm.arm_neutral, 0)
    assert arm.arm_limits_soft is not None                  # 软限位 = 硬限位内缩
    assert np.allclose(arm.arm_limits_soft.q_max, [2.52, -0.16, -0.16, 1.40, 1.41, 2.83])
    assert np.allclose(arm.end_limits.q_min, -1.8) and np.allclose(arm.end_limits.q_max, 3.8)
    assert arm.tcp_limits.workspace_box is not None         # tcp_limits 已加载


def test_initial_state():
    """初始状态：q=home、dq/tau=0、未使能、POSITION 模式、恒健康；tcp.pose 与 fkine 一致。"""
    arm = FakeArm()
    st = arm.get_arm_state()                                 # 不要求 connect
    assert np.allclose(st.joint.q, arm.arm_home)
    assert np.allclose(st.joint.dq, 0) and np.allclose(st.joint.tau, 0)
    assert not st.joint.enabled.any()                        # 初始未使能
    assert st.joint.comm_ok.all() and not st.joint.error.any()
    assert st.mode == ControlMode.POSITION and st.timestamp > 0
    pose = arm.fkine(arm.arm_home, arm.ee_frame_name)
    assert np.allclose(st.tcp.pose.position, pose.position)
    assert np.allclose(st.tcp.pose.orientation, pose.orientation)


def test_lifecycle():
    """生命周期：with 上下文置/清连接标志；使能/失能（含单关节）。"""
    arm = FakeArm()
    with arm:
        assert arm.connected
        arm.enable_arm()
        assert arm.get_arm_state().joint.enabled.all()
        arm.disable_arm(joint=0)
        en = arm.get_arm_state().joint.enabled
        assert not en[0] and en[1:].all()
    assert not arm.connected
    try:
        arm.enable_end(joint=5)
        assert False, "末端索引越界应报错"
    except ValueError:
        pass


def test_position_command():
    """位置指令：即时置位、越限裁剪、速度清零；未连接/模式不符报错。"""
    arm = FakeArm()
    try:
        arm.set_arm_command(q=_Q_STAR)
        assert False, "未 connect 应报错"
    except RuntimeError:
        pass
    arm.connect()
    try:
        arm.set_arm_command(ControlMode.VELOCITY, dq=np.zeros(6))
        assert False, "模式不符应报错"
    except RuntimeError:
        pass
    arm.set_arm_command(q=_Q_STAR)
    st = arm.get_arm_state()
    assert np.allclose(st.joint.q, _Q_STAR) and np.allclose(st.joint.dq, 0)
    arm.set_arm_command(q=np.full(6, 10.0))                  # 越限 → 裁到硬限位
    assert np.allclose(arm.get_arm_state().joint.q, arm.arm_limits.q_max)


def test_velocity_and_mit_commands():
    """速度/力矩语义：只记录（裁上限）不动位置；MIT kp 非零才动、缺参报错。"""
    arm = FakeArm()
    arm.connect()
    arm.set_mode_arm(ControlMode.VELOCITY)
    arm.set_arm_command(ControlMode.VELOCITY, dq=np.full(6, 1000.0))
    st = arm.get_arm_state()
    assert np.allclose(st.joint.dq, arm.arm_limits.dq_max)   # 裁到 dq_max
    assert np.allclose(st.joint.q, arm.arm_home)             # 位置不动、不积分
    arm.set_mode_arm(ControlMode.MIT)
    try:
        arm.set_arm_command(ControlMode.MIT, q=_Q_STAR)
        assert False, "MIT 缺 dq/tau 应报错"
    except ValueError:
        pass
    arm.set_arm_command(ControlMode.MIT, q=_Q_STAR, dq=np.zeros(6),
                        tau=np.full(6, 100.0), kp=np.zeros(6))   # 纯前馈：不动
    st = arm.get_arm_state()
    assert np.allclose(st.joint.q, arm.arm_home)
    assert np.allclose(st.joint.tau, arm.arm_limits.tau_max)     # tau 裁上限
    arm.set_arm_command(ControlMode.MIT, q=_Q_STAR, dq=np.zeros(6),
                        tau=np.zeros(6), kp=np.full(6, 20.0))    # kp>0：位置生效
    assert np.allclose(arm.get_arm_state().joint.q, _Q_STAR)


def test_single_joint_command():
    """单关节指令：标量并入该关节、其余保持；形状/索引错误报错。"""
    arm = FakeArm()
    arm.connect()
    arm.set_arm_command(q=_Q_STAR)
    arm.set_arm_command(q=1.0, joint=3)
    q = arm.get_arm_state().joint.q
    assert q[3] == 1.0 and np.allclose(np.delete(q, 3), np.delete(_Q_STAR, 3))
    try:
        arm.set_arm_command(q=np.zeros(6), joint=3)
        assert False, "joint 指定时给向量应报错"
    except ValueError:
        pass
    try:
        arm.set_arm_command(q=1.0, joint=9)
        assert False, "关节索引越界应报错"
    except ValueError:
        pass


def test_move_j():
    """move_j：返回 (ts, qs, dqs) 轨迹数组、首末帧=起终点、状态即时到位。"""
    arm = FakeArm()
    try:
        arm.move_j(_Q_STAR)
        assert False, "未 connect 应报错"
    except RuntimeError:
        pass
    arm.connect()
    try:
        arm.move_j(_Q_STAR)
        assert False, "未使能应报错"
    except RuntimeError:
        pass
    arm.enable_arm()
    try:
        arm.move_j(np.zeros(3))
        assert False, "维度不符应报错"
    except ValueError:
        pass
    ts, qs, dqs = arm.move_j(_Q_STAR, t=1.0, rate=100)   # 显式采样率，不依赖 config
    assert ts.shape == (101,) and qs.shape == (101, 6) and dqs.shape == (101, 6)
    assert np.allclose(qs[0], arm.arm_home) and np.allclose(qs[-1], _Q_STAR)
    assert np.allclose(dqs[0], 0) and np.allclose(dqs[-1], 0)   # 三次插值起末速度零
    assert np.allclose(arm.get_arm_state().joint.q, _Q_STAR)    # 状态即时到位
    ts1, qs1, _ = arm.move_j(_Q_STAR)                            # 零路程：单帧
    assert len(ts1) == 1 and np.allclose(qs1[-1], _Q_STAR)
    _, qs2, _ = arm.move_j(np.full(6, 10.0), t=0.5)              # 越限裁剪
    assert np.allclose(qs2[-1], arm.arm_limits.q_max)
    assert np.allclose(arm.get_arm_state().joint.q, arm.arm_limits.q_max)


def test_end_commands():
    """末端：位置即时生效（裁行程）、开/闭/归零、力矩只记录、状态字典。"""
    arm = FakeArm()
    try:
        arm.set_end_position(1.0)
        assert False, "未 connect 应报错"
    except RuntimeError:
        pass
    arm.connect()
    arm.set_end_position(100.0)                       # 越行程 → 裁到 q_max=3.8
    assert np.allclose(arm.get_end_state()["q"], 3.8)
    arm.set_end_open()
    assert np.allclose(arm.get_end_state()["q"], -1.8)
    arm.set_end_close()
    assert np.allclose(arm.get_end_state()["q"], 3.8)
    arm.set_end_zero()
    assert np.allclose(arm.get_end_state()["q"], 0.0)
    arm.set_end_tau(50.0)
    assert np.allclose(arm.get_end_state()["tau"], 1.0)     # 裁 tau_max=1.0
    st = arm.get_end_state(joint=0)                         # 单电机查询
    assert set(st) >= {"q", "dq", "tau", "enabled", "error", "comm_ok"}
    assert st["q"].shape == (1,) and st["comm_ok"].all()


def test_rand_q_arm():
    """随机采样：全部落在硬限位内；size 给 (N,n)；同种子结果可复现。"""
    arm = FakeArm()
    q = arm.rand_q_arm()
    assert q.shape == (6,) and (q >= arm.arm_limits.q_min).all() \
        and (q <= arm.arm_limits.q_max).all()
    qs = arm.rand_q_arm(size=50, rng=np.random.default_rng(42))
    assert qs.shape == (50, 6) and (qs >= arm.arm_limits.q_min).all()
    assert np.array_equal(qs, arm.rand_q_arm(size=50, rng=np.random.default_rng(42)))


def test_target_traj_protocol():
    """目标轨迹协议：None→空表、单帧→单帧列表、深拷贝隔离（规划器消费入口）。"""
    arm = FakeArm()
    assert arm.get_target_traj() is None
    arm.set_target_traj(None)
    assert arm.get_target_traj() == []
    frame = TrajFrame(time=time.time() + 1.0, q=_Q_STAR)
    arm.set_target_traj(frame)
    traj = arm.get_target_traj()
    assert len(traj) == 1 and np.allclose(traj[0].q, _Q_STAR)
    frame.q = np.zeros(6)                       # 改原对象不影响已写入目标
    assert np.allclose(arm.get_target_traj()[0].q, _Q_STAR)


def test_fkine_domain():
    """fkine 域：门面与求解器直调一致（arm 作为协议参数）；T 表示与批量。"""
    arm = FakeArm()
    solver = PinFkineSolver()
    for q in (arm.arm_home, _Q_STAR):
        p1, p2 = arm.fkine(q, "link_end"), solver.frame_pose(arm, q, "link_end")
        assert np.allclose(p1.position, p2.position)
        assert np.allclose(p1.orientation, p2.orientation)
    T = arm.fkine(_Q_STAR, "link_end", rep="T")
    assert T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1])
    Ts = arm.fkine(np.tile(_Q_STAR, (5, 1)), "link_end", rep="T")
    assert Ts.shape == (5, 4, 4)


def test_ikine_domain():
    """ikine 域：fkine→ikine 往返复位关节角；求解器直调等价；不可达目标失败。"""
    arm = FakeArm()
    solver = PinIkineSolver()
    target = arm.fkine(_Q_STAR, "link_end")
    res = arm.ikine(target, "link_end", arm.arm_neutral)
    assert res.success and np.allclose(res.q, _Q_STAR, atol=1e-2)
    res2 = solver.solve(arm, target, "link_end", arm.arm_neutral)
    assert np.allclose(res2.q, res.q, atol=1e-6)
    far = arm.fkine(_Q_STAR, "link_end")
    far.position = np.array([5.0, 5.0, 5.0])            # 远超工作空间
    assert not arm.ikine(far, "link_end", arm.arm_neutral).success


def test_jacobian_domain():
    """jacobian 域：形状/有限性、门面与求解器一致、可操作度、静力学 τ=JᵀF。"""
    arm = FakeArm()
    solver = PinJacobianSolver()
    J = arm.jac(_Q_STAR, "link_end")
    assert J.shape == (6, 6) and np.isfinite(J).all()
    assert np.allclose(J, solver.jac(arm, _Q_STAR, "link_end"))
    assert arm.manipulability(_Q_STAR, "link_end") > 0
    F = np.array([1.0, -2.0, 0.5, 0.1, 0.0, -0.3])
    assert np.allclose(arm.statics(_Q_STAR, F, "link_end"), J.T @ F)


def test_dynamics_domain():
    """dynamics 域：M(q) 对称且对角为正、G 有限、门面与求解器一致。"""
    arm = FakeArm()
    solver = PinDynamicsSolver()
    M = arm.mass_matrix(_Q_STAR)
    assert M.shape == (6, 6) and np.allclose(M, M.T)
    assert (np.diag(M) > 0).all()
    G = arm.gravity(_Q_STAR)
    assert G.shape == (6,) and np.isfinite(G).all()
    assert np.allclose(G, solver.gravity(arm, _Q_STAR))


def test_traj_planner_domain():
    """trajectory 域：ToJointTrajPlanner 以 FakeArm 规划——起点=当前状态、末端=目标。"""
    arm = FakeArm()
    planner = ToJointTrajPlanner()
    arm.set_target_traj(TrajFrame(time=time.time() + 2.0, q=_Q_STAR))   # 须 > now+dt_min(1s)
    assert planner.plan_once(arm) is True
    f0 = planner.sample_frame(time.time())                  # τ≈0：起点
    assert np.allclose(f0.q, arm.arm_home, atol=1e-4)
    f1 = planner.sample_frame(time.time() + 10.0)           # 超末端钳位：目标
    assert np.allclose(f1.q, _Q_STAR, atol=1e-6)
    assert f1.dq is not None and f1.ddq is not None


def test_controller_domain():
    """control 域：JointPositionController 以 FakeArm 计算——单周期步长裁剪生效。"""
    arm = FakeArm()
    ctrl = JointPositionController(ctrl_hz=200.0)
    assert ctrl.MODE == ControlMode.POSITION
    state = arm.get_arm_state()
    frame = TrajFrame(time=time.time(), q=np.full(6, 3.0))  # 远目标
    mode, cmd = ctrl.compute(arm, frame, state)
    assert mode == ControlMode.POSITION and set(cmd) == {"q"}
    step = np.asarray(arm.arm_limits.dq_max) / ctrl.ctrl_hz
    expect = state.joint.q + np.clip(3.0 - state.joint.q, -step, step)
    assert np.allclose(cmd["q"], expect)


def test_parity_with_joyarm():
    """与真 JoyArm 对拍：同 q 同 URDF → fkine/jac 结果一致；限位/关节名一致。"""
    cfg = load_config("joyarm_dm")
    cfg["robotics"] = {"fkine": "pin_fkine_solver", "jacobian": "pin_jacobian_solver"}
    real = JoyArm("joyarm_dm", config=cfg)                  # 离线构造（不连接）
    fake = FakeArm()
    assert fake.n_arm == real.n_arm
    assert fake.joint_names_arm == real.joint_names_arm
    assert np.allclose(fake.arm_limits.q_min, real.arm_limits.q_min)
    assert np.allclose(fake.arm_limits.q_max, real.arm_limits.q_max)
    rng = np.random.default_rng(7)
    for _ in range(3):
        q = fake.rand_q_arm(rng=rng)
        p1, p2 = fake.fkine(q, "link_end"), real.fkine(q, "link_end")
        assert np.allclose(p1.position, p2.position, atol=1e-9)
        assert np.allclose(p1.orientation, p2.orientation, atol=1e-9)
    assert np.allclose(fake.jac(_Q_STAR, "link_end"),
                       real.jac(_Q_STAR, "link_end"), atol=1e-9)


def test_get_config_deepcopy():
    """get_config：返回深拷贝（改返回值不影响对象内部配置）。"""
    arm = FakeArm()
    cfg = arm.get_config()
    cfg["basic"]["name"] = "mutated"
    cfg["joyarm"]["arm_home"] = [1.0] * 6
    assert arm.get_config()["basic"]["name"] == "joyarm_dm"
    assert np.allclose(arm.get_config()["joyarm"]["arm_home"], 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    fails = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"✓ {_name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"✗ {_name}: {type(e).__name__}: {e}")
    print("失败", fails, "项" if fails else "—— 全部通过")
    sys.exit(1 if fails else 0)
