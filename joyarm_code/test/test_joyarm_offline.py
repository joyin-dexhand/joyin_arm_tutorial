"""JoyArm 充分离线测试（不依赖硬件；与 test_joyarm_factory.py 互补）。

覆盖 test_joyarm_factory / test_default_solvers 未触达的路径：
① 构造失败面（URDF 侧：关节顺序/数量/末端帧/型号目录、后端选型名）与
   求解器构造参数注入；② tcp_limits 解析与缺省；③ check_config 补充项；
④ 连接生命周期（connect 失败收尾、with 上下文、disconnect 停管线）；
⑤ 无末端型号（n_end=0）行为；⑥ 指令/末端/参数分发；⑦ damping_mode 帧
序列、move_j 零时长、占位方法、is_in_position pose 分支、rand_q_arm；
⑧ 运动管线启停（含首帧同步发布）。

已知缺陷文档化：无末端型号 check_hardware 必报"末端状态读取失败"
（test_check_hardware_no_end_model_known_defect，断言当前行为；修复后翻转）。

运行：``python test/test_joyarm_offline.py`` 或 pytest。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from joyarm_core import (  # noqa: E402
    ArmState, ControlMode, JointState, JoyArm, Pose, clamp_to_limits,
)
from joyarm_core.backend import Backend  # noqa: E402
from joyarm_core.joyarm import load_config  # noqa: E402

_ROBOTICS_CFG = {
    "fkine": "pin_fkine_solver",
    "ikine": "pin_ikine_solver",
    "jacobian": "pin_jacobian_solver",
    "dynamics": "pin_dynamics_solver",
    "traj": "to_joint_traj_planner",
    "control": "joint_position_controller",
}

_END_METHODS = ("read_state_end", "state_age_end", "set_mode_end", "enable_end",
                "disable_end", "set_zero_end", "clear_fault_end", "send_position_end",
                "send_tau_end", "send_mit_end", "send_action_end",
                "read_param_end", "write_param_end")


def _cfg(rm_end: bool = False, robotics=None):
    """基础配置拷贝：可去掉末端段、可指定 robotics 段。"""
    cfg = copy.deepcopy(load_config("joyarm_dm"))
    if rm_end:
        del cfg["backend"]["end"]
        del cfg["joyarm"]["end_home"]
        cfg["joyarm"].pop("end_soft_limits", None)
    if robotics is not None:
        cfg["robotics"] = copy.deepcopy(robotics)
    return cfg


def _mk(rm_end: bool = False, robotics=None):
    return JoyArm(model="joyarm_dm", config=_cfg(rm_end, robotics))


def _full_state(n: int, q) -> ArmState:
    q_arr = np.full(n, q) if np.isscalar(q) else np.asarray(q, dtype=float)
    return ArmState(joint=JointState(
        q=q_arr, dq=np.zeros(n), tau=np.zeros(n),
        enabled=np.ones(n, bool), comm_ok=np.ones(n, bool),
        error=np.zeros(n, bool), angle_ok=np.ones(n, bool)))


class _RecBE(Backend):
    """记录型哑后端：记录全部调用；q 状态机向目标收敛（模真机跟随）；
    可模拟"无末端"语义（末端方法抛 RuntimeError，同 backend_dm）；
    可注入 connect 失败。模式缓存语义同 backend_dm。"""

    connected = True

    def __init__(self, n=6, converge=0.5, no_end=False, connect_fails=False,
                 cache_q=1.5):
        Backend.__init__(self, {})
        self.n, self.converge = n, converge
        self.no_end, self.connect_fails = no_end, connect_fails
        self.cache_q = cache_q
        self.calls = []
        self._q = np.zeros(n)
        self._target = np.zeros(n)
        self._mode_arm, self._mode_end = {}, {}
        self.age = {"arm": 0.0, "end": 0.0}

    # ---- 无末端守卫：模拟 backend_dm 的"本型号未配置末端" ----
    def _end_guard(self):
        if self.no_end:
            raise RuntimeError(
                "backend_dm.py - _end_motors_for：本型号未配置末端（config backend.end 缺失）")

    def end_call_names(self):
        return [c[0] for c in self.calls if c[0] in _END_METHODS]

    def names(self):
        return [c[0] for c in self.calls]

    # ---- 连接 ----
    def connect(self):
        if self.connect_fails:
            raise RuntimeError("串口打不开")
        self.calls.append(("connect",))

    def disconnect(self):
        self.calls.append(("disconnect",))

    # ---- 状态（缓存恒新鲜 q=1.5；同步读走 q 状态机）----
    def read_state_arm(self, joint=None):
        self._q = self._q + self.converge * (self._target - self._q)
        return ArmState(joint=JointState(
            q=self._q.copy(), dq=np.zeros(self.n), tau=np.zeros(self.n),
            enabled=np.ones(self.n, bool), comm_ok=np.ones(self.n, bool),
            error=np.zeros(self.n, bool), angle_ok=np.ones(self.n, bool)))

    def read_state_end(self, joint=None):
        self._end_guard()
        self.calls.append(("read_state_end", joint))
        return {"q": [0.0], "comm_ok": [True], "error": [False], "enabled": [True]}

    def read_state_cache_arm(self, joint=None):
        return _full_state(self.n, self.cache_q)

    def read_state_cache_end(self, joint=None):
        self._end_guard()
        return {"q": [0.25], "comm_ok": [True], "error": [False], "enabled": [True]}

    def state_age_arm(self, joint=None):
        return self.age["arm"]

    def state_age_end(self, joint=None):
        self._end_guard()
        return self.age["end"]

    # ---- 模式（缓存语义同 backend_dm：各电机一致才返回）----
    def set_mode_arm(self, mode, joint=None):
        self.calls.append(("set_mode_arm", mode, joint))
        for i in range(self.n):
            self._mode_arm[i] = mode

    def read_mode_arm(self, joint=None):
        if len(self._mode_arm) != self.n or len(set(self._mode_arm.values())) != 1:
            return None
        return list(self._mode_arm.values())[0]

    def set_mode_end(self, mode, joint=None):
        self._end_guard()
        self.calls.append(("set_mode_end", mode, joint))
        self._mode_end[0] = mode

    def read_mode_end(self, joint=None):
        self._end_guard()
        return list(self._mode_end.values())[0] if self._mode_end else None

    # ---- 指令内核：记录 + q 状态机跟随位置目标 ----
    def _send_position_arm(self, q, joint=None):
        self._target = np.asarray(q, dtype=float).copy()
        self.calls.append(("send_position_arm", self._target.copy(), joint))

    def _send_velocity_arm(self, dq, joint=None):
        self.calls.append(("send_velocity_arm", np.asarray(dq, dtype=float).copy(), joint))

    def _send_mit_arm(self, q, dq, tau_ff, kp=None, kd=None, joint=None):
        self._target = np.asarray(q, dtype=float).copy()
        self.calls.append(("send_mit_arm", self._target.copy(),
                           np.asarray(dq, dtype=float), np.asarray(tau_ff, dtype=float),
                           kp, kd, joint))

    def _send_position_end(self, position, joint=None):
        self._end_guard()
        self.calls.append(("send_position_end", np.asarray(position, dtype=float).copy(), joint))

    def _send_tau_end(self, tau, joint=None):
        self._end_guard()
        self.calls.append(("send_tau_end", np.asarray(tau, dtype=float).copy(), joint))

    def _send_mit_end(self, q, dq, tau_ff, kp=None, kd=None, joint=None):
        self._end_guard()
        self.calls.append(("send_mit_end", np.asarray(q, dtype=float),
                           np.asarray(dq, dtype=float), np.asarray(tau_ff, dtype=float),
                           kp, kd, joint))

    def send_action_end(self, action, joint=None):
        self._end_guard()
        self.calls.append(("send_action_end", action, joint))

    # ---- 其余接口：记录型空实现 ----
    def enable_arm(self, joint=None):
        self.calls.append(("enable_arm", joint))

    def disable_arm(self, joint=None):
        self.calls.append(("disable_arm", joint))

    def set_zero_arm(self, joint=None):
        self.calls.append(("set_zero_arm", joint))

    def clear_fault_arm(self, joint=None):
        self.calls.append(("clear_fault_arm", joint))

    def enable_end(self, joint=None):
        self._end_guard()
        self.calls.append(("enable_end", joint))

    def disable_end(self, joint=None):
        self._end_guard()
        self.calls.append(("disable_end", joint))

    def set_zero_end(self, joint=None):
        self._end_guard()
        self.calls.append(("set_zero_end", joint))

    def clear_fault_end(self, joint=None):
        self._end_guard()
        self.calls.append(("clear_fault_end", joint))

    def read_param_arm(self, key, joint=None):
        self.calls.append(("read_param_arm", key, joint))
        return 1.0

    def write_param_arm(self, key, value, joint=None, persist=False):
        self.calls.append(("write_param_arm", key, value, joint, persist))

    def read_param_end(self, key, joint=None):
        self._end_guard()
        self.calls.append(("read_param_end", key, joint))
        return 2.0

    def write_param_end(self, key, value, joint=None, persist=False):
        self._end_guard()
        self.calls.append(("write_param_end", key, value, joint, persist))


def _attach(arm, **kw):
    """换装记录型后端并置为已连接（不启动保活线程；需要时另调 connect）。"""
    be = _RecBE(n=arm.n_arm, **kw)
    arm._backend, arm.connected = be, True
    return be


def _expect(exc, fn, *subs):
    try:
        fn()
        raise AssertionError(f"应抛 {exc.__name__}")
    except exc as e:
        for s in subs:
            assert s in str(e), f"消息应含 {s!r}，实际：{e}"


# ============================================================
# ① 构造失败面与配置解析
# ============================================================
def test_construct_joint_order_mismatch():
    """config 关节顺序与 URDF 不一致 → 构造失败（防关节角装错轴）。"""
    cfg = _cfg()
    js = cfg["backend"]["arm"]["joints"]
    js[1], js[2] = js[2], js[1]
    _expect(ValueError, lambda: JoyArm("joyarm_dm", config=cfg), "顺序")


def test_construct_joint_count_mismatch():
    cfg = _cfg()
    del cfg["backend"]["arm"]["joints"][5]
    _expect(ValueError, lambda: JoyArm("joyarm_dm", config=cfg), "关节数")


def test_construct_bad_ee_frame():
    cfg = _cfg()
    cfg["basic"]["ee_frame"] = "nope"
    _expect(ValueError, lambda: JoyArm("joyarm_dm", config=cfg), "末端帧")


def test_construct_bad_robot_dir():
    cfg = _cfg()
    cfg["basic"]["robot"] = "nope"
    _expect(ValueError, lambda: JoyArm("joyarm_dm", config=cfg), "robot_model")


def test_construct_bad_backend_name():
    cfg = _cfg()
    cfg["backend"]["name"] = "bogus"
    _expect(ValueError, lambda: JoyArm("joyarm_dm", config=cfg), "backend_dm")


def test_solver_ctor_params_injected():
    """求解器构造参数经 config robotics.<域> 的 {name:..., **参数} 注入到实例。"""
    arm = _mk(robotics={"traj": {"name": "to_joint_traj_planner",
                                 "plan_hz": 3.0, "sample_hz": 111.0,
                                 "dt_min_required": 0.25}})
    assert arm.list_solvers("traj") == ["to_joint_traj_planner"]
    p = arm._traj_planners["to_joint_traj_planner"]
    assert (p.plan_hz, p.sample_hz, p.dt_min_required) == (3.0, 111.0, 0.25)


def test_tcp_limits_parsing_and_default():
    """workspace_box 两行写法自动转置为 (3,2)；速度/力上限直读；未配置为 None。"""
    arm = _mk()
    box = np.asarray(arm.tcp_limits.workspace_box, dtype=float)
    assert box.shape == (3, 2)
    assert np.all(box[:, 0] < box[:, 1])                    # 每行 [min, max]
    assert np.allclose(box[:, 0], [-0.5, -0.5, 0.0])        # yaml 的 min 行已转置
    assert arm.tcp_limits.v_lin_max == 1.0 and arm.tcp_limits.t_max == 5.0
    cfg_none = _cfg()
    del cfg_none["joyarm"]["tcp_limits"]
    assert JoyArm("joyarm_dm", config=cfg_none).tcp_limits.workspace_box is None


def test_check_config_more_failures():
    """补充失败项：end 缺 name 键 / robotics 规格非法 / 命名不一致 / 缺 basic 段。"""
    cfg = _cfg()
    del cfg["backend"]["end"]["joints"][0]["name"]
    _expect(ValueError, lambda: JoyArm.check_config("joyarm_dm", cfg), "name")
    cfg = _cfg()
    cfg["robotics"] = {"fkine": 123}
    _expect(ValueError, lambda: JoyArm.check_config("joyarm_dm", cfg), "类型非法")
    cfg = _cfg()
    cfg["basic"]["name"] = "other"
    _expect(ValueError, lambda: JoyArm.check_config("joyarm_dm", cfg), "不一致")
    cfg = _cfg()
    del cfg["basic"]
    _expect(ValueError, lambda: JoyArm.check_config("joyarm_dm", cfg), "basic")


# ============================================================
# ② 连接生命周期
# ============================================================
def test_connect_failure_clean():
    """connect 抛错：异常透传、connected 保持 False、无保活线程残留。"""
    arm = _mk()
    arm._backend = _RecBE(connect_fails=True)
    _expect(RuntimeError, arm.connect, "串口")
    assert arm.connected is False
    assert arm._state_thread is None
    _attach(arm)                                            # 换好后可正常连
    arm.connect()
    assert arm.connected and arm._state_thread is not None
    arm.disconnect()
    assert arm._state_thread is None and not arm.connected


def test_with_context_lifecycle():
    """with：进入未连接自动 connect；退出按序 stop→disable→disconnect；
    已连接时进入不重复 connect。"""
    arm = _mk()
    arm._backend = _RecBE()
    with arm:
        assert arm.connected
        assert "connect" in arm._backend.names()
    names = arm._backend.names()
    for expect_name in ("disable_arm", "disable_end", "disconnect"):
        assert expect_name in names
    assert names.index("disable_arm") < names.index("disconnect")
    assert not arm.connected
    be = _attach(arm)                                       # 已连接再进 with：不重复连
    arm.connected = True
    with arm:
        assert "connect" not in be.names()


def test_disconnect_stops_pipeline():
    """管线运行中 disconnect：自动停三线程（不切阻尼）再断开。"""
    arm = _mk(robotics=_ROBOTICS_CFG)
    be = _attach(arm)
    arm.start_motion()
    assert arm._motion_running and arm.get_current_frame() is not None
    mit_before = len([c for c in be.calls if c[0] == "send_mit_arm"])
    arm.disconnect()
    assert not arm._motion_running and arm._motion_threads == {}
    assert "disconnect" in be.names()
    # damping=False 路径：disconnect 停管线不额外发阻尼帧
    assert len([c for c in be.calls if c[0] == "send_mit_arm"]) == mit_before


# ============================================================
# ③ 无末端型号（n_end=0）
# ============================================================
def test_no_end_model_basics():
    """n_end=0：构造正常；get_end_state 清晰报错；keepalive 不触末端；阻尼可用。"""
    arm = _mk(rm_end=True)
    assert arm.n_end == 0 and arm.end_home.shape == (0,)
    be = _attach(arm, no_end=True)
    _expect(RuntimeError, arm.get_end_state, "末端")
    arm._keepalive_step()                                   # 无异常
    assert be.end_call_names() == []                        # 未触任何末端方法
    arm.damping_mode()                                      # 不抛错（末端失败仅告警）
    assert len([c for c in be.calls if c[0] == "send_mit_arm"]) >= 1


def test_check_hardware_no_end_model_known_defect():
    """已知缺陷（断言当前行为，修复后请翻转）：无末端型号 check_hardware 把
    "末端状态读取失败"当硬件问题上报——健康的无末端臂自检必失败。
    修复后 check_hardware() 应静默通过：届时把断言翻转为 ok is True。"""
    arm = _mk(rm_end=True)
    _attach(arm, no_end=True)
    try:
        arm.check_hardware()
        ok = True
    except RuntimeError:
        ok = False
    assert ok is False, ("check_hardware 已能对无末端型号静默通过——缺陷已修复，"
                         "请把本断言翻转为 True 并更新文档")


# ============================================================
# ④ 指令 / 末端 / 参数分发
# ============================================================
def test_set_arm_command_dispatch():
    arm = _mk()
    be = _attach(arm)
    arm.set_mode_arm(ControlMode.POSITION)
    arm.set_arm_command(ControlMode.POSITION, q=np.full(6, 0.1))
    assert np.allclose(be.calls[-1][1], 0.1)
    arm.set_mode_arm(ControlMode.VELOCITY)
    arm.set_arm_command(ControlMode.VELOCITY, dq=np.full(6, 0.2))
    assert be.calls[-1][0] == "send_velocity_arm"
    _expect(ValueError,
            lambda: arm.set_arm_command(ControlMode.VELOCITY), "dq")
    arm.set_mode_arm(ControlMode.MIT)
    _expect(ValueError,
            lambda: arm.set_arm_command(ControlMode.MIT, q=np.zeros(6), dq=np.zeros(6)),
            "tau")
    arm.set_arm_command(ControlMode.MIT, q=np.zeros(6), dq=np.zeros(6),
                        tau=np.zeros(6))                     # kp/kd 缺省 None 透传
    assert be.calls[-1][0] == "send_mit_arm" and be.calls[-1][4] is None


def test_end_actions_and_params():
    arm = _mk()
    be = _attach(arm)
    arm.set_mode_end(ControlMode.POSITION)
    for fn, args, expect_name, val in (
            (arm.set_end_open, (), "send_action_end", "open"),
            (arm.set_end_close, (), "send_action_end", "close"),
            (arm.set_end_zero, (), "send_action_end", "zero"),
            (arm.set_end_position, (0.3,), "send_position_end", 0.3),
            (arm.set_end_tau, (0.1,), "send_tau_end", 0.1)):
        fn(*args)
        assert be.calls[-1][0] == expect_name
        assert (be.calls[-1][1] == val if isinstance(val, str)
                else np.allclose(be.calls[-1][1], val))
    assert arm.read_param_arm("pos_kp") == 1.0               # 读返回值透传
    assert arm.read_param_end("pos_kp") == 2.0
    arm.write_param_arm("pos_kp", 5.0, persist=True)
    assert be.calls[-1] == ("write_param_arm", "pos_kp", 5.0, None, True)
    arm.write_param_end("uv", 30.0, joint=0)
    assert be.calls[-1][:3] == ("write_param_end", "uv", 30.0)


def test_damping_mode_frame_sequence():
    """阻尼序列：切 MIT（本体+末端）→ 阻尼帧 → 使能 → 再发一帧；kd 值正确。"""
    arm = _mk()
    be = _attach(arm)
    arm.damping_mode(kd=2.5)
    names = be.names()
    assert names.index("set_mode_arm") < names.index("send_mit_arm")
    assert "enable_arm" in names and "enable_end" in names
    mits = [c for c in be.calls if c[0] == "send_mit_arm"]
    assert len(mits) == 2                                   # 使能前后各一帧
    for m in mits:
        assert np.allclose(m[1], 0) and np.allclose(m[2], 0) and np.allclose(m[3], 0)
        assert np.allclose(m[4], 0) and np.allclose(m[5], 2.5)


# ============================================================
# ⑤ 运动与安全
# ============================================================
def test_move_j_zero_time_single_frame():
    """t=0 不插值：单帧直发目标；目标=当前 → 立即到位。"""
    arm = _mk()
    q = clamp_to_limits(np.full(6, 1.5), arm.arm_limits)    # 限内目标（部分关节上限 < 1.5）
    be = _attach(arm, cache_q=q)
    arm.move_j(q, t=0)
    frames = [c for c in be.calls if c[0] == "send_position_arm"]
    assert len(frames) == 1 and np.allclose(frames[0][1], q)
    assert arm.is_in_position(q=q)


def test_placeholders_raise():
    """四个占位方法均 NotImplementedError，且消息含方法名可定位。"""
    arm = _mk()
    for name, fn in (("move_l", lambda: arm.move_l(Pose())),
                     ("teach_start", arm.teach_start),
                     ("teach_play", arm.teach_play),
                     ("teleop_keyboard", arm.teleop_keyboard)):
        _expect(NotImplementedError, fn, name)


def test_is_in_position_pose_branch():
    """pose 分支（配 pin fkine）：当前位姿→True；扰动→False；非 Pose→TypeError。"""
    arm = _mk(robotics=_ROBOTICS_CFG)
    _attach(arm)                                            # 缓存态恒 q=1.5
    pose = arm.fkine(np.full(6, 1.5), arm.ee_frame_name)
    assert arm.is_in_position(pose=pose)
    pose2 = Pose(position=pose.position + 0.01, orientation=pose.orientation)
    assert not arm.is_in_position(pose=pose2)               # tol_pos=1e-3
    _expect(TypeError, lambda: arm.is_in_position(pose=[1, 2, 3]), "Pose")


def test_rand_q_arm_limits_size_rng():
    arm = _mk()
    rng = np.random.default_rng(7)
    q = arm.rand_q_arm(size=200, rng=rng)
    assert q.shape == (200, arm.n_arm)
    assert np.all(q >= arm.arm_limits.q_min) and np.all(q <= arm.arm_limits.q_max)
    a = arm.rand_q_arm(rng=np.random.default_rng(3))
    b = arm.rand_q_arm(rng=np.random.default_rng(3))
    assert np.array_equal(a, b)                             # 同种子可复现


# ============================================================
# ⑥ 运动管线
# ============================================================
def test_start_motion_errors_and_first_frame():
    """离线 / 域未配均显性报错；配齐后启动即同步发布首帧（空目标→home 回退）；
    停止后可重启。"""
    arm = _mk()                                             # 未配 traj/control
    _expect(RuntimeError, arm.start_motion, "connect")
    _attach(arm)
    _expect(RuntimeError, arm.start_motion, "traj")
    arm2 = _mk(robotics=_ROBOTICS_CFG)
    _attach(arm2)
    arm2.start_motion()
    fr = arm2.get_current_frame()
    assert fr is not None and fr.q is not None              # 首帧同步发布
    assert ("set_mode_arm", ControlMode.POSITION, None) in arm2._backend.calls
    arm2.stop_motion(damping=False)
    assert arm2.get_current_frame() is None
    arm2.start_motion()                                     # 重启正常
    assert arm2._motion_running
    arm2.stop_motion(damping=False)


if __name__ == "__main__":
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
