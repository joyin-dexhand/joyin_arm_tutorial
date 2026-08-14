# `joyarm_code/` 架构与 API

> 开发者向文档：按「整体结构 → 各 py 文件 → 函数和类 → API 参考」梳理 JoyArm 教程代码库。
> 设计详则（命名约定 / 设计原则 / 依赖与接口）见 [`joyarm/架构设计.md`](joyarm/架构设计.md)，本文不重复。
> 用法与安装见 [`README.md`](README.md)。

## 一、整体代码库结构

```
joyarm_code/
├── joyarm/                          # ★ 核心 SDK（ROS2-free）
│   ├── __init__.py                  #   公开 API 扁平导出（__all__）
│   ├── arms/                        #   设备模型层（Arm / joyarm_rebot_dm）
│   ├── robotics/                    #   算法层（fkine / ikine / jacobian / trajectory / dyn / control）
│   ├── safety/                      #   安全层（safety）
│   ├── backends/                    #   通信层（三层 Backend）
│   ├── utils/                       #   基础层（types / transforms / interfaces）
│   ├── robots/                      #   URDF + meshes 资产（运行期加载）
│   └── configs/                     #   per-model YAML（backend_mas / backend_end）
├── joyarm_ros2/                     # ROS2 兄弟包（依赖 joyarm + rclpy，占位）
│   ├── arm_nodes/                   #   机器人 → ROS2 节点（Ch10）
│   └── apps/{ros2,teleop,vision}.py #   ROS2 适配 / 遥操 / 视觉（Ch10/12/14）
├── chapt/                           # 章节教学示例脚本（一次性）
├── test/                            # pytest 测试套件
├── quickstart/                      # 快速上手（CLI/GUI/任务编排，占位）
└── pyproject.toml                   # 工程配置（装 joyarm + joyarm_ros2）
```

**分层依赖（自底向上、无环）**：

```
        joyarm_ros2/  （兄弟包，依赖 joyarm + rclpy）
              ↓
   ┌──────────────────────────────────────────────┐
   │  arms/      设备模型：Arm（本体+末端基类） · joyarm_rebot_dm  ← arm.xx 门面
   ├──────────────────────────────────────────────┤
   │  robotics/  算法    safety/  安全    backends/  通信   │ ← 仅依赖 utils + ArmProtocol
   ├──────────────────────────────────────────────┤
   │  utils/  types · transforms · interfaces(ArmProtocol) │
   └──────────────────────────────────────────────┘
      robots/  configs/   ← 数据/资产，由 arms 运行期加载
```

> 关键：`robotics`/`safety` **不 import `arms`**，仅依赖 `utils.ArmProtocol`（依赖倒置，保证无环）。

### 模块导览

| 文件（小写） | 类（驼峰） | 章节 | 状态 | 职责 |
|:---:|:---:|:---:|:---:|:---|
| `utils/types.py` | — | Ch2 | ✅ | 共享数据类型 + 枚举 + `clamp_to_limits` |
| `utils/transforms.py` | — | Ch2 | ✅ | 纯 numpy SO(3)/SE(3) 数学（23 函数） |
| `utils/interfaces.py` | `ArmProtocol` | Ch2 | ✅ | 算法层接口契约（依赖倒置） |
| `configs/joyarm_rebot_dm.yaml` | — | Ch2/7 | ✅ | 型号 YAML（限位/home/`backend_mas`/`backend_end`） |
| `arms/arm.py` | `Arm` | Ch2 | ✅ | 完整臂基类（本体+末端，持有 backend_mas + backend_end） |
| `arms/joyarm_rebot_dm.py` | `JoyArmRebotDM(Arm)` | Ch2 | ✅ | 型号预设（绑两个 backend 类） |
| `robotics/fkine.py` | `fkine` | Ch2 | ✅ | 正运动学（形状重载，默认 pin） |
| `robotics/{ikine,jacobian,trajectory,dyn,control}.py` | … | Ch3-9 | 🟡 | 占位（无 `method` 参数） |
| `safety/safety.py` | … | Ch11 | 🟡 | 占位（关节/末端/整机三层） |
| `backends/backend.py` | `Backend` | Ch2 | ✅ | 通用通信根（connect/disconnect/read_state） |
| `backends/backend_mas.py` | `BackendMas` | Ch2 | ✅ | 本体通信抽象 |
| `backends/backend_end.py` | `BackendEnd` | Ch2 | ✅ | 末端通信抽象 |
| `backends/backend_mas_rebot_dm.py` | `BackendMasRebotDM` | Ch6 | 🟡 | reBot-DM 本体后端（占位） |
| `backends/backend_end_joygripper.py` | `BackendEndJoyGripper` | Ch13 | 🟡 | Joy 夹爪后端（占位） |
| `joyarm_ros2/{arm_nodes,apps}/` | — | Ch10/12/14 | 🟡 | ROS2 节点 + rviz2（占位） |

## 二、各 py 文件结构

### `joyarm/`（核心 SDK）

| 包 | 文件 | 关键定义 | 状态 |
|:---:|:---|:---|:---:|
| `utils` | `transforms.py` | `rot_x/y/z`、`rpy_to_R`/`R_to_rpy`、`rodrigues`/`axis_angle_to_R`/`R_to_axis_angle`、`quat_*`、`Rp_to_T`/`T_to_Rp`/`T_inv`/`T_mul`/`adT`、`slerp` | ✅ |
| `utils` | `types.py` | 枚举 `ControlMode`/`Severity`/`SafetyAction`/`TrajectorySpace`；数据类 `Pose`/`JointState`/`TcpState`/`ArmState`/`JointLimits`/`TcpLimits`/`Wrench`/`Twist`/`Violation`/`IKResult`/`ComplianceParams`；`clamp_to_limits` | ✅ |
| `utils` | `interfaces.py` | `ArmProtocol`（`@runtime_checkable Protocol`） | ✅ |
| `backends` | `backend.py` | `Backend(ABC)`：`connect`/`disconnect`/`read_state` | ✅ |
| `backends` | `backend_mas.py` | `BackendMas(Backend)`：`enable`/`disable`/`set_zero`/`scan`/`send_position`/`send_velocity`/`send_torque`/`send_mit` | ✅ |
| `backends` | `backend_end.py` | `BackendEnd(Backend)`：`send_position`/`send_force`/`send_action` | ✅ |
| `backends` | `backend_mas_rebot_dm.py` | `BackendMasRebotDM(BackendMas)`（CAN，6 电机） | 🟡 |
| `backends` | `backend_end_joygripper.py` | `BackendEndJoyGripper(BackendEnd)`（CAN，2 指夹爪） | 🟡 |
| `robotics` | `fkine.py` | `fkine`（+ `_fkine_single`/`_fkine_batch` 内部） | ✅ |
| `robotics` | `ikine.py` / `jacobian.py` / `trajectory.py` / `dyn.py` / `control.py` | 逆运动学 / 雅可比 / 轨迹 / 动力学 / 控制（占位函数族） | 🟡 |
| `safety` | `safety.py` | `StateMonitor`/`SelfCollisionChecker`/`ExternalCollisionDetector`/`SafetySupervisor`/`joint_limits_check`/`tcp_limits_check` | 🟡 |
| `arms` | `arm.py` | `Arm`（本体+末端基类） | ✅ |
| `arms` | `joyarm_rebot_dm.py` | `JoyArmRebotDM(Arm)` | ✅ |

### `joyarm_ros2/`（ROS2 兄弟包，全部占位）

| 文件 | 关键定义 | 章节 |
|:---|:---|:---:|
| `apps/ros2.py` | `Ros2Adapter`、`to_joint_msg`、`from_pose_msg`（数据类 ↔ ROS2 消息适配） | Ch10 |
| `apps/teleop.py` | `Sample`/`JointSample`/`TcpSample`、`Recorder`/`Player`/`TeleopLoop`、`IdentityMapping`/`ScaledJointMapping`/`CartesianMapping`/`VrControllerMapping`、`Source`/`Sink` | Ch12 |
| `apps/vision.py` | `intrinsic_calibrate`、`hand_eye_calibrate`、`Detector`、`PoseEstimator` | Ch14 |
| `arm_nodes/` | 机器人 → ROS2 节点封装 | Ch10 |

### `chapt/` 与 `test/`

- `chapt/chapt2_T_demo.py`、`chapt2_pose_demo.py`：PySide6 + matplotlib GUI 示例（齐次变换 / 位姿表示），独立运行、不复用 SDK。
- `test/`：`conftest.py`（fixtures）+ `test_init.py`（导入烟测）/`test_interfaces.py`（`ArmProtocol` 契约）/`test_transforms.py`（23 变换）/`test_types.py`（类型与裁剪）。

## 三、函数和类

### 设备模型（`arms/`）继承链

```
JoyArmRebotDM ──▶ Arm（持有 backend_mas + backend_end）
        │
   绑两个 backend 类（mas + end）
   arm.fkine()/command()/end_open()/end_close()
```

- **`Arm`**：完整臂基类。持有 pinocchio `model`/`data` + 软硬限位 + `backend_mas` + `backend_end`；对外提供本体运动学/动力学/安全**门面**（薄委托 `robotics`/`safety`，默认 pinocchio）与执行类方法（`get_state`/`command`，连后可用），并直接提供末端夹爪语义（`end_open`/`end_close`/`set_end_position`/`set_end_force`）；`connect()` 同时连接本体 + 末端。
- **`JoyArmRebotDM(Arm)`**：具体型号预设。无参即用（自动解析随包 URDF + `configs/joyarm_rebot_dm.yaml`），绑定 `backend_mas_cls=BackendMasRebotDM`、`backend_end_cls=BackendEndJoyGripper`。

### Backend 三层（`backends/`）继承链

```
Backend(ABC) ──┬─▶ BackendMas ──▶ BackendMasRebotDM   （本体：关节电机，Ch6 占位）
               └─▶ BackendEnd ──▶ BackendEndJoyGripper（末端：夹爪，Ch13 占位）
```

按「硬件类型」再按「型号」派生；本体后端属性 `backend_mas`、末端后端属性 `backend_end`，均由 `configs/<model>.yaml` 的 `backend_mas`/`backend_end` 段实例化。

### 算法层（`robotics/` / `safety/`）

模块级函数，形参 `arm`（按 `ArmProtocol` 访问），**不 import `arms`**、不带 `method` 参数、默认 pinocchio。仅 `fkine` 已实现，其余为占位签名。

### 基础层（`utils/`）

- `transforms.py`：纯 numpy SO(3)/SE(3) 数学，无第三方依赖。
- `types.py`：跨层共享 `@dataclass` 快照 + 枚举（继承 `_ArrayEqMixin`，ndarray 字段安全比较）。
- `interfaces.py`：`ArmProtocol`——算法层对臂对象的最小契约（结构化鸭子类型）。

## 四、API 参考

> 以下为**精简签名**：`name(关键参数) → 返回 — 一句话`。完整参数 / 返回 / 异常说明见**源码 docstring** 或 `help(符号)`。✅ 已实现 / 🟡 占位。

### 设备模型 API

```python
Arm(urdf_path, ee_frame_name="ee", mesh_dirs=None, load_geometry=False, name="Arm", config=None)
    # 完整臂基类：建 pin model + 软硬限位 + backend_mas + backend_end（config 驱动）✅
# 类属性：backend_mas_cls / backend_end_cls（子类绑定具体型号后端）
# 本体方法
Arm.connect() / disconnect()                          # 连接 / 断开（本体 + 末端）✅
Arm.rand_q(size=None, rng=None) → ndarray             # 软限位内采样 (n,) 或 (N,n) ✅
Arm.clamp_q(q) → ndarray                              # 裁剪到软限位 ✅
Arm.is_q_valid(q) → bool                              # 是否在软限位内 ✅
Arm.frame_placement(q, frame=None) → (4,4)            # 底层单次 FK（含 T_base 偏移）✅
Arm.fkine(q, frame=None, rep="T")                     # 正运动学门面 ✅
Arm.ikine(T_target, q0=None, frame=None, **kw)        # 逆运动学门面 🟡
Arm.jac(q, frame=None, ref="local")                   # 雅可比门面 🟡
Arm.fdyn(q, dq, tau, **kw) / idyn(q, dq, ddq, **kw)   # 动力学门面 🟡
Arm.mass_matrix(q) / coriolis(q, dq) / gravity(q)     # 动力学项门面 🟡
Arm.cartesian_inertia(q, frame=None)                  # 笛卡尔惯量门面 🟡
Arm.check_joint_limits(state) / check_tcp_limits(state)  # 安全校验门面 🟡
Arm.get_state() → ArmState                            # 读状态（需 connect）✅
Arm.command(q=None, dq=None, tau=None, kp=None, kd=None, mode=ControlMode.POSITION)  # 下发指令（需 connect）✅
# 末端方法（需 connect）
Arm.end_open() / end_close()                          # 夹爪开/合 ✅
Arm.set_end_position(position) / set_end_force(force) # 末端位置/力 🟡
Arm.get_end_state() → dict                            # 末端状态 🟡
# 关键属性：model / data / n / nv / ee_frame_name / ee_frame_id / joint_limits / joint_limits_soft
#           / qlow / qhigh / q_neutral / tcp_limits / T_base / backend_mas / backend_end / connected

JoyArmRebotDM(urdf_path=None, ee_frame_name=None, mesh_dirs=None, load_geometry=False, name="JoyArmRebotDM")
    # 无参即用 ✅；属性 mdh_table / q_home；绑两个 backend 类
```

### Backend 三层 API

```python
Backend(ABC): connect() / disconnect() / read_state()                              # 通用根 ✅
BackendMas(Backend): enable/disable(joint=None) · set_zero(joint=None) · scan() → list[int]
    · read_state() → ArmState · send_position(q, joint=None) · send_velocity(dq, joint=None)
    · send_torque(tau, joint=None) · send_mit(q, dq, tau_ff, kp, kd, joint=None)   # 本体抽象 ✅
BackendEnd(Backend): read_state() → dict · send_position(position) · send_force(force) · send_action(action)  # 末端抽象 ✅
BackendMasRebotDM(BackendMas) / BackendEndJoyGripper(BackendEnd)                   # 具体型号，占位 🟡
```

### Robotics 算法 API（形参 `arm`，满足 `ArmProtocol`）

```python
fkine(arm, q, frame=None, rep="T")                                  # 正运动学（形状重载）✅
ikine(arm, T_target, q0=None, frame=None, **kw) → IKResult          # 逆运动学 🟡
ikine_constrained(...)                                              # 约束 IK 🟡
jac(arm, q, frame=None, ref="local")                                # 雅可比 🟡
manipulability(...) / cond_number(...) / statics(...)               # 性能指标 🟡
# 轨迹 🟡：Trajectory · joint_cubic / joint_quintic / joint_lspb / joint_waypoints
#          · cart_line / cart_arc / cart_to_joint · constant_velocity_retime · validate
# 动力学 🟡：fdyn / idyn / mass_matrix / coriolis / gravity / cartesian_inertia
# 控制 🟡：ControlLoop · play_trajectory · joint_position_control / joint_velocity_control
#          · torque_control · arm_position_control / arm_velocity_control
#          · computed_torque_control / inverse_dynamics_control · ForceTorqueSensor / JointTorqueSensor
#          · pure_force_control · HybridForcePosition · ImpedanceControl / AdmittanceControl
#          · compute_cartesian_impedance
```

### Safety API（`safety/`，占位 🟡）

```python
joint_limits_check(state, limits) / tcp_limits_check(state, limits)   # 限位校验纯函数 🟡
StateMonitor / SelfCollisionChecker / ExternalCollisionDetector / SafetySupervisor   # 监控器族 🟡
```

### Utils 数学 API（`transforms.py`，纯 numpy，全部 ✅）

```python
# 基本旋转：(3,3)
rot_x(angle) / rot_y(angle) / rot_z(angle)
# RPY ↔ R
rpy_to_R(rpy) → (3,3)          R_to_rpy(R) → (3,)              # R=Rz(y)·Ry(p)·Rx(r)
# 轴角 ↔ R
rodrigues(k, theta=None) → (3,3)   axis_angle_to_R(k, theta) → (3,3)   R_to_axis_angle(R) → (k, theta)
# 四元数 (w,x,y,z) ↔ R / 轴角 / RPY
quat_to_R(q) → (3,3)   R_to_quat(R) → (4,)   axis_angle_to_quat(k, theta) → (4,)
quat_to_axis_angle(q) → (k, theta)   rpy_to_quat(rpy) → (4,)   quat_to_rpy(q) → (3,)
quat_mul(q1, q2) → (4,)   quat_conj(q) → (4,)   quat_norm(q) → (4,)
# 齐次变换 (4,4)
Rp_to_T(R=None, p=None) → (4,4)   T_to_Rp(T) → (R, p)   T_inv(T) → (4,4)   T_mul(T1, T2) → (4,4)   adT(T) → (6,6)
# 插值
slerp(R0, R1, s) → (3,3)
```

### 类型 API（`types.py`，全部 ✅）

```python
# 枚举
ControlMode      # POSITION / VELOCITY / TORQUE / MIT
Severity         # INFO / WARNING / ERROR / CRITICAL
SafetyAction     # NONE / CLAMP / DAMPING_HOLD / FREEZE / ESTOP
TrajectorySpace  # JOINT / CARTESIAN
# 数据类（@dataclass，ndarray 安全相等）
Pose(position, orientation)        # from_T(T) → Pose；.T → (4,4)
JointState / TcpState / ArmState   # 状态快照（关节/末端/整机）
JointLimits / TcpLimits            # 限位声明（硬/软两实例）
Wrench(force, torque) / Twist(linear, angular)
Violation / IKResult / ComplianceParams
# 纯函数
clamp_to_limits(targets, limits: JointLimits) → ndarray   # 逐元素裁剪到限位内
```

### 接口 API（`interfaces.py`，✅）

```python
@runtime_checkable
class ArmProtocol(Protocol):
    # 属性：model / data / n / nv / ee_frame_name / ee_frame_id / T_base
    #       joint_limits / joint_limits_soft / tcp_limits / q_neutral
    def frame_placement(self, q, frame=None) -> (4,4): ...   # 算法层依赖的最小本体契约
```

## 五、维护说明

本文档与 [`README.md`](README.md) 为代码库两大入口，须随代码同步维护——具体强制原则与「精简概述、代码即详则」见根目录 [`../AGENTS.md`](../AGENTS.md) 的「代码库文档与注释维护」一节。
