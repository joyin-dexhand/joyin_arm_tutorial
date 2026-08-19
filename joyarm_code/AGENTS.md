# AGENTS.md —— `joyarm_code/` 子库项目记忆

> **定位**：`joyarm_code/` 代码库的维护者 / Agent 内部文档，不进教程站点。
> **依赖链**：根 [`AGENTS.md`](../AGENTS.md)（教程站点）→ 本文件（代码库）→ [`README.md`](README.md)（用户向基础）。
> [`README.md`](README.md)（给使用者）已覆盖：**定位、目录概览、基本架构（分层图与核心概念）、安装、快速开始、离线语义、测试运行、章节落地状态**——本文不重复，只在其上补充维护者向内容：**架构细化、命名与设计约定、模块与 API 速查、文档维护规则**。
> ✅ 本文件正常纳入版本管理（根 `.gitignore` 的 `AGENTS.md` 忽略规则已移除，直接 `git add` 即可）。
>
> **文档原则**：md 只写精简概述（结构速览 / API 速查），不复述代码已有内容；详细规则写进代码注释，查细节看源码 docstring 或 `help(符号)`。

## 1. 架构细化

> 补充 README「基本架构」：分层图与核心概念（`Arm` / Backend 三层 / `ArmProtocol`）见 README，本节给文件级细节。

### 1.1 目录树（文件级）

```
joyarm_code/
├── joyarm/                          # ★ 核心 SDK（ROS2-free）
│   ├── __init__.py                  #   公开 API 扁平导出（__all__）
│   ├── utils/                       #   基础层
│   │   ├── types.py                 #     数据类 / 枚举
│   │   ├── transforms.py            #     SO(3)/SE(3) 纯 numpy 数学
│   │   └── interfaces.py            #     ArmProtocol（算法层接口契约）
│   ├── robotics/                    #   算法层：fkine · ikine · jacobian · trajectory · dynamics · control
│   ├── safety/                      #   安全层（按监控层级分文件，Ch11）
│   │   ├── joint.py                 #     关节层：clamp_to_limits + joint_limits_check
│   │   ├── tcp.py                   #     末端层：tcp_limits_check
│   │   ├── machine.py               #     整机层：自碰撞检测
│   │   ├── external.py              #     外部层：外部碰撞检测
│   │   └── supervisor.py            #     跨层：StateMonitor + SafetySupervisor
│   ├── backends/                    #   通信层（三层 Backend）
│   │   ├── backend.py               #     Backend（通用根）
│   │   ├── backend_mas.py           #     BackendMas（本体抽象）
│   │   ├── backend_end.py           #     BackendEnd（末端抽象）
│   │   ├── backend_mas_rebot_dm.py  #     BackendMasRebotDM（reBot-DM 本体，占位）
│   │   └── backend_end_joygripper.py#     BackendEndJoyGripper（Joy 夹爪，占位）
│   ├── arms/                        #   设备模型层
│   │   ├── arm.py                   #     Arm 基类（本体+末端，持有两个后端）
│   │   └── joyarm_rebot_dm.py       #     JoyArmRebotDM(Arm)
│   ├── robots/                      #   URDF + meshes 资产（运行期加载）
│   └── configs/
│       └── joyarm_rebot_dm.yaml     #   per-model YAML（backend_mas/backend_end 分段）
├── joyarm_ros2_ws/                  # ROS2 colcon 工作空间（规划，Ch10 落地时创建；见 §1.4）
├── chapt/                           # 章节教学示例脚本（一次性，不复用 SDK）
├── test/                            # pytest 测试套件
├── quickstart/                      # 快速上手（CLI/GUI/任务编排，占位）
├── pyproject.toml                   # 工程配置（可编辑安装 joyarm；ws 不在此打包）
├── README.md                        # 用户向入口（定位/架构/安装/用法）
└── AGENTS.md                        # 本文件（维护者/Agent 项目记忆）
```

### 1.2 依赖规则（无环明细）

- `utils` → 仅 numpy；
- `robotics` / `safety` / `backends` → 仅依赖 `utils`；算法形参 `arm` 按 `ArmProtocol` 访问，**不 import `arms`**（依赖倒置）；
- `arms` → 门面委托 `robotics`/`safety`/`backends`，运行期加载 `robots/`、`configs/`；
- `joyarm_ros2_ws/src/*`（规划）→ 依赖 `joyarm`（pip 提供）+ `rclpy`（ROS 环境提供）；核心包保持 ROS2-free。

### 1.3 配置（yaml 分段）

`configs/<model>.yaml` 按 backend 分段（另含 urdf / ee_frame / q_home / mdh / joint_limits / tcp_limits；未装 PyYAML 时回退类常量）：

```yaml
backend_mas: {can_interface: can0, baudrate: 1000000, motor_ids: [1,2,3,4,5,6]}
backend_end: {can_interface: can0, baudrate: 1000000, id: 7}
```

### 1.4 ROS2 工作空间（`joyarm_ros2_ws/`，**规划——Ch10 落地时创建，当前未建**）

标准 colcon 工作空间，承载所有 ROS2 功能包（第十章起）；`build/ install/ log/` 为构建产物（.gitignore 已含忽略规则）。规划目录结构：

```
joyarm_ros2_ws/
└── src/
    └── joyarm_node/            # Ch10：机械臂节点功能包（ament_python）
        ├── package.xml / setup.py / setup.cfg / resource/   # setup.py 须含 tests_require=["pytest"]
        ├── joyarm_node/
        │   ├── arm_node.py     #   ArmNode：每机械臂实例一个节点（话题/服务，离线可跑）
        │   └── adapters.py     #   内部类型 ↔ ROS2 标准消息互转（四元数 wxyz↔xyzw 换序）
        ├── config/joyarm_rebot_dm.yaml   # ROS 参数（model/rate/offline；型号细节在核心库 YAML 单一来源）
        ├── launch/arm.launch.py          # ArmNode + robot_state_publisher(+rviz2)，namespace 即多臂
        └── test/test_adapters.py
```

**接口约定**（标准消息起步）：状态广播走话题（`joint_states` 相对名对齐 robot_state_publisher、`~/tcp_pose`）；连接/夹爪走服务（`~/connect`、`~/disconnect`、`~/set_end`，std_srvs）；单点位置指令走 `~/command` 订阅；**轨迹执行规划为 action**，待自定义接口包 `joyarm_interfaces`（ament_cmake，含 ControlMode/轨迹 action 等）随需要建立。规划中的应用包：`joyarm_teleop`(Ch12)、`joyarm_vision`(Ch14)、`joyarm_agent`(Ch15，语音/NFC/UWB/智能体)；纯算法（teleop/vision 管线）规划放 `joyarm/apps/`（ROS2-free），落地时再建。

**环境与构建**（教程第十章环境节同此；与 ROS2 生态习惯一致）：

```bash
python3 -m pip install --user -e joyarm_code     # 一次性：joyarm 装入 ~/.local（勿 sudo pip）
source /opt/ros/humble/setup.bash
cd joyarm_code/joyarm_ros2_ws && colcon build    # 系统 colcon：构建/运行解释器全链路一致
source install/setup.bash
ros2 launch joyarm_node arm.launch.py
```

> ⚠️ 边界：`joyarm_node` 定位"教 ROS2 概念 + 支撑应用篇"，**不做** ros2_control hardware_interface、不追求直接兼容 MoveIt2（那是 ros2_control 路线，教程如需再议）。
> ⚠️ 已知陷阱①：package.xml 的 XML **注释中禁止出现双连字符 `--`**（如 `--user`）——否则整份文件非法，catkin_pkg 解析失败后 colcon 会**静默回退**到普通 python task（仍"构建成功"，但缺 ament 索引 hook，`ros2 run/launch` 找不到包）。
> ⚠️ 已知陷阱②：ament_python 包的 `setup.py` 必须声明 `tests_require=["pytest"]`（或 `extras_require={"test": [...]}`），否则 `colcon test` 回退 unittest 发现 **0 个用例**。

## 2. 命名与设计约定

### 2.1 命名约定（全仓库强制，含 `chapt/` 章节脚本）

- **类名 = 驼峰（PascalCase）；文件名 = 小写 snake_case。**
- **设备模型**（`arms/`）：`Arm` = 完整机械臂基类（多轴本体 + 末端执行器），直接持有 `backend_mas`/`backend_end` 两个后端；具体型号 `JoyArmRebotDM(Arm)` 在子类绑定 `backend_mas_cls`/`backend_end_cls` 后端**类**。文件 `arms/arm.py`、`arms/joyarm_rebot_dm.py`。
- **Backend 三层**（`Backend` 前缀，`backends/`）：根 `Backend`（通用硬件通信抽象）→ 类型层 `BackendMas`/`BackendEnd`（按硬件类型派生）→ 型号层 `BackendMasRebotDM`/`BackendEndJoyGripper`（按具体型号派生）。文件 `backend.py` / `backend_mas.py` / `backend_end.py` / `backend_mas_rebot_dm.py` / `backend_end_joygripper.py`。
- **属性 / 形参**：本体后端 = `backend_mas`、末端后端 = `backend_end`（均由 `Arm` 直接持有）；robotics/safety 算法形参用 `arm`（依赖 `utils.ArmProtocol`）。
- **执行类方法命名**：本体方法带 `mas`、末端方法带 `end`，两者对应——本体 `get_mas_state` / `set_mas_command`；末端 `end_open` / `end_close` / `set_end_position` / `set_end_force` / `get_end_state`。
- **yaml**：`configs/<model>.yaml` 按 backend 分段——`backend_mas:` / `backend_end:`；子类绑 backend **类**，基类按 config 段实例化。
- **算法默认 pinocchio**：robotics 不带 `method` 参数，默认走 urdf+pin；手写实现请在 `Arm` 子类覆盖对应方法（FK 特例：覆盖 `frame_placement`——基本能力/协议唯一方法，fkine/ikine/自碰撞统一消费）。
- 约定针对**系统组件**（arm/backend）；项目品牌名 `joyarm`/`joyarm_code` 不改。

### 2.2 设计原则与编码约定

| 原则 | 落地 |
|:--|:--|
| **核心/ROS2 分层** | 核心 `joyarm/`（ROS2-free）+ colcon 工作空间 `joyarm_ros2_ws/src/`（rclpy，规划见 §1.4）；纯算法支撑层与 ws 薄封装分层同前述规划 |
| **设备模型** | `Arm` 基类（本体+末端）直接持有两个后端；`arm.fkine()` 运动学门面，`arm.end_open()` 操作末端 |
| **Backend 三层** | `Backend` → `BackendMas/BackendEnd` → 具体型号；按硬件类型、再按型号派生 |
| **单向导入、无环** | `arms` import `robotics/safety/backends`；后者不 import `arms`，只依赖 `utils.ArmProtocol`（依赖倒置） |
| **backend 由 yaml 驱动** | 子类绑 backend **类**，基类按 `configs/<model>.yaml` 段实例化 |
| **离线模式** | 默认 `connected=False`（计算类可用）；`connect()` 后才可执行类操作（语义详见 README「快速开始」）；ROS2 节点同语义（`offline:=true` 时指令等效执行） |
| **共享类型单定义** | 跨层数据类型只在 `utils/types.py` 定义一次；ROS2 消息仅在 ws 的 `adapters.py` 做互转（含四元数 wxyz↔xyzw 换序） |
| **核心轻依赖** | 核心仅 `numpy`/`pin`/`pyyaml`；`rclpy` 仅 `joyarm_ros2_ws` |

编码约定：`q=(n,)` 或 `(N,n)`（形状重载）、`T=(4,4)`；角度一律弧度；FK/IK 返回 numpy（进阶 `rep="se3"`）；软/硬限位分级；`ControlMode` 四态（POSITION/VELOCITY/TORQUE/MIT）；可视化不在核心包（rviz2 在 `joyarm_ros2_ws` 的 launch 中启动）。

### 2.3 库边界

SDK（`arm.xx`）→ `joyarm/`；ROS2 节点/launch → `joyarm_ros2_ws/src/`（规划）；CLI/GUI/任务编排 → `joyarm_code/quickstart/`；教学脚本 → `joyarm_code/chapt/`。

## 3. 模块与 API 速查

### 3.1 模块导览

`joyarm/`（核心 SDK）——✅ 已实现 / 🟡 占位：

| 文件 | 类 / 关键定义 | 章节 | 状态 |
|:---|:---|:---:|:---:|
| `utils/types.py` | 枚举 `ControlMode`/`Severity`/`SafetyAction`/`TrajectorySpace`；数据类 `Pose`/`JointState`/`TcpState`/`ArmState`/`JointLimits`/`TcpLimits`/`Wrench`/`Twist`/`Violation`/`IKResult`/`ComplianceParams` | Ch2 | ✅ |
| `utils/transforms.py` | 23 个纯 numpy 函数：`rot_x/y/z`、`rpy_to_R`/`R_to_rpy`、`rodrigues`/`axis_angle_to_R`/`R_to_axis_angle`、`quat_*`、`Rp_to_T`/`T_to_Rp`/`T_inv`/`T_mul`/`adT`、`slerp` | Ch2 | ✅ |
| `utils/interfaces.py` | `ArmProtocol`（`@runtime_checkable Protocol`） | Ch2 | ✅ |
| `backends/backend.py` | `Backend(ABC)`：`connect`/`disconnect`/`read_state` | Ch2 | ✅ |
| `backends/backend_mas.py` | `BackendMas(Backend)`：`enable`/`disable`/`set_zero`/`scan`/`send_position`/`send_velocity`/`send_torque`/`send_mit` | Ch2 | ✅ |
| `backends/backend_end.py` | `BackendEnd(Backend)`：`send_position`/`send_force`/`send_action` | Ch2 | ✅ |
| `backends/backend_mas_rebot_dm.py` | `BackendMasRebotDM(BackendMas)`（CAN，6 电机） | Ch6 | 🟡 |
| `backends/backend_end_joygripper.py` | `BackendEndJoyGripper(BackendEnd)`（CAN，2 指夹爪） | Ch13 | 🟡 |
| `robotics/fkine.py` | `fkine`（+ `_fkine_single`/`_fkine_batch` 内部） | Ch2 | ✅ |
| `robotics/ikine.py` 等 | 逆运动学 / 雅可比 / 轨迹 / 动力学 / 控制函数族 | Ch3-9 | 🟡 |
| `safety/joint.py` | `clamp_to_limits`（关节裁剪 ✅）+ `joint_limits_check`（🟡） | Ch11 | ✅/🟡 |
| `safety/`（tcp / machine / external / supervisor） | `tcp_limits_check` · `CollisionReport`+`SelfCollisionChecker` · `ExternalCollisionDetector` · `StateMonitor`+`SafetySupervisor`（均 🟡） | Ch11 | 🟡 |
| `arms/arm.py` | `Arm`（本体+末端基类） | Ch2 | ✅ |
| `arms/joyarm_rebot_dm.py` | `JoyArmRebotDM(Arm)` | Ch2 | ✅ |
| `configs/joyarm_rebot_dm.yaml` | 型号 YAML（限位 / home / backend 分段） | Ch2/7 | ✅ |

`joyarm/apps/` 与 `joyarm_ros2_ws/src/joyarm_node/`：**规划落点，当前未建**（接口与结构规划见 §1.4；teleop/vision 纯算法 + joyarm_node 节点封装，Ch10/12/14 落地时创建）。

`chapt/` 与 `test/`：

- `chapt/`：`chapt2_T_demo.py`、`chapt2_pose_demo.py`（PySide6 + matplotlib GUI，独立运行、不复用 SDK）；`chapt10_ros2_demo.py`（joyarm_node 最小 rclpy 客户端）随 §1.4 落地时创建。
- `test/`：`conftest.py`（fixtures）+ `test_init.py`（导入烟测）/ `test_interfaces.py`（`ArmProtocol` 契约）/ `test_transforms.py`（23 变换）/ `test_types.py`（类型）/ `test_safety.py`（安全层：裁剪 + 包契约）。

### 3.2 继承链与职责

```
JoyArmRebotDM ──▶ Arm（持有 backend_mas + backend_end）
        │  绑两个 backend 类（mas + end）；arm.fkine()/set_mas_command()/end_open()/end_close()
Backend(ABC) ──┬─▶ BackendMas ──▶ BackendMasRebotDM   （本体：关节电机，Ch6 占位）
               └─▶ BackendEnd ──▶ BackendEndJoyGripper（末端：夹爪，Ch13 占位）
```

- **`Arm`**：完整臂基类。持有 pinocchio `model`/`data` + 软硬限位 + `backend_mas` + `backend_end`；对外提供本体运动学/动力学/安全**门面**（薄委托 `robotics`/`safety`，默认 pinocchio）与执行类方法（`get_mas_state`/`set_mas_command`），并直接提供末端夹爪语义（`end_*`）；`connect()` 同时连接本体 + 末端。
- **`JoyArmRebotDM(Arm)`**：具体型号预设。无参即用（自动解析随包 URDF + `configs/joyarm_rebot_dm.yaml`）。
- **robotics / safety**：模块级函数，形参 `arm`（按 `ArmProtocol` 访问），不 import `arms`、无 `method` 参数、默认 pinocchio。

### 3.3 API 速查

> **精简签名**：`name(关键参数) → 返回 — 一句话`。完整参数 / 返回 / 异常说明见**源码 docstring** 或 `help(符号)`。

#### 设备模型 API

```python
Arm(name, urdf_path, ee_frame_name="ee", mesh_dirs=None, load_geometry=False, config=None)
    # 完整臂基类：建 pin model + 软硬限位 + backend_mas + backend_end（config 驱动）✅
# 类属性：backend_mas_cls / backend_end_cls（子类绑定具体型号后端）
# 本体方法
Arm.connect() / disconnect()                          # 连接 / 断开（本体 + 末端）✅
Arm.rand_q(size=None, rng=None) → ndarray             # 软限位内采样 (n,) 或 (N,n) ✅
Arm.clamp_q(q) → ndarray                              # 裁剪到软限位 ✅
Arm.is_q_valid(q) → bool                              # 是否在软限位内 ✅
Arm.frame_placement(q, frame=None) → (4,4)            # 底层单次 FK（含 T_base 偏移）✅
Arm.fkine(q, frame=None, rep="T")                     # 正运动学门面 ✅
Arm.ikine(T_target, q0=None, ...) / Arm.jac(q, ref="local")    # IK / 雅可比门面 🟡
Arm.fdyn / idyn / mass_matrix / coriolis / gravity / cartesian_inertia   # 动力学门面 🟡
Arm.check_joint_limits(state) / check_tcp_limits(state)        # 安全校验门面 🟡
Arm.get_mas_state() → ArmState                        # 读本体状态（需 connect）✅
Arm.set_mas_command(mode=ControlMode.POSITION, q=None, dq=None, tau=None, kp=None, kd=None)  # 下发指令（需 connect）✅
# 末端方法（需 connect）
Arm.end_open() / end_close()                          # 夹爪开/合 ✅
Arm.set_end_position(position) / set_end_force(force) # 末端位置/力 🟡
Arm.get_end_state() → dict                            # 末端状态 🟡
# 关键属性：model / data / n / nv / ee_frame_name / ee_frame_id / joint_limits(_soft) / qlow / qhigh / q_neutral / tcp_limits / T_base / backend_mas / backend_end / connected

JoyArmRebotDM(name="JoyArmRebotDM", urdf_path=None, ee_frame_name=None, mesh_dirs=None, load_geometry=False)
    # 无参即用 ✅；属性 mdh_table / q_home；绑两个 backend 类
```

#### Backend 三层 API

```python
Backend(ABC): connect() / disconnect() / read_state()                     # 通用根 ✅
BackendMas(Backend): enable/disable · set_zero · scan · read_state() → ArmState
    · send_position(q) / send_velocity(dq) / send_torque(tau) / send_mit(q, dq, tau, kp, kd)   # 本体 ✅
BackendEnd(Backend): read_state() → dict · send_position / send_force / send_action          # 末端 ✅
BackendMasRebotDM(BackendMas) / BackendEndJoyGripper(BackendEnd)          # 具体型号 🟡
```

#### Robotics 算法 API（形参 `arm`，满足 `ArmProtocol`；除 fkine ✅ 外均 🟡 占位）

```python
fkine(arm, q, frame=None, rep="T")                     # 正运动学（形状重载）✅
ikine(arm, T_target, q0=None, ...) → IKResult · ikine_constrained(...)    # 逆运动学 🟡
jac(arm, q, frame=None, ref="local") · manipulability / cond_number / statics   # 雅可比 / 性能指标 🟡
# 轨迹 🟡：Trajectory · joint_cubic/quintic/lspb/waypoints · cart_line/arc · cart_to_joint · constant_velocity_retime · validate
# 动力学 🟡：fdyn / idyn / mass_matrix / coriolis / gravity / cartesian_inertia
# 控制 🟡：ControlLoop · play_trajectory · joint_position/velocity_control · arm_position/velocity_control
#        · torque_control · computed_torque_control · inverse_dynamics_control · pure_force_control
#        · HybridForcePosition · ImpedanceControl / AdmittanceControl · ForceTorqueSensor / JointTorqueSensor
```

#### Safety API（`safety/`，按层分文件；除 clamp ✅ 外占位 🟡）

```python
clamp_to_limits(targets, limits: JointLimits) → ndarray   # 关节裁剪（joint.py）✅
joint_limits_check(state, limits) / tcp_limits_check(state, limits)   # 限位校验纯函数 🟡
StateMonitor / SelfCollisionChecker / ExternalCollisionDetector / SafetySupervisor   # 监控器族 🟡
```

#### Utils 数学 API（`transforms.py`，纯 numpy，全部 ✅）

```python
rot_x / rot_y / rot_z(angle) → (3,3)                  # 基本旋转
rpy_to_R(rpy) ↔ R_to_rpy(R)                           # RPY（R = Rz·Ry·Rx）
rodrigues(k, θ) / axis_angle_to_R ↔ R_to_axis_angle   # 轴角 ↔ R
quat_to_R ↔ R_to_quat · rpy_to_quat / quat_to_rpy · axis_angle_to_quat / quat_to_axis_angle   # 四元数 (w,x,y,z)
quat_mul / quat_conj / quat_norm                       # 四元数运算
Rp_to_T ↔ T_to_Rp · T_inv · T_mul · adT(T) → (6,6)     # 齐次变换 (4,4)
slerp(R0, R1, s) → (3,3)                               # 球面插值
```

#### 类型与接口 API（`types.py` / `interfaces.py`，全部 ✅）

```python
# 枚举：ControlMode（POSITION/VELOCITY/TORQUE/MIT）· Severity · SafetyAction · TrajectorySpace
# 数据类（@dataclass，ndarray 安全相等）：Pose（from_T(T) / .T 属性）· JointState / TcpState / ArmState（状态快照）
#   · JointLimits / TcpLimits（限位声明，硬/软两实例）· Wrench · Twist · Violation · IKResult · ComplianceParams
ArmProtocol  # @runtime_checkable Protocol——算法层最小本体契约
    # 属性：model / data / n / nv / ee_frame_name / ee_frame_id / T_base / joint_limits(_soft) / tcp_limits / q_neutral
    # 唯一方法：frame_placement(q, frame=None) → (4,4)
```

## 4. 文档同步维护

本文件与 [`README.md`](README.md) 是子库两大入口文档，**受众不同、须随代码同步维护**：README 给**使用者**（定位 / 架构 / 用法），本文件给 **agent / 开发者**（约定 / 速查 / 规则）；新内容按受众分流，不互相重复。

### 文档定位：精简概述、代码即详则

- 两份 md 只写**精简概述**（结构 / 用法 / API 速查），**不复述**代码已有内容；查细节看源码 docstring 或 `help(符号)`。
- **详细规则写进代码注释**：各包 `__init__.py` 写模块职责一句话；类与函数写精简 docstring（`:param:`/`:return:`/`:raises:`/ 章节标记）。

### 同步维护强制原则（每次变更必查）

> ⚠️ **每次 `joyarm_code/` 发生任何变化后（新增 / 修改 / 删除 `.py`、`.yaml`、目录结构、ws 功能包、章节脚本），必须同步更新本文件**，使其与代码保持一致；README 按受众需要同步：
>
> - 公开 API（`joyarm/__init__.py` 的 `__all__` 导出）变化 → 更新本文件「3.3 API 速查」与「3.1 模块导览」；
> - 目录 / 文件结构变化 → 更新本文件「1.1 目录树」（ws 部分含「1.4」）与 README「目录结构」；
> - 安装 / 用法 / 示例 / 章节落地状态变化 → 更新 README 对应小节；
> - 架构 / 约定变化 → 更新本文件 §1-2 与 README「基本架构」中的相应概念。
>
> 提交前自检：新增的公开符号已收录、且已补精简 docstring；删除的符号已从文档移除；文件清单与目录树一致。

### 行数上限：≤ 300 行

> 本文件硬性上限 **300 行**（`wc -l` 实测）。超出时**先精简再提交**：优先压缩 §3 速查——只留索引级签名，细节让位于源码 docstring；其次合并 §1-2 中的重复表述。禁止以"另开新节 / 另存新文件"绕过上限；新增内容与精简**同步进行**，只增不减必然超限。
