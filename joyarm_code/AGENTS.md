# AGENTS.md —— `joyarm_code/` 子库项目记忆

> **定位**：`joyarm_code/` 代码库的维护者 / Agent 内部文档，不进教程站点。
> **依赖链**：根 [`AGENTS.md`](../AGENTS.md)（教程站点）→ 本文件（代码库）→ [`README.md`](README.md)（用户向基础）。
> [`README.md`](README.md)（给使用者）已覆盖：**定位、目录概览、基本架构（分层图与核心概念）、安装、快速开始、离线语义、测试运行、章节落地状态**——本文不重复，只在其上补充维护者向内容：**架构细化、命名与设计约定、模块与 API 速查、文档维护规则**。
> ✅ 本文件正常纳入版本管理（根 `.gitignore` 的 `AGENTS.md` 忽略规则已移除，直接 `git add` 即可）。
>
> **文档原则**：md 只写精简概述（结构速览 / API 速查），不复述代码已有内容；详细规则写进代码注释，查细节看源码 docstring 或 `help(符号)`。

## 1. 架构细化

> 补充 README「基本架构」：分层图与核心概念（`JoyArm` / 求解器策略族 / Backend 整机两层）见 README，本节给文件级细节。

### 1.1 目录树（文件级）

```
joyarm_code/
├── joyarm_core/                          # ★ 核心 SDK（ROS2-free）
│   ├── __init__.py                  #   公开 API 扁平导出（__all__）
│   ├── utils/                       #   基础层
│   │   ├── types.py                 #     数据类 / 枚举
│   │   ├── transforms.py            #     SO(3)/SE(3) 纯 numpy 数学
│   │   └── limits.py                #     clamp_to_limits（指令路径限位守卫）
│   ├── robotics/                    #   算法层：一域一子包（策略 ABC + 实现 + REGISTRY + 函数式入口）
│   │   ├── fkine/                   #     FkineSolver · PinFkineSolver(✅) · MdhFkineSolver(🟡)
│   │   ├── ikine/                   #     IkineSolver · PinIkineSolver · AnalyticIkine6R（均 🟡）
│   │   ├── jacobian/                #     JacobianSolver · PinJacobianSolver · GeometricJacobianSolver（均 🟡）
│   │   ├── trajectory/              #     TrajPlanner · DefaultTrajPlanner · ToppraTrajPlanner(🟡) + segments.py 纯函数族
│   │   ├── dynamics/                #     DynamicsSolver · PinDynamicsSolver · LagrangianDynamicsSolver（均 🟡）
│   │   └── control/                 #     Controller · PositionController + kinematic/dynamics_based/force 三部
│   │      （监测已裁撤：指令守卫在 utils/limits.py；状态监测/日志归 ROS2 节点，Ch11）
│   ├── backends/                    #   通信层（整机 Backend + name 选型）
│   │   ├── backend.py               #     Backend（整机 ABC：*_arm/*_end + 参数读写方法族）
│   │   ├── backend_dm.py            #     BackendDM（DM 整机 7 电机；内含私有协议层 DmMotor/DmCanBus）✅
│   │   ├── u2can/                   #     厂商 DM 参考库（协议参照用，不依赖不导入）
│   │   └── __init__.py              #     REGISTRY + get_backend(name)
│   ├── joyarms/                     #   设备模型层（组合根 + 型号工厂）
│   │   ├── joyarm.py                #     JoyArm：策略成员组装 + 公开门面 + load_config
│   │   ├── joyarm_dm.py             #     JoyArmDM(JoyArm)
│   │   └── __init__.py              #     REGISTRY + JoyArmFactory/joyarm_factory（型号名选型）
│   ├── robots/                      #   URDF + meshes 资产（运行期加载）
│   └── configs/joyarm_dm.yaml       #   per-model YAML（solvers/backend 段，name 选型）
├── joyarm_ros2_ws/                  # ROS2 colcon 工作空间（规划，Ch10 落地时创建；见 §1.4）
├── chapt/                           # 章节教学示例脚本（一次性，不复用 SDK）
├── test/                            # 测试套件
├── quickstart/                      # 快速上手（CLI/GUI/任务编排，占位）
├── pyproject.toml                   # 工程配置（可编辑安装 joyarm_core；ws 不在此打包）
├── README.md                        # 用户向入口（定位/架构/安装/用法）
└── AGENTS.md                        # 本文件（维护者/Agent 项目记忆）
```

### 1.2 依赖规则（无环明细）

- `utils` → 仅 numpy；
- `robotics` / `backends` → 仅依赖 `utils`；求解器鸭子类型消费 `arm`（经公开门面/属性），**不 import `joyarms`**（组合根单向向下，架构约定）；
- `joyarms` → **组合根**：门面委托策略成员（`robotics`）与整机后端（`backends`），构造时按 config `solvers:` 段查各域 `REGISTRY` 组装成员、按 `backend:` 段 `name` 经 `get_backend` 构建后端，运行期加载 `robots/`、`configs/`；
- `joyarm_ros2_ws/src/*`（规划）→ 依赖 `joyarm_core`（pip 提供）+ `rclpy`（ROS 环境提供）；核心包保持 ROS2-free。

### 1.3 配置（yaml 分段）

`configs/<model>.yaml` 按 solvers / backend 分段（另含 name / robot / ee_frame / arm_mdh_and_limits / T_linkn_end / 特征位形 q_zero/q_home/q_neutral / tcp_limits；直用时缺文件回退类常量）。**命名链**：工厂入参 = 文件名 = `name` 字段 = joyarms 注册名（如 `joyarm_dm`），任一不符报『XX』型号在XX中未找到；backend 与机械臂型号 1:1（`backend_dm`）：

```yaml
solvers: {fkine: default, ikine: default, jacobian: default, dynamics: default, traj: default, control: default}  # 也可写注册名（pin/mdh/toppra/…），default=各域默认注册名别名
backend: {name: backend_dm,                 # name 选型键（REGISTRY 解析）
          arm: {channel, protocol, baud_rate, control_rate, joints},   # 本体子段
          end: {同构字段, joints}}           # 末端子段；channel 同 arm → 共享总线
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
        ├── config/joyarm_dm.yaml   # ROS 参数（model/rate/offline；型号细节在核心库 YAML 单一来源）
        ├── launch/arm.launch.py          # ArmNode + robot_state_publisher(+rviz2)，namespace 即多臂
        └── test/test_adapters.py
```

**接口约定**（标准消息起步）：状态广播走话题（`joint_states` 相对名对齐 robot_state_publisher、`~/tcp_pose`）；连接/夹爪走服务（`~/connect`、`~/disconnect`、`~/set_end`，std_srvs）；单点位置指令走 `~/command` 订阅；**轨迹执行规划为 action**，待自定义接口包 `joyarm_interfaces`（ament_cmake，含 ControlMode/轨迹 action 等）随需要建立。规划中的应用包：`joyarm_teleop`(Ch12)、`joyarm_vision`(Ch14)、`joyarm_agent`(Ch15，语音/NFC/UWB/智能体)；纯算法（teleop/vision 管线）规划放 `joyarm_core/apps/`（ROS2-free），落地时再建。

**环境与构建**（教程第十章环境节同此；与 ROS2 生态习惯一致）：

```bash
python3 -m pip install --user -e joyarm_code     # 一次性：joyarm_core 装入 ~/.local（勿 sudo pip）
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
- **设备模型**（`joyarms/`）：`JoyArm` = 完整机械臂基类（多轴本体 + 末端执行器），私有整机后端 `_backend`（config `backend.name` 选型构建，能力全部经公有门面暴露）；具体型号 `JoyArmDM(JoyArm)` 不绑后端类；`joyarms/__init__.py` 的 `joyarm_factory(model)` 按型号名创建（`REGISTRY` 选型 + 命名链校验），**外部推荐入口**。文件 `joyarms/joyarm.py`、`joyarms/joyarm_dm.py`、`joyarms/__init__.py`。
- **Backend 整机两层**（`Backend` 前缀，`backends/`）：根 `Backend`（整机后端 ABC，本体+末端一体，方法以 `*_arm`/`*_end` 后缀区分）→ 型号层 `BackendDM`（与机械臂型号 **1:1** 派生：`backend(_joyarm)_dm` ↔ `joyarm_dm`，结构同参数异）。文件 `backend.py` / `backend_dm.py`；`REGISTRY` + `get_backend(name)` 供 config 选型。
- **属性 / 形参**：整机后端 = `_backend`（私有，yaml `backend.name` 选型，不直接外露）；策略成员 = `_fkine_solver`/`_ikine_solver`/`_jacobian_solver`/`_dynamics_solver`/`_traj_planner`/`_controller`（私有，config 选型）；求解器形参用 `arm`（鸭子类型，经公开门面互调）。
- **执行类方法命名**：本体方法带 `arm`、末端方法带 `end`，两者对应——本体 `enable_arm`/`disable_arm`/`set_zero_arm`/`set_mode_arm(mode=POSITION, joint=None)`/`get_arm_state`/`set_arm_command`；末端 `enable_end`/`disable_end`/`set_zero_end`/`set_mode_end(mode=POSITION, joint=None)`/`end_open(joint=None)`/`end_close(joint=None)`/`end_zero(joint=None)`/`set_end_position`/`set_end_force`/`get_end_state(joint=None)`；参数读写 `read/write_param_arm(key[, value,] joint=None,…)`/`read/write_param_end(同构)`（key 由子类映射厂商寄存器，基类不泄漏 RID）。执行类方法统一带 `joint` 形参（None=全部，逐关节/多电机末端如灵巧手通用）；`set_mode_*` 默认位置模式（电机 POS_VEL）；参数读写 `joint=None` 时读返逐电机列表、写 `value` 可标量广播或等长列表。
- **yaml**：`configs/<model>.yaml` 单 `backend:` 段——`name` 选型键 + `arm:`/`end:` 子段（channel/protocol/baud_rate/control_rate/joints）；JoyArm 按 `name` 经 `REGISTRY` 构建整机后端。
- **算法可换（成员即策略）**：每域 = ABC + 实现们 + `REGISTRY`（一节点一文件）；config `solvers:` 按注册名选型，未知名报错列出可选项；黑盒默认 pin 系，白盒（`Mdh*`/`Geometric*`/`Lagrangian*`）是同 ABC 教学子类；运行期可 `arm._xxx = ...` 或 `arm.set_controller(name)` 换。
- 约定针对**系统组件**（joyarm/backend）；项目品牌名 `JoyArm`/`joyarm_code` 不改，核心 Python 包目录/导入名固定为 `joyarm_core`。

### 2.2 设计原则与编码约定

| 原则 | 落地 |
|:--|:--|
| **核心/ROS2 分层** | 核心 `joyarm_core/`（ROS2-free）+ colcon 工作空间 `joyarm_ros2_ws/src/`（rclpy，规划见 §1.4）；纯算法支撑层与 ws 薄封装分层同前述规划 |
| **组合根** | `JoyArm`（本体+末端）持有一个私有整机后端 `_backend` + 六个策略成员；公开门面（`fkine`/`plan_joint_p2p`/`play_joint`…）全部委托私有成员，换 config 即换算法 |
| **Backend 整机两层** | `Backend`（`*_arm`/`*_end` 方法族）→ 型号子类（结构同参数异）；yaml `name` 选型；arm/end 同 channel 共享总线、异 channel 独立开 |
| **单向导入、无环** | `joyarms` import `robotics`/`backends`；后者不 import `joyarms`（鸭子类型消费 arm，架构约定） |
| **组件由 yaml 驱动** | backend 按 `backend:` 段 `name` 经 REGISTRY 选型；求解器/规划器/控制律按 `solvers:` 段注册名选型（`_build_component` 统一组装） |
| **离线模式** | 默认 `connected=False`（计算类可用）；`connect()` 后才可执行类操作（语义详见 README「快速开始」）；ROS2 节点同语义（`offline:=true` 时指令等效执行） |
| **共享类型单定义** | 跨层数据类型只在 `utils/types.py` 定义一次；ROS2 消息仅在 ws 的 `adapters.py` 做互转（含四元数 wxyz↔xyzw 换序） |
| **核心轻依赖** | 核心仅 `numpy`/`pin`/`pyyaml`；`rclpy` 仅 `joyarm_ros2_ws` |

编码约定：`q=(n,)` 或 `(N,n)`（形状重载）、`T=(4,4)`；角度一律弧度；FK 返回 `rep` 三态（`quat`=Pose / `T`=4×4 / `se3`）、雅可比 `ref` 两态（`local`/`base`）；软/硬限位分级；`ControlMode` 三态（POSITION/VELOCITY/MIT，纯力矩经 MIT `kp=kd=0` 实现）；可视化不在核心包（rviz2 在 `joyarm_ros2_ws` 的 launch 中启动）。

### 2.3 库边界

SDK（`arm.xx`）→ `joyarm_core/`；ROS2 节点/launch → `joyarm_ros2_ws/src/`（规划）；CLI/GUI/任务编排 → `joyarm_code/quickstart/`；教学脚本 → `joyarm_code/chapt/`。

## 3. 模块与 API 速查

### 3.1 模块导览

`joyarm_core/`（核心 SDK）——✅ 已实现 / 🟡 占位：

| 文件 | 类 / 关键定义 | 章节 | 状态 |
|:---|:---|:---:|:---:|
| `utils/types.py` | 枚举 `ControlMode`/`Severity`/`SafetyAction`/`TrajectorySpace`；数据类 `Pose`/`JointState`/`TcpState`/`ArmState`/`JointLimits`/`TcpLimits`/`Wrench`/`Twist`/`Violation`/`IKResult`/`ComplianceParams` | Ch2 | ✅ |
| `utils/transforms.py` | 23 个纯 numpy 函数：`rot_x/y/z`、`rpy_to_R`/`R_to_rpy`、`rodrigues`/`axis_angle_to_R`/`R_to_axis_angle`、`quat_*`、`Rp_to_T`/`T_to_Rp`/`T_inv`/`T_mul`/`adT`、`slerp` | Ch2 | ✅ |
| `utils/limits.py` | `clamp_to_limits`（指令路径限位守卫） | Ch11 | ✅ |
| `backends/backend.py` | `Backend(ABC)` 整机契约：`connect`/`disconnect` + `connected` 属性 / `read_mode_*` 查询 + `*_arm`/`*_end` 方法族（enable/set_zero/set_mode/read_state/send_…/read·write_param_…） | Ch2 | ✅ |
| `backends/backend_dm.py` | `BackendDM(Backend)`（DM 7 电机：joint1~3=4340P、4~6=4310、夹爪=4310；协议参照 u2can 重写，私有 `DmMotor`/`DmCanBus` 协议层） | Ch6/13 | ✅ |
| `robotics/fkine/` | `FkineSolver`(ABC) · `PinFkineSolver`（模板方法：rep/批量在 ABC，内核 `frame_T`）✅ · `MdhFkineSolver` 🟡 | Ch2 | ✅/🟡 |
| `robotics/ikine/` | `IkineSolver` · `PinIkineSolver` · `AnalyticIkine6R`（均 🟡） | Ch3 | 🟡 |
| `robotics/jacobian/` | `JacobianSolver`（衍生量模板）· `PinJacobianSolver` · `GeometricJacobianSolver`（均 🟡） | Ch4 | 🟡 |
| `robotics/trajectory/` | `TrajPlanner` · `DefaultTrajPlanner`（segments 纯函数族分派）🟡 · `ToppraTrajPlanner` 🟡 | Ch5 | 🟡 |
| `robotics/dynamics/` | `DynamicsSolver`（Λ 模板）· `PinDynamicsSolver` · `LagrangianDynamicsSolver`（均 🟡） | Ch8 | 🟡 |
| `robotics/control/` | `Controller` · `PositionController`（均 🟡）+ `kinematic`/`dynamics_based`/`force` 三部函数（🟡） | Ch6/8/9 | 🟡 |
| `joyarms/joyarm.py` | `JoyArm`（本体+末端基类） | Ch2 | ✅ |
| `joyarms/joyarm_dm.py` | `JoyArmDM(JoyArm)` | Ch2 | ✅ |
| `joyarms/__init__.py` | `JoyArmFactory`/`joyarm_factory` + `REGISTRY`（型号名→型号类，命名链校验） | Ch2 | ✅ |
| `configs/joyarm_dm.yaml` | 型号 YAML（限位 / 特征位形 q_zero·q_home·q_neutral / backend 分段） | Ch2/7 | ✅ |

`joyarm_core/apps/`（规划，见 §1.4/§2.3）。

`chapt/` 与 `test/`：

- `chapt/`：`chapt2_T_demo.py`、`chapt2_pose_demo.py`（PySide6 + matplotlib GUI，独立运行、不复用 SDK）；`chapt10_ros2_demo.py`（joyarm_node 最小 rclpy 客户端）随 §1.4 落地时创建。
- `test/`：`test_backend_dm.py`（DM 协议离线单测：编解码纯函数往返、RX 帧分发、指令原语字节、BackendDM 离线约束、错误码语义回归——0=失能正常/1=使能正常/8~E=故障；`python test/test_backend_dm.py` 直跑）；`test_backend_dm_monitor.py`（真机失能监视：不使能电机，每 0.5 s 原地刷新全部关节/夹爪 q/dq/tau、使能/故障/通讯与驱动板/转子温度——状态帧 D6~7，旧固件恒 0 显"—"；电压/电流 DM 协议不提供故不显示，可手动搬动观察反馈；`python test/test_backend_dm_monitor.py [N]`，N 为周期数、缺省无限）；`test_backend_dm_debug.py`（真机分步调试：交互菜单按危险度递增逐电机逐功能测试——连接→读状态→读参数→写参数→设零位→安全使能（MIT 零阻抗 kp=kd=tau=0，防使能瞬间跳动的）→失能→模式切换→发指令（位置指令默认目标=当前 q；末端预设 open=-1/close=3/zero=0 rad，7 号电机）；`python test/test_backend_dm_debug.py`）。

### 3.2 继承链与职责

```
JoyArmDM ──▶ JoyArm（组合根：_backend + 6 个策略成员）
        │  按 config backend.name 选型后端、solvers: 组装成员；arm.fkine()/plan_joint_p2p()/play_joint()/end_open()
        │
FkineSolver ─▶ PinFkineSolver ✅ / MdhFkineSolver 🟡        （各域同构：ABC → 黑盒/白盒）
IkineSolver ─▶ PinIkineSolver / AnalyticIkine6R 🟡 · JacobianSolver ─▶ … · TrajPlanner ─▶ … ·
DynamicsSolver ─▶ … · Controller ─▶ PositionController 🟡
Backend(ABC，整机：*_arm/*_end 方法族) ──▶ BackendDM（达妙 DM 整机 7 电机 ✅）
```

- **`JoyArm`**：组合根。持有 pinocchio `model`/`data` + 软硬限位 + 特征位形（q_zero/q_home/q_neutral，config 驱动）+ 私有整机后端 `_backend` + 六个策略成员（`_fkine_solver` 等，config 选型）；公开门面（`fkine`/`ikine`/`jac`/动力学族/`plan_*`/`play_*`）全部委托成员；执行类方法（上电准备 `enable_*` 族 + `get_arm_state`/`set_arm_command` + 末端 `end_*` 族）；`connect()` 连接整机后端（本体 + 末端）。
- **`JoyArmDM(JoyArm)`**：具体型号预设。无参即用（自动解析随包 URDF + `configs/joyarm_dm.yaml`）；推荐经 `joyarm_factory("joyarm_dm")` 创建（config 注入 + 命名链校验）。
- **robotics 各域**：策略 ABC + 实现们 + `REGISTRY`；求解器鸭子类型消费 `arm`（互调走公开门面，如 ikine 迭代经 `arm.fkine`），不 import `joyarms`；函数式入口（`fkine(arm, q)` 等）保留作教学 API。

### 3.3 API 速查

> **精简签名**：`name(关键参数) → 返回 — 一句话`。完整参数 / 返回 / 异常说明见**源码 docstring** 或 `help(符号)`。

#### 设备模型 API

```python
JoyArm(name, urdf_path, ee_frame_name="ee", mesh_dirs=None, load_geometry=False, config=None)
    # 完整臂基类：建 pin model + 软硬限位 + 特征位形 + _backend（私有，config backend.name 选型）✅
# 本体方法
JoyArm.connect() / disconnect()                          # 连接 / 断开（本体 + 末端）✅
JoyArm.enable_arm(joint) / disable_arm · set_zero_arm · set_mode_arm(mode=POSITION, joint=None)   # 上电准备门面（需 connect）✅
JoyArm.rand_q(size=None, rng=None) → ndarray             # 软限位内采样 (n,) 或 (N,n) ✅
JoyArm.clamp_q(q) → ndarray                              # 裁剪到软限位 ✅
JoyArm.is_q_valid(q) → bool                              # 是否在软限位内 ✅
JoyArm.fkine(q, frame=None, rep="T")                     # 正运动学门面 → _fkine_solver（rep: quat/T/se3）✅
JoyArm.ikine(T_target, q0=None, ...) / JoyArm.ikine_constrained(...)    # IK 门面 🟡
JoyArm.jac(q, ref="local") · manipulability / cond_number / statics    # 雅可比及衍生量门面（ref: local/base）🟡
JoyArm.idyn / mass_matrix / coriolis / gravity / cartesian_inertia   # 动力学门面（无正动力学）🟡
JoyArm.plan_joint_p2p(q0, qf, method) · plan_joint_waypoints(qs, Ts)   # 关节轨迹门面（p2p/多点，C2）🟡
JoyArm.plan_cart_p2p(method="line") · plan_cart_waypoints(poses, Ts)   # 笛卡尔轨迹门面（line/arc，C2）🟡
JoyArm.play_joint(traj, mode, hz) · play_cart(traj, hz)（OSC）· set_controller(name)   # 回放 / 控制律切换 🟡
JoyArm.get_arm_state() → ArmState                        # 读本体状态（需 connect；tcp.pose 本层 fkine 填充）✅
JoyArm.set_arm_command(mode=ControlMode.POSITION, q=None, dq=None, tau=None, kp=None, kd=None)  # 下发指令（需 connect）✅
JoyArm.read_param_arm(key, joint=None) / write_param_arm(key, value, joint=None, persist=False) · read/write_param_end 同构   # 电机参数读写门面（需 connect；joint=None 读返列表、写 value 标量广播/列表）✅
# 末端方法（需 connect）
JoyArm.enable_end(joint) / disable_end · set_zero_end · set_mode_end(mode=POSITION, joint=None)   # 上电准备门面 ✅
JoyArm.end_open(joint=None) / end_close(joint=None) / end_zero(joint=None)   # 夹爪开/合/归零（作用于所选末端电机）✅
JoyArm.set_end_position(position, joint=None) / set_end_force(force, joint=None) / get_end_state(joint=None)   # 末端位置/力/状态 ✅
# 关键属性：model / data / n / nv / ee_frame_name / ee_frame_id / joint_limits(_soft) / qlow / qhigh / q_zero / q_home / q_neutral / tcp_limits / T_base / _backend / connected

JoyArmDM(name="JoyArmDM", urdf_path=None, ee_frame_name=None, mesh_dirs=None, load_geometry=False, config=None)
    # 无参即用 ✅（config 由工厂注入或自动加载）；属性 mdh_table / mdh_limits / T_linkn_end

joyarm_factory(model, **kwargs) → JoyArm · .create(model, **kwargs) · .list_models() → list
    # 型号名唯一参数创建（如 "joyarm_dm"；命名链：入参=configs 文件名=yaml name=joyarms 注册名）✅
```

#### Backend 整机 API

```python
Backend(ABC) 整机后端 ✅：connect() / disconnect() + connected 属性 + read_mode_arm/end(joint=None) 模式查询（本地缓存，一致才非 None）+ *_arm/*_end 方法族（统一 joint=None=全部）+ 参数读写
    _arm 族：enable_arm/disable_arm · set_zero_arm · set_mode_arm(mode=POSITION, joint=None) · read_state_arm(joint=None) → ArmState（指定 joint 时数组长度 1）
            · send_position_arm(q) / send_velocity_arm(dq) / send_mit_arm(q, dq, tau, kp=None, kd=None)
    _end 族（joint=None 全部，多电机末端通用）：enable_end/disable_end · set_zero_end · set_mode_end(mode=POSITION, joint=None) · read_state_end(joint=None) → dict（值为所选电机逐电机序列）
            · send_position_end(pos) / send_force_end(force) / send_action_end(action, joint=None)
    参数族：read/write_param_arm(key[, value,] joint=None, persist=False) · read/write_param_end 同构（joint=None 读返列表、写 value 标量广播/等长列表）
BackendDM(Backend) ✅  # DM 7 电机（4340P×3+4310×4）；协议参照 u2can/ 重写；pyserial 延迟导入；一发一收
get_backend(name) → type · REGISTRY     # config backend.name 选型入口 ✅
```

#### Robotics 算法 API（函数式入口形参 `arm`，委托门面；策略族经 config `solvers:` 选型）

```python
fkine(arm, q, frame=None, rep="T")                     # 正运动学（rep: quat/T/se3，形状重载）✅
# 策略族（JoyArm 成员的实现类，均可程序化注入）：
#   FkineSolver→PinFkineSolver ✅/MdhFkineSolver 🟡 · IkineSolver→PinIkineSolver/AnalyticIkine6R 🟡
#   JacobianSolver→… · DynamicsSolver→… · TrajPlanner→DefaultTrajPlanner/ToppraTrajPlanner 🟡 · Controller→PositionController 🟡
ikine(arm, T_target, q0=None, ...) → IKResult · ikine_constrained(...)    # 逆运动学 🟡
jac(arm, q, frame=None, ref="local") · manipulability / cond_number / statics   # 雅可比 / 性能指标（ref: local/base）🟡
# 轨迹 🟡：Trajectory · joint_cubic/quintic/lspb/waypoints · cart_line/arc/waypoints · cart_to_joint · constant_velocity_retime · validate
# 动力学 🟡：idyn / mass_matrix / coriolis / gravity / cartesian_inertia
# 控制 🟡：ControlLoop · play_joint / play_cart · joint_position/velocity_control · arm_position/velocity_control
#        · torque_control · computed_torque_control · inverse_dynamics_control · pure_force_control
#        · HybridForcePosition · ImpedanceControl / AdmittanceControl · ForceTorqueSensor / JointTorqueSensor
```

#### 限位守卫 API（`utils/limits.py`；状态监测/日志归 ROS2 节点）

```python
clamp_to_limits(targets, limits: JointLimits) → ndarray   # 指令路径逐元素裁剪 ✅
# JoyArm.clamp_q(q) 即软限位裁剪门面；控制律下发前统一消费
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

#### 类型 API（`types.py`，全部 ✅）

```python
# 枚举：ControlMode（POSITION/VELOCITY/MIT）· Severity · SafetyAction · TrajectorySpace
# 数据类（@dataclass，ndarray 安全相等）：Pose（from_T(T) / .T 属性）· JointState / TcpState / ArmState（状态快照）
#   · JointLimits / TcpLimits（限位声明，硬/软两实例）· Wrench · Twist · Violation · IKResult · ComplianceParams
# 算法层消费约定（鸭子类型，无协议类）：求解器只访问 arm 公开属性与门面（清单见 §1.2），
#   「不 import joyarms」为架构约定
```

## 4. 文档同步维护

本文件与 [`README.md`](README.md) 是子库两大入口文档，**受众不同、须随代码同步维护**：README 给**使用者**（定位 / 架构 / 用法），本文件给 **agent / 开发者**（约定 / 速查 / 规则）；新内容按受众分流，不互相重复。

### 文档定位：精简概述、代码即详则

- 两份 md 只写**精简概述**（结构 / 用法 / API 速查），**不复述**代码已有内容；查细节看源码 docstring 或 `help(符号)`。
- **详细规则写进代码注释**：各包 `__init__.py` 写模块职责一句话；类与函数写精简 docstring（`:param:`/`:return:`/`:raises:`/ 章节标记）。

### 同步维护强制原则（每次变更必查）

> ⚠️ **每次 `joyarm_code/` 发生任何变化后（新增 / 修改 / 删除 `.py`、`.yaml`、目录结构、ws 功能包、章节脚本），必须同步更新本文件**，使其与代码保持一致；README 按受众需要同步：
>
> - 公开 API（`joyarm_core/__init__.py` 的 `__all__` 导出）变化 → 更新本文件「3.3 API 速查」与「3.1 模块导览」；
> - 目录 / 文件结构变化 → 更新本文件「1.1 目录树」（ws 部分含「1.4」）与 README「目录结构」；
> - 安装 / 用法 / 示例 / 章节落地状态变化 → 更新 README 对应小节；
> - 架构 / 约定变化 → 更新本文件 §1-2 与 README「基本架构」中的相应概念。
>
> 提交前自检：新增的公开符号已收录、且已补精简 docstring；删除的符号已从文档移除；文件清单与目录树一致。

### 行数上限：≤ 300 行

> 本文件硬性上限 **300 行**（`wc -l` 实测）。超出时**先精简再提交**：优先压缩 §3 速查——只留索引级签名，细节让位于源码 docstring；其次合并 §1-2 中的重复表述。禁止以"另开新节 / 另存新文件"绕过上限；新增内容与精简**同步进行**，只增不减必然超限。
