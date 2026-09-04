# AGENTS.md —— `joyarm_code/` 子库项目记忆

> **定位**：`joyarm_code/` 代码库的维护者 / Agent 内部文档，不进教程站点。
> **依赖链**：根 [`AGENTS.md`](../AGENTS.md)（教程站点）→ 本文件（代码库）→ [`README.md`](README.md)（用户向基础）。
> [`README.md`](README.md)（给使用者）已覆盖：**定位、目录概览、基本架构（分层图与核心概念）、安装、快速开始、离线语义、测试运行**——本文不重复，只在其上补充维护者向内容：**架构约束（§0，持久化）、架构细化、命名与设计约定、模块与 API 速查、文档维护规则**。
> ✅ 本文件正常纳入版本管理（直接 `git add` 即可）。
>
> **文档原则**：md 只写精简概述（结构速览 / API 速查），不复述代码已有内容；详细规则写进代码注释，查细节看源码 docstring 或 `help(符号)`。

## 0. 架构约束（持久化，joyarm_core 全体贡献者必须遵循）

> 以下为 joyarm_core 架构的**持久化约束**：代码演进不得违背；如需变更须先更新本节并经人工确认。

1. **组合根·单类**：`joyarm/` 的 `JoyArm` 是组合根，基于 config 型号配置对其他层功能（backend、robot 资产、robotics 各求解器）做**可更换式组合**——能力全部委托私有子成员，`JoyArm` 只提供公共门面（`arm.*`）。**无型号子类**：型号差异全部由配置表达（config yaml + URDF 资产 + backend 后端类），型号标识为 `model` 属性（= 工厂入参 = yaml 文件名 = `basic.name`）。**新型号 = `configs/<型号>.yaml` + `robot_model/` 资产 + `backend/backend_*.py`**（backend `REGISTRY` 一行；型号与整机后端 1:1，如 `joyarm_dm` ↔ `backend_dm`）。
2. **接口先行·章节实现**：robotics 六域（fkine/ikine/jacobian/dynamics/traj/control）**仅保留 ABC 接口 + 空 `REGISTRY`**——具体算法为教程各章教学内容（fkine Ch2 / ikine Ch3 / jacobian Ch4 / traj Ch5 / control Ch6/8/9 / dynamics Ch8），章节实现后经对应域 `REGISTRY` 注册即接入（换 config 即换算法）。加载链：**工厂 → JoyArm → config → 指定的各子成员**；域未配置**静默跳过**（教学过渡正常态），注册名无效/实例化失败**置空 + 警告**。
3. **通用兼容**：`JoyArm` 兼容所有带末端执行器的 6R/7R 臂；末端功能兼容多种执行器（契约见 `backend/backend.py` 的 `*_end` 方法族，多电机末端如灵巧手通用）。
4. **软失败**：工厂创建时 config 缺失、命名链不符（`basic.name` ≠ 型号名）或初始化异常则**返回 `None` 并输出创建失败信息**（不抛异常）；各子成员加载同构：注册名不存在/实例化失败则**置空 + 警告**（对应门面调用时报清晰 `RuntimeError`），不中断创建。
5. **参数排序契约**：接口**通用参数在前**（任何实现都需要，如 ikine 的 `target`/`frame`/`q0`），**特有参数 keyword-only 在后**（仅特定算法需要，如数值法的 `tol`/`iters`，解析法可忽略）——各章实现求解器时遵守。
6. **功能全面性 + 字典化**：常见功能（读/设配置、配置自检、连接、硬件自检、使能/失能、单关节控制、末端控制、紧急阻尼……）在 `JoyArm` 完成定义。六域策略成员**统一字典化**：config 可指定单个或列表规格（全部加载进 `dict[注册名→实例]`，首个为活动），运行期 `set_solver` 切换。
7. **robotics 独立运行**：robotics 子模块不得 import `joyarm`（鸭子类型消费 `arm`）；各章算法实现须可脱离 JoyArm 独立运行与测试（直接实例化，构造参数自足）。
8. **教学数据读取路径**：config 在 `JoyArm.__init__` **一次性加载**存入 `self._config`；教学数据（如 MDH 参数 `joyarm.arm_mdh_and_limits`）由教学算法经 `arm.get_config()` 从**类内已加载数据**读取，**不重新加载 yaml 文件**；MDH 校验（yaml 可不含该字段）在算法层使用时做，不在 `check_config`。
9. **开闭原则**：常见变更（新型号、换电机、结构更新、升级算法）以最小改动完成（新型号三件套 / 换算法 = 改 config 注册名 + 注册表一行），对扩展开放、对修改关闭；新增实现不得改动 `JoyArm` 接口签名。

## 1. 架构细化

> 补充 README「基本架构」：分层图与核心概念见 README，本节给文件级细节。

### 1.1 目录树（文件级）

```
joyarm_code/
├── joyarm_core/                     # ★ 核心 SDK（ROS2-free）
│   ├── __init__.py                  #   公开 API 扁平导出（__all__）
│   ├── utils/                       #   基础层
│   │   ├── types.py                 #     数据类 / 枚举
│   │   ├── transforms.py            #     SO(3)/SE(3) 纯 numpy 数学
│   │   └── limits.py                #     clamp_to_limits + 限位构建辅助（joint_limits_from_model/soft_limits）
│   ├── robotics/                    #   算法层：一域一子包（ABC + 空 REGISTRY）
│   │   ├── fkine/                   #     FkineSolver(ABC)——MDH 白盒 FK 为 Ch2 教学内容
│   │   ├── ikine/                   #     IkineSolver(ABC)——数值/解析 IK 为 Ch3 教学内容
│   │   ├── jacobian/                #     JacobianSolver(ABC，衍生量模板)——Ch4
│   │   ├── trajectory/              #     TrajPlanner(ABC) + AutoTrajPlanner/ForceTrajPlanner
│   │   ├── dynamics/                #     DynamicsSolver(ABC，Λ 模板)——Ch8
│   │   └── control/                 #     Controller(ABC) + AutoController/ForceController 桩——Ch6/9
│   ├── backend/                    #   通信层（整机 Backend + name 选型）
│   │   ├── backend.py               #     Backend（整机 ABC：*_arm/*_end + 参数读写方法族）
│   │   ├── backend_dm.py            #     BackendDM（DM 整机 7 电机；内含私有协议层 DmMotor/DmCanBus）✅
│   │   ├── u2can/                   #     厂商 DM 参考库（协议参照用，不依赖不导入）
│   │   └── __init__.py              #     REGISTRY + get_backend(name)
│   ├── joyarm/                     #   设备模型层（组合根 + 型号工厂）
│   │   ├── joyarm.py                #     JoyArm（单类）：config 驱动构造 + 六域字典组装 + 公开门面
│   │   │                            #       + 自检 + tcp_limits 解析 + load_config/_build_domain
│   │   └── __init__.py              #     JoyArmFactory/joyarm_factory（软失败：失败→None+信息）
│   ├── robot_model/                 #   URDF + meshes 资产（运行期加载；robot=纯资产，加载逻辑在 JoyArm）
│   └── configs/joyarm_dm.yaml       #   per-model YAML（basic/joyarm/robotics/backend 四段）
├── joyarm_ros2_ws/                  # ROS2 colcon 工作空间（规划，Ch10 落地时创建；见 §1.4）
├── chapt/                           # 章节教学示例脚本（一次性，不复用 SDK）
├── test/                            # 测试套件（离线为主 + 真机分层测试）
│   ├── test_backend_dm*.py          #   DM 协议离线单测 + 真机监视/调试/全覆盖（4 个）
│   ├── test_joyarm_full.py          #   JoyArm 真机分层测试（29 项九层风险递增；回放/求解器类随各章实现补回）
│   ├── test_joyarm_recovery.py      #   事故后复测（backend 级 4 步；每步预案卡人工审阅 + 速度守护）
│   ├── test_joyarm_factory.py       #   工厂软失败/六域字典机制/自检/配置 API
│   └── test_lock.py                 #   GIL 竞态教学实验
├── pyproject.toml                   # 工程配置（可编辑安装 joyarm_core）
├── README.md                        # 用户向入口
└── AGENTS.md                        # 本文件
```

### 1.2 依赖规则（无环明细）

- `utils` → 仅 numpy；
- `robotics` / `backend` → 仅依赖 `utils`；求解器鸭子类型消费 `arm`（经公开门面/属性），**不 import `joyarm`**（约束7）；
- `joyarm` → **组合根**：门面委托六域策略字典（`robotics`）与整机后端（`backend`），构造时按 config `robotics:` 段查各域 `REGISTRY` 组装成员、按 `backend:` 段 `name` 经 `get_backend` 构建后端，运行期加载 `robot_model/`、`configs/`；
- `joyarm_ros2_ws/src/*`（规划）→ 依赖 `joyarm_core` + `rclpy`；核心包保持 ROS2-free。

### 1.3 配置（yaml 分段）

`configs/<model>.yaml` 四段：`basic`（基本信息 + robot_model URDF 索引 + utils 守卫）/ `joyarm`（设备模型）/ `robotics`（六域算法选型）/ `backend`（通信层）。**命名链**：工厂入参 = 文件名 = `basic.name` 字段，任一环节不符即软失败返回 `None`（约束4）；backend 与机械臂型号 1:1（`backend_dm`）：

```yaml
basic: {name, robot, ee_frame, utils: {joint_limits_soft_margin}}
joyarm: {arm_mdh_and_limits, T_linkn_end, tcp_limits, q_zero, q_home, q_neutral}   # MDH/T_linkn_end 为教学数据（算法层按需读取，约束8）；workspace_box 兼容 (2,3)/(3,2)
robotics: {六域契约规格}   # 六域选型（注册名 / {name, **参数} / 规格列表）
backend: {name: backend_dm, arm: {channel, protocol, baud_rate, control_rate, joints}, end: {同构, joints}}
```

运行期可设参数走 `arm.set_config(path, value)` 白名单（`basic.utils.joint_limits_soft_margin`），**不回写 yaml**（文件为单一事实来源，重启以 yaml 为准）。

### 1.4 ROS2 工作空间（`joyarm_ros2_ws/`，**规划——Ch10 落地时创建，当前未建**）

标准 colcon 工作空间，承载所有 ROS2 功能包（第十章起）；`build/ install/ log/` 为构建产物。规划目录：`src/joyarm_node/`（ament_python：`arm_node.py` 每机械臂一节点 + `adapters.py` 内部类型 ↔ ROS2 消息互转，四元数 wxyz↔xyzw 换序）、`config/joyarm_dm.yaml`（model/rate/offline）、`launch/arm.launch.py`（+robot_state_publisher/rviz2，namespace 即多臂）。接口约定：状态广播走话题（`joint_states`、`~/tcp_pose`）；连接/夹爪走服务（std_srvs）；轨迹执行规划为 action（自定义 `joyarm_interfaces`）。规划中的应用包：`joyarm_teleop`(Ch12)、`joyarm_vision`(Ch14)、`joyarm_agent`(Ch15)；纯算法管线规划放 `joyarm_core/apps/`（ROS2-free）。

**环境与构建**：`python3 -m pip install --user -e joyarm_code` → `source /opt/ros/humble/setup.bash` → `colcon build` → `source install/setup.bash`。

> ⚠️ 边界：`joyarm_node` 不做 ros2_control hardware_interface、不追求直接兼容 MoveIt2。
> ⚠️ 陷阱①：package.xml 的 XML 注释中禁止双连字符 `--`，否则 catkin_pkg 静默回退，`ros2 run/launch` 找不到包。
> ⚠️ 陷阱②：ament_python 包 `setup.py` 须声明 `tests_require=["pytest"]`，否则 `colcon test` 发现 0 用例。

## 2. 命名与设计约定

### 2.1 命名约定（全仓库强制，含 `chapt/`）

- **类名 = 驼峰（PascalCase）；文件名 = 小写 snake_case。**
- **设备模型**（`joyarm/`）：`JoyArm` = 组合根单类（无型号子类）；`joyarm_factory(model)` 外部推荐入口（失败 → `None` + 失败信息）。文件 `joyarm.py` / `__init__.py`。
- **Backend 整机两层**（`backend/`）：根 `Backend`（`*_arm`/`*_end` 方法族）→ 型号层 `BackendDM`（1:1：`backend_dm` ↔ `joyarm_dm`）。`REGISTRY` + `get_backend(name)` 供 config 选型。
- **属性约定**：型号名 = `model`（str）；pinocchio 构型产物 = `pin_model`/`pin_data`（**勿与型号名混淆**）；整机后端 = `_backend`（私有）；**六域策略成员字典** = `_fkine_solvers`/`_ikine_solvers`/`_jacobian_solvers`/`_dynamics_solvers`/`_traj_planners`/`_controllers`（`dict[注册名→实例]`）+ `_active_name`（域→活动注册名）；config = `self._config`（`get_config()` 深拷贝读取）。求解器形参用 `arm`（鸭子类型）。
- **方法命名**：本体带 `arm`、末端带 `end` 一一对应（`enable_arm`/`enable_end`…）；执行类统一 `joint` 形参（None=全部）；`set_mode_*` 默认 POSITION；参数读写 `read/write_param_{arm,end}`（joint=None 读返列表、写标量广播或等长列表）；求解切换 `set_solver(domain, name)`（`set_controller` 为 control 域别名）、查询 `list_solvers(domain)`。
- **算法可换（成员即策略）**：每域 = ABC + `REGISTRY`（实现一节点一文件，各章新增）。

### 2.2 设计原则与编码约定

| 原则 | 落地 |
|:--|:--|
| **组合根·单类** | `JoyArm` 持 `_backend` + 六域成员字典；公开门面全部委托活动成员，换 config 即换算法；型号差异全在 config（约束1/6） |
| **接口先行·章节实现** | robotics 六域仅 ABC + 空 REGISTRY；域未配置静默跳过、无效注册名置空+警告；各章实现注册后 config 选型接入（约束2） |
| **软失败** | 工厂 config 缺失/命名链不符/初始化异常→`None`+信息；成员注册名无效→置空+警告，门面调用报 `RuntimeError`；backend 无效→置空（离线计算仍可用）（约束4） |
| **教学数据读取** | config 一次性加载存 `self._config`；教学算法经 `arm.get_config()` 读类内数据，不重读 yaml（约束8） |
| **单向导入、无环** | `joyarm` → `robotics`/`backend` → `utils`；robotics/backend 不 import joyarm（约束7） |
| **离线模式** | 默认 `connected=False`（已注册域计算可用）；`connect()` 后执行类可用；`state` property 已连接现读、离线 None |
| **共享类型单定义** | 跨层类型只在 `utils/types.py`；ROS2 消息仅在 ws 的 `adapters.py` 互转 |
| **核心轻依赖** | 核心仅 `numpy`/`pin`/`pyyaml`；`rclpy` 仅 ws |
| **开闭原则** | 新型号 = config yaml + robot_model 资产 + backend 文件（REGISTRY 一行）；换算法 = 改 config 注册名（约束9） |

编码约定：`q=(n,)` 或 `(N,n)`、`T=(4,4)`、角度弧度；FK `rep` 三态、雅可比 `ref` 两态（local/base）；软/硬限位分级（`clamp_to_limits`）；`ControlMode` 三态（纯力矩经 MIT `kp=kd=0`）；接口参数**通用在前、特有 keyword-only 在后**（约束5）；可视化不在核心包。

### 2.3 库边界

SDK（`arm.xx`）→ `joyarm_core/`；ROS2 → `joyarm_ros2_ws/src/`（规划）；CLI/GUI → `quickstart/`（占位）；教学脚本 → `chapt/`。

## 3. 模块与 API 速查

### 3.1 模块导览（✅ 已实现 / 🟡 教学章节待实现）

| 文件 | 类 / 关键定义 | 章节 | 状态 |
|:---|:---|:---:|:---:|
| `utils/types.py` | 枚举 + 数据类（Pose/TrajFrame/JointState/ArmState/JointLimits/TcpLimits/IKResult…） | Ch2 | ✅ |
| `utils/transforms.py` | 23 个纯 numpy 函数（rpy/rodrigues/quat/T 族/slerp…） | Ch2 | ✅ |
| `utils/limits.py` | `clamp_to_limits` + `joint_limits_from_model`/`soft_limits` | Ch11 | ✅ |
| `backend/backend.py` | `Backend(ABC)` 整机契约（`*_arm`/`*_end` + 参数读写） | Ch2 | ✅ |
| `backend/backend_dm.py` | `BackendDM`（DM 7 电机，私有 DmMotor/DmCanBus） | Ch6/13 | ✅ |
| `robotics/fkine/` | `FkineSolver(ABC)`（批量/rep 模板在 ABC）；MDH 白盒 FK 待 Ch2 实现 | Ch2 | 🟡 |
| `robotics/ikine/` | `IkineSolver(ABC)`（solve 单解 + solve_all 全解 + `_shift_2pi`/`_select_nearest` 助手）；数值/解析实现待 Ch3 | Ch3 | 🟡 |
| `robotics/jacobian/` | `JacobianSolver(ABC)`（`manipulability`/`cond_number`/`statics` 衍生量模板在 ABC） | Ch4 | 🟡 |
| `robotics/trajectory/` | `TrajPlanner(ABC)`（`plan` 模板 + `_plan`/`sample_frame` 内核 + `_check_targets`）+ AutoTrajPlanner/ForceTrajPlanner 桩 | Ch5/9 | 🟡 |
| `robotics/dynamics/` | `DynamicsSolver(ABC)`（Λ=J⁻ᵀMJ⁻¹ 模板在 ABC）；实现待 Ch8 | Ch8 | 🟡 |
| `robotics/control/` | `Controller(ABC)`（`step` 模板 + `_compute` 内核 + 限位守卫）+ AutoController/ForceController 桩 | Ch6/9 | 🟡 |
| `joyarm/joyarm.py` | `JoyArm`（单类组合根：config 驱动构造 + 六域字典 + 门面 + 自检）+ `load_config` + `_build_domain` | — | ✅ |
| `joyarm/__init__.py` | `JoyArmFactory`（软失败；`list_models` 扫描 configs） | — | ✅ |
| `configs/joyarm_dm.yaml` | 型号 YAML 四段（robotics 段已预配置各章注册名契约，未注册时告警置空） | — | ✅ |

### 3.2 组装与数据流

```
JoyArm（单类组合根：_backend + 六域成员字典 + config/_config）
   │  config 驱动：basic.robot→URDF（pin_model/pin_data）；backend.name→后端；
   │  robotics:→各域 REGISTRY 字典化组装（首个活动；域未配置=空，教学过渡态）
   │  教学数据（MDH 等）：算法层经 arm.get_config() 读类内 config（不重读 yaml）
Backend(ABC) ──▶ BackendDM ✅
六域 ABC + 空 REGISTRY ──▶ 各章实现注册接入（fkine Ch2 / ikine Ch3 / jacobian Ch4 /
   traj Ch5 / control Ch6/8/9 / dynamics Ch8）
```

### 3.3 API 速查

> **精简签名**：`name(关键参数) → 返回 — 一句话`。完整说明见源码 docstring 或 `help(符号)`。

#### 设备模型 API（`joyarm_factory("joyarm_dm")` 创建；失败 → `None`+信息）

```python
# 配置
JoyArm.get_config() → dict                                    # 深拷贝快照（教学数据读取口）✅
JoyArm.set_config(path, value)                                # 运行期白名单（soft_margin）✅
JoyArm.check_config() → None（异常 ValueError）               # 配置最小自检（basic 段/命名链/urdf/位形/关节数；正常静默）✅
JoyArm.check_hardware() → None（异常 RuntimeError）           # 硬件最小自检（需 connect：通讯/故障/编码器；温度等运行期监控归 ROS2）✅
# 成员字典
JoyArm.set_solver(domain, name) / set_controller(name) · list_solvers(domain)   # 运行期切换/查询 ✅
# 计算门面（已注册域可用；参数排序：通用前/特有 keyword-only 后，约束5）
JoyArm.fkine(q, frame, rep="pose") · ikine(target, frame, q0, **kw) → IKResult 单解（限位剔除+q0 最近） · ikine_all(target, frame) → 全解 (K,n)（±2π 归位）
JoyArm.jac(q, frame, ref="base") · manipulability / cond_number / statics(q, F, frame)
JoyArm.idyn / mass_matrix / coriolis / gravity · cartesian_inertia(q, frame)
# 连接 / 执行 / 参数 / 末端 ✅（read_mode_arm/end 为本地缓存离线可查；set_arm_command(..., joint=None) 单关节）
JoyArm.connect() / disconnect() · enable/disable_{arm,end} · set_zero_{arm,end} · set_mode_{arm,end}(mode=POSITION, joint=None) · read_mode_{arm,end}
JoyArm.get_arm_state() → ArmState（fkine 已注册时填 tcp.pose）· set_arm_command(mode, q/dq/tau/kp/kd, joint=None)（下发前守卫：q 软限位裁剪、dq/tau 幅值裁剪，越界告警） · read/write_param_{arm,end}
JoyArm.end_open/end_close/end_zero(joint=None) · set_end_position / set_end_force / get_end_state
JoyArm.damping_mode(kd=10.0)                    # 紧急阻尼：任何状态全电机（含末端）MIT 纯阻尼 ✅
JoyArm.state → ArmState|None · rand_q / clamp_q / is_q_valid
JoyArm.set/get_target_traj(targets) · set/get_current_frame(frame)   # 轨迹桥：应用→规划→控制（TrajFrame；发布即不可变+原子交换）✅
# 关键属性：model / pin_model / pin_data / n / nv / ee_frame_* / joint_limits(_soft) / qlow / qhigh / q_zero / q_home / q_neutral / tcp_limits / T_base / _backend / connected
#   六域字典：_fkine_solvers/_ikine_solvers/_jacobian_solvers/_dynamics_solvers/_traj_planners/_controllers + _active_name
#   轨迹桥（私有）：_target_traj（List[TrajFrame]，写者=应用线程）/ _current_frame（TrajFrame，写者=规划线程）
```

#### Robotics 独立 API（各域 ABC；实现为教学章节内容）

```python
FkineSolver / IkineSolver / JacobianSolver / DynamicsSolver / TrajPlanner / Controller   # ABC 契约 🟡
TrajPlanner(plan_hz, sample_hz)：plan(arm, targets)→_plan / sample_frame(t_abs)→TrajFrame   # 目标驱动规划 🟡
AutoTrajPlanner（关节/位姿 × 单值/序列 四情形）/ ForceTrajPlanner（阻抗/导纳/力位混合）   # 注释桩 🟡
Controller(ctrl_hz)：step(arm, frame, state=None)→_compute→限位守卫(q 软限位/dq/tau 幅值)→set_arm_command   # 模板已实现 ✅
AutoController（type 五类型）/ ForceController（impedance/hybrid）   # 注释桩 🟡
# 求解器用法二选一：子类实例.solve(arm, q)（arm 鸭子类型，课堂/单测）或 arm.* 门面（活动成员，应用）
```

#### Backend / 限位 / 数学 / 类型 API

```python
Backend(ABC) 整机契约 ✅：connect/disconnect + connected + read_mode_* + *_arm/*_end 方法族（joint=None=全部）+ 参数读写；末端含 send_mit_end（MIT 阻尼通道，damping_mode 消费）
get_backend(name) · REGISTRY     # config backend.name 选型 ✅
clamp_to_limits(targets, limits) · joint_limits_from_model(model) · soft_limits(hard, margin)   # ✅
transforms.py 23 函数（rpy/rodrigues/quat/T/adT/slerp，纯 numpy）✅ · types.py 枚举+数据类 ✅
```

## 4. 文档同步维护

本文件与 [`README.md`](README.md) 是子库两大入口文档，**受众不同、须随代码同步维护**：README 给**使用者**，本文件给 **agent / 开发者**；新内容按受众分流，不互相重复。

- 两份 md 只写**精简概述**；详细规则写进代码 docstring（各包 `__init__.py` 一句话职责 + `:param:`/`:return:`/`:raises:`/章节标记）。
- **每次 `joyarm_code/` 任何变化后必须同步更新本文件**：公开 API 变 → §3；目录变 → §1.1；安装/用法/章节状态变 → README；架构/约定变 → §0-2。**架构约束（§0）变更须人工确认后先改本节再动代码。**
- 提交前自检：新增公开符号已收录、已补 docstring；删除符号已移除；目录树一致。

### 行数上限：≤ 300 行

> 本文件硬性上限 **300 行**。超出时先精简再提交：优先压缩 §3 速查（只留索引级签名）；其次合并 §1-2 重复表述。禁止以"另存新文件"绕过上限；新增内容与精简同步进行。
