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
2. **接口先行·内置默认实现**：robotics 六域（fkine/ikine/jacobian/dynamics/traj/control）= **ABC 接口 + 内置默认实现 + `REGISTRY`**（Pin 四域 `pin_*_solver` / 到关节目标规划 `to_joint_traj_planner` / 关节位置控制 `joint_position_controller`，5/6/7 轴通用、config 选配启用）——六域均为**纯计算内核、无线程**（traj/control 的周期调度与指令门控归 JoyArm 运动管线三线程，控制器经 `MODE` 类属性声明所需电机模式）；各章自定义算法为实现类加 `@register` 装饰器（`robotics/_registry.py`）即自动注册接入（**实现类名 = 算法前缀 + 域基类名，如 `PinFkineSolver`；注册名 = 类名小写+下划线，如 `pin_fkine_solver`**，换 config 即换算法）。加载链：**工厂 → JoyArm → config → 指定的各子成员**；域未配置即无成员（对应门面调用显性 `RuntimeError`，过渡正常态），**配置的注册名无效/实例化失败即构造失败（硬失败，`ValueError`）**。
3. **通用兼容**：`JoyArm` 兼容所有带末端执行器的 6R/7R 臂；末端功能兼容多种执行器（契约见 `backend/backend.py` 的 `*_end` 方法族，多电机末端如灵巧手通用）。
4. **软化仅限工厂入口**：工厂创建时 config 缺失、命名链不符（`basic.name` ≠ 型号名）或初始化异常则**返回 `None` 并输出创建失败信息**（不抛异常）；`JoyArm` 直用为**硬失败**——config 必需（缺失/自检不通过即抛）、backend 必配、robotics 配置的成员必须全部创建成功（约束2）。
5. **参数排序契约**：接口**通用参数在前**（任何实现都需要，如 ikine 的 `target`/`frame`/`q0`），**特有参数 keyword-only 在后**（仅特定算法需要，如数值法的 `tol`/`iters`，解析法可忽略）——各章实现求解器时遵守。
6. **功能全面性 + 字典化**：常见功能（读配置、配置自检、连接、硬件自检、使能/失能、单关节控制、末端控制、紧急阻尼……）在 `JoyArm` 完成定义。六域策略成员**统一字典化**：config 可指定单个或列表规格（全部加载进 `dict[注册名→实例]`，首个为激活、有且仅有一个），运行期 `set_solver` 切换。
7. **robotics 独立运行**：robotics 子模块不得 import `joyarm`（鸭子类型消费 `arm`）；各章算法实现须可脱离 JoyArm 独立运行与测试（直接实例化，构造参数自足）。
8. **教学数据读取路径**：config 在 `JoyArm.__init__` **一次性深拷贝**存入 `self._config`（隔离外部引用，`get_config()` 另返回深拷贝）；教学数据（如 MDH 参数 `joyarm.arm_mdh_and_limits`）由教学算法经 `arm.get_config()` 从**类内已加载数据**读取，**不重新加载 yaml 文件**；MDH 校验（yaml 可不含该字段）在算法层使用时做，不在 `check_config`。
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
│   │   └── limits.py                #     clamp_to_limits + 限位构建辅助
│   ├── robotics/                    #   算法层：一域一子包（ABC + 默认实现 + REGISTRY + @register）
│   │   ├── fkine/                   #     FkineSolver(ABC) + PinFkineSolver 默认实现——MDH 白盒 FK 为 Ch2 教学内容
│   │   ├── ikine/                   #     IkineSolver(ABC) + PinIkineSolver 默认实现——解析 IK 为 Ch3 教学内容
│   │   ├── jacobian/                #     JacobianSolver(ABC，衍生量模板) + PinJacobianSolver——Ch4
│   │   ├── trajectory/              #     TrajPlanner(ABC) + ToJointTrajPlanner 到关节目标规划默认实现——Ch5
│   │   ├── dynamics/                #     DynamicsSolver(ABC，Λ 模板) + PinDynamicsSolver——Ch8
│   │   └── control/                 #     Controller(ABC，compute+MODE 声明) + JointPositionController——Ch6/9
│   ├── backend/                    #   通信层（整机 Backend + name 选型）
│   │   ├── backend.py               #     Backend（整机 ABC：*_arm/*_end + 参数读写；限位 __init__ 构建 + send_* 守卫模板）
│   │   ├── backend_dm.py            #     BackendDM（DM 整机 7 电机；内含私有协议层 DmMotor/DmCanBus）✅
│   │   ├── backend_dm_mujoco.py     #     BackendDMMujoco（DM 机械臂 MuJoCo 仿真后端桩，待实现）🟡
│   │   ├── u2can/                   #     厂商 DM 参考库（协议参照用，不依赖不导入）
│   │   └── __init__.py              #     REGISTRY + get_backend(name)
│   ├── joyarm/                     #   设备模型层（组合根 + 型号工厂）
│   │   ├── joyarm.py                #     JoyArm（单类）：config 驱动构造 + 六域字典组装 + 公开门面
│   │   │                            #       + 自检 + tcp_limits 解析 + load_config/_build_domain
│   │   │                            #       + 私有周期线程机制 _run_periodic/_PeriodicThread（保活+运动三线程）
│   │   │                            #       + _cubic_traj 三次插值（move_j/safe_* 直连路径，单消费者内联）
│   │   └── __init__.py              #     JoyArmFactory/joyarm_factory（仅工厂入口软化：失败→None+信息；JoyArm 直用硬失败）
│   ├── robot_model/                 #   URDF + meshes 资产（运行期加载；robot=纯资产，加载逻辑在 JoyArm）；joyarm_dm/ 为旧版原始构型参照（现行配置用 joyarm_dm_fixend）
│   └── configs/                     #   per-model YAML（basic/joyarm/robotics/backend 四段）
│       ├── joyarm_dm.yaml           #     正式型号配置
│       └── joyarm_template.yaml     #     型号配置模板（复制为 <型号>.yaml 填写；本文件不入 list_models）
├── joyarm_ros2_ws/                  # ROS2 colcon 工作空间（规划，Ch10 落地时创建；见 §1.4）
├── chapt/                           # 章节教学示例脚本（一次性；基础设施优先复用 joyarm_core，章节算法可自行实现）
├── test/                            # 测试套件（离线回归 + 真机全覆盖）
│   ├── test_backend_dm.py           #   DM 协议层离线单测（打包/解包/帧分发/错误码语义）
│   ├── test_backend_dm_full.py      #   DM 真机全覆盖（42 项，风险递增、逐项确认）
│   ├── test_backend_base.py         #   Backend 基类限位守卫离线单测（arm/end 两族，哑后端）
│   ├── test_limits.py               #   utils/limits.py 离线单测（限位构建/直配软限位 + 裁剪）
│   ├── test_joyarm_full.py          #   JoyArm 真机分层测试（29 项九层风险递增；回放/求解器类随各章实现补回）
│   └── test_joyarm_factory.py       #   工厂软化+直用硬失败/六域字典机制/静态自检/配置 API/前置校验族
├── pyproject.toml                   # 工程配置（可编辑安装 joyarm_core）
├── README.md                        # 用户向入口
└── AGENTS.md                        # 本文件
```

### 1.2 依赖规则（无环明细）

- `utils` → 仅 numpy；
- `robotics` / `backend` → 依赖 `utils`（+ numpy）；robotics 的 Pin 默认实现另需 `pinocchio`；求解器按属性约定消费 `arm`（经公开门面/属性），**不 import `joyarm`**（约束7）；
- `joyarm` → **组合根**：门面委托六域策略字典（`robotics`）与整机后端（`backend`），构造时按 config `robotics:` 段查各域 `REGISTRY` 组装成员、按 `backend:` 段 `name` 经 `get_backend` 构建后端，运行期加载 `robot_model/`、`configs/`；
- `joyarm_ros2_ws/src/*`（规划）→ 依赖 `joyarm_core` + `rclpy`；核心包保持 ROS2-free。

### 1.3 配置（yaml 分段）

`configs/<model>.yaml` 四段：`basic`（基本信息 + robot_model URDF 索引）/ `joyarm`（设备模型 + 直配软限位）/ `robotics`（六域算法选型）/ `backend`（通信层 + 硬限位四键）。**命名链**：工厂入参 = 文件名 = `basic.name` 字段，任一环节不符即工厂返回 `None`（约束4）；backend 与机械臂型号 1:1（`backend_dm`）：

```yaml
basic: {name, robot, ee_frame}
joyarm: {arm_mdh_and_limits, T_linkn_end, arm_soft_limits, end_soft_limits, tcp_limits, arm_home, end_home}   # 软限位四键直值 {q_min, q_max, dq_max, tau_max}（arm/end 分开，标量或 n/n_end 元列表，须位于硬限位内；供上层状态判断，不参与指令裁剪）；zero/neutral 按自由度全零（代码固定，不经 config）；MDH/T_linkn_end 为教学数据（约束8）；workspace_box 兼容 (2,3)/(3,2)
robotics: {六域契约规格}   # 六域选型（注册名 / {name, **参数} / 规格列表，全部加载、首个激活）；当前整段注释过渡——注册名实现并注册后取消注释接入（硬失败语义）
backend: {name: backend_dm, arm: {channel, protocol, baud_rate, control_rate, joints}, end: {channel, protocol, baud_rate, joints}}   # arm/end joints 条目四限位键 q_min/q_max/dq_max/tau_max 同构（= 硬限位来源，backend `__init__` 自解析；arm 数值须与 URDF limit 标定保持一致，JoyArm init 三类自检告警：关节缺失/数值不一致/URDF 多余非 mimic 活动关节）；
```

限位语义：**backend 下发指令只裁硬限位**（`Backend.send_*` 守卫模板，唯一执行点）；软限位由 JoyArm 直配加载（`arm_limits_soft`/`end_limits_soft`，缺省软=硬）仅作上层状态判断依据（超软限位→状态异常→急停恢复，后续实现）；config 改动重启生效，**不回写 yaml**。

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
- **属性约定**：型号名 = `model`（str，product model）；pinocchio 构型产物 = `pin_model`/`pin_data`（**勿与型号名混淆**）；关节数 = `n_arm`/`n_end`；硬限位 = `arm_limits`/`end_limits`（config `backend.*.joints` 四键，初始化后固定，下发守卫唯一依据）；软限位 = `arm_limits_soft`/`end_limits_soft`（config `joyarm.*_soft_limits` 直配，缺省软=硬，仅上层状态判断用）；整机后端 = `_backend`（私有，必配）；**六域策略成员字典** = `_fkine_solvers`/`_ikine_solvers`/`_jacobian_solvers`/`_dynamics_solvers`/`_traj_planners`/`_controllers`（`dict[注册名→实例]`）+ `_active_name`（域→激活注册名）；config = `self._config`（构造时深拷贝，`get_config()` 深拷贝读取）。求解器形参用 `arm`（鸭子类型）。
- **方法命名**：本体带 `arm`、末端带 `end` 一一对应（`enable_arm`/`enable_end`…，末端动作 `set_end_open/close/zero`）；执行类统一 `joint` 形参（None=全部）；`set_mode_*` 默认 POSITION；参数读写 `read/write_param_{arm,end}`（joint=None 读返列表、写标量广播或等长列表）；求解切换 `set_solver(domain, name)`、查询 `list_solvers(domain)`（每域有且仅一个激活）。
- **算法可换（成员即策略）**：每域 = ABC + `REGISTRY`（实现一节点一文件，各章新增）。

### 2.2 设计原则与编码约定

| 原则 | 落地 |
|:--|:--|
| **组合根·单类** | `JoyArm` 持 `_backend` + 六域成员字典；公开门面全部委托激活成员，换 config 即换算法；型号差异全在 config（约束1/6） |
| **接口先行·内置默认实现** | 六域 = ABC + 内置默认实现（Pin 四域/到关节目标规划/关节位置控制，config 选配）；域未配置即无成员（门面显性报错）；**注册名无效→构造失败（硬失败）**；章节实现 @register 后同样 config 选型（约束2） |
| **软化仅限工厂** | 工厂 config 缺失/命名链不符/初始化异常→`None`+信息；`JoyArm` 直用硬失败：config 必需（`check_config` 静态自检通过才初始化）、backend 必配、域成员必成（约束4） |
| **教学数据读取** | config 一次性深拷贝存 `self._config`；教学算法经 `arm.get_config()` 读类内数据，不重读 yaml（约束8） |
| **单向导入、无环** | `joyarm` → `robotics`/`backend` → `utils`；robotics/backend 不 import joyarm（约束7） |
| **离线模式** | 默认 `connected=False`（计算类可用）；`connect()` 后执行类可用；执行前置校验族 connected/enabled/mode（`_require_*`） |
| **共享类型单定义** | 跨层类型只在 `utils/types.py`；ROS2 消息仅在 ws 的 `adapters.py` 互转 |
| **核心轻依赖** | 核心仅 `numpy`/`pin`/`pyyaml`；`rclpy` 仅 ws |
| **开闭原则** | 新型号 = config yaml + robot_model 资产 + backend 文件（REGISTRY 一行）；换算法 = 改 config 注册名（约束9） |

编码约定：`q=(n,)` 或 `(N,n)`、`T=(4,4)`、角度弧度；FK `rep` 三态、雅可比 `ref` 两态（local/base）；**限位单点守卫**（backend 下发只裁硬限位；裁剪唯一执行点 = `Backend.send_*_{arm,end}` 基类守卫模板，`Controller.compute` 等上游一律原样透传不重复裁剪；硬限位自 config `backend.*.joints` 四键解析（arm/end 一致由 backend `__init__` 自解析；arm 数值与 URDF limit 标定一致，JoyArm init 自检不一致项告警）；软限位 config 直配仅加载、不参与指令裁剪；越限告警节流每 0.5s 至多一条）；`ControlMode` 三态（纯力矩经 MIT `kp=kd=0`）；接口参数**通用在前、特有 keyword-only 在后**（约束5）；可视化不在核心包；异常消息统一格式**『文件名 - 故障的功能：具体原因』**（含 NotImplementedError 教学桩，新代码一律遵守）。

### 2.3 库边界

SDK（`arm.xx`）→ `joyarm_core/`；ROS2 → `joyarm_ros2_ws/src/`（规划）；CLI/GUI → `quickstart/`（占位）；教学脚本 → `chapt/`。

## 3. 模块与 API 速查

### 3.1 模块导览（✅ 已实现 / 🟡 教学章节待实现）

| 文件 | 类 / 关键定义 | 章节 | 状态 |
|:---|:---|:---:|:---:|
| `utils/types.py` | 枚举 + 数据类（Pose/TrajFrame/JointState/ArmState/JointLimits/TcpLimits/IKResult…） | Ch2 | ✅ |
| `utils/transforms.py` | 23 个纯 numpy 函数（rpy/rodrigues/quat/T 族/slerp…） | Ch2 | ✅ |
| `utils/limits.py` | `clamp_to_limits` + `joint_limits_from_model`/`limits_from_joint_cfgs`/`soft_limits_from_cfg`（直配软限位）+ `rand_within_limits`（限位内采样） | Ch11 | ✅ |
| `backend/backend.py` | `Backend(ABC)` 整机契约（`*_arm`/`*_end` + 参数读写 + 限位构建 + send_* 两族守卫模板 + `read_state_cache_*`/`state_age_*` 只读缓存契约，默认 NotImplementedError） | Ch2 | ✅ |
| `backend/backend_dm.py` | `BackendDM`（DM 7 电机，私有 DmMotor/DmCanBus；`read_state_cache_*` 零总线帧组装缓存槽 + `state_age_*` 陈旧度，`DmMotor.t_state` 应答时间戳） | Ch6/13 | ✅ |
| `backend/backend_dm_mujoco.py` | `BackendDMMujoco` DM 机械臂 MuJoCo 仿真后端桩（与 `backend_dm` 同型号配套；实现后 REGISTRY 注册 `backend_dm_mujoco`） | — | 🟡 |
| `robotics/fkine/` | `FkineSolver(ABC)`（批量/rep 模板在 ABC）+ `PinFkineSolver`（forwardKinematics+updateFramePlacement，含帧名哨兵防御）✅；MDH 白盒 FK 为 Ch2 教学 | Ch2 | ✅ |
| `robotics/ikine/` | `IkineSolver(ABC)`（solve_all 全解 + `_shift_2pi`/`_select_nearest` 助手）+ `PinIkineSolver`（LM 自适应阻尼 + 多起点重启，不可达返 success=False）✅；解析 IK 为 Ch3 教学 | Ch3 | ✅ |
| `robotics/jacobian/` | `JacobianSolver(ABC)`（衍生量模板）+ `PinJacobianSolver`（computeFrameJacobian，base/local 双参考系）✅ | Ch4 | ✅ |
| `robotics/trajectory/` | `TrajPlanner(ABC)`（校验/剔除/回退管线）+ `ToJointTrajPlanner`（到关节目标点，五次多项式六边界条件，采样出 q/dq/ddq；目标可带 dq/ddq）✅ | Ch5 | ✅ |
| `robotics/dynamics/` | `DynamicsSolver(ABC)`（fdyn/Λ 模板）+ `PinDynamicsSolver`（rnea/crba 对称化/nonLinearEffects，f_ext 待力控章节）✅ | Ch8 | ✅ |
| `robotics/control/` | `Controller(ABC)`（compute 内核 + MODE 声明）+ `JointPositionController` 关节位置控制器（步长 ≤ dq_max/ctrl_hz 裁剪+节流告警）✅；运行防护 is_normal 在 JoyArm 管线 | Ch6 | ✅ |
| `joyarm/joyarm.py` | `JoyArm`（单类组合根：config 驱动构造 + 六域字典 + 门面 + 自检 + 运动管线三线程）+ `load_config`/`_build_domain`/`_parse_domain_specs` + 私有周期线程机制 `_run_periodic`/`_PeriodicThread`（单消费者，不入 utils） | — | ✅ |
| `joyarm/__init__.py` | `JoyArmFactory`（仅工厂入口软化；`list_models` 扫描 configs） | — | ✅ |
| `configs/joyarm_dm.yaml` | 型号 YAML 四段（robotics 段注释过渡：注册名实现并注册后取消注释接入，硬失败语义） | — | ✅ |

### 3.2 组装与数据流

```
JoyArm（单类组合根：_backend + 六域成员字典 + config/_config）
   │  config 驱动：basic.robot→URDF（pin_model/pin_data）；backend.name→后端（必配）；
   │  robotics:→各域 REGISTRY 字典化组装（首个激活；域未配置=空，门面显性报错）
   │  教学数据（MDH 等）：算法层经 arm.get_config() 读类内 config（不重读 yaml）
Backend(ABC) ──▶ BackendDM ✅
六域 ABC + 内置默认实现（Pin 四域 / to_joint_traj_planner / joint_position_controller）──▶
   章节自定义实现 @register 后并列选配（fkine Ch2 / ikine Ch3 / jacobian Ch4 /
   traj Ch5 / control Ch6/8/9 / dynamics Ch8）
```

### 3.3 API 速查

> **精简签名**：`name(关键参数) → 返回 — 一句话`。完整说明见源码 docstring 或 `help(符号)`。

#### 设备模型 API（`joyarm_factory("joyarm_dm")` 创建；失败 → `None`+信息）

```python
# 配置
JoyArm.get_config() → dict                                  # 深拷贝快照（教学数据读取口）✅
JoyArm.check_config(model, config) → None（异常 ValueError）   # **静态**自检（basic/命名链/URDF/backend 必配/限位四键/位形长度；init 第一步自动调用）✅
JoyArm.check_hardware() → None（异常 RuntimeError）           # 硬件自检（自包含：临时连接失能状态检查→断开，手动调用；温度等运行期监控归 ROS2）✅
# 成员字典
JoyArm.set_solver(domain, name) · list_solvers(domain)   # 运行期切换/查询（有且仅一个激活） ✅
# 计算门面（已配置域可用；参数排序：通用前/特有 keyword-only 后，约束5）
JoyArm.fkine(q, frame, rep="pose") · ikine(target, frame, q0, **kw) → IKResult 单解（限位剔除+q0 最近） · ikine_all(target, frame) → 全解 (K,n)（±2π 归位）
JoyArm.jac(q, frame, ref="base") · fkine_vel(q, dq, frame, ref)（V=J·q̇ 六维速度旋量） · ikine_vel(q, V, frame, ref, damping=1e-3)（q̇=J*·V 微分逆解） · manipulability（w=Πσᵢ） / statics(q, F∈R^6, frame)（τ=JᵀF ∈ R^n_arm）；求解统一经 _solve（RLock 串行化，内核免线程安全）
JoyArm.idyn / mass_matrix / coriolis / gravity · cartesian_inertia(q, frame)
# 连接 / 执行 / 参数 / 末端 ✅（read_mode_arm/end 为本地缓存离线可查；set_arm_command(..., joint=None) 单关节）
JoyArm.connect() / disconnect()（支持 with 上下文：enter 自动 connect，exit 尽力 disable→disconnect；connect 自动启动**状态保活线程** joyarm-state@10Hz——空闲期缓存陈旧（age>0.15s）时主动刷新，控制流期间随指令帧刷新零总线开销）· enable/disable_{arm,end} · set_zero_{arm,end} · clear_fault_{arm,end}（验证式清错复位） · set_mode_{arm,end}(mode=POSITION, joint=None) · read_mode_{arm,end}
JoyArm.get_arm_state() → ArmState（**新鲜度感知**：缓存新鲜（age≤0.15s）零总线帧组装，陈旧同步刷新；fkine 已配置时填 tcp.pose）· get_end_state() 同构 · refresh_state()（强制同步刷新） · set_arm_command(mode, q/dq/tau/kp/kd, joint=None)（前置校验 connected+mode 后委托后端；限位守卫在 Backend 基类模板） · read/write_param_{arm,end}
JoyArm.set_end_open/close/zero(joint=None) · set_end_position / set_end_tau / get_end_state   # 末端守卫同在后端模板（tau 按 ±tau_max 数值裁剪）
JoyArm.damping_mode(kd=10.0)                    # 紧急阻尼：任何状态全电机（含末端）MIT 纯阻尼 ✅
# 运动便利与安全层 ✅（move_j 为独立功能、与规划管线并行、仅直接调用；常规运动走轨迹桥→规划器→控制器）
JoyArm.hold_position(kp,kd,tau)（MIT 阻抗保持当前姿态，tau 缺省重力前馈）· lock_position()（急停锁定：切位置模式锁当前 q）
JoyArm.move_j(q, t=None, rate/tol/timeout kw)（三次多项式阻塞运动，精确定时器逐帧下发+到位等待；q 入口裁硬限位，判定以裁剪后为准）· move_l(pose, t)（占位）· teach_mode(on=True)（占位：拖动示教，待重力补偿）
JoyArm.safe_home(t) · home_to_zero(t)（先校验位于 home）· safe_zero()（安全起停组合：home→zero；arm+end 均执行，本体 MIT 阻抗+末端位置模式）· is_in_position(q 或 pose, 容差) → bool（单入口双判断）
JoyArm.rand_q_arm(size, rng)（本体硬限位内采样，委托 utils.rand_within_limits）· joint_names_arm/end · joint_index_arm/end(name)
JoyArm.set/get_target_traj(targets) · set/get_current_frame(frame)   # 轨迹桥：应用→规划→控制（TrajFrame；发布即不可变+原子交换，写入口深拷贝隔离）✅
JoyArm.start_motion() / stop_motion(damping=True)   # 运动管线公共启停：**三线程归 JoyArm**（traj-plan/traj-sample/ctrl-step，每周期派发激活成员→运行中 set_solver 即热切换、即时生效）；start 按 Controller.MODE 自动 set_mode_arm + 同步首帧；stop 默认切纯阻尼防下坠（damping=False 关）+ 清当前帧；disconnect/__exit__ 自动先停管线 ✅
# 关键属性：model / pin_model / pin_data / n_arm / n_end / ee_frame_* / arm_limits(_soft) / end_limits(_soft) / arm_zero/home/neutral / end_zero/home/neutral / joint_names_arm/end / tcp_limits / _backend / connected / is_normal（运行状态标志：True=正常；False=控制器循环跳过 cmd 下发，急停/恢复由直连 safe_* 负责；本期无状态管理联动）
#   六域字典：_fkine_solvers/_ikine_solvers/_jacobian_solvers/_dynamics_solvers/_traj_planners/_controllers + _active_name
#   轨迹桥（私有）：_target_traj（List[TrajFrame]，写者=应用线程）/ _current_frame（TrajFrame，写者=规划线程）
```

#### Robotics 独立 API（各域 ABC；实现为教学章节内容）

```python
FkineSolver / IkineSolver / JacobianSolver / DynamicsSolver / TrajPlanner / Controller   # ABC 契约 🟡
TrajPlanner(plan_hz, sample_hz, dt_min_required)：纯计算内核——plan_once(arm)（读桥目标→_check_frame 规则/超时/NaN 剔除→空回退 q_home→_plan）/ sample_frame(t_abs)（按绝对时间采样产帧）；频率属性供管线起节拍（切换变频热重整）🟡
# 并发契约：_plan 系数打包为单一不可变对象原子赋值发布（发布即不可变，同轨迹桥约定）；arm 按属性约定调用：get_target_traj/get_arm_state/arm_home
Controller(ctrl_hz=200) + MODE 类属性：纯计算内核——compute(arm, frame, state)→(模式, 指令字典)；MODE 声明所需电机模式（管线激活/热切换时自动 set_mode_arm）；门控（is_normal/无帧）在管线 ctrl 线程 🟡
# 门控仅作用控制循环，直连 set_arm_command/move_j/safe_* 永不受门控；arm 鸭子契约：is_normal/get_current_frame/get_arm_state/set_arm_command；ctrl_hz 默认 200（DM 串口桥带宽 ~39%，500 近饱和）
# 求解器用法二选一：子类实例.solve(arm, q)（arm 鸭子类型，课堂/单测）或 arm.* 门面（活动成员，应用）
```

#### Backend / 限位 / 数学 / 类型 API

```python
Backend(ABC, cfg) ✅   # 只存硬限位（本体/末端均自解析 config 对应 joints 四键，缺键 ±∞）；属性 arm_limits / end_limits；下发守卫只裁硬限位
send_position/velocity/mit_{arm,end} 守卫模板→抽象内核 _send_*_{arm,end}（限位裁剪唯一执行点；末端标量广播逐电机裁剪、力度按 ±tau_max 数值裁剪；越限告警节流每 0.5s 至多一条（加锁防并发漏节流）；send_action_end 离散不模板化）；软限位仅 JoyArm 加载、运行期替换未实现
get_backend(name) · REGISTRY     # config backend.name 选型 ✅
BackendDMMujoco（DM 机械臂 MuJoCo 仿真后端桩，待实现、未注册 REGISTRY）🟡
clamp_to_limits(targets, limits) · joint_limits_from_model(model) · limits_from_joint_cfgs(joint_cfgs) · soft_limits_from_cfg(cfg, n) · rand_within_limits(limits, size, rng)   # ✅
transforms.py 23 函数（rpy/rodrigues/quat/T/adT/slerp，纯 numpy）✅ · types.py 枚举+数据类 ✅
```

## 4. 文档同步维护

本文件与 [`README.md`](README.md) 是子库两大入口文档，**受众不同、须随代码同步维护**：README 给**使用者**，本文件给 **agent / 开发者**；新内容按受众分流，不互相重复。

- 两份 md 只写**精简概述**；详细规则写进代码 docstring（各包 `__init__.py` 一句话职责 + `:param:`/`:return:`/`:raises:`/章节标记）。
- **每次 `joyarm_code/` 任何变化后必须同步更新本文件**：公开 API 变 → §3；目录变 → §1.1；安装/用法/章节状态变 → README；架构/约定变 → §0-2。**架构约束（§0）变更须人工确认后先改本节再动代码。**
- 提交前自检：新增公开符号已收录、已补 docstring；删除符号已移除；目录树一致。

### 行数上限：≤ 300 行

> 本文件硬性上限 **300 行**。超出时先精简再提交：优先压缩 §3 速查（只留索引级签名）；其次合并 §1-2 重复表述。禁止以"另存新文件"绕过上限；新增内容与精简同步进行。
