# AGENTS.md —— `joyarm_code/` 机械臂代码库的 Agent 项目记忆

> **定位**：`joyarm_code/` 代码库的维护者（Agent）内部文档，不进教程站点。
> **依赖链**：根 [`AGENTS.md`](../AGENTS.md)（教程站点）→ 本文件（代码库）→ [`README.md`](README.md)（用户向）。
> **文档原则**：只写精简概述（结构速览 / API 速查），不复述代码已有细节；详细规则写进代码注释，查细节看源码文件头部的 docstring。
> **去重原则**：每条规则只在权威小节完整陈述一次——架构约束归 §0，架构/配置/限位语义归 §1，命名/编码归 §2，文件与 API 细节归 §3；他处仅留指针（如「（约束N）」「见 §1.3」）。

## 0. 架构约束（持久化，joyarm_core 全体贡献者必须遵循）

> 以下为 joyarm_core 架构**持久化约束**：代码演进不得违背；如需变更约束，须先经人工显式确认后再更新本节。

1. **组合根·单类**：`joyarm/` 的 `JoyArm` 是组合根，基于 config 具体型号配置对其他层功能（backend、robot_model 资产、robotics 各求解器）做**可更换式组合**——能力委托私有子成员，`JoyArm` 只提供公共门面（`arm.*`）。**机械臂新型号 = `config/<型号>.yaml` + `robot_model/<型号>/` 资产 + `backend/backend_*.py`**（backend `REGISTRY` 一行；型号与整机后端 1:1，如 `joyarm_dm` ↔ `backend_dm`；型号名不强制 `joyarm_` 前缀）。
2. **接口先行**：robotics 六域（fkine/ikine/jacobian/dynamics/trajectory/control）= **ABC 接口 + 子类算法实现 + `REGISTRY`注册**（config 选配启用）——均为**纯计算内核、无线程**；自定义算法在所在域 `REGISTRY` 显式加一行注册（import 类 + 表中一行）即接入（**子类名 = 前缀 + 基类名，如 `PinFkineSolver`；注册名 = 类名小写+下划线，如 `pin_fkine_solver`**）。加载链：**工厂 → JoyArm → config → 指定的各子成员**；域未配置即无成员（对应门面调用显性 `RuntimeError`），**配置的注册名无效/实例化失败即构造失败（硬失败，`ValueError`）**。backend = 基类(backend)定义功能接口 + 子类（backend_*）实现具体型号功能，`backend_*` 与 `<型号>` 一一对应（型号名不强制 `joyarm_` 前缀；约束1）。
3. **通用兼容**：`JoyArm` 兼容所有带末端执行器的 5R/6R/7R 臂；末端功能兼容多种执行器（契约见 `backend/backend.py` 的 `*_end` 方法族）。
4. **失败语义（抛异常，不返 None）**：工厂 `joyArm(model)` 四步固定顺序校验（①config 存在且解析为非空 dict → ②`backend.name` ∈ `REGISTRY` → ③robot_model 四路径齐全 → ④`JoyArm` 构造自检），任一步失败**抛携带具体原因的异常**（消息附可用型号列表 = `listArm()` 结果）；`JoyArm` 直用为**硬失败**——config 必需（缺失/自检不通过即抛）、backend 必配、robotics 配置的成员必须全部创建成功（约束2）。
5. **参数排序契约**：接口**通用参数在前**（任何实现都需要，如 ikine 的 `pose`/`frame`/`q0`），**特有参数在后**（仅特定算法需要，如 ikine 数值法的 `tol`/`iters`，解析法可忽略）。
6. **功能全面性 + 字典化**：机械臂的常见功能（读配置、配置自检、连接/失连、使能/失能、状态读取、指令下发、运动状态机、安全监视、运行状态可视化……）在 `JoyArm` 类内完成定义。robotics 六域策略成员**统一字典化**：config 可指定单个或列表规格（全部加载进 `dict[注册名→实例]`，首个为激活、有且仅有一个被激活），运行期经 `set_<域>_solver` 热切换。
7. **robotics 独立运行**：robotics 子模块不得 import `joyarm`（鸭子类型消费 `arm`）；各章算法实现须可脱离 JoyArm 独立运行与测试（直接实例化，构造参数自足）。
8. **教学数据读取路径**：config 在 `JoyArm.__init__` **一次性深拷贝**存入 `self._cfg`（隔离外部引用，`get_config()` 另返回深拷贝）；教学数据（如 MDH 参数）由教学算法经 `arm.get_config()` 从**类内已加载数据**读取，**不重新加载 yaml 文件**；MDH 等非通用参数不做 cfg 跨源自检（对比项见 §1.3）。
9. **开闭原则**：常见变更（新机械臂型号、换电机、结构更新、升级算法）以最小改动完成（新型号三件套见约束1；换算法 = 改 config 注册名 + 注册表一行），对扩展开放、对修改关闭；新增实现不得改动 `JoyArm` 接口签名。
10. **全局优先（执行约束）**：任何代码改动均须追求**全局最优**而非局部最优——源码可见结构必须与运行时行为一致（**所见即所得**：禁止 import 副作用隐式注册、动态改写等造成「打开文件看到的」与「运行时实际的」分歧的机制）；局部加固（防错/便利）与全局一致性冲突时局部让位、必要时阻挡局部修改；改动审查从整个代码库与读者/使用者心智模型出发，不因小失大。
11. **大道至简·第一性原理（执行约束）**：**用最简单的操作做复杂的事**——先看透问题本质，提炼能覆盖整类问题的本质原语并把它用深用透（如config 驱动之于型号差异），一个原语胜过堆叠多个浅层机制；代码实现与文档说人话、精简、易于理解，少用黑科技（动态建类/反射/元编程/隐式回调链等）及术语黑话，仅当常规手段无法实现或存在明显缺陷时才引入深层技术并就地文档说明——简洁是深度理解事物本质的产物，不是能力妥协。
12. **utils 归置（共享基础层）**：凡**跨域、基础、可复用且非特定功能/任务**的功能与数据类型，一律定义于 `utils/`——整个核心代码库的共享功能与数据类型层（依赖规则见 §1.2）；特定于某域/某层（robotics 六域、backend、joyarm）的功能不下沉 utils（如 JoyArm 私有周期线程机制），通用能力不得在域内重复实现——先复用 utils，缺则补入。

13. **状态机 / 监控 / 显示职责分离**：FSM 与 plan/ctrl 只负责状态迁移、维持与运动执行，不因发送失败、算法异常或到位超时设置安全事件；运行期自动安全判定与事件/目标状态设置只归 monitor，自定义规则只在监视线程执行。运动状态机启动入口仅校验连接、新鲜有效反馈、无电机故障及算法配置，初始化 `current=DISABLE / target=MOTION / event=NORMAL`；启停生命周期与外部显式 `set_event` 保留。状态可视化只读显示。

## 1. 架构细化

> 补充 README「基本架构」：分层图与核心概念见 README，本节给文件级细节。

### 1.1 目录树（目录级速览；文件职责/章节/状态见 §3.1）

```
joyarm_code/
├── joyarm_core/                     # ★ 核心 SDK（ROS2-free）
│   ├── __init__.py                  #   公开 API 扁平导出（__all__）
│   ├── utils/                       #   基础层（跨域共享功能/数据类型，约束12）
│   │   ├── types.py                 #     数据类 / 枚举
│   │   ├── transforms.py            #     SO(3)/SE(3) 纯 numpy 数学
│   │   └── limits.py                #     限位裁剪 + 硬/软/tcp 限位构建辅助
│   ├── robotics/                    #   算法层：一域一子包（ABC + 默认实现 + 显式 REGISTRY）
│   │   ├── fkine / ikine / jacobian / dynamics     #     求解器四域
│   │   └── trajectory / control                    #     规划域 / 控制域
│   ├── backend/                     #   通信层（整机 Backend + name 选型）
│   │   ├── backend.py               #     Backend（整机 ABC：三段式——公开薄委托/族通用 impl/29 个抽象内核：24 单 joint + 4 总线级 + 1 整组动作）✅
│   │   ├── backend_dm.py            #     BackendDM（DM 整机 7 电机，U2CAN 串口转 CAN 桥；RX 收取线程+全操作内核验证）✅
│   │   ├── backend_dm_mujoco.py     #     BackendDMMujoco（DM 机械臂 MuJoCo 仿真后端桩；文件待创建）🟡
│   │   ├── backend_template.py      #     BackendTemplate（子类实现模板：复制为实现，不入 REGISTRY）✅
│   │   ├── backend_verify.py        #     Backend 子类真机验收 CLI（上层视角 00~50 六组分项；独立工具，不入门面）
│   │   ├── u2can/                   #     厂商 DM 参考库（协议参照用，不依赖不导入）
│   │   └── __init__.py              #     REGISTRY（backend_dm）+ get_backend(name) 选型
│   ├── joyarm/                      #   设备模型层（主类 + FSM + 监视器 + 可视化 + 对象创建入口）
│   │   ├── joyarm.py                #     JoyArm 主类：初始化自检/硬件委托面/六域两线程/数学门面/MotionFSM/safe_monitor/状态可视化
│   │   └── __init__.py              #     JoyArmFactory / joyArm / listArm / load_config（对象创建入口）
│   ├── robot_model/                 #   URDF + meshes 资产（运行期加载；robot=纯资产，加载逻辑在 JoyArm）
│   └── config/                      #   per-model YAML（joyarm/robotics/backend 三段）
│       ├── joyarm_dm.yaml           #     正式型号配置
│       └── template.yaml            #     型号配置模板（复制为 <型号>.yaml 填写；本文件不入 listArm）
├── joyarm_ros2_ws/                  # ROS2 colcon 工作空间（规划，Ch10 落地时创建；见 §1.4）
├── chapt/                           # 章节教学示例脚本（一次性；基础设施优先复用 joyarm_core，章节算法可自行实现）
├── test/                            # 测试套件（test_backend_base.py：Backend 基类 Dummy 全内核模拟；test_robotics.py：robotics 六域；test_joyarm.py：JoyArm FSM 离线回归；joyarm_test.py：手动冒烟脚本）
├── quickstart/                      # 快速上手示例脚本（7 个；CLI/GUI 入口规划见 §2.3）
├── pyproject.toml                   # 工程配置（可编辑安装 joyarm_core）
├── README.md                        # 用户向入口
└── AGENTS.md                        # 本文件
```

### 1.2 依赖规则（无环明细）

- `utils` → 仅 numpy；
- `robotics` / `backend` → 依赖 `utils`（+ numpy）；robotics 的 Pin 默认实现另需 `pinocchio`；求解器按属性约定消费 `arm`（经公开门面/属性），**不 import `joyarm`**（约束7）；
- `joyarm` → **组合根**（组装方式见约束1）：按 config `robotics:` 段查各域 `REGISTRY` 组装成员、按 `backend:` 段 `name` 经 `get_backend` 构建后端，运行期加载 `robot_model/`、`config/`；**运行状态可视化在 joyarm 包内**；
- `joyarm_ros2_ws/src/*`（规划）→ 依赖 `joyarm_core` + `rclpy`；核心包保持 ROS2-free。

### 1.3 配置（yaml 分段 + 限位语义）

`config/<model>.yaml` 三段：`joyarm`（设备模型）/ `robotics`（六域算法选型）/ `backend`（通信层 + 硬限位四键）。**型号唯一标识 = yaml 文件名**（= 工厂入参；URDF 目录 `robot_model/<型号>/` 同名）；backend 与机械臂型号 1:1（约束1）；三段数值叶一律须有限且非空（null/NaN/inf 构造期硬失败）：

```yaml
joyarm: {frame_end, arm_mdh, T_linkn_endtcp, joint_limits_soft_arm/end, tcp_limits, home_arm/end, MotionFSM, arrival_tol, runtime}   # 软限位四键直值 {q_min,q_max,dq_max,tau_max}（q_min/q_max 必配、缺键拒绝构造，dq_max/tau_max 缺键回退 ±∞；标量广播或 n/n_end 元列表，须位于硬限位内；供上层状态判断，不参与指令裁剪）；zero=全零（构造期校验在各关节硬限位内）、neutral=软限位中点（代码固定，不经 config）；home 逐关节须在软限位内；arm_mdh（6×4：α/a/d/θ）/T_linkn_endtcp 为教学数据（约束8）；tcp_limits.workspace_box 为 (3,2) 每行 [min,max]（兼容旧 (2,3) 布局），v_lin_max/v_ang_max/f_max/t_max 须 >0；MotionFSM 四键 {fsm_hz(缺省50), damping_kd(必配，DAMP 态 MIT kd，标量广播或 n 元列表), damping_k_gravity(必配，DAMP 态重力补偿率 [0,1]，1=完全补偿；dynamics 域未配置时退化纯阻尼), monitor_hz(缺省=fsm_hz)}；arrival_tol 缺省 0.05/1e-3/1e-2；runtime.move_wait_timeout 缺省 10.0（stop_motion_fsm 回零等待超时；同段其余键暂无消费方）
robotics: {六域契约规格}   # 六域选型（注册名 / {name, **参数} / 规格列表，全部加载、首个激活）；显式配置，注册名无效/实例化失败即硬失败（约束2）；traj 每规格必配 plan_hz、control 每规格必配 ctrl_hz（int 或整值浮点 ≥1，JoyArm 剥离消费、不传构造器；电机模式由控制器 MODE 类属性声明）
backend: {name: backend_dm, 可选运行键 state_refresh_hz/write_settle（默认 20Hz/0.1s；state_refresh_hz=状态新鲜度承诺、刷新线程按 0.25/hz 四倍频绝对对齐节拍供给，读侧帧龄 >0.5/hz 补查询后即返当前快照（t 如实带旧帧时刻，不上抛；新鲜度由上层按 t 判定）；backend_dm 另有 rx_hz 默认 1000Hz、上限 5000Hz=RX 收取线程节拍）, arm/end 同构：{channel, protocol, baud_rate, default_mode（连接后自动设置，缺省 MIT；无 MIT 型号配 position/velocity）, joints}}   # joints 条目：四限位键 q_min/q_max/dq_max/tau_max（= 硬限位来源）+ 协议标识三键 motor_id/feedback_id/model + 能力段 MIT {kp,kd}（MIT 指令发送默认）与 POS_VEL {pos_kp,pos_ki,vel_kp,vel_ki,vlim,flim}（切 POSITION/VELOCITY 随写增益寄存器；整段缺省静默 NaN、段在键缺 warn），均由 backend `__init__` 自解析；cfg.backend ↔ URDF 硬校验（任一不符构造失败，ValueError 列全部问题）：arm 关节须 joint1~jointN、end 须 joint_end1~joint_endM 且与 URDF 驱动关节同名同序；只登记非 fixed 非 mimic 驱动关节且须全部登记；四限位键与 URDF <limit> 逐键差 ≤1e-4；q_min≤q_max；joint/link 名无重复
```

**限位语义**：**backend 下发指令只裁硬限位**（`Backend.send_*` 守卫模板，唯一执行点）；软限位由 JoyArm 直配加载（`joint_limits_soft_arm/end`，必配）作上层状态判断依据（到位判断容差域、neutral 中点、home 校验区间、`rand_q_*` 缺省采样域），不拦指令；config 改动重启生效，**不回写 yaml**。硬限位解析鲁棒性（`limits_from_joint_cfgs`）：限位键值为 NaN/inf → 该键回退未限位（`q_min`→`-inf`、其余→`+inf`）并 warn（防 `np.clip` 产出 NaN 指令静默下发）；`q_min > q_max` 由 cfg↔URDF 硬校验拦截（utils 层仅 warn，勿依赖）。

### 1.4 ROS2 工作空间（`joyarm_ros2_ws/`，**规划——Ch10 落地时创建，当前未建**）

标准 colcon 工作空间，承载所有 ROS2 功能包（第十章起）；`build/ install/ log/` 为构建产物。规划目录：`src/joyarm_node/`（ament_python：`arm_node.py` 每机械臂一节点 + `adapters.py` 内部类型 ↔ ROS2 消息互转，四元数 wxyz↔xyzw 换序）、`config/joyarm_dm.yaml`（model/rate/offline）、`launch/arm.launch.py`（+robot_state_publisher/rviz2，namespace 即多臂）。接口约定：状态广播走话题（`joint_states`、`~/tcp_pose`）；连接/夹爪走服务（std_srvs）；轨迹执行规划为 action（自定义 `joyarm_interfaces`）。规划中的应用包：`joyarm_teleop`(Ch12)、`joyarm_vision`(Ch14)、`joyarm_agent`(Ch15)；纯算法管线规划放 `joyarm_core/apps/`（ROS2-free）。

**环境与构建**：`python3 -m pip install --user -e joyarm_code` → `source /opt/ros/humble/setup.bash` → `colcon build` → `source install/setup.bash`。

> ⚠️ 边界：`joyarm_node` 不做 ros2_control hardware_interface、不追求直接兼容 MoveIt2。
> ⚠️ 陷阱①：package.xml 的 XML 注释中禁止双连字符 `--`，否则 catkin_pkg 静默回退，`ros2 run/launch` 找不到包。
> ⚠️ 陷阱②：ament_python 包 `setup.py` 须声明 `tests_require=["pytest"]`，否则 `colcon test` 发现 0 用例。

## 2. 命名与设计约定

### 2.1 命名约定（全仓库强制，含 `chapt/`）

- **类名 = 驼峰（PascalCase）；文件名 = 小写 snake_case。**
- **设备模型**（`joyarm/`）：`JoyArm` = 组合根单类（无型号子类）；`joyArm(model)` 外部推荐创建入口（`JoyArmFactory` 实例、可直接调用；失败语义见约束4）；`listArm()` 列可用型号。文件 `joyarm.py` / `__init__.py`。
- **Backend 整机两层**（`backend/`）：根 `Backend`（`*_arm`/`*_end` 方法族）→ 型号层 `BackendDM`（1:1 对应见约束1）。`REGISTRY` + `get_backend(name)` 供 config 选型。
- **属性约定**：类内成员一律 `_` 前缀私有，类外访问经只读 `@property`（限位/位形数组只读化后零拷贝返本体）或 `get_*/set_*` 方法；型号名/教学数据经 `arm.get_config()` 读（约束8）。常用只读属性：`pin_model`（pinocchio 缩减构型，nq=n_arm；**勿与型号名混淆**）/ `n_arm`/`n_end` / `joint_limits_hard_arm/end` / `joint_limits_soft_arm/end`（限位语义见 §1.3）/ `zero_/neutral_/home_arm/end` / `tcp_limits` / `plan_hz`/`ctrl_hz` / `is_connected_arm/end`/`is_enabled_arm/end`、`joint_state_arm/end`（均与 backend 同名直连；joint_state 为缓存快照被动只读、不触发读帧，需实时走 `get_state_*`）；关节名查询 `get_joint_names_arm/end(joint)`（1 基序号口径，全文统一）；整机后端 = `_backend`（私有，必配，经 §3.3 委托面访问）。求解器形参用 `arm`（鸭子类型）。
- **六域策略成员**：每域两成员 `_<域>_solvers`（`dict[注册名→实例]`）+ `_activated_<域>_solver`（激活**对象**指针）；域缩写 `fkine/ikine/jac/dyn/traj/ctrl`（cfg 段键 `fkine/ikine/jacobian/dynamics/trajectory/control`）。公开方法三件套：`get_<域>_solvers()` / `set_<域>_solver(name)` / `get_activated_<域>_solver()`（热切换协议见 §3.3）。
- **方法命名**：本体带 `arm`、末端带 `end` 一一对应（`enable_arm`/`enable_end`…，末端离散动作 `set_action_end(action, **kw)`）；三模式整组下发 `send_mit/send_position/send_vel_{arm,end}`（无逐 joint 子集参数）；`set_mode_*` 默认 MIT；参数读写 `read/write_param_{arm,end}`（族级：读返列表、写等长列表）；数学门面前缀 `solve_`（solve_fkine/solve_ikine/solve_ikine_all/solve_idyn）或 `get_`（get_jac/get_jac_pinv_damped/get_mass_matrix/get_coriolis/get_gravity）；到位判断 `is_in_q_arm/end`、`is_in_pose_arm`；目标通道 `set/get_target_trajframe` + 便捷封装 `set_target_joint/set_target_cart`；FSM `start/stop_motion_fsm`、`get_current/get_target_motion_state`；安全 `set/get_event`、`start/stop_monitor`、`set_monitor_rules`；可视化 `start/stop_state_viz`；销毁 `close()`。
- **算法可换（成员即策略）**：每域 = ABC + `REGISTRY`（实现一节点一文件，各章新增）。

### 2.2 编码与运行约定（架构类原则见 §0，此处不重复）

| 原则 | 落地 |
|:--|:--|
| **离线模式** | 计算类显式传参即离线可用（数学门面缺省值取硬件状态，无状态抛 `RuntimeError`）；执行类需连接 |
| **共享类型单定义** | 跨层类型只在 `utils/types.py`（约束12）；ROS2 消息仅在 ws 的 `adapters.py` 互转 |
| **核心轻依赖** | 核心包代码仅 `numpy`/`pin`/`pyyaml`/`pyserial`/`meshcat`（pyserial 供硬件后端串口、meshcat 供可视化 3D——直接依赖顶层导入，初始化失败降级为仅 HTML 面板）；`rclpy` 仅 ws；matplotlib/PySide6/mujoco 为工程级教学依赖 |

编码约定：`q=(n,)`、`T=(4,4)`、角度弧度、四元数 wxyz；关节序号 1 基（cfg 列表位置）；`ControlMode` 三态（纯力矩经 MIT `kp=kd=0`）；接口参数排序遵约束5；**运行状态可视化在 joyarm 包内**（joyarm.py §8，HTTP 轮询零新增依赖 + meshcat 直接依赖）；异常消息统一格式**『文件名 - 故障的功能：具体原因』**（含 NotImplementedError 教学桩，新代码一律遵守；backend 子类模板桩例外——前缀自动取注册名 `self._name`（cfg `backend.name`，按 1:1 约定 = 文件名去 `.py`），未传 name 回退占位前缀 `backend_<型号>.py`）。

### 2.3 库边界

SDK（`arm.xx`）→ `joyarm_core/`；ROS2 → `joyarm_ros2_ws/src/`（规划）；快速上手示例脚本 → `quickstart/`；教学脚本 → `chapt/`。

## 3. 模块与 API 速查（随代码进度实时更新）

### 3.1 模块导览（✅ 已实现 / 🟡 教学章节待实现）——文件级细节权威表

| 文件 | 类 / 关键定义 | 章节 | 状态 |
|:---|:---|:---:|:---:|
| `utils/types.py` | 枚举（ControlMode/MotionState/SafetyEvent）+ 数据类（Pose/TrajFrame/JointState/Wrench/Twist/Accel/JointLimits/TcpLimits/IKResult/ComplianceParams 等） | Ch2 | ✅ |
| `utils/transforms.py` | 23 个纯 numpy 函数（rpy/rodrigues/quat/T 族/slerp…） | Ch2 | ✅ |
| `utils/limits.py` | 限位助手：硬/软/tcp 限位构建 + 指令裁剪 + 限位内采样（签名见 §3.3） | Ch11 | ✅ |
| `backend/backend.py` | `Backend(ABC)` 整机契约：三段式（公开薄委托 → 族通用 `_impl`（守卫/裁剪/验证/保活线程）→ 29 个抽象内核：24 单 joint + 4 总线级 + 1 整组动作）；三模式发送（cfg `default_mode` 可配、缺省 MIT，connect 自动 best-effort 设定）；动态数据读取：`get_mode_*`/`get_state_*` **读时刷新**；`joint_state_*` 只读快照 @property（被动直读槽位、不触发读帧，供监视器/可视化等无需实时者）；`is_connected_*`/`is_enabled_*` 三值标志 @property；无 `mode_*` 直读属性；`send_mit_*` 签名 `(q, dq=None, tau=None, kp=None, kd=None)`（`q` 必填；`dq`/`tau` 缺省置 `zeros(n)` 不广播；`kp`/`kd` 缺省取 cfg MIT 增益成员；三指令校验全在 `*_impl`、公开接口零校验）；cfg 预解析硬限位/发送默认/POS_VEL 四增益（切 POSITION/VELOCITY 随写寄存器）/baud/协议标识三键/两运行键（state_refresh_hz、write_settle）；`cfg` @property 返 MappingProxyType 只读视图、`joint_limits_*` 数组 writeable=False 零拷贝；成员变量全私有化；运行期 warn/info 直接输出 | Ch2 | ✅ |
| `backend/backend_dm.py` | `BackendDM`（DM 7 电机经 U2CAN 串口转 CAN 桥，模块私有 `_DMMotor`/`_DMBus`：RX 收取线程纯收取默认 1000Hz（cfg `rx_hz`，上限 5000）、状态快照单一 tuple 原子换入、状态读非阻塞（新鲜直返/陈旧发 0xCC 即返当前快照、t 如实带旧帧时刻不上抛，新鲜度由上层按 t 判定）；模式寄存器 RID10 MIT/VEL/Torque_Pos=1/3/4（POSITION 走 0x300+id pos_force 帧）、切 POSITION/VELOCITY 随写 4 增益（RID25-28，NaN 跳过）；反馈帧含双温度（data[6]=T_MOS、data[7]=T_Rotor）；全操作内核验证+重试（参数 0x55 回显核对/0x33 应答、使能/失能/设零/清错 0xFC/FD/FE/FB 新状态帧核对 err，应答按电机+RID+时刻配对防洗白）；僵尸总线重连先关再重建、断连清扫无主总线；组帧/解码与厂商库比特级一致；无 save；末端离散动作 open/home/close/position（-5.0/-3.0/0.0 rad/kwarg `position` 透传：自动切 POSITION 随写增益 + cfg vlim/flim 发送默认 + 位置经基类管道裁硬限位） | Ch6/13 | ✅ |
| `backend/backend_dm_mujoco.py` | `BackendDMMujoco` DM 机械臂 MuJoCo 仿真后端桩（与 `backend_dm` 同型号配套；文件待创建，实现后 REGISTRY 注册 `backend_dm_mujoco`） | — | 🟡 |
| `backend/backend_template.py` | `BackendTemplate` 子类实现模板（29 个抽象方法全 NotImplementedError，桩消息前缀自动取 `self._name`（cfg `backend.name` 注册名，未传回退占位前缀 `backend_<型号>.py`，复制后无需手改）+ 逐方法实现指引注释 + 统一 `:param:/:return:/:raises:` 契约块（类型/单位/范围/维度与基类已做校验）+ 全量私有成员台账（直接消费 / 按需借用 / 勿动分档）+ 最小可用子集说明：必实现 14（连接/断连/使能/失能/读状态/读模式/设模式 ×2 族）、按需 15（参数读写/设零/三模式发送/离散动作/清错），不支持的能力保留桩由基类转 warn 降级；复制为 `backend_<型号>.py` 用，不入 REGISTRY、不被 `__init__` 导入；含 DM 类关节电机/Feetech 类舵机双家族差异指引（覆盖混合硬件 arm 电机 + end 舵机、无 end 型号适配）、联调自检清单与「实现后 docstring 改写为型号介绍说明」指引（七步接入），受 test_69 冒烟保护：29 桩全量消息格式校验（含前缀随 name 自动切换）+ 基类内核集漂移守护 | — | ✅ |
| `backend/backend_verify.py` | Backend 子类真机验收 CLI（上层视角只调基类公开面；00~50 六组分项 = 初始化/连接/只读/控制/运动/清错，PASS/SKIP/FAIL 三分 + 高危交互确认；`--param`/`--action` 适配子类） | — | ✅ |
| `robotics/fkine/` | `FkineSolver(ABC)`（solve 单内核）+ `PinFkineSolver`（forwardKinematics+updateFramePlacement，含帧名哨兵防御；每算新建私有 pin.Data）✅ + `MdhFkineSolver`（fkine_solver_mdh.py：MDH 白盒 FK——config `arm_mdh` 逐关节连乘 + `T_linkn_endtcp` 右乘，仅末端帧；与 URDF 标定一致至 1e-13，解析 IK 复用其计算函数）✅ | Ch2 | ✅ |
| `robotics/ikine/` | `IkineSolver(ABC)`（solve 单解 / solve_all 全解默认 `NotImplementedError`）+ `PinIkineSolver`（LM 自适应阻尼 + 多起点重启，不可达返 success=False）✅ + `AnalyticIkineSolver`（ikine_solver_analytic.py：2/3/4 轴平行、5/6 轴相交构型解析解——枚举肩/肘/腕分支 ≤8 组候选、MDH FK 回代剔伪解去重、±2π 限位内归位；solve 软限位剔除后取距 q0 最近，solve_all 不剔越限且 frame 可缺省=末端帧；首次求解缓存 MDH/TCP 逆变换，按 arm 身份复用、配置变更须重建实例）✅ | Ch3 | ✅ |
| `robotics/jacobian/` | `JacobianSolver(ABC)`（衍生量模板：jac/jac_pinv_damped）+ `PinJacobianSolver`（computeFrameJacobian，基座系）✅ | Ch4 | ✅ |
| `robotics/trajectory/` | `TrajPlanner(ABC)`（plan/sample 两内核：目标单点/序列 → 轨迹存对象内、按绝对时间采样产帧）+ 双实现：`SingleJointHoldTrajPlanner`（单点直传：仅取单帧 q 恒定输出、dq/ddq=0；q 空/维度不符/多帧回退 q_home）与 `SingleJointCubicTrajPlanner`（单点三次多项式：当前 q/dq 起点 + 目标 q/dq 终点四约束，T=目标时刻−当前、τ 钳位超末保持；无效回退恒定 q_home、已过期直接恒定发布目标），config 默认激活直传 ✅ | Ch5 | ✅ |
| `robotics/dynamics/` | `DynamicsSolver(ABC)`（idyn/M/C/G 四内核）+ `PinDynamicsSolver`（rnea/crba 对称化/nonLinearEffects）✅ | Ch8 | ✅ |
| `robotics/control/` | `Controller(ABC)`（step 内核 + `MODE` 类属性声明所需电机模式：基类缺省 MIT、子类覆写，FSM 在 MOTION 进入/维持时依此自愈模式与使能）+ `JointPositionController` 关节位置控制器（`MODE=POSITION`；step 跟踪当前帧 q 经 `send_position_arm` 下发，NaN/缺 q 拒绝下发）✅ | Ch6 | ✅ |
| `joyarm/joyarm.py` | `JoyArm` 主类：两阶段初始化（私有成员声明 → 按依赖序赋真值）+ cfg 全套自检（三段数值叶有限非空 + zero∈硬限位 + cfg↔URDF 硬校验）+ pin 缩减模型（锁定非 cfg 名单活动关节 → nq=n_arm）+ 硬件委托面（§3.3）+ 六域求解器与热切换 + 目标帧通道 + plan/ctrl 两线程 + 数学门面 9 方法（q/dq 维度校验、frame=0 合法直传）+ MotionFSM（六状态 + 路线表 + start/stop + 进入/维持自愈，失败保留状态，下拍重试（职责见约束13）；最终收尾禁止自愈后停线程再失能）+ SafetyEvent/safe_monitor（硬规则 + 自定义规则）+ 运行状态可视化（HTTP 轮询 + meshcat 依赖 + zmq 超时看门狗降级与 server 回收）+ `close()` 销毁路径；模块级 `load_config` 与私有周期线程机制 `_run_periodic`/`_PeriodicThread`（含 prepare() 预登记线程句柄；FSM/监视/可视化/运动线程共用，不入 utils） | — | ✅ |
| `joyarm/__init__.py` | `JoyArmFactory`/`joyArm`/`listArm`/`load_config`（对象创建入口，四步校验失败语义见约束4；`listArm` 三条件过滤与失败消息共用实现） | — | ✅ |
| `joyarm/joyarm.py` 内 `_VizWindow` | 监控页原生窗口封装类（JoyArm 成员 `_viz_window`：init 占位 None、start 时创建赋值、stop 终止；`launch()` 经 `python -c` 拉起子进程跑 `child_main()`——`QMainWindow`+`QWebEngineView` 内嵌 Chromium 加载三栏页，内置早夭看护（主动停止标志防误告警、stop 超时 kill 兜底）；零数据逻辑，数据仍来自 HTTP 服务；关窗即退出子进程） | — | ✅ |
| `config/joyarm_dm.yaml` | 型号 YAML 三段（结构与限位语义见 §1.3） | — | ✅ |
| `quickstart/01_offline_model.py` | 离线模型与数学：无需硬件的模型信息浏览 + 正逆运动学/雅可比/动力学求解与六域求解器切换 | 使用篇 | ✅ |
| `quickstart/02_connect_read.py` | 连接与状态读取：连接、使能与状态读取（两种状态口径、故障码与电机参数读取） | 使用篇 | ✅ |
| `quickstart/03_direct_command.py` | 直接指令与夹爪：FSM 未运行时的直接指令（位置/速度/MIT 三模式与末端夹爪动作） | 使用篇 | ✅ |
| `quickstart/04_fsm_motion.py` | FSM 关节与笛卡尔运动：状态机闭环运动（关节/笛卡尔目标、TrajFrame 通道与安全事件切换） | 使用篇 | ✅ |
| `quickstart/05_monitor_viz.py` | 监视器与可视化：自定义安全监视规则与三栏实时可视化面板 | 使用篇 | ✅ |
| `quickstart/06_teach_replay.py` | 拖动示教与回放：阻尼态拖动示教录制轨迹并逐路点回放 | 使用篇 | ✅ |
| `quickstart/07_pick_demo.py` | 综合抓取：逆解 + FSM 运动 + 夹爪的完整抓取—放置闭环 | 使用篇 | ✅ |

### 3.2 组装与数据流

```
JoyArm（单类组合根：_backend + 六域成员字典 + _cfg + FSM/监视/可视化线程）
   │  config 驱动：型号名→URDF（pin_model 缩减模型）；backend.name→后端（必配）；
   │  robotics:→各域 REGISTRY 字典化组装（首个激活；域未配置=空，门面显性报错）
   │  教学数据（MDH 等）：算法层经 arm.get_config() 读类内 config（约束8）
   │  运动数据流：应用线程 set_target_*（原子发布）→ plan 线程逐拍规划（存激活规划器内）
   │  → ctrl 线程逐拍 sample(time.time())→控制器 step→经 JoyArm 委托面下发 _backend
Backend(ABC) ──▶ BackendDM ✅（U2CAN 串口转 CAN 桥）
六域 ABC + 内置默认实现（Pin 四域 + MDH FK + 解析 IK / single_joint 双规划器 hold+cubic / joint_position_controller）──▶
   章节自定义实现显式入表后并列选配（fkine Ch2 / ikine Ch3 / jacobian Ch4 /
   traj Ch5 / control Ch6/8/9 / dynamics Ch8）
```

### 3.3 API 速查

> **精简签名**：`name(关键参数) → 返回 — 一句话`。完整说明见源码 docstring 或 `help(符号)`。
> 域缩写：<域> ∈ {fkine, ikine, jac, dyn, traj, ctrl}。

#### 工厂与型号（joyarm/__init__.py）

```python
joyArm(name) → JoyArm      # 工厂实例直调（= JoyArmFactory.create）：四步校验（见约束4）任一失败抛 ValueError ✅
listArm() → list[str]      # 可用型号三条件过滤（三顶层键 + backend.name ∈ REGISTRY + 四路径齐全；模板 yaml 由路径条件自然排除）✅
load_config(model) → dict  # config 加载（缺失/解析失败/顶层非映射抛 ValueError）✅
```

#### 设备模型 JoyArm 公开面（joyarm.py）

```python
# 配置与只读属性
JoyArm.get_config() → dict    # 深拷贝快照（教学数据读取口，约束8）✅
@property：pin_model · n_arm/n_end · joint_limits_hard_arm/end · joint_limits_soft_arm/end · zero_/neutral_/home_arm/end · tcp_limits · plan_hz/ctrl_hz（随热切换更新，未配置 None）· is_connected_arm/end · is_enabled_arm/end · joint_state_arm/end（backend 同名直连，缓存快照被动只读）    # 限位/位形数组只读化零拷贝返本体 ✅
# 关节查询与采样
JoyArm.get_joint_names_arm/end(joint="all"|1基序号) · rand_q_arm/end(joint_limits=None)    # 缺省软限位内均匀采样 ✅
# 硬件后端委托面（§4.1 总表同构直连；动态数据读时刷新）
connect/disconnect_arm/end · enable/disable_arm/end · read/write_param_arm/end · get/set_mode_arm/end · get_state_arm/end() → JointState · set_zero_arm/end · get/check/clear_error_arm/end · send_mit_arm/end(q,dq=None,tau=None,kp=None,kd=None) · send_position_arm/end(q,vlim=None,flim=None) · send_vel_arm/end(dq) · set_action_end(action,**kw) ✅
#   FSM 外部指令互斥：FSM 运行中外部线程调 arm 侧指令类方法（enable/disable/set_mode/set_zero/clear_error/send_*_arm）空操作+warn；ctrl 线程仅在 MOTION、NORMAL 事件且当前控制器已由 FSM 确认就绪时豁免；句柄经 prepare() 预登记；运行中 plan/sample/step 失败或控制反馈缺失只跳拍，异常交周期线程记日志；end 侧与参数读写、状态读取不受互斥
# 六域求解器管理（每域三方法）
JoyArm.get_<域>_solvers() → list[str] · set_<域>_solver(name) → bool · get_activated_<域>_solver() → str ✅
#   热切换协议：traj 域在运动锁内先试 plan，成功才替换并同步 plan_hz；ctrl 域在同锁内仅选择算法/更新 ctrl_hz（可离线、不切模式），撤销下发许可，由 MOTION 进入/维持按 MODE 核验模式/使能后恢复；plan/sample/step 与切换串行，join 不持锁；其余四域直接换指针；坏名保持原激活
# 目标帧通道（应用线程写、规划线程每拍整体重读；deepcopy 后单条赋值原子发布）
JoyArm.set_target_trajframe(单帧|序列|None) · get_target_trajframe() · set_target_joint(t, q, dq?, ddq?, tau?) · set_target_cart(t, pose, twist?, accel?, wrench?)    # 便捷封装内部组帧走同一通道；t=Unix 秒、q=(n,)、pose=Pose 或 (7,) ✅
# 数学门面 9 方法（显式传参离线可用；q/dq 缺省取 joint_state_arm 被动快照（无有效快照抛 RuntimeError）、frame 缺省 frame_end（0=universe 帧合法直传）、ddq 缺省零；显式 q/dq 验维度 (n_arm,)、pose 非 Pose 抛 TypeError；域未配置抛 RuntimeError）
JoyArm.solve_fkine(q?, frame?) → Pose · solve_ikine(pose, frame?, q0?, **kw) → IKResult · solve_ikine_all(pose, frame?)（仅解析法；数值法转 RuntimeError）· get_jac(q?, frame?) → (6,n) · get_jac_pinv_damped(q?, frame?, damping=1e-3) → (n,6) · solve_idyn(q?, dq?, ddq?) · get_mass_matrix(q?) → (n,n) · get_coriolis(q?, dq?) · get_gravity(q?) ✅
# 到位判断（读 joint_state_arm/end 被动快照，过期/未来/无效时间戳或位置维度错误均判未到位；tol_q 标量统一或 n 元逐关节、缺省 cfg joyarm.arrival_tol）
JoyArm.is_in_q_arm/end(q, tol_q?) → bool · is_in_pose_arm(pose, tol_pos?, tol_rot?, frame?) → bool ✅
# 运动状态机 MotionFSM（六状态 DISABLE/ZERO/HOME/MOTION/HOLD/DAMP；home(3) 中枢路线表；状态内硬件操作直连 _backend；进入/维持均核验模式与使能，不符自愈、失败跳过下发，失败/异常不改安全事件（约束13）；ZERO 到位与 DISABLE 只保持失能、HOLD 自愈不重采位置；进入失败保留旧态必要缓存，不盲目恢复旧运动；周期等待被 stop 唤醒后不再执行一步）
JoyArm.start_motion_fsm()（前置已连接、反馈新鲜有效、无电机故障且 traj/ctrl 域已配置，否则 RuntimeError；非阻塞；自动 DISABLE→HOME→MOTION，3→4 每次复核新鲜 home 到位反馈；初始化 DISABLE/MOTION/NORMAL；不调用监视规则）· stop_motion_fsm()（阻塞安全停机：事件危险度高于 ZEROING 时直接失能不回零，否则回零失能、等待中升档或失连立即收尾；超时 move_wait_timeout 强制推进；最终收尾先禁止自愈/ctrl 再停线程、最后失能；未连接抛 RuntimeError）· get_current_motion_state() · get_target_motion_state() · wait_for_motion_state(fsm_state, timeout=5.0)（只读轮询等待，不受 FSM 互斥拦截：到达返 1；超时/FSM 未运行（含中途停止）/非枚举返 0 各带 warn；timeout 非正有限抛 ValueError）✅
# 安全事件与监视器（事件→目标状态映射：NORMAL→MOTION…DISABLE 危险度最高；FSM 每拍消费，实际响应延迟还受总线 I/O 与线程调度影响）
JoyArm.set_event(SafetyEvent) · get_event() · start_monitor() / stop_monitor() · set_monitor_rules(rules)    # 监视器独立 daemon 线程（monitor_hz）：硬规则（error≥2→DAMPING；断连或连续 0.1s 无新鲜快照→DISABLE，快照过期过滤判 None）与自定义规则命中合并、统一取危险度最高、事件锁保证比较与升档发布原子化；rule(state_arm, state_end, motion, event) → SafetyEvent|None ✅
# 运行状态可视化（HTTP 轮询、零新增依赖、纯被动快照零总线帧、与 FSM/规划控制线程正交）
JoyArm.start_state_viz(traj_points=200) / stop_state_viz()    # 单页面三栏监控网页（中间 1:1、两侧 1:2 内部纵滚），start 后默认自动弹出 PySide6 QtWebEngine 原生窗口（joyarm.py 内 `_VizWindow` 子进程加载同页、JoyArm 成员 `_viz_window` 持有，stop 随手关窗；`JOYARM_VIZ_NOWINDOW=1`/无显示器/启动失败降级仅打 URL 日志）：左静态（/static.json，start 时装配一次、页面加载拉取一次，六组：标识（型号/frame_end/URDF）/关节与限位（arm、end 两表：行=硬限位上/下限、软限位上/下限）/末端与限位/特征位形（arm、end 两表 zero/home/neutral）/控制默认值（h2 下四子表：电机默认模式、运动状态机、MIT 模式默认值、POS_VEL 电机内部增益；cfg 原文折叠）+ 中 3D（iframe 嵌 meshcat（初始化失败降级占位）；arm+end 联动的可视化专属缩减 pin 模型（数学用 _pin_model 不动），display q 按关节名映射快照拼装、无效保持上一帧、构建失败降级 arm-only）+ 右动态（JS ~5Hz fetch /state.json：运行状态两表（表1 行=arm/end×列=连接/使能/状态快照；表2 状态机运行状态（fsm.running）/当前/目标/安全事件）、末端位姿与速度（首行末端系 frame_end）、arm/end 关节状态（行=数据项 q/dq/tau/T_mos/T_rot/err、表头=关节名）、当前求解器（traj/ctrl 附频率；目标帧摘要不渲染））；全部单元格居中、fitTables() 表超栏宽整体等比缩字号（兼容任意 n_arm+n_end）；单一只读 daemon 线程（绝不下发指令）；meshcat 刷新带 zmq 超时看门狗（连续失败降级仅 HTML）、stop 回收 meshcat-server 子进程；轨迹显示可选（需 MOTION 态，开关在中栏 3D 角部）✅
# 销毁（支持 with 上下文）
JoyArm.close()    # stop_motion_fsm → stop_monitor → stop_state_viz → disable_arm/end → disconnect_arm/end → backend.close（各步异常转 warn 不阻断收尾；arm 断连时仍停线程清缓存；with 异常退出先置 DISABLE，不回零；close 后需新建对象再连接）✅
```

#### Robotics 独立 API（各域 ABC）

```python
FkineSolver / IkineSolver / JacobianSolver / DynamicsSolver / TrajPlanner / Controller   # ABC 契约 ✅
TrajPlanner：纯计算——plan(arm, target)（目标单点/序列→轨迹存对象内、拷贝后原子发布）/ sample(t)（按绝对时间采样产帧）；内置双实现 SingleJointHoldTrajPlanner（单点直传恒定输出）/ SingleJointCubicTrajPlanner（单点三次多项式，详见 §3.1）✅
Controller：step(arm, target, jointstate) → bool（当前帧+JointState，算 cmd 并经 arm 下发）；门控在 JoyArm ctrl 线程（暂停开关/无帧跳拍）✅
# 求解器用法二选一：实例.method(arm, ...)（arm 鸭子类型，课堂/单测）或 arm.* 门面（激活成员，应用）
# 鸭子契约（内置实现）：pin 系求解器 arm.pin_model；MDH/解析算法 arm.get_config()（教学数据）；规划器 arm.home_arm（三次法另读 arm.get_state_arm() 取起点）；控制器 arm.send_position_arm；电机模式由控制器子类类属性 MODE 声明（基类缺省 MIT、JointPositionController=POSITION），起搏为 config 规格键 plan_hz/ctrl_hz（必配、JoyArm 剥离消费）
```

#### Backend / 限位 / 数学 / 类型 API

```python
Backend(ABC, cfg) ✅   # 公开面：connect_*/disconnect_*（连接成功自动 best-effort 设 cfg default_mode，缺省 MIT）、enable/disable（逐 joint 内核编排、返 1/0）、get_mode（一致返模式并更新缓存，不一致/失败 None）、get_state（JointState 纯物理量含 t）、read/write_param（写入无静置；读回验证由子类内核决定——DM 参数 0x55 回显核对）、set_zero、set_mode（默认 MIT；成功后乐观更新模式缓存、门禁立即放行，一致性经 get_mode 读回纠正）、get/check/clear_error（读状态快照 error 码 list → 判 0/1 正常 → 发清错指令；读回验证由子类内核决定——DM 清错后核 err<2）、send_action_end、close（显式停刷新线程）；@property：cfg（MappingProxyType 只读视图）/ name / n_joints_*/ is_connected_*/ is_enabled_*/ joint_limits_*（只读，数组 writeable=False 零拷贝返本体）/ joint_state_*（只读快照槽位本体，被动直读不触发读帧）——无 mode_* 直读属性；限位裁剪唯一执行点（语义见 §1.3）
send_mit_{arm,end}(q, dq=None, tau=None, kp=None, kd=None) · send_position_{arm,end}(q, vlim=None, flim=None) · send_vel_{arm,end}(dq)   # 三模式整组下发（公开薄委托+缺省直接解析：q 必填 (n,)；dq/tau 缺省 zeros(n) 不广播；kp/kd 缺省取 _kp/_kd_mit_default_*、vlim/flim 缺省取 _vlim/_flim_default_* 成员；校验全在 *_impl：①维度→②空值/越限裁剪→③连接/模式门禁→④逐 joint 内核）
get_backend(name) · REGISTRY     # config backend.name 选型（REGISTRY 现有 backend_dm；新型号接入 = 复制 backend_template.py 七步接入）✅
clamp_to_limits(targets, limits) · limits_from_joint_cfgs(joint_cfgs) · soft_limits_from_cfg(cfg, n) · tcp_limits_from_cfg(tl) · rand_within_limits(limits)   # ✅
transforms.py（23 函数，清单见 §3.1）✅ · types.py 枚举+数据类（见 §3.1）✅
```

## 4. 文档同步维护

本文件与 [`README.md`](README.md) 是子库两大入口文档，**受众不同、须随代码同步维护**：README 给**使用者**，本文件给 **agent / 开发者**；新内容按受众分流，不互相重复。

- 两份 md 只写**精简概述**；详细规则写进代码 docstring（各包 `__init__.py` 一句话职责 + `:param:`/`:return:`/`:raises:`/章节标记）。
- **每次代码或注释发生变动，均须同步更新一遍本文件与 [`README.md`](README.md)**：公开 API/行为语义变 → §3；目录/文件变 → §1.1；安装/用法/章节状态变 → README；架构/约定变 → §0-2。**架构约束（§0）变更须人工确认后先改本节再动代码。**
- 提交前自检：新增公开符号已收录、已补 docstring；删除符号已移除；目录树一致。

### 篇幅上限：≤ 300 行，同内容一行

> 本文件硬性上限 **300 行**。超出时先精简再提交：优先压缩 §3 速查（只留索引级签名）；其次合并 §1-2 重复表述。禁止以"另存新文件"绕过上限；新增内容与精简同步进行。**行宽不设上限：同内容保持一行表达、不折行拆行，行超长时精简文字而非换行。**
