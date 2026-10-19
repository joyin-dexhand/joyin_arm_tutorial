# joyarm 重构规格

> 📌 **概述**：本文档是 `joyarm_core/joyarm/` 的重构执行规格——「工厂 `joyArm` + `JoyArm` 主类」结构，覆盖工厂校验、初始化自检、硬件委托、求解器与运动线程、运动状态机、安全监视、可视化七大功能块。含校验清单、方法映射表、状态行为表与配置及类型变更清单。

## 1 总则

- 重构属复杂任务：**用子 agent 与工作流并行执行**，动手前先充分探索 `joyarm_core` 现有代码。
- 现 `joyarm/joyarm.py` 仅作参考（原实现已过时），**不受其现状制约**。
- 明确废弃的现实现路径：
  - 三线程运动管线与 `sample_hz` → 改 plan + ctrl 两线程（§5.6）；
  - 工厂失败 `return None` → 改抛异常（§2）；
  - 0 基关节索引（`list.index` 实现）→ 改 1 基序号（§3.5）；
  - 旧配置键 `end_frame` / `arm_home` / `end_home` / `arm_soft_limits` / `end_soft_limits` → 一律以 yaml 现键为准（`frame_end` / `home_arm` / `home_end` / `joint_limits_soft_arm` / `joint_limits_soft_end`）。
- 遇疑问、歧义或有更优方案：**询问用户决策，不擅自定**。

## 2 工厂 joyArm 与 listArm

`joyArm(name: str)` 负责构造前的外部资源校验，四步固定顺序；任一步失败**抛携带具体原因的异常**（不返回 None），错误消息附可用型号列表（= `listArm()` 结果）。四步全过则 `logger.info`『创建成功』（含型号名）。调用方式保持 `from joyarm_core import joyArm; joyArm("joyarm_dm")`。

| 步骤 | 校验内容 | 失败消息要素 |
|:---:|:---:|:---:|
| ① config | `config/<name>.yaml` 存在且解析为非空 dict（空文件、纯注释、yaml 语法错误均判失败） | 文件路径 + 原始解析错误  + `listArm()` 结果|
| ② backend | cfg 的 `backend.name` ∈ `backend/__init__.py` 的 `REGISTRY` 键 | 非法值 + `REGISTRY` 全部可用键 |
| ③ robot_model | 四路径按序存在：`robot_model/<name>/`、`meshes/`、`urdf/`、`urdf/<name>.urdf`（型号名 = 目录名 = URDF 文件名） | 逐项缺失清单 + `listArm()` 结果 |
| ④ 构造 | 调用 `JoyArm(name)` 做类内初始化自检（§3） | 异常向上抛、保留原因链 + `listArm()` 结果 |

`listArm()`：模块级函数，返回可用型号 `list[str]` 并打印，与工厂失败消息共用同一实现。可行型号须**同时**满足：

1. `config/<name>.yaml` 存在、非空 dict、含 `joyarm` / `robotics` / `backend` 三顶层键；
2. `backend.name` ∈ `backend/__init__.py` 的 `REGISTRY` 键；
3. 上表③的四路径齐全。

`joyarm_template.yaml` 三键齐全但无 robot_model 目录，由条件 3 自然排除（**不做文件名特判**）。

## 3 初始化与自检

`JoyArm.__init__(name: str)`：类内成员一律 `_` 前缀私有，需类外访问者配只读 `@property`（防篡改：默认返回拷贝；数组/位形类成员只读化规则见 §3.1 `_joint_limits_hard_*` 行）；类外可写/可触发入口一律 `set_*`（snake_case），`@property` 不设 setter。类外访问接口（set/get/@property）不限于本规格所列清单，但增补接口前须先询问用户确认。任一自检失败抛具体异常、对象不构造。

**两阶段初始化**：先按 §3.1 清单把全部私有成员声明并赋安全默认值（None / 空容器 / 缺省数），再按依赖顺序赋真值：cfg 加载自检 → `_pin_model` → 关节数/名 → 限位 → 特征位形（zero/neutral/home）→ `_tcp_limits` → `_backend` → 六域求解器与激活项 → `_plan/_ctrl_hz`。

类注释两级：一级三行 banner（等号对齐）、二级一行 banner（等号对齐），规则同 backend 基类。

### 3.1 私有成员清单

| 私有成员 | 来源与构建 | 类外访问 | 关键校验与备注 |
|:---:|:---:|:---:|:---:|
| `_cfg` | `config/<name>.yaml` 全量 dict，经 §3.2 全套自检 | `get_config()`，返回 deepcopy | — |
| `_pin_model` | URDF 构建锁定缩减（§3.3） | `pin_model` @property，只读共享 | `frame_end` 帧存在校验 |
| `_backend` | `get_backend(cfg.backend.name)(backend 段 dict 去 name 键)` | 不直接暴露，经 §4 委托面访问 | 未知名自带可用列表报错；构造异常 → `JoyArm` 构造失败 |
| `_n_arm` / `_n_end` | `len(cfg.backend.arm/end.joints)` | `n_arm` / `n_end` @property | 与 URDF 同名同序活动（非 fixed 非 mimic）关节数一致，arm、end 均校验 |
| `_joint_names_arm/end` | `cfg.backend.arm/end.joints` 各 `name`（内部存 tuple） | `get_joint_names_*`（§3.5） | cfg 顺序 = URDF 顺序 = q 向量顺序 |
| `_joint_limits_hard_arm/end` | `utils.limits.limits_from_joint_cfgs(backend.arm/end.joints)` → `JointLimits` | @property 只读引用（init 构造后内部数组 writeable=False，零拷贝、外部原地改抛异常） | `q_min>q_max` 由自检⑤硬校验拦截（utils 内仅 warn，勿依赖） |
| `_joint_limits_soft_arm/end` | `cfg.joyarm.joint_limits_soft_arm/end` 经 `soft_limits_from_cfg` | @property 只读引用（同上行） | 标量广播全关节；未知键 / 长度≠n / 负幅值 / `q_min>q_max` 抛错；逐键位于硬限位内（±∞ 维跳过） |
| `_tcp_limits` | `cfg.joyarm.tcp_limits` → `TcpLimits` | @property 只读引用（同上行） | 格式校验（先校验后解析，`tcp_limits_from_cfg` 对缺键回退 inf）：`workspace_box` 为 3 行 2 列（yaml改格式）且首列逐元素 < 第二轮列、`v_lin_max/v_ang_max/f_max/t_max` 非空且 > 0，违规抛错；解析后 `workspace_box` 每行 `[min,max]` |
| `_zero_arm/end` | `np.zeros(n)` | @property 只读引用（同上行） | — |
| `_neutral_arm/end` | 软限位逐关节中点 `(q_min+q_max)/2` | @property 只读引用（同上行） | 软限位不存在则抛错并初始化失败 |
| `_home_arm/end` | `cfg.joyarm.home_arm/home_end` | @property 只读引用（同上行） | 逐关节校验在软限位 `[q_min,q_max]` 内，越界抛异常（指明关节、home 值、软限位区间） |
| `_plan_hz` / `_ctrl_hz` | 初始化取激活规划器 `plan_hz` / 激活控制器 `ctrl_hz` | @property | 两键由 JoyArm 消费、**不入求解器构造器**（cfg 自检④必配，无缺省回退）；切换求解器时更新（§5.4）；动态重读见 §5.6 |

六域求解器成员见 §5.1，`_target_trajframe` 见 §5.5，FSM 与安全事件成员见 §6.3 / §7.2。

### 3.2 cfg 内容自检清单

单成员自身校验以 §3.1 各成员行为权威，本清单仅列跨成员/跨源对比项；utils 等仅实现层，判据以本规格为准。cfg 加载后逐项自检，任一项失败抛异常，消息指明哪一条、期望 vs 实际：

| # | 校验内容 | 失败行为 |
|:---:|:---:|:---:|
| ① | `joyarm` / `robotics` / `backend` 三顶层键存在 | 抛异常，指明缺失键 |
| ② | `home_arm/home_end` 与 `joint_limits_soft_arm/end` 各键维度 = `backend.arm/end.joints` 条数（end 软限位标量按全关节广播） | 抛异常，指明键与维度 |
| ③ | robotics 六域子求解器注册名逐一在对应子包 `REGISTRY`（cfg 段键见 §5.1） | 抛异常，列可用注册名 |
| ④ | traj 域每个规划器必配 `plan_hz`、control 域每个控制器必配 `ctrl_hz`；接受 int 或整值浮点（如 `10.0`）且 ≥1 | 缺省 / 非整值 / <1 抛异常 |
| ⑤ | cfg.backend ↔ URDF 对比（细则见下） | 任一不符抛异常 |
| ⑥ | 软限位逐键位于硬限位内（细则见 §3.1 `_joint_limits_soft_*` 行） | 抛异常，指明关节与区间 |

**⑤ cfg.backend ↔ URDF 对比细则**：

- arm 活动关节须为 `joint1~jointN`、end 为 `joint_end1~joint_endM`，且与 URDF **同名同序**；
- cfg 只登记非 fixed、非 mimic 的驱动关节：URDF 中 fixed 的、mimic 从动的不要求也不允许出现在 `cfg.backend.joints`；
- 额外 link 留在 URDF，由 `_pin_model` 全量加载容纳；
- 各 cfg 关节 `q_min/q_max/dq_max/tau_max` 与 URDF `<limit>` 逐键差 ≤ 1e-4；
- `q_min ≤ q_max`；cfg 关节名无重复；URDF joint/link 名无重复。

### 3.3 _pin_model 构建

cfg 自检通过后：

1. `pin.buildModelFromUrdf(robot_model/<name>/urdf/<name>.urdf)` 得全量模型；
2. `pin.buildReducedModel` 锁定「URDF 活动关节中**不在** `cfg.backend.arm.joints` 名单内者」（= end 关节 + mimic 手指等，锁定位形取零位），得 `nq = n_arm` 的缩减模型——锁定判据用 cfg 名单而非硬编码关节名集合，型号无关；
3. 所有 link / 帧保留，保证整臂质量惯量正确；
4. 构建后校验 `cfg.joyarm.frame_end` 在 `pin_model` 帧表中存在：`getFrameId` 越界抛错并列可用帧名。

### 3.4 初始化完成 info

自检全部通过、成员装配完成后，`logger.info` 首行：

```text
<机械臂型号> 初始化和自检完毕（仅必要校验，不确保完全无误）！
```

随后附配置摘要（多行），最低字段集：型号名、backend 注册名、`frame_end`、`n_arm/n_end` 及关节名列表、各关节硬/软限位范围、zero / neutral / home 位形、六域当前激活求解器注册名、`plan_hz/ctrl_hz`、`tcp_limits` 摘要、`fsm_hz`。

### 3.5 关节查询与随机采样

| 方法 | 签名 | 语义 |
|:---:|:---:|:---:|
| `get_joint_names_arm/end` | `(joint: int\|"all"="all") -> str\|tuple[str, ...]` | `"all"`（默认）返回内部关节名 tuple 本体（顺序口径见 §3.1）；传入 1 基序号（cfg 列表位置）返回该关节名；越界 `[1,n]` 或类型非法 warn 并返回 None |
| `rand_q_arm/end` | `(joint_limits: JointLimits\|None=None) -> np.ndarray` | `(n,)` 在 `[q_min,q_max]` 均匀采样（委托 `rand_within_limits`）；缺省 `None` 在方法体内回退**软限位**（签名默认值不能引用 self）；仅此单参数，不加 `size`/`rng` |

关节序号（= cfg 列表位置，1 基）为全文统一的关节编号口径。

## 4 硬件后端委托面

除注明外均为**直接委托、透传参数与返回值**（backend 签名按 `backend.py` 现状）；JoyArm 不重复实现 backend 已有的校验与门禁。动态数据（控制模式、关节状态快照）一律经 `get_*` 方法**读时刷新**；joyarm 和 backend 不保留直返私有缓存的 `mode_arm/end`、`joint_state_arm/end` @property（删除，§10⑧，防陈旧读）。

**FSM 外部指令互斥**：FSM 运行期，arm 侧指令类公开方法被**外部线程**调用时空操作 + warn——`enable/disable/set_mode/set_zero` 返回 0、`clear_error` 返回 False、`send_mit/send_position/send_vel` 的 `_arm` 变体空操作；JoyArm 自家 ctrl 线程自身豁免（照常委托下发）；end 侧方法、参数读写、状态读取等只读/非 arm 指令接口不受互斥。

### 4.1 委托面方法总表

| JoyArm 方法 | backend 方法 | 参数 → 返回 | 细则 |
|:---:|:---:|:---:|:---:|
| `connect_arm/end` | `connect_arm/end` | `(channel: str\|None=None, protocol: str\|None=None) -> bool\|None` | §4.2 |
| `disconnect_arm/end` | `disconnect_arm/end` | `() -> None` | §4.2 |
| `enable_arm/end` | `enable_arm/end` | `() -> int` | §4.2；arm 侧 FSM 运行中被外部线程调用空操作+warn 返 0 |
| `disable_arm/end` | `disable_arm/end` | `() -> int` | §4.2；arm 侧 FSM 运行中被外部线程调用空操作+warn 返 0 |
| `is_connected_arm/end` | `is_connected_arm/end` | `@property -> Optional[bool]` | §4.2 |
| `is_enabled_arm/end` | `is_enabled_arm/end` | `@property -> Optional[bool]` | §4.2 |
| `read_param_arm/end` | `read_param_arm/end` | `(key: str) -> list\|None` | §4.3 |
| `write_param_arm/end` | `write_param_arm/end` | `(key: str, values: Sequence[float]) -> int` | §4.3 |
| `get_mode_arm/end` | `get_mode_arm/end` | `() -> ControlMode\|None` | §4.4 |
| `set_mode_arm/end` | `set_mode_arm/end` | `(mode: ControlMode=ControlMode.MIT) -> int` | §4.4；arm 侧 FSM 运行中被外部线程调用空操作+warn 返 0 |
| `get_state_arm/end` | `get_state_arm/end` | `() -> JointState\|None` | §4.5 |
| `set_zero_arm/end` | `set_zero_arm/end` | `() -> int` | §4.6；arm 侧 FSM 运行中被外部线程调用空操作+warn 返 0 |
| `get_error_arm/end` | `get_error_arm/end` | `() -> list\|None` | §4.6 |
| `check_error_arm/end` | `check_error_arm/end` | `() -> bool` | §4.6 |
| `clear_error_arm/end` | `clear_error_arm/end` | `() -> bool` | §4.6；arm 侧 FSM 运行中被外部线程调用空操作+warn 返 False |
| `send_mit_arm/end` | `send_mit_arm/end` | `(q, dq=None, tau=None, kp=None, kd=None) -> None`，缺省细则见 §4.7 | §4.7；arm 侧 FSM 运行中被外部线程调用空操作+warn |
| `send_position_arm/end` | `send_position_arm/end` | `(q, vlim=None, flim=None) -> None` | §4.7；arm 侧 FSM 运行中被外部线程调用空操作+warn |
| `send_vel_arm/end` | `send_vel_arm/end` | `(dq) -> None` | §4.7；arm 侧 FSM 运行中被外部线程调用空操作+warn |
| `set_action_end` | `send_action_end` | `(action: str, **kw) -> None` | §4.8 |

### 4.2 连接与使能

- `connect_arm/end(channel, protocol)`：透传可选参（缺省用 backend cfg 成员值，显式传入则同步更新）后委托，返回 `is_connected_arm/end`（True 成功 / False 失败 / None 意外）。
- `disconnect_arm/end()`：委托，无返回值（断连结果经 `is_connected_*` 查询）。
- backend 副作用（方法 docstring 写明）：connect 成功后自动设 cfg `default_mode` 并等 `write_settle` 秒生效；disconnect 前自动尽力失能再断总线、断开后清模式缓存；连接失败仅 warn 不抛。
- `enable/disable_arm/end() -> int`：1 = 全部成功、0 = 任一关节失败或未连接（失败细节经日志 warn，不抛；使能结果另经 `is_enabled_*` 核实）；FSM 运行中被外部线程调用：arm 侧空操作 + warn 返 0（end 侧不受互斥）。
- `is_connected_arm/end`、`is_enabled_arm/end` 四 `@property`：与 backend **同名直连**（改名见 §10⑧），返回 `Optional[bool]` 三态（使能缓存按各关节 error 码推导（码义见 §4.5）：全 1→True、全 0→False、混合或含 ≥2→None，可自愈；推导自快照，鲜度可用 `JointState.t` 判）。

### 4.3 参数读写

- `read_param_arm/end(key)`：逐关节参数值列表（int/float 混合，uint32 寄存器为 int）；任一失败 / 未连接返回 None。
- `write_param_arm/end(key, values)`：`values` 长度须 = 关节数且全为有限值；成功 1、失败 0；前置已连接。
- DM 参数键：`pos_kp/pos_ki/vel_kp/vel_ki/pmax/vmax/tmax/hw_ver/sw_ver/sn`，其中 `hw_ver/sw_ver/sn` 只读；写入均为 RAM 态、断电丢失。

### 4.4 控制模式

- `get_mode_arm/end()`：全组模式一致才返回该模式；不一致 / 失败 / 未连接返回 None。
- `set_mode_arm/end(mode)`：入参必须为 `ControlMode` 枚举（MIT / POSITION / VELOCITY），非枚举直接 0；成功返回 1 并**乐观更新**模式缓存使发送门禁立即放行，硬件一致性经 `get_mode` 读回纠正；FSM 运行中被外部线程调用：arm 侧空操作 + warn 返 0。
- 切 POSITION / VELOCITY 自动随写 POS_VEL 四增益（缺配置跳过、沿用电机内部值）。

### 4.5 状态读取

- `get_state_arm/end()`：返回本次装配并原子换入的快照（`t` = 本批各电机数据帧到达时刻的最早值 Unix 秒；`q` rad、`dq` rad/s、`tau` N·m、`temp_mos/temp_rotor` ℃；硬件不提供的量整组 None；任一关节读取失败 / 数据异常跳过本轮更新返回 None 且保持上次快照）。
- **error 状态码语义**（全文约定）：0 = 失能、1 = 使能（均正常）、≥2 = 故障码——使能标志推导（§4.2）、FSM 转移检查（§6.1，经 `is_enabled_arm`）与 safe_monitor 硬规则（§7.3）直接依赖此语义。
- backend 低频刷新线程（cfg `backend.state_refresh_hz`）持续维护快照，JoyArm 层无需自建轮询。

### 4.6 零点与错误

- `set_zero_arm/end() -> int`：将当前位置记为零点（影响后续 q 读数基准）；1 = 全部成功、0 = 任一失败（warn 不抛）；FSM 运行中被外部线程调用：arm 侧空操作 + warn 返 0。
- `get_error_arm/end()`：读缓存快照 error 段；快照过期、未连接或 error 段缺失返回 None。
- `check_error_arm/end() -> bool`：全 0/1 判正常 True；含 ≥2（码义见 §4.5）/ 未连接 / 快照过期 False。
- `clear_error_arm/end() -> bool`：逐关节发清错指令、无读回验证；FSM 运行中被外部线程调用：arm 侧空操作 + warn 返 False（注意返 False 非 0）。

### 4.7 关节指令下发

- `send_mit_arm/end(q, dq=None, tau=None, kp=None, kd=None)`：电机内部执行 `τ = tau + kp·(q_d−q) + kd·(dq_d−dq)`，发送时五参均为 `(n,)`——**`q` 必填**；`dq`/`tau` 缺省置 `zeros(n)`；`kp`/`kd` 缺省取成员 `_kp/_kd_mit_default_arm/end`。下发校验（维度校验、`tau/q/dq` 越硬限位就近裁剪 + warn、连接 + MIT 模式门禁）**全部在 `_send_mit_impl`**，公开接口零校验、JoyArm 直接委托 backend。
- `send_position_arm/end(q, vlim=None, flim=None)`：`q` 必填 `(n,)` rad；`vlim` rad/s、`flim` 归一化力矩电流 0~1，缺省取成员 `_vlim/_flim_default_arm/end`；`q` 越硬限位 / `vlim` 超 `[0,dq_max]` / `flim` 超 `[0,1]` 就近裁剪 + warn、连接 + POSITION 门禁——校验全部在 `_send_position_impl`。
- `send_vel_arm/end(dq)`：`dq` 必填 `(n,)` rad/s、无缺省；幅值越 `|dq_max|` 就近裁剪 + warn、连接 + VELOCITY 门禁——校验全部在 `_send_vel_impl`。

### 4.8 send_action_end 动作集

JoyArm 公共入口 `set_action_end(action: str, **kw)` 纯透传 `backend.send_action_end`（连接校验与分发由 backend 完成）。
未知动作或缺 `position` 参数：backend 抛 `ValueError` 经基类转 warn——JoyArm 层可观察行为为 warn 不抛。

## 5 求解器与运动线程

### 5.1 六域成员

- 域缩写：`fkine / ikine / jac / dyn / traj / ctrl`；cfg 段键维持 yaml 现状（`fkine/ikine/jacobian/dynamics/trajectory/control`）。
- 每域两成员：`_<域>_solvers`（`{注册名: 求解器对象}` 字典）+ `_activated_<域>_solver`（直接指向当前激活求解器**对象**，非名字符串）。
- 默认激活 = `cfg.robotics.<域>` 段**首个**子求解器（dict 保序）。
- 域未配置任何成员时对应成员为 `None`，且该域方法抛 `RuntimeError`。

### 5.2 构建与校验

- 前置：cfg 自检③④通过（§3.2）。
- 注册名不存在 / 规格缺 `name` / 构造参数错 / 实例化异常 → 抛具体异常，列可用注册名。
- **构造参数透传**（如 `PinIkineSolver(damping, step_max, restarts)`）；`plan_hz/ctrl_hz` 为管线键（归属见 §3.1）。各实例自持构造参数，切换激活指针即携带，无共享参数通道。

### 5.3 按域方法（×6 域）

| 方法 | 签名 | 语义 |
|:---:|:---:|:---:|
| `get_<域>_solvers()` | `() -> list[str]` | 返回该域 `[注册名]` 列表**深拷贝** |
| `set_activated_<域>_solver(name)` | `(name: str) -> bool` | 名称不在 `_<域>_solvers` 中则 warn 列出可用名、**保持原激活不变**；成功则按 §5.4 协议替换 |
| `get_activated_<域>_solver()` | `() -> str` | 返回当前激活求解器注册名（= `_activated_<域>_solver` 所指对象的注册名，经 `_<域>_solvers` 反查） |

### 5.4 热切换协议

- **traj 域固定时序**：① 用新规划器 `plan(self, _target_trajframe)` 一次（成功才继续，失败 warn 保持原激活）→ ② 替换 `_activated_traj_solver` → ③ 更新 `_plan_hz` 与 plan 线程频率。
- **先 plan 后替换的原因**：规划结果缓存在规划器实例内部，先换指针后规划可能会使控制线程在 `sample` 时出错（未 plan 即 sample），旧缓存亦不可复用。
- **ctrl 域**：替换 `_activated_ctrl_solver` → 按控制器 `MODE` 类属性联动电机模式 → 更新 `_ctrl_hz` 与 ctrl 线程频率。
- **其余四域**（fkine/ikine/jac/dyn）直接换指针。

### 5.5 _target_trajframe

- `TrajFrame` 单帧或帧序列（字段：`time/pose/twist/accel/wrench/q/dq/ddq/tau`），规划线程 plan 时传入。
- `set_target_trajframe(单帧|序列)`（None / 空 = 清空目标）：`copy.deepcopy` 后单条赋值语句整体原子替换引用（规划线程每拍整体重读，无锁原子）。
- `get_target_trajframe` 返回当前只读引用。
- **上层便捷封装**：免手写完整 `TrajFrame`，内部组帧后走 `set_target_trajframe` 同一原子通道；均为单帧目标，多帧序列仍直接走 `set_target_trajframe`；参数值可为list或np.ndarray类型。

| 方法 | 签名 | 校验 |
|:---:|:---:|:---:|
| `set_target_joint` | `(t, q, dq=None, ddq=None, tau=None)` | `t`（Unix 秒标量）与 `q`（`(n,)` rad）必填；`dq` rad/s、`ddq` rad/s²、`tau` N·m 可选（缺省 None）；给出的全部数组维度须为 `(n,)`，不符抛异常 |
| `set_target_cart` | `(t, pose, twist=None, accel=None, wrench=None)` | `t` 与 `pose`（`Pose` 类型或`(7,)` list/ndarray）必填；`twist`/`accel`/`wrench` 可选（缺省 None），须为 `Twist`/`Accel`/`Wrench` 类型或`(6,)` list/ndarray |

### 5.6 两线程架构

- **`_start_motion_thread`**：激活 traj 求解器 `plan(self, _target_trajframe)` 一次（保证首帧可用）→ 启动 plan 线程（按 `_plan_hz` 周期 `plan(self, _target_trajframe)`）→ 启动 ctrl 线程（按 `_ctrl_hz` 每拍 `_activated_ctrl_solver.step(self, frame = _activated_traj_solver.sample(time.time()), _backend.get_state_arm() 的 JointState)`，step 经 JoyArm 公开 send_*_arm 下发；FSM 运行中依赖外部指令互斥拦截 + ctrl 线程自身豁免才不被自身拦死；`get_state_arm()` 为 None 时跳过本拍 step + 节流 warn）→ 清暂停开关。
- **旧线程残留防护**：`_start_motion_thread` 启动前检查 plan/ctrl 线程句柄，上次 stop 超时未确认退出（句柄保留）则抛错拒绝另起新线程（防双循环；MOTION 进入失败由 FSM 下拍重试）。
- **hz 动态更新**：两线程节拍每拍重读 `_plan_hz/_ctrl_hz`（或等价 set_hz 机制），切换求解器后下一拍生效。
- **`_stop_motion_thread`**：先停 ctrl 线程再停 plan 线程，超时回收线程；目标帧缓存 `_target_trajframe` 仅在 FSM start/stop 边界清理（§6.5），MOTION 态退出不清（保护性往返后续走用户目标）。
- 方法名单数 `_thread` 但统一管理两条线程；不设 sample 线程与 `sample_hz`。

### 5.7 数学门面 9 方法

全部落到 JoyArm（后四个为现状缺失、需新增）：

| 方法 | 签名 | 返回 |
|:---:|:---:|:---:|
| `solve_fkine` | `(q=None, frame=None)` | `Pose` |
| `solve_ikine` | `(pose, frame=None, q0=None, **kw)`（求解器特有参数仅限关键字传入） | `IKResult` |
| `solve_ikine_all` | `(pose, frame=None, **kw)` | `IKResult` |
| `get_jac` | `(q=None, frame=None)` | `(6,n)` |
| `get_jac_pinv_damped` | `(q=None, frame=None, damping=1e-3)` | `(n,6)` |
| `solve_idyn` | `(q=None, dq=None, ddq=None)` | `(n,)` |
| `get_mass_matrix` | `(q=None)` | `(n,n)` |
| `get_coriolis` | `(q=None, dq=None)` | `(n,)` |
| `get_gravity` | `(q=None)` | `(n,)` |

- 缺省值统一：`q/dq` 缺省取 `_backend.get_state_arm()` 的 q/dq（返回 None 时抛 `RuntimeError`；显式传全参则离线可用）；`frame` 缺省取 `cfg.joyarm.frame_end`；`q0` 缺省当前状态 q；`ddq` 缺省零向量。
- `solve_ikine_all`：激活求解器未实现 `solve_all` 时 `NotImplementedError` 转 `RuntimeError` 并提示换解析法。

## 6 运动状态机 MotionFSM

### 6.1 总纲

- **`MotionState` 枚举**定义到 `utils/types.py`（成员大写，对齐 `ControlMode` 风格）：`DISABLE` / `ZERO` / `HOME` / `MOTION` / `HOLD` / `DAMPING`。
- 状态机跑**独立 daemon 线程、固定频率**：`cfg.joyarm.MotionFSM.fsm_hz`（默认 50 Hz） 可配，与 `plan/ctrl_hz` 解耦。
- 前置：初始化自检通过且 arm 已连接。
- FSM 状态动作与切换中全部硬件操作（enable/disable、get/set_mode、send_position/send_mit、get_state、clear_error）直连 `self._backend.*`，不经 JoyArm 公开委托面，减少转发开销；线程启停与缓存管理仍走 JoyArm 内部方法；安全性不降（理由同 §4 总则）。
- **连接/使能/模式检查与设置仅在状态转移（enter）时执行一次**（经 `self._backend` 读取 `is_connected_arm` 标志、`is_enabled_arm` 标志、`get_mode_arm()`），状态内维持循环不再每拍检查、只执行必要动作（降低开销）；转移时发现未连接 → 抛错。
- 各状态参数配置在 `cfg.joyarm.MotionFSM.*`。

### 6.2 六状态行为规格

总表（编号 1=disable、2=zero、3=home、4=motion、5=hold、6=damping，与 §6.4 路线表一致）。各状态 enter 先执行 §6.1 的转移时一次检查与设置，enter 列仅列状态特有动作：

| # | 状态 | 语义 | 电机模式 | 进入时（enter） | 每拍核心动作 | 退出清理 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `DISABLE` | 失能 | — | 执行一次 `_disable_clear_disable_arm`（§9，「失能→清错→失能」序列）并 info 状态转移信息 | 直接轮发 `self._backend.disable_arm()`（幂等、无检查无 warn） | — |
| 2 | `ZERO` | 零位失能 | POSITION | 使能+设模式+发送 `_zero_arm` 位置指令+ 置「移动中」标志 | 子阶段机：移动中 → 逐拍 `send_position_arm(_zero_arm)` + 查 `is_in_q_arm(_zero_arm)`，到位 → `disable_arm()` 失能并转「零位失能维持」子阶段（此后每拍轮发 disable_arm，同 DISABLE）；任何切换可立即打断（无阻塞等待） | 清到位轮询/子阶段标志 |
| 3 | `HOME` | 空闲保持 | POSITION | 使能+设置模式 | 逐拍 `send_position_arm(_home_arm)` + `is_in_q_arm(_home_arm)` 到位监测（迁移门槛见 §6.3） | — |
| 4 | `MOTION` | 正常运动 | 控制器 MODE | 清理缓存和残留+使能+设置mode为控制器 MODE；然后启动 plan/ctrl 两线程（§5.6 `_start_motion_thread`） | 维持 plan+ctrl 两线程 ；运行中热切换求解器按 §5.4 | 停线程（§5.6 `_stop_motion_thread`；目标帧缓存仅在 FSM start/stop 边界清理，见 §6.5） |
| 5 | `HOLD` | 原位锁定 | POSITION | 使能+ 模式设置+`_hold_q` 取 `_backend.get_state_arm().q`（采样失败则迁移推迟下拍重试） | 逐拍 `send_position_arm(_hold_q)` | 清 `_hold_q` |
| 6 | `DAMPING` | 阻尼防护 | MIT | 使能+模式设置 | 逐拍 `send_mit_arm(q=0, dq=0, tau=0, kp=0, kd=_mit_kd_damping)`（`q/dq/tau/kp` 均为 `zeros(n)` 数组，不广播） | — |

**damping**：`_mit_kd_damping` 在 init 解析：`cfg.joyarm.MotionFSM.mit_kd_damping` 为正有限数（标量广播或 n 元列表）则采用；未配置 / 非法 → init 抛错初始化失败。

### 6.3 状态成员与切换时序

- `_current_motion_state: MotionState`（FSM 线程独占写，初始 `DISABLE`）+ 公共 `get_current_motion_state()`；`_target_motion_state: MotionState` + `get_target_motion_state()` + 私有 `_set_target_motion_state(state)`（原子赋值）。
- FSM 每拍：
  1. 消费 `_safe_event` 变化更新 `_target_motion_state`（通路与映射见 §7.1/§7.2）；
  2. `target≠current` 时按 §6.4 路线表取下一跳 `next` → 执行现状态 **exit 清理**（motion/hold 清单见 §6.2 退出清理列——motion 经 §5.6 `_stop_motion_thread`；zero：清到位轮询/子阶段标志）→ 执行 `next` 的 **enter**（= §6.1 转移前置检查一次完成 + 状态初始化：hold 采样 `_hold_q`、motion 起线程等）→ `current=next`；`next==target` 转维持，否则下一拍继续迁移；
  3. `target==current` 执行该状态维持动作。
- 迁移到位门槛：current==HOME 且 target∈{ZERO, MOTION} 时须 home 到位判定 `is_in_q_arm(_home_arm)`（tol 从 `cfg.joyarm.arrival_tol` 取（joyarm的私有成员变量），arm/end 分组逐关节）成立才执行该次迁移；其余迁移不等到位。home 未到位时的出路 = 保护性事件直达（disable/damping/hold 不受门槛限制）或 `stop_motion_fsm`（带超时强制推进，§6.5）。
- 外部指令互斥兜底（§4 总则）：状态运行中被外部失能/改模式不再每拍检测（§6.1）——软件层由 FSM 外部指令互斥阻断（arm 侧指令类公开方法对**外部线程**空操作 + warn；JoyArm 自家 ctrl 线程豁免、照常下发，FSM 内部硬件操作直连 `_backend` 亦不经公开面、不受影响）；硬件侧异常由 safe_monitor 硬规则（error≥2 → DAMPING、断连 → DISABLE，见 §7.3）监测，命中触发转移、在转移时重新执行 §6.1 的检查与设置。

### 6.4 状态转移路线表

（from 左列 = 起点，to 上行 = 终点）

| from\to | 1 disable | 2 zero | 3 home | 4 motion | 5 hold | 6 damping |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1 disable** | — | 1→3→2 | 1→3 | 1→3→4 | 1→5 | 1→6 |
| **2 zero** | 2→1 | — | 2→3 | 2→3→4 | 2→5 | 2→6 |
| **3 home** | 3→1 | 3→2 | — | 3→4 | 3→5 | 3→6 |
| **4 motion** | 4→1 | 4→3→2 | 4→3 | — | 4→5 | 4→6 |
| **5 hold** | 5→1 | 5→3→2 | 5→3 | 5→3→4 | — | 5→6 |
| **6 damping** | 6→1 | 6→3→2 | 6→3 | 6→3→4 | 6→5 | — |

结构规律：`home(3)` 为中枢，各态与 home 互达均直达；除「各态→disable / →hold / →damping」三组保护性直达外，其余迁移（→zero、非 home 起点→motion）一律经 home 中转。

### 6.5 start_motion_fsm / stop_motion_fsm

**start_motion_fsm**：

1. 前置：`is_connected_arm is True`（False 或 None 抛 `RuntimeError` 拒启）；
2. 若规划/控制线程尚在运行则先停止，然后幂等清理 FSM 全部缓存（`_hold_q`、plan/ctrl 线程句柄与轨迹帧缓存；`_safe_event` 置 NORMAL、`_target_motion_state` 置 MOTION）；
3. 启动 FSM 线程（固定 `_fsm_hz`）：初始 `current=DISABLE`、`event=NORMAL`、`target=MOTION`，按路线表自动走 1→3→4 迁移至 MOTION（与「先 disable 再 home 再 motion」序列一致；3→4 受 §6.3 home 到位门槛）；非阻塞。

**stop_motion_fsm**：

1. 前置：`is_connected_arm is True`，否则抛 `RuntimeError`（断连急停不走本方法——无法下发任何指令，走销毁路径 §6.6）；
2. 置 `set_event(SafetyEvent.ZEROING)`（统一事件与目标，同步 `_target_motion_state=ZERO`）：按路线表经 HOME 中转迁移，**阻塞等待**含 home 到位（3→2 迁移门槛见 §6.3）+ 等 ZERO 子阶段推进完成（到位 + 失能）；超时沿用 `cfg.joyarm.runtime.move_wait_timeout`（10 s），超时 warn 并强制推进（disable → 清缓存 → 退出，见步骤 3/4），避免死等；
3. `self._backend.disable_arm()` 失能 arm 全部电机（end 失能由调用方或销毁路径处理）；
4. 停 FSM 线程（停机 Event + join 带超时，超时 warn 不强杀）、清空全部 FSM 缓存、状态机退出（再用需重新 start）。

### 6.6 销毁安全退出

`__exit__`（with 退出）与显式 `close()` 第一步均为 `stop_motion_fsm()`（best-effort：异常捕获转 warn 不阻断收尾），其后安全收尾序列 `disable_arm` → `disable_end` → `disconnect`。`__del__` 不做执行时序承诺，docstring 注明请用 with 语句或显式 `close()`。

### 6.7 到位判断函数

三个独立函数（tol 参数从 `cfg.joyarm.arrival_tol` 取）：

| 函数 | 签名 | 判据 |
|:---:|:---:|:---:|
| `is_in_pose_arm` | `(pose, tol_pos, tol_rot, frame=None) -> bool` | 末端位姿由激活 fkine 对 `get_state_arm().q` 现算（frame 同 §5.7），位置误差 ≤ `tol_pos` 且姿态误差角 ≤ `tol_rot` 同时满足；`pose` 须为 `Pose` 类型；tol 从 `cfg.joyarm.arrival_tol` 取 |
| `is_in_q_arm` | `(q, tol_q=None) -> bool` | `np.max(|q_cur−q|) ≤ tol_q` 全关节同时满足；tol从 `cfg.joyarm.arrival_tol` 取（n 元列表逐关节，joyarm 私有成员变量） |
| `is_in_q_end` | `(q, tol_q=None) -> bool` | 同上（end 侧） |

`q_cur` 取 backend `get_state_arm/end().q`；`q` 必传（zero / home 态到位判断即 `is_in_q_arm(self._zero_arm/_home_arm)`）。

## 7 安全事件与监视器

### 7.1 SafetyEvent 枚举与状态映射

**`SafetyEvent` 枚举**定义到 `utils/types.py`（成员大写）：`NORMAL` / `HOLDING` / `HOMING` / `ZEROING` / `DAMPING` / `DISABLE`，危险程度递增（DISABLE 最高）。事件表达安全监督层的判定（要求进入的降级等级），状态机按映射表执行：

| SafetyEvent | 语义 | 目标 MotionState |
|:---:|:---:|:---:|
| `NORMAL` | 运行正常 | `MOTION` |
| `HOLDING` | 原位锁定 | `HOLD` |
| `HOMING` | 回退锁定 | `HOME` |
| `ZEROING` | 回零失能 | `ZERO` |
| `DAMPING` | 阻尼防护 | `DAMPING` |
| `DISABLE` | 电机失能 | `DISABLE` |

### 7.2 set/get_event

- 私有成员 `_safe_event: SafetyEvent`（初始 `DISABLE`）。
- 公共 `set_event(event: SafetyEvent)`：原子赋值 `_safe_event` 并同步 `_target_motion_state = 事件映射状态`；`get_event() -> SafetyEvent` 返回当前值。
- FSM 线程每拍优先消费事件变化，响应延迟 ≤ 1/fsm_hz（默认 ≤ 20 ms）。

### 7.3 safe_monitor 监视器

- 独立 daemon 线程；`start_monitor()` / `stop_monitor()` 启停；频率 `cfg.joyarm.MotionFSM.monitor_hz`（缺省 = `fsm_hz`）。
- 每拍：只读采集（`get_state_arm/end`、`get_current_motion_state`、`get_event`）→ 按规则判断 → 命中且危险程度高于当前事件才 `set_event`（**只升不降**，防止覆写并发设置的更高危险事件；外部 `set_event` 不受限）。**只读：绝不调用控制/下发方法**；事件经 §7.2 通路驱动 FSM，天然非阻塞。
- **触发规则集（最小硬规则）**：

| 监测条件 | set 事件 |
|:---:|:---:|
| `get_state_arm().error` 任一 ≥2（码义见 §4.5） | `DAMPING` |
| `is_connected_arm is False`，或 `get_state_arm()` 连续 0.1 s 返回 None（快照过期/读取失败） | `DISABLE` |

- **自定义规则**：签名 `rule(state_arm: JointState|None, state_end: JointState|None, motion: MotionState, event: SafetyEvent) -> SafetyEvent|None`（状态读取失败时仍以 None 调用，规则须自判；返回 None = 无意见、不触发）；绑定方法 `set_monitor_rules(rules)`，传单条或列表；仲裁：同拍多命中取危险程度最高者，内置硬规则优先于自定义。

## 8 运行状态可视化(只读和状态刷新)

- 一级功能区，入核心包；配套修订已完成（`joyarm_code/AGENTS.md` §1.2/§2.2 已纳入可视化与 meshcat 可选依赖）。
- **单页面监控网页**：左侧 iframe 嵌 meshcat 3D 视图（`src = meshcat.Visualizer().url()`），右侧自写 HTML 状态面板；轻量 HTTP 服务托管页面 + 面板 JS 约 5 Hz 轮询 `fetch /state.json`（HTTP 轮询、零新增依赖）。
- 右侧面板内容（arm/end分组）：关节的 q/dq/tau、双温度（`temp_mos/temp_rotor`）、error 码、末端位姿/速度、FSM 当前/目标状态、安全事件、连接/使能/模式。
- **单一纯只读 daemon 线程**（`_state_viz_worker`），仅调只读接口刷新 meshcat 模型姿态与面板数据，绝不调用控制/下发方法；启停 `start_state_viz()` / `stop_state_viz()`。
- **meshcat 为可选依赖**：`start_state_viz()` 内才 import；缺包（ImportError）warn 并降级为仅 HTML 状态面板（3D 区占位）。pyproject 增可选依赖组（如 `vis = ["meshcat>=0.3.2"]`）。
- 可选（用户可选开启：start 时或启动后手动点击界面）轨迹显示功能（需运动状态机已启动，状态为 motion 状态时才显示）：显示 `frame_end` 的当前位姿与目标位姿，并从当前激活的规划器采样轨迹点绘制。采样规则：单帧目标取区间 [当前时刻, 目标 t]、多帧目标取 [当前时刻, 末帧时刻]；取 200 点（start 时可配，默认 200）；区间长度 ≤ 0.1 s 则不采样不绘制。

## 9 私有助手

| 助手 | 职责 |
|:---:|:---:|
| `_fsm_worker` / `_fsm_tick` | FSM 线程体与单拍循环（固定 `_fsm_hz`，执行 §6.3 每拍流程） |
| `_enter_motion_state` / `_exit_motion_state` | 状态进入/退出动作分发与缓存清理 |
| `_disable_clear_disable_arm` | 「失能→清错→失能」序列（DISABLE 态 enter 用；直连规则见 §6.1） |
| `_monitor_worker` | safe_monitor 线程体（§7.3 规则集） |
| `_state_viz_worker` | 状态可视化只读监控线程体（§8） |

周期线程类助手沿用现有私有周期线程机制形态（`_run_periodic` / `_PeriodicThread`），留在 joyarm、不入 utils。

## 10 cfg 与类型变更清单

| # | 变更 | 内容 |
|:---:|:---:|:---:|
| ① | 新增段 `cfg.joyarm.MotionFSM` | `fsm_hz`（§6.1）/ `mit_kd_damping`（§6.2）/ `monitor_hz`（§7.3） |
| ② | 修订值 | `home_arm: [0.0, -1.0, -1.0, -0.5, 0.0, 0.0]`、`home_end: [-3.0]`（已逐关节核对在软限位内；校验规则见 §3.1 `_home_*` 行，保留） |
| ③ | pyproject | 增可选依赖组 `vis = ["meshcat>=0.3.2"]`（缺包降级行为见 §8） |
| ④ | 基准 | 其余键一律以 yaml 现键为准、不做键名迁移；旧键 `end_frame` / `arm_home` / `end_home` / `arm_soft_limits` / `end_soft_limits` 废弃 |
| ⑤ | 类型变更 `utils/types.py` | 新增 `Accel` 数据类（`linear` `(3,)` m/s²、`angular` `(3,)` rad/s²，与 `Twist` 成对同构）；`TrajFrame` 增加笛卡尔加速度子项 `accel: Optional[Accel]`（置于 `twist` 与 `wrench` 之间）——关节侧已有 `ddq`、力侧已有 `tau`/`wrench`，独缺笛卡尔加速度 |
| ⑥ | 契约变更 | backend 基类 `send_mit_arm/end` 同步 §4.7 签名（`q` 必填、`dq/tau` 缺省默认 `zeros(n)`、`kp/kd` 缺省取 `_kp/_kd_mit_default_*` 成员）；三指令下发校验全部在 `*_impl`、公开接口零校验、JoyArm 纯委托（backend.py；backend_dm.py / backend_template.py 无公开覆写） |
| ⑦ | 配套变更 | backend 基类 `joint_limits_arm/end` @property 内部数组置 writeable=False 返本体；`cfg` @property 返回 MappingProxyType 只读视图（顶层防增删，嵌套层注明勿深层改动） |
| ⑧ | 契约变更 | backend 删除 `mode_arm/end`、`joint_state_arm/end` 直读 @property（直返私有缓存、可能陈旧）——动态数据一律经 `get_mode_*` / `get_state_*` 读时刷新；`is_abled_arm/end` @property **改名** `is_enabled_arm/end`（与 JoyArm 委托面同名，形态保持 @property）；`is_connected_arm/end` @property 不变（连接标志，非快照数据） |
| ⑨ | 新增键 `cfg.joyarm.arrival_tol` | 到位判据统一容差：arm `{q: n 元列表, pos: float, rot: float}`、end `{q: n 元列表}`；缺省 0.05 / 1e-3 / 1e-2 标量广播（q 逐关节、pos/rot 单值）；消费方见 §6.7 三函数与 §6.3 到位门槛 |
