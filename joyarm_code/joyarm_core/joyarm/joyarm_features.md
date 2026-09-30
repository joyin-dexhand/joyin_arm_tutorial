# JoyArm（`joyarm.py`）功能清单

## 一、模块级内容（类外）

### 1.1 常量

| 名称 | 简介 |
|:--|:--|
| `_CONFIG_DIR` | `config/` 目录（包根下；按型号名找 `<model>.yaml`） |
| `_ROBOT_MODEL_DIR` | `robot_model/` 目录（URDF + meshes 资产，运行期加载） |
| `_ARM_JOINT_NAMES` | 本体关节名集合 `{joint1…joint9}`——URDF 里叫这些名字才算本体关节，其余（末端/手指）不参与关节数与顺序校验 |
| `_URDF_LIMIT_TOL` | config 四键限位与 URDF limit 核对容差（1e-3；超出只告警，数值以 config 为准） |
| `_ERR_LOG_INTERVAL` / `_JOIN_TIMEOUT` | 周期线程：同故障日志最小间隔 0.5 s / `stop()` 等待上限 1 s |
| `_DOMAIN_REGISTRIES` | 六域 → 各域 `REGISTRY` 映射（fkine / ikine / jacobian / dynamics / traj / control） |

### 1.2 函数

| 函数 | 简介 |
|:--|:--|
| `load_config(model, strict=False)` | **公开**（`__all__`）：读 `config/<model>.yaml`；`strict=True` 时缺失/解析错抛 `ValueError`（列可用型号），默认返 `None`（交工厂软化） |
| `_soft_beyond_hard(soft, hard)` | 判软限位是否越出硬限位（某键 ±∞ = 该维度不设软限，跳过比较；JoyArm/FakeArm 共用） |
| `_parse_domain_specs(domain, spec)` | config `robotics.<域>` 段 → `[(注册名, 参数字典), …]`（单项为注册名字符串或 `{name, …, **构造参数}`，非法抛错） |
| `_build_domain(domain, registry, spec)` | 按规格实例化一域成员 → `{注册名: 实例}`；域未配置返空；注册名无效/实例化失败硬失败 `ValueError` |
| `_as_hz(hz)` | 频率参数归一：数值，或"返回数值的无参函数" |
| `_run_periodic(stop, hz, step, name)` | 周期循环体（线程 target）：绝对时间对齐节拍（单步迟了只跳拍不追赶）、`step()` 异常仅记日志下周期重试（同故障日志限频）、每周期重取 hz（改频即时生效） |
| `_cubic_traj(q0, q1, t, rate)` | 三次多项式插值 → `(ts, q, dq)`（起末速度零；`t ≤ 0` 退化为单帧）；仅 `move_j` / `_safe_move` 直连路径使用 |

## 二、JoyArm 类

### 2.1 构造 `__init__(model, config=None)`

config 缺省自动加载 `config/<model>.yaml`；缺失或 `check_config` 不过即抛（硬失败）。流程：

1. 深拷贝 config → `check_config` 静态自检；
2. 解析 URDF → pin 模型（`_build_pin_model`，锁非本体关节）→ 末端帧存在性校验；
3. 关节数/顺序校验（config `backend.arm.joints` vs URDF，不一致抛错——顺序错位会静默装错轴）；
4. 硬限位（config 四键，拦指令）+ URDF 限位核对（只告警）+ 软限位（须位于硬限位内，超出抛错）；
5. 特征位形（zero/neutral 恒全零；home 取 config，越限自动裁剪并告警）+ `tcp_limits`；
6. 运行参数（`joyarm.runtime` 四键，缺省 10 Hz / 0.15 s / 0.05 s / 10 s）；
7. 创建通信后端（`backend.name` 经 `get_backend`；失败即构造失败）；
8. 创建六域算法成员（任一失败即构造失败；第一个设为激活）。

### 2.2 成员变量

（`__init__` 先全部赋默认值再逐段填真值；`_` 前缀为私有）

| 分组 | 变量 | 简介 |
|:--|:--|:--|
| 基本信息 | `model` | 型号名（= yaml 文件名 = URDF 目录名） |
| | `_config` | 配置快照（构造时深拷贝；`get_config()` 另返深拷贝，约束8） |
| | `_urdf_path` | URDF 文件路径 |
| | `connected` | 是否已连真机（创建后默认 `False`） |
| | `ee_frame_name` | 末端坐标系名（config `joyarm.end_frame`，缺省 `ee`） |
| 运动学模型 | `pin_model` | pinocchio 构型模型（只读共享；锁定非本体关节后 nq = n_arm） |
| | `pin_data` | 给用户直连的 `pin.Data`（内部求解另建私有 Data，不共用） |
| | `ee_frame_id` | 末端坐标系在模型里的编号 |
| 本体 arm | `n_arm` | 本体关节数（config `backend.arm.joints` 条数） |
| | `arm_limits` | 本体硬限位（真正拦指令） |
| | `arm_limits_soft` | 本体软限位（只供上层状态判断，不拦指令；缺省软=硬） |
| | `_arm_joint_names` | 本体关节名列表（config 顺序 = q 向量顺序） |
| | `_arm_zero` / `_arm_home` / `_arm_neutral` | 零位（全零）/ home 位形（config）/ 数值求解默认初值（全零） |
| | `_arm_enabled` | 使能标志缓存（`enable/disable_arm` 维护；connect 时复位 `False`） |
| 末端 end | `n_end` | 末端电机数（无末端为 0） |
| | `end_limits` / `end_limits_soft` | 末端硬/软限位（语义同 arm） |
| | `_end_joint_names` | 末端电机名列表（config 顺序） |
| | `_end_zero` / `_end_home` / `_end_neutral` | 语义同 arm 对应项 |
| 空间限位 | `tcp_limits` | 末端空间限位（config `joyarm.tcp_limits` 配置后生效） |
| 六域算法 | `_fkine_solvers` / `_ikine_solvers` / `_jacobian_solvers` / `_dynamics_solvers` / `_traj_planners` / `_controllers` | 六个 `{注册名: 实例}` 字典（config 选型、全部加载） |
| | `_active_name` | 各域当前激活的注册名（有且仅有一个） |
| 通信后端 | `_backend` | 整机通信后端（config `backend.name` 经 `get_backend` 构建） |
| 轨迹桥 | `_target_traj` | 目标帧序列（应用线程写；写入口深拷贝） |
| | `_current_frame` | 当前轨迹帧（规划线程写；首帧发布前为 `None`） |
| 运行状态 | `is_normal` | 运行正常标志（`False` 时控制线程不下发指令；由应用置位） |
| | `_state_thread` | 状态保活线程（connect 时启动） |
| | `_motion_running` | 运动管线是否在跑（防重复启停） |
| | `_motion_threads` | 管线三线程 `{线程名: _PeriodicThread}`（traj-plan / traj-sample / ctrl-step） |
| | `_traj_planned_by` | 最近完成规划的规划器实例（运行中换规划器的过渡保护） |
| | `_switching` | 换算法暂停开关（`threading.Event`；置位期间控制线程暂停派发） |
| 运行参数 | `_keepalive_hz` | 空闲状态保活频率（config `runtime.state_keepalive_hz`，缺省 10 Hz） |
| | `_stale_after` | 缓存数据过期阈值（`state_stale_timeout`，缺省 0.15 s） |
| | `_poll_interval` | 到位轮询间隔（`move_poll_interval`，缺省 0.05 s） |
| | `_wait_timeout` | 运动等待超时（`move_wait_timeout`，缺省 10 s） |
| 类常量 | `_SPIN_MARGIN` | 精确发送定时余量（0.001 s：先 sleep 到点前 1 ms 再自旋） |

### 2.3 私有方法

| 分组 | 方法 | 简介 |
|:--|:--|:--|
| 初始化助手 | `_resolve_robot_urdf(robot)`（static） | 找 `robot_model/<robot>/urdf/<robot>.urdf`；缺失抛 `ValueError`（列可用型号） |
| | `_build_pin_model(urdf_path)`（static） | 解析 URDF → pin 模型：锁定 joint1~9 之外的活动关节（nq 收敛为 n_arm、帧全保留）；URDF 仍为几何/限位/命名的权威源 |
| | `_urdf_arm_nq(pin_model)`（static） | 数本体关节（名字在 `_ARM_JOINT_NAMES` 内）自由度总数 |
| | `_urdf_joint_limits(urdf_path)`（static） | 解析 URDF 各活动关节限位 → `{关节名: {q_min/q_max/dq_max/tau_max, mimic}}` |
| | `_check_limits_vs_urdf(urdf_path, arm_jcfgs, end_jcfgs)`（static） | config 限位 vs URDF 三类核对（关节缺失 / 数值不一致 / URDF 多余非 mimic 本体关节），只告警不拦初始化 |
| | `_apply_tcp_limits(tl)` | 用 config `tcp_limits` 段覆盖默认末端空间限位（解析在 utils） |
| 六域成员管理 | `_members(domain)` | 域 → 成员字典 |
| | `_active(domain)` | 取当前激活成员；域未配置/未加载抛 `RuntimeError` |
| | `_pick(domain, name)` | 按注册名取已加载成员（`None` = 当前激活） |
| | `_solve(domain, method, *args, **kw)` | 统一求解入口：取激活成员调其 `method`（纯转发；Pin 求解器每次新建私有 Data，多线程安全） |
| 前置校验 | `_require_connected()` | 未连接抛 `RuntimeError` |
| | `_require_enabled_arm()` | 本体未全使能抛错（查 `_arm_enabled` 缓存；运动/下发指令前用，急停类不用） |
| | `_require_enabled_end()` | 末端未全使能抛错（查 `get_end_state()` 实时状态；无末端跳过） |
| | `_require_mode_arm(mode, joint)` | 本体当前模式 ≠ `mode` 抛错（查本地缓存；`set_arm_command` 用） |
| 状态保活线程 | `_start_state_keepalive()` | 启动保活线程（connect 自动调；已在跑不重复启动） |
| | `_stop_state_keepalive()` | 停止保活线程（disconnect 自动调） |
| | `_keepalive_step()` | 保活单步：缓存陈旧（说明无指令在发）才主动查电机——发指令期间零总线占用；后端不支持查数据新旧时静默跳过 |
| 运动管线单步 | `_tick_plan()` | 规划线程单步：`plan_once` **成功**才记录 `_traj_planned_by` |
| | `_tick_sample()` | 采样线程单步：当前规划器 = 最近完成规划者才发布帧（换规划器过渡保护） |
| | `_tick_ctrl()` | 控制线程单步：`_switching` 置位期间先不发（切换流程自己补一拍） |
| | `_dispatch_ctrl()` | 控制派发一步：`is_normal` → 读当前帧 → 控制器 `compute` → `set_arm_command` |
| | `_sync_traj_tick()` | 同步跑一拍"规划 + 采样发布"（启动/换算法时当前帧立即可用；失败沿用旧帧不抛错，线程按节拍重试自愈） |
| 紧急阻尼 | `_damping_frames(kd=5.0)` | 阻尼指令序列：切 MIT → 发阻尼帧 → 尽力使能 → 再发一帧（arm + end，每步尽力、失败只告警） |
| 直连运动内核 | `_paced_send(ts, send)` | 按时间表逐帧下发：sleep + 自旋精确定时；某帧迟了跳过等待不追赶（`move_j`/`_safe_move` 共用） |
| | `_safe_move(q_arm, q_end, t, rate, wait_tol, wait_timeout)` | 安全运动内核（safe_home/zero 用）：本体 MIT 阻抗三次曲线（kp/kd 缺省 config 增益、重力前馈取起点常值）+ 末端位置模式单目标 + 末段等到位 |
| | `_is_end_in_position(q_end, tol_q)` | 末端到位判断（每个电机都进容差；无末端恒 `True`） |
| | `_wait_in_position(q, tol_q, timeout)` | 按轮询间隔查到位；超时前最后再查一次并返回结果 |

### 2.4 公开接口（源码十二大类）

#### 一、配置与自检

| 方法 | 简介 |
|:--|:--|
| `get_config()` | 返回配置深拷贝（教学数据读取口，如 MDH；勿重读 yaml） |
| `check_config(model, config)`（静态） | 配置静态自检（不接硬件；`__init__` 第一步自动调）：joyarm 段齐全（`end_frame`）、URDF 存在、backend 必配、四键限位齐全且 `q_min ≤ q_max`、关节名无重复、channel 必填且同总线参数一致、同总线电机 ID 无冲突、`control_rate`/runtime 为正数、home/软限位长度匹配、robotics 段格式合法；一条消息列出全部问题 |
| `check_hardware()` | 硬件自检：临时连接（不使能）逐个查本体/末端通讯与故障，查完还原连接状态；有问题抛 `RuntimeError` 列全部 |

#### 二、连接与断开

| 方法 | 简介 |
|:--|:--|
| `connect()` | 连接真机（本体 + 末端；只连不使能）；复位 `_arm_enabled`、启动状态保活线程 |
| `disconnect()` | 断开：停管线 → 停保活 → 后端断开，各步尽力执行（失败只告警继续） |
| `__enter__` / `__exit__` | `with arm:` 支持：进入未连接自动 connect；退出停管线 → 失能本体/末端 → 断开（各步尽力） |

#### 三、使能与控制模式

| 方法 | 简介 |
|:--|:--|
| `enable_arm(joint=None)` / `disable_arm(joint=None)` | 本体使能/失能（维护 `_arm_enabled` 缓存；重连后须重新使能） |
| `enable_end(joint=None)` / `disable_end(joint=None)` | 末端使能/失能（不维护缓存，走实时状态查询） |
| `set_mode_arm(mode=POSITION, joint=None)` / `set_mode_end(…)` | 切控制模式（位置/速度/MIT；默认位置模式） |
| `read_mode_arm(joint=None)` / `read_mode_end(joint=None)` | 查当前模式（读本地缓存不发总线；`joint=None` 时各关节一致才返回该模式，否则 `None`） |

#### 四、读状态

| 方法 | 简介 |
|:--|:--|
| `get_arm_state()` | 本体状态 `ArmState`（关节角/速度/力矩/温度、故障标志、通讯异常汇总）；**新鲜度感知**——缓存新鲜（age ≤ 0.15 s）零总线、陈旧先查一次；配了 fkine 时填充 `tcp.pose`（外壳拷贝防污染缓存） |
| `get_end_state(joint=None)` | 末端状态 dict（q/dq/tau、使能、故障、通讯、温度）；同样新鲜度感知 |

#### 五、手动下发指令

| 方法 | 简介 |
|:--|:--|
| `set_arm_command(mode=POSITION, q/dq/tau/kp/kd, joint=None)` | 手动单条指令：POSITION 要 `q`、VELOCITY 要 `dq`、MIT 要 `q/dq/tau`（`kp/kd` 缺省 config MIT 增益）；前置校验 connected + 模式一致 |

#### 六、末端执行器（夹爪）

| 方法 | 简介 |
|:--|:--|
| `set_end_open(joint=None)` | 张开到最大（默认行程/力度） |
| `set_end_close(joint=None)` | 闭合（夹到默认力度即停） |
| `set_end_zero(joint=None)` | 归零（目标 = 电机 0 弧度，越行程自动裁剪） |
| `set_end_position(position, joint=None)` | 位置控制（连续量，如夹爪电机弧度） |
| `set_end_tau(tau, joint=None)` | 力矩控制（电机力矩 N·m） |

#### 七、运动（`move_j` 为独立直连通道，与规划管线并行）

| 方法 | 简介 |
|:--|:--|
| `move_j(q, t=None, rate=None, wait_tol=0.05, wait_timeout=None)` | 关节运动到目标角（阻塞到到位/超时）：入口裁硬限位（判定以裁剪后为准）→ 三次多项式 → 切位置模式按频率逐帧下发 → 轮询等到位；`t` 缺省按最大关节行程估计（隐含峰值约 1.5 rad/s）、`rate` 缺省 config `control_rate`（≤1000） |
| `safe_home(t=None, wait_tol, wait_timeout)` | 安全回 home（本体 + 末端）：本体 MIT 阻抗 + 末端位置模式 |
| `home_to_zero(t=None, wait_tol, wait_timeout)` | home → 零位；先确认当前在 home 容差内，否则 `RuntimeError` |
| `safe_zero(t=None, wait_tol, wait_timeout)` | 安全回零 = `safe_home` + `home_to_zero` |
| `move_l(pose, t=None)` 🟡 | 末端直线运动占位（待 ikine + 笛卡尔规划） |
| `teach_start()` / `teach_play()` 🟡 | 拖动示教 / 回放占位（待重力补偿：MIT + gravity 前馈 + kp=0） |
| `teleop_keyboard()` 🟡 | 笛卡尔键盘遥操作占位 |

#### 八、安全与急停（任何状态可用；出事先 `damping_mode`）

| 方法 | 简介 |
|:--|:--|
| `damping_mode(kd=5.0)` | 紧急阻尼：**任何状态**全电机（含末端）切 MIT 纯阻尼（默认 5.0 = DM kd 编码量程上限） |
| `hold_position(kp=None, kd=None, tau=None)` | 原地 MIT 保持（当前位置为目标；`tau` 缺省自动重力前馈，dynamics 未配置则置零并告警） |
| `lock_position()` | 急停锁定：切位置模式锁当前关节角（不需先使能） |
| `is_in_position(q=None, pose=None, frame=None, tol_q=0.05, tol_pos=1e-3, tol_rot=1e-2)` | 到位判断（`q`/`pose` 恰给其一）：关节全进 `tol_q` 容差；或末端位置误差 ≤ `tol_pos` 且姿态误差角 ≤ `tol_rot`（现算 fkine，不用缓存 `tcp.pose`） |

#### 九、标零、故障清除与电机参数读写

| 方法 | 简介 |
|:--|:--|
| `set_zero_arm(joint=None)` / `set_zero_end(joint=None)` | 零位标定（把当前位置记为零点；失能/无故障门禁在后端） |
| `clear_fault_arm(joint=None)` / `clear_fault_end(joint=None)` | 清电机故障（验证式清错复位；未恢复则后端抛错汇总） |
| `read_param_arm(key, joint=None)` / `read_param_end(…)` | 读电机参数（`joint` 指定返单值，`None` 返逐电机列表） |
| `write_param_arm(key, value, joint=None, persist=False)` / `write_param_end(…)` | 写电机参数（`persist=True` 同时存闪存） |

#### 十、基本信息与只读属性（均返回拷贝）

| 成员 | 简介 |
|:--|:--|
| `arm_zero` / `arm_home` / `arm_neutral`（property） | 本体零位 / home 位形 / 数值求解初值 `(n_arm,)` |
| `end_zero` / `end_home` / `end_neutral`（property） | 末端对应项 `(n_end,)` |
| `joint_names_arm` / `joint_names_end`（property） | 关节名列表（config 顺序 = q 向量顺序） |
| `joint_index_arm(name)` / `joint_index_end(name)` | 关节名 → 索引（未找到抛 `ValueError` 列可用名） |
| `rand_q_arm(size=None, rng=None)` | 本体硬限位内随机采关节角（`size=N` 给 `(N, n_arm)`；教程工作空间采样用） |
| `__repr__` | 型号 / 关节数 / 末端帧 / 各域激活名 / 连接态一览 |

#### 十一、数学计算（不连真机可用；统一经 `_solve` 派发给激活成员）

| 方法 | 简介 |
|:--|:--|
| `fkine(q, frame, rep="pose")` | 正运动学（`frame` 必填：帧名或编号；`rep`：pose / T / se3） |
| `ikine(target, frame, q0, **kw)` | 逆运动学单解（`q0` 必填 = 数值法迭代起点） |
| `ikine_all(target, frame, **kw)` | 逆运动学全解 `(K, n_arm)`（逐解 ±2π 平移尽量落进限位；数值法不支持抛 `RuntimeError`） |
| `jac(q, frame, ref="base")` | 雅可比（`ref`：local 末端系 / base 基座系） |
| `manipulability(q, frame)` | 可操作度（越大离奇异越远） |
| `statics(q, F, frame)` | 静力学 `τ = JᵀF`（`F` 六维：力 N + 力矩 N·m） |
| `mass_matrix(q)` / `gravity(q)` | 关节空间惯量矩阵 M(q) / 重力项 G(q) |
| （不设门面） | idyn / coriolis / cartesian_inertia / fkine_vel / ikine_vel——经激活求解器实例调用（实例由 `set_solver` 返回值取得） |

#### 十二、进阶：运动管线与运行中换算法

数据流：`set_target_traj` 写目标 → 规划线程算系数 → 采样线程产当前帧 → 控制线程算指令并自动下发。

| 方法 | 简介 |
|:--|:--|
| `set_target_traj(targets)` | 写运动目标（`TrajFrame` 单帧或帧列表；写深拷贝；`None`/空 = 清空 → 规划器回退规划回 q_home） |
| `get_target_traj()` | 读当前目标序列（返回内部快照引用，规划期间勿改动） |
| `set_current_frame(frame)` / `get_current_frame()` | 写/读当前轨迹帧（规划线程写、控制线程读；首帧发布前 `None`） |
| `start_motion()` | 启动管线三线程 traj-plan / traj-sample / ctrl-step（频率取规划器 `plan_hz`/`sample_hz` 与控制器 `ctrl_hz`）；按 `Controller.MODE` 自动切电机模式 + 同步算出首帧；已在跑无副作用；traj/control 域未配置抛 `RuntimeError` |
| `stop_motion(damping=True)` | 停三线程（卡住 1 s 抛 `RuntimeError` 列出卡者）；清当前帧回初始态；默认紧接着切纯阻尼防下坠（`damping=False` 关闭） |
| `set_solver(domain, name)` | 切某域激活算法（每域有且仅一个激活），返回算法实例；管线运行中切 control / traj 走**热切换**：暂停一拍 → 切换（control 先按新 `MODE` 切电机模式）→ 立即用新算法算一拍 → 调线程频率 → 恢复 |
| `list_solvers(domain)` | 列某域已加载算法名（当前激活排第一，其余按字母序） |
