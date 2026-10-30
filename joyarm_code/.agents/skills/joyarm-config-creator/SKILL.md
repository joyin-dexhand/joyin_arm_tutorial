---
name: joyarm-config-creator
description: "为 JoyArm 机械臂新型号创建、修改或排错型号配置文件 config/<型号>.yaml，或检查配置合法性时使用；覆盖型号配置、config yaml、template 模板起步、三段结构（joyarm/robotics/backend）逐键填写、软硬限位与注册名在册校验、与 URDF 限位一致性核对等触发场景。"
---

# 型号配置文件创建（joyarm-config-creator）

JoyArm 是**配置驱动**的整机对象：`joyArm("<型号>")` 构造时按 `joyarm_core/config/<型号>.yaml` 把整机参数（joyarm 段）、算法层（robotics 段）、通信层（backend 段）一次性装配成机械臂对象，并与 `robot_model/<型号>/` 下的 URDF 交叉核对。本 skill 是**新型号创建四件套的第一步**：提供配置模板 [assets/template.yaml](assets/template.yaml) 与完整性检查脚本 [scripts/config_check.py](scripts/config_check.py)，覆盖配置的**创建 → 完善更新 → 测试**全流程；URDF 数字模型见 [joyarm-urdf-creator](../joyarm-urdf-creator/)、backend 通信实现见 [joyarm-backend-creator](../joyarm-backend-creator/)、整机集成验收见 [joyarm-arm-creator](../joyarm-arm-creator/)。四件套依序为 config → urdf → backend → arm。

**命名链**（三处必须同名，任一不符即校验失败）：型号名 = `config/<型号>.yaml` 文件名 = `robot_model/<型号>/` 目录名 = URDF 文件名。

## 1 型号配置文件的三段结构

| 顶层段 | 层次 | 职责 | 主要消费方 |
|:---:|:---:|:---:|:---:|
| `joyarm` | 整机层 | 末端坐标系、软限位、末端空间限位、特征位形、运行与状态机参数 | JoyArm 整机对象 |
| `robotics` | 算法层 | 六域求解器/规划器/控制律的加载规格（按注册名） | robotics 六子包 REGISTRY |
| `backend` | 通信层 | 硬件后端选择、arm/end 逐关节电机表（含硬限位与增益） | backend REGISTRY 与型号后端 |

### 1.1 joyarm 段（整机层）逐键说明

| 键 | 含义 | 类型 / 单位 | 约束与备注 |
|:---:|:---:|:---:|:---:|
| `frame_end` | 指定的末端坐标系名（TCP） | 字符串 | 须为 URDF 中存在的 pin 模型帧，缺失则构造失败 |
| `joint_limits_soft_arm` | arm 本体软限位 | 四键 `q_min/q_max/dq_max/tau_max`，各为 n_arm 元列表（rad / rad/s / N·m） | 须整体位于硬限位（backend joints 四键）内；超软限位 → 状态异常 → 急停恢复；`neutral` = 软限位中点（代码实现） |
| `joint_limits_soft_end` | end 末端软限位（电机空间） | 同上四键，标量或 n_end 元列表 | 同上；无 end 关节时可不配 |
| `tcp_limits` | `frame_end` 系三维空间限位 | `workspace_box` 为 3 行 `[min, max]`（x/y/z，m）；`v_lin_max`(m/s)、`v_ang_max`(rad/s)、`f_max`(N)、`t_max`(N·m) | `workspace_box` 每轴须 min < max；四个上限须 > 0 |
| `home_arm` | arm 归位位形 | n_arm 元列表（rad） | 逐关节须在**软限位**内（软 ⊆ 硬，故亦在硬限位内） |
| `home_end` | end 归位位形 | n_end 元列表（rad） | 同上 |
| `runtime` | 运行参数 | `move_wait_timeout`（s） | 运动等待超时；非法时告警并沿用缺省 10.0 s |
| `MotionFSM` | 运动状态机参数 | `fsm_hz`、`monitor_hz`(Hz)；`damping_kd`（n_arm 元，正有限）；`damping_k_gravity`（n_arm 元，[0,1]） | DAMP 阻尼态逐关节速度阻尼与重力补偿率；`damping_kd/k_gravity` 未配置或非法则构造失败 |
| `arrival_tol` | 到位判断容差 | `arm.q`(n_arm 元，rad)、`arm.pos`(m)、`arm.rot`(rad)、`end.q`(n_end 元，rad) | 须为正有限数；缺省回退 `arm.q`=`end.q`=0.05 rad、`arm.pos`=1e-3 m、`arm.rot`=1e-2 rad（两组关节容差共用 0.05 标量缺省；标量广播或逐关节列表均可） |

要点：

1. **硬限位不写在本段**——写在各关节电机表（backend 段 `arm/end.joints` 条目）的 `q_min/q_max/dq_max/tau_max` 四键，backend 下发指令按硬限位裁剪；
2. **zero 与 neutral 不写进配置**——zero 为全零位形（须落在硬限位内）、neutral 为软限位中点，均由代码实现；
3. **可选键** `arm_mdh`（6×4 名义 MDH 参数）与 `T_linkn_endtcp`（末连杆系→末端系固定位姿）：仅当 `robotics.fkine` 使用 `mdh_fkine_solver` 时必配，写法参考 [rebot_arm_b601_dm.yaml](../../../joyarm_core/config/rebot_arm_b601_dm.yaml)；
4. 待标定数值建议行尾标注 `# ！ 待标定`（参考型号同款写法），真机标定后回填。

### 1.2 robotics 段（算法层）逐键说明

每个域的**规格**有两种等价写法：注册名字符串，或 `{name: 注册名, **构造参数}` 字典（见 [assets/template.yaml](assets/template.yaml) 的 robotics 段）。注册名规则：**类名小写 + 下划线**（如 `PinFkineSolver` → `pin_fkine_solver`）。

| cfg 段键 | 域缩写 | REGISTRY 注册名（当前全部） | 常用构造参数 |
|:---:|:---:|:---:|:---:|
| `fkine` | fkine | `pin_fkine_solver`、`mdh_fkine_solver` | mdh 实现须另配 joyarm 段 `arm_mdh`/`T_linkn_endtcp` |
| `ikine` | ikine | `pin_ikine_solver`、`analytic_ikine_solver` | 数值法：`damping`、`step_max`、`restarts` |
| `jacobian` | jac | `pin_jacobian_solver` | —— |
| `dynamics` | dyn | `pin_dynamics_solver` | —— |
| `trajectory` | traj | `single_joint_hold_traj_planner`、`single_joint_cubic_traj_planner` | `plan_hz`（**必配**） |
| `control` | ctrl | `joint_position_controller` | `ctrl_hz`（**必配**） |

要点：

1. 每域可列多条（列表首位 = 默认激活），运行期 `set_<域>_solver()` 热切换；
2. `trajectory` 条目的 `plan_hz` 与 `control` 条目的 `ctrl_hz` 为**起搏键**：被 JoyArm 消费（不透传求解器构造），须为 int 或整值浮点且 ≥ 1，缺失即构造失败；
3. 注册名不在对应域 REGISTRY → 构造失败（错误消息附可用注册名）；新增求解器须先在 `joyarm_core/robotics/<域>/__init__.py` 的 REGISTRY 注册。

### 1.3 backend 段（通信层）逐键说明

顶层运行键：

| 键 | 含义 | 缺省行为 |
|:---:|:---:|:---:|
| `name` | backend 注册名（`backend/__init__.py` REGISTRY 的键），与型号对应 | 无缺省，**必配**且须在册（新型号先完成 [joyarm-backend-creator](../joyarm-backend-creator/)） |
| `state_refresh_hz` | 状态新鲜度承诺（Hz） | 缺省 20 |
| `write_settle` | 指令帧生效等待（s；连接后默认模式帧 / 断连前失能帧等） | 缺省 0.1 |
| `rx_hz` | RX 收取线程节拍（Hz，串口阻塞读超时 = 1/rx_hz） | 缺省 1000 |

`arm` / `end` 两子段同构（end 为执行器电机组，可多电机），共同键：

| 键 | 含义 | 备注 |
|:---:|:---:|:---:|
| `channel` | 总线设备（如 `/dev/ttyACM0`） | end 与 arm 相同 → 共享单总线（一个 RX 线程 + 一把 TX 锁）；不同 → 独立总线 |
| `protocol` | 总线类型（如 `serial_can`） | 信息性标识，实际总线由型号 backend 指定 |
| `baud_rate` | 通讯波特率 | —— |
| `default_mode` | 连接成功后自动设置的控制模式 | `mit` / `position` / `velocity` |
| `joints` | 逐关节电机表（列表，每条目一个字典） | arm 命名 `joint1~jointN`、end 命名 `joint_end1~joint_endM`，与 URDF 同名同序 |

逐关节电机表各键：

| 键 | 含义 | 单位 / 取值 | 约束 |
|:---:|:---:|:---:|:---:|
| `name` | 关节名（状态报告 / 模式缓存的键） | 字符串 | 与 URDF 驱动关节同名同序，不可重复 |
| `motor_id` | CAN 指令 ID | 十六进制（如 `0x01`） | 总线内唯一 |
| `feedback_id` | 应答帧 CAN ID | 十六进制 | 总线内唯一 |
| `model` | 电机型号 | 字符串（如 `"4340P"`） | 信息性 |
| `q_min` / `q_max` | 行程下 / 上限（**硬限位**） | rad | 须与 URDF `<limit lower/upper>` 一致（容差 1e-4），且区间须含零位 |
| `dq_max` | 速度上限（硬限位） | rad/s | 须与 URDF `<limit velocity>` 一致 |
| `tau_max` | 力矩上限（硬限位） | N·m | 须与 URDF `<limit effort>` 一致 |
| `MIT.kp` / `MIT.kd` | MIT 模式默认增益（指令未传 kp/kd 时使用） | kp max=500；kd max=5 | 建议配非零值 |
| `POS_VEL.pos_kp` / `pos_ki` | 位置环增益（电机寄存器） | —— | 位置/速度两模式共用 |
| `POS_VEL.vel_kp` / `vel_ki` | 速度环增益（电机寄存器） | —— | 两模式共用 |
| `POS_VEL.vlim` | 位置/速度指令限速 | rad/s | —— |
| `POS_VEL.flim` | 归一化力矩电流上限 | 0~1 | flim 随**位置/速度指令帧**下发：DM 电机 POSITION 模式实为力位混合（Torque_Pos）模式，控制帧 `pos_force`（CAN ID `0x300+SlaveID`）内的电流字段按 0-10000 标幺——下发时 `flim × 10000` 打包进该字段，如 flim=0.5 表示允许输出 50% 电流上限 |

## 2 从 template.yaml 起步的创建流程

> 📌 **前置依赖**：本 skill 只创建四件套中的第 1 件（config yaml）。H13（backend.name 在册）依赖 [joyarm-backend-creator](../joyarm-backend-creator/) 已完成后端注册；H14/H15（关节名与四键限位与 URDF 一致）依赖 [joyarm-urdf-creator](../joyarm-urdf-creator/) 已就位模型资产——这两项未完成时对应检查行会 FAIL，**属预期**，按 urdf → backend → arm 顺序补齐其余件后复跑本脚本直至 SUCCESS。**硬限位数值的最初来源**是电机数据手册 / 机械结构设计（行程、额定速度、额定力矩），新型号先据手册定下四键数值（先填本篇 config 或先填 urdf 篇 URDF 均可），再保证两侧逐键一致。

1. **复制模板**（文件名即型号名；以下命令均在 `joyarm_code/` 目录下执行，路径方才成立）：

   ```bash
   cp .agents/skills/joyarm-config-creator/assets/template.yaml joyarm_core/config/<型号>.yaml
   ```

2. **填 joyarm 段**：按真机填写软限位（= 硬限位整体内缩）、`home` 位形、`tcp_limits`、`MotionFSM` 阻尼增益等；模板数值仅示范结构，须逐项按真机修改；
3. **填 robotics 段**：按 §1.2 注册名表选择各域实现并写构造参数（可直接参照参考型号配置）；
4. **填 backend 段**：`backend.name` 指向新型号后端注册名（先在 [joyarm-backend-creator](../joyarm-backend-creator/) 完成子类实现与 REGISTRY 注册），再逐关节填电机表——四键硬限位须与 URDF `<limit>` 标定一致；
5. **放置模型资产**：URDF + meshes 放入 `joyarm_core/robot_model/<型号>/urdf/` 与 `meshes/`（见 [joyarm-urdf-creator](../joyarm-urdf-creator/)）；
6. **运行 §5 测试方法验收**，直至 `SUCCESS`。

## 3 硬性约束清单

以下约束与库 `JoyArm` 初始化自检同口径，任一违反 → 构造抛 `ValueError`（消息附具体原因与可用型号列表）：

| 编号 | 约束 | 具体口径 |
|:---:|:---:|:---:|
| C1 | 软限位在硬限位内 | arm/end 软限位四键的有限值逐维不越出对应硬限位（±∞ 维不设软限） |
| C2 | zero 在硬限位内 | 全零位形（ZERO 态回零目标）逐关节落在 `[q_min, q_max]` 内（1e-12 容差） |
| C3 | home 在软限位内 | `home_arm/home_end` 维度 = 关节数且逐关节在软限位内（软 ⊆ 硬，故亦在硬限位内） |
| C4 | robotics 六键为各域 REGISTRY 注册名 | 每条规格须为「注册名字符串或含 `name` 的字典」，注册名在对应域 REGISTRY 在册；traj/ctrl 起搏键必配且整值 ≥1 |
| C5 | backend.name 在 backend REGISTRY | 须为 `backend/__init__.py` REGISTRY 的键 |
| C6 | 关节名与 URDF 一致 | cfg 登记与 URDF 驱动关节（非 fixed、非 mimic）**同名同序且全部登记**；命名 arm `jointN`、end `joint_endN` 从 1 连续；fixed / mimic 关节不允许登记 |
| C7 | 硬限位四键与 URDF `<limit>` 一致 | 逐关节 `q_min/q_max/dq_max/tau_max` 与 URDF 比对，容差 1e-4 |
| C8 | 基础完整性 | 顶层三段齐全且为字典；`joyarm.frame_end` 已配置；`backend.arm.joints` 非空且条目含 `name` |

## 4 完善更新方法

1. **改配置须重建对象**：cfg 仅在 `joyArm("<型号>")` 构造时读取并一次性装配，运行中的对象**不会回读**配置文件——修改 `<型号>.yaml` 后，须对旧对象 `close()` 再重新 `joyArm("<型号>")` 才生效；
2. **只读快照**：`arm.get_config()` 返回 cfg 深拷贝快照，用于核对当前对象装配所用配置，对快照的修改不影响对象；
3. **待标定项迭代**：软限位、增益、阻尼参数等以 `# ！ 待标定` 标注的项，按「真机标定 → 回填 yaml → 重建对象 → 复跑 §5 测试」循环收敛；
4. 软限位取值方法：以硬限位（backend joints 四键 = URDF `<limit>`，最初来源见 §2 前置依赖的说明）为基准**整体内缩**，参考型号内缩量约 0.03~0.3 rad。

## 5 测试方法

> 📌 **SUCCESS 的完整含义**：H1~H15 全过 + `joyArm` 离线构造成功，其中 H13 依赖 backend 注册、H14/H15 依赖 URDF 就位——严格按 config → urdf → backend 顺序先做本篇时，这几行 FAIL 属预期（见 §2 前置依赖），补齐其余件后复跑直至 SUCCESS。

1. **完整性检查脚本**（离线）：

   ```bash
   # 在 joyarm_code/ 目录下（--model 缺省 rebot_arm_b601_dm）
   python .agents/skills/joyarm-config-creator/scripts/config_check.py --model <型号>
   ```

   输出 ①~⑤ 五节：目标定位 → 基础校验表（B1~B3）→ 硬性约束表（H1~H15，逐条覆盖 §3 的 C1~C8）→ 终极校验 → 总结。全过打印 `SUCCESS`（退出码 0），任一 `FAIL` 退出码 1（每行说明列给出具体原因与修复指引）；
2. **终极校验**：脚本 ④ 节即 `joyArm("<型号>")` 离线构造——通过库内全套自检（cfg 交叉校验 → URDF 装配 → 求解器实例化），等价于工厂四步校验；通过后 `listArm()` 应能列出该型号（三条件：config 三段齐全 + backend.name 在册 + robot_model 四路径齐全）。自行验证方式（在 `joyarm_code/` 目录下执行）：

   ```bash
   python -c "from joyarm_core import listArm; listArm()"
   ```

3. **后续真机验证**：配置与构造通过后，按 [02 连接与状态读取](../../../quick_start/02_connect_read.py) 连接真机核对限位/位形读数，整机创建全流程验收见 [joyarm-arm-creator](../joyarm-arm-creator/)。
