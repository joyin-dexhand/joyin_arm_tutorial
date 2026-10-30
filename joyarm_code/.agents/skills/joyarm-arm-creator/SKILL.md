---
name: joyarm-arm-creator
description: "为 JoyArm 机械臂新型号做整机集成时使用——四件套装配核对、joyArm 工厂四步校验排障、整机自检与「离线→真机」验收阶梯；覆盖整机创建失败排查、缺件定位、新型号交付验收、型号维护/固件升级后的回归冒烟等场景，串联 joyarm-config-creator / joyarm-urdf-creator / joyarm-backend-creator 三个制作 skill。"
---

# JoyArm 整机集成创建

把 [joyarm-config-creator](../joyarm-config-creator/)、[joyarm-urdf-creator](../joyarm-urdf-creator/)、[joyarm-backend-creator](../joyarm-backend-creator/) 三个制作 skill 的产出装配成一台可被 `joyArm("<型号>")` 直接创建的完整机械臂：核对四件套、跑四步校验定位缺件、按验收阶梯逐级交付。

## 1. 新型号 = 四件套

一台新型号机械臂由四件套构成，前三件分别由三个制作 skill 制作（各自配有件级自检脚本），第四件「注册与自检」由本 skill 完成：

| # | 件 | 库内位置 | 制作 skill | 件级自检 |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 型号配置 | `joyarm_core/config/<型号>.yaml` | [joyarm-config-creator](../joyarm-config-creator/) | `config_check.py` |
| 2 | 模型资产 | `joyarm_core/robot_model/<型号>/`（URDF + meshes） | [joyarm-urdf-creator](../joyarm-urdf-creator/) | `urdf_check.py` |
| 3 | 硬件后端 | `joyarm_core/backend/backend_<型号>.py`（REGISTRY 注册） | [joyarm-backend-creator](../joyarm-backend-creator/) | `backend_check.py` |
| 4 | 注册与自检 | 无新增文件（四步校验 + 验收阶梯） | 本 skill | `arm_check.py` |

前三件由**型号名**贯穿：`config/<型号>.yaml` 文件名 = `robot_model/<型号>/` 目录名 = `urdf/<型号>.urdf` 文件名；backend 与型号一一对应，注册命名因来源而异（第三方 `backend_<型号>` ↔ `<型号>`，官方 `backend_<版本>` ↔ `joyarm_<版本>`）。`listArm()` 列出的「可用型号」正是三件外部资源全部就位的型号（config 三顶层段齐全 + backend 命中 REGISTRY + 模型四路径齐全），自行验证（在 `joyarm_code/` 目录下执行）：

```bash
python -c "from joyarm_core import listArm; listArm()"
```

## 2. 四步校验流程

`joyArm("<型号>")` 工厂在创建对象前按**固定顺序**做四步校验（口径见 `joyarm_core/joyarm/__init__.py` 文件头「工厂四步固定顺序校验」）：任一步失败抛携带具体原因的异常，错误消息附可用型号列表；四步全过才进入对象使用。

| 步骤 | 校验内容 | 典型失败原因 | 不通过时回退 |
|:---:|:---:|:---:|:---:|
| ① config | `config/<型号>.yaml` 存在且解析为非空 dict | 文件未建 / 空文件 / yaml 语法错 | [joyarm-config-creator](../joyarm-config-creator/) |
| ② backend | cfg 的 `backend.name` ∈ `backend/__init__.py` 的 `REGISTRY` 键 | 注册名拼写不一致 / 子类未导入注册 | [joyarm-backend-creator](../joyarm-backend-creator/)（注册）；[joyarm-config-creator](../joyarm-config-creator/)（yaml 填写） |
| ③ robot_model | 四路径按序存在：`robot_model/<型号>/`、`meshes/`、`urdf/`、`urdf/<型号>.urdf` | 目录未建 / 文件名与型号名不符 | [joyarm-urdf-creator](../joyarm-urdf-creator/) |
| ④ 构造自检 | `JoyArm(<型号>)` 类内初始化自检（cfg → pin 模型 → 限位 → backend → 求解器依序装配） | config 数值与 URDF 不一致 / URDF 内容不合规 | 按报错前缀定位，回 [config](../joyarm-config-creator/) 或 [urdf](../joyarm-urdf-creator/) creator |

第 ①②③ 步查**外部资源就位**（缺文件、错命名立即暴露）；第 ④ 步查**内容正确**（限位逐键一致、`frame_end` 帧存在等由构造自检把关）。件级自检（三个制作 skill 的 check 脚本）通过 ≠ 整机可创建——四件之间的**互检**（如 ② 要求 config 的 yaml 与 backend 的注册名一致）只能由整机自检发现，三件产出后务必运行 `arm_check.py`。

## 3. 整机自检：scripts/arm_check.py

在 `joyarm_code/` 目录下运行（离线脚本，无需硬件）：

```bash
python .agents/skills/joyarm-arm-creator/scripts/arm_check.py --model <型号>    # --model 缺省 rebot_arm_b601_dm
```

脚本把工厂的四步校验**拆开逐项执行**：哪一步 FAIL 就打印该步的具体原因与回退 skill 指引，不必对着工厂一整串报错逐行猜；第 ④ 步用 `with joyArm(<型号>)` 离线构造（不触碰任何硬件），构造成功即打印关节数、关节名、规划/控制频率、home 位形等装配证据。判定标准：**四步全 PASS → 打印验收阶梯建议、退出码 0**；任一 FAIL → 打印回退指引、退出码 1（收尾时「未连接……豁免跳过」类日志属正常提示，非报错）。

## 4. 验收阶梯

整机四步全 PASS 只代表「能创建」；「能用」须按以下阶梯逐级验收——先离线后真机、先只读后运动，均通过即可交付该型号。各级命令均在 `joyarm_code/` 目录下运行：

| 级 | 例程 | 硬件 | 验证内容 | 通过产出 | 不通过时回退 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [`quick_start/01_model_create.py`](../../../quick_start/01_model_create.py) | 离线 | 对象创建与核心信息（限位 / 位形 / 求解器六域） | 控制台整机核心信息表 | 重跑 `arm_check.py` 定位缺件 → 回对应 skill |
| 2 | [`quick_start/05_offline_math.py`](../../../quick_start/05_offline_math.py) | 离线 | fk / ik / jac / dyn 离线计算与求解器设置切换 | 各域数值结果与切换日志 | 数值异常多为 config 与 URDF 参数不一致 → [config](../joyarm-config-creator/) / [urdf](../joyarm-urdf-creator/) |
| 3 | [`quick_start/02_connect_read.py`](../../../quick_start/02_connect_read.py) | 真机 | 连接 / 断开、双口径状态读取、模式与状态码 | 连接成功 + 结构化状态表 | 总线联调 → [backend](../joyarm-backend-creator/)；通道/波特率等参数 → [config](../joyarm-config-creator/) |
| 4 | [`quick_start/04_direct_command.py`](../../../quick_start/04_direct_command.py) | 真机 | 使能、清错、位置 / 速度 / MIT 直通指令、夹爪动作 | 指令执行与到位确认 | 协议实现 → [backend](../joyarm-backend-creator/)；增益 / 限位 → [config](../joyarm-config-creator/) |
| 5 | [`quick_start/06_fsm_motion.py`](../../../quick_start/06_fsm_motion.py) | 真机 | 运动状态机启停、安全事件、关节 / 笛卡尔运动 | FSM 状态转移与小幅运动完成 | MotionFSM 段与 robotics 段参数 → [config](../joyarm-config-creator/) |
| 6 | [`test/joyarm_test.py`](../../../test/joyarm_test.py) | 真机 | 真机冒烟：创建 + 连接 + 状态可视化（FSM 段按需启用） | 监控窗口 / 页面正常显示 | 按失败环节定位，回对应 skill |

阶梯使用要点：

1. **逐级推进，不过不升阶**：低级失败时先按回退列修正对应件、重跑 `arm_check.py` 确认四步仍全 PASS，再重试本级；
2. **真机级（3~6 级）前置**：清空机械臂工作半径内的人员与障碍物、手持急停手段，各级脚本自身的安全须知见其文件头；
3. 第 6 级 `test/joyarm_test.py` 的「FSM 段」指脚本内 ④『MotionFSM 真机测试』注释段（`arm.start_motion_fsm()` 等语句默认被注释）——需要验证 FSM 状态转移时取消该段注释再运行，启用后运动会真实发生，注意现场安全；该脚本亦是日后型号维护、固件升级后的快速回归入口；
4. 阶梯各级脚本首次运行前的环境准备（uv 安装、`uv sync`、激活环境）见各脚本文件头【环境与运行】段。
