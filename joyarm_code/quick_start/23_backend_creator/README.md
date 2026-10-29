# 23 硬件后端创建（backend_creator）

> 📌 **概述**：本单元是机械臂新型号创建全流程（21 配置 → 22 模型 → **23 后端** → [24 整机](../24_arm_creator/)）的第三步——为型号编写硬件后端子类 `joyarm_core/backend/backend_<型号>.py`，把厂商总线协议接入 `Backend` 抽象基类的统一接口。本文给出七步接入流程、DM 关节电机与 Feetech 舵机双家族差异总览，以及「本文件夹离线桩冒烟 → 真机六分组验收」的测试阶梯。**第一个子类的参考实现**：`joyarm_core/backend/backend_rebot_arm_b601_dm.py`（DM 族完整实现，逐方法对照模板阅读）。

## 1. 总述：backend 在框架中的位置

JoyArm 自上而下分三层，上层只面向抽象基类编程，型号差异全部收敛在子类：

| 层 | 代码 | 职责 |
|:---:|:---:|:---:|
| 编排层 | `joyarm_core/joyarm/joyarm.py` | 面向用户的整机功能（状态机、安全监视、求解器调度……） |
| 抽象基类 | `joyarm_core/backend/backend.py` | 校验/裁剪/模式门禁/状态刷新等全部通用编排 |
| 型号子类 | `joyarm_core/backend/backend_<型号>.py` | 只做单关节/单总线的厂商协议操作（本单元产物） |

选型链路：型号子类在 `joyarm_core/backend/__init__.py` 的 `REGISTRY` 注册表登记 → 型号 yaml 的 `backend.name` 键填注册名 → 创建 `joyArm("<型号>")` 时按 yaml 自动装配对应子类。注册名 = 子类文件名去 `.py`（如 `backend_rebot_arm_b601_dm` ↔ 类 `BackendRebotArmB601DM`）。

本文件夹三件套：

| 文件 | 作用 |
|:---:|:---:|
| `backend_template.py` | Backend 子类实现模板：29 个抽象方法的桩 + 逐方法实现约束（迁自 `joyarm_core/backend/`，已改为绝对导入） |
| `23_backend_check.py` | 离线自检脚本：注册链路 + 模板 29 桩两级检查（无需硬件） |
| `README.md` | 本文：七步接入流程 + 双家族差异 + 测试阶梯 |

子类三条架构约束（详见模板文件头）：**不得新增公开方法/属性**（上层只面向基类）；**不在子类重复基类已做的操作**（校验/裁剪/门禁由基类负责）；失败**上抛异常或返回 `False`/`None`** 即可（基类统一转 warn 优雅降级）。

## 2. 七步接入新硬件后端

以下七步与 `backend_template.py` 文件头 docstring 的步骤一一对应；以新型号 `<型号>`（如 `my_arm_v1`）为例。

### 2.1 第 1 步：拷贝模板、改名

以下命令在 `joyarm_code/` 目录下执行（源路径 `quick_start/23_backend_creator/...` 才成立）：

```bash
cp quick_start/23_backend_creator/backend_template.py joyarm_core/backend/backend_<型号>.py
```

随后全文改名：类名 `BackendTemplate` → `Backend<型号>`（驼峰，如 `BackendMyArmV1`）；模块级 `logger` 名同步替换——模板中仅**一处**定义（约第 43 行 `logger = logging.getLogger("joyarm_core.backend_<型号>")`），全局搜索原字符串 `joyarm_core.backend_<型号>` 并替换为 `joyarm_core.backend_<你的型号>` 即可。

### 2.2 第 2 步：复制 config 模板

若 `joyarm_core/config/<型号>.yaml` 尚不存在，从 21 单元复制配置模板并重命名：

```bash
cp quick_start/21_config_creator/template.yaml joyarm_core/config/<型号>.yaml
```

### 2.3 第 3 步：完善 backend 段

模板 yaml 的 `backend:` 段键结构（逐关节条目以 `arm.joints[0]` 为例，`end.joints[]` 同构）：

| 键 | 位置 | 作用 |
|:---:|:---:|:---:|
| `name` | 段顶层 | backend 注册名（第 6 步回填，见 2.6） |
| `state_refresh_hz` / `write_settle` / `rx_hz` | 段顶层 | 可选运行参数：状态新鲜度承诺 Hz / 指令帧生效等待 s / 接收线程节拍 Hz（缺省沿用默认并 warn） |
| `channel` / `protocol` / `baud_rate` | `arm`、`end` 段 | 总线通道（串口设备名）/ 总线类型标识 / 波特率；end 与 arm 同 channel 则共享单总线 |
| `default_mode` | `arm`、`end` 段 | 连接成功后自动设置的控制模式（`mit`/`position`/`velocity`；无 MIT 的型号配 `position`） |
| `name` / `motor_id` / `feedback_id` / `model` | `joints[]` | 关节名 / 电机总线地址 / 应答帧标识 / 电机型号（子类按型号查量程表） |
| `q_min` / `q_max` / `dq_max` / `tau_max` | `joints[]` | 硬限位四键——**数值须与 URDF limit 标定一致**（22 单元产物），发送越限由基类按此裁剪 |
| `MIT.kp` / `MIT.kd` | `joints[]` | MIT 指令未传 kp/kd 时的发送默认 |
| `POS_VEL.pos_kp` 等 / `vlim` / `flim` | `joints[]` | 位置/速度模式的电机寄存器闭环增益（设模式时子类直取写入）/ 指令限速 / 归一化力矩电流上限 0~1 |

无 end 执行器的型号：`end:` 段留空（关节数 0）即可，end 系列方法仍须定义（保留桩）以满足抽象方法强制。

### 2.4 第 4 步：实现 29 个抽象方法

Python 的 abc 机制要求 29 个 `@abstractmethod` 内核**全部被定义**才允许实例化；**必实现 14 个**是任何电机/舵机型号的可用底线，**按需实现 15 个**未实现时保留模板的 `NotImplementedError` 桩（= 能力缺失经基类转 warn 优雅降级）。逐方法的实现要点、参数与返回约定见 `backend_template.py` 各方法 docstring（第一~五部分）。

**必实现 14 个**：

| 分组 | 方法（× arm/end 两族） | 要点 |
|:---:|:---:|:---:|
| 连接（4） | `_connect_arm/_connect_end`、`_disconnect_arm/_disconnect_end` | 打开/关闭总线与协议层资源；同 channel 共享总线，后连方复用 |
| 使能（4） | `_enable_joint_arm/_enable_joint_end`、`_disable_joint_arm/_disable_joint_end` | 单电机上电闭环/退出闭环帧 |
| 状态（2） | `_read_joint_state_arm/_read_joint_state_end` | 返回可用量 dict（缺键=硬件不提供；`error` 归一化 0 失能/1 使能/≥2 故障） |
| 模式（4） | `_read_joint_mode_arm/_read_joint_mode_end`、`_set_joint_mode_arm/_set_joint_mode_end` | 协议无模式回读时 `_read_*` 用本地镜像（恒返 `None` 会使 `get_mode_*` 核实永远失败）；`_set_*` 切 POSITION/VELOCITY 须一并写 POS_VEL 增益寄存器 |

**按需实现 15 个**：

| 分组 | 方法（× arm/end 两族） | 何时需要 |
|:---:|:---:|:---:|
| 参数读写（4） | `_read_joint_param_arm/_end`、`_write_joint_param_arm/_end` | 协议支持寄存器/RAM 表读写 |
| 设零（2） | `_set_joint_zero_arm/_set_joint_zero_end` | 有专用设零帧或零位寄存器 |
| 发送（6） | `_send_joint_mit_arm/_end`、`_send_joint_position_arm/_end`、`_send_joint_vel_arm/_end` | 型号支持哪种模式实现哪种（舵机通常仅位置） |
| 离散动作（1） | `_send_action_end` | 仅离散动作末端（夹爪 open/close/home 等） |
| 清错（2） | `_clear_joint_error_arm/_clear_joint_error_end` | 固件支持错误清除命令 |

实现完成后对照模板文件头的**联调自检清单**逐项核对（空 cfg 可构造、连接后 `is_connected_*` 为 `True`、状态字段约定、模式乐观更新、越限裁剪、参数标量返回、断连重连可恢复）。

### 2.5 第 5 步：REGISTRY 注册

编辑 `joyarm_core/backend/__init__.py`：顶部导入子类，`REGISTRY` 加一行：

```python
from .backend_<型号> import Backend<型号>

REGISTRY = {
    "backend_rebot_arm_b601_dm": BackendRebotArmB601DM,
    "backend_<型号>": Backend<型号>,          # ← 新增行
}
```

### 2.6 第 6 步：yaml `backend.name` 填注册名

`joyarm_core/config/<型号>.yaml` 的 `backend.name` 填第 5 步的注册名（= 子类文件名去 `.py`），选型链路即闭环。

### 2.7 第 7 步：docstring 改型号介绍

把拷贝来的模板注释由「子类型号模板和实现约束」风格改为「子类型号的介绍说明」风格——文件头 docstring（接入步骤段落换成型号介绍）、类 docstring、各方法 docstring 三层，说明该型号的硬件组成与协议特点。

## 3. DM / Feetech 双家族差异总览

两类典型硬件的实现取舍（总览引自 `backend_template.py` 类 docstring，具体实施因型号而异）：

| 维度 | DM 关节电机族（达妙，如 rebot_arm_b601_dm） | Feetech 舵机族 |
|:---:|:---:|:---:|
| 总线/收发 | 应答式：接收线程 + 发送锁 + 请求-应答 Event 同步 | 半双工轮询：总线锁包住「发帧 + 等应答」整个事务 |
| 模式 | 有模式寄存器、可回读 | 通常无模式概念 → 本地镜像恒返 `POSITION`，yaml 配 `default_mode: position` |
| 发送面 | MIT / 位置 / 速度三模式全支持 | 按硬件能力实现（通常仅位置，部分有速度/力矩可映射 VELOCITY/MIT）；未实现保留桩 = 优雅降级 |
| 状态 | 反馈帧含 q/dq/tau/error（协议无温度则 `temp_*` 缺键） | 轮询读位置/负载/温度，硬件没有的量直接缺键 |
| 参数 | 寄存器 RID 表读写 | RAM 表读写，键名由子类定义 |

补充两点：**混合硬件属常态**——同一子类可 arm 用关节电机 + end 用舵机（两族 `channel` 不同即各开各的总线，相同才共享一条）；**无 end 的型号** end 段留空即可（见 2.3）。

## 4. 测试阶梯

| 阶梯 | 工具 | 环境 | 覆盖 | 通过判据 |
|:---:|:---:|:---:|:---:|:---:|
| 一、离线桩冒烟 | 本文件夹 `23_backend_check.py` | 无硬件 | 注册链路（可导入/继承/离线构造）+ 模板 29 桩齐全与消息格式 | 退出码 0 |
| 二、真机验收 | `joyarm_core/backend/backend_verify.py` | 真机 | 六分组：00 初始化 / 10 连接 / 20 只读 / 30 控制 / 40 运动发送 / 50 清错 | 全 PASS/SKIP 退出码 0 |

### 4.1 第一阶：离线桩冒烟（本文件夹）

```bash
# 在 joyarm_code/ 目录下（离线脚本，无需硬件）
python quick_start/23_backend_creator/23_backend_check.py --backend backend_<型号>
```

两级检查分工明确：**第一级**的对象是 `--backend` 指定的**你自己新建的注册名**（核对可从 `REGISTRY` 取到、对应类是 `Backend` 子类、空 cfg 与型号 yaml `backend:` 段均可离线构造）；**第二级**的对象始终是本文件夹的模板 `backend_template.py`（核对 29 个抽象方法桩齐全、逐桩调用按约定抛 `NotImplementedError`、核对表与 `Backend.__abstractmethods__` 集合完全对齐，与 `--backend` 参数无关）。你自己子类的方法完整性不靠第二级把关——由 abc 机制在实例化时强制（缺一个抽象方法即 `TypeError`，见 2.4），真机行为由 4.2 的 `backend_verify.py` 验收。任一 FAIL 退出码 1。

### 4.2 第二阶：真机验收（backend_verify.py 六分组）

```bash
# 在 joyarm_code/ 目录下，接好真机后按风险递进执行
python joyarm_core/backend/backend_verify.py --sub=backend_<型号> --step=offline   # 00 组初始化（缺省，低危）
python joyarm_core/backend/backend_verify.py --sub=backend_<型号> --step=10,20     # 连接 + 只读（低危）
python joyarm_core/backend/backend_verify.py --sub=backend_<型号> --step=all       # 全部（不含 38 设零）
```

安全机制：编号按风险递进；中危交互确认（`--yes` 跳过）；**高危（使能/写参数/运动发送/清错）强制交互输入 yes，`--yes` 不放行，非终端自动 SKIP**；38 设置零点为极高危，仅显式 `--step=38` 且二次确认才执行。退出码：全 PASS/SKIP = 0，任一 FAIL = 1，用法错误 = 2。

基类契约的完整回归测试另见 `joyarm_code/test/test_backend_base.py`（由工作流统一运行，日常接入不必手动执行）。
