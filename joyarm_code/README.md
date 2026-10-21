# `joyarm_code/` —— JoyArm 代码库

## 1. 目录结构

```
joyarm_code/
├── joyarm_core/     # ★ 核心 SDK ：joyarm / robotics / backend / utils / robot_model / config
├── joyarm_ros2_ws/  # ROS2 工作空间（规划）
├── chapt/           # 章节教学示例脚本（一次性演示）
├── test/            # 测试套件
├── quickstart/      # 快速上手（待补）
└── pyproject.toml   # 工程配置与依赖
```

## 2. 基本架构（待人工提炼）

核心包 `joyarm_core` 自底向上分层，导入单向、无环：

```
   joyarm_ros2_ws/src/joyarm_node 
              ↓ 依赖
   ┌──────────────────────────────────────────────────────────────┐
   │  joyarm/   组合根：JoyArm（主类） · joyArm（对象创建入口）        │
   ├──────────────────────────────────────────────────────────────┤
   │  robotics/  算法层（子包：ABC基类功能定义+子类具体功能实现+显式注册表）│
   │             ← 仅依赖 utils（按属性约定调用 arm，不反向依赖joyarm） │
   │  backend/  硬件后端通信（基类定义功能 + 子类实现具体型号功能和选型）  │
   ├──────────────────────────────────────────────────────────────┤
   │  utils/    types（共享类型） · transforms（数学） · limits（限位）│
   └──────────────────────────────────────────────────────────────┘
   robot_model/（URDF+meshes）  config/（型号配置 YAML） ← 资产，joyarm 运行期加载
```

核心概念：

- **组合根 `JoyArm`（单类，无型号子类）**：完整机械臂 = arm 本体 + end 执行器。**型号差异全部由配置表达**——新型号 = `config/<型号>.yaml` + `robot_model/` URDF 资产 + `backend/backend_*.py`（型号与整机后端 1:1）。持有一个**整机通信后端** `_backend`（私有，yaml `backend.name` 选型构建、必配）与**六域策略成员字典**（`_fkine_solvers` / `_ikine_solvers` / …，config 可指定加载多个、首个为激活；JoyArm 型号同时配有 Pin 数值 IK 和 2/3/4 轴平行、5/6 轴相交的解析 IK）；公开门面（`arm.solve_fkine()` / `arm.set_action_end("open")`…）委托激活成员，`connect_arm()` 后才可执行硬件操作。config 必需且构造时经类内自检（任一项失败抛异常、对象不构造）；构造时一次性深拷贝存为类内成员，数据经 `arm.get_config()` 读取、不重读 yaml。运动能力由**运动状态机 MotionFSM**（启动后自动经 home 进入正常运动态）与 plan/ctrl 两线程提供，另配安全监视器与运行状态可视化（网页监控）。
- **成员即策略（通过 config 配置、字典化加载）**：`config/<model>.yaml` 的 `robotics:` 段按各域注册名选型（全部加载+首个激活；子类类名 = 前缀 + 基类名（如 `PinFkineSolver`），注册名 = 类名小写+下划线（如`pin_fkine_solver`），在域 `REGISTRY` 显式加一行注册）；运行期通过 `arm.set_<域>_solver(名)` 切换（有且仅有一个被激活；traj/control 管线运行时切换即**热切换**——规划器先试规划成功才替换、控制器自动切换电机模式，即时生效）。**配置的注册名必须已在注册表且实例化成功，否则构造失败（硬失败）**。
- **对象创建入口 `joyArm`**：按型号名（唯一参数）创建机械臂——四步固定顺序校验（config 文件 / backend 注册名 / robot_model 资产路径 / 构造自检），**任一步失败抛携带具体原因的异常（不返回 None）**，错误消息附可用型号列表；型号唯一标识 = yaml 文件名，URDF 目录 `robot_model/<型号>/` 同名。`listArm()` 列出当前可用型号。
- **Backend 两层**：`Backend` 基类（方法以 `*_arm` / `*_end` 后缀区分本体与末端）→ `BackendDM` 等子类（与具体机械臂型号 **1:1** 派生），新型号接入 = 复制 `backend/backend_template.py`（接入见该文件头）；config yaml 文件 `backend:` 段 `name` 选型，arm/end 同 channel 则共享总线、异 channel 则独立。

## 3. 环境安装

**核心库（第 1~9 章，uv 工作流）**：

```bash
cd joyarm_code
curl -LsSf https://astral.sh/uv/install.sh | sh # 若未安装uv则执行，若已安装则可跳过 
uv sync                    # 创建 .venv + 按 uv.lock 安装全部依赖
source .venv/bin/activate  # 激活后直接用 python；或免激活用 uv run <命令>
```

**ROS2 部分（第十章起）**：
工作空间 `joyarm_ros2_ws/` 与环境步骤**规划中**，落地时补充。

## 4. 快速开始

```python
import time
from joyarm_core import joyArm, MotionState   # 创建入口 + 状态枚举（型号名 = config/joyarm_dm.yaml 文件名）

arm = joyArm("joyarm_dm")            # 四步校验任一失败 → 抛携带具体原因的异常（不返回 None）
arm.connect_arm()                    # 连接真机本体（末端 connect_end() 同构）
arm.enable_arm()                     # 使能本体电机（状态机进入各状态时也会自动检查使能）
arm.start_motion_fsm()               # 启动运动状态机（自动迁移 DISABLE→HOME→MOTION）
arm.wait_for_motion_state(MotionState.MOTION)  # 等待就绪再发定时目标（1=到达 / 0=超时；真机行程远可传大 timeout）
arm.set_target_joint(time.time() + 3.0, arm.home_arm)  # 关节目标：3 秒后到达 home 位形
arm.stop_motion_fsm()                # 停止状态机（阻塞安全停机：回零失能；更高危事件时直接失能）
arm.close()                          # 安全退出（失能 + 断开；亦可用 with 语句）
```

## 5. 运行教学章节示例

```bash
cd joyarm_code/chapt && python chapt2_T_demo.py         # 运行章节示例（PySide6 GUI）
```
