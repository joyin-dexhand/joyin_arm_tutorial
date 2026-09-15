# `joyarm_code/` —— JoyArm 代码库

## 1. 目录结构

```
joyarm_code/
├── joyarm_core/     # ★ 核心 SDK ：joyarm / robotics / backend / utils / robot_model / configs
├── joyarm_ros2_ws/  # ROS2 工作空间（Ch10 落地创建）
├── chapt/           # 章节教学示例脚本（一次性，不复用）
├── test/            # 测试脚本，随时删除
├── quickstart/      # 快速上手（使用示例）
└── pyproject.toml   # 工程配置与依赖
```

## 2. 基本架构

核心包 `joyarm_core` 自底向上分层，导入单向、无环：

```
   joyarm_ros2_ws/src/joyarm_node 
              ↓ 依赖
   ┌──────────────────────────────────────────────────────────────┐
   │  joyarm/   组合根：JoyArm（单类） · JoyArmFactory(按型号选型)    │
   ├──────────────────────────────────────────────────────────────┤
   │  robotics/  算法层（一域一子包：ABC + 内置默认实现 + 显式注册表； │
   │             实现为教程各章教学内容）← 仅依赖 utils（按属性约定     │
   │             调用 arm，不反向依赖 joyarm）                        │
   │  backend/  通信（整机 Backend + name 选型）                     │
   ├──────────────────────────────────────────────────────────────┤
   │  utils/  types（共享类型） · transforms（数学） · limits（限位）│
   └──────────────────────────────────────────────────────────────┘
      robot_model/（URDF + meshes）  configs/（per-model YAML）   ← 资产，由 joyarm 运行期加载
```

核心概念：

- **组合根 `JoyArm`（单类，无型号子类）**：完整机械臂 = arm 本体 + end 执行器。**型号差异全部由配置表达**——新型号 = `configs/<型号>.yaml` + `robot_model/` URDF 资产 + `backend/backend_*.py`（型号与整机后端 1:1）。持有一个**整机通信后端** `_backend`（私有，yaml `backend.name` 选型构建、必配）与**六域策略成员字典**（`_fkine_solvers` / `_ikine_solvers` / …，config 可指定加载多个、首个为激活）；公开门面（`arm.fkine()` / `arm.set_end_open()`…）全部委托激活成员，`connect()` 后才可执行硬件操作。config 必需且构造前经 `JoyArm.check_config` 静态自检（通过才初始化）；构造时一次性深拷贝存为类内成员，教学数据（如 MDH 参数）经 `arm.get_config()` 读取、不重读 yaml。
- **成员即策略（config 可换、字典化加载）**：`configs/<model>.yaml` 的 `robotics:` 段按各域注册名选型（值可为单个规格或列表，全部加载、首个激活；实现类名 = 算法前缀 + 域基类名（如 `PinFkineSolver`），注册名 = 类名小写+下划线（`pin_fkine_solver`），在域 `REGISTRY` 显式加一行注册，默认实现居首）；运行期 `arm.set_solver(域, 名)` 切换（有且仅一个激活；traj/control 管线运行中切换即**热切换**——同步刷新一拍、模式自动切换、即时生效）。各域内置默认实现（`pin_fkine_solver`/`pin_ikine_solver`/`pin_jacobian_solver`/`pin_dynamics_solver`/`to_joint_traj_planner`/`joint_position_controller`，5/6/7 轴通用）供 config 选配——域未配置即无成员（门面调用显性报错）；**配置的注册名必须已在注册表且实例化成功，否则构造失败（硬失败）**。
- **型号工厂 `joyarm_factory`**：按型号名（唯一参数）创建机械臂——自动加载 `configs/<型号>.yaml` 并校验命名链（入参 = 文件名 = `basic.name`）；**型号不存在或初始化失败时返回 `None` 并输出失败信息**（不抛异常；仅工厂入口软化，`JoyArm` 直用为硬失败）。
- **Backend 整机两层**：`Backend`（整机根，方法以 `*_arm` / `*_end` 后缀区分本体与末端）→ `BackendDM`（与机械臂型号 **1:1** 派生：`backend_dm` ↔ `joyarm_dm`）；yaml `backend:` 段 `name` 选型，arm/end 同 channel 共享总线、异 channel 独立。

## 3. 环境安装

**核心库（第 1~9 章，uv 工作流）**：

```bash
cd joyarm_code
# 若未安装uv，则先安装uv: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                    # 创建 .venv + 按 uv.lock 安装全部依赖 + 可编辑安装 joyarm_core
source .venv/bin/activate  # 激活后直接用 python；或免激活用 uv run <命令>
```

**ROS2 部分（第十章起）**：
工作空间 `joyarm_ros2_ws/` 与环境步骤**规划中**，落地时补充。

> 安装后 `import joyarm_core` 全局可用；核心包不依赖 rclpy。

## 4. 快速开始

```python
from joyarm_core import joyarm_factory, JoyArm

arm = joyarm_factory("joyarm_dm")  # 推荐：型号名唯一参数；默认未连接（离线），
                                   # 自动加载 configs/joyarm_dm.yaml + URDF（config 必需、
                                   # 自检通过才初始化）；型号不存在时返回 None 并输出失败信息
Q = arm.rand_q_arm(size=100_000)   # 本体硬限位内采样 (N, 6)
JoyArm.check_config("joyarm_dm", arm.get_config())   # 配置静态自检：正常静默通过，异常 ValueError
mdh = arm.get_config()["joyarm"]["arm_mdh_and_limits"]   # 教学数据从类内 config 读取
# arm.fkine(Q, arm.ee_frame_name) / arm.ikine(...)  # 求解器门面：六默认实现已内置注册，
                                                   # config robotics: 段取消注释选配即离线可用
# arm.check_hardware()   # 硬件自检：自包含（临时连接→失能状态检查→断开），手动调用
# arm.connect(); arm.enable_arm()
# 等价直用：from joyarm_core import JoyArm; arm = JoyArm("joyarm_dm")
```

> **离线语义**：`connected=False`（默认）时计算类（`rand_q_arm`/`clamp_to_limits`/…）可用；执行类（`get_arm_state`/`set_arm_command`/`set_end_open()`）`raise RuntimeError`，`connect()` 后可用（支持 `with` 上下文自动连接与安全收尾）。

## 5. 运行章节示例

```bash
cd joyarm_code/chapt && python chapt2_T_demo.py         # 运行章节示例（PySide6 GUI）
```
