# `joyarm_code/` —— JoyArm 代码库

## 1. 目录结构

```
joyarm_code/
├── joyarm_core/     # ★ 核心 SDK ：joyarms / robotics / backends / utils / robots / configs
├── joyarm_ros2_ws/  # ROS2 工作空间（Ch10 落地创建）
├── chapt/           # 章节教学示例脚本（一次性，不复用）
├── test/            # 测试套件
├── quickstart/      # 快速上手（使用示例/任务编排）
└── pyproject.toml   # 工程配置
```

## 2. 基本架构

核心包 `joyarm_core` 自底向上分层，导入单向、无环：

```
   joyarm_ros2_ws/src/joyarm_node 
              ↓ 依赖
   ┌──────────────────────────────────────────────────────────────┐
   │  joyarms/   组合根：JoyArm（单类） · JoyArmFactory(按型号选型)    │
   ├──────────────────────────────────────────────────────────────┤
   │  robotics/  算法接口（一域一子包：ABC + 空 REGISTRY，            │
   │             实现为教程各章教学内容）← 仅依赖 utils（鸭子类型）     │
   │  backends/  通信（整机 Backend + name 选型）                     │
   ├──────────────────────────────────────────────────────────────┤
   │  utils/  types（共享类型） · transforms（数学） · limits（限位守卫）│
   └──────────────────────────────────────────────────────────────┘
      robots/（URDF + meshes）  configs/（per-model YAML）   ← 资产，由 joyarms 运行期加载
```

核心概念：

- **组合根 `JoyArm`（单类，无型号子类）**：完整机械臂 = arm 本体 + end 执行器。**型号差异全部由配置表达**——新型号 = `configs/<型号>.yaml` + `robots/` URDF 资产 + `backends/backend_*.py`（型号与整机后端 1:1）。持有一个**整机通信后端** `_backend`（私有，yaml `backend.name` 选型构建）与**六域策略成员字典**（`_fkine_solvers` / `_ikine_solvers` / …，config 可指定加载多个、首个为活动）；公开门面（`arm.fkine()` / `arm.end_open()`…）全部委托活动成员，`connect()` 后才可执行硬件操作。config 在构造时一次性加载存为类内成员，教学数据（如 MDH 参数）经 `arm.get_config()` 读取、不重读 yaml。
- **成员即策略（config 可换、字典化加载）**：`configs/<model>.yaml` 的 `robotics:` 段按各域注册名选型（值可为单个规格或列表，全部加载）；运行期 `arm.set_solver(域, 名)` 切换。各域 `REGISTRY` 默认仅有 ABC 接口（空表）——具体算法为教程各章教学内容（fkine Ch2 / ikine Ch3 / jacobian Ch4 / traj Ch5 / control Ch6 / dynamics Ch8），章节实现注册后经 config 选型接入。
- **型号工厂 `joyarm_factory`**：按型号名（唯一参数）创建机械臂——自动加载 `configs/<型号>.yaml` 并校验命名链（入参 = 文件名 = `basic.name`）；**型号不存在或初始化失败时返回 `None` 并输出失败信息**（不抛异常）。
- **Backend 整机两层**：`Backend`（整机根，方法以 `*_arm` / `*_end` 后缀区分本体与末端）→ `BackendDM`（与机械臂型号 **1:1** 派生：`backend_dm` ↔ `joyarm_dm`）；yaml `backend:` 段 `name` 选型，arm/end 同 channel 共享总线、异 channel 独立。

## 3. 环境安装

**核心库（第 1~9 章，venv 工作流）**：

```bash
cd joyarm_code
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e .            # 核心 joyarm_core（含 numpy/pin/pyyaml）
```

**ROS2 部分（第十章起）**：
工作空间 `joyarm_ros2_ws/` 与环境步骤**规划中**，落地时补充。

> 安装后 `import joyarm_core` 全局可用；核心包不依赖 rclpy。

## 4. 快速开始

```python
from joyarm_core import joyarm_factory

arm = joyarm_factory("joyarm_dm")  # 推荐：型号名唯一参数；默认未连接（离线），
                                   # 自动加载 configs/joyarm_dm.yaml + URDF，按 robotics: 组装成员
                                   # 型号不存在时返回 None 并输出失败信息
Q = arm.rand_q(size=100_000)       # 软限位内采样 (N,6)
arm.check_config()                 # 配置自检 → list[Violation]（空 = 通过）
mdh = arm.get_config()["joyarm"]["arm_mdh_and_limits"]   # 教学数据从类内 config 读取
# arm.fkine(Q) / arm.ikine(...)    # 求解器门面：各章实现算法并注册后即可用（当前注册表为空）
# arm.connect(); arm.check_hardware(); arm.enable_arm()
# 等价直用：from joyarm_core import JoyArm; arm = JoyArm("joyarm_dm")
```

> **离线语义**：`connected=False`（默认）时已注册域的计算类（`rand_q`/`clamp_q`/…）可用；执行类（`get_arm_state`/`set_arm_command`/`end_open()`）`raise RuntimeError`，`connect()` 后可用。

## 5. 运行章节示例

```bash
cd joyarm_code/chapt && python chapt2_T_demo.py         # 运行章节示例（PySide6 GUI）
```
