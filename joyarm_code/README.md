# `joyarm_code/` —— JoyArm 机械臂教程配套代码库

> JoyArm 教程的代码工程根目录：核心 SDK 库 [`joyarm_core/`](joyarm_core/)（ROS2-free）、章节示例
> [`chapt/`](chapt/) 与测试 [`test/`](test/)；ROS2 工作空间 `joyarm_ros2_ws/` 规划中（第十章落地）。
> 教程正文（原理讲解）见[线上教程站点](https://joyin-dexhand.github.io/joyin_arm_tutorial/)；
> 面向开发者 / Agent 的维护文档见 [`AGENTS.md`](AGENTS.md)。

## 1. 目录结构

```
joyarm_code/
├── joyarm_core/          # ★ 核心 SDK（ROS2-free）：joyarms / robotics / safe_monitors / backends / utils / robots / configs
├── joyarm_ros2_ws/  # ROS2 colcon 工作空间（规划，Ch10 落地时创建；见 AGENTS.md §1.4）
├── chapt/           # 章节教学示例脚本（一次性，不复用）
├── test/            # pytest 测试套件
├── quickstart/      # 快速上手（CLI/GUI/任务编排，占位）
└── pyproject.toml   # 工程配置：可编辑安装 joyarm_core（ws 功能包由 colcon 另行构建）
```

## 2. 基本架构

核心包 `joyarm_core` 自底向上分层，导入单向、无环：

```
   joyarm_ros2_ws/src/joyarm_node  （ROS2 功能包：joyarm_core(pip) + rclpy(ROS)；规划中）
              ↓ 依赖
   ┌──────────────────────────────────────────────┐
   │  joyarms/      设备模型：JoyArm（本体+末端） · joyarm_dm  ← arm.xx 门面
   ├──────────────────────────────────────────────┤
   │  robotics/  算法    safe_monitors/  安全    backends/  通信    │ ← 仅依赖 utils + ArmProtocol
   ├──────────────────────────────────────────────┤
   │  utils/  types（共享类型） · transforms（数学） · interfaces（ArmProtocol） │
   └──────────────────────────────────────────────┘
      robots/（URDF + meshes）  configs/（per-model YAML）   ← 资产，由 joyarms 运行期加载
```

三个核心概念：

- **设备模型 `JoyArm`**：完整机械臂 = 多轴本体 + 末端执行器，直接持有两个通信后端 `backend_arm` / `backend_end`；`arm.fkine()` 等为运动学**门面**，`arm.end_open()` 操作末端，`connect()` 后才可执行硬件操作。
- **Backend 三层**：`Backend`（通用根）→ `BackendArm` / `BackendEnd`（按硬件类型）→ `BackendArmDM` / `BackendEndGripper`（按具体型号）。
- **算法层解耦**：`robotics` / `safe_monitors` 通过最小接口 `ArmProtocol` 访问臂对象，不依赖具体实现（依赖倒置）；算法默认基于 pinocchio（URDF 驱动）。

## 3. 环境安装

**核心库（第 1~9 章，venv 工作流）**：

```bash
cd joyarm_code
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e .            # 核心 joyarm_core（含 numpy/pin/pyyaml）
```

**ROS2 部分（第十章起）**：工作空间 `joyarm_ros2_ws/` 与环境步骤**规划中**，落地时补充（规划见 [`AGENTS.md`](AGENTS.md) §1.4）。

> 安装后 `import joyarm_core` 全局可用；`pytest` 无需手动设 `PYTHONPATH`（由 `test/conftest.py` 注入 `sys.path`）。核心包不依赖 rclpy。

## 4. 快速开始（离线，无需真机）

```python
from joyarm_core import JoyArmDM

arm = JoyArmDM()              # 默认未连接（离线）；JoyArm = 本体 + 末端，自动加载 configs + URDF
Q   = arm.rand_q(size=100_000)     # 软限位内采样 (N,6)
P   = arm.fkine(Q, rep="pos")      # 门面 arm.fkine → (N,3)
# arm.connect(); arm.end_open()    # 真机：先 connect() 再操作末端
```

> **离线语义**：`connected=False`（默认）时计算类（`fkine`/`rand_q`/…）可用；执行类（`get_arm_state`/`set_arm_command`/`end_open()`）`raise RuntimeError`，`connect()` 后可用。

## 5. 运行测试 / 章节示例

```bash
cd joyarm_code && pytest                                # 核心库测试套件
cd joyarm_code/chapt && python chapt2_T_demo.py         # 运行章节示例（PySide6 GUI）
```

