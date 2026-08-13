# `joyarm_code/` —— JoyArm 机械臂教程配套代码库

> JoyArm 教程的代码工程根目录：核心 SDK 库 [`joyarm/`](joyarm/)（ROS2-free）、ROS2 兄弟包
> [`joyarm_ros2/`](joyarm_ros2/)、章节示例 [`chapt/`](chapt/) 与测试 [`test/`](test/)。
> 设计详则见 [`joyarm/架构设计.md`](joyarm/架构设计.md)，结构与 API 速查见 [`架构与API.md`](架构与API.md)。

## 目录结构

```
joyarm_code/
├── joyarm/        # ★ 核心 SDK（ROS2-free）：arms / robotics / safety / backends / utils / robots / configs
├── joyarm_ros2/   # ROS2 兄弟包（依赖 joyarm + rclpy）：arm_nodes / apps（Ch10/12/14 占位）
├── chapt/         # 章节教学示例脚本（一次性，不复用）
├── test/          # pytest 测试套件
├── quickstart/    # 快速上手（CLI/GUI/任务编排，占位）
└── pyproject.toml # 工程配置：可编辑安装 joyarm + joyarm_ros2
```

## 核心包 joyarm 速览

**设备三概念**：`Mas`（多轴本体）/ `End`（末端执行器）/ `Arm`（= Mas + End）。
**Backend 三层**：`Backend` → `BackendMas`/`BackendEnd` → 型号层（`BackendMasRebotDM`/`BackendEndJoyGripper`）。

```
arms/       设备模型：Mas · End · Arm(Mas+End) · joyarm_rebot_dm   ← arm.xx 门面
robotics/   算法：fkine · ikine · jacobian · trajectory · dyn · control
safety/     安全
backends/   通信（三层 Backend）
utils/      基础：transforms(数学) · types(类型) · interfaces(MasProtocol)
robots/     URDF + meshes     configs/  per-model YAML（backend_mas/backend_end）
```

> 导入方向（自底向上、无环）：`utils` → `robotics`/`safety`/`backends` → `arms`；
> `robotics`/`safety` 仅依赖 `MasProtocol`（不 import `arms`）。算法默认 pinocchio。

## 环境安装

```bash
cd joyarm_code
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e .            # 核心 joyarm + 兄弟包 joyarm_ros2（含 numpy/pin/pyyaml）
# uv pip install -e ".[ros2]"  # 额外装 ROS2 依赖（rclpy；通常由 ROS2 环境提供）
```

> 安装后 `import joyarm` 全局可用；`pytest` 无需手动设 `PYTHONPATH`（见 `pyproject.toml`）。核心包不依赖 rclpy。

## 快速开始（离线，无需真机）

```python
from joyarm import JoyArmRebotDM

arm = JoyArmRebotDM()              # 默认未连接（离线）；Arm = Mas + End，自动加载 configs + URDF
Q   = arm.rand_q(size=100_000)     # 软限位内采样 (N,6)
P   = arm.fkine(Q, rep="pos")      # 门面 arm.fkine → (N,3)
# arm.connect(); arm.end.open()    # 真机：先 connect() 再操作末端
```

> **离线语义**：`connected=False`（默认）时计算类（`fkine`/`rand_q`/…）可用；执行类（`get_state`/`command`/`end.open()`）`raise RuntimeError`，`connect()` 后可用。

## 运行测试 / 章节示例

```bash
cd joyarm_code && pytest                      # 运行测试套件
cd joyarm_code/chapt && python chapt2_T_demo.py        # 运行章节示例（PySide6 GUI）
```

## 章节落地状态

| 章节 | 内容 | 状态 |
|:---:|:---|:---:|
| Ch2 | `utils/*`、`arms/{mas,end,arm,joyarm_rebot_dm}`、`configs`、`backends` 抽象层、`robotics/fkine`、`__init__` | ✅ |
| Ch3+ | `robotics/{ikine,jacobian,trajectory,dyn,control}`、`safety`、`backends` 型号层、`joyarm_ros2/*` | 🟡 |

## 进阶文档

| 文档 | 说明 |
|:---:|:---|
| [`架构与API.md`](架构与API.md) | 开发者向：整体结构 → 各 py 文件 → 类与函数 → API 速查 |
| [`joyarm/架构设计.md`](joyarm/架构设计.md) | 核心包架构定稿：命名约定、设计原则、依赖与接口 |
| [`../spec/项目架构.md`](../spec/项目架构.md) | 仓库整体架构（文档站点 + 代码工程） |
