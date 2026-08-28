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
   │  joyarms/   组合根：JoyArm · joyarm_dm · JoyArmFactory(按型号选型) │
   ├──────────────────────────────────────────────────────────────┤
   │  robotics/  算法（一域一子包：ABC + 实现 + REGISTRY）              │
   │             · fkine / ikine / jacobian / trajectory            │
   │             · dynamics / control       ← 仅依赖 utils（鸭子类型）│
   │  backends/  通信（整机 Backend + name 选型）                     │
   ├──────────────────────────────────────────────────────────────┤
   │  utils/  types（共享类型） · transforms（数学） · limits（限位守卫）│
   └──────────────────────────────────────────────────────────────┘
      robots/（URDF + meshes）  configs/（per-model YAML）   ← 资产，由 joyarms 运行期加载
```

三个核心概念：

- **组合根 `JoyArm`**：完整机械臂 = arm 本体 + end 执行器。持有一个**整机通信后端** `_backend`（私有，本体+末端一体，yaml `backend.name` 选型构建，能力全部经公有门面暴露）与六个**策略成员**（`_fkine_solver` / `_ikine_solver` / `_jacobian_solver` / `_dynamics_solver` / `_traj_planner` / `_controller`）；公开门面（`arm.fkine()` / `arm.plan_joint_p2p()` / `arm.play_joint()` / `arm.end_open()`…）全部委托私有成员，`connect()` 后才可执行硬件操作。
- **成员即策略（config 可换）**：`configs/<model>.yaml` 的 `robotics:` 段按注册名选型（`fkine: pin|mdh`、`traj: default|toppra`…）。
- **型号工厂 `joyarm_factory`**：按型号名（唯一参数）创建机械臂——自动加载 `configs/<型号>.yaml` 并校验命名链（入参 = 文件名 = yaml `basic.name` 字段 = joyarms 注册名，如 `joyarm_dm`），不一致报『XX』型号在XX中未找到。
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

arm = joyarm_factory("joyarm_dm")  # 推荐：型号名唯一参数（命名链校验）；默认未连接（离线），
                                   # 自动加载 configs/joyarm_dm.yaml + URDF，按 robotics: 组装成员
Q   = arm.rand_q(size=100_000)     # 软限位内采样 (N,6)
T   = arm.fkine(Q)                 # 门面 → _fkine_solver.solve → (N,4,4)（rep: quat/T/se3）
traj = arm.plan_joint_p2p(arm.q_neutral, arm.q_home)   # 轨迹规划门面（Ch5 实现）
# arm.connect(); arm.end_open()    # 真机：先 connect() 再操作末端
# 等价直用：from joyarm_core import JoyArmDM; arm = JoyArmDM()
```

> **离线语义**：`connected=False`（默认）时计算类（`fkine`/`rand_q`/…）可用；执行类（`get_arm_state`/`set_arm_command`/`end_open()`）`raise RuntimeError`，`connect()` 后可用。

## 5. 运行章节示例

```bash
cd joyarm_code/chapt && python chapt2_T_demo.py         # 运行章节示例（PySide6 GUI）
```
