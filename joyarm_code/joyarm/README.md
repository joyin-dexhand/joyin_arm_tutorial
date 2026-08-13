# `joyarm/` —— JoyArm 机械臂教程核心 SDK 库（ROS2-free）

> 核心库，被 `joyarm_code/chapt/` 与上位机 `import` 使用。ROS2 封装在兄弟包
> [`joyarm_ros2/`](../joyarm_ros2/)。完整架构见 [`架构设计.md`](架构设计.md)。

## 命名约定（类驼峰 / 文件小写）
- **设备**：`Mas`（多轴本体，不含末端）/ `End`（末端执行器）/ `Arm`（= Mas + End）。
- **Backend 三层**：`Backend` → `BackendMas`/`BackendEnd` → `BackendMasRebotDM`/`BackendEndJoyGripper`。

## 分层架构
```
arms/         设备模型：Mas · End · Arm(Mas+End) · joyarm_rebot_dm   ← arm.xx
─────────────────────────────────────────────
robotics/     算法：fkine · ikine(Ch3) · jacobian(Ch4) · trajectory(Ch5) · dyn(Ch8) · control(Ch6/8/9)
safety/       安全：safety(Ch11)
backends/     通信（三层）：Backend → BackendMas/BackendEnd → 型号层
─────────────────────────────────────────────
utils/        基础：transforms · types · interfaces(MasProtocol)
robots/       URDF + meshes    configs/  per-model YAML（backend_mas/backend_end）
```
> 导入方向（自底向上）：`utils` → `robotics`/`safety`/`backends` → `arms`；`robotics`/`safety`
> 仅依赖 `MasProtocol`（**不 import `arms`**，无环）。ROS2 在兄弟包 `joyarm_ros2/`。

## 模块导览
| 文件（小写） | 类（驼峰） | 章节 | 状态 | 职责 |
|:---:|:---:|:---:|:---:|:---|
| `utils/types.py` | — | Ch2 | ✅ | 共享数据类型 + 枚举 + `clamp_to_limits` |
| `utils/transforms.py` | — | Ch2 | ✅ | 纯 numpy SO(3)/SE(3) |
| `utils/interfaces.py` | `MasProtocol` | Ch2 | ✅ | robotics/safety 接口契约 |
| `configs/joyarm_rebot_dm.yaml` | — | Ch2/7 | ✅ | 型号 YAML（限位/MDH/home/`backend_mas`/`backend_end`） |
| `arms/mas.py` | `Mas` | Ch2 | ✅ | 多轴本体（model/限位/`backend_mas`/运动学门面） |
| `arms/end.py` | `End` | Ch13 | 🟡 | 末端执行器（`backend_end`） |
| `arms/arm.py` | `Arm(Mas)` | Ch2 | ✅ | 完整臂 = Mas + End |
| `arms/joyarm_rebot_dm.py` | `JoyArmRebotDM(Arm)` | Ch2 | ✅ | 型号预设（绑两个 backend 类） |
| `robotics/fkine.py` | `fkine` | Ch2 | ✅ | 正运动学（形状重载，默认 pin） |
| `robotics/{ikine,jacobian,trajectory,dyn,control}.py` | … | Ch3-9 | 🟡 | 占位（无 `method` 参数） |
| `safety/safety.py` | … | Ch11 | 🟡 | 占位 |
| `backends/backend.py` | `Backend` | Ch2 | ✅ | 通用硬件通信根 |
| `backends/backend_mas.py` | `BackendMas` | Ch2 | ✅ | 本体通信抽象 |
| `backends/backend_end.py` | `BackendEnd` | Ch2 | ✅ | 末端通信抽象 |
| `backends/backend_mas_rebot_dm.py` | `BackendMasRebotDM` | Ch6 | 🟡 | reBot-DM 本体后端（占位） |
| `backends/backend_end_joygripper.py` | `BackendEndJoyGripper` | Ch13 | 🟡 | Joy 夹爪后端（占位） |
| `../joyarm_ros2/{arm_nodes,apps}/` | — | Ch10/12/14/15 | 🟡 | ROS2 节点 + rviz2 |

## 依赖安装
```bash
cd joyarm_code
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e .            # 核心 joyarm + 兄弟包 joyarm_ros2（含 numpy/pin/pyyaml）
# joyarm_ros2（Ch10+）：需 ROS2 环境（rclpy），不影响核心包
```
> 安装后 `import joyarm` 全局可用、`pytest` 无需 `PYTHONPATH`（见 `pyproject.toml`）。核心不依赖 meshcat/rclpy。

## 最小示例（离线，无需真机）
```python
from joyarm import JoyArmRebotDM

arm = JoyArmRebotDM()              # 默认未连接（离线）；Arm = Mas + End，自动加载 configs + URDF
Q   = arm.rand_q(size=100_000)     # 软限位内采样 (N,6)
P   = arm.fkine(Q, rep="pos")      # 门面 arm.fkine → (N,3)
# arm.end.open()                   # 末端（夹爪）；需先 arm.connect()
```
**离线语义**：`connected=False`（默认）时计算类（fkine/jac/...）可用；执行类（`get_state`/`command`/`end.open()`）`raise RuntimeError`，`connect()` 后可用。

## 关键约定
1. **设备三概念**：`Mas`（本体）/ `End`（末端）/ `Arm`（Mas+End）。
2. **Backend 三层**：根→类型→型号；本体后端 `backend_mas`、末端后端 `backend_end`。
3. **backend 由 yaml 驱动**：子类绑 backend 类，基类按 `backend_mas`/`backend_end` 段实例化。
4. **算法默认 pinocchio**：robotics 无 `method` 参数；手写请在 `Mas` 子类覆盖。
5. **形状重载 / 弧度 / FK 返回 numpy / 软硬限位分级 / ControlMode 四态**。

## 章节落地
- ✅ **Ch2 即用**：`utils/*`、`arms/{mas,end,arm,joyarm_rebot_dm}`、`configs`、`backends` 三层抽象、`robotics/fkine`、`__init__`。
- 🟡 **Ch3+ 占位**：`robotics/{ikine,jacobian,trajectory,dyn,control}`、`safety`、`backends` 型号层、`joyarm_ros2/*`。
