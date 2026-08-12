# `joyarm/` —— JoyArm 机械臂教程核心 SDK 库

> 本目录是 JoyArm 教程的**可复用核心库**（SDK）：被 `joyarm_code/chapt/` 教学脚本
> 与未来的 `joyarm_code/quickstart/` 上位机/CLI 直接 `import` 使用。完整架构设计见
> [`架构设计.md`](架构设计.md)。

## 设计原则（一句话）

**分层解耦、库即被调用、`Arm` 作组合根、Backend 仅真机、型号↔通信正交、
双版本（`method="auto"`/`"manual"`）、形状重载、共享类型只定义一次、
核心轻依赖、函数式为主、渐进式落地。**

## 分层架构

```
application/   上层应用：可视化 viz · 遥操作 teleop(Ch12) · 适配器 ros2(Ch10)/vision(Ch14)
─────────────────────────────────────────────
safety/        安全层：safety(Ch11) 限位 / 自碰撞 / 外部碰撞监控
─────────────────────────────────────────────
robotics/      算法层：control(Ch6/8/9) · trajectory(Ch5) · dyn(Ch8)
               · fkine · ikine(Ch3) · jacobian(Ch4)
─────────────────────────────────────────────
arm/           设备模型层：arm · joyarm_rebot_dm · gripper(Ch13)
─────────────────────────────────────────────
backends/      通信层：arm_backend · joyarm_rebot_dm_backend · end_backend · gripper_backend
─────────────────────────────────────────────
utils/         基础层：transforms(数学) · types(共享类型)
```

**依赖方向**：仅允许向下依赖。`utils/`（`types`/`transforms`）处最底层、零机器人学耦合，
可被 `joyarm_code/chapt/` 教学脚本单独复用。

## 模块导览

| 模块 | 章节 | 状态 | 职责 |
|:---:|:---:|:---:|:---|
| `utils/types.py` | Ch2.2 | ✅ | 共享数据类型（dataclass）+ 枚举 + `clamp_to_limits` |
| `utils/transforms.py` | Ch2.2 | ✅ | 纯 numpy SO(3)/SE(3)（零外部依赖） |
| `arm/arm.py` | Ch2.2 | ✅ | `Arm` 基类（组合根）+ 离线模式语义 |
| `arm/joyarm_rebot_dm.py` | Ch2.2 | ✅ | `JoyArmRebotDM(Arm)` 型号预设 |
| `arm/gripper.py` | Ch13 | 🟡 | 两指夹爪（设备模型，占位） |
| `robotics/fkine.py` | Ch2.2 | ✅ | 正运动学（形状重载 + `method`） |
| `robotics/ikine.py` | Ch3 | 🟡 | 逆运动学（占位） |
| `robotics/jacobian.py` | Ch4 | 🟡 | 速度运动学 + 静力学（占位） |
| `robotics/trajectory.py` | Ch5 | 🟡 | 关节/笛卡尔轨迹生成（占位） |
| `robotics/dyn.py` | Ch8 | 🟡 | 动力学（占位） |
| `robotics/control.py` | Ch6/8/9 | 🟡 | 运动学/动力学/力控（占位） |
| `safety/safety.py` | Ch11 | 🟡 | 状态监测与安全软防护（占位） |
| `application/viz.py` | Ch2.2 | ✅ | meshcat / matplotlib 可视化 |
| `application/teleop.py` | Ch12 | 🟡 | 示教记录/回放 + 遥操映射（占位） |
| `application/ros2.py` | Ch10 | 🟡 | ROS2 适配器（可选，占位） |
| `application/vision.py` | Ch14 | 🟡 | 视觉（可选，占位） |
| `backends/arm_backend.py` | Ch2.2 | ✅ | `ArmBackend` 机械臂通信抽象基类 |
| `backends/joyarm_rebot_dm_backend.py` | Ch6 | 🟡 | `JoyArmRebotDMBackend` 机械臂真机后端（占位） |
| `backends/end_backend.py` | Ch2.2 | ✅ | `EndBackend` 末端执行器通信抽象基类 |
| `backends/gripper_backend.py` | Ch13 | 🟡 | `GripperBackend` 夹爪真机后端（占位） |
| `robots/` | Ch2.2/Ch7 | — | 随包 URDF 资产（`reBot-DevArm_fixend/`） |

> ✅ = 完整实现；🟡 = 签名 + docstring + `raise NotImplementedError("ChN 实现")` 占位。
> 占位模块保证 `import joyarm` 永不抛 `ImportError`，下游脚本与 IDE 补全始终可用。

## 依赖安装

**Python 3.10**，推荐用 [uv](https://github.com/astral-sh/uv) 管理环境：

```bash
# 新建并激活环境
uv venv --python 3.10
source .venv/bin/activate

# 核心依赖（Ch2.2 即用）
uv pip install numpy pin meshcat matplotlib
#  - numpy:    数值计算
#  - pin:      pinocchio，URDF 加载 + 运动学/动力学（重依赖）
#  - meshcat:  浏览器 3D 可视化（viz.py）
#  - matplotlib: 无浏览器环境备用 3D 点云（viz.plot_points_mpl）
```

> `pinocchio` 也可用 conda 安装：`conda install -c conda-forge pinocchio`。
> 后续章节的可选适配器（ROS2 / 视觉 / VR / PySide6）按需安装，不进核心层。

## URDF（随包已提供）

`JoyArmRebotDM()` 缺省时通过 `importlib.resources` 自动解析随包 URDF：

```
robots/reBot-DevArm_fixend/urdf/reBot-DevArm_fixend.urdf
```

无需用户额外操作即可直接实例化：

```python
from joyarm import JoyArmRebotDM
arm = JoyArmRebotDM()   # 自动解析随包 reBot-DevArm_fixend.urdf
```

> 当前随包 URDF 为过渡型号 `reBot-DevArm_fixend`（纯 6 转动臂，`nq=6`）；
> 第七章导出正式版 `joyarm1.urdf` 后将替换 `DEFAULT_URDF`。
> 若显式传入不存在的 `urdf_path`，`Arm.__init__` 会抛出带定位信息的 `FileNotFoundError`。

## 最小示例（离线模式，无需真机）

```python
from joyarm import JoyArmRebotDM, fkine, viz

arm = JoyArmRebotDM()                            # backend=None 离线模式
Q  = arm.rand_q(size=100_000)                    # 软限位内采样 (N,6)
P  = fkine(arm, Q, rep="pos")                    # 批量 FK → (N,3)
v  = viz.make_viewer(); viz.plot_points(v, P)    # meshcat 工作空间点云

# 或无浏览器环境：
# ax = viz.plot_points_mpl(P)
```

**离线模式语义**：

- **计算类方法**（`fkine`/`frame_placement`/`rand_q`/`clamp_q`/`is_q_valid`/
  `jac`/`fdyn`/`idyn` 等）**不依赖 backend**，无硬件也能跑——教学、工作空间
  绘制、轨迹预演等场景的前提。
- **执行类方法**（`get_state`/`command`）在 `backend=None` 时
  `raise RuntimeError("离线模式不可执行")`。

## 工厂函数

```python
from joyarm import load_arm

arm = load_arm("joyarm_rebot_dm")                # 离线（默认）
arm = load_arm("joyarm_rebot_dm", "real")        # 真机（JoyArmRebotDMBackend，Ch6 实现）
```

型号（结构：URDF/限位/home）与 backend（通信）正交组合：
`load_arm(model=, backend=)`。

## 关键约定

1. **形状重载**：`fkine` 等按输入形状自动切换批量/非批量，无单独 `*_batch` 函数。
2. **`method` 双版本**：核心算法统一支持 `method="auto"`（pinocchio 黑盒）/
   `"manual"`（手写白盒，先占位）。
3. **角度单位**：内部一律**弧度**。
4. **返回值**：FK/IK 一律返回 numpy（非 `pin.SE3`），降低教学门槛。
5. **软/硬限位分级**：`Arm.joint_limits`（硬，URDF 极限）/ `Arm.joint_limits_soft`
   （软，内缩 5% 余量）；软限位违规 → `SafetyAction.CLAMP`，硬限位违规 →
   `Severity.ERROR/CRITICAL`。
6. **零状态污染**：模块级无全局可变状态；`Arm` 仅持有 `model`/`data`。

## 章节落地状态

- ✅ **Ch2.2 即用**（完整实现）：`utils/types`、`utils/transforms`、`arm/arm`、
  `arm/joyarm_rebot_dm`、`robotics/fkine`、`application/viz`、`backends/arm_backend`、
  `backends/end_backend`、`__init__`、`README`。
- 🟡 **Ch3+ 占位**：`robotics/ikine`(Ch3)、`robotics/jacobian`(Ch4)、`robotics/trajectory`(Ch5)、
  `robotics/dyn`(Ch8)、`robotics/control`(Ch6/8/9)、`safety/safety`(Ch11)、`arm/gripper`(Ch13)、
  `application/teleop`(Ch12)、`application/ros2`(Ch10)、`application/vision`(Ch14)、
  `backends/joyarm_rebot_dm_backend`(Ch6)、`backends/gripper_backend`(Ch13)。

接口已冻结，章节推进时按 [`架构设计.md` §十七](架构设计.md) 顺序补实现即可。
