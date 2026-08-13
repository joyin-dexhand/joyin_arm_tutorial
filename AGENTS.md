# 项目说明（ZCode 项目记忆）

本仓库为 JoyArm 机械臂教程文档站点，使用 MkDocs + Material 主题构建。

## 项目架构与维护文档（spec/）

> 💡 **接手任何任务前，先读 [`spec/项目架构.md`](spec/项目架构.md)** —— 它用最短篇幅讲清仓库目录结构、四篇十五章组织、命名/编号约定、构建部署链路与常见陷阱，是快速建立全局认知的入口。

`spec/` 为面向维护者与 Agent 的内部文档区（**不进站点**，且在 `.gitignore` 中，新增文件需 `git add -f` 才会被跟踪）：

| 文档 | 作用 |
|:---:|:---|
| [项目架构.md](spec/项目架构.md) | 仓库架构速览（目录、编号、构建、Agent 上手清单）—— **首选入口** |
| [spec_readme.md](spec/spec_readme.md) | MkDocs 环境安装/预览/构建/部署命令速查 |

## `docs/chapter*.md`写作规范

### 表格

- **所有表格的单元格默认居中**（水平 + 垂直）
- Markdown 表格统一用 `:---:` 对齐符（每列都写）
- 新建/修改表格时无需再向用户确认是否居中，默认居中

### 标题编号

- H1（章标题）：使用篇不加序号，其余篇格式为“第N章”，N为汉字，如:`第一章`
- H2：阿拉伯数字 `1, 2, 3`（每章内从 1 重新编号） 
- H3：`父H2.序号`，如 `2.1`
- H4+：依次类推 `2.1.1`
- `mkdocs.yml` 侧边栏的章序号用中文数字（第一章、第二章…），使用篇无章号

### 文档风格

- 文档保持简洁、清晰、无歧义，避免废话和讲解不清晰
- 结构化有条理，不允许思维过于跳跃，要对新手友好
- 每章内容（分别对应于一个独立的md文件）不设置本章小结，采用概要去总结全章内容（类似学术论文摘要），摘要不设置标题等级，直接紧随在章标题后面（第二行）

### 教程各章对应的脚本文件

- 各章对应的脚本文件存放于`./joyarm_code/chapt/<章序_功能>.py`，如`chapt2_T_demo.py`
- 各章脚本文件前几行必须添加使用说明的段落注释，包括：脚本功能概要以及新建环境、激活环境、安装py和依赖库、切换目录、脚本运行等必要说明
- 各章脚本文件仅作为讲解时的示例，可复用的核心功能和库均位于`./joyarm_code/joyarm/`，如：fkine、dyn、Arm 类、JoyArm类等。各章的脚本将调用这些功能

### 机械臂代码库

位置：`./joyarm_code/joyarm`（核心 SDK，ROS2-free）+ `./joyarm_code/joyarm_ros2`（ROS2 兄弟包）。

#### 命名约定（类 / 文件 / backend）—— 全仓库强制遵循

- **类名 = 驼峰（PascalCase）；文件名 = 小写 snake_case。**
- **设备三概念**（`arms/`）：
  - `Mas` = 多轴系统（Multi-Axis System，**不含**末端执行器的本体）。文件 `arms/mas.py`。
  - `End` = 末端执行器（夹爪/灵巧手/吸附…）。文件 `arms/end.py`。
  - `Arm` = 完整机械臂（= `Mas` + `End`）。`Arm(Mas)` 组合一个 `End`；文件 `arms/arm.py`。
  - 具体型号：`JoyArmRebotDM(Arm)` 等（"Arm" 表示完整臂，名仍对）。
- **Backend 三层**（`Backend` 前缀，`backends/`）：
  - 根：`Backend`（`backend.py`）—— 通用硬件通信抽象（`connect/disconnect/read_state`）。
  - 类型层（按硬件类型派生）：`BackendMas` / `BackendEnd` …。
  - 型号层（按具体型号派生）：`BackendMasRebotDM` / `BackendEndJoyGripper` …。
  - 文件名：`backend.py` / `backend_mas.py` / `backend_end.py` / `backend_mas_rebot_dm.py` / `backend_end_joygripper.py`。
- **属性 / 形参**：本体后端 = `backend_mas`（`Mas` 持有）；末端后端 = `backend_end`（`End` 持有）；robotics/safety 算法形参用 `mas`（依赖 `utils.MasProtocol`，**不 import** `arms`）。
- **yaml**：`configs/<model>.yaml` 按 backend 分段——`backend_mas:` / `backend_end:`（不再用单一 `comm:`）。子类绑 backend **类**（`backend_mas_cls` / `backend_end_cls`），基类按 config 段实例化。
- **算法默认 pinocchio**：robotics 不带 `method` 参数，默认走 urdf+pin；手写实现请在 `Mas` 子类覆盖对应方法。
- 约定针对**系统组件**（mas/end/arm/backend）；项目品牌名 `joyarm`/`joyarm_code` 不改。
- 详见 `joyarm_code/joyarm/架构设计.md`。
