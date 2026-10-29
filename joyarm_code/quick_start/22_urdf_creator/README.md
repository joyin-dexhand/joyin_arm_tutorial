# 22 URDF 模型资产创建（urdf_creator）

本文件夹是新型号机械臂接入 joyarm_core 的第 22 单元：负责 `robot_model/` **模型资产**（URDF + 网格 + MuJoCo 模型）的创建、完善与测试。型号接入共四个单元依序协作：21 配置文件（`config/<型号>.yaml`）→ **22 模型资产（本单元）** → 23 硬件后端（`backend/backend_<型号>.py`，见 [23_backend_creator](../23_backend_creator/)）→ 24 整机创建与联调（见 [24_arm_creator](../24_arm_creator/)）；本单元产出合格后，由 24 单元统一装配。

本文件夹内容一览：

| 文件 | 说明 |
|:---:|:---:|
| `README.md` | 本文档：目录约定、URDF 规范要点、创建/完善/测试三步流程 |
| `机械臂URDF格式规范.md` | URDF 详细格式规范（命名总则、关节轴与限位、检查清单） |
| `22_urdf_check.py` | 离线检查脚本：七项检查逐项 PASS/FAIL，全部通过退出码 0 |

## 1 目录约定

模型资产统一存放在 `joyarm_core/robot_model/<型号>/` 下，目录名 = 型号名 = URDF 文件名（去扩展名）。JoyArm 创建对象时按这一固定命名寻址（`JoyArmFactory` 第 ③ 步按序校验 `robot_model/<型号>/`、`meshes/`、`urdf/`、`urdf/<型号>.urdf` 四路径存在，见 `joyarm_core/joyarm/__init__.py`），命名不符将无法接入：

```text
joyarm_core/robot_model/<型号>/
├── urdf/
│   └── <型号>.urdf     # URDF 模型文件（pinocchio 加载入口）
├── meshes/
│   └── *.STL           # 外观/碰撞网格（文件名与所属 link 同名）
└── mjcf/               # MuJoCo 仿真模型（可选）
    └── <型号>.xml
```

| 路径 | 内容 | 必备性 |
|:---:|:---:|:---:|
| `robot_model/<型号>/` | 型号根目录（目录名 = 型号名） | 必备 |
| `urdf/<型号>.urdf` | URDF 模型文件（文件名 = 型号名） | 必备 |
| `meshes/*.STL` | 连杆网格，文件名与所属 link 同名（`link2.STL` ↔ `link2`） | 必备 |
| `mjcf/<型号>.xml` | MuJoCo 仿真模型，由 URDF 转换，mesh 同样以 `../meshes/*.STL` 相对引用 | 可选 |

URDF 内 mesh 采用**相对路径**引用（如 `../meshes/link0.STL`），因此 `urdf/` 与 `meshes/` 必须整体存放、一并迁移，不可拆开。

## 2 URDF 规范要点摘要

下表是接入 joyarm_core 的**硬性要点**（检查脚本 `22_urdf_check.py` 逐项把关的就是这些）：

| 要点 | 要求 |
|:---:|:---:|
| 纯 URDF | 不使用 `xacro`，单文件自包含 |
| 型号标识 | `<robot name>` = 型号名 = `robot_model/<型号>/` 目录名 |
| mesh 引用 | 一律相对路径（`../meshes/*.STL`），且文件真实存在 |
| 单位 | 一律 SI：长度 m、质量 kg、角度 rad、时间 s |
| 四限位 | 每个主动关节 `lower / upper / effort / velocity` 四键完整；数值还须与 config 硬限位逐键一致（容差 1e-4，见规范 md 检查清单第 8 项） |
| mimic 从动 | 从动关节必须带 `<mimic joint=… multiplier=… offset=…/>` |
| 末端帧 | 提供 `frame_end_<位置>` 工具参考系（如 `frame_end_tcp`），空 link + fixed joint |

link / joint / frame 命名总则：小写 `snake_case`，编号从 0 起，同编号 link、joint 成对出现；`link*` 为有质量/几何的实体连杆，`frame_*` 为无质量纯坐标系：

| 类别 | 命名 | 说明 |
|:---:|:---:|:---:|
| 基座连杆 | `link0` | 固定基座（含电机 1 定子部分） |
| arm 连杆 | `link1`~`linkn` | 各级连杆（与驱动关节同号） |
| arm 关节 | `joint1`~`jointn` | 连接同编号 link 与上一编号 link，编号连续无下划线 |
| end 基座 | `link_end0` + `joint_end0`（fixed） | 固连至 arm 末连杆 `linkn` |
| end 主动关节 | `joint_end<N>` | 末端执行器驱动关节（如夹爪旋转轴 `joint_end1`） |
| end 从动 | `link_end_<名称>` + `joint_end_<名称>` | 带 `<mimic>` 从动（如手指 `link_end_fingerleft`） |
| end 参考系 | `frame_end_<位置>` + `joint_frame_end_<位置>` | 空 link（无惯量无几何），fixed 到 `link_end0`（如 `frame_end_tcp`） |

**详细规则**见同文件夹 [机械臂URDF格式规范.md](./机械臂URDF格式规范.md)（含关节轴与电机轴同向/反向重合约定、九项检查清单）；**导出操作**（SolidWorks + sw_urdf_exporter 插件逐步流程）见教程第七章《构型结构设计与 URDF 导出》，位于仓库根 `docs/chapt7_urdf.md`（自 `joyarm_code/` 起算即 `../docs/chapt7_urdf.md`）。

## 3 创建 / 完善 / 测试三步流程

新型号模型资产按「先建骨架 → 再对照规范逐项打磨 → 最后跑检查脚本」三步完成；每步产出与判定标准如下。

### 3.1 创建：建立目录骨架并导入 URDF 与网格

1. 按第 1 节约定新建 `joyarm_core/robot_model/<型号>/{urdf/, meshes/}` 目录骨架（`mjcf/` 为可选项，需要 MuJoCo 仿真时再建，见 3.2 末尾说明）；
2. 推荐在 SolidWorks 中由装配体经 `sw_urdf_exporter` 插件导出 URDF 与 STL（逐步操作见第七章第 5 节），URDF 存为 `urdf/<型号>.urdf`，各连杆 STL 存入 `meshes/`。没有 SolidWorks 时的替代路径：URDF 是纯文本 XML，可按同文件夹[《机械臂URDF格式规范.md》](./机械臂URDF格式规范.md)手工编写（或用其他三维软件导出 URDF/STL 后按规范改名整理），STL 网格也可由任意三维软件导出——只要文件名与所属 link 同名、引用为相对路径即可；
3. 导出后手动补充工具参考系（插件不导出空 link）：在 `</robot>` 前添加 `frame_end_tcp` 等空 link 与对应 fixed joint（写法见规范 md 第 3 节）。

### 3.2 完善：按规范逐项打磨

对照第 2 节要点与规范 md 的检查清单逐项修复：`<robot name>` 与型号名一致、link/joint 命名合规、每个主动关节四限位完整且与 config（21 单元产出）硬限位一致、从动关节带 mimic、关节轴与电机轴同向/反向重合；修改几何或拓扑后应重新核对 mesh 引用与末端帧位置。

四限位数值与 config 硬限位（21 单元 `backend.arm/end.joints` 四键）**同源**：最初数值来自电机数据手册 / 机械结构设计（行程、额定速度、额定力矩），先定一侧再逐键对齐另一侧（容差 1e-4），先填 URDF 或先填 config 均可。

需要 MuJoCo 仿真时再由定稿 URDF 转换生成 `mjcf/<型号>.xml`：使用 MuJoCo 官方发行版自带的转换工具——`compile` 命令行（`compile <型号>.urdf mjcf/<型号>.xml`）或 `simulate` 拖入 URDF 后经 Save as XML 另存（工具说明见 [MuJoCo 官方文档](https://mujoco.readthedocs.io/)）；转换后核对 mesh 仍以 `../meshes/*.STL` 相对引用。

### 3.3 测试：运行检查脚本

> 📌 **前置依赖**：检查⑥（`frame_end` 帧存在）与检查⑦（活动关节与 config arm 段一致）
> 需读取 21 单元产出的 `config/<型号>.yaml`（`joyarm.frame_end` 与 `backend.arm.joints`
> 关节表）；21 未完成时这两项会 FAIL 并提示 config 加载失败，属预期——先回
> [21_config_creator](../21_config_creator/) 建好 config 再复跑。

在 `joyarm_code/` 目录下运行（离线，无需硬件）：

```bash
python quick_start/22_urdf_creator/22_urdf_check.py --model <型号>
```

脚本对七项要求逐项输出 `[PASS]/[FAIL]` 及明细表，判定标准为**七项全 PASS → 打印 SUCCESS、退出码 0**：

| # | 检查项 | 通过标准 |
|:---:|:---:|:---:|
| ① | URDF 文件存在 | `robot_model/<型号>/urdf/<型号>.urdf` 存在 |
| ② | pinocchio 可加载 | `pin.buildModelFromUrdf` 解析成功 |
| ③ | `<robot name>` = 型号名 | 根元素 name 属性与型号名一致 |
| ④ | 逐关节四限位完整 | 非 fixed 关节 `lower/upper/effort/velocity` 四键齐全 |
| ⑤ | mesh 相对路径文件存在 | 全部 `<mesh filename>` 为相对路径且文件存在 |
| ⑥ | `frame_end` 帧存在 | config `joyarm.frame_end` 指定的帧存在于模型 frames |
| ⑦ | 活动关节与 config arm 段一致 | 锁定非 arm 名单后缩减模型关节与 config 同名同序、nq 一致（同 `JoyArm._build_pin_model` 口径） |

任一项 FAIL 时按明细提示修复 URDF/网格后重跑，直至 SUCCESS；随后进入 23（backend，见 [23_backend_creator](../23_backend_creator/)）与 24（整机创建，见 [24_arm_creator](../24_arm_creator/)）单元。脚本中 URDF 加载、`frame_end`、活动关节比对的判据与 `joyarm.py` 完全同源；**限位数值与 config 硬限位的逐键一致**（容差 1e-4）不在本脚本七项检查之内——由 JoyArm 构造自检把关（21 单元检查脚本的 H15 行），完善阶段请对照本节上方七项检查表，另按 §2 要点表第 6 行「四限位」条目与规范 md 检查清单第 8 项人工逐键核对。
