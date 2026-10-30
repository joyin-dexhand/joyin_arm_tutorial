---
name: joyarm-urdf-creator
description: "为 JoyArm 新型号机械臂创建或修复 robot_model 模型资产（URDF、mesh 网格、MuJoCo mjcf）时使用——新建型号目录骨架、SolidWorks 导出或手写 URDF、mesh 整理、命名与四限位规范核对、运行七项 URDF 校验并修复报错"
---

# JoyArm URDF 模型资产创建

为新型号机械臂接入 joyarm_core 创建 `robot_model/` **模型资产**（URDF + 网格 + MuJoCo 模型），负责创建、完善与测试。型号接入由四个 skill 依序协作：[joyarm-config-creator](../joyarm-config-creator/) 配置文件（`config/<型号>.yaml`）→ **本 skill 模型资产** → [joyarm-backend-creator](../joyarm-backend-creator/) 硬件后端（`backend/backend_<型号>.py`）→ [joyarm-arm-creator](../joyarm-arm-creator/) 整机创建与联调；本 skill 产出合格后，由 joyarm-arm-creator 统一装配。

本 skill 自带文件：

| 文件 | 说明 |
|:---:|:---:|
| `scripts/urdf_check.py` | 离线检查脚本：七项检查逐项 PASS/FAIL，全部通过退出码 0 |
| `references/机械臂URDF格式规范.md` | URDF 详细格式规范（命名总则、关节轴与限位、检查清单） |

## 1 目录约定

模型资产统一存放在 `joyarm_core/robot_model/<型号>/` 下，目录名 = 型号名 = URDF 文件名（去扩展名）。JoyArm 创建对象时按这一固定命名寻址（`JoyArmFactory` 第 ③ 步按序校验 `robot_model/<型号>/`、`meshes/`、`urdf/`、`urdf/<型号>.urdf` 四路径存在，见 `joyarm_core/joyarm/__init__.py`），命名不符将无法接入：

```text
joyarm_core/robot_model/<型号>/
├── urdf/<型号>.urdf     # URDF 模型文件（必备，pinocchio 加载入口）
├── meshes/*.STL        # 连杆网格（必备，文件名与所属 link 同名，如 link2.STL ↔ link2）
└── mjcf/<型号>.xml      # MuJoCo 仿真模型（可选，由 URDF 转换生成）
```

URDF 内 mesh 采用**相对路径**引用（如 `../meshes/link0.STL`），因此 `urdf/` 与 `meshes/` 必须整体存放、一并迁移，不可拆开。

## 2 URDF 规范要点

下表是接入 joyarm_core 的**硬性要点**（检查脚本 `scripts/urdf_check.py` 逐项把关的就是这些）：

| 要点 | 要求 |
|:---:|:---:|
| 纯 URDF | 不使用 `xacro`，单文件自包含 |
| 型号标识 | `<robot name>` = 型号名 = `robot_model/<型号>/` 目录名 |
| mesh 引用 | 一律相对路径（`../meshes/*.STL`），且文件真实存在 |
| 单位 | 一律 SI：长度 m、质量 kg、角度 rad、时间 s |
| 四限位 | 每个主动关节 `lower / upper / effort / velocity` 四键完整；数值还须与 config 硬限位逐键一致（容差 1e-4，见规范 md 检查清单第 8 项） |
| mimic 从动 | 从动关节必须带 `<mimic joint=… multiplier=… offset=…/>` |
| 末端帧 | 提供 `frame_end_<位置>` 工具参考系（如 `frame_end_tcp`），空 link + fixed joint |

命名总则：link / joint / frame 一律小写 `snake_case`，编号从 0 起，同编号 link、joint 成对出现；`link*` 为有质量/几何的实体连杆，`frame_*` 为无质量纯坐标系。核心命名模式：

| 部位 | 命名 | 说明 |
|:---:|:---:|:---:|
| arm 链 | `link0`（固定基座）→ `link1`~`linkn` + `joint1`~`jointn` | 连杆与驱动关节同号，编号连续无下划线 |
| end 基座 | `link_end0` + `joint_end0`（fixed） | 固连至 arm 末连杆 `linkn` |
| end 关节 | 主动 `joint_end<N>`；从动 `link_end_<名称>` + `<mimic>` | 从动如手指 `link_end_fingerleft` |
| end 参考系 | `frame_end_<位置>` + `joint_frame_end_<位置>`（fixed 到 `link_end0`） | 空 link 无惯量无几何，如 `frame_end_tcp` |

**详细规则**见同 skill 内 [机械臂URDF格式规范.md](references/机械臂URDF格式规范.md)（含关节轴与电机轴同向/反向重合约定、九项检查清单）；**导出操作**（SolidWorks + sw_urdf_exporter 插件逐步流程）见教程第七章《构型结构设计与 URDF 导出》（[docs/chapt7_urdf.md](../../../../docs/chapt7_urdf.md)）。

## 3 创建 / 完善 / 测试三步流程

新型号模型资产按「先建骨架 → 再对照规范逐项打磨 → 最后跑检查脚本」三步完成。

### 3.1 创建：建立目录骨架并导入 URDF 与网格

1. 按第 1 节约定新建 `joyarm_core/robot_model/<型号>/{urdf/, meshes/}` 目录骨架（`mjcf/` 需要 MuJoCo 仿真时再建）；
2. 推荐在 SolidWorks 中由装配体经 `sw_urdf_exporter` 插件导出 URDF 与 STL（逐步操作见第七章第 5 节）；没有 SolidWorks 时可按规范 md 手工编写 URDF（纯文本 XML），或用其他三维软件导出 URDF/STL 后按规范改名整理——只要 mesh 文件名与所属 link 同名、引用为相对路径即可；
3. 导出后手动补充工具参考系（插件不导出空 link）：在 `</robot>` 前添加 `frame_end_tcp` 等空 link 与对应 fixed joint（写法见规范 md 第 3 节）。

### 3.2 完善：按规范逐项打磨

对照第 2 节要点与规范 md 检查清单逐项修复：`<robot name>` 与型号名一致、link/joint 命名合规、主动关节四限位完整且与 config（joyarm-config-creator 产出）硬限位一致、从动关节带 mimic、关节轴与电机轴同向/反向重合；修改几何或拓扑后重新核对 mesh 引用与末端帧位置。

四限位数值与 config 硬限位（joyarm-config-creator 的 `backend.arm/end.joints` 四键）**同源**：最初数值来自电机数据手册 / 机械结构设计（行程、额定速度、额定力矩），先定一侧再逐键对齐另一侧（容差 1e-4），先填 URDF 或先填 config 均可。

需要 MuJoCo 仿真时，由定稿 URDF 经 MuJoCo 官方转换工具生成 `mjcf/<型号>.xml`：命令行 `compile <型号>.urdf mjcf/<型号>.xml`，或 `simulate` 拖入 URDF 后 Save as XML 另存（工具说明见 [MuJoCo 官方文档](https://mujoco.readthedocs.io/)）；转换后核对 mesh 仍以 `../meshes/*.STL` 相对引用。

### 3.3 测试：运行检查脚本

> 📌 **前置依赖**：检查⑥（`frame_end` 帧）与检查⑦（活动关节比对）需读取 joyarm-config-creator 产出的 `config/<型号>.yaml`；config 未就绪时这两项会 FAIL 并提示加载失败，属预期——先回 [joyarm-config-creator](../joyarm-config-creator/) 建好 config 再复跑。

在 `joyarm_code/` 目录下运行（离线，无需硬件）：

```bash
python .agents/skills/joyarm-urdf-creator/scripts/urdf_check.py --model <型号>
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

任一项 FAIL 按明细提示修复后重跑，直至 SUCCESS；随后进入 [joyarm-backend-creator](../joyarm-backend-creator/)（backend）与 [joyarm-arm-creator](../joyarm-arm-creator/)（整机创建）。注意两点：脚本中 URDF 加载、`frame_end`、活动关节比对的判据与 `joyarm.py` 完全同源；**限位数值与 config 硬限位的逐键一致**（容差 1e-4）不在七项检查之内——由 JoyArm 构造自检把关（joyarm-config-creator 检查脚本 H15 行），完善阶段请按第 2 节要点表「四限位」条目与规范 md 检查清单第 8 项人工核对。
