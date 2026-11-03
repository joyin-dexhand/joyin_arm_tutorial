# 机械臂 URDF 格式规范

> **适用范围**：`joyarm_core/robot_model/` 所有型号机械臂的 URDF。

## 1. 总原则

1. **纯 URDF**：不使用 `xacro`。
2. **型号标识**：`<robot name>` = 型号名（<型号>） = `robot_model/<型号>/` 目录名。
3. **单位**：一律 SI——长度 m、质量 kg、角度 rad、时间 s；`rpy` 为rad。

## 2. 文件与目录结构

robot_model/<型号>/
├── urdf/<型号>.urdf      # URDF 文件
└── meshes/*.STL           # mesh 文件

- urdf 中的 mesh 标签采用**相对路径**引用（如 `../meshes/link0.STL`），便于移植。
- mesh 文件名与所属 link 同名（`link2.STL` ↔ `link2`）。

## 3. link / joint 命名
 
总则：小写 `snake_case`；编号从 0 起；同编号 `<link>`、`<joint>` 成对出现（`link*` <-> `joint*`）。

| 类别 | 命名 | 说明 |
|:---:|:---:|:---:|
| 基座连杆 | `link0` | 机械臂固定基座（含电机1定子部分） |
| arm 连杆 | `link1`~`linkn` | 各级连杆（与驱动关节同号，含固连于连杆的电机组件） |
| arm 关节 | `joint1`~`jointn` | 连接同编号link和上一编号link |
| end 基座连杆 | `link_end0` | end 的基座部分，固连至 arm 的 linkn 末连杆 |
| end 基座关节 | `joint_end0` | linkn → link_end0，fixed |
| end 主动关节 | `joint_end<N>` | 如 `joint_end1`（二指夹爪旋转驱动轴） |
| end 从动连杆与关节 | `link_end_<N>` + `joint_end_<N>` | N为具体型号名 + `<mimic>`；joint 名与 link 同名，如：`<N>` 为 `fingerleft` |
| end 虚拟参考系 | `frame_end_<位置>` + `joint_frame_end_<位置>` | 如 `frame_end_tcp`、`frame_end_fingertip`；link 为空标签 `<link/>`（无惯量无几何），joint fixed 到 link_end0 |

- **前缀区分**：`link*` = 有质量/几何的实体连杆；`frame_*` = 无质量纯坐标系（TCP、指尖点等参考系）。
- **从动关节**：必须带 `<mimic joint=... multiplier=... offset=.../>`。
- **分区分节**：`arm` / `end` 为一级分区，三行banner；每个 link / joint 一条 banner 注释，`=` 数全文对齐一致。

## 4. 关节轴与限位

1. **`关节轴`需与`电机轴`同向重合（定子位于基座侧连杆上时）或反向重合（定子位于末端侧连杆上时）**，推荐采用MDH法建系。
2. **每个主动关节必须包含完整四限位**：`lower / upper / effort / velocity`。

## 5. 检查清单

| # | 检查项 | 通过标准 |
|:---:|:---:|:---:|
| 1 | XML 与型号名 | well-formed；`<robot name>` = 型号名（<型号>） |
| 2 | mesh 引用完整 | 每个 `../meshes/*.STL` 存在，且无孤立 mesh |
| 3 | arm 关节命名 | `joint1`~`jointN` 无下划线，N<=9 |
| 4 | arm 连杆命名 | `link0`~`linkN` 无下划线，N<=9 |
| 5 | end 关节命名 | end基座 `joint_end0`，主动 `joint_end<正整数>`，从动`joint_end_<名称>`，参考帧 `joint_frame_end_<frame名称>` |
| 6 | end 连杆命名 | end基座 `link_end0`，主动 `link_end<正整数>`，从动`link_end_<名称>`,参考帧 `frame_end_<frame名称>` |
| 7 | 关节轴 | axis 单位化；fixed 无 axis |
| 8 | 限位 4 键 | 齐全，且与 config 四限位逐键差 ≤1e-4（joint1~N + end 主动关节）；关节名与 config 同名同序、编号连续、驱动关节全部登记 |
| 9 | mimic 从动关节 | 带 mimic，multiplier 与实测行程比一致，按传动比计算限速和限力 |

## 非强制建议

1. 建议在 `Solidworks` 中由机械臂装配体 `CAD模型` 经 `sw_urdf_exporter` 插件导出 urdf；
2. 建议采用MDH法建连杆系。
