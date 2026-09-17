# 机械臂 URDF 导出规范

> **适用范围**：`joyarm_core/robot_model/` 下所有机械臂型号的 URDF 资产；参考实现为[`joyarm_dm/`](./joyarm_dm/)。

## 1. 总原则

1. **权威数据源**：URDF 为几何/限位/命名的权威数据源；`config/<model>.yaml` 中的同类字段需对齐或等效对齐至URDF文件。
2. **纯 URDF**：不使用 `xacro`；建议在 `Solidworks` 中由机械臂装配体 `CAD模型` 经 `sw_urdf_exporter` 插件导出 urdf。
3. **型号标识**：`<robot name>` = 型号名（joyarm_*） = `robot_model/<model>/` 目录名 = `config/<model>.yaml` 文件名。
4. **单位**：一律 SI——长度 m、质量 kg、角度 rad、时间 s；`rpy` 为rad。

## 2. 文件与目录结构

```
robot_model/<model>/
├── urdf/<model>.urdf      # URDF 文件
└── meshes/*.STL           # mesh 文件
```

- urdf 中的 mesh 采用**相对路径**引用（如 `../meshes/link0.STL`），便于模型移植。
- mesh 文件名与所属 link 同名（`link2.STL` ↔ `link2`）。

## 3. link / joint 命名
 
总则：小写 `snake_case`；编号从 0 起；同编号 `<link>`、`<joint>` 成对出现（`link_*` <-> `joint_*`）；**关节顺序 = URDF 中出现顺序 = config `joints` 列表顺序 = q 向量顺序**（硬校验）。

| 类别 | 命名 | 说明 |
|:---:|:---:|:---:|
| 基座连杆 | `link0` | 机械臂固定基座（含电机1定子部分） |
| arm 连杆 | `link1`~`linkn` | 各级连杆（与驱动关节同号，含固连于连杆的电机组件） |
| arm 关节 | `joint1`~`jointn` | **代码硬约定**：仅 `joint1`~`joint9` 计入本体关节数与顺序校验（`_ARM_JOINT_NAMES`），不得带下划线或其它前缀 |
| end 基座连杆 | `link_end0` | end 的基座部分，固连至 arm 的 linkn 末连杆 |
| end 基座关节 | `joint_end0` | linkn → link_end0，fixed |
| end 主动关节 | `joint_end<N>` | 如 `joint_end1`（二指夹爪旋转驱动轴），与 config `backend.end.joints[].name` 一致 |
| end 从动连杆与关节 | `link_end_<N>` + `joint_end_<N>` | N为具体型号名 + `<mimic>`；joint 名与 link 同名，如：`<N>` 为 `fingerleft` |
| end 虚拟参考系 | `frame_end_<位置>` + `joint_frame_end_<位置>` | 如 `frame_end_tcp`、`frame_end_fingertip`；link 为空标签 `<link/>`（无惯量无几何），joint fixed 到 link_end0 |

- **前缀区分**：`link*` = 有质量/几何的实体连杆；`frame_*` = 无质量纯坐标系（TCP、指尖点等参考系）。
- **从动关节**：必须带 `<mimic joint=... multiplier=... offset=.../>`。
- **分区分节**：`arm` / `end` 为一级分区；每个 link / joint 一条 banner 注释，`=` 数全文对齐一致。

## 4. 关节轴与限位

1. **关节轴取为`电机轴`（定子位于基座侧连杆上时）或`电机轴反向`（定子位于末端侧连杆上时）**，推荐采用MDH法建系。
2. **电机的标定零位取为关节角零位**。
3. **每个主动关节必须包含完整四限位**：`lower / upper / effort / velocity`。
4. **urdf 与 config 同步对齐**：`config/<model>.yaml` 的 `backend.arm.joints` 限位 `q_min、q_max、dq_max、tau_max` 必须与 URDF `<limit>` 一致（否则警告）；end 的主动关节（`joint_end<N>`）同样保持两文件一致。**改任一侧必须同步另一侧。**

## 5. 导出自检清单

| # | 检查项 | 通过标准 |
|:---:|:---:|:---:|
| 1 | XML 与型号名 | well-formed；`<robot name>` = 型号名 |
| 2 | mesh 引用完整 | 每个 `../meshes/*.STL` 存在，且无孤立 mesh |
| 3 | arm 关节命名 | `joint1`~`jointN` 无下划线，N<=9 |
| 4 | arm 连杆命名 | `link0`~`linkN` 无下划线，N<=9 |
| 5 | end 关节命名 | end基座 `joint_end0`，主动 `joint_end<正整数>`，从动`joint_end_<名称>`，参考帧 `joint_frame_end_<frame名称>` |
| 6 | end 连杆命名 | end基座 `link_end0`，主动 `link_end<正整数>`，从动`link_end_<名称>`,参考帧 `frame_end_<frame名称>` |
| 7 | 关节轴 | axis 单位化；fixed 无 axis |
| 8 | 限位 4 键 | 齐全，且与 config 数值一致（joint1~N + end 主动关节） |
| 9 | mimic 从动关节 | 带 mimic，multiplier 与实测行程比一致，按传动比计算限速和限力 |
