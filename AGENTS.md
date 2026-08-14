# 项目说明（ZCode 项目记忆）

本仓库是 **JoyArm 机械臂教程的文档站点源码**：用 MkDocs + Material for MkDocs 主题，把"使用 / 基础 / 进阶 / 应用"四篇教程编译成静态网站，经 GitHub Pages 发布。教程正文（`docs/*.md`）是核心产出；`joyarm_code/` 为配套代码（可复用核心库 + 章节演示脚本）。

> ⚠️ **两类内容**：`docs/` 下是面向读者的教程正文（会渲染上网）；`AGENTS.md` 等内部文档不进站点。
> **文档层级**（自上而下只概述 + 链接，不重复展开）：根 `AGENTS.md`（教程站点）→ [`joyarm_code/AGENTS.md`](./joyarm_code/AGENTS.md)（代码库，agent/开发者向）→ [`joyarm_code/README.md`](./joyarm_code/README.md)（使用者向基础）。子库 AGENTS 依赖其 README 并在其上**补充**维护信息，不重复；README 给人看、AGENTS 给 agent 看。
> **职责分离**：本文件只维护**教程站点**部分；代码库 `joyarm_code/` 的架构、命名约定、API 与维护规则由子库自含的 [`joyarm_code/AGENTS.md`](./joyarm_code/AGENTS.md) 负责，用户向介绍见 [`joyarm_code/README.md`](./joyarm_code/README.md)。本文件提及代码库时只**概述 + 链接**，不展开细节。

## 仓库结构

```
joyin_arm_tutorial/
├── docs/                 # ★ 教程正文源（MkDocs 文档根）——核心产出
│   ├── index.md          #   站点首页
│   ├── *.md              #   各章正文（命名规则见下）
│   ├── images/           #   章内配图（按正文文件同名子目录，如 images/chapt1_intro/）
│   ├── reference/        #   参考资料
│   ├── stylesheets/      #   extra.css（表格居中等）、mermaid.css
│   └── javascripts/      #   mathjax.js（数学公式渲染配置）
├── joyarm_code/          # ★ 配套代码库（用户向见其 README.md；架构/约定/维护规则见其 AGENTS.md）
├── mkdocs.yml            # ★ 站点配置（主题、扩展、nav 目录树）
├── .github/workflows/    # deploy.yml：推 main 自动构建发布到 gh-pages
├── AGENTS.md             # 本文件：教程站点部分的项目记忆
└── site/                 # 构建产物（gitignore，勿提交、勿手改）
```

带 `★` 的为日常维护最常接触的目录/文件。

### 四篇各章教程与文件命名

教程按"先用起来 → 讲清原理 → 进阶 → 应用"的顺序组织。**使用篇用主题式命名（无章号），其余篇用 `chapt<全局章号>_<主题>.md`**（全局章号 = 侧边栏"第N章"，跨篇连续累加）：

| 篇 | 文件命名 | 章范围 | 定位 |
|:---:|:---:|:---:|:---:|
| **使用篇** | 主题式：`quickstart` / `joint_motor` / `arm_6dof` / `gripper` | 无章号 | 面向"想立刻用起来"，无需理论，按操作让机械臂动起来 |
| **基础篇** | `chapt1~6_<主题>.md` | 第 1~6 章 | 机器人学必备知识（位姿、运动学、轨迹、控制） |
| **进阶篇** | `chapt7~10_<主题>.md` | 第 7~10 章 | 构型设计/动力学/力控制/ROS2 等专业进阶 |
| **应用篇** | `chapt11~15_<主题>.md` | 第 11~15 章 | 状态监测、示教遥操作、末端执行器、用户接口、综合项目 |

- 例：`chapt2_fkine.md` = 第二章 空间位姿与位置正运动学。完整映射见 `mkdocs.yml` 的 `nav` 段。
- 配图目录与正文文件同名：`docs/images/<正文文件名>/`。
- 跨章引用一律用相对链接（如 `[第三章](chapt3_ikine.md)`），勿用绝对路径。

## 构建与部署

- **本地预览**：`python3 -m mkdocs serve`（默认 `http://127.0.0.1:8000`），改 `docs/` 或 `mkdocs.yml` 自动刷新。
- **自动部署**：`.github/workflows/deploy.yml` 监听 `main` 分支 push → `mkdocs gh-deploy --force` → 构建产物推 `gh-pages` 分支 → GitHub Pages。
- **线上地址**：https://joyin-dexhand.github.io/joyin_arm_tutorial/
- `site/` 为构建产物（已 `.gitignore`），**不要提交、不要手改**。

## `docs/chapt*.md`写作规范

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

- 文档务必保持**结构化、清晰、表达简洁、指代无歧义**，**避免啰嗦和讲解不清晰**
- **结构化有条理**，**不允许思维过于跳跃**，要对**新手友好**
- 每章（对应一个独立 md 文件）不设"本章小结"，用**概要**总结全章（类似论文摘要）：概要不设标题等级，直接紧随章标题后（第二行），用 `> 📌 **概述**：……` 引用块形式；正文按 H2 → H3 → H4 展开，最后一节通常为"本章实践"（对应程序脚本位于 `joyarm_code/chapt/` ）

### 教程各章对应的脚本文件

- 各章脚本存放于 `joyarm_code/chapt/<章序_功能>.py`（如 `chapt2_T_demo.py`），仅作一次性教学示例，**优先调用**核心库 `joyarm_code/joyarm/` 的功能（`fkine`、`Arm` 类等）。
- 脚本前几行必须添加使用说明段落注释：功能概要，以及新建/激活环境、安装 Python 与依赖、切换目录、运行脚本等步骤。
- 代码规范（命名 / 注释 / 结构）与代码库维护规则见 [`joyarm_code/AGENTS.md`](./joyarm_code/AGENTS.md)，本文件不展开。

## 常见陷阱

- ❌ 在本文件详细展开代码库内容 → ✅ 只概述并链接 [`joyarm_code/AGENTS.md`](./joyarm_code/AGENTS.md)。
- ❌ 忘记把 AGENTS.md 纳入提交 → ✅ 两份 AGENTS.md（根 + `joyarm_code/`）**正常跟踪**。
- ❌ 在 `docs/` 写"本章小结" → ✅ 改为章标题第二行的概要引用块。
- ❌ 手改或提交 `site/` → ✅ 它是构建产物，已忽略。
- ❌ 把内部文档放进 `docs/`（会被渲染上网）→ ✅ 放 `spec/`（需 `git add -f` 跟踪）。
- ❌ 跨章链接用绝对路径 → ✅ 用相对链接（如 `chapt3_ikine.md`）。
- ❌ 章号写成阿拉伯数字"第2章" → ✅ H1 与 nav 均用汉字"第二章"。
