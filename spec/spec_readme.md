# spec_readme

本项目使用 [MkDocs](https://www.mkdocs.org/) + [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 主题构建文档站点。下面是开发与维护中常用的指令。

> 💡 下文命令默认在项目根目录（含 `mkdocs.yml` 的目录）执行。建议用 `python3 -m mkdocs ...` 形式调用，避免与系统其他 mkdocs 冲突。

## 1 环境安装

```bash
# 安装核心框架 + Material 主题（本项目的全部依赖）
pip install mkdocs mkdocs-material

# 验证安装
python3 -m mkdocs --version
```

> 本项目当前使用：MkDocs 1.6.1 + Material 9.x。如需固定版本，可创建 `requirements.txt`：
> ```text
> mkdocs==1.6.1
> mkdocs-material
> ```

## 2 本地预览（开发调试）

```bash
# 启动本地实时预览服务器（默认 http://127.0.0.1:8000）
python3 -m mkdocs serve

# 指定地址和端口
python3 -m mkdocs serve --dev-addr 127.0.0.1:8000

# 仅预览，不监控文件变化（不自动刷新）
python3 -m mkdocs serve --no-livereload
```

启动后修改 `docs/` 下任意 `.md` 或 `mkdocs.yml`，浏览器会**自动刷新**。

## 3 构建静态站点

```bash
# 构建到 site/ 目录（默认）
python3 -m mkdocs build

# 严格模式：把警告视为错误，用于发布前检查（推荐）
python3 -m mkdocs build --strict

# 构建前先清理 site/，避免残留旧文件
python3 -m mkdocs build --clean

# 构建到自定义目录
python3 -m mkdocs build --site-dir path/to/output
```

> `site/` 为构建产物，已加入 `.gitignore`，不要提交。

## 4 部署到 GitHub Pages

本项目已配置 [GitHub Actions](../.github/workflows/deploy.yml) 自动部署：**推送 `main` 分支即自动构建并发布**，无需手动操作。

如需本地手动触发部署（用于调试 CI 或紧急发布）：

```bash
# 一键构建并推送到 gh-pages 分支（GitHub Pages 托管）
mkdocs gh-deploy --force
```

部署成功后访问：**https://joyin-dexhand.github.io/joyin_arm_tutorial/**

> 首次部署需在 GitHub 仓库 **Settings → Pages** 中，将 Source 设为 `gh-pages` 分支 `/ (root)` 目录。

## 5 项目维护常用命令

| 操作 | 命令 | 说明 |
|:---|:---|:---|
| 查看帮助 | `python3 -m mkdocs -h` | 列出所有子命令 |
| 查看某命令帮助 | `python3 -m mkdocs serve -h` | 查看具体参数 |
| 新建项目 | `python3 -m mkdocs new [目录名]` | 生成初始结构（本项目已存在，无需用） |
| 查看版本 | `python3 -m mkdocs --version` | 查看版本信息 |

## 6 项目配置要点（mkdocs.yml）

| 配置项 | 当前设置 | 作用 |
|:---|:---|:---|
| `site_name` | JoyArm 机械臂教程 | 站点标题 |
| `theme.name` | material | 使用 Material 主题 |
| `theme.features` | navigation.expand | 左侧目录默认全部展开 |
| `theme.palette` | default / slate | 亮/暗模式切换，记忆用户选择 |
| `markdown_extensions` | pymdownx.superfences (mermaid) | 支持 Mermaid 流程图 |
| `extra_css` | stylesheets/mermaid.css、extra.css | 自定义样式（配色、布局） |
| `nav` | 使用篇/基础篇/进阶篇/应用篇 | 左侧目录结构与跳转 |

## 7 常见问题

**Q：修改了文档但浏览器没刷新？**
A：确认 `mkdocs serve` 正在运行；强刷浏览器（Ctrl+F5）清缓存。

**Q：`--strict` 构建报错"link not found"？**
A：通常是 Markdown 里引用了不存在的图片或页面。检查相对路径是否正确（图片需放在 `docs/` 下）。

**Q：Mermaid 流程图显示为代码块而非图形？**
A：确认 `mkdocs.yml` 已配置 `pymdownx.superfences` 的 mermaid custom fence（本项目已配置）。

**Q：部署后站点没更新？**
A：GitHub Pages 有 1~2 分钟缓存延迟；检查 [Actions 页面](https://github.com/joyin-dexhand/joyin_arm_tutorial/actions) 构建是否成功。
