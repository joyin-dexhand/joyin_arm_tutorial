# 关节电机

> 📌 **本章定位**：介绍如何通过 Python、C++ 程序控制单关节电机。

## 1 硬件准备

* USB转CAN线(图)
* *待补充*机械臂(图)

## 2 环境准备(Python)

### 2.1 安装 uv

本教程统一使用 [uv](https://docs.astral.sh/uv/) 管理 Python 虚拟环境与依赖。

**Linux**

    curl -LsSf https://astral.sh/uv/install.sh | sh

**Windows PowerShell**

    irm https://astral.sh/uv/install.ps1 | iex

### 2.2 拉取代码

```shell
git clone *待补充*
cd joyin_arm_tutorial/joyarm_code
```

### 2.3 创建环境并安装依赖

```shell
uv sync   # 创建 .venv + 按 uv.lock 安装全部依赖 + 可编辑安装 joyarm_core
source .venv/bin/activate
```


