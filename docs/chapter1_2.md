# 关节电机

本文将介绍如何通过Python、C++程序控制xxx电机。

## 1 硬件准备

* USB转CAN线(图)
* xxx机械臂(图)

## 2 环境准备(Python)

### 2.1 安装 uv

本教程统一使用 [uv](https://docs.astral.sh/uv/) 管理 Python 虚拟环境与依赖。

**Linux**

    curl -LsSf https://astral.sh/uv/install.sh | sh

**Windows PowerShell**

    irm https://astral.sh/uv/install.ps1 | iex

### 2.2 拉取代码并创建虚拟环境

**拉取代码：**

```shell
git clone xxxx
```

**创建并激活 uv 虚拟环境：**

```shell
cd joyin_arm_tutorial
uv venv --python 3.10
source .venv/bin/activate
```

### 2.3 安装依赖

```shell
uv pip install PySide6 numpy matplotlib
```


