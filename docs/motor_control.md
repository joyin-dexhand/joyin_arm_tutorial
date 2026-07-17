# 电机控制指南

---

## 概述

本文将介绍如何通过Python、C++程序控制xxx电机。

## 硬件准备

* USB转CAN线(图)
* xxx机械臂(图)

## 环境准备(Python)

### 1.创建conda或uv环境 

**安装 conda 环境(可选)：**

**Linux**

    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh

**Windows PowerShell**

    wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -outfile ".\miniconda.exe"
    Start-Process -FilePath ".\miniconda.exe" -ArgumentList "/S" -Wait
    del .\miniconda.exe

    
**安装 uv 环境(可选)：**

**Linux**

    curl -LsSf https://astral.sh/uv/install.sh | sh

**Windows PowerShell**

    irm https://astral.sh/uv/install.ps1 | iex

### 2.拉取代码并运行虚拟环境

**拉取代码：**

    git clone xxxx

**运行conda环境(可选)：**

    source ~/miniconda3/bin/activate
    conda create -n my_arm python=3.10
    conda activate my_arm

**运行uv环境(可选)：**

    cd xxx
    uv venv --python 3.10
    source .venv/bin/activate

### 3.安装依赖




