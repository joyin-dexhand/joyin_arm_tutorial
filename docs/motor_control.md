# 电机控制指南

---

## 概述

本文将介绍如何通过Python、C++程序控制xxx电机。

## 硬件准备

- USB转CAN线(图)
- xxx机械臂(图)

## 环境准备

- 创建conda或者uv环境 (Linux)
  (conda)
  ```
  mkdir -p ~/miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm ~/miniconda3/miniconda.sh
  ```
  
  (uv)
  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  
- 安装python依赖
  
  pip install
  
- 将机械臂上电

## 内容

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
