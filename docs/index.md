# JoyArm 机械臂教程

[机械臂渲染图]()

JoyArm 是一套结合实机的协作机械臂学习教程，按 **使用、基础、进阶、应用** 四篇组织：先用起来，再讲清原理，最后完成综合实践任务。每个知识点都配可复现的实操，适合从零开始、想真正做出一台好用易用机械臂的学习者。

[演示动图或视频]()

---

## 特点与定位

| 维度 | 主要特点 | 区别于 | 适合读者 |
| :---: | :---: | :---: | :---: |
| **理论** | 必要理论配实操，概念直观 | 纯理论教材与公式堆砌 | 想理解"为什么"的初学者 |
| **实践** | 贴近真实任务，步骤可复现 | 脱离实机的纸上项目 | 想做出能用的机械臂的开发者 |
| **体系** | 从单关节到整机，全栈贯通 | 只讲单点的碎片教程 | 想系统性掌握的学习者 |

---

## 项目指引

CAD结构件：*仓库链接*

控制代码：*仓库链接*

完整BOM表和外购标准件：[快速上手](chapter1_1.md)

快速上手：[快速上手](chapter1_1.md)

---

## 篇章速览

> 全教程共 4 篇：使用篇（操作上手）、基础篇（机器人学必备）、进阶篇（专业进阶）、应用篇（实机应用）。

<table>
  <thead>
    <tr>
      <th>篇标题</th>
      <th>章标题</th>
      <th>内容概览</th>
      <th>跳转链接</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4" align="center"><strong>使用篇</strong></td>
      <td align="center">快速上手</td>
      <td align="center">让机械臂快速动起来的操作流程</td>
      <td align="center"><a href="chapter1_1.md">快速上手</a></td>
    </tr>
    <tr>
      <td align="center">关节电机</td>
      <td align="center">通过 Python/C++ 程序控制单关节电机</td>
      <td align="center"><a href="chapter1_2.md">关节电机</a></td>
    </tr>
    <tr>
      <td align="center">六轴整臂</td>
      <td align="center">整臂上电、关节与位置控制、示教复现</td>
      <td align="center"><a href="chapter1_3.md">六轴整臂</a></td>
    </tr>
    <tr>
      <td align="center">两指夹爪</td>
      <td align="center">夹爪接线、位置与力度控制（仅操作）</td>
      <td align="center"><a href="chapter1_4.md">两指夹爪</a></td>
    </tr>
    <tr>
      <td rowspan="6" align="center"><strong>基础篇</strong></td>
      <td align="center">第一章 机器人与机械臂初识</td>
      <td align="center">机器人发展历史、分类与核心概念速扫盲</td>
      <td align="center"><a href="chapter2_1.md">第一章</a></td>
    </tr>
    <tr>
      <td align="center">第二章 空间位姿与位置正运动学</td>
      <td align="center">位姿矩阵表示、MDH 建系、正运动学求解</td>
      <td align="center"><a href="chapter2_2.md">第二章</a></td>
    </tr>
    <tr>
      <td align="center">第三章 位置逆运动学</td>
      <td align="center">逆解可解性、代数/几何/数值解法</td>
      <td align="center"><a href="chapter2_3.md">第三章</a></td>
    </tr>
    <tr>
      <td align="center">第四章 速度运动学与静力学</td>
      <td align="center">速度传递、雅可比、奇异性、静力学</td>
      <td align="center"><a href="chapter2_4.md">第四章</a></td>
    </tr>
    <tr>
      <td align="center">第五章 轨迹生成</td>
      <td align="center">关节空间与笛卡尔空间轨迹规划</td>
      <td align="center"><a href="chapter2_5.md">第五章</a></td>
    </tr>
    <tr>
      <td align="center">第六章 运动控制</td>
      <td align="center">单关节电机控制、整机位置/速度控制</td>
      <td align="center"><a href="chapter2_6.md">第六章</a></td>
    </tr>
    <tr>
      <td rowspan="4" align="center"><strong>进阶篇</strong></td>
      <td align="center">第七章 构型与结构设计、关节选型、装配体URDF导出</td>
      <td align="center">构型设计、结构设计、电机选型、URDF 导出</td>
      <td align="center"><a href="chapter3_1.md">第七章</a></td>
    </tr>
    <tr>
      <td align="center">第八章 动力学及控制实现</td>
      <td align="center">牛顿欧拉递推、动力学方程、基于动力学的控制</td>
      <td align="center"><a href="chapter3_2.md">第八章</a></td>
    </tr>
    <tr>
      <td align="center">第九章 力控制</td>
      <td align="center">纯力控、力位混合、阻抗、导纳控制</td>
      <td align="center"><a href="chapter3_3.md">第九章</a></td>
    </tr>
    <tr>
      <td align="center">第十章 ROS2</td>
      <td align="center">工作空间、话题/服务通信、moveit2 集成</td>
      <td align="center"><a href="chapter3_4.md">第十章</a></td>
    </tr>
    <tr>
      <td rowspan="5" align="center"><strong>应用篇</strong></td>
      <td align="center">第十一章 状态监测与安全软防护</td>
      <td align="center">关节层/末端层/整机层状态监测与防护</td>
      <td align="center"><a href="chapter4_1.md">第十一章</a></td>
    </tr>
    <tr>
      <td align="center">第十二章 示教和遥操作</td>
      <td align="center">同臂/主从同构/主从异构示教、VR 遥操作</td>
      <td align="center"><a href="chapter4_2.md">第十二章</a></td>
    </tr>
    <tr>
      <td align="center">第十三章 末端执行器统一接口与自识别</td>
      <td align="center">夹爪/灵巧手/吸附等各类末端原理与接口</td>
      <td align="center"><a href="chapter4_3.md">第十三章</a></td>
    </tr>
    <tr>
      <td align="center">第十四章 用户接口</td>
      <td align="center">UI、语音、视觉、NFC、UWB、体感</td>
      <td align="center"><a href="chapter4_4.md">第十四章</a></td>
    </tr>
    <tr>
      <td align="center">第十五章 JoyArm 综合项目实践</td>
      <td align="center">感知/规划/控制/交互全栈综合项目</td>
      <td align="center"><a href="chapter4_5.md">第十五章</a></td>
    </tr>
  </tbody>
</table>

---

## 主要参考的资料和项目

### 主要参考书籍

| 名称 | 简介 | 链接 |
|:---:|:---|:---:|
| 《机器人学导论（原书第四版）》 | John J. Craig 著，运动学、雅可比、动力学与力控制的经典教材，本教程基础篇与进阶篇的主要理论参考 | [仓库内 PDF](reference/机器人学导论（第4版）.pdf) |
| 《机器人学》（第四版） | 蔡自兴、谢斌 编著（清华大学出版社，2022，ISBN 9787302598220），国内高校广泛使用的中文经典教材，涵盖空间描述、运动学、动力学、控制与轨迹规划 | *待补充* |

### 主要参考开源项目

| 名称 | 简介 | 链接 |
|:---:|:---|:---:|
| reBot 机械臂 | Seeed 出品的开源六轴机械臂项目（6 DOF + 夹爪）：reBot-DevArm 提供全套软硬件图纸；reBotArm_control_py 是基于 Pinocchio 的 Python 控制库 | [DevArm](https://github.com/Seeed-Projects/reBot-DevArm) / [control_py](https://github.com/vectorBH6/reBotArm_control_py) |
| Panthera-HT_SDK | Panthera-HT 六轴机械臂官方 C++/Python SDK，支持位置/速度/力矩控制、运动学/动力学建模与主从遥操作 | https://github.com/HighTorque-Robotics/Panthera-HT_SDK |
| Dummy-Robot | 稚晖君（peng-zhihui）的超迷你六轴机械臂机器人项目，固件核心为运动学姿态解算，已完整开源 | https://github.com/peng-zhihui/Dummy-Robot |

---

## 开源许可证

*Apachi-2.0*

---