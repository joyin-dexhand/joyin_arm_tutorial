# 第二章 空间位姿与位置正运动学

> 📌 **概述**：本章将建立机械臂的正运动学模型，求解给定各关节角时的机械臂末端空间位姿（三维位置+三维姿态）。
> 正运动学建模前，首先需知悉位姿的数学表示形式——位姿矩阵 $^A_B\bm{T}$，以及 RPY、轴角、四元数等常见的等价表示与转换关系；然后采用改进的 DH法（MDH法）为机械臂各连杆建系并将真实机械臂结构关系转化为统一的 MDH 参数表，进而得到相邻连杆系间的位姿矩阵$^{i-1}_i\bm{T}$，经链式连乘便可求得机械臂的**位置正运动学方程**（已知各关节角，求末端位姿）；最后封装正运动学求解算法并通过 JoyArm 调用，对各关节角经蒙特卡洛法随机采样后代入正运动学关系，绘制并分析末端工作空间（机械臂末端可触达的空间范围）。

## 1 空间位姿及矩阵表示

在进行正运动学建模前，首先需建立位姿的概念及其数学表示形式：被描述的物体相对于某参考系的三维空间位置和三维姿态朝向。

### 1.1 什么是空间位姿

#### 1.1.1 位姿：三维位置 + 三维姿态

想象机械臂要抓取桌上的某个物体，那么就需要知道被抓取的这个物品的：

- **位置（Position）**：被抓取物体在哪里。
- **姿态（Orientation）**：被抓取物体的形状朝向如何。

故而让机械臂执行某项任务，必须先精确描述目标**在哪里、朝哪个方向**——这就是**空间位姿**。

| 概念 | 英文 | 回答的问题 |
|:---:|:---:|:---:|
| 位置 | Position | 在哪里？ |
| 姿态 | Orientation | 朝哪个方向？ |
| 位姿 | Pose | “位置 + 姿态”完整状态 |

#### 1.1.2 为什么是六维

- **位置 3 维**：三维空间中一个点的位置需 $(x, y, z)$ 三个坐标表示 —— 前后、左右、上下。
- **姿态 3 维**：朝向可分解为"绕三个轴各转一定角度" —— 如人的：左右歪头、左右转头、上下低头抬头。

> 💡 **空间六维位姿 $\Leftrightarrow$ 关节六个电机**。这正是 **6R 机械臂**成为"黄金标配"的原因：六个关节电机具有 6 个独立控制量，恰好对应位姿的 6 个维度。

#### 1.1.3 必须指定参考系

位姿是**相对**的，脱离参考系谈位姿没有意义（同一物体相对桌子、地面等不同参考系的位姿必然不同）。在机械臂实际开发应用中，通常以基座系或工作台系作为默认的全局参考系。

若记参照物固连的坐标系为 $\{A\}$，被描述物体固连的坐标系为 $\{B\}$，则 $\{B\}$ 相对于 $\{A\}$ 的位姿记作 “$^A_B\bm{T}$” ，读作 “B 相对 A 的位姿变换（Translation）”。

### 1.2 空间位姿的数学表示

#### 1.2.1 位姿矩阵 $^A_B\bm{T}$

位姿的数学表示形式，最基础和常用的是 $4\times 4$的 **齐次变换矩阵 $^A_B\bm{T}$** 。

每个坐标系均由**一个原点**和**三条互相垂直的单位轴**构成。若设 $\{A\}$ 三轴为 $\hat{X}_A, \hat{Y}_A, \hat{Z}_A$，$\{B\}$ 三轴为 $\hat{X}_B, \hat{Y}_B, \hat{Z}_B$，$\{B\}$ 原点在 $\{A\}$ 中的位置为 $^A\bm p_B$，那么 $\{B\}$ 相对 $\{A\}$ 的位姿 $^A_B\bm{T}$ 由两部分刻画：

- **姿态 $^A_B\bm{R}$ **：$\{B\}$ 系的三个单位轴在 $\{A\}$ 中的坐标表示。
- **位置 $^A\bm p_B$ **：$\{B\}$ 系的原点在 $\{A\}$ 中的位置坐标。

将 $^A_B\bm{R}$ 与 $^A\bm p_B$ 合并表示，即**齐次变换矩阵** $^A_B\bm{T}$：

$$
^A_B\bm{T} = 
\begin{bmatrix} 
^A_B\bm{R}          & ^A\bm p_B \\ 
\bm{0}_{1\times3}   & 1 
\end{bmatrix} 
= \begin{bmatrix} 
\vert       & \vert         & \vert             & \vert \\ 
^A\hat{X}_B & ^A\hat{Y}_B   & ^A\hat{Z}_B       & ^A\bm p_B \\ 
\vert       & \vert         & \vert             & \vert \\ 
0           & 0             & 0                 & 1 
\end{bmatrix}
= \begin{bmatrix} 
r_{11}      & r_{12}        & r_{13}        & p_x \\ 
r_{21}      & r_{22}        & r_{23}        & p_y \\ 
r_{31}      & r_{32}        & r_{33}        & p_z \\ 
0           & 0             & 0             & 1 
\end{bmatrix}
$$

| 分块 | 大小 | 含义 | 对应物理量 |
|:---:|:---:|:---:|:---:|
| 左上 $^A_B\bm{R}$ | $3\times3$ | $\{B\}$ 系三单位轴列向量在 $\{A\}$ 系中的坐标 | **姿态** |
| 右上 $^A\bm p_B$ | $3\times1$ | $\{B\}$ 系原点在 $\{A\}$ 系中的坐标 | **位置** |
| 左下 | $1\times3$ 全 0 | 齐次形式占位 | —— |
| 右下 | 1 | 齐次坐标比例因子 | 固定为 1 |

> 💡 **为什么用 $4\times4$？** —— 把"旋转"和"平移"统一成**一次矩阵乘法**：三维坐标下平移是加法，而旋转是乘法，混算繁琐；$4\times4$ 只需矩阵相乘。

> 💡 旋转矩阵 $^A_B\bm{R}$ 有9个变量，但坐标系三轴均为单位轴（3个约束）且两两垂直（3个约束），故独立变量数为 9-3-3 = 3，这便是前面提到的姿态的三个维度。

#### 1.2.2 位姿矩阵的两个用途

**用途 1：坐标变换**——把点从一个系换算到另一个系。若点 $P$ 在 $\{B\}$ 中坐标为 $^B\!P$，那么在 $\{A\}$ 中：

$$
^A\!P = {}^A_B\bm{T}\;{}^B\!P
$$

**用途 2：位姿复合**——串联多个相对位姿。若 $\{B\}$ 相对 $\{A\}$ 为 $^A_B\bm{T}$，$\{C\}$ 相对 $\{B\}$ 为 $^B_C\bm{T}$，则 $\{C\}$ 相对 $\{A\}$：

$$
^A_C\bm{T} = {}^A_B\bm{T}\;{}^B_C\bm{T}
$$

这正是正运动学的灵魂：**末端系相对基座系的位姿 = 一串相邻连杆变换矩阵的连乘**。

> 📌 **记忆**：连乘时矩阵上下标"**斜对角相同，则可约去**" —— $^A_{\boxed{B}}\bm{T}\cdot{}^{\boxed{B}}_C\bm{T}= {}^A_C\bm{T}$。字母链对齐，便不会写错顺序。

#### 1.2.3 位姿矩阵的逆矩阵

位姿是相对的：知道"$\{B\}$ 相对 $\{A\}$"，便同时知道了"$\{A\}$ 相对 $\{B\}$"——后者即前者的**逆矩阵**。表达式：

$$
{}^B_A\bm{T} = (^A_B\bm{T})^{-1} =
\begin{bmatrix}
(^A_B\bm{R})^\top & -(^A_B\bm{R})^\top\,{}^A\bm p_B \\
\bm{0}_{1\times3} & 1
\end{bmatrix}
$$

**几何意义**（各分块与正向矩阵一一对应）：

| 分块 | 逆矩阵取值 | 含义 |
|:---:|:---:|:---:|
| 姿态 | ${}^B_A\bm{R} = (^A_B\bm{R})^\top$ | $\{A\}$ 系三单位轴在 $\{B\}$ 系中的坐标 |
| 位置 | ${}^B\bm p_A = -(^A_B\bm{R})^\top\,{}^A\bm p_B$ | $\{A\}$ 系原点在 $\{B\}$ 系中的坐标 |

> 💡 位置列**不是**简单取负 $-{}^A\bm p_B$——必须先左乘 $(^A_B\bm{R})^\top$ 把 $^A\bm p_B$ 换算到 $\{B\}$ 系再取负。

**用途 1：坐标反变换**——把点从 $\{A\}$ 系换算回 $\{B\}$ 系：

$$
{}^B\!P = {}^B_A\bm{T}\;{}^A\!P = (^A_B\bm{T})^{-1}\;{}^A\!P
$$

**用途 2：位姿链倒推**——从位姿复合式两端解出未知的中间环节。如已知 $\{B\}$、$\{C\}$ 各自相对 $\{A\}$ 的位姿，求 $\{C\}$ 相对 $\{B\}$：

$$
{}^B_C\bm{T} = (^A_B\bm{T})^{-1}\;{}^A_C\bm{T}
$$

> 💡 **程序实现**：直接套用上方公式，比通用求逆方法更快更稳。库方法：Pinocchio `T.inverse()`、Spatial Math `T.inv()`。

### 1.3 位姿变换可视化演示

用 Python 脚本演示位姿矩阵 $^A_B\bm{T}$ 如何同时驱动**坐标系**与**位置点**：全局参考系 $\{A\}$ 固定不动，固连系 $\{B\}$ 被 $^A_B\bm{T}$ 搬移，定义在 $\{B\}$ 中的点 $P$（$^B\!P=[x,y,z,1]^T$）也随之被换算到 $\{A\}$ 系中——坐标变换 $^A\!P={}^A_B\bm{T}\,{}^B\!P$ 的直观呈现。

| 元素 | 作用 | 可视化 |
|:---:|:---:|:---:|
| 固定参考系 $\{A\}$ | 观察基准，不移动 | 原点 + RGB虚线单位轴 |
| 固连系 $\{B\}$ | 被 $^A_B\bm{T}$ 驱动 | 原点 + RGB实线坐标轴 |
| 点 $P$ | 定义在 $\{B\}$ 下的点 | 黑色点标记 + 灰色虚线（$\{B\}$原点$\to P$）+ 橙色虚线（$\{A\}$原点$\to P$） |

界面操作：在右侧面板输入 $^A_B\bm{T}$ 与 $^B\!P$ 的 $x,y,z$，然后点击**开始计算**：

| 结果项 | 含义 |
|:---:|:---:|
| $P$ 相对 $B$ 位置 | 即输入的 $^B\!P$ |
| $P$ 相对 $A$ 位置 | $^A\!P={}^A_B\bm{T}\,{}^B\!P$ 的结果 |
| $B$ 相对 $A$ 位置 | $^A\bm p_B$，即矩阵右上位置列 |
| $B$ 相对 $A$ 姿态 | 矩阵左上 $3\times3$ 旋转块及换算得到的 RPY 角 与四元数（详见 §2） |

修改 $^A_B\bm{T}$ 各分量，可观察三类典型变换：

| 变换类型 | $^A_B\bm{T}$ 的修改部分 | 效果 |
|:---:|:---:|:---:|
| 纯平移 | 右上位置列 $(p_x,p_y,p_z)$ | $\{B\}$ 与 $P$ 一起平移、三色轴方向不变 |
| 纯旋转（绕单轴） | 左上 $3\times3$ 旋转块 | $\{B\}$ 原地转动、$P$ 随之绕 $\{B\}$ 原点转 |
| 复合变换 | 同时改位置与姿态 | $\{B\}$ 边移边转、$P$ 跟随到新位置 |

对应脚本位于 `joyarm_code/chapt/chapt2/chapt2_T_demo.py`，使用说明见脚本头部注释。

> 💡 建议先手改矩阵数字或点坐标，预测 $\{B\}$ 和 $P$ 在 $\{A\}$ 的位置，再点击"开始计算"核对结果，建立"矩阵数字"与"空间运动"的直觉；若左上 $3\times3$ 旋转矩阵不满足正交约束（三轴须单位长且两两垂直），程序会自动归一正交化并提示。

## 2 空间位姿其他常见表示及其转换关系

位姿矩阵 $\bm{T}$ 中的旋转矩阵 $\bm{R}$ 有 **9 个元素**（实际 3 个独立量），实际应用中（尤其涉及数据传输或存储时）常采用更紧凑的形式来表示姿态。本节介绍最经典常用的**RPY（与欧拉角）、轴角、四元数**三种。

### 2.1 欧拉角和固定角

任何姿态都可通过"绕三个坐标轴分别依次旋转"得到。旋转的方式有两种：绕原坐标系旋转（外旋）和绕旋转后的新坐标系旋转（内旋）。

| 类型 | 绕谁的轴 | 常见别称 |
|:---:|:---:|:---:|
| **内旋欧拉角** | 绕新**运动**坐标系 $\{B\}$ 的轴（每次转动后更新） | Intrinsic / Euler |
| **外旋固定角** | 绕原**固定**参考系 $\{A\}$ 的轴（始终不变） | Extrinsic / Fixed-angle / RPY |

#### 2.1.1 三个基本旋转矩阵

$$
\bm{R}_X(\gamma)=\begin{bmatrix}1&0&0\\0&c\gamma&-s\gamma\\0&s\gamma&c\gamma\end{bmatrix},\quad
\bm{R}_Y(\beta)=\begin{bmatrix}c\beta&0&s\beta\\0&1&0\\-s\beta&0&c\beta\end{bmatrix},\quad
\bm{R}_Z(\alpha)=\begin{bmatrix}c\alpha&-s\alpha&0\\s\alpha&c\alpha&0\\0&0&1\end{bmatrix}
$$

上式符号含义如下：

- $\bm{R}_X(\gamma),\bm{R}_Y(\beta),\bm{R}_Z(\alpha)$：分别绕 $\hat{X},\hat{Y},\hat{Z}$ 轴旋转的基本旋转矩阵，括号内的 $\gamma,\beta,\alpha$ 为对应的旋转角度；
- $c\cdot=\cos(\cdot)$、$s\cdot=\sin(\cdot)$：三角函数简写，如 $c\gamma=\cos\gamma$（下同）。

后面的复合旋转矩阵或姿态矩阵，便是这三个基本旋转矩阵按不同规则复合的结果。

#### 2.1.2 内旋欧拉角（Z-Y-X 顺序）

**内旋**指每一步都绕**当前（已经转动过的）坐标系 $\{B\}$** 的轴旋转。令 $\{B\}$ 初始时与 $\{A\}$ 重合，按 Z-Y-X 顺序旋转的过程：

| 步骤 | 绕哪根轴 | 转角 | 直观动作 |
|:---:|:---:|:---:|:---:|
| 1 | 当前 $\hat{Z}_B$（重合于 $\hat{Z}_A$） | $\alpha$ | 偏航 Yaw：如人的左右扭头 |
| 2 | 转动后的 $\hat{Y}_B$ | $\beta$ | 俯仰 Pitch：如人的上下点头 |
| 3 | 再转动后的 $\hat{X}_B$ | $\gamma$ | 横滚 Roll：如人的左右歪头 |

关键点：**每绕"运动轴"旋转一次，矩阵从右侧乘入（B系侧）**。因此三次旋转连乘为：

$$
^A_B\bm{R}_{内旋ZYX}(\alpha,\beta,\gamma)=\bm{R}_Z(\alpha)\,\bm{R}_Y(\beta)\,\bm{R}_X(\gamma)
$$

> 💡 直观理解：设 ${}^A_B R = {}^0_1 R \cdot {}^1_2 R \cdot {}^2_3 R $，初始时三矩阵为单位阵（A系B系重合），第一次旋转后 ${}^0_1 R$ 不再为单位阵，然后是 ${}^1_2 R$ 和 ${}^2_3 R$ 不再为单位阵。

#### 2.1.3 外旋固定角（X-Y-Z，即 RPY 角）

**外旋**指每一步都绕**固定参考系 $\{A\}$** 的轴旋转，轴的方向始终不变。令 $\{B\}$ 初始时与 $\{A\}$ 重合，按 X-Y-Z 顺序旋转的过程：

| 步骤 | 绕哪根轴 | 转角 | 直观动作 |
|:---:|:---:|:---:|:---:|
| 1 | 固定系 $\hat{X}_A$ | $\gamma$ | 横滚 Roll |
| 2 | 固定系 $\hat{Y}_A$ | $\beta$ | 俯仰 Pitch |
| 3 | 固定系 $\hat{Z}_A$ | $\alpha$ | 偏航 Yaw |

这就是常说的 **RPY（Roll-Pitch-Yaw）角**。

关键点：**每绕"固定轴"转一次，矩阵从左侧乘入（A系侧）**。因此三次旋转连乘为：

$$
^A_B\bm{R}_{外旋XYZ}(\gamma,\beta,\alpha)=\bm{R}_Z(\alpha)\,\bm{R}_Y(\beta)\,\bm{R}_X(\gamma)
$$

> 💡 直观理解：设 $ {}^A_B R = {}^0_1 R \cdot {}^1_2 R \cdot {}^2_3 R $ ，初始时三矩阵为单位阵（A系B系重合），第一次旋转后 ${}^2_3 R$ 不再为单位阵，然后是 ${}^1_2 R$ 和 ${}^0_1 R$ 不再为单位阵。

#### 2.1.4 为何两者表达式相同

原因在于矩阵相乘时"内旋后乘"与"外旋前乘"互为对偶。从绕轴、乘法、连乘顺序、数学表达、直观理解上逐一对比：

| 对比维度 | **内旋**（欧拉角） | **外旋**（固定角 / RPY） |
|:---:|:---:|:---:|
| 绕哪根轴 | 绕**运动**系 $\{B\}$ 自身轴（每次转动后更新） | 绕**固定**系 $\{A\}$ 的轴（始终不变） |
| 矩阵乘法 | **后乘**（右乘） | **前乘**（左乘） |
| 连乘顺序 vs 转动顺序 | **相同**（先转的矩阵排最左） | **相反**（后转的矩阵排最左） |
| 表达式 | $\bm{R}_Z(\alpha)\bm{R}_Y(\beta)\bm{R}_X(\gamma)$ | $\bm{R}_Z(\alpha)\bm{R}_Y(\beta)\bm{R}_X(\gamma)$ |
| 直观理解 | 先转**肩**、再转**肘**、最后转**腕** | 先转**腕**、再转**肘**、最后转**肩** |

由此可见"内旋 Z-Y-X"的连乘与"外旋 X-Y-Z"的反序前乘得到的结果恰好相等——**先肩后肘再腕**与**先腕后肘再肩**虽然动作顺序相反，但最终结果完全一致。

#### 2.1.5 万向节死锁问题

$$
^A_B\bm{R}=\bm{R}_Z(\alpha)\bm{R}_Y(\beta)\bm{R}_X(\gamma)=\begin{bmatrix}c\alpha\,c\beta & c\alpha\,s\beta\,s\gamma-s\alpha\,c\gamma & c\alpha\,s\beta\,c\gamma+s\alpha\,s\gamma\\ s\alpha\,c\beta & s\alpha\,s\beta\,s\gamma+c\alpha\,c\gamma & s\alpha\,s\beta\,c\gamma-c\alpha\,s\gamma\\ -s\beta & c\beta\,s\gamma & c\beta\,c\gamma\end{bmatrix}
$$

当俯仰角 $\beta=\pm90°$ 时，$c\beta=0$、$s\beta=\pm1$，代入化简后旋转矩阵

$$
\beta=+90°:\quad {}^A_B\bm{R}=\begin{bmatrix}0&-s(\alpha-\gamma)&c(\alpha-\gamma)\\0&c(\alpha-\gamma)&s(\alpha-\gamma)\\-1&0&0\end{bmatrix}
$$

$$
\beta=-90°:\quad {}^A_B\bm{R}=\begin{bmatrix}0&-s(\alpha+\gamma)&-c(\alpha+\gamma)\\0&c(\alpha+\gamma)&-s(\alpha+\gamma)\\1&0&0\end{bmatrix}
$$

只依赖 $(\alpha\mp\gamma)$ ，这表明 Roll（$\gamma$）与 Yaw（$\alpha$）的作用相当，两者的轴在**同一直线**上、总旋转效果唯一但无法具体区分二者。三个旋转轴中有两个共线，**丢失了一个自由度**。

> ⚠️ **万向锁**：当 $\beta=\pm90°$ 时，Roll 与 Yaw 退化为同轴，姿态不再由三个独立角度决定（逆解不再唯一）。这是**所有三参数姿态表示**（含 RPY 与各种欧拉角）的通病，也是工程上引入轴角（[2.2](#22)）与四元数（[2.3](#23)）的重要动因之一。

### 2.2 轴角表示

**欧拉旋转定理**：三维空间任何姿态变化都可通过**绕某轴 $\hat{k}$ 旋转角度 $\theta$** 得到。于是姿态可用单位轴 $\hat{k}=(k_x,k_y,k_z)$ + 转角 $\theta$ 来描述（由于 $\|\hat{k}\|=1$ ，实际也是 3 个独立变量）。

由轴角求旋转矩阵（罗德里格斯公式）：

$$
\bm{R}(\hat{k},\theta)=\cos\theta\,\bm{I}+(1-\cos\theta)\,\hat{k}\hat{k}^\top+\sin\theta\,\bm{K},\quad
\bm{K}=\begin{bmatrix}0&-k_z&k_y\\k_z&0&-k_x\\-k_y&k_x&0\end{bmatrix}
$$

### 2.3 四元数

四元数是轴角的"合体版"，用四维向量 $\bm{q}=(w,x,y,z)$ 描述姿态：

$$
\bm{q}=\cos\frac{\theta}{2}+\sin\frac{\theta}{2}\,(k_x\hat{i}+k_y\hat{j}+k_z\hat{k}) = \bigl( w,x,y,z \bigr)
$$

$w=\cos(\theta/2)$ 为实部，$(x,y,z)=\sin(\theta/2)\hat{k}$ 为虚部，满足 $w^2+x^2+y^2+z^2=1$ （故独立变量个数也为 3）。由四元数求旋转矩阵：

$$
\bm{R}(\bm{q})=\begin{bmatrix}
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy)\\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx)\\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{bmatrix}
$$

### 2.4 四种姿态表示法对比

四种姿态表示的横向对比如下：

| 表示法 | 参数个数 | 优点 | 缺点 | 典型用途 |
|:---:|:---:|:---:|:---:|:---:|
| 旋转矩阵 $\bm{R}$ | 9（独立 3） | 无奇异、计算直接（矩阵乘）、所有方法的"通用语言" | 冗余、不直观、存储/传输开销大 | 内部计算、作为转换枢纽 |
| 欧拉角与固定角 | 3 | 直观、符合直觉 | **万向锁**（Pitch=$\pm90°$）；存在多解 | 上位机示教、参数显示 |
| 轴角 $(\hat{k},\theta)$ | 4（独立 3） | 直观、无万向锁、转角可直接读出 | $\theta\to0$ 时轴 $\hat{k}$ 不确定；非唯一（$\theta\leftrightarrow2\pi-\theta$、$\hat{k}\leftrightarrow-\hat{k}$） | 单次旋转、关节运动、插值中间量 |
| 四元数 $\bm{q}$ | 4（独立 3） | 无万向锁、可平滑插值（SLERP）、数值稳定 | 不直观；$\pm\bm{q}$ 双覆盖 | 渲染引擎、姿态滤波、轨迹插值 |

> 💡 **总结**：RPY 直观但存在万向锁现象，轴角紧凑但零旋转奇异，四元数稳健但抽象，$\bm{R}$ 矩阵冗余但通用。上位机显示常用 RPY，内部计算/插值常用四元数，多表示方法的转换则一律通过 $\bm{R}$ 中转。

### 2.5 四种姿态表示法的转换

设 $\bm{R}=\begin{bmatrix}r_{11}&r_{12}&r_{13}\\r_{21}&r_{22}&r_{23}\\r_{31}&r_{32}&r_{33}\end{bmatrix}$、$\mathrm{tr}(\bm{R})=r_{11}+r_{22}+r_{33}$，则前面四种姿态表示方法间的相互转换关系为：

| 转换前的表示 | 转换后的表示 | 转换公式 | Pinocchio（`pinocchio`） | Spatial Math（`spatialmath`） | 注意点 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\bm{R}$ | ZYX欧拉角和RPY | $\beta=\arctan2(-r_{31},\sqrt{r_{11}^2+r_{21}^2})$；$\alpha=\arctan2(r_{21},r_{11})$；$\gamma=\arctan2(r_{32},r_{33})$ | `pinocchio.rpy.matrixToRpy(R)` | `rmat.rpy()` | **$\beta\ne\pm90°$**，否则万向锁 |
| $\bm{R}$ | 轴角 | $\theta=\arccos\!\frac{\mathrm{tr}(\bm{R})-1}{2}$；$\hat{k}=\frac{1}{2\sin\theta}\begin{bmatrix}r_{32}-r_{23}\\r_{13}-r_{31}\\r_{21}-r_{12}\end{bmatrix}$ | `pinocchio.log3(R)` | `rmat.angvec()` | **$\theta\ne0,\pi$**，轴不确定 |
| $\bm{R}$ | 四元数 | $w=\tfrac{1}{2}\sqrt{1+\mathrm{tr}(\bm{R})}$；$x=\frac{r_{32}-r_{23}}{4w}$；$y=\frac{r_{13}-r_{31}}{4w}$；$z=\frac{r_{21}-r_{12}}{4w}$ | `pinocchio.Quaternion(R).coeffs()` | `UnitQuaternion(rmat)` | **$\mathrm{tr}(\bm{R})>-1$**（即 $w\ne0$） |
| ZYX欧拉角和RPY | $\bm{R}$ | $\bm{R}=\bm{R}_Z(\alpha)\bm{R}_Y(\beta)\bm{R}_X(\gamma)$ | `pinocchio.rpy.rpyToMatrix(rpy)` | `SE3.RPY(rpy, order='zyx')` | 见 [2.1](#21) |
| 轴角 | $\bm{R}$ | 罗德里格斯公式 $\bm{R}=\cos\theta\,\bm{I}+(1-\cos\theta)\hat{k}\hat{k}^\top+\sin\theta\,\bm{K}$，$\bm{K}=\begin{bmatrix}0&-k_z&k_y\\k_z&0&-k_x\\-k_y&k_x&0\end{bmatrix}$ | `pinocchio.exp3(theta*k)` | `SE3.AngVec(theta, k)` | 见 [2.2](#22) |
| 四元数 | $\bm{R}$ | $\bm{R}(\bm{q})=\begin{bmatrix}1-2(y^2+z^2)&2(xy-wz)&2(xz+wy)\\2(xy+wz)&1-2(x^2+z^2)&2(yz-wx)\\2(xz-wy)&2(yz+wx)&1-2(x^2+y^2)\end{bmatrix}$ | `pinocchio.Quaternion(q).toRotationMatrix()` | `SE3(UnitQuaternion(q))` | 见 [2.3](#23) |
| 轴角 | 四元数 | $\bm{q}=\bigl(\cos\tfrac{\theta}{2},\;\sin\tfrac{\theta}{2}\hat{k}\bigr)$ | `pinocchio.Quaternion(pin.AngleAxis(theta, k))` | `UnitQuaternion.AngVec(theta, k)` | 无奇异，最常用直接法 |
| 四元数 | 轴角 | $\theta=2\arccos w$；$\hat{k}=\frac{(x,y,z)}{\sqrt{x^2+y^2+z^2}}=\frac{(x,y,z)}{\sin(\theta/2)}$ | —（由四元数分量直接算） | `q.angvec()` | $w=\pm1$（$\theta\to0$）时轴不定 |

> 📌 **表中未列出的其余转换经 $\bm{R}$ 中转**：RPY↔轴角、RPY↔四元数，"先转为 $\bm{R}$、再转为目标形式"两步求解。

> 💡 **选库建议**：侧重运动学分析与教学用 `spatialmath`（语义清晰）；侧重动力学与高效仿真用 `pinocchio`（与 URDF 深度集成，考虑到后续实际开发，本教程主要用`pinocchio`）。

### 2.6 四种姿态表示转换的可视化演示

用 Python 脚本演示四种姿态表示如何描述**同一个旋转**：全局参考系 $\{A\}$ 固定不动，系 $\{B\}$ 被驱转动（纯姿态）。右侧数值面板修改其中一个，点击**开始计算**后便立刻刷新其余三个面板数据与左侧 3D 可视化。

| 元素 | 作用 | 可视化 |
|:---:|:---:|:---:|
| 固定参考系 $\{A\}$ | 观察基准，不移动 | 原点 + RGB 虚线单位轴 |
| 固连系 $\{B\}$ | 被 $^A_B\bm{R}$ 驱动，绕原点旋转 | 原点 + RGB 实线坐标轴 |

四个输入面板（角度均为度，四元数取 $(w,x,y,z)$）**开始计算**后的规范化处理如下：

| 面板 | 输入形式 | 规范化处理 |
|:---:|:---:|:---:|
| RPY 角 | $r_x,r_y,r_z$ 三个数值框（°） | 显示规范解 |
| 轴角 | $\hat{k}=(k_x,k_y,k_z)$ 任意非零向量 + $\theta$（°） | 归一化单位轴与 $[0°, 180°]$ 内转角 |
| 四元数 | $w,x,y,z$ 四个数值框 | 归一化单位四元数 |
| 旋转矩阵 $\bm{R}$ | $3\times3$ 可编辑表格 | 不正交时则正交化 |

万向锁：点**万向锁示例**填入 $(30°, 90°, 60°)$ 计算，再把 RPY 改为 $(0°, 90°, 30°)$ 计算，两组 RPY 角度不同但**姿态相同**（其余三组表示形式的结果不变）。

对应脚本位于 `joyarm_code/chapt/chapt2/chapt2_R_demo.py`，使用说明见脚本头部注释。

> 💡 建议先在一个面板手改数字、预测其余三种表示的计算结果，再点"开始计算"核对结果。

## 3 JoyArm 连杆建系与MDH参数

本节采用 **改进 D-H（Modified Denavit-Hartenberg，MDH）参数法**把机械臂构型关系转化为 MDH 参数表：为每个连杆建立坐标系，并用 4 个参数描述相邻连杆系间的位姿关系。

### 3.1 连杆建系步骤

#### 3.1.1 关节与连杆的编号

![图片占位：JoyArm 关节与连杆标注实物图]()

以上述 JoyArm 机械臂为例，该机械臂为 **6 DOF 串联型旋转关节机械臂**（6R）：6 个旋转关节 ，相邻两关节间的刚性杆件为连杆，关节 $i$ 驱动着与之固连的连杆 $i$。

#### 3.1.2 MDH 建系步骤

**步骤 1**：机械臂摆放至零位构型，标出各关节轴及其延长线，自基座向末端编号依次为 1 至 n。关节轴 $i$ 与 $i+1$ 间的连杆为连杆 $x$。

**步骤 2**：找出两相邻关节轴 $\{i\}$ 和 $\{i+1\}$ 的公垂线（若两者平行，通常取符合直觉、结构对称且便于计算的公垂线）或关节轴 $\{i\}$ 和 $\{i+1\}$ 的交点，以公垂线与关节轴 $\{i\}$ 的交点或关节轴 $\{i\}$ 和 $\{i+1\}$ 的交点作为连杆系 $\{i\}$ 的原点。

**步骤 3**：规定连杆系 $\{i\}$ 的 $\hat{Z}_i$ 轴沿关节轴线 $\{i\}$ 的方向。

**步骤 4**：规定连杆系 $\{i\}$ 的 $\hat{X}_i$ 轴沿关节轴 $\{i\}$ 和 $\{i+1\}$ 的公垂线方向并指向轴 $\{i+1\}$ ；若轴 $\{i\}$ 和 $\{i+1\}$ 相交，则取它们的公垂线方向。

**步骤 5**：在 $\hat{X}_i$ 与 $\hat{Z}_i$ 已确定的基础上，按右手系规则确定出 $\hat{Y}_i$ 轴。

**步骤 6**：基座系 $\{0\}$ 通常与零位位姿下的连杆系 $\{1\}$ 重合；也可沿其 $\hat{Z}_1$ 轴进行平移，使基座系 $\{0\}$ 原点位于基座的安装底面。

**步骤 7**：末端工具系 $\{e\}$（可选）固连于末端执行器，其原点取工具中心点（TCP），三轴朝向按依具体工具朝向选取，末端工具系 $\{e\}$ 可通过固定偏移 $^n_E\bm{T}$ 与 末连杆系 $\{n\}$ 相接。

**步骤 8**：末连杆系 $\{n\}$ 的原点和 $\hat{X}_n$ 方向可任意选取，但通常会使连杆系 $\{n\}$ 原点与连杆系 $\{n-1\}$ 原点重合，且 $\hat{X}_n$ 方向通常指向末端工具系 $\{e\}$。

#### 3.1.3 JoyArm 的连杆系

遵循上述建系规则对 JoyArm 机械臂建系，得到下图的连杆系、基座系和工具系。图中标注了各系的 $Z$ 轴（关节转轴）与 $X$ 轴（公垂线），是推导 MDH 参数表的依据。

![图片占位：JoyArm 连杆坐标系 1-6 标注图（各关节轴、各连杆系、末端系、基座系）]()

> 📌 建系结果**不唯一**（$Z$ 轴正负、$X$ 轴方向有多种合法选择），但最后计算得到的位姿 $^0_n\bm{T}$ 或 $^0_E \bm{T}$ 应当是唯一的。

### 3.2 MDH 参数表

#### 3.2.1 四个参数的几何含义

![图片占位：MDH 连杆参数示意图]()

相邻连杆系 $\{i-1\}$ 与 $\{i\}$ 的相对位姿由以下 4 个参数完全描述：

| 参数 | 符号 | 几何含义 | 物理对应 |
|:---:|:---:|:---:|:---:|
| 连杆扭角 | $\alpha_{i-1}$ | $\hat{Z}_{i-1}$ 绕 $\hat{X}_{i-1}$ 转到 $\hat{Z}_i$ 所需的角度 | 相邻关节轴线的**夹角**（连杆扭转角） |
| 连杆长度 | $a_{i-1}$ | $\hat{Z}_{i-1}$ 与 $\hat{Z}_i$ 的公垂线长度 | 相邻关节轴线的**最短距离**（连杆长度） |
| 连杆偏距 | $d_i$ | $\hat{X}_{i-1}$ 沿 $\hat{Z}_i$ 平移到 $\hat{X}_i$ 的长度 | 关节轴上的**偏移量**（转动关节为常数） |
| 关节角 | $\theta_i$ | $\hat{X}_{i-1}$ 绕 $\hat{Z}_i$ 转到 $\hat{X}_i$ 的角度 | 关节**转角**（转动关节的变量） |

> 💡 **记忆**：$ \alpha_{i-1},a_{i-1}$ 描述**连杆几何**（常数）；$d_i, \theta_i$ 描述**关节运动**。对于全旋转关节机械臂，**$\theta_i$ 为变量**，其余三参数为常量。

#### 3.2.2 JoyArm 的 MDH 参数表

| 序号 $i$ | $\alpha_{i-1}(rad)$ | $a_{i-1}(m)$ | $d_i(m)$ | $\theta_i(rad)$ | 范围 $(rad)$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.0 | 0.0 | 0.1402 | $\theta_1$ | -2.8 ~ 2.8 |
| 2 | $\pi/2$ | 0.02 | 0.0 | $\theta_2+\pi$ | -$\pi$ ~ 0.0 |
| 3 | -$\pi$ | 0.264 | 0.0 | $\theta_3+2.9226$ | -$\pi$ ~ 0.0 |
| 4 | 0.0 | 0.2485 | 0.0 | $\theta_4+0.2189$ | -$\pi/2$ ~ $\pi/2$ |
| 5 | $-\pi/2$ | 0.07835 | 0.0025 | $\theta_5+\pi/2$ | $-\pi/2$ ~ $\pi/2$ |
| 6 | $\pi/2$ | 0.0 | 0.0 | $\theta_6$ | $-\pi/2$ ~ $\pi/2$ |

末端工具系 $\{e\}$ 相对末连杆系 $\{n\}$ 的位姿

$$
{}^n_e\bm{T} = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & p_x \\
r_{21} & r_{22} & r_{23} & p_y \\
r_{31} & r_{32} & r_{33} & p_z \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

## 4 JoyArm 位置正运动学求解

得到 MDH 参数表后，求解机械臂位置正运动学（Forward Kinematics, FK）的过程就是：**分别写出各相邻连杆系间的位姿矩阵，然后从基座系连乘到末端系（或工具系）**。

### 4.1 相邻连杆系间的位姿矩阵

#### 4.1.1 位姿矩阵通用表达式

$\{i\}$ 相对 $\{i-1\}$ 的变换矩阵 $^{i-1}_i\bm{T}$ 由四个参数按如下顺序复合：

$$
^{i-1}_i\bm{T}=\bm{R}_x(\alpha_{i-1})\,\bm{D}_x(a_{i-1})\,\bm{R}_z(\theta_i)\,\bm{D}_z(d_i)
$$

**$R$** 为 $4 \times 4$ 旋转算子，**$D$** 为 $4 \times 4$ 平移算子。展开后：

$$
^{i-1}_i\bm{T}=\begin{bmatrix}
c\theta_i & -s\theta_i & 0 & a_{i-1}\\
s\theta_i\,c\alpha_{i-1} & c\theta_i\,c\alpha_{i-1} & -s\alpha_{i-1} & -s\alpha_{i-1}\,d_i\\
s\theta_i\,s\alpha_{i-1} & c\theta_i\,s\alpha_{i-1} & c\alpha_{i-1} & c\alpha_{i-1}\,d_i\\
0 & 0 & 0 & 1
\end{bmatrix}
$$

上式中， $c\cdot=\cos(\cdot)$、$s\cdot=\sin(\cdot)$ 。

#### 4.1.2 JoyArm 各相邻连杆位姿矩阵

将 MDH 参数表按行代入上式，得：

$$
^0_1\bm{T}=\begin{bmatrix}
c_1 & -s_1 & 0 & a_0\\
s_1 c_{\alpha_0} & c_1 c_{\alpha_0} & -s_{\alpha_0} & -s_{\alpha_0} d_1\\
s_1 s_{\alpha_0} & c_1 s_{\alpha_0} & c_{\alpha_0} & c_{\alpha_0} d_1\\
0 & 0 & 0 & 1
\end{bmatrix},\quad c_1=\cos\theta_1,\;s_1=\sin\theta_1
$$

$$
^1_2\bm{T}=\begin{bmatrix}
c_2 & -s_2 & 0 & a_1\\
s_2 c_{\alpha_1} & c_2 c_{\alpha_1} & -s_{\alpha_1} & -s_{\alpha_1} d_2\\
s_2 s_{\alpha_1} & c_2 s_{\alpha_1} & c_{\alpha_1} & c_{\alpha_1} d_2\\
0 & 0 & 0 & 1
\end{bmatrix},\quad c_2=\cos\theta_2,\;s_2=\sin\theta_2
$$

$$
^2_3\bm{T}=\begin{bmatrix}
c_3 & -s_3 & 0 & a_2\\
s_3 c_{\alpha_2} & c_3 c_{\alpha_2} & -s_{\alpha_2} & -s_{\alpha_2} d_3\\
s_3 s_{\alpha_2} & c_3 s_{\alpha_2} & c_{\alpha_2} & c_{\alpha_2} d_3\\
0 & 0 & 0 & 1
\end{bmatrix},\quad c_3=\cos\theta_3,\;s_3=\sin\theta_3
$$

$$
^3_4\bm{T}=\begin{bmatrix}
c_4 & -s_4 & 0 & a_3\\
s_4 c_{\alpha_3} & c_4 c_{\alpha_3} & -s_{\alpha_3} & -s_{\alpha_3} d_4\\
s_4 s_{\alpha_3} & c_4 s_{\alpha_3} & c_{\alpha_3} & c_{\alpha_3} d_4\\
0 & 0 & 0 & 1
\end{bmatrix},\quad c_4=\cos\theta_4,\;s_4=\sin\theta_4
$$

$$
^4_5\bm{T}=\begin{bmatrix}
c_5 & -s_5 & 0 & a_4\\
s_5 c_{\alpha_4} & c_5 c_{\alpha_4} & -s_{\alpha_4} & -s_{\alpha_4} d_5\\
s_5 s_{\alpha_4} & c_5 s_{\alpha_4} & c_{\alpha_4} & c_{\alpha_4} d_5\\
0 & 0 & 0 & 1
\end{bmatrix},\quad c_5=\cos\theta_5,\;s_5=\sin\theta_5
$$

$$
^5_6\bm{T}=\begin{bmatrix}
c_6 & -s_6 & 0 & a_5\\
s_6 c_{\alpha_5} & c_6 c_{\alpha_5} & -s_{\alpha_5} & -s_{\alpha_5} d_6\\
s_6 s_{\alpha_5} & c_6 s_{\alpha_5} & c_{\alpha_5} & c_{\alpha_5} d_6\\
0 & 0 & 0 & 1
\end{bmatrix},\quad c_6=\cos\theta_6,\;s_6=\sin\theta_6
$$

### 4.2 完整正运动学

根据位姿复合的连乘规则，末端工具系 $\{e\}$ 相对基座系 $\{0\}$ 的位姿矩阵：

$$
^0_e\bm{T}(\theta_1,\cdots,\theta_6)=^0_1\bm{T}(\theta_1)\cdots{ }^{n-1}_n\bm{T}(\theta_n)\cdot^n_e\bm{T}
$$

上式即为机械臂的位置正运动学方程**：给定各关节角，解得末端位姿 $^0_e\bm{T}$。

### 4.3 URDF + 库的正运动学求解方法

在实际项目和应用中，手写推算 $\bm{T}$ 既繁琐又易错，且较后者不够通用。更易用且现代做法是：将机器人CAD模型导出为统一机器人描述格式（URDF），程序中**用 URDF 描述机器人几何，由机器人学库自动建系、求解**。当我们理解并掌握了位姿变换和正运动学的基本原理后，实际运用过程中完全可以转向**“URDF + 机器人学计算库”**的组合。

#### 4.3.1 URDF：机械臂的"数字孪生"

**URDF（Unified Robot Description Format，统一机器人描述格式）** 是 XML 格式的机器人数字模型，用 link + joint 描述：

| URDF 元素 | 描述内容 | 对应物理量 |
|:---:|:---:|:---:|
| `<link>` | 连杆的视觉、碰撞、惯性属性 | 连杆几何、质量、惯量 |
| `<joint>` | 相邻连杆的连接关系 | 关节类型（revolute/fixed）、轴向量、零位偏移 |

如，机械臂基座、关节 1 和连杆 1 的 URDF 示例如下：

```xml
<link name="link0">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="../meshes/link0.STL"/>
    </geometry>
  </visual>
</link>

<joint name="joint1" type="revolute">
  <origin xyz="0 0 0.1402" rpy="0 0 0"/>
  <parent link="link0"/>
  <child  link="link1"/>
  <axis xyz="0 0 1"/>
  <limit lower="-2.8" upper="2.8" effort="30" velocity="5"/>
</joint>

<link name="link1">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="../meshes/link1.STL"/>
    </geometry>
  </visual>
</link>
```

URDF 使用"父子连杆 + 关节偏移"的树状结构描述机器人几何结构，**不依赖特定 D-H 约定，比 MDH 法更通用、更贴近结构设计 CAD**。JoyArm 的 URDF 便是基于[第七章](chapt7_urdf.md)的方法由 CAD 装配体导出的。

JoyArm 机械臂 URDF 在 [URDF Studio](https://urdf.d-robotics.cc/) （也可使用 vscode 中的URDF可视化插件进行预览）的可视化结果如下

![图片占位：URDF Studio 中的 JoyArm URDF可视化]()

URDF 几乎兼容所有的主流仿真器和机器人学计算库，它可以导入仿真器作为仿真的数字模型，也可加载进机器人学计算库进行机器人的规划和控制。

#### 4.3.2 用库自动求解正运动学

以 **URDF + Pinocchio** 计算正运动学，核心流程示例：

```python
import numpy as np
import pinocchio as pin

# 1. 从 URDF 加载模型（含运动链与关节限位）
model = pin.buildModelFromUrdf("joyarm.urdf") # 模型（不可变数据）
data  = model.createData() # 可变数据

# 2. 给定关节角（rad）
q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 

# 3. 求解末端（"frame_end_tcp" 帧，即末端 TCP 系）相对基座的位姿
pin.forwardKinematics(model, data, q) # 计算正运动学
T_0e = pin.updateFramePlacement(model, data, model.getFrameId("frame_end_tcp")) # 获取结果
```

## 5 本章实践：正运动学求解器实现与工作空间绘制分析

本节把第 4 节的两种正运动学解法落地为程序：先在核心库中封装成可切换的求解器并学会调用（5.1~5.2），再用蒙特卡洛采样绘制并分析 JoyArm 的工作空间（5.3~5.5），回答"目标点够不够得到"。全程离线计算，无需连接机械臂。

### 5.1 正运动学求解器的算法实现

核心库把正运动学求解封装在 robotics/fkine 里——一个基类规定接口，多个子类各自实现算法；本章实现两个子类，分别对应 §4.2 的 MDH 连乘与 §4.3 的 URDF + Pinocchio。

#### 5.1.1 求解器接口：基类 FkineSolver

基类只规定一个方法 `solve(arm, q, frame) → Pose`——**给定关节角，求某一帧的位姿**。输入与输出如下：

| 输入 / 输出 | 名称 | 说明 |
|:---:|:---:|:---:|
| 输入 | `arm` | 机械臂对象（提供配置与 URDF 模型） |
| 输入 | `q` | `(n,)` 关节角，rad |
| 输入 | `frame` | 目标帧名，如 `"frame_end_tcp"`（末端 TCP 系） |
| 返回 | `Pose` | `position`：`(3,)` 位置；`orientation`：`(4,)` 四元数 $(w,x,y,z)$；`.T`：$4\times4$ 位姿矩阵 |

基类写成代码只需几行：

```python
from abc import ABC, abstractmethod

class FkineSolver(ABC):
    """正运动学求解器接口：所有子类只须实现一个 solve 方法。"""

    @abstractmethod
    def solve(self, arm, q, frame):
        """给定关节角 q（rad），求 frame 帧在基座系下的位姿（Pose）。"""
```

#### 5.1.2 MDH 版求解器

MDH 版是 4.2 节连乘公式的直接翻译，共五步：

| 步骤 | 操作 | 得到 |
|:---:|:---:|:---:|
| 1 | 读 MDH 参数表与末端偏移 | config 的 `arm_mdh`（$n\times4$）与 `T_linkn_endtcp` |
| 2 | 逐行代入 [4.1.1 节](#411) 展开式 | 相邻连杆位姿矩阵 $^{i-1}_i\bm{T}$ |
| 3 | 按 [4.2 节](#42) 自基座连乘 | $^0_n\bm{T}$ |
| 4 | 右乘末端工具系偏移 | $^0_e\bm{T}={}^0_n\bm{T}\,{}^n_e\bm{T}$ |
| 5 | 转为 `Pose` 返回 | 位置 + 四元数 |

精简实现（一个求单节矩阵的小函数 + 求解器类）：

```python
import numpy as np
from joyarm_core import Pose

def mdh_transform(row, q):
    """MDH 参数表一行 [α, a, d, θ0] + 关节角 q → 相邻连杆位姿矩阵（4.1.1 节展开式）。"""
    alpha, a, d, t0 = row
    sa, ca, st, ct = np.sin(alpha), np.cos(alpha), np.sin(t0 + q), np.cos(t0 + q)
    return np.array([[ct, -st, 0, a],
                     [st * ca, ct * ca, -sa, -sa * d],
                     [st * sa, ct * sa,  ca,  ca * d],
                     [0, 0, 0, 1]])

class MdhFkineSolver(FkineSolver):
    """MDH 连乘版：逐行求相邻矩阵、自基座连乘，再右乘末端偏移。"""

    def solve(self, arm, q, frame):
        cfg = arm.get_config()["joyarm"]                    # arm_mdh 参数表 + 末端偏移
        T = np.eye(4)
        for row, qi in zip(cfg["arm_mdh"], q):              # ② 逐行代入 → ③ 连乘
            T = T @ mdh_transform(row, qi)
        return Pose.from_T(T @ np.asarray(cfg["T_linkn_endtcp"]))  # ④ 右乘 → ⑤ 转 Pose
```

完整实现位于 `joyarm_code/joyarm_core/robotics/fkine/fkine_solver_mdh.py`，在上述逻辑之外另含参数缓存与维度、帧检查（仅支持末端帧）。

#### 5.1.3 Pinocchio 版求解器

Pin 版与 [4.3.2 节](#432) 同理：`forwardKinematics` 沿运动链逐关节传递位姿，`updateFramePlacement` 取出目标帧结果：

```python
import numpy as np
import pinocchio as pin
from joyarm_core import Pose

class PinFkineSolver(FkineSolver):
    """Pinocchio 版：沿运动链逐关节传递位姿，取出目标帧结果。"""

    def solve(self, arm, q, frame):
        model, data = arm.pin_model, arm.pin_model.createData()
        pin.forwardKinematics(model, data, np.asarray(q, dtype=float))    # 正运动学
        return Pose.from_T(pin.updateFramePlacement(                      # 取目标帧位姿
            model, data, model.getFrameId(frame)).homogeneous)
```

完整实现位于 `joyarm_code/joyarm_core/robotics/fkine/fkine_solver_pin.py`，含帧名合法性检查，支持模型内任意帧。

#### 5.1.4 注册与配置

两个子类写好后，在包的注册表 `REGISTRY` 里各登记一行（注册名 = 类名小写 + 下划线），config 再按名单加载：

```python
REGISTRY = {
    "pin_fkine_solver": PinFkineSolver,  # 默认实现：URDF 通用
    "mdh_fkine_solver": MdhFkineSolver,  # MDH 解析实现
}
```

```yaml
robotics:
  fkine:
    - { name: pin_fkine_solver }  # 列表首位 = 默认激活
    - { name: mdh_fkine_solver }
```

两版求解器对比如下：

| 对比项 | MDH 版 | Pinocchio 版 |
|:---:|:---:|:---:|
| 数据来源 | config 的 MDH 参数表 | URDF 模型 |
| 支持的帧 | 仅末端帧 | 任意帧（`link3`、`frame_end_tcp` 等） |
| 适用场景 | 教学演示、与手推公式对照 | 通用计算（默认激活） |
| 特点 | 公式白盒、过程可见 | 高效通用、无需维护公式 |

### 5.2 正运动学求解器的调用

创建机械臂对象时按 config 加载全部 fkine 求解器，`arm.solve_fkine` 自动转给当前激活的那个。离线调用示例（无需连接硬件）：

```python
import numpy as np
from joyarm_core import joyArm

arm = joyArm("rebot_arm_b601_dm")               # 离线创建对象，不连接硬件
print(arm.get_fkine_solvers())                  # ['mdh_fkine_solver', 'pin_fkine_solver']
print(arm.get_activated_fkine_solver())         # pin_fkine_solver（config 名单首位）
q = np.array([0.0, -1.0, -1.0, 0.3, 0.0, 0.0])  # home 位形（config 的 home_arm）
pose = arm.solve_fkine(q)                       # frame 缺省为末端帧 frame_end_tcp
print(pose.position)                            # [ 0.3447 -0.0065  0.3442]
print(pose.orientation)                         # [ 0.9887 -0.0023  0.1494 -0.0152]（w,x,y,z）
print(pose.T)                                   # 4×4 位姿矩阵（左上姿态、右上位置）
print(arm.solve_fkine(q, "link3").position)     # [-0.1226  0.      0.3623]（可求任意中间帧）
arm.set_fkine_solver("mdh_fkine_solver")        # 切换激活求解器（返回 True）
print(arm.solve_fkine(q).position)              # [ 0.3447 -0.0065  0.3442]（与上面一致）
arm.close()
```

常用方法一览：

| 方法 | 作用 |
|:---:|:---:|
| `get_fkine_solvers()` | 列出已加载的全部求解器注册名 |
| `get_activated_fkine_solver()` | 查询当前激活的求解器注册名 |
| `set_fkine_solver(name)` | 切换激活求解器，成功返回 `True` |
| `solve_fkine(q, frame=None)` | 正运动学求解；`frame` 缺省为末端帧（pin 版可传任意帧，MDH 版仅末端帧） |

> 💡 两版求末端位姿的结果一致（末端位置差约 $10^{-13}$ m）——想换算法，改 config 名单或调用 `set_fkine_solver` 即可，调用代码一行不用动。

### 5.3 工作空间及求解思路

工作空间回答"末端能到哪里"。本节先给出定义，再说明蒙特卡洛采样求法与体积估算方法。

#### 5.3.1 什么是工作空间

**工作空间（Workspace）** 是末端参考点（通常取 TCP）所能到达的所有位置集合，直接回答"目标点是否够得到"。最常用的有两类：

| 类别 | 定义 | 工程意义 |
|:---:|:---:|:---:|
| **可达工作空间** | 末端至少以一种姿态能到达的所有位置 | "最大活动范围"，选型时参考 |
| **灵巧工作空间** | 末端以任意姿态都能到达的所有位置 | 高姿态要求任务的可行域 |

> 📌 本章聚焦**可达工作空间**（最常用、易计算）。灵巧工作空间的严格求解涉及逆运动学（[第三章](chapt3_ikine.md)）。

#### 5.3.2 求解思路：蒙特卡洛采样

解析求取工作空间边界需联立关节限位与运动学方程，非常困难。**实用做法是蒙特卡洛采样**——用大量随机关节角样本逼近工作空间：

| 步骤 | 操作 | 原理 |
|:---:|:---:|:---:|
| 1 | 在各关节角范围内均匀随机采样 $\theta_i$ | 覆盖关节空间全部可达构型 |
| 2 | 对每组采样调用 `arm.solve_fkine(q)` 求末端位置 $(p_x,p_y,p_z)$ | 关节空间映射到笛卡尔空间 |
| 3 | 收集末端位置点构成点云 | 包络即近似工作空间 |
| 4 | 可视化点云（3D 散点 / 投影截面） | 直观看出形状与尺寸 |

> 💡 采样数越多、点云越密、边界越清晰。本章脚本默认取 $2\times10^5$ 组样本。

#### 5.3.3 体积估算：体素计数

点云只给形状，量化"活动范围"还得算**体积**。最简单的办法是体素计数：把空间切成边长为 $h$ 的小方格，数一数有多少格子里落进了点；记被占用的格子数为 $N$，则体积近似为

$$V \approx N\,h^3$$

> 💡 体积是**近似值**：格子越小、边界越精细，但需要的采样点也越多——点太少时漏掉的空格子让体积偏小，格子太大时边界上的整格计入又让体积偏大。

### 5.4 工作空间绘制方法

用脚本把 5.3 节的思路"画"出来：硬限位内随机撒 $N$ 组关节角，逐组调用 `arm.solve_fkine` 求末端位置，点云边算边画、逐批刷新，结束后打印分析数据。流程与每步调用的核心库功能如下：

| 步骤 | 操作 | 调用的核心库功能 |
|:---:|:---:|:---:|
| ① | 创建对象（离线，不连接硬件），选定求解器 | `joyArm("<型号>")`、`arm.set_fkine_solver(SOLVER)` |
| ② | 硬限位内均匀随机采样 $N$ 组关节角 | `rand_within_limits(arm.joint_limits_hard_arm, size=N)` |
| ③ | 画布与 home 位形骨架：求各关节帧位置，基座→关节→末端连成折线 | `arm.solve_fkine(arm.home_arm, frame)` |
| ④ | 分批求末端位置并刷新画面（每批估一次体积） | `arm.solve_fkine(q)` |
| ⑤ | 打印分析数据（范围 / 臂展 / 体积 / 占比 / 耗时） | —— |

画面自左向右共三幅子图（3D 视图、俯视图、侧视图），各元素的作用与画法：

| 元素 | 作用 | 可视化 |
|:---:|:---:|:---:|
| 点云 | 工作空间本体（全部采样点的末端位置） | 三幅子图同画，点色 = 到基座距离（紫色近、黄色远） |
| 机械臂骨架 | home 位形下的参考姿态 | 基座 → 各关节 → 末端的黑色折线 |
| 末端坐标系 | home 位形下的末端朝向 | 末端原点处的红绿蓝三根轴（长 0.1 m） |
| 俯视图（X-Y） | 水平方向的覆盖范围 | 点云投影到 $X\text{-}Y$ 平面，等比例 |
| 侧视图（X-Z） | 竖直方向的覆盖范围 | 点云投影到 $X\text{-}Z$ 平面，等比例 |

脚本顶部的可调常量如下（改数字即可调参）：

| 常量 | 含义 | 默认值 |
|:---:|:---:|:---:|
| `N_SAMPLES` | 采样组数 | 200000 |
| `N_BATCHES` | 分批数（每批刷新一次画面） | 10 |
| `SEED` | 随机种子（固定后结果可复现） | 0 |
| `VOXEL` | 体素边长 (m) | 0.05 |
| `SOLVER` | 正运动学求解器注册名 | `"pin_fkine_solver"` |
| `SHOW_MAX` | 画面最多显示的点数（统计仍用全部点） | 20000 |

运行时须指定型号，如：

```bash
python ./chapt/chapt2_workspace.py rebot_arm_b601_dm
```

对应脚本位于 `joyarm_code/chapt/chapt2_workspace.py`，使用说明（新建环境、激活环境、安装依赖、切换目录、运行命令）见脚本头部注释。

> 💡 先用默认参数跑通，再增大 `N_SAMPLES` 观察体积怎样一点点变大（边界逐渐填实）；把 `SOLVER` 改为 `"mdh_fkine_solver"` 再跑一次，可核对两版结果一致。

### 5.5 工作空间用途与分析

工作空间的形状与大小是机械臂"活动能力"的直接体现。本节先看工程用途，再用实测数据分析 JoyArm。

#### 5.5.1 工作空间的用途

工作空间分析对**选型、布置、任务规划**有直接指导：

| 应用场景 | 指导作用 |
|:---:|:---:|
| 机械臂选型 | 目标作业区域必须完全落在可达工作空间内 |
| 安装布置 | 工件/料仓应布置在工作空间密集区域，避开边缘死角 |
| 任务可行性 | 先判断目标点是否在工作空间内，再做逆运动学求解 |
| 性能评估 | 工作空间体积与形状是衡量"活动能力"的基本指标 |

#### 5.5.2 JoyArm 工作空间分析

先交代条件：下列数据取自脚本默认参数的一次运行——型号 `rebot_arm_b601_dm`，硬限位内均匀采样 $N=2\times10^5$ 组（随机种子 0），求解器 `pin_fkine_solver`。

| 指标 | 数值 | 说明 |
|:---:|:---:|:---:|
| x 范围 (m) | [-0.721, +0.754] | 正前方比正后方略远（关节 1 限位的痕迹） |
| y 范围 (m) | [-0.765, +0.764] | 左右基本对称 |
| z 范围 (m) | [-0.341, +0.888] | 上至基座上方 0.89 m，下探安装面以下 0.34 m |
| 最大臂展 (m) | 0.890 | 末端到基座原点的最远距离 |
| 最近距离 (m) | 0.007 | 末端能折回到基座近旁 |
| 体积 (m³) | 约 1.584 | 体素法近似值（边长 0.05 m），估算方法见 5.3.3 节 |
| 安装面以下点占比 | 16.1% | $z<0$ 的采样点，台面安装时不可达 |
| 求解耗时 | 2.48 s | 单次约 12.4 µs |

对照运行画面读形状：

- **整体形状**：近似**实心的球状区域**——点云靠近基座最密、向外渐稀（画面呈亮核淡边），但末端折回后能到达离基座 0.007 m 的近旁，基座正上方同样有点，因此没有中心空洞。
- **俯视图近似整圆**：关节 1 只能转 ±2.8 rad（约 ±160°），看似该在背后留下缺口；但肩关节行程接近 $-\pi$，整条臂能翻到基座正后方把这个方向补上——留下的痕迹只是 $x$ 方向的不对称：正前方 +0.754 m、正后方 -0.721 m。
- **侧视图呈穹顶形**：最高点 $z\approx0.888$ m 在基座正上方附近，向两侧逐渐降低；下边界最低到 $z\approx-0.341$ m。

> ⚠️ 纯正运动学采样只受关节限位约束，**没考虑连杆自碰撞与桌面遮挡**：表中"最近距离 0.007 m"这类基座近旁的点、以及安装面以下 16.1% 的点，实际多半被机身挡住或低于台面，工程使用时应结合碰撞检测再确认有效范围。
