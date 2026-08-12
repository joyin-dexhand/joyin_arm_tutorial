"""纯 numpy SO(3)/SE(3) 数学

本模块实现旋转矩阵、欧拉角/RPY、轴角、四元数、齐次变换矩阵之间的互转与运算

约定（务必阅读）：
----------------

- 旋转矩阵 ``R``：``(3,3)`` numpy 数组，右手坐标系。
- **RPY 角**：``rpy = (r, p, y)`` 传入（roll/pitch/yaw），弧度；默认按 ``R = Rz(y)·Ry(p)·Rx(r)`` 计算。
- **四元数**：``(w, x, y, z)`` 顺序，实部在前，单位四元数。
- **齐次变换矩阵** ``T``：``(4,4)``。

"""
from __future__ import annotations

import numpy as np

__all__ = [
    # 基本旋转
    "rot_x",
    "rot_y",
    "rot_z",
    # RPY
    "rpy_to_R",
    "R_to_rpy",
    # 轴角
    "rodrigues",
    "R_to_axis_angle",
    "axis_angle_to_R",
    # 四元数
    "quat_to_R",
    "R_to_quat",
    "quat_to_axis_angle",
    "axis_angle_to_quat",
    "rpy_to_quat",
    "quat_to_rpy",
    "quat_mul",
    "quat_conj",
    "quat_norm",
    # 齐次变换
    "make_T",
    "T_to_Rp",
    "T_inv",
    "T_mul",
    "adT",
    # 插值
    "slerp",
]


# ============================================================
# 基本旋转矩阵
# ============================================================
def rot_x(angle: float) -> np.ndarray:
    """绕 X 轴旋转 ``angle`` 弧度的 ``(3,3)`` 旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)       # 预算 cos、sin，避免重复计算
    # X 轴不变（第一行/列 [1,0,0]），Y、Z 在 YZ 平面内转
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    """绕 Y 轴旋转 ``angle`` 弧度的 ``(3,3)`` 旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    # Y 轴不变（中间 [1]），X、Z 在 XZ 平面内转
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    """绕 Z 轴旋转 ``angle`` 弧度的 ``(3,3)`` 旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    # Z 轴不变（右下 [1]），X、Y 在 XY 平面内转
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ============================================================
# RPY ↔ 旋转矩阵
# ============================================================
def rpy_to_R(rpy) -> np.ndarray:
    """RPY 角 → 旋转矩阵。

    按 ``R = Rz(y)·Ry(p)·Rx(r)`` 计算（外旋固定角 X-Y-Z，即 RPY 约定）。

    :param rpy: ``(3,)`` 数组 ``(roll, pitch, yaw)``，弧度。
    :return: ``(3,3)`` 旋转矩阵。
    """
    rpy = np.asarray(rpy, dtype=float).reshape(3)  # 统一成 (3,) float 数组
    r, p, y = rpy                            # 拆出 roll/pitch/yaw
    # 外旋 X-Y-Z：先绕 X 转 r，再绕 Y 转 p，最后绕 Z 转 y
    # 矩阵相乘顺序与"绕固定轴旋转"的顺序相反（右乘）
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def R_to_rpy(R) -> np.ndarray:
    """旋转矩阵 → RPY 角 ``(roll, pitch, yaw)``，弧度。

    与 :func:`rpy_to_R` 互为逆运算。当 ``pitch`` 接近 ±90° 时
    进入万向锁奇异，仅 roll 与 yaw 之和有效。

    :param R: ``(3,3)`` 旋转矩阵。
    :return: ``(3,)`` 数组 ``(r, p, y)``。
    """
    R = np.asarray(R, dtype=float)
    # 由 R[2,0] 反算 pitch：sin(p) = -R[2,0]，clip 防浮点误差越界
    p = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    # 用 atan2 处理 ±90° 边界（atan2 比 acos 数值更稳定）
    if abs(abs(R[2, 0]) - 1.0) > 1e-9:        # 非奇异情况
        r = np.arctan2(R[2, 1], R[2, 2])      # roll = atan2(R21, R22)
        y = np.arctan2(R[1, 0], R[0, 0])      # yaw  = atan2(R10, R00)
    else:
        # 万向锁：pitch=±90° 时 roll 与 yaw 耦合，解不唯一
        # 约定取 yaw=0；roll 的符号须按 pitch 的正负分支（sp = sin(p)）：
        #   pitch=+π/2 时 R[0,1]=sin(r-y)   → r=atan2( R[0,1], R[1,1])
        #   pitch=-π/2 时 R[0,1]=-sin(r+y)  → r=atan2(-R[0,1], R[1,1])
        # 合并写成 r = atan2(sp·R[0,1], R[1,1])，sp=sin(p)=±1，保证 R 仍能精确重建
        sp = np.sin(p)
        r = np.arctan2(sp * R[0, 1], R[1, 1])
        y = 0.0
    return np.array([r, p, y])


# ============================================================
# 轴角 ↔ 旋转矩阵
# ============================================================
def axis_angle_to_R(k, theta: float) -> np.ndarray:
    """轴角 → 旋转矩阵（Rodrigues 公式）。

    :param k: ``(3,)`` 旋转轴（无需单位化，内部归一化）。
    :param theta: 旋转角，弧度。
    :return: ``(3,3)`` 旋转矩阵。
    """
    k = _normalize(k)
    return rodrigues(k, theta)


def rodrigues(k, theta=None) -> np.ndarray:
    """Rodrigues 公式：轴角 → 旋转矩阵。

    :param k: ``(3,)`` 旋转轴。若 ``theta`` 为 ``None``，``k`` 视为
              旋转向量（方向为轴、模长为角），内部拆解。
    :param theta: 旋转角，弧度；为 ``None`` 时从 ``k`` 的模长取。
    :return: ``(3,3)`` 旋转矩阵。
    """
    k = np.asarray(k, dtype=float).reshape(3)  # 统一成 (3,) float 数组
    if theta is None:
        # 没给角度：把 k 当"旋转向量"——方向是轴、长度是角
        theta = np.linalg.norm(k)               # 角度 = 向量模长
        if theta > 1e-12:                        # 避免除零
            k = k / theta                        # 单位化旋转轴
    else:
        k = _normalize(k)                        # 给了角度，只需单位化轴

    if theta < 1e-12:
        return np.eye(3)                         # 角度≈0 → 单位阵（不转）

    # 反对称矩阵 K（叉积矩阵），让公式能写成纯矩阵乘法
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    # 罗德里格斯公式：R = I + sin(θ)·K + (1-cos(θ))·K²
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def R_to_axis_angle(R) -> tuple[np.ndarray, float]:
    """旋转矩阵 → 轴角。

    :param R: ``(3,3)`` 旋转矩阵。
    :return: ``(k, theta)``，``k`` 为 ``(3,)`` 单位轴，``theta`` 为弧度。
    """
    R = np.asarray(R, dtype=float)
    # 迹公式：cos(θ) = (trace(R) - 1) / 2；clip 防浮点误差超出 [-1,1]
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)              # 反算转角 θ

    if theta < 1e-12:                         # θ≈0：旋转轴不确定，约定取 X 轴
        return np.array([1.0, 0.0, 0.0]), 0.0
    if abs(theta - np.pi) < 1e-6:
        # 接近 π：常规公式分母 2·sin(θ)→0 会数值爆炸，需用对称部分恢复轴
        # 数学关系：θ=π 时 R = 2·k·kᵀ − I（k 为单位轴）
        A = (R + np.eye(3)) / 2.0             # A = k·kᵀ
        k_abs = np.sqrt(np.clip(np.diag(A), 0.0, None))   # |k_i| = √A 对角元
        # 对角元只给绝对值，符号需由非对角项 A[i,j] = k_i·k_j 确定：
        # 取最大分量 idx 作参考并固定为正，其余分量符号 = sign(A[idx, j])
        idx = int(np.argmax(k_abs))
        if k_abs[idx] < 1e-12:
            return np.array([1.0, 0.0, 0.0]), np.pi
        k = np.zeros(3)
        k[idx] = k_abs[idx]
        for j in range(3):
            if j == idx:
                continue
            # 用 A 的非对角项确定 k_j 相对 k_idx 的符号
            sign = 1.0 if A[idx, j] >= 0.0 else -1.0
            k[j] = sign * k_abs[j]
        k = k / np.linalg.norm(k)             # 归一化（保险，理论已单位）
        return k, np.pi

    # 一般情况：轴向量从 R 的反对称部分提取，再除以 2·sin(θ) 归一化
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    k = k / (2.0 * np.sin(theta))
    return k, theta


# ============================================================
# 四元数 ↔ 旋转矩阵 / 轴角 / RPY
# ============================================================
def quat_to_R(q) -> np.ndarray:
    """四元数 ``(w,x,y,z)`` → ``(3,3)`` 旋转矩阵。

    :param q: ``(4,)`` 四元数，实部在前。无需单位化，内部归一化。
    """
    q = quat_norm(q)                          # 先单位化（旋转只用单位四元数）
    w, x, y, z = q
    # 四元数 → 旋转矩阵的标准公式（每个元素由 w/x/y/z 的二次项组合而成）
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_quat(R) -> np.ndarray:
    """旋转矩阵 → 四元数 ``(w,x,y,z)``（实部在前，单位四元数）。

    采用稳定的 Shepperd 分支方法，避免 sqrt 数值问题。
    """
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)                          # 迹 = R00 + R11 + R22
    q = np.empty(4)

    # Shepperd 分支法：根据"哪个对角元最大"选最优分支，避免 sqrt 出现负数或除以小数
    if tr > 0.0:                              # 分支 1：迹 > 0，先算 w
        s = np.sqrt(tr + 1.0) * 2.0           # s = 4w
        q[0] = 0.25 * s                       # w
        q[1] = (R[2, 1] - R[1, 2]) / s        # x
        q[2] = (R[0, 2] - R[2, 0]) / s        # y
        q[3] = (R[1, 0] - R[0, 1]) / s        # z
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):   # 分支 2：R00 最大，先算 x
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # s = 4x
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:                   # 分支 3：R11 最大，先算 y
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # s = 4y
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:                                     # 分支 4：R22 最大，先算 z
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # s = 4z
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
    return quat_norm(q)                       # 再单位化一次，保证数值精度


def axis_angle_to_quat(k, theta: float) -> np.ndarray:
    """轴角 → 四元数 ``(w,x,y,z)``。

    :param k: ``(3,)`` 旋转轴（内部归一化）。
    :param theta: 旋转角，弧度。
    """
    k = _normalize(k)                         # 单位化旋转轴
    half = theta / 2.0                        # 四元数用的是半角
    s = np.sin(half)
    # 实部 w = cos(θ/2)；虚部 (x,y,z) = sin(θ/2) · 单位轴
    return np.array([np.cos(half), k[0] * s, k[1] * s, k[2] * s])


def quat_to_axis_angle(q) -> tuple[np.ndarray, float]:
    """四元数 → 轴角 ``(k, theta)``，``theta ∈ [0, π]``。"""
    q = quat_norm(q)
    # 单位四元数 q 与 -q 表示同一旋转；为使 θ 落在 [0,π]，w<0 时取 -q（走短弧）
    if q[0] < 0.0:
        q = -q
    w = np.clip(q[0], -1.0, 1.0)              # 实部，clip 防浮点误差越界
    theta = 2.0 * np.arccos(w)                # 转角 θ = 2·arccos(w)（半角公式的逆）
    s = np.sqrt(1.0 - w * w)                  # 虚部模长 = sin(θ/2)
    if s < 1e-8:                              # θ≈0：轴不确定，约定取 X 轴
        return np.array([1.0, 0.0, 0.0]), 0.0
    k = q[1:4] / s                            # 单位轴 = 虚部 / sin(θ/2)
    return k, theta


def rpy_to_quat(rpy) -> np.ndarray:
    """RPY 角 ``(r,p,y)`` → 四元数 ``(w,x,y,z)``。"""
    rpy = np.asarray(rpy, dtype=float).reshape(3)
    r, p, y = rpy
    # 各角的半角 sin/cos（四元数恒以半角形式出现）
    cr, sr = np.cos(r / 2.0), np.sin(r / 2.0)
    cp, sp = np.cos(p / 2.0), np.sin(p / 2.0)
    cy, sy = np.cos(y / 2.0), np.sin(y / 2.0)
    # 由三个半角的 sin/cos 组合出四元数四分量（标准公式）
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return quat_norm(np.array([w, x, y_, z]))


def quat_to_rpy(q) -> np.ndarray:
    """四元数 ``(w,x,y,z)`` → RPY 角 ``(r,p,y)``，弧度。"""
    return R_to_rpy(quat_to_R(q))


def quat_mul(q1, q2) -> np.ndarray:
    """四元数乘法（Hamilton 积），返回单位四元数 ``(w,x,y,z)``。"""
    w1, x1, y1, z1 = quat_norm(q1)
    w2, x2, y2, z2 = quat_norm(q2)
    # Hamilton 四元数乘法公式（注意虚部交叉项的符号规则）
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return quat_norm(np.array([w, x, y, z]))


def quat_conj(q) -> np.ndarray:
    """四元数共轭（与逆等价，对单位四元数）。"""
    q = quat_norm(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_norm(q) -> np.ndarray:
    """四元数归一化 → 单位四元数 ``(w,x,y,z)``。"""
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)                     # 四元数模长
    if n < 1e-12:                             # 零四元数无法归一，返回单位四元数（零旋转）
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n                              # 除以模长 → 单位四元数


# ============================================================
# 齐次变换矩阵
# ============================================================
def make_T(R=None, p=None) -> np.ndarray:
    """由旋转 ``R`` 与平移 ``p`` 组装 ``(4,4)`` 齐次变换矩阵。

    :param R: ``(3,3)`` 旋转矩阵，缺省为单位阵。
    :param p: ``(3,)`` 平移向量，缺省为零向量。
    :return: ``(4,4)`` 齐次变换矩阵。
    """
    T = np.eye(4)                             # 先建 4x4 单位阵（最后一行固定 [0,0,0,1]）
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float)    # 左上 3x3 块填旋转矩阵
    if p is not None:
        T[:3, 3] = np.asarray(p, dtype=float).reshape(3)   # 右上 3x1 列填平移
    return T


def T_to_Rp(T) -> tuple[np.ndarray, np.ndarray]:
    """拆解 ``(4,4)`` 齐次变换矩阵 → ``(R, p)``。

    :return: ``(R(3,3), p(3,))``。
    """
    T = np.asarray(T, dtype=float)
    # 左上 3x3 = 旋转 R；右上 3x1 = 平移 p；copy() 防止外部修改影响原矩阵
    return T[:3, :3].copy(), T[:3, 3].copy()


def T_inv(T) -> np.ndarray:
    """齐次变换矩阵的逆（利用旋转正交性，避免通用 4×4 求逆）。

    对 ``T = [R p; 0 1]``，逆为 ``[Rᵀ  -Rᵀp; 0 1]``。
    """
    R, p = T_to_Rp(T)
    R_inv = R.T                              # 旋转矩阵的逆 = 转置（正交性）
    # 平移部分的逆 = -Rᵀ·p（把原点变换回去）
    return make_T(R_inv, -R_inv @ p)


def T_mul(T1, T2) -> np.ndarray:
    """两个 ``(4,4)`` 齐次变换矩阵相乘。"""
    return np.asarray(T1, dtype=float) @ np.asarray(T2, dtype=float)


def adT(T) -> np.ndarray:
    """齐次变换 ``T`` 的 6×6 伴随矩阵 ``Ad_T``（Ch4/Ch9 用）。

    用于旋量/六维力在不同坐标系间的变换：

    .. math::

        \\mathrm{Ad}_T = \\begin{bmatrix} R & \\hat{p} R \\\\
                                       0 & R \\end{bmatrix}

    其中 :math:`\\hat{p}` 为 ``p`` 的反对称矩阵。

    :param T: ``(4,4)`` 齐次变换矩阵。
    :return: ``(6,6)`` 伴随矩阵。
    """
    R, p = T_to_Rp(T)
    # p 的反对称矩阵 p̂（3x3），用于把叉积 p× 写成矩阵乘法
    phat = np.array([[0.0, -p[2], p[1]], [p[2], 0.0, -p[0]], [-p[1], p[0], 0.0]])
    # 组装 6x6 伴随矩阵：[[R, p̂·R], [0, R]]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R                           # 左上：旋转作用于角速度部分
    Ad[:3, 3:] = phat @ R                    # 右上：平移与旋转的耦合项
    Ad[3:, 3:] = R                           # 右下：旋转作用于线速度部分
    return Ad


# ============================================================
# 插值
# ============================================================
def slerp(R0, R1, s: float) -> np.ndarray:
    """球面线性插值两个旋转矩阵（Ch5 轨迹、Ch12 遥操用）。

    :param R0: ``(3,3)`` 起始旋转。
    :param R1: ``(3,3)`` 终止旋转。
    :param s: 插值参数，``[0,1]``；``0`` 返回 ``R0``，``1`` 返回 ``R1``。
    :return: ``(3,3)`` 插值后的旋转矩阵。
    """
    # 先把两个旋转矩阵转成四元数（四元数便于球面插值）
    q0 = R_to_quat(R0)
    q1 = R_to_quat(R1)
    # 若点积为负，翻转 q1 以走最短弧（q 和 -q 表示同一旋转，选更近的那个）
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)   # 两四元数的夹角余弦

    if dot > 0.9995:
        # 两旋转极接近：slerp 分母→0 不稳定，改用线性插值 + 归一化（nlerp）
        q = quat_norm(q0 + s * (q1 - q0))
    else:
        # 球面线性插值：在四元数球面上沿大圆走
        theta_0 = np.arccos(dot)              # 两旋转间的总夹角
        theta = theta_0 * s                   # 当前要走的夹角
        sin_0 = np.sin(theta_0)
        # 两个插值权重（保证结果仍在单位球面上）
        s0 = np.sin((1.0 - s) * theta_0) / sin_0
        s1 = np.sin(theta) / sin_0
        q = s0 * q0 + s1 * q1
    return quat_to_R(q)                  # 四元数转回旋转矩阵


# ============================================================
# 内部工具
# ============================================================
def _normalize(v) -> np.ndarray:
    """归一化 3 维向量为单位向量；零向量返回 (1,0,0)。"""
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)                     # 向量模长
    if n < 1e-12:                             # 零向量无法归一，约定返回 X 轴
        return np.array([1.0, 0.0, 0.0])
    return v / n                              # 除以模长 → 单位向量
