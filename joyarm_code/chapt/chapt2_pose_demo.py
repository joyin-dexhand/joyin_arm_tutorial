"""
==============================================================================
四种姿态表示可视化演示  (第二章 §2.6)
==============================================================================
【功能概要】
    用 PySide6 GUI + matplotlib 3D 演示 RPY、轴角、四元数、旋转矩阵 R 四种姿态表示的实时互转：

      - 左侧 3D：固定参考系 {A}（虚线 RGB 单位轴）+ 单位球固连系 {B}（实线 RGB 坐标轴 + 半透明线框单位球）

      - 右侧四组控件，任改一组，其余三组数值与 3D 视图实时跟随：
          · RPY 角   ：rx ry rz（滑块 + 数值框，度）
          · 轴角     ：kx ky kz（单位轴）+ theta（转角，度）
          · 四元数   ：w x y z（只读数值框，随其他组刷新）
          · 旋转矩阵 ：3×3 表格（只读，随其他组刷新）

    拖动 RPY 推到 Pitch=±90° 时，可直观体会"万向锁"——轴角/四元数仍正常，
    而 RPY 自身退化（Roll/Yaw 无法区分）。

【环境与运行】
    # 1. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 2. 创建虚拟环境并同步依赖
    cd ~/joyarm_code
    uv sync
    source .venv/bin/activate
    # Linux 缺少 libxcb-cursor.so.0 时执行
    sudo apt-get install -y libxcb-cursor0 

    # 3. 运行本脚本
    cd ./chapt
    python chapt2_pose_demo.py

    # 注：RPY 滑块的旋转矩阵 R = R_Z(rz)·R_Y(ry)·R_X(rx) —— 按 x-y-z 顺序的外旋（等价于按 z-y-x 顺序的内旋），即标准 RPY。详见第二章的讲解。
         轴角表示的数值在计算时会自动单位化，可随意输入任意非零向量。

==============================================================================
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSlider, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QGroupBox, QGridLayout, QDoubleSpinBox,
)

# ----------------------------------------------------------------------------
# 全局常量
# ----------------------------------------------------------------------------
ROT_MIN, ROT_MAX = -180, 180           # RPY 旋转角滑块范围 (度)
AXIS_MIN, AXIS_MAX = -1.0, 1.0         # 轴向量分量范围
THETA_MIN, THETA_MAX = 0, 180          # 轴角转角范围 (度)
SLIDER_STEPS = 400                      # 滑块离散步数（越大越平滑）
AXIS_COLORS = ("#e6194b", "#3cb44b", "#2b6cb0")   # RGB 对应 X/Y/Z
PLOT_LIM = 1.8                          # 3D 坐标轴显示范围


# ----------------------------------------------------------------------------
# 旋转矩阵（与第二章 §2.1 公式一致：RPY = R_Z(rz)·R_Y(ry)·R_X(rx)，单位：弧度）
# ----------------------------------------------------------------------------
def rot_x(a: float) -> np.ndarray:
    """绕 X 轴旋转 a 弧度的 3x3 旋转矩阵（X 轴不变，Y/Z 平面内转）。"""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])


def rot_y(a: float) -> np.ndarray:
    """绕 Y 轴旋转 a 弧度的 3x3 旋转矩阵（Y 轴不变，X/Z 平面内转）。"""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])


def rot_z(a: float) -> np.ndarray:
    """绕 Z 轴旋转 a 弧度的 3x3 旋转矩阵（Z 轴不变，X/Y 平面内转）。"""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])


def rpy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """由 RPY 角（弧度）合成旋转矩阵，与第二章 §2.1 外旋 XYZ 约定一致：
       R = R_Z(rz)·R_Y(ry)·R_X(rx)  （rz=Yaw, ry=Pitch, rx=Roll）。"""
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def rpy_from_matrix(R: np.ndarray):
    """由旋转矩阵反算 RPY 角（弧度），与 rpy_to_matrix 的 R = R_Z·R_Y·R_X 顺序对应。
    β=±90°（万向锁）时用 arctan2 稳定形式，不报错（但 Roll/Yaw 解不唯一，R 仍一致）。"""
    ry = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))  
    rz = np.arctan2(R[1, 0], R[0, 0])               
    rx = np.arctan2(R[2, 1], R[2, 2])                   
    return rx, ry, rz


# ----------------------------------------------------------------------------
# 轴角 ↔ 旋转矩阵（与第二章 §2.2 罗德里格斯公式一致）
# ----------------------------------------------------------------------------
def rodrigues(k: np.ndarray, theta: float) -> np.ndarray:
    """由轴角 (k, theta) 求旋转矩阵（罗德里格斯公式）。k 自动单位化。"""
    k = np.asarray(k, dtype=float)       
    norm = np.linalg.norm(k)                  
    if norm < 1e-12:                   
        return np.eye(3)
    k = k / norm                         
    ct, st = np.cos(theta), np.sin(theta)    
    kkt = np.outer(k, k)                      
    # 反对称矩阵 K（叉积矩阵），让公式能写成纯矩阵乘法
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    # 罗德里格斯公式：R = cosθ·I + (1-cosθ)·(k⊗k) + sinθ·K
    return ct * np.eye(3) + (1 - ct) * kkt + st * K


def axis_angle_from_matrix(R: np.ndarray):
    """由旋转矩阵反算轴角 (k, theta)。θ≈0 时轴不确定，取默认 [1,0,0]。"""
    # 迹公式：cosθ = (trace(R) - 1) / 2；clip 防止浮点误差超出 [-1,1]
    ct = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(ct)                 
    s = 2 * np.sin(theta)           
    if s < 1e-6:                    
        return np.array([1.0, 0.0, 0.0]), 0.0
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s
    return k, theta


# ----------------------------------------------------------------------------
# 四元数 ↔ 旋转矩阵 / 轴角（与第二章 §2.3、§2.5 公式一致）
# ----------------------------------------------------------------------------
def quat_to_matrix(q) -> np.ndarray:
    """由四元数 q=(w,x,y,z) 求旋转矩阵（自动单位化）。"""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)       
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_to_quat(k: np.ndarray, theta: float) -> np.ndarray:
    """由轴角求四元数 q=(cos θ/2, sin θ/2 · k̂)。k 自动单位化。"""
    k = np.asarray(k, dtype=float)
    norm = np.linalg.norm(k)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])        
    k = k / norm              
    return np.concatenate(([np.cos(theta / 2)], np.sin(theta / 2) * k))


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """由旋转矩阵求四元数（经轴角中转：R→(k,θ)→q），返回单位四元数。"""
    k, theta = axis_angle_from_matrix(R)     
    return axis_angle_to_quat(k, theta)      


# ----------------------------------------------------------------------------
# 单位球线框顶点：用于可视化 {B} 的姿态（球随 R 旋转后形状改变）
# ----------------------------------------------------------------------------
def unit_ball_lines():
    """返回单位球上若干经/纬线折线段的顶点列表；每个元素是 (n,3) 的点序列，
    绘制时逐相邻两点连线，并对整体做 R 变换。"""
    theta = np.linspace(0, np.pi, 24)
    phi = np.linspace(0, 2 * np.pi, 48)
    lines = []
    for p in phi[::4]:
        x = np.sin(theta) * np.cos(p)
        y = np.sin(theta) * np.sin(p)
        z = np.cos(theta)
        lines.append(np.stack([x, y, z], axis=-1))

    for t in theta[2:-2:4]:
        x = np.sin(t) * np.cos(phi)
        y = np.sin(t) * np.sin(phi)
        z = np.full_like(phi, np.cos(t))   
        lines.append(np.stack([x, y, z], axis=-1))
    return lines


# ============================================================================
# 主窗口
# ============================================================================
class PoseRepWindow(QMainWindow):
    """演示窗口：左侧 3D 画布 + 右侧四组姿态控件（RPY/轴角/四元数/R 矩阵）。

    设计核心：以旋转矩阵 R 作为"唯一真相"(single source of truth)。
    任一组控件变化，都先折算成 R，再由 R 反推出另三组的显示值并刷新 3D。"""

    def __init__(self):
        super().__init__()                
        self.setWindowTitle("四种姿态表示可视化演示  (RPY / 轴角 / 四元数 / R)")
        self.resize(1320, 800)

        self._R = np.eye(3)

        self._build_ui()
        self._refresh_from_R(self._R, source=None)   

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        """搭建主界面：左 3D 画布 + 右控制面板（四组）的横向布局。"""
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)          

        # --- 左侧：3D 画布 ---
        self.fig = plt.figure(figsize=(7, 7))
        self.canvas = FigureCanvas(self.fig) 
        self.ax = self.fig.add_subplot(111, projection="3d") 
        self._setup_axes()                  
        h.addWidget(self.canvas, stretch=3)   

        # --- 右侧：控制面板（四组） ---
        panel = QWidget()
        pv = QVBoxLayout(panel)           
        pv.addWidget(self._build_rpy_group())       
        pv.addWidget(self._build_axis_angle_group())  
        pv.addWidget(self._build_quat_group())         
        pv.addWidget(self._build_matrix_group())      
        pv.addStretch(1)                   
        h.addWidget(panel, stretch=2)       

    def _setup_axes(self):
        """固定 3D 坐标系样式（等比例、正方向、网格）。"""
        ax = self.ax
        ax.set_xlim(-PLOT_LIM, PLOT_LIM)
        ax.set_ylim(-PLOT_LIM, PLOT_LIM)
        ax.set_zlim(-PLOT_LIM, PLOT_LIM)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=22, azim=-55)        # 设定观察视角（仰角 22°、方位 -55°）

    # ---------------- RPY 组：rx ry rz（滑块 + 数值框，度） ----------------
    def _build_rpy_group(self):
        """构建 RPY 控件组：Roll/Pitch/Yaw 三轴，各带滑块和数值框。"""
        box = QGroupBox("RPY 角  (rx ry rz, 单位:°)")
        g = QGridLayout(box)             
        self.rpy_sliders = []     
        self.rpy_spins = []             
        for i, name in enumerate(("rx (Roll)", "ry (Pitch)", "rz (Yaw)")):
            g.addWidget(QLabel(name), i, 0)
            s = QSlider(Qt.Horizontal)
            s.setMinimum(0)
            s.setMaximum(SLIDER_STEPS)
            s.setValue(SLIDER_STEPS // 2)            # 中点 = 0°
            # 滑块值改变时回调：lambda 捕获当前轴索引 idx
            s.valueChanged.connect(lambda v, idx=i: self._on_rpy_slider(idx, v))
            g.addWidget(s, i, 1)
            sp = self._make_spin(ROT_MIN, ROT_MAX, 1)
            sp.valueChanged.connect(lambda v, idx=i: self._on_rpy_spin(idx, v))
            g.addWidget(sp, i, 2)
            self.rpy_sliders.append(s)
            self.rpy_spins.append(sp)
        return box

    # ---------------- 轴角组：kx ky kz + theta（滑块 + 数值框） ----------------
    def _build_axis_angle_group(self):
        """构建轴角控件组：旋转轴 k=(kx,ky,kz) 三分量 + 转角 θ。"""
        box = QGroupBox("轴角  (k̂=(kx,ky,kz), θ, 单位:°)")
        g = QGridLayout(box)
        self.k_sliders = []             
        self.k_spins = []                 
        for i, name in enumerate(("kx", "ky", "kz")):
            g.addWidget(QLabel(name), i, 0)
            s = QSlider(Qt.Horizontal)
            s.setMinimum(0)
            s.setMaximum(SLIDER_STEPS)
            s.setValue(SLIDER_STEPS // 2)            # 中点 = 0
            s.valueChanged.connect(lambda v, idx=i: self._on_axis_slider(idx, v))
            g.addWidget(s, i, 1)
            sp = self._make_spin(AXIS_MIN, AXIS_MAX, 4)
            sp.valueChanged.connect(lambda v, idx=i: self._on_axis_spin(idx, v))
            g.addWidget(sp, i, 2)
            self.k_sliders.append(s)
            self.k_spins.append(sp)
        # theta 行：单独一行放转角 θ
        g.addWidget(QLabel("θ (转角)"), 3, 0)
        s = QSlider(Qt.Horizontal)
        s.setMinimum(0)
        s.setMaximum(SLIDER_STEPS)
        s.setValue(0)                                # 默认 0°
        s.valueChanged.connect(lambda v: self._on_theta_slider(v))
        g.addWidget(s, 3, 1)
        sp = self._make_spin(THETA_MIN, THETA_MAX, 1)
        sp.setValue(0.0)
        sp.valueChanged.connect(lambda v: self._on_theta_spin(v))
        g.addWidget(sp, 3, 2)
        self.theta_slider = s
        self.theta_spin = sp
        return box

    # ---------------- 四元数组：w x y z（只读数值框） ----------------
    def _build_quat_group(self):
        """构建四元数显示组：w/x/y/z 四个只读数值框，随其他组刷新。"""
        box = QGroupBox("四元数  q=(w,x,y,z)  (随其他组刷新)")
        g = QGridLayout(box)
        self.quat_spins = []
        for i, name in enumerate(("w", "x", "y", "z")):
            g.addWidget(QLabel(name), 0, i)
            sp = self._make_spin(-1.0, 1.0, 4)
            sp.setReadOnly(True)          
            g.addWidget(sp, 1, i)
            self.quat_spins.append(sp)
        return box

    # ---------------- 旋转矩阵 R 组：3×3 只读表格 + 重置按钮 ----------------
    def _build_matrix_group(self):
        """构建旋转矩阵显示组：3x3 只读表格 + 重置为单位阵按钮。"""
        box = QGroupBox("旋转矩阵 R  (随其他组刷新)")
        v = QVBoxLayout(box)
        self.table = QTableWidget(3, 3)
        self.table.setFixedHeight(110)
        for r in range(3):
            for c in range(3):
                it = QTableWidgetItem("+0.000")
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)   # 只读：去掉可编辑标志
                self.table.setItem(r, c, it)
        v.addWidget(self.table)
        b_reset = QPushButton("重置为单位阵")
        b_reset.clicked.connect(self._reset_identity)
        v.addWidget(b_reset)
        return box

    # ------------------------------------------------------------------ 控件工厂
    def _make_spin(self, lo, hi, decimals):
        """构造一个数值输入框（指定范围与小数位）。"""
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSingleStep(0.01)                 # 上下箭头的步长
        sp.setKeyboardTracking(False)          # 仅回车/失焦触发，防抖
        return sp

    # ------------------------------------------------------------------ 滑块<->值 换算
    @staticmethod
    def _to_slider(v, lo, hi):
        """真实值 v -> 滑块离散整数。把 [lo,hi] 线性映射到 [0, SLIDER_STEPS]。"""
        return int(round((v - lo) / (hi - lo) * SLIDER_STEPS))

    @staticmethod
    def _from_slider(s, lo, hi):
        """滑块离散值 -> 真实值。逆映射回 [lo,hi] 物理区间。"""
        return lo + s.value() / SLIDER_STEPS * (hi - lo)

    # ------------------------------------------------------------------ RPY 回调
    def _on_rpy_slider(self, idx, _v):
        """RPY 滑块拖动 -> 同步数值框 -> 折算 R。"""
        val = self._from_slider(self.rpy_sliders[idx], ROT_MIN, ROT_MAX)
        with QSignalBlocker(self.rpy_spins[idx]):      # 阻断 spin 回调，防递归
            self.rpy_spins[idx].setValue(val)
        self._commit_from_rpy()

    def _on_rpy_spin(self, idx, val):
        """RPY 数值框输入 -> 同步滑块 -> 折算 R。"""
        with QSignalBlocker(self.rpy_sliders[idx]):    # 阻断 slider 回调，防递归
            self.rpy_sliders[idx].setValue(self._to_slider(val, ROT_MIN, ROT_MAX))
        self._commit_from_rpy()

    def _commit_from_rpy(self):
        """由 RPY 三控件折算权威 R，再分发刷新另三组。"""
        degs = [self.rpy_spins[i].value() for i in range(3)]  
        R = rpy_to_matrix(*np.deg2rad(degs))                  
        self._refresh_from_R(R, source="rpy")              

    # ------------------------------------------------------------------ 轴角回调
    def _on_axis_slider(self, idx, _v):
        """轴分量滑块拖动 -> 修改该分量并归一化 -> 折算 R。"""
        val = self._from_slider(self.k_sliders[idx], AXIS_MIN, AXIS_MAX)
        self._set_axis_component(idx, val)
        self._commit_from_axis_angle()

    def _on_axis_spin(self, idx, val):
        """轴分量数值框输入 -> 修改该分量并归一化 -> 折算 R。"""
        self._set_axis_component(idx, val)
        self._commit_from_axis_angle()

    def _set_axis_component(self, idx: int, val: float):
        """修改轴的第 idx 个分量为 val，并实时归一化其余两分量，保持 ‖k̂‖=1。
        归一化规则：其余两分量按原比例缩放，使其平方和补足 (1 - val²)；
        若其余两分量本就为零（无方向可保留），则均分剩余模长。
        归一化后的三个分量回写各自滑块+数值框（阻断信号防递归）。"""
        val = float(np.clip(val, AXIS_MIN, AXIS_MAX)) 
        k = [self.k_spins[i].value() for i in range(3)]
        k[idx] = val                               
        others = [i for i in range(3) if i != idx]  
        rem_sq = max(0.0, 1.0 - val * val)       
        cur_sq = sum(k[i] * k[i] for i in others)
        if cur_sq < 1e-12:
            each = (rem_sq / len(others)) ** 0.5 
            for i in others:
                k[i] = each
        else:
            scale = (rem_sq / cur_sq) ** 0.5   
            for i in others:
                k[i] *= scale
        for i, v in enumerate(k):
            with QSignalBlocker(self.k_sliders[i]), QSignalBlocker(self.k_spins[i]):
                self.k_sliders[i].setValue(self._to_slider(v, AXIS_MIN, AXIS_MAX))
                self.k_spins[i].setValue(v)

    def _on_theta_slider(self, _v):
        """转角 θ 滑块拖动 -> 同步数值框 -> 折算 R。"""
        val = self._from_slider(self.theta_slider, THETA_MIN, THETA_MAX)
        with QSignalBlocker(self.theta_spin):
            self.theta_spin.setValue(val)
        self._commit_from_axis_angle()

    def _on_theta_spin(self, val):
        """转角 θ 数值框输入 -> 同步滑块 -> 折算 R。"""
        with QSignalBlocker(self.theta_slider):
            self.theta_slider.setValue(self._to_slider(val, THETA_MIN, THETA_MAX))
        self._commit_from_axis_angle()

    def _commit_from_axis_angle(self):
        """由轴角控件折算权威 R，再分发刷新另三组。"""
        k = np.array([self.k_spins[i].value() for i in range(3)]) 
        theta = np.deg2rad(self.theta_spin.value())   
        R = rodrigues(k, theta)                     
        self._refresh_from_R(R, source="axis")         

    # ------------------------------------------------------------------
    def _refresh_from_R(self, R: np.ndarray, source):
        """以 R 为唯一真相，刷新四组控件 + 3D。source 标记触发源，跳过其自身回写防递归。
        非源组一律用 QSignalBlocker 阻断回调。"""
        self._R = R

        if source != "rpy":
            rx, ry, rz = rpy_from_matrix(R)     
            degs = np.rad2deg([rx, ry, rz])      
            for i, (s, sp, v) in enumerate(zip(self.rpy_sliders, self.rpy_spins, degs)):
                with QSignalBlocker(s), QSignalBlocker(sp):   # 阻断两者防递归
                    s.setValue(self._to_slider(v, ROT_MIN, ROT_MAX))
                    sp.setValue(v)

        # 轴角组（非 axis 触发时回写）
        if source != "axis":
            k, theta = axis_angle_from_matrix(R) 
            deg = np.rad2deg(theta)           
            for i, (s, sp, v) in enumerate(zip(self.k_sliders, self.k_spins, k)):
                with QSignalBlocker(s), QSignalBlocker(sp):
                    s.setValue(self._to_slider(v, AXIS_MIN, AXIS_MAX))
                    sp.setValue(v)
            with QSignalBlocker(self.theta_slider), QSignalBlocker(self.theta_spin):
                self.theta_slider.setValue(self._to_slider(deg, THETA_MIN, THETA_MAX))
                self.theta_spin.setValue(deg)

        q = quat_from_matrix(R)                   
        for sp, v in zip(self.quat_spins, q):
            with QSignalBlocker(sp):
                sp.setValue(v)

        self._update_matrix_table()

        # 3D 视图
        self._render()

    def _update_matrix_table(self):
        """把当前 self._R 写入 3×3 只读表格。"""
        R = self._R
        for r in range(3):
            for c in range(3):
                self.table.item(r, c).setText(f"{R[r, c]:+.3f}") 

    def _reset_identity(self):
        """重置为单位阵：直接走分发（source=None 触发全部回写）。"""
        self._refresh_from_R(np.eye(3), source=None)

    # ------------------------------------------------------------------
    def _render(self):
        """重绘 3D 视图：清空 -> 画固定系 {A} -> 画被驱动的系 {B} + 球。"""
        ax = self.ax
        ax.cla()            
        self._setup_axes()        

        T_A = np.eye(4)
        self._draw_frame(ax, T_A, dashed=True, prefix="{A}")

        T_B = np.eye(4)
        T_B[:3, :3] = self._R               
        self._draw_frame(ax, T_B, dashed=False, prefix="{B}")
        self._draw_ball(ax, T_B)

        self.canvas.draw_idle()                # 异步触发重绘，不阻塞 UI

    def _draw_frame(self, ax, T, dashed: bool, prefix: str):
        """画一个坐标系：原点(黑点) + 三条 RGB 单位轴。dashed=True 画虚线。"""
        origin = T[:3, 3]                  
        length = 1.0                  
        for i, color in enumerate(AXIS_COLORS): 
            direction = T[:3, i]              
            end = origin + direction * length 
            ls = "--" if dashed else "-"     
            lw = 1.5 if dashed else 2.4        
            ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]],
                    color=color, linestyle=ls, linewidth=lw)
            ax.text(end[0], end[1], end[2], f"{prefix}", color=color, fontsize=8)
        ax.scatter(*origin, color="k", s=18)   # 原点画个黑点

    def _draw_ball(self, ax, T):
        """画单位球：把球面经纬线用 T 变换到 {A} 下后绘制（半透明线框，让旋转可见）。"""
        R, p0 = T[:3, :3], T[:3, 3]
        for pts in unit_ball_lines():
            p = (R @ pts.T).T + p0           
            ax.plot(p[:, 0], p[:, 1], p[:, 2],
                    color="#4a5568", linewidth=0.5, alpha=0.45)


# ============================================================================
# 入口
# ============================================================================
def main():
    """程序入口：创建 Qt 应用 -> 显示主窗口 -> 进入事件循环。"""
    app = QApplication(sys.argv)              # 每个 Qt 程序需要一个 QApplication 对象
    win = PoseRepWindow()
    win.show()                                # 显示窗口
    sys.exit(app.exec())                      # 进入 Qt 事件循环；退出时用状态码关闭进程


if __name__ == "__main__":
    main()
