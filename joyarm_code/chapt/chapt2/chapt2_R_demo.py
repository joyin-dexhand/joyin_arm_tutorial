"""
==============================================================================
四种姿态表示转换可视化演示  (第二章 §2.6)
==============================================================================
【功能概要】
    用 PySide6 GUI + matplotlib 3D 演示 RPY、轴角、四元数、旋转矩阵 R 四种姿态表示的互相转换：

      - 固定参考系 {A}：原点 + RGB 虚线单位轴（红=X、绿=Y、蓝=Z），观察基准
      - 固连系 {B}：原点 + RGB 实线坐标轴，被姿态 R 驱动（纯旋转，无平移）

      - 右侧四个输入分区（RPY / 轴角 / 四元数 / 旋转矩阵），各有一个"开始计算"按钮：
        点击后以该分区输入为准计算出姿态 R，其余分区与 3D 场景随之刷新

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    # 若 Linux 缺少 libxcb-cursor.so.0 ，则先安装：
    sudo apt-get install -y libxcb-cursor0

    # 2. 运行本脚本
    uv run python chapt/chapt2/chapt2_R_demo.py

    # 数值简单实例：RPY = (0, 0, 90)（单位：度），即绕 Z 轴转 90°
      特点：{B} 的 x 轴转到 {A} 的 y 方向、y 轴转到 -x 方向、z 轴不变；
      
      "开始计算"后另外三组应为：
      · 轴角     k = (0, 0, 1)，θ = 90°
      · 四元数   q = (0.707, 0, 0, 0.707)
      · 旋转矩阵 R = R_Z(90°)

    # 注：转换函数由 joyarm_core.utils.transforms 提供（R = R_Z·R_Y·R_X，标准 RPY 约定；四元数 (w,x,y,z) 实部在前）。

==============================================================================
"""

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QGroupBox, QDoubleSpinBox, QMessageBox, QHeaderView,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装时也可直接运行
from joyarm_core.utils.transforms import (
    rpy_to_R, R_to_rpy, axis_angle_to_R, R_to_axis_angle, quat_to_R, R_to_quat,
)

# ----------------------------------------------------------------------------
# 全局常量
# ----------------------------------------------------------------------------
RPY_MIN, RPY_MAX = -180.0, 180.0        # RPY 角输入范围 (度)
AXIS_MIN, AXIS_MAX = -1.0, 1.0          # 旋转轴分量输入范围
THETA_MIN, THETA_MAX = 0.0, 180.0       # 轴角转角输入范围 (度)
QUAT_MIN, QUAT_MAX = -1.0, 1.0          # 四元数分量输入范围
AXIS_COLORS = ("#e6194b", "#3cb44b", "#2b6cb0")  # RGB 对应 X/Y/Z
PLOT_LIM = 1.8                          # 3D 坐标轴显示范围（内容恒为单位轴，固定即可）


# ============================================================================
# 主窗口
# ============================================================================
class RConvDemoWindow(QMainWindow):
    """演示窗口：左侧 3D 画布 + 右侧四个姿态输入分区（RPY/轴角/四元数/R 矩阵）。

    设计核心：以旋转矩阵 R 作为"唯一真相"(single source of truth)。
    任一分区点"开始计算"，都先由该分区输入折算出 R，再把 R 的规范表示回写其余
    分区并刷新 3D（RPY 分区作为输入源时保留输入原值）。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("四种姿态表示转换可视化演示  (RPY / 轴角 / 四元数 / R)")
        self.resize(1320, 800)

        self._R = np.eye(3)             # 最近一次"开始计算"生效的旋转矩阵

        self._build_ui()
        self._reset_defaults()               # 输入回默认（对应 R = 单位阵）
        self._compute_from_rpy()             # 以默认输入立即算一次，界面初始即有结果

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        """搭建主界面：左 3D 画布 + 右控制面板（四组 + 全局重置）的横向布局。"""
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)

        # --- 左侧：3D 画布 ---
        self.fig = plt.figure(figsize=(7, 7))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._setup_axes()
        h.addWidget(self.canvas, stretch=3)

        # --- 右侧：控制面板（自上而下：RPY -> 轴角 -> 四元数 -> R 矩阵 -> 全局按钮） ---
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setSpacing(12)                    # 四个分区之间留出行距
        pv.addWidget(self._build_rpy_group())
        pv.addWidget(self._build_axis_angle_group())
        pv.addWidget(self._build_quat_group())
        pv.addWidget(self._build_matrix_group())
        pv.addLayout(self._build_button_row())
        pv.addStretch(1)
        h.addWidget(panel, stretch=2)

    def _setup_axes(self):
        """固定 3D 坐标系样式（等比例、正方向、固定视野）。"""
        ax = self.ax
        ax.set_xlim(-PLOT_LIM, PLOT_LIM)
        ax.set_ylim(-PLOT_LIM, PLOT_LIM)
        ax.set_zlim(-PLOT_LIM, PLOT_LIM)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=22, azim=-55)      # 设定观察视角（仰角 22°、方位 -55°）

    # ---------------- RPY 组：rx ry rz 数值框（度） ----------------
    def _build_rpy_group(self):
        """构建"RPY 角"输入组：Roll/Pitch/Yaw 三个数值框 + 万向锁示例 + 开始计算。"""
        box = QGroupBox("RPY 角  (rx ry rz, 单位:°)")
        g = QGridLayout(box)
        self.rpy_spins = []
        for i, name in enumerate(("rx (Roll)", "ry (Pitch)", "rz (Yaw)")):
            g.addWidget(QLabel(name), i, 0)
            sp = self._make_spin(RPY_MIN, RPY_MAX, 1)
            self.rpy_spins.append(sp)
            g.addWidget(sp, i, 1)
        btn_row = QHBoxLayout()
        b_gimbal = QPushButton("万向锁示例")
        b_gimbal.clicked.connect(self._apply_gimbal_preset)
        b_calc = QPushButton("开始计算")
        b_calc.clicked.connect(self._compute_from_rpy)
        btn_row.addWidget(b_gimbal)
        btn_row.addWidget(b_calc)
        g.addLayout(btn_row, 3, 0, 1, 2)
        hint = QLabel("万向锁示例：填入 (30, 90, 60) 后点\"开始计算\"，再改为 (0, 90, 30) 后点\"开始计算\" ：两组 RPY 角度不同但表示的姿态相同。")
        hint.setWordWrap(True)
        g.addWidget(hint, 4, 0, 1, 2)
        return box

    # ---------------- 轴角组：kx ky kz + theta 数值框 ----------------
    def _build_axis_angle_group(self):
        """构建"轴角"输入组：旋转轴 k=(kx,ky,kz) 三分量 + 转角 θ + 开始计算。"""
        box = QGroupBox("轴角  (k=(kx,ky,kz) 任意非零向量, θ, 单位:°)")
        g = QGridLayout(box)
        self.k_spins = []
        for i, name in enumerate(("kx", "ky", "kz")):
            g.addWidget(QLabel(name), i, 0)
            sp = self._make_spin(AXIS_MIN, AXIS_MAX, 4)
            self.k_spins.append(sp)
            g.addWidget(sp, i, 1)
        g.addWidget(QLabel("θ (转角)"), 3, 0)
        self.theta_spin = self._make_spin(THETA_MIN, THETA_MAX, 1)
        g.addWidget(self.theta_spin, 3, 1)
        btn = QPushButton("开始计算")
        btn.clicked.connect(self._compute_from_axis_angle)
        g.addWidget(btn, 4, 0, 1, 2)
        hint = QLabel("k 无需单位化，计算时自动归一化并回写本组；零向量无法确定方向，会提示且不计算。")
        hint.setWordWrap(True)
        g.addWidget(hint, 5, 0, 1, 2)
        return box

    # ---------------- 四元数组：w x y z 数值框 ----------------
    def _build_quat_group(self):
        """构建"四元数"输入组：w/x/y/z 四个数值框 + 开始计算。"""
        box = QGroupBox("四元数  q=(w,x,y,z)")
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        self.quat_spins = []
        for name in ("w", "x", "y", "z"):
            row.addWidget(QLabel(name))
            sp = self._make_spin(QUAT_MIN, QUAT_MAX, 4)
            self.quat_spins.append(sp)
            row.addWidget(sp)
        v.addLayout(row)
        btn = QPushButton("开始计算")
        btn.clicked.connect(self._compute_from_quat)
        v.addWidget(btn)
        hint = QLabel("无需单位化，计算时自动归一化并回写本组；q 与 -q 表示同一旋转。")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return box

    # ---------------- 旋转矩阵 R 组：3x3 可编辑表格 ----------------
    def _build_matrix_group(self):
        """构建"旋转矩阵 R"输入组：3x3 可编辑表格（列 = {B} 的三根轴）+ 开始计算。"""
        box = QGroupBox("旋转矩阵 R  (9 个单元格可直接编辑)")
        v = QVBoxLayout(box)
        self.table = QTableWidget(3, 3)
        self.table.setHorizontalHeaderLabels(["x(B)", "y(B)", "z(B)"])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(26)   # 固定行高，防布局压缩出滚动条
        # 固定为"表头 + 3 行"的高度：QSizePolicy.Maximum 会被布局压缩、裁掉第 3 行
        self.table.setFixedHeight(
            self.table.horizontalHeader().sizeHint().height()
            + 3 * 26 + self.table.frameWidth() * 2 + 2)
        for c in range(3):              # 三列等宽铺满面板，避免数值挤压或留白
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
        for r in range(3):
            for c in range(3):
                self.table.setItem(r, c, QTableWidgetItem("0"))
        v.addWidget(self.table)
        btn = QPushButton("开始计算")
        btn.clicked.connect(self._compute_from_matrix)
        v.addWidget(btn)
        hint = QLabel("直接编辑单元格（回车生效）；不正交时（三轴须单位长且两两垂直）"
                      "自动投影到最近的旋转矩阵并回写。")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return box

    # ---------------- 全局按钮行 ----------------
    def _build_button_row(self):
        """全局按钮行：重设默认值（四组输入回默认，仍需点某组\"开始计算\"生效）。"""
        row = QHBoxLayout()
        b_reset = QPushButton("重设默认值")
        b_reset.clicked.connect(self._reset_defaults)
        row.addWidget(b_reset)
        return row

    # ---------------- 控件工厂 ----------------
    def _make_spin(self, lo, hi, decimals):
        """构造一个数值输入框（指定范围与小数位）。"""
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSingleStep(0.1)
        sp.setValue(0.0)
        sp.setKeyboardTracking(False)   # 仅回车/失焦触发，防抖
        return sp

    # ------------------------------------------------------------------ 预设 / 重置
    def _apply_gimbal_preset(self):
        """万向锁示例预设：把 RPY 一键填入 (30, 90, 60)（仍需"开始计算"生效）。"""
        for sp, v in zip(self.rpy_spins, (30.0, 90.0, 60.0)):
            with QSignalBlocker(sp):
                sp.setValue(v)

    def _reset_defaults(self):
        """全局"重设默认值"：四组输入回默认（对应 R = 单位阵）。仅重置输入；
        3D 与各分区数值等"开始计算"再刷新。"""
        for sp, v in zip(self.rpy_spins, (0.0, 0.0, 0.0)):
            with QSignalBlocker(sp):
                sp.setValue(v)
        for sp, v in zip(self.k_spins, (1.0, 0.0, 0.0)):    # θ=0 时规范轴约定取 X 轴
            with QSignalBlocker(sp):
                sp.setValue(v)
        with QSignalBlocker(self.theta_spin):
            self.theta_spin.setValue(0.0)
        for sp, v in zip(self.quat_spins, (1.0, 0.0, 0.0, 0.0)):
            with QSignalBlocker(sp):
                sp.setValue(v)
        with QSignalBlocker(self.table):
            self._update_matrix_table(np.eye(3))

    # ------------------------------------------------------------------ 计算
    def _compute_from_rpy(self):
        """由 RPY 分区折算 R：三个角（度）-> 弧度 -> 旋转矩阵；本组保留输入原值不回写。"""
        rpy_deg = np.array([sp.value() for sp in self.rpy_spins])
        R = rpy_to_R(np.deg2rad(rpy_deg))
        self._refresh_from_R(R, write_rpy=False)

    def _compute_from_axis_angle(self):
        """由轴角分区折算 R：轴自动单位化（零向量无法确定方向，提示后不计算）。"""
        k = np.array([sp.value() for sp in self.k_spins])
        theta = np.deg2rad(self.theta_spin.value())
        if np.linalg.norm(k) < 1e-12:
            QMessageBox.warning(self, "输入错误", "旋转轴为零向量，无法确定方向，请修改后再试。")
            return
        R = axis_angle_to_R(k, theta)
        self._refresh_from_R(R)

    def _compute_from_quat(self):
        """由四元数分区折算 R：自动归一化（零四元数无法归一化，提示后不计算）。"""
        q = np.array([sp.value() for sp in self.quat_spins])
        if np.linalg.norm(q) < 1e-12:
            QMessageBox.warning(self, "输入错误", "四元数模长为 0，无法归一化，请修改后再试。")
            return
        R = quat_to_R(q)
        self._refresh_from_R(R)

    def _compute_from_matrix(self):
        """由矩阵分区折算 R：解析 9 个单元格 -> 校验/正交化 -> 回写各分区并重绘 3D。"""
        try:
            R = np.zeros((3, 3))
            for r in range(3):
                for c in range(3):
                    txt = self.table.item(r, c).text().strip() or "0"
                    R[r, c] = float(txt)
        except ValueError:
            QMessageBox.warning(self, "输入错误", "矩阵单元格存在无法识别的数值，请检查后重试。")
            return
        if not np.all(np.isfinite(R)):
            QMessageBox.warning(self, "输入错误", "矩阵单元格存在非有限数值（NaN/inf），请检查后重试。")
            return
        if not self._is_rotation(R):        # 不正交：投影到最近的旋转矩阵并提示
            R = self._orthonormalize(R)
            QMessageBox.information(
                self, "已自动正交化",
                "矩阵不是正交旋转矩阵（三轴须单位长且两两垂直），\n"
                "已自动投影到最接近的旋转矩阵进行计算。")
        self._refresh_from_R(R)

    @staticmethod
    def _is_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
        """判断 3x3 矩阵是否为正交旋转矩阵（R R^T = I 且 det(R) = +1）。"""
        return (np.allclose(R @ R.T, np.eye(3), atol=tol)
                and abs(np.linalg.det(R) - 1.0) < tol)

    @staticmethod
    def _orthonormalize(R: np.ndarray) -> np.ndarray:
        """SVD 投影到最近的旋转矩阵（det = +1），用于容错手改后的矩阵。"""
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:             # 反射时翻转最后一列，保证 det = +1
            U[:, -1] *= -1
            R = U @ Vt
        return R

    # ------------------------------------------------------------------ 回写与刷新
    def _refresh_from_R(self, R: np.ndarray, write_rpy: bool = True):
        """以 R 为唯一真相，把 R 的规范表示回写各分区，并重绘 3D。
        write_rpy=False 时跳过 RPY 分区（输入源为 RPY，保留用户输入原值）。
        所有回写均以 QSignalBlocker 阻断信号，防止触发额外刷新。"""
        self._R = R

        # RPY 分区：规范解（本组为输入源时保留原值，便于对比不同 RPY 对应同一姿态）
        if write_rpy:
            rpy_deg = np.degrees(R_to_rpy(R))
            for sp, v in zip(self.rpy_spins, rpy_deg):
                with QSignalBlocker(sp):
                    sp.setValue(v)

        # 轴角分区：单位轴 + 转角（θ 恒落在 [0°, 180°]）
        k, theta = R_to_axis_angle(R)
        for sp, v in zip(self.k_spins, k):
            with QSignalBlocker(sp):
                sp.setValue(v)
        with QSignalBlocker(self.theta_spin):
            self.theta_spin.setValue(np.degrees(theta))

        # 四元数分区：归一化后的单位四元数
        q = R_to_quat(R)
        for sp, v in zip(self.quat_spins, q):
            with QSignalBlocker(sp):
                sp.setValue(v)

        # 矩阵分区：3 位小数
        with QSignalBlocker(self.table):
            self._update_matrix_table(R)

        self._render()

    def _update_matrix_table(self, R: np.ndarray):
        """把 R 写入 3x3 表格（调用方须以 QSignalBlocker(self.table) 阻断 itemChanged）。"""
        for r in range(3):
            for c in range(3):
                # 先按显示精度取整再加 +0.0，消除 "-0.000" 的负零显示
                self.table.item(r, c).setText(f"{round(float(R[r, c]), 3) + 0.0:+.3f}")

    # ------------------------------------------------------------------ 渲染
    def _render(self):
        """重绘 3D 视图：清空 -> 画固定系 {A} -> 画被姿态驱动的 {B}。"""
        ax = self.ax
        ax.cla()
        self._setup_axes()

        self._draw_frame(ax, np.eye(4), dashed=True, prefix="{A}")

        T_B = np.eye(4)                  # 纯姿态：{B} 原点与 {A} 重合，只转动不平移
        T_B[:3, :3] = self._R
        self._draw_frame(ax, T_B, dashed=False, prefix="{B}")

        self.canvas.draw_idle()          # 异步触发重绘，不阻塞 UI

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


# ============================================================================
# 入口
# ============================================================================
def main():
    """程序入口：创建 Qt 应用 -> 显示主窗口 -> 进入事件循环。"""
    app = QApplication(sys.argv)              # 每个 Qt 程序需要一个 QApplication 对象
    win = RConvDemoWindow()
    win.show()                                # 显示窗口
    sys.exit(app.exec())                      # 进入 Qt 事件循环；退出时用状态码关闭进程


if __name__ == "__main__":
    main()
