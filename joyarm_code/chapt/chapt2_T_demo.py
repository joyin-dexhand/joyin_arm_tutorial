"""
==============================================================================
位姿变换可视化演示  (第二章 §1.3)
==============================================================================
【功能概要】
    用 PySide6 GUI + matplotlib 3D 演示齐次变换矩阵 ^A_B T 对坐标系 B 与点 P 的变换：
      - 固定参考系 {A}：原点 + RGB 虚线单位轴（红=X、绿=Y、蓝=Z），观察基准
      - 坐标系 {B}：原点 + RGB 实线坐标轴，被 ^A_B T 驱动
      - 点 P：定义在 {B} 下的点，坐标 ^B P = [x, y, z, 1]，随 {B} 被搬移

    右侧输入 ^A_B T 与点 P，点"开始计算"后刷新 3D 场景与结果区（P相对{B}和{A}的位置以及{B}相对{A}的位置与姿态）。

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    source .venv/bin/activate
    # 若 Linux 缺少 libxcb-cursor.so.0 ，则先安装：
    sudo apt-get install -y libxcb-cursor0

    # 2. 运行本脚本
    python ./chapt/chapt2_T_demo.py

    # 输入数值简单实例：T = 绕 Z 轴旋转 90° + 平移 (0.5, 0, 0)，P = (0.5, 0.5, 0.5)
    T = [[ 0, -1, 0, 0.5],
         [ 1,  0, 0, 0.0],
         [ 0,  0, 1, 0.0],
         [ 0,  0, 0, 1.0]]
    特点：{B} 的 x 轴转到 {A} 的 y 方向、y 轴转到 -x 方向、z 轴不变，原点沿 x 平移 0.5 m；
    点 P 被换算到 {A} 下 (0.0, 0.5, 0.5)，可在 3D 场景与结果区核对。

    # 注：RPY / 四元数的表示原理详见第二章 §2.1 / §2.3；矩阵 <-> RPY / 四元数的转换
    #     由核心库 joyarm_core.utils.transforms 提供（R = R_Z·R_Y·R_X，标准 RPY 约定）。

==============================================================================
"""

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QGroupBox, QDoubleSpinBox, QMessageBox, QSizePolicy, QHeaderView,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装时也可直接运行
from joyarm_core.utils.transforms import rpy_to_R, Rp_to_T, R_to_rpy, R_to_quat

# ----------------------------------------------------------------------------
# 全局常量
# ----------------------------------------------------------------------------
TRANS_MIN, TRANS_MAX = -2.0, 2.0        # 点 P 坐标输入范围 (m)
AXIS_COLORS = ("#e6194b", "#3cb44b", "#2b6cb0")  # RGB 对应 X/Y/Z
LINE_A2B = "#805ad5"                    # {A} 原点 -> {B} 原点连线颜色（紫）
LINE_B2P = "#718096"                    # {B} 原点 -> P 连线颜色（灰）
LINE_A2P = "#dd6b20"                    # {A} 原点 -> P 连线颜色（橙）
ROW4_BG = QColor("#e2e8f0")             # 矩阵第 4 行（固定行）底色
ROW4_FG = QColor("#1a202c")             # 矩阵第 4 行文字色（深色主题默认文字近白，浅底须固定深色）
MIN_VIEW = 1.8                          # 3D 视野最小半宽（内容很小时的下限）
VIEW_PAD = 0.6                          # 视野自适应余量（容纳轴端标签文字）


# ============================================================================
# 主窗口
# ============================================================================
class PoseDemoWindow(QMainWindow):
    """演示窗口：左侧 3D 画布 + 右侧控制面板（矩阵/点P/按钮/结果）。"""

    # 结果表第 1 列的项目名（其后各列为数值，3 位小数）
    RESULT_ROWS = (
        "P 相对 B 位置 (m)",
        "P 相对 A 位置 (m)",
        "B 相对 A 位置 (m)",
        "B 相对 A 姿态 RPY (°)",
        "B 相对 A 姿态 四元数 (wxyz)",
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("位姿变换可视化演示  (位姿矩阵 T 与点的坐标变换)")
        self.resize(1280, 760)

        self._T = np.eye(4)             # 最近一次"开始计算"生效的 ^A_B T
        self._p_B = np.zeros(3)         # 最近一次"开始计算"生效的 ^B P（前三维）

        self._build_ui()
        self._reset_defaults()               # 输入回默认（单位阵 + P=(0,1,0)）
        self._compute()                      # 以默认输入立即算一次，界面初始即有结果

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        """搭建主界面：左 3D 画布 + 右控制面板的横向布局。"""
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)

        # --- 左侧：3D 画布 ---
        self.fig = plt.figure(figsize=(7, 7))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._setup_axes()
        h.addWidget(self.canvas, stretch=3)

        # --- 右侧：控制面板（自上而下：矩阵 -> 点P -> 按钮 -> 结果） ---
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.addWidget(self._build_matrix_group())
        pv.addWidget(self._build_point_group())
        pv.addLayout(self._build_button_row())
        pv.addWidget(self._build_result_group())
        pv.addStretch(1)
        h.addWidget(panel, stretch=2)

    def _setup_axes(self):
        """固定 3D 坐标系样式（等比例、正方向、网格）；视野范围由 _render 自适应。"""
        ax = self.ax
        ax.set_box_aspect((1, 1, 1))    # 三轴等比例，避免视觉上被压扁
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=22, azim=-55)

    # ---------------- 4x4 矩阵表格 + 预设 ----------------
    def _build_matrix_group(self):
        """构建"位姿矩阵 T"分组：前 3 行可编辑表格 + 三类典型变换预设按钮；第 4 行固定。"""
        box = QGroupBox("位姿矩阵 T：{B} 相对 {A}（前 3 行可手改）")
        v = QVBoxLayout(box)
        self.table = QTableWidget(4, 4)
        self.table.setHorizontalHeaderLabels(["x(B)", "y(B)", "z(B)", "原点"])
        self.table.verticalHeader().setVisible(False)
        for c in range(4):              # 四列等宽铺满面板，避免数值挤压或留白
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
        for r in range(4):
            for c in range(4):
                item = QTableWidgetItem("0")
                if r == 3:              # 第 4 行：齐次坐标固定行 [0, 0, 0, 1]，不可编辑
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setBackground(ROW4_BG)
                    item.setForeground(ROW4_FG)
                    item.setText("1" if c == 3 else "0")
                self.table.setItem(r, c, item)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)  # 高度贴合 4 行
        v.addWidget(self.table)

        preset_row = QHBoxLayout()
        b_trans = QPushButton("纯平移")
        b_trans.clicked.connect(lambda: self._apply_preset("translate"))
        b_rot = QPushButton("纯旋转(绕Z)")
        b_rot.clicked.connect(lambda: self._apply_preset("rotate"))
        b_comp = QPushButton("复合变换")
        b_comp.clicked.connect(lambda: self._apply_preset("composite"))
        for b in (b_trans, b_rot, b_comp):
            preset_row.addWidget(b)
        v.addLayout(preset_row)

        hint = QLabel("直接编辑前 3 行单元格（回车生效）；预设按钮一键填入典型变换；"
                      "第 4 行为齐次坐标固定行，不可修改。")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return box

    # ---------------- 点 P 坐标 ----------------
    def _build_point_group(self):
        """构建"点 P"输入组：x/y/z 数值框可输入，第 4 元固定为 1（齐次坐标比例因子）。"""
        box = QGroupBox("点 P 在 {B} 系下的坐标 [x, y, z, 1]")
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        self.p_spins = []
        for name in ("x", "y", "z"):
            row.addWidget(QLabel(name))
            sp = self._make_spin(TRANS_MIN, TRANS_MAX, 2)
            self.p_spins.append(sp)
            row.addWidget(sp)
        one = QLabel("1（固定）")
        one.setStyleSheet("color: #718096;")        # 与矩阵固定行同风格的只读占位
        row.addWidget(one)
        v.addLayout(row)
        hint = QLabel("P 定义在 {B} 下：x/y/z 可输入（单位 m）；第 4 元为齐次坐标比例因子，固定为 1。")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return box

    # ---------------- 全局按钮行 ----------------
    def _build_button_row(self):
        """全局按钮行：重设默认值（T->单位阵、P->(0,1,0,1)）+ 开始计算（统一刷新）。"""
        row = QHBoxLayout()
        b_reset = QPushButton("重设默认值")
        b_reset.clicked.connect(self._reset_defaults)
        b_calc = QPushButton("开始计算")
        b_calc.clicked.connect(self._compute)
        row.addWidget(b_reset)
        row.addWidget(b_calc)
        return row

    # ---------------- 只读结果表 ----------------
    def _build_result_group(self):
        """构建"计算结果"只读表格：第 1 列项目名自适应，其余四列等宽铺满（数值不挤压）。"""
        box = QGroupBox("计算结果（只读）")
        v = QVBoxLayout(box)
        self.result = QTableWidget(len(self.RESULT_ROWS), 5)
        self.result.setHorizontalHeaderLabels(["项目", "值1", "值2", "值3", "值4"])
        self.result.verticalHeader().setVisible(False)
        self.result.setEditTriggers(QTableWidget.NoEditTriggers)
        header = self.result.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for c in range(1, 5):
            header.setSectionResizeMode(c, QHeaderView.Stretch)
        for r, label in enumerate(self.RESULT_ROWS):
            self.result.setItem(r, 0, QTableWidgetItem(label))
            for c in range(1, 5):
                cell = QTableWidgetItem("")
                cell.setTextAlignment(Qt.AlignCenter)
                self.result.setItem(r, c, cell)
        self.result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)  # 高度贴合 5 行
        v.addWidget(self.result)
        hint = QLabel("点击\"开始计算\"后按当前输入刷新；各行列序：xyz / r p y / w x y z。")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return box

    def _make_spin(self, lo, hi, decimals):
        """构造一个数值输入框（指定范围与小数位）。"""
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSingleStep(0.1)
        sp.setValue(0.0)
        sp.setKeyboardTracking(False)
        return sp

    # ------------------------------------------------------------------ 表格读写
    def _update_table(self, T: np.ndarray):
        """把 T 前 3 行写入矩阵表格（调用方须以 QSignalBlocker(self.table) 阻断 itemChanged）。"""
        for r in range(3):
            for c in range(4):
                self.table.item(r, c).setText(f"{T[r, c]:+.3f}")

    # ------------------------------------------------------------------ 预设 / 重置
    def _apply_preset(self, kind: str):
        """典型变换预设：把预设 T 一次性写入矩阵表格（仍需"开始计算"生效）。"""
        if kind == "translate":
            t, rpy = (1.0, 0.0, 0.5), (0.0, 0.0, 0.0)
        elif kind == "rotate":
            t, rpy = (0.0, 0.0, 0.0), (0.0, 0.0, 90.0)
        elif kind == "composite":        # 复合 = 上两者叠加：绕 Z 转 90° + 平移 (1, 0, 0.5)
            t, rpy = (1.0, 0.0, 0.5), (0.0, 0.0, 90.0)
        else:
            return
        T = Rp_to_T(rpy_to_R(np.deg2rad(rpy)), np.array(t))
        with QSignalBlocker(self.table):
            self._update_table(T)

    def _reset_defaults(self):
        """全局"重设默认值"：T -> 单位阵（表格回写）、P -> (0, 1, 0, 1)。仅重置输入；
        3D 与结果区等"开始计算"再刷新。"""
        with QSignalBlocker(self.table):
            self._update_table(np.eye(4))
        for sp, v in zip(self.p_spins, (0.0, 1.0, 0.0)):
            with QSignalBlocker(sp):
                sp.setValue(v)

    # ------------------------------------------------------------------ 计算
    @staticmethod
    def _is_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
        """判断 3x3 矩阵是否为正交旋转矩阵（R R^T = I 且 det(R) = +1）。"""
        return (np.allclose(R @ R.T, np.eye(3), atol=tol)
                and abs(np.linalg.det(R) - 1.0) < tol)

    @staticmethod
    def _orthonormalize(R: np.ndarray) -> np.ndarray:
        """SVD 投影到最近的旋转矩阵（det = +1），用于容错手改后的旋转块。"""
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:             # 反射时翻转最后一列，保证 det = +1
            U[:, -1] *= -1
            R = U @ Vt
        return R

    def _compute(self):
        """"开始计算"：解析输入 -> 校验/正交化 -> 算出各项结果 -> 填只读表 -> 重绘 3D。"""
        # 1. 解析矩阵前 3 行（第 4 行固定，不参与解析）
        try:
            vals = np.zeros((3, 4))
            for r in range(3):
                for c in range(4):
                    txt = self.table.item(r, c).text().strip() or "0"
                    vals[r, c] = float(txt)
        except ValueError:
            QMessageBox.warning(self, "输入错误", "矩阵单元格存在无法识别的数值，请检查后重试。")
            return
        if not np.all(np.isfinite(vals)):
            QMessageBox.warning(self, "输入错误", "矩阵单元格存在非有限数值（NaN/inf），请检查后重试。")
            return
        R, p_BA = vals[:3, :3], vals[:3, 3]

        # 2. 旋转块须为正交旋转矩阵（三轴单位长且两两垂直），否则 SVD 投影到最近旋转矩阵
        if not self._is_rotation(R):
            R = self._orthonormalize(R)
            with QSignalBlocker(self.table):
                self._update_table(Rp_to_T(R, p_BA))   # 校正后的有效矩阵回写表格
            QMessageBox.information(
                self, "已自动正交化",
                "矩阵左上 3x3 块不是正交旋转矩阵（三轴须单位长且两两垂直），\n"
                "已自动投影到最接近的旋转矩阵进行计算，并回写至上方矩阵表格。")

        # 3. 读点 P 前三维（第 4 元固定 1），计算并保存生效值
        p_B = np.array([sp.value() for sp in self.p_spins])
        self._T = Rp_to_T(R, p_BA)
        self._p_B = p_B
        p_A = R @ p_B + p_BA                # 坐标变换：^A P = ^A_B T · ^B P
        rpy_deg = np.degrees(R_to_rpy(R))
        quat = R_to_quat(R)

        # 4. 填只读结果表（"P 相对 B 位置"即输入回显，便于与"P 相对 A 位置"对照）
        def fmt(vec):
            # 先按显示精度取整再加 +0.0，消除 "-0.000" 的负零显示
            return [f"{round(float(v), 3) + 0.0:+.3f}" for v in vec]

        rows = (
            fmt(p_B) + [""],
            fmt(p_A) + [""],
            fmt(p_BA) + [""],
            fmt(rpy_deg) + [""],
            fmt(quat),
        )
        for r, values in enumerate(rows):
            for c, text in enumerate(values):
                self.result.item(r, c + 1).setText(text)

        # 5. 重绘 3D 场景
        self._render()

    # ------------------------------------------------------------------ 渲染
    def _render(self):
        """重绘 3D 视图：{A}/{B} 两系 + 三条指示虚线 + 点 P，视野随内容自适应。"""
        ax = self.ax
        ax.cla()
        self._setup_axes()
        self._draw_frame(ax, np.eye(4), dashed=True, prefix="{A}")
        self._draw_frame(ax, self._T, dashed=False, prefix="{B}")

        # 三条指示虚线：A->B（B 相对 A 的位置 ^A p_B）、B->P（^B P）、A->P（^A P）
        T = self._T
        oB = T[:3, 3]                       # {B} 原点在 {A} 下的位置
        pA = T[:3, :3] @ self._p_B + oB     # P 相对 A 的位置
        ax.plot([0.0, oB[0]], [0.0, oB[1]], [0.0, oB[2]],
                color=LINE_A2B, linestyle="--", linewidth=1.6,
                label=r"$\{A\}\rightarrow\{B\}$  ($^{A}p_{B}$)")
        ax.plot([oB[0], pA[0]], [oB[1], pA[1]], [oB[2], pA[2]],
                color=LINE_B2P, linestyle="--", linewidth=1.6,
                label=r"$\{B\}\rightarrow P$  ($^{B}p$)")
        ax.plot([0.0, pA[0]], [0.0, pA[1]], [0.0, pA[2]],
                color=LINE_A2P, linestyle="--", linewidth=1.6,
                label=r"$\{A\}\rightarrow P$  ($^{A}p$)")
        ax.scatter(*pA, color="k", s=30)    # 点 P 本体
        ax.text(pA[0], pA[1], pA[2], " P", color="k", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)

        # 视野自适应：覆盖两系轴端与 P，留标签余量；内容很小时取最小视野，避免过度放大
        pts = [oB, pA, [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]     # {A} 三轴端点 + B 原点 + P
        pts += [T[:3, i] + oB for i in range(3)]                  # {B} 三轴端点
        lim = max(MIN_VIEW, float(np.abs(np.asarray(pts)).max()) + VIEW_PAD)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        self.canvas.draw_idle()

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
    win = PoseDemoWindow()
    win.show()                                # 显示窗口
    sys.exit(app.exec())                      # 进入 Qt 事件循环；退出时用状态码关闭进程


if __name__ == "__main__":
    main()
