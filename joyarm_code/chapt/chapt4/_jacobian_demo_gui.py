"""第四章雅可比 Demo 的 PySide6 交互界面（章内实现，不属于公共层）。"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import font_manager
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QButtonGroup, QComboBox, QDoubleSpinBox, QGridLayout, QHeaderView,
    QGraphicsDropShadowEffect, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
    QSlider, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from chapt.common.exercise_metrics import frame_end_name, jacobian_metrics

_cjk_font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _cjk_font_path.exists():
    font_manager.fontManager.addfont(_cjk_font_path)
    _cjk_font = font_manager.FontProperties(fname=_cjk_font_path).get_name()
    matplotlib.rcParams["font.sans-serif"] = [_cjk_font, "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

MODE_COLORS = {"雅可比": "#2563eb", "奇异": "#dc2626", "静力学": "#059669"}
SINGULAR_PRESETS = {
    "腕部近奇异": [0, -35, -70, 0, -80.787, 0],
    "肩部近奇异": [-44.6, -73.0, -149.0, 25.5, 0, -89.7],
    "肘部伸展奇异": [0, -87.8, -170.7, 0, 0, 0],
}
SINGULAR_DESCRIPTIONS = (
    "第四轴与第六轴接近平行。精确重合约在 q5=-91.76°，超出该型号硬限位，因此预设采用软限位内最接近的安全姿态。",
    "腕心接近第一关节的基座轴线，第一关节转动对末端位置的作用减弱。",
    "上臂与前臂接近共线，肘部在伸展方向附近失去运动能力。",
)


class FloatSlider(QWidget):
    """带数值标签的浮点滑块。"""

    def __init__(self, name, low, high, value, suffix="", decimals=1):
        super().__init__()
        self.low, self.high, self.decimals = float(low), float(high), decimals
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(name)
        label.setFixedWidth(58)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 10000)
        self.value_label = QLabel()
        self.value_label.setFixedWidth(76)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.suffix = suffix
        layout.addWidget(label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self.set_value(value)
        self.slider.valueChanged.connect(self._show_value)

    def value(self):
        return self.low + (self.high - self.low) * self.slider.value() / 10000

    def set_value(self, value):
        ratio = (float(value) - self.low) / (self.high - self.low)
        self.slider.setValue(round(10000 * np.clip(ratio, 0, 1)))
        self._show_value()

    def _show_value(self):
        self.value_label.setText(f"{self.value():.{self.decimals}f}{self.suffix}")


class JacobianWindow(QMainWindow):
    """机械臂可视化与三种雅可比概念的联动窗口。"""

    def __init__(self, arm, regular_q, joint_velocity, wrench):
        super().__init__()
        self.arm, self.mode = arm, "雅可比"
        self.regular_q = np.asarray(regular_q)
        self.default_dq = np.asarray(joint_velocity)
        self.wrench = np.asarray(wrench, dtype=float).copy()
        self.frame = frame_end_name(arm)
        preset_q = np.deg2rad(next(iter(SINGULAR_PRESETS.values())))
        singular_jacobian = arm.get_jac(preset_q, self.frame)
        self.singular_target = 0.03 * np.linalg.svd(singular_jacobian)[0][:, -1]
        self.setWindowTitle("第四章｜雅可比、奇异性与静力学交互演示")
        self.resize(1400, 840)
        self.setStyleSheet("""
            QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: #f5f6f8; }
            QGroupBox { background: white; border: none; border-radius: 9px;
                        margin-top: 14px; padding: 12px 10px 10px 10px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { min-height: 28px; border: none; border-radius: 5px;
                          background: #f2f4f7; padding: 2px 12px; }
            QPushButton:hover { background: #f1f5f9; }
            QComboBox, QDoubleSpinBox { min-height: 28px; border: 1px solid transparent;
                                      border-radius: 5px; background: #f2f4f7; padding: 0 8px; }
            QComboBox:hover, QDoubleSpinBox:hover { background: #eceff3; }
            QComboBox:focus, QDoubleSpinBox:focus { border: 1px solid #94a3b8; background: white; }
            QTableWidget { border: none; alternate-background-color: #f9f9f9; background: white; }
            QHeaderView::section { background: white; border: none; border-bottom: 1px solid #e0e0e0;
                                   padding: 4px; color: #64748b; font-size: 12px; font-weight: 600; }
            QSlider::groove:horizontal { height: 4px; background: #d8dde5; border-radius: 2px; }
            QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px; }
        """)
        self._base_style = self.styleSheet()
        self._build_ui()
        self._connect()
        self._change_mode(0)
        self._set_q(self.regular_q)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        page = QHBoxLayout(root)
        page.setContentsMargins(10, 10, 10, 10)
        page.setSpacing(12)
        left = QVBoxLayout()
        left.setSpacing(8)
        page.addLayout(left, 3)
        buttons = QHBoxLayout()
        left.addLayout(buttons)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for index, name in enumerate(MODE_COLORS):
            button = QPushButton(name)
            button.setCheckable(True)
            button.setFixedHeight(36)
            button.setStyleSheet(f"QPushButton {{background:transparent;border:none;border-radius:0;font-size:14px;color:#475569;}} QPushButton:hover {{background:#f1f5f9;}} QPushButton:checked {{background:transparent;color:{MODE_COLORS[name]};border-bottom:3px solid {MODE_COLORS[name]};font-weight:bold;}}")
            self.mode_group.addButton(button, index)
            buttons.addWidget(button)
            if index == 0:
                button.setChecked(True)
        self.meshcat_view = None
        try:
            from chapt.chapt4._meshcat_demo_view import MeshcatDemoView
            self.meshcat_view = MeshcatDemoView(self)
            left.addWidget(self.meshcat_view, 1)
        except Exception as error:
            warnings.warn(f"MeshCat 嵌入失败，回退到连杆视图：{error}", RuntimeWarning)
            self.figure = plt.Figure(figsize=(8, 7), tight_layout=True, facecolor="#f5f6f8")
            self.canvas = FigureCanvas(self.figure)
            self.axes = self.figure.add_subplot(111, projection="3d")
            left.addWidget(self.canvas, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page.addWidget(scroll, 2)
        panel = QWidget()
        scroll.setWidget(panel)
        self.right = QVBoxLayout(panel)
        self.right.setContentsMargins(12, 8, 12, 8)
        self.right.setSpacing(14)
        self.title = QLabel()
        self.title.setMinimumHeight(32)
        self.title.setStyleSheet("font-size:20px;font-weight:bold")
        self.summary = QLabel()
        self.summary.setMinimumHeight(42)
        self.summary.setWordWrap(True)
        self.right.addWidget(self.title)
        self.right.addWidget(self.summary)
        self._build_joint_controls()
        self._build_mode_panels()
        self.right.addStretch()

    def _build_joint_controls(self):
        box = QGroupBox("关节角度（拖动后所有结果实时更新）")
        self._add_shadow(box)
        grid = QGridLayout(box)
        limits = self.arm.joint_limits_soft_arm
        self.q_sliders = []
        for i in range(self.arm.n_arm):
            slider = FloatSlider(f"关节 {i + 1}", np.rad2deg(limits.q_min[i]), np.rad2deg(limits.q_max[i]), 0, "°")
            self.q_sliders.append(slider)
            grid.addWidget(slider, i % 3, i // 3)
        reset = QPushButton("恢复常规位形")
        reset.clicked.connect(lambda: self._set_q(self.regular_q))
        grid.addWidget(reset, 3, 0, 1, 2)
        self.right.addWidget(box)

    def _build_mode_panels(self):
        self.velocity_box = QGroupBox("关节速度与末端速度")
        self._add_shadow(self.velocity_box)
        layout = QVBoxLayout(self.velocity_box)
        layout.setSpacing(9)
        formula = QLabel("<span style='font-family:STIX Two Math,serif;font-size:20px'><i>V</i> = <i>J</i>(<i>q</i>)q̇ = <i>J</i><sub>1</sub>q̇<sub>1</sub> + <i>J</i><sub>2</sub>q̇<sub>2</sub> + ··· + <i>J</i><sub>6</sub>q̇<sub>6</sub></span>")
        formula.setAlignment(Qt.AlignCenter)
        formula.setStyleSheet("font-size:18px;font-weight:bold")
        layout.addWidget(formula)
        speed_grid = QGridLayout()
        layout.addLayout(speed_grid)
        self.dq_sliders = []
        for i, value in enumerate(np.rad2deg(self.default_dq)):
            slider = FloatSlider(f"dq{i + 1}", -30, 30, value, "°/s")
            self.dq_sliders.append(slider)
            speed_grid.addWidget(slider, i % 3, i // 3)
        components = ["vx", "vy", "vz", "ωx", "ωy", "ωz"]
        self.jacobian_table = self._table(6, 6, components, [f"J{i}" for i in range(1, 7)], 150)
        layout.addWidget(QLabel("雅可比矩阵 J（每一列对应一个关节）"))
        layout.addWidget(self.jacobian_table)
        self.velocity_table = self._table(6, 7, components, [*(f"J{i}q̇{i}" for i in range(1, 7)), "合成 V"], 150)
        layout.addWidget(QLabel("列贡献相加得到末端速度（前三行为 m/s，后三行为 rad/s）"))
        layout.addWidget(self.velocity_table)

        self.singular_box = QGroupBox("奇异位形与逆速度放大")
        self._add_shadow(self.singular_box)
        layout = QVBoxLayout(self.singular_box)
        layout.setSpacing(11)
        self.preset = QComboBox()
        self.preset.addItems(SINGULAR_PRESETS)
        self.preset.setMaximumWidth(320)
        jump = QPushButton("跳转到所选奇异位形")
        jump.clicked.connect(self._jump_singular)
        layout.addWidget(self.preset)
        layout.addWidget(jump)
        self.preset_note = QLabel()
        self.preset_note.setWordWrap(True)
        layout.addWidget(self.preset_note)
        chain = QLabel("σmin ↓   →   条件数 ↑   →   普通伪逆速度放大\n阻尼伪逆：限制放大，但不能消除几何奇异性")
        chain.setAlignment(Qt.AlignCenter)
        chain.setStyleSheet("font-size:18px;font-weight:bold;color:#991b1b;background:#fff1f2;border-radius:6px;padding:10px")
        layout.addWidget(chain)
        cards = QGridLayout()
        self.singular_cards = []
        for index, caption in enumerate(("最小奇异值 σmin", "条件数 κ", "普通伪逆 ‖q̇‖", "阻尼伪逆 ‖q̇‖")):
            card = QLabel()
            card.setAlignment(Qt.AlignCenter)
            card.setMinimumHeight(92)
            card.setStyleSheet("font-size:16px;background:#f3f4f6;border:1px solid #d1d5db;border-radius:6px")
            card.setToolTip(caption)
            self.singular_cards.append(card)
            cards.addWidget(card, index // 2, index % 2)
        layout.addLayout(cards)
        note = QLabel("选择预设时确定一个固定的期望末端速度；随后拖动关节，比较同一任务在不同位形下需要的关节速度。")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.statics_box = QGroupBox("末端载荷与关节力矩")
        self._add_shadow(self.statics_box)
        layout = QVBoxLayout(self.statics_box)
        layout.setSpacing(12)
        wrench_grid = QGridLayout()
        wrench_grid.setHorizontalSpacing(12)
        wrench_grid.setVerticalSpacing(8)
        self.wrench_inputs = []
        names = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
        for i, (name, value) in enumerate(zip(names, self.wrench)):
            spin = QDoubleSpinBox()
            spin.setRange(-100, 100)
            spin.setDecimals(2)
            spin.setValue(value)
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spin.setFont(QFont("DejaVu Sans", 11, QFont.DemiBold))
            spin.setMinimumWidth(88)
            spin.setMaximumWidth(108)
            spin.setSuffix(" N" if i < 3 else " N·m")
            spin.setAlignment(Qt.AlignRight)
            self.wrench_inputs.append(spin)
            wrench_grid.addWidget(QLabel(name), i // 3, 2 * (i % 3))
            wrench_grid.addWidget(spin, i // 3, 2 * (i % 3) + 1)
        layout.addLayout(wrench_grid)
        layout.addWidget(QLabel("关节力矩（N·m）"))
        self.statics_table = self._table(6, 2, [f"关节{i}" for i in range(1, 7)], ["产生力矩", "保持力矩"])
        layout.addWidget(self.statics_table)
        self.capacity_box = QGroupBox("工程检查：保持力矩占电机最大力矩的比例")
        self.capacity_box.setCheckable(True)
        self.capacity_box.setChecked(False)
        capacity_layout = QVBoxLayout(self.capacity_box)
        self.capacity_note = QLabel("占用率 = |保持力矩| / 最大允许力矩 × 100%。它不是静力学公式本身，只用于判断当前负载是否接近硬件能力。")
        self.capacity_note.setWordWrap(True)
        self.capacity_table = self._table(6, 1, [f"关节{i}" for i in range(1, 7)], ["力矩占用率 %"], 190)
        capacity_layout.addWidget(self.capacity_note)
        capacity_layout.addWidget(self.capacity_table)
        self.capacity_box.toggled.connect(self._toggle_capacity)
        self._toggle_capacity(False)
        layout.addWidget(self.capacity_box)
        for box in (self.velocity_box, self.singular_box, self.statics_box):
            self.right.addWidget(box)

    @staticmethod
    def _table(rows, columns, row_names, column_names, height=225):
        table = QTableWidget(rows, columns)
        table.setVerticalHeaderLabels(row_names)
        table.setHorizontalHeaderLabels(column_names)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setFont(QFont("DejaVu Sans Mono", 9))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setFixedHeight(height)
        return table

    @staticmethod
    def _add_shadow(widget):
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(16)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(15, 23, 42, 24))
        widget.setGraphicsEffect(shadow)

    def _toggle_capacity(self, visible):
        self.capacity_note.setVisible(visible)
        self.capacity_table.setVisible(visible)

    def _connect(self):
        self.mode_group.idClicked.connect(self._change_mode)
        for slider in self.q_sliders + self.dq_sliders:
            slider.slider.valueChanged.connect(self.update_view)
        for spin in self.wrench_inputs:
            spin.valueChanged.connect(self.update_view)

    def _change_mode(self, index):
        self.mode = tuple(MODE_COLORS)[index]
        color = MODE_COLORS[self.mode]
        self.setStyleSheet(self._base_style + f"QSlider::sub-page:horizontal {{background:{color};border-radius:2px;}} QSlider::handle:horizontal {{background:{color};}}")
        self.update_view()

    def _q(self):
        return np.deg2rad([slider.value() for slider in self.q_sliders])

    def _dq(self):
        return np.deg2rad([slider.value() for slider in self.dq_sliders])

    def _set_q(self, q):
        for slider, value in zip(self.q_sliders, np.rad2deg(q)):
            slider.set_value(value)
        self.update_view()

    def _jump_singular(self):
        q = np.deg2rad(SINGULAR_PRESETS[self.preset.currentText()])
        jacobian = self.arm.get_jac(q, self.frame)
        self.singular_target = 0.03 * np.linalg.svd(jacobian)[0][:, -1]
        self.preset_note.setText(SINGULAR_DESCRIPTIONS[self.preset.currentIndex()])
        self._set_q(q)

    @staticmethod
    def _fill(table, values, signed_colors=False):
        for row, values_row in enumerate(np.asarray(values)):
            for column, value in enumerate(values_row):
                item = QTableWidgetItem(f"{value:.4f}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if signed_colors and value != 0:
                    item.setForeground(QBrush(QColor("#9f1239" if value > 0 else "#1e40af")))
                table.setItem(row, column, item)

    def update_view(self, *_):
        q = self._q()
        jacobian = self.arm.get_jac(q, self.frame)
        self.velocity_box.setVisible(self.mode == "雅可比")
        self.singular_box.setVisible(self.mode == "奇异")
        self.statics_box.setVisible(self.mode == "静力学")
        self.title.setText(f"{self.mode}模式")
        if self.mode == "雅可比":
            contributions = (jacobian * self._dq()[None, :]).T
            twist = contributions.sum(axis=0)
            self._fill(self.jacobian_table, jacobian)
            self._fill(self.velocity_table, np.column_stack((contributions.T, twist)))
            self.summary.setText(f"J(q)·dq 将 6 个关节贡献相加。当前末端线速度 ‖v‖={np.linalg.norm(twist[:3]):.4f} m/s，角速度 ‖ω‖={np.linalg.norm(twist[3:]):.4f} rad/s。")
        elif self.mode == "奇异":
            sigma, condition = jacobian_metrics(jacobian)
            pinv = np.linalg.pinv(jacobian) @ self.singular_target
            damped = self.arm.get_jac_pinv_damped(q, self.frame, damping=0.03) @ self.singular_target
            sigma_text = f"{sigma:.3e}" if sigma < 1e-3 else f"{sigma:.6f}"
            condition_text = "∞" if not np.isfinite(condition) else f"{condition:.1f}"
            values = (("最小奇异值 σmin", sigma_text, ""), ("条件数 κ", condition_text, ""),
                      ("普通伪逆", f"{np.linalg.norm(pinv):.3f}", "rad/s"),
                      ("阻尼伪逆", f"{np.linalg.norm(damped):.3f}", "rad/s"))
            for card, value in zip(self.singular_cards, values):
                label, number, unit = value
                card.setText(f'<span style="font-size:13px;color:#64748b">{label}</span><br><span style="font-size:25px;font-weight:700">{number}</span> <span style="font-size:12px">{unit}</span>')
            self.singular_cards[2].setStyleSheet("font-size:22px;font-weight:bold;background:#fee2e2;color:#991b1b;border:2px solid #ef4444;border-radius:6px")
            self.singular_cards[3].setStyleSheet("font-size:22px;font-weight:bold;background:#dcfce7;color:#166534;border:2px solid #22c55e;border-radius:6px")
            self.summary.setText("拖动关节或跳转预设：σmin 越小、条件数越大，普通伪逆速度越容易放大；阻尼伪逆会牺牲少量精度来限制放大。")
        else:
            wrench = np.array([spin.value() for spin in self.wrench_inputs])
            tau_external = jacobian.T @ wrench
            tau_motor = -tau_external
            utilization = 100 * np.abs(tau_motor) / self.arm.joint_limits_hard_arm.tau_max
            self._fill(self.statics_table, np.column_stack((tau_external, tau_motor)), signed_colors=True)
            self._fill(self.capacity_table, utilization[:, None])
            self.summary.setText("τ外 = JᵀF。末端外力会在各关节产生力矩；为了保持机械臂不动，电机必须提供大小相等、方向相反的保持力矩。")
        self._draw_robot(q, jacobian)

    def _draw_robot(self, q, jacobian):
        model = self.arm.pin_model
        data = model.createData()
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        points = [np.zeros(3)] + [data.oMi[model.getJointId(f"joint{i}")].translation.copy() for i in range(1, self.arm.n_arm + 1)]
        points.append(data.oMf[model.getFrameId(self.frame)].translation.copy())
        points = np.asarray(points)
        tcp = points[-1]
        if self.meshcat_view is not None:
            contributions = (jacobian * self._dq()[None, :]).T if self.mode == "雅可比" else None
            force = np.array([spin.value() for spin in self.wrench_inputs[:3]]) if self.mode == "静力学" else None
            self.meshcat_view.update_scene(q, tcp, self.mode, contributions=contributions, force=force)
            return
        ax = self.axes
        ax.clear()
        ax.set_facecolor("#fcfcfd")
        color = MODE_COLORS[self.mode]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], "-o", color=color, linewidth=5, markersize=8)
        ax.scatter(*tcp, color="black", s=55, label="TCP")
        if self.mode == "雅可比":
            contributions = (jacobian * self._dq()[None, :]).T
            for item in contributions:
                ax.quiver(*tcp, *(2.5 * item[:3]), color="#93c5fd", alpha=.65)
            ax.quiver(*tcp, *(2.5 * contributions.sum(axis=0)[:3]), color="#111827", linewidth=3, label="合成线速度")
        elif self.mode == "静力学":
            force = np.array([spin.value() for spin in self.wrench_inputs[:3]])
            if np.linalg.norm(force):
                ax.quiver(*tcp, *(0.012 * force), color="#dc2626", linewidth=3, label="末端力")
        center = points.mean(axis=0)
        radius = max(np.ptp(points, axis=0).max() * .65, .35)
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(max(-.15, center[2]-radius), center[2]+radius)
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        ax.set_zlabel("Z / m")
        ax.set_title(f"JoyArm 机械臂｜{self.mode}")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = (0.72, 0.76, 0.82, 0.20)
            axis.set_pane_color((0.985, 0.988, 0.995, 1.0))
        ax.grid(True)
        legend = ax.legend(loc="upper left", fontsize=8, framealpha=.62, borderpad=.4, handlelength=1.5)
        legend.get_frame().set_edgecolor("none")
        ax.view_init(elev=25, azim=-55)
        self.canvas.draw_idle()

    def closeEvent(self, event):
        if self.meshcat_view is not None:
            self.meshcat_view.shutdown()
        super().closeEvent(event)


def run_gui(arm, regular_q, joint_velocity, wrench):
    """启动第四章交互窗口。"""
    app = QApplication.instance()
    if app is None:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        app = QApplication(sys.argv)
    window = JacobianWindow(arm, regular_q, joint_velocity, wrench)
    window.show()
    app.exec()
