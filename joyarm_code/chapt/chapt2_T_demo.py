"""
==============================================================================
位姿变换可视化演示  (第二章 §1.3)
==============================================================================
【功能概要】
    用 PySide6 GUI + matplotlib 3D 演示齐次变换矩阵 ^A_B T 对空间的变换效果：
      - 固定参考系 {A}：原点 + RGB 虚线单位轴（红=X、绿=Y、蓝=Z），观察基准
      - 单位球固连系 {B}：原点 + RGB 实线坐标轴 + 半透明线框单位球，被 ^A_B T 驱动
    通过滑块/矩阵表格实时修改 ^A_B T，可直观观察四类典型变换：
      纯平移 / 纯旋转(绕单轴) / 复合变换 / 连乘级联。

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
    python chapt2_T_demo.py

    # T矩阵数值简单实例：绕 Z 轴旋转 90° + 平移 (0.5, 0.0, 0.0)
    输入下面数值到表格中，观察效果：
    T = [[ 0, -1, 0, 0.5],
         [ 1,  0, 0, 0.0],
         [ 0,  0, 1, 0.0],
         [ 0,  0, 0, 1.0]]

    特点：x轴转到了y轴（0 1 0），y轴转到了-x轴 （-1 0 0），z轴不变 （0 0 1），原点沿 x 平移了0.5米

    # 注：滑块控制的旋转矩阵 R = R_Z(rz)·R_Y(ry)·R_X(rx) —— 按 x-y-z 顺序的外旋（等价于按 z-y-x 顺序的内旋），即标准 RPY。具体原理详见欧拉角和固定角的讲解。

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
TRANS_MIN, TRANS_MAX = -2.0, 2.0        # 平移滑块范围 (m)
ROT_MIN, ROT_MAX = -180, 180           # 旋转滑块范围 (度)
SLIDER_STEPS = 400                      # 滑块离散步数（越大越平滑）
AXIS_COLORS = ("#e6194b", "#3cb44b", "#2b6cb0")   # RGB 对应 X/Y/Z
PLOT_LIM = 2.4                          # 3D 坐标轴显示范围


# ----------------------------------------------------------------------------
# 旋转矩阵（与第二章 §2.1 公式一致：RPY = R_Z * R_Y * R_X，标准 RPY / 外旋 XYZ，单位：弧度）
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
    """由 RPY 角（弧度）合成旋转矩阵 R = R_Z(rz) R_Y(ry) R_X(rx)（标准 RPY：外旋 XYZ）。"""
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """由 3x3 旋转矩阵与 3x1 位置向量拼成 4x4 齐次变换矩阵 ^A_B T。"""
    T = np.eye(4)      
    T[:3, :3] = R     
    T[:3, 3] = p       
    return T


# ----------------------------------------------------------------------------
# 单位球线框顶点：用于可视化 {B} 的姿态（球随 T 旋转后形状改变）
# ----------------------------------------------------------------------------
def unit_ball_lines():
    """返回单位球上若干经/纬线折线段的顶点列表；每个元素是 (n,3) 的点序列，
    绘制时逐相邻两点连线，并对整体做 T 变换。"""
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
class PoseDemoWindow(QMainWindow):
    """演示窗口：左侧 3D 画布 + 右侧控制面板（滑块/表格/按钮）。"""

    def __init__(self):
        super().__init__()         
        self.setWindowTitle("位姿变换可视化演示  (位姿矩阵 T)")
        self.resize(1280, 760)

        self._T = np.eye(4)
        self._cascade: list[np.ndarray] = []

        self._build_ui()      
        self._refresh_from_params()   

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

        # --- 右侧：控制面板 ---
        panel = QWidget()
        pv = QVBoxLayout(panel)          
        pv.addWidget(self._build_transform_group())  
        pv.addWidget(self._build_matrix_group())    
        pv.addWidget(self._build_action_group())   
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
        ax.view_init(elev=22, azim=-55)   

    # ---------------- 平移/旋转 滑块 + 数值输入组 ----------------
    def _build_transform_group(self):
        """构建"变换参数"分组：tx/ty/tz 平移 + rx/ry/rz 旋转，各带滑块和数值框。"""
        box = QGroupBox("变换参数（滑块拖动 / 数值输入）")
        g = QGridLayout(box)             

        self.t_sliders = []          
        self.t_spins = []              
        for i, name in enumerate(("tx", "ty", "tz")):
            g.addWidget(QLabel(f"平移 {name} (m)"), i, 0)
            s = QSlider(Qt.Horizontal)  
            s.setMinimum(0)
            s.setMaximum(SLIDER_STEPS)
            s.setValue(SLIDER_STEPS // 2)    
            s.valueChanged.connect(lambda v, idx=i: self._on_slider_changed("t", idx, v))
            g.addWidget(s, i, 1)
            sp = self._make_spin(TRANS_MIN, TRANS_MAX, 2)
            sp.valueChanged.connect(lambda v, idx=i: self._on_spin_changed("t", idx, v))
            g.addWidget(sp, i, 2)
            self.t_sliders.append(s)
            self.t_spins.append(sp)

        self.r_sliders = []
        self.r_spins = []
        for i, name in enumerate(("rx", "ry", "rz")):
            g.addWidget(QLabel(f"旋转 {name} (°)"), i + 3, 0)
            s = QSlider(Qt.Horizontal)
            s.setMinimum(0)
            s.setMaximum(SLIDER_STEPS)
            s.setValue(SLIDER_STEPS // 2)
            s.valueChanged.connect(lambda v, idx=i: self._on_slider_changed("r", idx, v))
            g.addWidget(s, i + 3, 1)
            sp = self._make_spin(ROT_MIN, ROT_MAX, 1)  
            sp.valueChanged.connect(lambda v, idx=i: self._on_spin_changed("r", idx, v))
            g.addWidget(sp, i + 3, 2)
            self.r_sliders.append(s)
            self.r_spins.append(sp)

        reset_row = QHBoxLayout()
        b_reset_t = QPushButton("重置 txyz")
        b_reset_t.clicked.connect(lambda: self._reset_group("t"))
        b_reset_r = QPushButton("重置 rxyz")
        b_reset_r.clicked.connect(lambda: self._reset_group("r"))
        b_reset_all = QPushButton("全部重置")
        b_reset_all.clicked.connect(self._reset_identity)
        for b in (b_reset_t, b_reset_r, b_reset_all):
            reset_row.addWidget(b)
        g.addLayout(reset_row, 6, 0, 1, 3)   

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

    # ---------------- 4x4 矩阵表格 ----------------
    def _build_matrix_group(self):
        """构建"位姿矩阵 T"表格：4x4 单元格可直接编辑，与滑块双向同步。"""
        box = QGroupBox("位姿矩阵 T（可手改）")
        v = QVBoxLayout(box)
        self.table = QTableWidget(4, 4)   
        self.table.setHorizontalHeaderLabels(["x̂_B", "ŷ_B", "ẑ_B", "原点"])
        self.table.verticalHeader().setVisible(False)
        for r in range(4):
            for c in range(4):
                self.table.setItem(r, c, QTableWidgetItem("0"))
        self.table.itemChanged.connect(self._on_table_edited)
        v.addWidget(self.table)
        hint = QLabel("直接编辑单元格数值（回车生效）；与左侧滑块双向同步。")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return box

    # ---------------- 预设 / 动作按钮 ----------------
    def _build_action_group(self):
        """构建"四类典型变换 / 操作"按钮组：预设、级联、重置。"""
        box = QGroupBox("四类典型变换 / 操作")
        v = QVBoxLayout(box)

        row1 = QHBoxLayout()
        b_trans = QPushButton("纯平移")
        b_trans.clicked.connect(lambda: self._apply_preset("translate"))
        b_rot = QPushButton("纯旋转(绕Y)")
        b_rot.clicked.connect(lambda: self._apply_preset("rotate"))
        row1.addWidget(b_trans)
        row1.addWidget(b_rot)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        b_comp = QPushButton("复合变换")
        b_comp.clicked.connect(lambda: self._apply_preset("composite"))
        b_add = QPushButton("添加为级联步骤")
        b_add.clicked.connect(self._add_cascade_step)
        row2.addWidget(b_comp)
        row2.addWidget(b_add)
        v.addLayout(row2)

        row3 = QHBoxLayout()
        b_clear = QPushButton("清空级联")
        b_clear.clicked.connect(self._clear_cascade)
        b_reset = QPushButton("重置为单位阵")
        b_reset.clicked.connect(self._reset_identity)
        row3.addWidget(b_clear)
        row3.addWidget(b_reset)
        v.addLayout(row3)

        self.lbl_cascade = QLabel("级联步骤数：0")
        v.addWidget(self.lbl_cascade)
        return box

    # ------------------------------------------------------------------ 数值 <-> 滑块 互转
    @staticmethod
    def _trans_to_slider(v: float) -> int:
        """平移量 (m) -> 滑块离散值。把连续区间线性映射到 [0, SLIDER_STEPS]。"""
        return int(round((v - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * SLIDER_STEPS))

    @staticmethod
    def _rot_to_slider(v: float) -> int:
        """旋转角 (度) -> 滑块离散值。同样线性映射到滑块整数范围。"""
        return int(round((v - ROT_MIN) / (ROT_MAX - ROT_MIN) * SLIDER_STEPS))

    def _slider_value(self, kind, idx):
        """读取某轴的"真实数值"（从滑块反算，保证与拖动一致）。"""
        s = self.t_sliders[idx] if kind == "t" else self.r_sliders[idx]
        lo, hi = (TRANS_MIN, TRANS_MAX) if kind == "t" else (ROT_MIN, ROT_MAX)
        t = s.value() / SLIDER_STEPS     
        return lo + t * (hi - lo)        

    def _on_slider_changed(self, kind, idx, _v):
        """滑块拖动 -> 同步数值框 -> 重算 T。"""
        val = self._slider_value(kind, idx)
        spins = self.t_spins if kind == "t" else self.r_spins
        with QSignalBlocker(spins[idx]):    
            spins[idx].setValue(val)
        self._refresh_from_params()

    def _on_spin_changed(self, kind, idx, val):
        """数值框输入 -> 同步滑块 -> 重算 T。"""
        sliders = self.t_sliders if kind == "t" else self.r_sliders
        to_slider = self._trans_to_slider if kind == "t" else self._rot_to_slider
        with QSignalBlocker(sliders[idx]):   
            sliders[idx].setValue(to_slider(val))
        self._refresh_from_params()

    def _refresh_from_params(self):
        """由当前所有滑块/数值框重新计算 T，刷新表格与 3D 视图。"""
        p = np.array([self._slider_value("t", i) for i in range(3)])
        degs = [self._slider_value("r", i) for i in range(3)]
        rads = np.deg2rad(degs)     
        R = rpy_to_matrix(*rads)           
        self._T = make_T(R, p)              
        self._update_table(silent=True)     
        self._render()                     

    def _reset_group(self, kind):
        """重置某一组（t 平移 / r 旋转）归零。"""
        sliders = self.t_sliders if kind == "t" else self.r_sliders
        spins = self.t_spins if kind == "t" else self.r_spins
        mid = SLIDER_STEPS // 2          
        for s, sp in zip(sliders, spins):
            with QSignalBlocker(s), QSignalBlocker(sp):  
                s.setValue(mid)
                sp.setValue(0.0)
        self._refresh_from_params()

    def _on_table_edited(self, _item):
        """表格手改后，解析为 T、最近正交化，并回写滑块/数值框（双向同步）。"""
        try:
            vals = np.zeros((4, 4))
            for r in range(4):
                for c in range(4):
                    txt = self.table.item(r, c).text().strip() or "0"
                    vals[r, c] = float(txt)
            R = vals[:3, :3]
            p = vals[:3, 3]

            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            if np.linalg.det(R) < 0:      
                U[:, -1] *= -1     
                R = U @ Vt
        except ValueError:
            return  # 非法输入忽略
        rx, ry, rz = self._matrix_to_rpy(R)
        tvals = np.clip(p, TRANS_MIN, TRANS_MAX)
        rvals = np.clip(np.degrees([rx, ry, rz]), ROT_MIN, ROT_MAX)
        for s, sp, v in zip(self.t_sliders, self.t_spins, tvals):
            with QSignalBlocker(s), QSignalBlocker(sp):
                s.setValue(self._trans_to_slider(float(v)))
                sp.setValue(float(v))
        for s, sp, v in zip(self.r_sliders, self.r_spins, rvals):
            with QSignalBlocker(s), QSignalBlocker(sp):
                s.setValue(self._rot_to_slider(float(v)))
                sp.setValue(float(v))
        self._T = make_T(R, p)
        self._render()

    @staticmethod
    def _matrix_to_rpy(R: np.ndarray):
        """由旋转矩阵反算固定轴 XYZ 的 RPY 角（弧度，与核心库 R_to_rpy 及
        chapt2_pose_demo 同一约定）；与第二章 §2.5 公式一致。GUI 显示层就地转度。"""
        ry = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
        rz = np.arctan2(R[1, 0], R[0, 0])
        rx = np.arctan2(R[2, 1], R[2, 2])
        return rx, ry, rz

    def _update_table(self, silent: bool):
        """把当前 self._T 写入表格。silent=True 时阻断 itemChanged 信号防递归。"""
        if silent:
            self.table.blockSignals(True)     
        T = self._T
        labels = (
            f"{T[0,0]:+.3f}", f"{T[0,1]:+.3f}", f"{T[0,2]:+.3f}", f"{T[0,3]:+.3f}",
            f"{T[1,0]:+.3f}", f"{T[1,1]:+.3f}", f"{T[1,2]:+.3f}", f"{T[1,3]:+.3f}",
            f"{T[2,0]:+.3f}", f"{T[2,1]:+.3f}", f"{T[2,2]:+.3f}", f"{T[2,3]:+.3f}",
            "0", "0", "0", "1",
        )
        for idx, txt in enumerate(labels):
            r, c = divmod(idx, 4)           
            self.table.item(r, c).setText(txt)
        if silent:
            self.table.blockSignals(False)    

    # ------------------------------------------------------------------ 预设
    def _apply_preset(self, kind: str):
        """应用四类典型变换预设（设置滑块位置）。"""
        if kind == "translate":
            tvals = (0.8, 0.5, 0.6)
            rvals = (0.0, 0.0, 0.0)
        elif kind == "rotate":         
            tvals = (0.0, 0.0, 0.0)
            rvals = (0.0, 60.0, 0.0)
        elif kind == "composite":      
            tvals = (0.7, -0.4, 0.5)
            rvals = (20.0, 45.0, -30.0)
        else:
            return
        self._set_sliders(tvals, rvals)

    def _set_sliders(self, tvals, rvals):
        """预设：同时设置滑块与数值框（阻断信号防递归），再统一刷新。"""
        for s, sp, v in zip(self.t_sliders, self.t_spins, tvals):
            with QSignalBlocker(s), QSignalBlocker(sp):
                s.setValue(self._trans_to_slider(v))
                sp.setValue(v)
        for s, sp, v in zip(self.r_sliders, self.r_spins, rvals):
            with QSignalBlocker(s), QSignalBlocker(sp):
                s.setValue(self._rot_to_slider(v))
                sp.setValue(v)
        self._refresh_from_params()

    def _reset_identity(self):
        """重置为单位阵：清空级联 + 所有滑块归零。"""
        self._clear_cascade()
        mid = SLIDER_STEPS // 2
        for s, sp in zip(self.t_sliders, self.t_spins):
            with QSignalBlocker(s), QSignalBlocker(sp):
                s.setValue(mid)
                sp.setValue(0.0)
        for s, sp in zip(self.r_sliders, self.r_spins):
            with QSignalBlocker(s), QSignalBlocker(sp):
                s.setValue(mid)
                sp.setValue(0.0)
        self._refresh_from_params()

    # ------------------------------------------------------------------ 级联
    def _add_cascade_step(self):
        """把当前 T 存入级联历史，后续渲染时会把所有历史步骤连乘起来。"""
        self._cascade.append(self._T.copy())  
        self.lbl_cascade.setText(f"级联步骤数：{len(self._cascade)}")
        self._render()

    def _clear_cascade(self):
        """清空级联历史。"""
        self._cascade.clear()
        self.lbl_cascade.setText("级联步骤数：0")
        self._render()

    def _effective_T(self) -> np.ndarray:
        """连乘级联：T_total = T_1 @ T_2 @ ... @ T_n @ T_current。"""
        T = np.eye(4)               
        for step in self._cascade:
            T = T @ step
        return T @ self._T                 

    # ------------------------------------------------------------------ 渲染
    def _render(self):
        """重绘 3D 视图：清空 -> 画固定系 {A} -> 画被驱动的系 {B} + 球 + 级联路径。"""
        ax = self.ax
        ax.cla()                      
        self._setup_axes()                    
        self._draw_frame(ax, np.eye(4), dashed=True, prefix="{A}")
        Tb = self._effective_T()
        self._draw_frame(ax, Tb, dashed=False, prefix="{B}")
        self._draw_ball(ax, Tb)

        if len(self._cascade) >= 1:
            pts = [np.zeros(3)]              
            T_acc = np.eye(4)                
            for step in self._cascade:
                T_acc = T_acc @ step          
                pts.append(T_acc[:3, 3])     
            pts.append(Tb[:3, 3])             
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="gray",
                    linestyle=":", linewidth=1.2, alpha=0.8)

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
    win = PoseDemoWindow()
    win.show()                                # 显示窗口
    sys.exit(app.exec())                      # 进入 Qt 事件循环；退出时用状态码关闭进程


if __name__ == "__main__":
    main()
