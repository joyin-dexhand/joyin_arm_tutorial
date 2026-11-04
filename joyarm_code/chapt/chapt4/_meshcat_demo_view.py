"""第四章 GUI 的嵌入式 MeshCat 真实模型视图。"""
from __future__ import annotations

import meshcat.geometry as mg
import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QVBoxLayout, QWidget
from PySide6.QtWebEngineWidgets import QWebEngineView

from chapt.common.model_visualization import MODEL_NAME, create_viewer


class MeshcatDemoView(QWidget):
    """把 MeshCat WebGL 页面嵌入 PySide6，并管理第四章叠加图形。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.visualizer = create_viewer(MODEL_NAME, open_browser=False)
        self.scene = self.visualizer.viewer
        self._style_scene()
        self.web_view = QWebEngineView(self)
        self.web_view.setContextMenuPolicy(Qt.NoContextMenu)
        self.web_view.loadFinished.connect(lambda _ok: self._style_scene())
        self.web_view.load(QUrl(self.scene.url()))
        layout.addWidget(self.web_view)

    def _style_scene(self):
        """使用浅灰白背景与低对比度网格，保留空间参照。"""
        self.scene["/Background"].set_property("top_color", [0.95, 0.96, 0.975])
        self.scene["/Background"].set_property("bottom_color", [0.995, 0.995, 1.0])
        self.scene["/Grid"].set_property("visible", False)
        self.scene["/Axes"].set_property("visible", False)
        segments = []
        for value in np.linspace(-0.6, 0.6, 13):
            segments.extend(([-0.6, value, 0.0], [0.6, value, 0.0]))
            segments.extend(([value, -0.6, 0.0], [value, 0.6, 0.0]))
        points = np.asarray(segments, dtype=float).T
        environment = self.scene["chapter4_environment"]
        environment["grid"].set_object(
            mg.LineSegments(
                mg.PointsGeometry(points),
                mg.LineBasicMaterial(color=0xD8DEE8, linewidth=1.0),
            )
        )
        environment["origin"].set_object(mg.triad(0.10))

    @staticmethod
    def _line(scene, name, start, vector, color, width=3.0):
        points = np.column_stack((np.asarray(start), np.asarray(start) + np.asarray(vector)))
        scene[name].set_object(
            mg.Line(
                mg.PointsGeometry(points),
                mg.LineBasicMaterial(color=color, linewidth=width),
            )
        )

    def update_scene(self, q, tcp, mode, *, contributions=None, force=None):
        """刷新真实模型姿态及当前教学模式的速度/外力叠加图形。"""
        self.visualizer.display(np.asarray(q, dtype=float))
        overlay = self.scene["chapter4"]
        overlay.delete()
        overlay["tcp"].set_object(
            mg.Sphere(0.012), mg.MeshLambertMaterial(color=0x111827)
        )
        tcp_transform = np.eye(4)
        tcp_transform[:3, 3] = tcp
        overlay["tcp"].set_transform(tcp_transform)
        if mode == "雅可比" and contributions is not None:
            for index, item in enumerate(contributions):
                self._line(overlay, f"velocity/joint{index + 1}", tcp, 2.5 * item[:3], 0x93C5FD, 2.0)
            self._line(overlay, "velocity/result", tcp, 2.5 * contributions.sum(axis=0)[:3], 0x111827, 5.0)
        elif mode == "静力学" and force is not None and np.linalg.norm(force) > 0:
            self._line(overlay, "force", tcp, 0.012 * np.asarray(force), 0xDC2626, 5.0)

    def shutdown(self):
        """回收 MeshCat 本地服务进程。"""
        try:
            process = self.scene.window.server_proc
            process.terminate()
            process.wait(timeout=2.0)
        except Exception:
            pass
