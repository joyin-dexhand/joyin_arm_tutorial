"""JoyArm 监控页原生窗口（PySide6 QtWebEngine 子进程入口）。

由 `JoyArm.start_state_viz()` 以 `python -m joyarm_core.joyarm.viz_window <url>`
拉起：独立桌面窗口内嵌 Chromium 引擎加载监控页（左静态/中 3D/右动态三栏），
数据仍来自 JoyArm 的 HTTP 服务，本进程零数据逻辑；关闭窗口即退出本进程。
"""

import sys

from PySide6.QtCore import QUrl, Qt
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView


def main() -> int:
    """入口：按命令行参数打开监控窗口并进入事件循环。

    :param url: 监控页 URL（`http://127.0.0.1:<port>/`）。
    :param title: 窗口标题（缺省「JoyArm 运行状态监控」）。
    :return: 进程退出码（参数缺失返回 2，正常关闭返回 0）。
    """
    if len(sys.argv) < 2:
        print("用法：python -m joyarm_core.joyarm.viz_window <url> [title]")
        return 2
    url, title = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "JoyArm 运行状态监控")
    # QtWebEngine 官方建议：QApplication 创建前开启 OpenGL 上下文共享（WebGL 稳定）。
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv[:1])
    win = QMainWindow()
    win.setWindowTitle(title)
    win.resize(1440, 820)
    view = QWebEngineView()
    view.load(QUrl(url))
    win.setCentralWidget(view)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
