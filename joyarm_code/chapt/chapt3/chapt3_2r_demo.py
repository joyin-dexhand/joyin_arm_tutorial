"""
==============================================================================
平面 2R 逆运动学可视化  (第三章 §2)
==============================================================================
【功能概要】
    拖动目标点坐标与两段连杆长度，切换查看肘上、肘下两组逆解；
    目标移出可达圆环时显示无解，移到边界时观察两解重合。
    本脚本只演示平面 2R 几何，不连接 JoyArm，也不实现 6R 求解器。

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code
    uv sync
    # Windows PowerShell:  .venv/Scripts/Activate.ps1

    # 2. 运行本脚本
    uv run python chapt/chapt3/chapt3_2r_demo.py

【命令行参数】
    无。用窗口下方的滑块调节 x、y、l1、l2，左下角切换肘形。
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle
from matplotlib.widgets import RadioButtons, Slider

chinese_font = next((path for path in font_manager.findSystemFonts()
                     if "NotoSansCJK-Regular" in path), None)
if chinese_font:
    font_manager.fontManager.addfont(chinese_font)
    plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=chinese_font).get_name()]
plt.rcParams["axes.unicode_minus"] = False


def solve_2r(x: float, y: float, l1: float, l2: float):
    """按第三章式 (2)、(3) 求两组关节角；返回 (解列表, 状态)。"""
    r2 = x * x + y * y
    cosine = (r2 - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if cosine < -1 - 1e-10 or cosine > 1 + 1e-10:
        return [], "目标不可达：无实数逆解"

    if r2 < 1e-12 and abs(l1 - l2) < 1e-10:
        return [(0.0, np.pi)], "目标在原点：有无限多组折叠解，图中仅画一组"

    elbow_angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    if abs(np.sin(elbow_angle)) < 1e-10:
        angles = (elbow_angle,)
        status = "工作空间边界：肘上、肘下两解重合"
    else:
        angles = (elbow_angle, -elbow_angle)
        status = "目标可达：两组构型解"

    solutions = []
    for q2 in angles:
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        solutions.append((q1, q2))
    return solutions, status


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.32)
    ax_branch = fig.add_axes((0.02, 0.09, 0.12, 0.12))
    ax_x = fig.add_axes((0.18, 0.22, 0.68, 0.025))
    ax_y = fig.add_axes((0.18, 0.17, 0.68, 0.025))
    ax_l1 = fig.add_axes((0.18, 0.12, 0.68, 0.025))
    ax_l2 = fig.add_axes((0.18, 0.07, 0.68, 0.025))
    sx = Slider(ax_x, "x", -3.0, 3.0, valinit=1.15)
    sy = Slider(ax_y, "y", -3.0, 3.0, valinit=0.55)
    sl1 = Slider(ax_l1, "l1", 0.25, 1.5, valinit=1.0)
    sl2 = Slider(ax_l2, "l2", 0.25, 1.5, valinit=0.75)
    branch = RadioButtons(ax_branch, ("肘下", "肘上"))

    def redraw(_=None):
        x, y, l1, l2 = sx.val, sy.val, sl1.val, sl2.val
        solutions, status = solve_2r(x, y, l1, l2)
        ax.clear()
        ax.add_patch(Circle((0, 0), l1 + l2, fill=False, ls="--", ec="0.6"))
        ax.add_patch(Circle((0, 0), abs(l1 - l2), fill=False, ls=":", ec="0.6"))
        ax.plot(x, y, "k*", ms=12, label="目标点")
        ax.plot(0, 0, "ko", ms=5)

        if solutions:
            selected = 0 if len(solutions) == 1 or branch.value_selected == "肘下" else 1
            q1, q2 = solutions[selected]
            label = "边界/折叠解" if len(solutions) == 1 else branch.value_selected
            color = "tab:blue" if selected == 0 else "tab:orange"
            elbow = (l1 * np.cos(q1), l1 * np.sin(q1))
            ax.plot([0, elbow[0], x], [0, elbow[1], y], "o-", color=color,
                    lw=2.5, label=f"{label}: q1={np.rad2deg(q1):.1f}°, "
                                  f"q2={np.rad2deg(q2):.1f}°")

        if len(solutions) == 2:
            angle_text = "\n".join(
                f"{label}: q1={np.rad2deg(q1):.1f}°, q2={np.rad2deg(q2):.1f}°"
                for label, (q1, q2) in zip(("肘下", "肘上"), solutions)
            )
            ax.text(0.02, 0.02, angle_text, transform=ax.transAxes, va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"})

        ax.set_title(status)
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-3.1, 3.1)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.canvas.draw_idle()

    for slider in (sx, sy, sl1, sl2):
        slider.on_changed(redraw)
    branch.on_clicked(redraw)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
