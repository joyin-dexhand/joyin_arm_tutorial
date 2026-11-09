"""
==============================================================================
fkine 正运动学交互测试
==============================================================================
【功能概要】
    正运动学求解器的图形界面测试工具，不连硬件即可运行：
      - 左侧 3D 视图：机械臂随输入关节角摆位；
      - 右侧输入框、「计算」按钮和三张表：
          自检结论表——数学性质自检 / 输入边界鲁棒性 / 异常路径测试的结论；
          求解器结果表——末端位置、姿态四元数与耗时；
          批量对比表——多组关节角下的末端结果，汇总最大位置误差与最大姿态误差。

    启动即完成全部检查并按零位形填表；之后输入任意关节角点「计算」，表格与 3D 视图随之刷新。

【环境与运行】
    # 0. 安装 uv（仅首次）
    curl -LsSf https://astral.sh/uv/install.sh | sh            # Linux
    # irm https://astral.sh/uv/install.ps1 | iex               # Windows PowerShell

    # 1. 创建虚拟环境并同步依赖
    cd <仓库路径>/joyarm_code && uv sync

    # 2. 运行（离线，无需硬件）
    uv run python test/fkine_test.py <型号> 如：rebot_arm_b601_dm

【预期输出】
    控制台先打印求解器清单与启动检查结果，随后弹出窗口；输入关节角点「计算」，三张表与 3D 视图随之刷新。
==============================================================================
"""

import logging                                            # 收尾时临时静默离线脚本的既有告警
import sys                                               # 解析命令行参数与注入模块搜索路径
import time                                              # 求解耗时统计（perf_counter 计时）
from pathlib import Path                                 # URDF 路径的跨平台路径处理

import numpy as np                                       # 数值计算：向量/矩阵运算与有限性判定

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 未安装 joyarm_core 时也可直接运行
import joyarm_core.robot_model as robot_model            # 资产包：仅用其包路径定位 URDF（不 import 私有成员）
import meshcat                                           # meshcat 服务器（左栏 3D 场景，失败整体降级）
import pinocchio as pin                                  # pinocchio：构建 COLLISION/VISUAL 几何模型
from pinocchio.visualize import MeshcatVisualizer        # meshcat 可视化器（与 start_state_viz 同法）
from PySide6.QtCore import QEvent, Qt, QUrl              # Qt 事件类型/常量/URL（WebEngine 加载 meshcat 页）
from PySide6.QtWebEngineCore import QWebEnginePage       # 网页页抽象（覆写 JS 控制台回调做静音）
from PySide6.QtWebEngineWidgets import QWebEngineView    # 内嵌网页视图（显示 meshcat 页）
from PySide6.QtWidgets import (                          # Qt 部件：窗口/布局/输入框/表格
    QApplication, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from joyarm_core import Pose, joyArm, listArm            # JoyArm 创建入口 + 位姿数据类 + 型号清单
from joyarm_core.utils.limits import rand_within_limits  # 硬/软限位内随机采样（固定种子可复现）
from joyarm_core.utils.transforms import quat_conj, quat_mul  # 相对四元数（姿态角距稳健算法）

# ============================================================
# 全局常量（检查规模与阈值可微调；改动后须同步注释与表头口径）
# ============================================================
N_TIMING = 5                                             # 单求解器计时次数：剔除最大最小后取中间 3 次均值
N_EDGE_RAND = 10_000                                     # 检查 2：硬限位内随机采样组数（1e4，秒级）
N_RAND_Q = 5                                             # 检查 1：软限位内随机位形组数（另有 zero/home/当前 q）
SEED_RAND_Q = 2026                                       # 检查 1 随机种子（与既有 test 脚本同口径）
SEED_EDGE = 2027                                         # 检查 2 随机种子（与检查 1 区分）
MATH_TOL = 1e-9                                          # 检查 1 阈值：SE(3) 结构/往返一致最大允许偏差
CONSISTENCY_TOL = 1e-9                                   # 表 3 阈值：各求解器末端两两互差上限
N_BATCH_RAND = 8                                         # 表 3：软限位内随机位形组数（另有 zero/home 共 10 组）
SEED_BATCH = 2028                                        # 表 3 随机种子（与检查 1/2 的种子区分）
N_BATCH_SHOW = 10                                        # 表 3 截断上限：用例数超过 10 仅测试前 10（规则显式生效）
ROW_H = 26                                               # 表格固定行高（px）：表高 = 表头 + 行数 × 行高


# ============================================================
# 格式化与数值辅助（控制台和表格统一口径）
# ============================================================
def fit_table_height(table: QTableWidget) -> None:
    """表高精确贴合内容：固定行高，表高 = 表头 + 全部行 + 边框，关闭竖向滚动条。

    不用 QSizePolicy.Maximum：它依据 QTableWidget 的默认 sizeHint（与行数无关），
    行少时留白、行多时被布局压缩出滚动条并裁掉末行（与 chapt2_R_demo 同法规避）。
    """
    vh = table.verticalHeader()                          # 行表头（已隐藏，只用其行高设置）
    vh.setSectionResizeMode(QHeaderView.Fixed)           # 行高固定，不随布局伸缩
    vh.setDefaultSectionSize(ROW_H)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 全部行可见，无需内滚动
    table.setFixedHeight(table.horizontalHeader().sizeHint().height()
                         + table.rowCount() * ROW_H + table.frameWidth() * 2 + 2)


def centered_item(text: str) -> QTableWidgetItem:
    """新建居中对齐的只读单元格（三张表统一对齐口径）。"""
    item = QTableWidgetItem(text)                        # 单元格文本
    item.setTextAlignment(Qt.AlignCenter)                # 水平垂直居中
    return item


def fmt_vec(vec, nd: int = 3) -> str:
    """向量格式化为 [+0.123, -0.456, +0.789]（定宽正负号，表格统一格式）。"""
    return "[" + ", ".join(f"{float(v):+.{nd}f}" for v in vec) + "]"  # 逗号+空格分隔便于阅读


def fmt_ms(seconds: float) -> str:
    """求解耗时格式化（毫秒，3 位小数）。"""
    return f"{seconds * 1e3:.3f} ms"                     # 秒转毫秒


def brief(msg, width: int = 96) -> str:
    """异常/消息单行摘要：折叠换行与多余空白，超宽截断加省略号。"""
    text = " ".join(str(msg).split())                    # pinocchio C++ 消息为多行，压成单行
    return text if len(text) <= width else text[:width] + "……"  # 控制台一行放得下


def quat_angle(q1, q2) -> float:
    """两单位四元数的姿态角距（rad）：q_rel = q1* ⊗ q2，θ = 2·atan2(‖v_rel‖, |w_rel|)。

    不用 2·arccos(|q1·q2|)：arccos 在 1 附近病态，点积差 1e-16 即得约 1e-8 rad 假角距；
    atan2 形式对小角度保持机器精度。取 |w_rel| 消除 q 与 −q 双号歧义，姿态相同则为 0。
    """
    q_rel = quat_mul(quat_conj(q1), q2)                  # 相对旋转四元数（q1 → q2）
    return 2.0 * float(np.arctan2(np.linalg.norm(q_rel[1:]), abs(q_rel[0])))


def facade_solve(arm, name: str, q, frame: str):
    """门面求解助手：切激活求解器为 name 后经 arm.solve_fkine(q, frame) 求位姿。

    全部测试求解统一走此路径（不构造求解器直连实例）：门面对 q 先做维度校验
    再分发（joyarm.py solve_fkine），故 q 维度错误用例呈门面统一 ValueError；
    逐求解器批量循环处可先切一次激活、循环内直接调 arm.solve_fkine（同为门面）。
    """
    if not arm.set_fkine_solver(name):
        raise RuntimeError(f"fkine_test.py - facade_solve：激活 {name!r} 失败（不在 cfg 已加载成员中）")
    return arm.solve_fkine(q, frame)


# ============================================================
# 检查 1：数学性质自检（SE(3) 结构 + Pose↔T 往返一致）
# ============================================================
def math_selfcheck(arm, name: str, frame_end: str, qs) -> tuple:
    """逐位形求末端位姿，检查五项数学性质并返回最大偏差。

    ① max|RᵀR − I|（旋转正交性）；② |det R − 1|；③ T 底行偏离 [0,0,0,1]；
    ④ 四元数范数偏差；⑤ T 与 position/orientation 重装 T 的往返一致性。

    :param arm: 离线 JoyArm 对象。
    :param name: 求解器注册名（经门面切激活后逐位形求解）。
    :param frame_end: 末端帧名。
    :param qs: 位形组（zero + home + 随机 + 当前 q）。
    :return: (是否全过, 最大偏差)，阈值 MATH_TOL。
    """
    worst = 0.0                                          # 五项检查的滚动最大偏差
    bottom = np.array([0.0, 0.0, 0.0, 1.0])              # 齐次矩阵固定底行
    arm.set_fkine_solver(name)                           # 切激活一次，循环内经门面求解
    for q in qs:
        pose = arm.solve_fkine(q, frame_end)             # 门面求解末端位姿（含 q 校验）
        T = np.asarray(pose.T, dtype=float)              # 位姿的 4×4 齐次表示
        R = T[:3, :3]                                    # 旋转块
        dev = float(np.abs(R.T @ R - np.eye(3)).max())   # ① 正交性
        dev = max(dev, abs(float(np.linalg.det(R)) - 1.0))  # ② 行列式 = 1
        dev = max(dev, float(np.abs(T[3] - bottom).max()))  # ③ 底行固定
        dev = max(dev, abs(float(np.linalg.norm(pose.orientation)) - 1.0))  # ④ 四元数单位范数
        # ⑤ 往返一致：由 position/orientation 两字段重装 T，与原 T 逐元素对比
        again = Pose(position=np.asarray(pose.position, dtype=float),
                     orientation=np.asarray(pose.orientation, dtype=float)).T
        dev = max(dev, float(np.abs(np.asarray(again) - T).max()))
        worst = max(worst, dev)                          # 更新滚动最大偏差
    return worst <= MATH_TOL, worst


# ============================================================
# 检查 2：输入边界鲁棒性（限位边界 + 硬限位内大批量随机）
# ============================================================
def edge_robustness(arm, name: str, frame_end: str) -> tuple:
    """位形组 = 逐关节软上/下限（其余 0）+ 全上/全下 + N_EDGE_RAND 组硬限位内
    均匀随机（固定种子 SEED_EDGE 可复现）；逐组求解并检查结果全部有限。

    :param arm: 离线 JoyArm 对象。
    :param name: 求解器注册名（经门面切激活后逐组求解）。
    :param frame_end: 末端帧名。
    :return: (是否全部有限, 位形组数)。
    """
    soft = arm.joint_limits_soft_arm                     # 软限位（边界位形用）
    hard = arm.joint_limits_hard_arm                     # 硬限位（随机采样域）
    n = arm.n_arm                                        # 关节数
    qs = []                                              # 边界位形组
    for i in range(n):
        for bound in (soft.q_min[i], soft.q_max[i]):     # 第 i 关节分别取软下/上限
            q = np.zeros(n)                              # 其余关节取 0
            q[i] = bound
            qs.append(q)
    qs.append(np.asarray(soft.q_max, dtype=float))       # 全上限位形
    qs.append(np.asarray(soft.q_min, dtype=float))       # 全下限位形
    rng = np.random.default_rng(SEED_EDGE)               # 固定种子：边界检查可复现
    qs.extend(rand_within_limits(hard, size=N_EDGE_RAND, rng=rng))  # 硬限位内大批量随机
    arm.set_fkine_solver(name)                           # 切激活一次，循环内经门面求解
    for q in qs:
        pose = arm.solve_fkine(q, frame_end)
        if not (bool(np.all(np.isfinite(pose.position)))
                and bool(np.all(np.isfinite(pose.orientation)))):
            return False, len(qs)                        # 出现 NaN/inf 即未通过
    return True, len(qs)


# ============================================================
# 检查 3：异常路径（实测呈现类型与消息，不写死实现、不臆造）
# ============================================================
def exception_paths(arm, names: list, frame_end: str, frame_mid: str, verbose: bool = True) -> dict:
    """逐求解器经门面触发三类异常用例并记录实测行为（不写死实现，接什么呈现什么）。

    实测口径（离线、q 取零位形，异常与位形取值无关；逐求解器用例均经
    facade_solve 切激活后走 arm.solve_fkine，无直连路径）：
    - 未知帧名：应抛 ValueError（消息原文如实记录）；
    - 中间帧 frame_mid（按 pin 模型动态选取的真实帧名）：两种行为均记 ✓——
      正常返回 = 支持任意帧，抛 ValueError = 仅末端帧实现；抛其他异常才记 ✗；
    - q 维度错误（len = n−1 向量）：门面 arm.solve_fkine 的 q 校验先于求解器分发，
      逐求解器该用例均呈门面统一 ValueError，与所切求解器实现无关——如实呈现门面行为。

    :param arm: 离线 JoyArm 对象。
    :param names: 求解器注册名列表。
    :param frame_end: 末端帧名。
    :param frame_mid: 中间帧名（须为 pin 模型内真实存在的非末端帧）。
    :param verbose: True 时逐用例打印消息原文（启动一次）；False 静默（「计算」重算）。
    :return: {注册名: [(用例名, 是否符合, 短标记, 详情), ...]}。
    """
    q0 = np.zeros(arm.n_arm)                             # 零位形（仅作触发输入）
    q_bad = np.zeros(arm.n_arm - 1)                      # 维度错误向量（len = n−1）
    out = {}                                             # {注册名: 用例记录列表}
    for name in names:
        cases = []                                       # 本求解器的用例记录
        # 用例 1：未知帧名——应抛 ValueError（门面不校验帧名，透传求解器）
        try:
            facade_solve(arm, name, q0, "no_such_frame")
            cases.append(("未知帧名 no_such_frame", False, "✗", "未抛异常（非预期）"))
        except ValueError as e:
            cases.append(("未知帧名 no_such_frame", True, "✓", f"ValueError：{e}"))
        # 用例 2：合法中间帧——正常返回 = 支持任意帧；抛 ValueError = 仅末端帧（两种契约均 ✓）
        if not frame_mid:                                # 模型仅基座+末端帧：无中间帧可测
            cases.append(("中间帧", True, "—", "模型仅基座与末端帧，跳过本用例"))
        else:
            label = f"中间帧 {frame_mid}"                # 用例名（含中间帧名）
            try:
                pose = facade_solve(arm, name, q0, frame_mid)
                cases.append((label, True, "✓",
                              f"正常返回（支持任意帧）：位置 {fmt_vec(pose.position)}"))
            except ValueError as e:
                cases.append((label, True, "✓", f"ValueError（仅末端帧实现）：{e}"))
            except Exception as e:                       # 其他异常类型 = 行为异常
                cases.append((label, False, "✗", f"{type(e).__name__}：{e}"))
        # 用例 3：q 维度错误——经门面由 joyarm.py solve_fkine 的 q 校验先行统一抛
        # ValueError（先校验再分发），逐求解器呈现同一门面消息，属门面口径的如实记录
        try:
            facade_solve(arm, name, q_bad, frame_end)
            cases.append(("q 维度错误（门面）", False, "✗", "未抛异常（非预期）"))
        except ValueError as e:
            cases.append(("q 维度错误（门面）", True, "✓", f"ValueError：{e}"))
        out[name] = cases
        if verbose:                                      # 启动打印消息原文备查；重复计算静默
            for label, _, _, detail in cases:
                print(f"      [{name}] {label} → {detail}")
    # 门面必测：q 校验先于求解器路由、与激活项无关
    try:
        arm.solve_fkine(q_bad)
        facade = ("q 维度错误（arm.solve_fkine 门面）", False, "✗", "未抛异常（非预期）")
    except ValueError as e:
        facade = ("q 维度错误（arm.solve_fkine 门面）", True, "✓", f"ValueError：{e}")
    out["门面"] = [facade]
    if verbose:
        print(f"      [门面] {facade[0]} → {facade[3]}")
    return out


def solver_consistency(poses: dict) -> tuple:
    """同一 q 下各求解器末端位姿两两互差的最大值（表 3 互差列数据源）。

    :param poses: {注册名: Pose}——同一 q 的求解结果。
    :return: (位置最大互差 m, 姿态四元数角距最大值 rad)。
    """
    names = sorted(poses)                                # 固定遍历顺序
    d_pos = d_ang = 0.0                                  # 位置互差 / 角距滚动最大值
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = poses[a], poses[b]                  # 一对求解器的同 q 结果
            d_pos = max(d_pos, float(np.linalg.norm(pa.position - pb.position)))
            d_ang = max(d_ang, quat_angle(pa.orientation, pb.orientation))
    return d_pos, d_ang


# ============================================================
# 左栏：meshcat 3D 视图（构建失败整体降级为文字提示，绝不崩溃）
# ============================================================
class _QuietPage(QWebEnginePage):
    """静音网页页：吞掉 meshcat 页面的 JS 控制台消息。

    Qt 默认把网页 console 输出透传到终端（meshcat 加载 STL 会刷数十行
    「_meshfile is deprecated」弃用告警）；meshcat 前端日志对本工具无行动
    价值，页面整体异常已由 MeshcatPanel 的降级路径兜底，故全部静默。
    """

    def javaScriptConsoleMessage(self, level, message, line, source) -> None:
        """覆写为空：不转发任何 JS 控制台消息（QtWebEngine 约定回调）。"""
        pass


class MeshcatPanel(QWidget):
    """URDF 真实外观的 3D 视图面板（QWebEngineView 内嵌 meshcat 页面）。

    场景构建与 joyarm.py 的 start_state_viz 同法：URDF 按
    robot_model/<型号>/urdf/<型号>.urdf 推导（由 robot_model 包定位）→
    pin.buildGeomFromUrdf（COLLISION/VISUAL 各一次，package_dirs 指向 URDF
    所在目录）→ meshcat.Visualizer → MeshcatVisualizer.initViewer +
    loadViewerModel；姿态刷新经 viz.display(q)。任一步失败（含离屏渲染受限）
    → 面板降级为提示 QLabel，不影响右侧测试功能。
    """

    def __init__(self, arm, model: str, parent=None):
        """构建 meshcat 场景并嵌入网页视图；失败降级。

        :param arm: 离线 JoyArm 对象（提供 pin_model）。
        :param model: 型号名（= robot_model/<型号>/ 目录名）。
        """
        super().__init__(parent)
        self._viz = None                                 # MeshcatVisualizer（构建成功时持有）
        self._server_proc = None                         # meshcat-server 子进程（收尾终止用）
        self.view = None                                 # QWebEngineView（构建成功时持有）
        layout = QVBoxLayout(self)                       # 布局：成功放视图，失败放提示
        layout.setContentsMargins(0, 0, 0, 0)
        try:
            urdf_dir = Path(next(iter(robot_model.__path__))) / model / "urdf"  # URDF 所在目录
            urdf = urdf_dir / f"{model}.urdf"            # robot_model/<型号>/urdf/<型号>.urdf
            geom_kw = dict(package_dirs=[str(urdf_dir)])  # URDF 内 mesh 为相对路径，须给包目录
            collision = pin.buildGeomFromUrdf(arm.pin_model, str(urdf),  # 碰撞几何
                                              pin.GeometryType.COLLISION, **geom_kw)
            visual = pin.buildGeomFromUrdf(arm.pin_model, str(urdf),  # 外观几何
                                           pin.GeometryType.VISUAL, **geom_kw)
            viewer = meshcat.Visualizer()                # 拉起 meshcat-server 子进程
            self._server_proc = viewer.window.server_proc  # 立即登记：后续步骤失败也能回收
            viz = MeshcatVisualizer(arm.pin_model, collision, visual)  # 缩减模型 + 两类几何
            viz.initViewer(viewer=viewer)
            viz.loadViewerModel()
            viz.display(np.zeros(arm.n_arm))             # 初始零位形
            self.view = QWebEngineView(self)             # 共享 OpenGL 上下文（入口已设）
            self.view.setPage(_QuietPage(self.view))     # 静音页：拦截 meshcat 的 JS 控制台刷屏
            self.view.load(QUrl(viewer.url()))           # 加载 meshcat 页面
            layout.addWidget(self.view)
            self._viz = viz                              # 全部成功才登记，供 display 使用
            print(f"   meshcat 3D 视图已就绪（{viewer.url()}）")
        except Exception as e:                           # 构建失败：整体降级，不崩溃
            tip = QLabel("3D 视图不可用（meshcat 场景构建失败），右侧测试功能不受影响。\n"
                         f"原因：{type(e).__name__}: {e}")  # 降级提示标签
            tip.setWordWrap(True)
            tip.setAlignment(Qt.AlignCenter)
            layout.addWidget(tip)
            print(f"   meshcat 3D 视图构建失败，降级为文字提示：{brief(e)}")

    def display(self, q) -> None:
        """刷新机械臂姿态到关节角 q（失败即停用并提示，避免反复报错）。"""
        if self._viz is None:
            return
        try:
            self._viz.display(np.asarray(q, dtype=float))
        except Exception as e:
            self._viz = None                             # 置空：后续调用直接跳过
            print(f"   meshcat 姿态刷新失败，3D 视图停用：{brief(e)}")

    def shutdown(self) -> None:
        """收尾：预约删除网页视图，并终止 meshcat-server 子进程。

        视图删除经 deleteLater 预约，由 dispose_window 统一冲刷执行（须先于
        QApplication 析构，否则 WebEnginePage 晚于 profile 释放会告警甚至段错误）。
        """
        if self.view is not None:
            self.view.deleteLater()                      # 预约删除（dispose_window 冲刷执行）
            self.view = None
        if self._server_proc is not None:
            try:
                self._server_proc.terminate()
                self._server_proc.wait(timeout=2.0)
            except Exception:
                pass                                     # 收尾尽力而为，异常不再上抛
            self._server_proc = None                     # 幂等：重复调用不再终止


# ============================================================
# 主窗口：左 3D + 右（输入行 / 表 1 自检结论 / 表 2 求解器结果）
# ============================================================
class FkineTestWindow(QMainWindow):
    """fkine 交互测试主窗口：启动检查填两表，「计算」按输入 q 全量刷新。"""

    CHECK_HEADERS = ("求解器注册名", "数学性质自检", "输入边界鲁棒性", "异常路径测试")
    RESULT_HEADERS = ("求解器注册名", "末端位置 (m)", "末端姿态四元数 (w,x,y,z)", "门面耗时")

    def __init__(self, arm, model: str):
        """构建窗口并完成启动检查（秒级同步执行，规模见模块常量注释）。

        :param arm: 离线 JoyArm 对象。
        :param model: 型号名。
        """
        super().__init__()
        self._arm = arm                                  # 离线 JoyArm 对象（全部求解经其门面）
        self._model = model                              # 型号名（窗口标题与 URDF 定位）
        self._frame_end = str(arm.get_config()["joyarm"]["frame_end"])  # cfg 末端帧名
        # 中间帧按 pin 模型动态选取（去基座 universe 与末端帧后取中位）：
        # 帧名写死会在换型号后失效——不存在的帧被「仅末端帧」误判（实测踩过）
        mids = [f.name for f in arm.pin_model.frames[1:] if f.name != self._frame_end]
        self._frame_mid = mids[len(mids) // 2] if mids else ""  # 模型仅基座+末端时留空（跳过该用例）
        self._names = arm.get_fkine_solvers()            # fkine 域已加载求解器（与 cfg 一致，已排序）
        self._q_cur = np.zeros(arm.n_arm)                # 当前生效关节角（初始 zero）
        self._math_pass = {}                             # 检查 1 结论 {注册名: bool}
        self._math_worst = {}                            # 检查 1 最大偏差 {注册名: float}
        self._edge = {}                                  # 检查 2 结论 {注册名: (bool, 组数)}
        self._exc = {}                                   # 检查 3 用例 {注册名|"门面": [...]}
        self._timing = {}                                # 门面耗时（去极值均值，秒）{注册名: float}
        self._batch_rows = []                            # 表 3 数据行 [(位形名, q, {注册名: Pose}, 位置差, 角距)]
        self._batch_max = (0.0, 0.0)                     # 表 3 批量最大值 (位置差 m, 角距 rad)
        self.setWindowTitle(f"fkine 正运动学交互测试（{model}）")
        self.resize(1500, 920)
        print("=" * 78)
        print(f"fkine 正运动学交互测试：型号 {model}（离线，无需硬件）")
        print(f"   n_arm={arm.n_arm}，末端帧 frame_end={self._frame_end!r}，"
              f"pin 模型 {len(arm.pin_model.frames)} 帧")
        print(f"   fkine 域求解器（arm.get_fkine_solvers()，与 cfg 加载一致）：{self._names}")
        self._build_ui()
        self._recompute_all(verbose=True)                # 按零位形算好三张表（详情打印一次）

    # ---------------- UI 构建 ----------------
    def _build_ui(self) -> None:
        """搭建界面：左右分栏——左 meshcat 3D，右输入行 + 表 1 + 表 2 + 表 3。"""
        central = QWidget()                              # 中央部件（承载左右分栏）
        self.setCentralWidget(central)
        h = QHBoxLayout(central)                         # 左右分栏布局
        self.meshcat_panel = MeshcatPanel(self._arm, self._model, self)  # 左栏（失败自降级）
        h.addWidget(self.meshcat_panel, 3)
        right = QWidget()                                # 右栏容器（便于按比例分配宽度）
        rv = QVBoxLayout(right)                          # 右栏自上而下：输入 / 表 1 / 表 2 / 表 3
        rv.addWidget(self._build_input_group())
        rv.addWidget(self._build_check_table())
        rv.addWidget(self._build_result_table())
        rv.addWidget(self._build_batch_table())
        rv.addStretch(1)                                 # 各表高度贴合内容，余量留在底部
        scroll = QScrollArea()                           # 求解器增多致总高超窗时整栏滚动（不裁单表）
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(right)
        h.addWidget(scroll, 5)

    def _build_input_group(self) -> QGroupBox:
        """输入区：说明标签 + 单行输入框（初始全 0）+「计算」按钮 + 提示标签。"""
        box = QGroupBox("关节角输入")                    # 输入分组框
        v = QVBoxLayout(box)                             # 分组内纵向布局
        row = QHBoxLayout()                              # 输入行：标签 + 输入框 + 按钮
        row.addWidget(QLabel(f"q（{self._arm.n_arm} 个）:"))
        self.q_edit = QLineEdit(", ".join(["0"] * self._arm.n_arm))  # 初始内容 = 全 0 关节角
        self.q_edit.setToolTip("单位弧度，逗号分隔；非法输入红字提示且不刷新")  # 输入格式说明
        self.calc_btn = QPushButton("计算")              # 「计算」按钮（点击触发 _on_compute）
        self.calc_btn.clicked.connect(self._on_compute)
        row.addWidget(self.q_edit, 1)                    # 输入框吃满余宽
        row.addWidget(self.calc_btn)
        v.addLayout(row)
        self.hint = QLabel("启动检查完成，当前为 zero 位形结果；修改 q 后点击「计算」刷新。")  # 提示（非法红字）
        self.hint.setWordWrap(True)
        v.addWidget(self.hint)
        return box

    def _build_check_table(self) -> QGroupBox:
        """表 1「自检结论表」：行 = 求解器，列 = 数学性质/边界鲁棒性/异常路径检查结论。"""
        box = QGroupBox("自检结论表")                    # 表 1 分组框
        v = QVBoxLayout(box)                             # 分组内纵向布局
        n_col = len(self.CHECK_HEADERS)                  # 列数（4）
        self.check_table = QTableWidget(len(self._names), n_col)  # 行 = 求解器数
        self.check_table.setHorizontalHeaderLabels(self.CHECK_HEADERS)
        self.check_table.verticalHeader().setVisible(False)
        self.check_table.setEditTriggers(QTableWidget.NoEditTriggers)  # 只读结果表
        header = self.check_table.horizontalHeader()     # 表头（逐列设伸缩模式）
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 注册名列贴合内容
        for c in range(1, n_col):
            header.setSectionResizeMode(c, QHeaderView.Stretch)  # 结论列等宽铺满
        for r, name in enumerate(self._names):
            self.check_table.setItem(r, 0, centered_item(name))
            for c in range(1, n_col):
                self.check_table.setItem(r, c, centered_item("—"))  # 未计算前显「—」
        fit_table_height(self.check_table)               # 表高贴合行数（无留白、无内滚动）
        v.addWidget(self.check_table)
        return box

    def _build_result_table(self) -> QGroupBox:
        """表 2「求解器结果表」：每行一个求解器，只列当前 q 的结果（检查结论在表 1）。"""
        box = QGroupBox("求解器结果表")                  # 表 2 分组框
        v = QVBoxLayout(box)                             # 分组内纵向布局
        n_col = len(self.RESULT_HEADERS)                 # 列数（4）
        self.result_table = QTableWidget(len(self._names), n_col)  # 行数 = 注册数
        self.result_table.setHorizontalHeaderLabels(self.RESULT_HEADERS)
        self.result_table.horizontalHeaderItem(3).setToolTip(  # 耗时口径入悬浮提示
            f"同一关节角经门面调用 {N_TIMING} 次，剔除最大最小后取中间 3 次平均"
            f"（含门面参数校验开销）")
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)  # 只读结果表
        header = self.result_table.horizontalHeader()    # 表头（逐列设伸缩模式）
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 注册名列贴合内容
        for c in range(1, n_col):
            header.setSectionResizeMode(c, QHeaderView.Stretch)  # 其余列等宽铺满
        for r, name in enumerate(self._names):
            self.result_table.setItem(r, 0, centered_item(name))
            for c in range(1, n_col):
                self.result_table.setItem(r, c, centered_item("—"))  # 未计算前显「—」
        fit_table_height(self.result_table)              # 表高贴合行数（不再吞占剩余高度）
        v.addWidget(self.result_table)
        return box

    def _build_batch_table(self) -> QGroupBox:
        """表 3「求解器批量对比表」：列随已加载求解器数自适应，数据行 + 汇总 1 行。"""
        headers = ("位形", *(f"{name} 末端位置 (m)" for name in self._names),
                   "两两最大位置差 (m)", "两两最大姿态角距 (rad)")  # 列 = 位形 + 逐求解器 + 互差
        box = QGroupBox("求解器批量对比表")              # 表 3 分组框
        v = QVBoxLayout(box)                             # 分组内纵向布局
        n_col = len(headers)                             # 列数（2 + 求解器数）
        self.batch_table = QTableWidget(N_BATCH_SHOW + 1, n_col)  # 数据 10 行 + 汇总 1 行
        self.batch_table.setHorizontalHeaderLabels(headers)
        self.batch_table.verticalHeader().setVisible(False)
        self.batch_table.setEditTriggers(QTableWidget.NoEditTriggers)  # 只读结果表
        header = self.batch_table.horizontalHeader()     # 表头（逐列设伸缩模式）
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 位形列贴合内容
        for c in range(1, n_col):
            header.setSectionResizeMode(c, QHeaderView.Stretch)  # 其余列等宽铺满
        for r in range(N_BATCH_SHOW + 1):
            for c in range(n_col):
                self.batch_table.setItem(r, c, centered_item("—"))  # 未计算前显「—」
        for c in range(n_col):                           # 表尾汇总行加粗，与数据行区分
            cell = self.batch_table.item(N_BATCH_SHOW, c)
            font = cell.font()
            font.setBold(True)
            cell.setFont(font)
        fit_table_height(self.batch_table)               # 11 行全部可见（不再被压出滚动条）
        v.addWidget(self.batch_table)
        return box

    # ---------------- 全量重算（启动与「计算」同一路径） ----------------
    def _recompute_all(self, verbose: bool = True) -> None:
        """按 self._q_cur 重算全部数据并填三张表（启动与「计算」共用同一路径）。

        边界鲁棒性/异常路径/批量对比与 q 无关，仍一并重算——保证任何时刻三张表
        与最新状态一致、无过期单元格；批量规模控制在秒级（见模块常量注释）。
        verbose=True 打印逐用例异常详情（启动一次），False 只打汇总行（「计算」）。
        """
        arm, fe = self._arm, self._frame_end             # 局部别名（行宽友好）
        t0 = time.perf_counter()                         # 计时起点
        if verbose:
            print("-" * 78)
            print(f"   启动检查：② 输入边界鲁棒性（{N_EDGE_RAND} 组硬限位内随机，种子 {SEED_EDGE}）"
                  " / ③ 异常路径 / 表 3 批量对比 ……")
        for name in self._names:                         # 检查 2：边界鲁棒性
            self._edge[name] = edge_robustness(arm, name, fe)
        if verbose:
            print("   ③ 异常路径（逐求解器实测类型与消息原文）：")
        self._exc = exception_paths(arm, self._names, fe, self._frame_mid, verbose=verbose)  # 检查 3
        self._build_batch_data()                         # 表 3：各求解器批量对比
        qs = self._math_q_cases()                        # 检查 1 位形组（含当前 q）
        poses = {}                                       # {注册名: 当前 q 末端位姿}
        for name in self._names:
            poses[name] = facade_solve(arm, name, self._q_cur, fe)  # 当前 q 位姿（兼作计时预热）
            samples = []                                 # 逐次耗时（秒）
            for _ in range(N_TIMING):                    # 同一 q 经门面求解 N_TIMING 次，逐次计时
                t1 = time.perf_counter()                 # 单次计时起点
                arm.solve_fkine(self._q_cur, fe)         # 含门面参数校验开销（耗时列口径）
                samples.append(time.perf_counter() - t1) # 单次耗时入表
            mid = sorted(samples)[1:-1]                  # 剔除最大与最小，余 N_TIMING - 2 次
            self._timing[name] = sum(mid) / len(mid)     # 中间段均值（5 次 → 3 次均值）
            self._math_pass[name], self._math_worst[name] = math_selfcheck(arm, name, fe, qs)
        self._fill_batch_rows()                          # 表 3
        self._fill_result_rows(poses)                    # 表 2
        self._fill_check_rows()                          # 表 1
        print(f"   批量对比（{len(self._batch_rows)} 组位形 × {len(self._names)} 个求解器）："
              f"两两最大位置差 {self._batch_max[0]:.1e} m、最大角距 {self._batch_max[1]:.1e} rad"
              f"（阈 {CONSISTENCY_TOL:.0e}）")
        print(f"   {'启动检查' if verbose else '重算'}完成，用时 {time.perf_counter() - t0:.2f} s")

    def _build_batch_data(self) -> None:
        """构建表 3 数据：zero + home + N_BATCH_RAND 组软限位内随机（种子固定）。

        截断规则显式实现：用例总数超过 N_BATCH_SHOW 时仅测试前 N_BATCH_SHOW 组
        （当前 2 + 8 = 10 恰好等于上限，规则仍生效兜底；位形组数变动时自动截断）。
        逐组各求解器均经门面求解（facade_solve 切激活 + arm.solve_fkine），两两互差
        复用 solver_consistency；仅加载一个求解器时无互差可算，相应列显「—」。
        """
        arm, fe = self._arm, self._frame_end             # 局部别名
        rng = np.random.default_rng(SEED_BATCH)          # 固定种子：批量对比可复现
        cases = [("zero", np.asarray(arm.zero_arm, dtype=float)),
                 ("home", np.asarray(arm.home_arm, dtype=float))]
        cases += [(f"rand-{i + 1}", q) for i, q in enumerate(
            rand_within_limits(arm.joint_limits_soft_arm, size=N_BATCH_RAND, rng=rng))]
        cases = cases[:N_BATCH_SHOW]                     # 截断规则：超过 10 组仅测前 10
        rows, dp_max, da_max = [], 0.0, 0.0              # 数据行 / 滚动最大值
        for label, q in cases:
            poses = {name: facade_solve(arm, name, q, fe) for name in self._names}  # 逐求解器求解
            dp, da = (solver_consistency(poses) if len(self._names) > 1 else (None, None))
            if dp is not None:                           # 仅一个求解器时无互差
                dp_max, da_max = max(dp_max, dp), max(da_max, da)
            rows.append((label, q, poses, dp, da))
        self._batch_rows = rows                          # [(位形名, q, {注册名: Pose}, 位置差, 角距)]
        self._batch_max = (dp_max, da_max)               # 表尾汇总与表 1 一致性说明引用

    def _math_q_cases(self) -> list:
        """检查 1 位形组：zero + home + N_RAND_Q 组软限位内随机（固定种子）+ 当前 q。"""
        arm = self._arm                                  # 局部别名
        rng = np.random.default_rng(SEED_RAND_Q)         # 固定种子：位形组可复现
        qs = [np.asarray(arm.zero_arm, dtype=float), np.asarray(arm.home_arm, dtype=float)]  # zero + home
        qs.extend(rand_within_limits(arm.joint_limits_soft_arm, size=N_RAND_Q, rng=rng))
        qs.append(np.array(self._q_cur, dtype=float))    # 当前 q（「计算」后随之更新）
        return qs

    def _fill_result_rows(self, poses: dict) -> None:
        """填表 2：逐求解器当前 q 的末端位置、姿态四元数与耗时（检查结论在表 1）。"""
        for r, name in enumerate(self._names):
            pose = poses[name]                           # 本行求解器的当前 q 结果
            self._set_cell(self.result_table, r, 1, fmt_vec(pose.position))
            self._set_cell(self.result_table, r, 2, fmt_vec(pose.orientation, nd=4))
            self._set_cell(self.result_table, r, 3, fmt_ms(self._timing[name]))

    def _fill_batch_rows(self) -> None:
        """填表 3：逐位形数据行（位形 + 逐求解器位置 + 两两互差）+ 表尾汇总行。"""
        c_diff = 1 + len(self._names)                    # 互差两列的起始列号（位形 + N 位置列之后）
        for r, (label, _q, poses, dp, da) in enumerate(self._batch_rows):
            self.batch_table.item(r, 0).setText(label)
            for c, name in enumerate(self._names, start=1):  # 逐求解器位置列（列序与表头同构）
                self.batch_table.item(r, c).setText(fmt_vec(poses[name].position))
            self.batch_table.item(r, c_diff).setText(f"{dp:.1e}" if dp is not None else "—")
            self.batch_table.item(r, c_diff + 1).setText(f"{da:.1e}" if da is not None else "—")
        r = len(self._batch_rows)                        # 汇总行号（数据行之后第一行）
        self.batch_table.item(r, 0).setText(f"最大值（{len(self._batch_rows)} 组）")
        for c in range(1, c_diff):                       # 各求解器位置列不适用，显「—」
            self.batch_table.item(r, c).setText("—")
        has_diff = len(self._names) > 1                  # 仅一个求解器时无互差可汇总
        self.batch_table.item(r, c_diff).setText(f"{self._batch_max[0]:.1e} m" if has_diff else "—")
        self.batch_table.item(r, c_diff + 1).setText(f"{self._batch_max[1]:.1e} rad" if has_diff else "—")

    def _fill_check_rows(self) -> None:
        """填表 1：行 = 求解器，三列检查结论（数值细节进悬浮提示）。"""
        for r, name in enumerate(self._names):
            ok_m, worst = self._math_pass[name], self._math_worst[name]  # 检查 1 结论与最大偏差
            self._set_cell(self.check_table, r, 1, "通过" if ok_m else "未通过",
                           f"最大偏差 {worst:.1e}（阈值 {MATH_TOL:.0e}；位形组 zero/home/"
                           f"{N_RAND_Q} 组随机/当前 q）")
            ok_e, n_group = self._edge[name]             # 检查 2 结论与组数
            self._set_cell(self.check_table, r, 2, "通过" if ok_e else "未通过",
                           f"{n_group} 组位形（含 {N_EDGE_RAND} 组硬限位内随机）结果全部有限")
            cases = self._exc[name]                      # 检查 3 用例记录
            ok_x = all(case[1] for case in cases)        # 全部用例符合实测
            short = "；".join(case[2] for case in cases)  # 逐用例短标记（✓/✗）
            self._set_cell(self.check_table, r, 3, ("通过：" if ok_x else "未通过：") + short,
                           "\n".join(f"{case[0]} → {case[3]}" for case in cases))

    def _set_cell(self, table: QTableWidget, row: int, col: int, text: str, tip: str = "") -> None:
        """写指定表格单元格（居中）；tip 非空时作悬浮提示（完整数值/消息原文）。"""
        item = table.item(row, col)                      # 目标单元格
        item.setText(text)
        item.setToolTip(tip if tip else text)

    # ---------------- 「计算」按钮 ----------------
    def _on_compute(self) -> None:
        """「计算」：解析输入 q（非法 → 红字提示、不刷新）→ 按新 q 重算刷新全部表格。"""
        q, err = self._parse_q(self.q_edit.text())       # 解析输入（非法时 q=None）
        if q is None:
            self.hint.setStyleSheet("color: #dc2626;")   # 红字说明原因
            self.hint.setText(f"输入非法：{err}（表格与 3D 视图均不刷新）")
            return
        self._q_cur = q                                  # 更新当前生效 q
        self._recompute_all(verbose=False)               # 三张表按新 q 全部重算刷新
        self.meshcat_panel.display(q)                    # 3D 姿态同步
        self.hint.setStyleSheet("")                      # 恢复正常配色
        self.hint.setText(f"已按 q = {fmt_vec(q)} 重算并刷新全部表格")

    def _parse_q(self, text: str) -> tuple:
        """解析输入框文本 → (q, 错误说明)；合法要求个数 = n_arm 且全为有限数。"""
        parts = [t for t in text.replace("，", ",").replace(" ", "").split(",") if t]  # 去空格按逗号切分
        try:
            vals = [float(t) for t in parts]             # 逐段转 float（中文逗号/空格容错）
        except ValueError:
            return None, "存在无法识别为数字的内容"
        if len(vals) != self._arm.n_arm:
            return None, f"关节角个数须为 {self._arm.n_arm}，实际 {len(vals)}"
        q = np.asarray(vals, dtype=float)                # 个数合法后转数组
        if not bool(np.all(np.isfinite(q))):
            return None, "含非有限数值（NaN/inf）"
        return q, ""


# ============================================================
# QApplication 工厂与窗口回收
# ============================================================
def new_qapp() -> QApplication:
    """创建 QApplication；AA_ShareOpenGLContexts 须在构造之前设置（WebEngine 要求）。"""
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    return QApplication(sys.argv)


def dispose_window(win: FkineTestWindow) -> None:
    """销毁主窗口：回收 3D 视图与 meshcat-server → 删窗口 → 立即冲刷延迟删除。

    WebEnginePage 须先于 QApplication 析构释放，否则退出时报「Release of profile
    requested but WebEnginePage still not deleted」并可能段错误（本脚本实测）；
    不在事件循环内时 deleteLater 不会自动执行，故显式 sendPostedEvents(DeferredDelete)，
    重复数轮以覆盖视图析构级联投递的新删除事件。
    """
    win.meshcat_panel.shutdown()
    win.close()
    win.deleteLater()                                    # 整棵部件树随窗口一并删除
    for _ in range(3):                                   # 级联删除可能再投递，冲刷数轮
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QApplication.processEvents()


# ============================================================
# 命令行入口
# ============================================================
def main() -> None:
    """命令行入口：python test/fkine_test.py <型号>。

    缺型号参数 → 打印用法 + listArm() 清单并以退出码 1 结束。
    """
    args = [a for a in sys.argv[1:] if not a.startswith("--")]  # 位置参数（型号）
    if not args:
        print("用法：python test/fkine_test.py <型号>")
        print("示例：python test/fkine_test.py rebot_arm_b601_dm")
        print(f"可用型号：{listArm()}")
        sys.exit(1)
    model = args[0]                                      # 型号名（首个位置参数）
    arm = joyArm(model)                                  # 离线创建（无需硬件）
    app = QApplication.instance() or new_qapp()          # 复用或新建 QApplication
    win = FkineTestWindow(arm, model)                    # 主窗口（构造即跑启动检查）
    win.show()
    print("窗口已显示：左侧 3D 视图；右侧输入 q 并点击「计算」重算刷新全部表格。")
    app.exec()                                           # 交互事件循环（关窗返回）
    dispose_window(win)                                  # 视图/窗口先于 QApplication 销毁
    close_quiet(arm)                                     # 离线收尾（幂等；静默未连接类既有告警）


def close_quiet(arm) -> None:
    """离线收尾并静默既有的「未连接豁免跳过/操作跳过」类告警。

    本脚本全程离线，close() 必然触发 joyarm/backend 的未连接告警（每次退出
    固定数行、无信息量）；临时压低日志级别到 ERROR 屏蔽之，收尾后恢复，
    真正的 ERROR 及以上仍会输出。
    """
    logging.disable(logging.WARNING)                     # 屏蔽 WARNING 及以下（含 INFO）
    try:
        arm.close()                                      # 离线收尾（幂等）
    finally:
        logging.disable(logging.NOTSET)                  # 恢复日志级别（不影响后续）


if __name__ == "__main__":                               # 直接运行本文件时执行入口
    main()                                               # 解析参数后进 GUI 事件循环
