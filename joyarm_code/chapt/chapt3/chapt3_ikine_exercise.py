"""
第三章实践：使用 Pinocchio 实现 JoyArm 数值逆运动学求解器

功能概要：
    继承 ``IkineSolver``，使用 Pinocchio 的正运动学和雅可比接口，在
    ``solve()`` 中实现单起点阻尼最小二乘逆运动学。实践中只调用 Pinocchio
    获取雅可比，不需要推导或手工构造雅可比矩阵。

实践任务：
    阅读并独立完成 ``PracticePinIkineSolver.solve()`` 中标出的
    “实践实现区”。目标位姿、连续初值、回代校验和显示由辅助框架提供。

环境与运行：
    cd ~/joyin_arm_tutorial/joyarm_code
    uv sync
    source .venv/bin/activate
    python chapt/chapt3/chapt3_ikine_exercise.py

    只查看终端结果、不启动 MeshCat：
    python chapt/chapt3/chapt3_ikine_exercise.py --headless
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

# 允许从 joyarm_code/ 目录直接运行本文件。
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from joyarm_core import IKResult, IkineSolver

try:
    from ._ikine_exercise_support import run_exercise
except ImportError:
    from _ikine_exercise_support import run_exercise


def _frame_id(model, frame) -> int:
    """把目标帧名转换为 Pinocchio 帧编号，并检查帧是否存在。"""
    frame_id = model.getFrameId(str(frame))
    if not 0 <= frame_id < model.nframes:
        raise ValueError(f"目标帧不存在：{frame!r}")
    return frame_id


def _pose_error(target_T, current) -> np.ndarray:
    """返回基座参考系下的六维位姿误差。"""
    position_error = target_T[:3, 3] - current.translation
    rotation_error = pin.log3(target_T[:3, :3] @ current.rotation.T)
    return np.hstack((position_error, rotation_error))


class PracticePinIkineSolver(IkineSolver):
    """等待读者完成的固定阻尼数值逆运动学实践求解器。"""

    def __init__(self, damping: float = 0.03, step_max: float = 0.15):
        if damping <= 0.0:
            raise ValueError("damping 必须大于 0")
        if step_max <= 0.0:
            raise ValueError("step_max 必须大于 0")
        self.damping = float(damping)
        self.step_max = float(step_max)

    def solve(
        self,
        arm,
        pose,
        frame: str,
        q0: np.ndarray,
        *,
        tol: float = 1e-3,
        iters: int = 200,
        **kwargs,
    ) -> IKResult:
        """从 ``q0`` 出发，使用 Pinocchio 迭代求一组 JoyArm 逆解。"""
        # ======================== 实践实现区开始 ========================
        # TODO：在这里完成固定阻尼数值逆运动学。
        #
        # ｜准备数据
        #   - 模型：arm.pin_model
        #   - 私有计算数据：pin.Data(model)
        #   - 目标帧编号：_frame_id(model, frame)
        #   - 目标齐次矩阵：pose.T
        #   - 初始关节角：把 q0 整理为一维浮点数组，并保留独立副本





        # ｜每轮迭代
        #   1. 使用 forwardKinematics() 和 updateFramePlacement() 计算当前位姿；
        #   2. 使用 _pose_error() 得到六维误差，前三维为位置、后三维为姿态；
        #   3. 用两部分误差范数之和作为残差，并与 tol 比较；
        #   4. 使用 computeFrameJacobian(..., pin.LOCAL_WORLD_ALIGNED) 计算雅可比；
        #   5. 按下式计算关节修正量：
        #      Δq = J.T @ solve(J @ J.T + damping² I, error)
        #   6. 将 ||Δq|| 限制在 self.step_max 内，再用 pin.integrate() 更新 q；
        #   7. 使用 arm.joint_limits_hard_arm 把 q 裁剪到关节硬限位内。





        # ｜返回结果
        #   - 收敛时返回 IKResult(q, True, residual, iteration + 1)；
        #   - 始终记录残差最小的关节角；达到 iters 后返回该结果；
        #   - 未达到 tol 时 success=False，n_iter=int(iters)。





        raise NotImplementedError(
            "请完成 PracticePinIkineSolver.solve() 中的实践实现区"
        )
        # ========================= 实践实现区结束 =========================


if __name__ == "__main__":
    run_exercise(PracticePinIkineSolver())
