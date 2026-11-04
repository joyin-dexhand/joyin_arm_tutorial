"""第四章实践：实现雅可比模块，并计算末端负载对应的关节负载。

功能概要：
    任务 1：继承 JacobianSolver，在 jac() 中补全 Pinocchio 帧雅可比计算；
    任务 2：根据末端外载荷和雅可比矩阵，计算静力平衡所需的关节力矩。
    第四章公共框架负责固定轨迹、结果检查和 MeshCat 显示。

环境与运行：
    cd ~/joyin_arm_tutorial/joyarm_code
    uv sync
    source .venv/bin/activate
    python chapt/chapt4/chapt4_jacobian_exercise.py
    python chapt/chapt4/chapt4_jacobian_exercise.py --meshcat

    第三至第六章全部完成后，可在 chapt/common/motion_exercise.py 中进行综合实验。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 允许从 joyarm_code/ 目录直接运行本文件。
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from joyarm_core import JacobianSolver, Wrench  # noqa: E402
from chapt.chapt4._jacobian_exercise_support import run_jacobian_exercise  # noqa: E402


class PracticeJacobianSolver(JacobianSolver):
    """读者实现的 JoyArm 末端几何雅可比求解器。"""

    def jac(self, arm, q: np.ndarray, frame: str) -> np.ndarray:
        """返回形状为 (6, n) 的基座系雅可比；前三行为线速度。"""
        # ======================== 实践实现区开始 ========================
        # TODO：计算指定末端帧在构型 q 下的几何雅可比。
        #
        # 1. 从 arm.pin_model 取得模型，为本次计算创建 pin.Data(model)。
        # 2. 将 q 整理为一维浮点数组。
        # 3. 用 model.getFrameId(frame) 查询帧编号。
        # 4. 调用 pin.computeFrameJacobian(model, data, q, frame_id,
        #    pin.LOCAL_WORLD_ALIGNED)，返回计算结果。
        #
        # 提示：LOCAL_WORLD_ALIGNED 将线速度和角速度都表示在基座系；
        # 不要用 pin.LOCAL，否则与公共流程的参考系约定不一致。
        raise NotImplementedError(
            "请完成 PracticeJacobianSolver.jac() 中的实践实现区"
        )
        # ========================= 实践实现区结束 =========================


def motor_torque_from_external_load(
    jacobian: np.ndarray,
    external_load: Wrench,
) -> np.ndarray:
    """计算平衡末端外载荷所需的关节力矩。"""
    # ======================== 任务 2 实现区开始 ========================
    # TODO：根据末端外载荷计算各关节电机需要提供的静力平衡力矩。
    #
    # 1. 把 external_load.force 和 external_load.torque 按顺序拼成六维向量：
    #       F_ext = [fx, fy, fz, nx, ny, nz]
    # 2. 外载荷作用于机械臂时，电机平衡力矩为：
    #       tau_motor = -jacobian.T @ F_ext
    # 3. 返回形状为 (n,) 的 tau_motor，单位为 N·m。
    #
    # 负号来自受力方向约定：F_ext 是环境施加给机械臂的力；电机必须提供
    # 方向相反的力矩维持静力平衡。
    raise NotImplementedError(
        "请完成 motor_torque_from_external_load() 中的实践实现区"
    )
    # ========================= 任务 2 实现区结束 =========================


def main() -> None:
    """把两个读者接口交给第四章公共实验框架。"""
    parser = argparse.ArgumentParser(description="第四章 JoyArm 雅可比实践")
    parser.add_argument("--meshcat", action="store_true", help="播放轨迹与最小奇异值")
    args = parser.parse_args()

    run_jacobian_exercise(
        PracticeJacobianSolver(),
        motor_torque_from_external_load,
        meshcat=args.meshcat,
    )


if __name__ == "__main__":
    main()
