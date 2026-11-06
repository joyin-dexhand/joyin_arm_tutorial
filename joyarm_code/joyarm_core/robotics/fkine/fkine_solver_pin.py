"""PinFkineSolver —— 基于 Pinocchio 的正运动学实现（URDF 通用实现）。

算法：调用 Pinocchio 原生 `forwardKinematics` + `updateFramePlacement`；
数据：消费 `arm.pin_model`（末端固定的缩减模型，nq = n_arm）；每次计算新建私有 `pin.Data`；
通用：适配任意 URDF 构型，通常作为 cfg 中默认激活的 fkine_solver。

用法：
    arm.set_fkine_solver("pin_fkine_solver")   # 设置要使用的正运动学求解器
    arm.solve_fkine(q, frame)                  # 通过 arm 调用, q 默认取 arm.joint_state_arm.q, frame 默认取 cfg 中的 frame_end
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from .fkine_solver import FkineSolver
from ...utils.types import Pose

__all__ = ["PinFkineSolver"]


def _frame_id(model, frame) -> int:
    """frame 名/序号 → 模型内帧索引。

    pin `getFrameId` 查不到帧名时静默返回 `nframes` 哨兵，须在此拦截；
    越界与负索引一并拒绝（负值经无符号 C++ 类型产生晦涩错误）。

    :param model: pin 模型。
    :param frame: frame 名或序号。
    :return: frame 索引 `int`。
    :raises ValueError: frame 在模型中不存在或索引越界。
    """
    fid = int(frame) if isinstance(frame, (int, np.integer)) else model.getFrameId(str(frame))
    if not 0 <= fid < model.nframes:
        raise ValueError(
            f"fkine_solver_pin.py - PinFkineSolver：frame {frame!r} 在模型中未找到；"
            f"可用frame：{[f.name for f in model.frames]}")
    return fid


class PinFkineSolver(FkineSolver):
    """基于 Pinocchio 的正运动学实现：`q` → 指定 frame 在基座系下的位姿。"""

    def solve(self, arm, q: np.ndarray, frame: str) -> Pose:
        """求目标 frame 在基座系下的位姿。

        每次计算新建私有 `pin.Data`（~5 µs）。

        :param arm: 机械臂对象。
        :param q: `(n,)` 关节角，弧度。
        :param frame: 目标 frame 名或序号。
        :return: `Pose`（xyz+四元数），该 frame 在基坐标系下的位姿。
        :raises ValueError: frame 在模型中不存在或索引越界。
        """
        model = arm.pin_model
        data = pin.Data(model)
        fid = _frame_id(model, frame)
        pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
        return Pose.from_T(pin.updateFramePlacement(model, data, fid).homogeneous)
