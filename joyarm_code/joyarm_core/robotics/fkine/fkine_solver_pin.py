"""PinFkineSolver —— 基于 pinocchio 的正运动学默认实现。

直接调用 pin 原生 API（``forwardKinematics`` + ``updateFramePlacement``），
按属性约定消费 ``arm.pin_model``（每次计算新建私有 ``pin.Data``），适配任意构型（5/6/7 轴通用）。
config ``robotics.fkine`` 段写注册名 ``pin_fkine_solver``即选用。
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from . import FkineSolver
from ...utils.types import Pose

__all__ = ["PinFkineSolver"]


def _frame_id(model, frame) -> int:
    """帧名/索引 → 帧索引；未找到抛错（pin ``getFrameId`` 查不到时静默返回
    ``nframes`` 哨兵，须显式拦截才能给出可定位的错误消息）。"""
    fid = int(frame) if isinstance(frame, (int, np.integer)) \
        else model.getFrameId(str(frame))
    if not 0 <= fid < model.nframes:   # 拒越界与负索引（负值经无符号 C++ 类型产生晦涩错误）
        raise ValueError(
            f"fkine_solver_pin.py - PinFkineSolver：帧 {frame!r} 在模型中未找到；"
            f"可用帧：{[f.name for f in model.frames]}")
    return fid


class PinFkineSolver(FkineSolver):
    """正运动学默认实现：``q`` → 指定帧在基座系下的位姿（pinocchio）。"""

    def frame_pose(self, arm, q: np.ndarray, frame) -> Pose:
        """内核：``forwardKinematics`` + ``updateFramePlacement`` → ``Pose``。

        每次计算新建私有 ``pin.Data``（~5 µs）——无共享可变状态，线程安全。
        """
        model = arm.pin_model
        data = pin.Data(model)
        fid = _frame_id(model, frame)
        pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
        return Pose.from_T(pin.updateFramePlacement(model, data, fid).homogeneous)
