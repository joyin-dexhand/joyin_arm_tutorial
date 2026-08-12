"""示教与遥操作（应用支撑层，Ch12 占位）。

四种数据流共用「记录流 → (变换) → 回放」管线（对齐
``chapter4_2.md`` 第十二章）：

1. **§12.1 同臂示教**（:class:`IdentityMapping`）：本机关节记录 → 本机回放。
2. **§12.2 主从缩比同构臂**（:class:`ScaledJointMapping`）：
   主臂关节 × 缩比 → 从臂关节。
3. **§12.3 主从异构臂末端**（:class:`CartesianMapping`）：
   主臂 TCP 位姿 → 从臂 IK。
4. **§12.4 VR 空间手柄**（:class:`VrControllerMapping`）：
   手柄位姿 → 从臂 IK。

轨迹载体**直接复用** :class:`joyarm.robotics.trajectory.Trajectory`（teleop 不另
定义子类）；:class:`Sample`（JointSample / TcpSample）作为采样点类型，
``save/load`` 复用 trajectory 模块的 npz 持久化。

对应章节：Ch12（``chapter4_2.md``）。
当前状态：仅签名 + docstring + ``raise NotImplementedError("Ch12 实现")``。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..utils.types import ControlMode, Pose, Twist

__all__ = [
    "Sample",
    "JointSample",
    "TcpSample",
    "Recorder",
    "Player",
    "TeleopLoop",
    # 映射
    "IdentityMapping",
    "ScaledJointMapping",
    "CartesianMapping",
    "VrControllerMapping",
    # 流式契约
    "Source",
    "Sink",
]

# 【给新手的话】本文件是"占位文件"——函数体目前都是 raise NotImplementedError。
# 这不是 bug，而是教学安排：第十二章会真正实现示教与遥操作。
# 核心思路是统一的"记录 → (变换) → 回放"管线：
#   - 示教：人拖着机械臂走一遍，记录关节轨迹，再让机器自己重复。
#   - 遥操：主端（人操作的设备）实时采集动作，经"映射"换算后驱动从端机械臂。
# 四种映射对应四种用法：同臂复现、主从缩比、主从末端、VR 手柄。


# ============================================================
# 采样点
# ============================================================
class Sample:
    """带时间戳的采样点（基类）。"""

    def __init__(self, t: float = 0.0):
        self.t: float = t


class JointSample(Sample):
    """关节空间采样点。"""

    def __init__(self, q: np.ndarray, t: float = 0.0, dq: Optional[np.ndarray] = None):
        super().__init__(t)
        self.q: np.ndarray = np.asarray(q, dtype=float)
        self.dq: Optional[np.ndarray] = dq


class TcpSample(Sample):
    """笛卡尔空间采样点（VR 手柄即用此类型）。"""

    def __init__(self, pose: Pose, t: float = 0.0, twist: Optional[Twist] = None):
        super().__init__(t)
        self.pose: Pose = pose
        self.twist: Optional[Twist] = twist


# ============================================================
# 记录 / 回放
# ============================================================
class Recorder:
    """定频记录器：源无关（采样器为 Callable）。

    用法::

        rec = Recorder(sampler=lambda: arm.get_state().joint.q, hz=100)
        rec.start()
        ...
        traj = rec.stop()   # -> Trajectory
    """

    def __init__(self, sampler: Callable, hz: int = 100):
        self.sampler = sampler
        self.hz = hz

    def start(self) -> None:
        """开始记录（独立线程）。"""
        # 占位：Ch12 实现——开一个后台线程，按固定频率调用 sampler 采样并存盘。
        raise NotImplementedError("Recorder.start 待 Ch12 实现")

    def record_sample(self) -> None:
        """手动记录一帧（供非线程化使用）。"""
        # 占位：Ch12 实现——调一次 sampler，把结果加进缓冲区（单步调试用）。
        raise NotImplementedError("Recorder.record_sample 待 Ch12 实现")

    def stop(self):
        """停止并返回记录的 :class:`joyarm.robotics.trajectory.Trajectory`。"""
        # 占位：Ch12 实现——结束线程，把缓冲区打包成 Trajectory 对象返回。
        raise NotImplementedError("Recorder.stop 待 Ch12 实现")


class Player:
    """回放器：加载轨迹 → 按模式回放。

    支持变速 / 循环；回放过 Ch11 :mod:`joyarm.safety` 限位校验。
    """

    def __init__(self, **kwargs):
        pass

    def load(self, traj) -> None:
        """加载 :class:`joyarm.robotics.trajectory.Trajectory`。"""
        # 占位：Ch12 实现——读入轨迹，准备逐帧回放。
        raise NotImplementedError("Player.load 待 Ch12 实现")

    def play(self, arm, mode: ControlMode = ControlMode.POSITION) -> None:
        """按 ``mode`` 回放到 ``arm``。"""
        # 占位：Ch12 实现——按时间戳逐帧下发指令给 arm，中途过 safety 校验。
        raise NotImplementedError("Player.play 待 Ch12 实现")


class TeleopLoop:
    """实时遥操循环：Source + 映射 + Sink。

    每个 tick：从 Source 读主端采样 → 经映射变换 → 写入 Sink（从端指令）。
    """

    def __init__(self, source: "Source", mapping, sink: "Sink", hz: int = 100):
        self.source = source
        self.mapping = mapping
        self.sink = sink
        self.hz = hz

    def start(self) -> None:
        # 占位：Ch12 实现——启动循环：每个 tick 读主端→映射→写从端，循环直至 stop。
        raise NotImplementedError("TeleopLoop.start 待 Ch12 实现")

    def stop(self) -> None:
        # 占位：Ch12 实现——结束遥操循环，安全停机。
        raise NotImplementedError("TeleopLoop.stop 待 Ch12 实现")


# ============================================================
# 映射
# ============================================================
class IdentityMapping:
    """§12.1 同臂示教：恒等映射（主 == 从）。"""

    def __call__(self, sample: Sample) -> Sample:
        # 占位：Ch12 实现——原样返回采样点（同一台臂记录后回放，无需变换）。
        raise NotImplementedError("IdentityMapping.__call__ 待 Ch12 实现")


class ScaledJointMapping:
    """§12.2 主从缩比同构臂：``q_follower = scale * q_master + offset``。"""

    def __init__(self, scale: float = 1.0, offset: Optional[np.ndarray] = None):
        self.scale = scale
        self.offset = offset if offset is not None else 0.0

    def __call__(self, sample: JointSample) -> JointSample:
        # 占位：Ch12 实现——把主臂关节角乘缩比、加偏移，得到从臂关节角（同构臂用）。
        raise NotImplementedError("ScaledJointMapping.__call__ 待 Ch12 实现")


class CartesianMapping:
    """§12.3 主从异构臂末端：master TCP → follower IK。"""

    def __call__(self, sample: TcpSample) -> JointSample:
        # 占位：Ch12 实现——取主臂末端位姿，对从臂做逆运动学（IK），得从臂关节角。
        raise NotImplementedError("CartesianMapping.__call__ 待 Ch12 实现")


class VrControllerMapping:
    """§12.4 VR 空间手柄：手柄位姿 → 从臂 IK。"""

    def __call__(self, sample: TcpSample) -> JointSample:
        # 占位：Ch12 实现——把 VR 手柄的位姿当作末端目标，对从臂做 IK。
        raise NotImplementedError("VrControllerMapping.__call__ 待 Ch12 实现")


# ============================================================
# 流式契约（主端 / 从端抽象）
# ============================================================
class Source:
    """主端采样源抽象（重力补偿臂 / 第二臂 / VR 手柄等）。"""

    def read(self) -> Sample:
        # 占位：Ch12 实现——从主端设备读一个采样点（关节角或末端位姿）。
        raise NotImplementedError("Source.read 待 Ch12 实现")


class Sink:
    """从端指令汇抽象（实机指令流）。"""

    def write(self, sample: Sample) -> None:
        # 占位：Ch12 实现——把变换后的指令下发给从端机械臂执行。
        raise NotImplementedError("Sink.write 待 Ch12 实现")
