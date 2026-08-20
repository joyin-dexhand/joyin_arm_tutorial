"""``joyarm_core.backends`` —— 硬件通信后端子包（三层继承）。

继承层次（类名驼峰、文件名小写）::

    Backend                      # 通用根（backend.py）
    ├── BackendArm               # 多轴本体 → BackendArmDM
    └── BackendEnd               # 末端执行器 → BackendEndGripper

按**硬件类型**派生、再按**具体型号**派生。不设物理仿真后端：仿真 / 预演 / 教学
走 ``JoyArm`` 离线模式（``connected=False``，纯运动学计算）；可视化在兄弟包
:mod:`joyarm_ros2_ws` 用 rviz2 呈现。
"""
from .backend import Backend
from .backend_arm import BackendArm
from .backend_end import BackendEnd
from .backend_arm_dm import BackendArmDM
from .backend_end_gripper import BackendEndGripper

__all__ = [
    "Backend",
    "BackendArm",
    "BackendEnd",
    "BackendArmDM",
    "BackendEndGripper",
]
