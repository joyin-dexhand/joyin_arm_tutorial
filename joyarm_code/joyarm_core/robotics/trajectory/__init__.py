"""轨迹域（策略接口 + 轨迹载体）。

TrajPlanner(ABC) 与 Trajectory 数据类为通用契约；具体规划算法为教程 Ch5
教学内容，实现后经 ``REGISTRY`` 注册接入。``method`` 选同族方式，config 选
整个规划器，两层不混淆。
"""
from .traj_planner import TrajPlanner, Trajectory

REGISTRY: dict = {}

__all__ = ["TrajPlanner", "Trajectory", "REGISTRY"]
