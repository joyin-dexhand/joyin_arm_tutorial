"""轨迹域（策略接口 + 轨迹载体）。

TrajPlanner(ABC) 与 Trajectory 数据类为通用契约；具体规划算法为教程 Ch5
教学内容，实现后经 ``REGISTRY`` 注册接入。用法二选一：直接实例化子类规划
（``arm`` 鸭子类型，课堂/单测用）；或 config ``robotics.traj`` 选型由 JoyArm
装入成员字典（门面方法随 Ch5 实现补充）。
"""
from .traj_planner import TrajPlanner, Trajectory

REGISTRY: dict = {}

__all__ = ["TrajPlanner", "Trajectory", "REGISTRY"]
