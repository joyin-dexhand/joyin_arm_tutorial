"""轨迹规划：规划器接口 + 注册表。

- ``TrajPlanner``：规划器接口（ABC，见 traj_planner.py）；
- ``REGISTRY``：注册表 ``{注册名: 规划器子类}``
                子类加 ``@register``装饰器即自动入表（注册名 = 类名小写 + 下划线）
                机械臂型号 config 文件的 ``robotics.trajectory`` 段按名引用。
"""
from .._registry import register
from .traj_planner import TrajPlanner

REGISTRY: dict = {}

__all__ = ["TrajPlanner", "REGISTRY", "register"]
