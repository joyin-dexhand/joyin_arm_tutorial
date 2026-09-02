"""正运动学子包：求解器接口 + 注册表。

正运动学 = 已知各关节角，算出末端（或任意连杆）在基座坐标系下的位置与姿态。

- ``FkineSolver``：求解器接口（ABC，见 fkine_solver.py）；具体算法是第二章
  教学内容（如 MDH 递推），实现子类后接入；
- ``REGISTRY``：注册表 ``{注册名: 求解器类}``——教程各章把实现的类填进来，
  config ``robotics.fkine`` 段写同名，JoyArm 即按名实例化启用。

用法二选一：直接实例化子类 ``求解器().solve(arm, q)``（``arm`` 鸭子类型，
课堂实现/单测用）；或门面 ``arm.fkine(q)``（config 选出的活动成员，应用用）。
"""
from .fkine_solver import FkineSolver

REGISTRY: dict = {}

__all__ = ["FkineSolver", "REGISTRY"]
