"""``joyarm_ros2.arm_nodes`` —— 机器人 → ROS2 节点封装【Ch10 占位】。

把 :mod:`joyarm` 某一型号机械臂封装为一个 ROS2 node，对外暴露：

- 话题：发布 :class:`~joyarm.utils.types.ArmState` / ``JointState``，订阅控制指令；
- 服务：逆运动学求解、单点控制；
- 动作：轨迹执行（接 :mod:`joyarm.robotics.trajectory`）。

当前为占位，第十章（ROS2）落地。内部经 ``from joyarm import ...`` 复用核心 SDK。
"""
__all__: list[str] = []
