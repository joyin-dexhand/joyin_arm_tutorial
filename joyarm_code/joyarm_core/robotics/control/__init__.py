"""控制律：控制律接口 + 注册表。

- ``Controller``：控制律接口（ABC，见 controller.py）；
- ``REGISTRY``：注册表 ``{注册名: 控制律子类}``
        新增实现：import 后在本表加一行（注册名 = 类名小写+下划线）， config 文件 ``robotics.control``段按名引用。
"""
from .controller import Controller
from .controller_joint_position import JointPositionController

REGISTRY: dict = {
    "joint_position_controller": JointPositionController,   # 默认实现（首位）
}

__all__ = ["Controller", "JointPositionController", "REGISTRY"]
