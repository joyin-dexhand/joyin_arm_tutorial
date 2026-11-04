"""控制律：基类接口 + 子类注册表。

- `Controller`：控制律接口（ABC，见 controller.py）；
- `REGISTRY`：注册表 `{注册名: 控制律子类}`

新子类实现步骤：
    1. 在本包内创建 `controller_<new>.py` 子类文件，实现子类 `<New>Controller`（继承自 `controller.py` 的 `Controller` 基类）；
    2. 在本文件中 import 新子类，并在 `REGISTRY` 显式加一行注册（注册名 = 类名小写+下划线）；
"""
from .controller import Controller
from .controller_joint_position_direct import JointPositionDirectController
from .controller_joint_drag_teach import JointDragTeachController

REGISTRY: dict = {
    "joint_position_direct_controller": JointPositionDirectController,
    "joint_drag_teach_controller": JointDragTeachController,
}

__all__ = ["Controller", "REGISTRY", "JointPositionDirectController", "JointDragTeachController"]
