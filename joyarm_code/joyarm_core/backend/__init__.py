"""``joyarm_core.backend`` —— 整机硬件通信后端子包（两层继承 + 注册表）。

继承层次（类名驼峰、文件名小写）::

    Backend                  # 整机后端抽象根（backend.py）：本体+末端一体，方法以 _arm / _end 后缀区分
    └── BackendDM            #   真机：DM 7 电机经U2CAN 串口转 CAN 桥，型号说明见 backend_dm.py 模块 docstring

子类与机械臂型号 **1:1 对应**派生（``backend_dm`` ↔ ``joyarm_dm``，
即每个型号一个专属整机后端）；接入新型号 = 复制 ``backend_template.py``（六步接入见该文件头）；
``REGISTRY`` 供 config ``backend.name`` 选型（JoyArm 解析 yaml 后经 :func:`get_backend` 构建子类实例）。
"""
from .backend import Backend
from .backend_dm import BackendDM

REGISTRY = {"backend_dm": BackendDM}

__all__ = ["Backend", "REGISTRY", "get_backend"]


def get_backend(name: str) -> type:
    """按注册名解析后端类（config ``backend.name`` 选型入口）。

    :param name: 注册名（如 ``"backend_dm"``）。
    :return: 后端类（调用时传 yaml ``backend:`` 段字典，``name`` 已弹出）。
    :raises ValueError: 注册名未知时抛出，并列出全部可选项。
    """
    if name not in REGISTRY:
        raise ValueError(
            f"backend/__init__.py - get_backend：『{name}』型号在 backend 中未找到；"
            f"可用：{sorted(REGISTRY)}")
    return REGISTRY[name]
