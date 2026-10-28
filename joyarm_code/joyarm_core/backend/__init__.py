"""`joyarm_core.backend` —— 整机硬件通信后端

继承层次（类名驼峰、文件名小写）：

    Backend
    └── Backend<型号1>
    └── Backend<...>

每个机械臂型号对应一个单独的硬件后端子类（`backend_<型号>` ↔ `<型号>`）；
新后端创建流程见 `backend_template.py`文件头 docstring。
"""
from .backend import Backend
from .backend_rebot_arm_b601_dm import BackendRebotArmB601DM

REGISTRY = {"backend_rebot_arm_b601_dm": BackendRebotArmB601DM}

__all__ = ["Backend", "REGISTRY", "get_backend"]


def get_backend(name: str) -> type:
    """按注册名解析后端类（config `backend.name` 选型入口）。

    :param name: 注册名（如 `"backend_rebot_arm_b601_dm"`）。
    :return: 后端类（构造传 yaml `backend:` 段字典值）。
    :raises ValueError: 注册名未知时抛出，并列出全部可选项。
    """
    if name not in REGISTRY:
        raise ValueError(
            f"backend/__init__.py - get_backend：『{name}』型号在 backend 中未找到；"
            f"可用：{sorted(REGISTRY)}")
    return REGISTRY[name]
