"""``joyarm_core.backend`` —— 整机硬件通信后端子包（两层继承 + 注册表）。

继承层次（类名驼峰、文件名小写）::

    Backend                  # 整机后端抽象根（backend.py）：本体+末端一体，
    └── BackendDM            #   方法以 _arm / _end 后缀区分两组（真机）
                             #   （backend_dm_mujoco.py 为 MuJoCo 仿真后端占位，实现时再导入注册）

子类与机械臂型号 **1:1 对应**派生（``backend(_joyarm)_dm`` ↔ ``joyarm_dm``，
即每个型号一个专属整机后端）；``REGISTRY`` 供 config ``backend.name`` 选型
（JoyArm 解析 yaml 后经 :func:`get_backend` 构建子类实例）。
"""
from .backend import Backend
from .backend_dm import BackendDM

REGISTRY = {
    "backend_dm": BackendDM,
}

__all__ = ["Backend", "BackendDM", "REGISTRY", "get_backend"]


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
