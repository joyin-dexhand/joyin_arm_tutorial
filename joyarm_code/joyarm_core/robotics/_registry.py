"""_registry —— 注册工具（六域共用）。

新增算法求解器只需两步：
    1. 继承所在域的 ABC（如 ``FkineSolver``）并加上 ``@register`` 装饰器，类即按「类名小写 + 下划线」自动注册进该域的 ``REGISTRY``
    2. 在 config ``robotics:`` 段用该注册名引用::

    # joyarm_core/robotics/fkine/pin_fkine.py
    from .._registry import register
    from ..fkine import FkineSolver

    @register
    class PinFkineSolver(FkineSolver):   # 注册名自动为 "pin_fkine_solver"
        ...

命名规则：实现类名 = 算法/方法前缀 + 域基类名（如 ``Pin`` + ``FkineSolver``；
规划器/控制器同构，如 ``ToJointTrajPlanner``、``JointPidController``），
注册名由类名机械转换（``PinFkineSolver`` → ``pin_fkine_solver``）。
"""
from __future__ import annotations

import re
import sys
from typing import Optional

__all__ = ["register"]


def _snake(name: str) -> str:
    """CamelCase → snake_case（``PinFkineSolver`` → ``pin_fkine_solver``、
    ``DLSController`` → ``dls_controller``：连续大写视为一个缩写词）。"""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def register(cls=None, *, registry: Optional[dict] = None):
    """类装饰器：把域 ABC 的子类注册进其所在子包的 ``REGISTRY``。

    裸用 ``@register``（实现位于域子包内时）或带参 ``@register(registry=...)``
    （测试/包外定义时）。注册名 = 类名小写 + 下划线（与 config ``robotics:``
    段引用名一致）；重名注册后者覆盖前者（教程代码不设防）。
    """

    def _wrap(c):
        reg = registry
        if reg is None:
            mod_name = c.__module__
            if "." not in mod_name:
                raise RuntimeError(
                    f"_registry.py - register：[{c.__name__}] 定义于顶层模块 "
                    f"{mod_name!r}（无所属包，取不到 REGISTRY）；请把实现放在域"
                    f"子包内，或改用 ``@register(registry=...)``")
            pkg = sys.modules[mod_name.rsplit(".", 1)[0]]
            reg = getattr(pkg, "REGISTRY", None)
            if not isinstance(reg, dict):
                raise RuntimeError(
                    f"_registry.py - register：[{c.__name__}] 所在包 {pkg.__name__} "
                    f"未定义 REGISTRY 注册表，无法注册")
        reg[_snake(c.__name__)] = c
        return c

    return _wrap if cls is None else _wrap(cls)
