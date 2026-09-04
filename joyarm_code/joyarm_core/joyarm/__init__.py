"""``joyarm_core.joyarm`` —— joyarm 机械臂类（组合根）+ 具体型号工厂。

JoyArm：机械臂类（策略成员组装 + 公开门面，:mod:`joyarm_core.joyarm.joyarm`）；
JoyArmFactory：按型号名创建（命名链校验）——外部推荐入口::

    from joyarm_core import joyarm_factory
    arm = joyarm_factory("joyarm_dm")   # configs/joyarm_dm.yaml → JoyArm

命名链校验规则：
    工厂入参 ↔ configs/<型号>.yaml  ↔ yaml ``basic.name`` 字段；
    urdf（``basic.robot``）、backend（``backend.name``）、求解器（``robotics:`` 段），yaml 各字段驱动各层 REGISTRY。
    
    任一环节（config 文件 / 命名链 / 初始化异常）缺失或失败，**返回 ``None`` 并输出创建失败信息**（不抛异常）；
    各子功能成员的加载同构：不存在则置空 + 警告（见:func:`joyarm_core.joyarm.joyarm._build_domain`）。



**joyarm 新型号开发步骤**：

    创建配置文件    ：  configs/<型号>.yaml
    放置并修改 URDF ：  robot_model/资产 
    配置硬件通信    ：  backend/backend_*.py**（如 joyarm_dm ↔ backend_dm）。


"""
from __future__ import annotations

import logging

from .joyarm import JoyArm, load_config

__all__ = ["JoyArm", "JoyArmFactory", "joyarm_factory", "load_config"]

logger = logging.getLogger("joyarm_core.joyarm")


class JoyArmFactory:
    """按型号名创建机械臂：加载 ``configs/<型号>.yaml`` → 校验命名链 → 实例化 JoyArm。

    每次调用返回新实例（机械臂对象有状态、可连真机，不做缓存）；
    ``create(model, **kwargs)`` 的额外参数透传 :class:`JoyArm`（如 ``load_geometry``）。
    创建失败（config 缺失 / 命名链不符 / 初始化异常）返回 ``None`` 并输出失败信息。
    """

    # ----------------------------------------------------------
    # factory 工厂方法
    # ----------------------------------------------------------
    def create(self, model: str, **kwargs) -> JoyArm | None:
        """创建指定型号的 JoyArm 实例

        :param model: 型号名（如 ``"joyarm_dm"``）——须与 configs 文件名、yaml ``basic.name`` 字段一致。
        :return: 实例；**任一环节未找到 / 初始化失败时返回 ``None``**（失败信息经日志输出，可用型号见信息中列表）。
        """
        # 1) 型号 config
        cfg = load_config(model, strict=False)
        if cfg is None:
            logger.error("创建失败：configs/%s.yaml 未找到或解析失败", model)
            return None
        # 2) 命名链
        cname = (cfg.get("basic") or {}).get("name")
        if cname != model:
            logger.error("创建失败：configs/%s.yaml 的 basic.name 字段为 %r（应为 %r）",model, cname, model)
            return None
        # 3) 创建 + 初始化（异常软化：输出信息并返回 None）
        try:
            return JoyArm(model=model, config=cfg, **kwargs)
        except Exception as e:
            logger.exception("创建失败：『%s』初始化异常：%s", model, e)
            return None

    __call__ = create

    def list_models(self) -> list:
        """列出可用型号名（扫描 ``configs/*.yaml``）。"""
        from .joyarm import _CONFIGS_DIR

        try:
            import os

            return sorted(f[:-5] for f in os.listdir(_CONFIGS_DIR) if f.endswith(".yaml"))
        except OSError:
            return []


joyarm_factory = JoyArmFactory()
