"""``joyarm_core.joyarm`` —— JoyArm 机械臂类 + 型号工厂。

JoyArm：机械臂类（负责把各子功能按config装配起来，对外提供统一调用入口，见 :mod:`joyarm_core.joyarm.joyarm`）；
JoyArmFactory：按型号名创建（三处名称须一致：工厂入参 = configs/<型号>.yaml 文件名 = config文件 ``basic.name`` 字段）
                ——外部推荐入口::

    from joyarm_core import joyarm_factory
    arm = joyarm_factory("joyarm_dm")   # configs/joyarm_dm.yaml → JoyArm

工厂创建对象的流程：
① 传入型号名 → ② 加载对应 config 文件（configs/<型号>.yaml）→ ③ 按 config 配置初始化并创建 JoyArm 对象（默认不连接真机）。

**joyarm 新型号机械臂接入**：

    ① 创建配置文件：configs/<型号>.yaml（复制 joyarm_template.yaml 模板填写）
    ② 放置模型资产：robot_model/<robot>/urdf + meshes（建议参考已有urdf命名规范并由 sw_urdf_exporter 导出）
    ③ 新写通信后端：backend/backend_<型号>.py（backend子包注册表加一行；型号↔后端一一对应，如 joyarm_dm ↔ backend_dm）
"""
from __future__ import annotations

import logging

from .joyarm import JoyArm, load_config

__all__ = ["JoyArm", "JoyArmFactory", "joyarm_factory", "load_config"]

logger = logging.getLogger("joyarm_core.joyarm")


class JoyArmFactory:
    """按型号名创建机械臂：加载 ``configs/<型号>.yaml`` → 校验命名链 → 实例化 JoyArm。

    每次调用返回新实例（机械臂对象有状态、可连真机，不做缓存）。
    创建失败（config 缺失 / 命名链不符 / 初始化异常）返回 ``None`` 并输出失败信息。
    """

    # ----------------------------------------------------------
    # factory 工厂方法
    # ----------------------------------------------------------
    def create(self, model: str) -> JoyArm | None:
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
            logger.error("创建失败：configs/%s.yaml 的 basic.name 字段为 %r（应为 %r）", model, cname, model)
            return None
        # 3) 创建 + 初始化（异常软化：输出信息并返回 None；JoyArm 构造本身为硬失败）
        try:
            return JoyArm(model=model, config=cfg)
        except Exception as e:
            logger.exception("创建失败：『%s』初始化异常：%s", model, e)
            return None

    __call__ = create

    def list_models(self) -> list:
        """列出可用型号名（扫描 ``configs/*.yaml``，排除 ``*_template.yaml`` 模板）。"""
        from .joyarm import _CONFIGS_DIR

        try:
            import os

            return sorted(f[:-5] for f in os.listdir(_CONFIGS_DIR)
                          if f.endswith(".yaml") and not f.endswith("_template.yaml"))
        except OSError:
            return []


joyarm_factory = JoyArmFactory()
