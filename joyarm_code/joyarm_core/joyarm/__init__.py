"""``joyarm_core.joyarm`` —— JoyArm 机械臂类 + 对象创建入口。

JoyArm：机械臂类（负责把各子功能按config装配起来，对外提供统一调用入口，见 :mod:`joyarm_core.joyarm.joyarm`）；
JoyArmFactory：按型号名创建机械臂对象（命名链：工厂入参 = config/<型号>.yaml 文件名 = robot_model/<型号>/ 文件夹及URDF文件名）

创建 JoyArm 对象方式：
    from joyarm_core import joyArm
    
    arm1 = joyArm("joyarm_<型号>") 

对象的创建流程：
① 传入型号名 → ② 加载对应 config 文件 → ③ 按 config 配置初始化并创建 JoyArm 对象。

命名约定：类名驼峰，文件名小写 snake_case。

**新型号接入步骤**：

    1. 完善配置文件：拷贝 ``config/joyarm_template.yaml`` 为 `joyarm_<型号>.yaml` 并完成型号的配置填写；
    2. 放置模型资产：按照本库约定，将导出或修复后的机械臂URDF和Mesh资产放入到``robot_model/joyarm_<型号>/`` ；
    3. 连接硬件后端：拷贝 ``backend/backend_template.py`` 为 `backend/backend_<型号>.py`，并实现子类(详见``backend/backend_template.py``)。
"""
from __future__ import annotations

import logging

from .joyarm import JoyArm, load_config

__all__ = ["JoyArm", "JoyArmFactory", "joyArm", "load_config"]

logger = logging.getLogger("joyarm_core.joyarm")


class JoyArmFactory:
    """按型号名创建机械臂：加载 ``config/<型号>.yaml`` → 实例化 JoyArm。

    每次调用返回新实例（机械臂对象有状态、可连真机，不做缓存）。
    创建失败（config 缺失 / 初始化异常）返回 ``None`` 并输出失败信息。
    """

    # ----------------------------------------------------------
    # factory 工厂方法
    # ----------------------------------------------------------
    def create(self, model: str) -> JoyArm | None:
        """创建指定型号的 JoyArm 实例

        :param model: 型号名（如 ``"joyarm_dm"``）——= config 文件名（URDF 目录 ``robot_model/<型号>/`` 同名）。
        :return: 实例；**任一环节未找到 / 初始化失败时返回 ``None``**（失败信息经日志输出，可用型号见信息中列表）。
        """
        # 1) 型号 config
        cfg = load_config(model, strict=False)
        if cfg is None:
            logger.error("创建失败：config/%s.yaml 未找到或解析失败", model)
            return None
        # 2) 创建 + 初始化（异常软化：输出信息并返回 None；JoyArm 构造本身为硬失败）
        try:
            return JoyArm(model=model, config=cfg)
        except Exception as e:
            logger.exception("创建失败：『%s』初始化异常：%s", model, e)
            return None

    __call__ = create

    def list_models(self) -> list:
        """列出可用型号名（扫描 ``config/*.yaml``，排除 ``*_template.yaml`` 模板）。"""
        from .joyarm import _CONFIG_DIR

        try:
            import os

            return sorted(f[:-5] for f in os.listdir(_CONFIG_DIR)
                          if f.endswith(".yaml") and not f.endswith("_template.yaml"))
        except OSError:
            return []

joyArm = JoyArmFactory()
