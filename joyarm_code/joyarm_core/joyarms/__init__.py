"""``joyarm_core.joyarms`` —— 设备模型层（组合根）+ 型号工厂。

JoyArm：策略成员组装 + 公开门面（:mod:`joyarm_core.joyarms.joyarm`）；
JoyArmDM：具体型号预设（:mod:`joyarm_core.joyarms.joyarm_dm`）；
JoyArmFactory/joyarm_factory：按型号名创建（命名链校验 + REGISTRY 选型，
与 backends 层的 ``REGISTRY`` + ``get_backend`` 同构）——外部推荐入口::

    from joyarm_core import joyarm_factory
    arm = joyarm_factory("joyarm_dm")   # configs/joyarm_dm.yaml → JoyArmDM

命名链（任一环节不符即报『XX』型号在XX中未找到）：
工厂入参 ↔ configs/<型号>.yaml 文件名 ↔ yaml ``basic.name`` 字段 ↔ ``REGISTRY``
注册名 ↔ ``joyarms/<型号>.py``；backend/robot/robotics 由 yaml 各段字段继续
驱动各层 REGISTRY（型号与整机后端 1:1，如 joyarm_dm ↔ backend_dm）。
"""
from .joyarm import JoyArm, load_config
from .joyarm_dm import JoyArmDM

__all__ = ["JoyArm", "JoyArmDM", "JoyArmFactory", "joyarm_factory", "load_config", "REGISTRY"]

# 型号注册表（新增型号：joyarms/<型号>.py + configs/<型号>.yaml + 此处一行）
REGISTRY = {
    "joyarm_dm": JoyArmDM,
}


class JoyArmFactory:
    """按型号名创建机械臂：加载 ``configs/<型号>.yaml`` → 校验命名链 → REGISTRY 实例化。

    每次调用返回新实例（机械臂对象有状态、可连真机，不做缓存）；
    ``create(model, **kwargs)`` 的额外参数透传型号类（如 ``load_geometry``）。
    """

    def create(self, model: str, **kwargs) -> JoyArm:
        """创建指定型号的 JoyArm 实例（离线，未连接真机）。

        :param model: 型号名（如 ``"joyarm_dm"``）——须与 configs 文件名、
            yaml ``basic.name`` 字段、joyarms 注册名一致。
        :raises ValueError: 命名链任一环节未找到该型号时抛出（列出可用项）。
        """
        cfg = load_config(model, strict=True)
        cname = (cfg.get("basic") or {}).get("name")
        if cname != model:
            raise ValueError(
                f"『{model}』型号在 configs/{model}.yaml 的 basic.name 字段中未找到"
                f"（实际为 {cname!r}）"
            )
        if model not in REGISTRY:
            raise ValueError(f"『{model}』型号在 joyarms 中未找到；可用：{sorted(REGISTRY)}")
        return REGISTRY[model](name=model, config=cfg, **kwargs)

    __call__ = create

    def list_models(self) -> list:
        """列出已注册型号名。"""
        return sorted(REGISTRY)


joyarm_factory = JoyArmFactory()
