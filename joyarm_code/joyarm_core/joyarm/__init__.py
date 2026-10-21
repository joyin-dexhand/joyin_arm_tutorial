"""`joyarm_core.joyarm` —— JoyArm 机械臂类 + 对象创建入口。

JoyArm：机械臂类（负责把各子功能按 config 装配起来，对外提供统一调用入口，见 :mod:`joyarm_core.joyarm.joyarm`）；
JoyArmFactory：按型号名创建机械臂对象（命名链：工厂入参 = config/<型号>.yaml 文件名 = robot_model/<型号>/ 文件夹及URDF文件名）；
listArm：列出当前可用的机械臂型号（三条件过滤，与工厂失败消息共用同一实现）。

创建 JoyArm 对象方式：

    from joyarm_core import joyArm

    arm1 = joyArm("joyarm_<型号>")    # 任一步校验失败抛携带具体原因的异常（不返回 None）

工厂四步固定顺序校验（任一步失败抛异常，错误消息附可用型号列表 = `listArm()` 结果）：

① config——`config/<型号>.yaml` 存在且解析为非空 dict（空文件、纯注释、yaml 语法错误均判失败）；
② backend——cfg 的 `backend.name` ∈ `backend/__init__.py` 的 `REGISTRY` 键；
③ robot_model——四路径按序存在：`robot_model/<型号>/`、`meshes/`、`urdf/`、`urdf/<型号>.urdf`；
④ 构造——调用 `JoyArm(<型号>)` 做类内初始化自检（异常向上抛、保留原因链）。

对象的创建流程：
① 传入型号名 → ② 加载对应 config 文件 → ③ 按 config 配置初始化并创建 JoyArm 对象。

命名约定：类名驼峰，文件名小写 snake_case。

**新型号接入步骤**：

    1. 完善配置文件：拷贝 `config/joyarm_template.yaml` 为 `joyarm_<型号>.yaml` 并完成型号的配置填写；
    2. 放置模型资产：按照本库约定，将导出或修复后的机械臂URDF和Mesh资产放入到`robot_model/joyarm_<型号>/` ；
    3. 连接硬件后端：拷贝 `backend/backend_template.py` 为 `backend/backend_<型号>.py`，并实现子类(详见`backend/backend_template.py`)。
"""
from __future__ import annotations

import logging
import os

import yaml

from .joyarm import JoyArm, load_config
from .joyarm import _CONFIG_DIR, _ROBOT_MODEL_DIR
from ..backend import REGISTRY

__all__ = ["JoyArm", "JoyArmFactory", "joyArm", "listArm", "load_config"]

logger = logging.getLogger("joyarm_core.joyarm")


def _model_paths_missing(name: str) -> list:
    """工厂第③步四路径的缺失清单（空列表 = 齐全）。

    四路径按序：`robot_model/<name>/`、`<name>/meshes/`、`<name>/urdf/`、
    `<name>/urdf/<name>.urdf`（型号名 = 目录名 = URDF 文件名）。
    """
    base = os.path.join(_ROBOT_MODEL_DIR, name)
    paths = [base,
             os.path.join(base, "meshes"),
             os.path.join(base, "urdf"),
             os.path.join(base, "urdf", f"{name}.urdf")]
    return [p for p in paths if not os.path.exists(p)]


def _available_models() -> list:
    """listArm 的共享实现（与工厂失败消息共用）：扫描 `config/*.yaml` 逐型号过滤。

    可行型号须**同时**满足（三条件）：

    1. `config/<name>.yaml` 存在、非空 dict、含 `joyarm` / `robotics` / `backend` 三顶层键；
    2. `backend.name` ∈ `backend/__init__.py` 的 `REGISTRY` 键；
    3. 四路径齐全（`_model_paths_missing` 为空）—— 模板 yaml 无 robot_model 目录，自然排除。
    """
    out = []
    try:
        files = sorted(os.listdir(_CONFIG_DIR))
    except OSError:
        return out
    for f in files:
        if not f.endswith(".yaml"):
            continue
        name = f[:-5]
        try:
            with open(os.path.join(_CONFIG_DIR, f), encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
        except Exception:
            continue     # 解析失败 → 不可用
        if not isinstance(cfg, dict) or not cfg:
            continue
        if not all(isinstance(cfg.get(k), dict) for k in ("joyarm", "robotics", "backend")):
            continue     # 段缺失或写成非 dict（如列表）→ 不可用
        if cfg["backend"].get("name") not in REGISTRY:
            continue
        if _model_paths_missing(name):
            continue
        out.append(name)
    return out


def listArm() -> list:
    """列出可用机械臂型号（返回 `list[str]` 并打印；与工厂失败消息共用实现）。

    三条件见 :func:`_available_models`；`joyarm_template.yaml` 三键齐全但无
    robot_model 目录，由路径条件自然排除（不做文件名特判）。
    """
    models = _available_models()
    print(f"可用机械臂型号：{models}")
    return models


class JoyArmFactory:
    """按型号名创建机械臂：四步固定顺序外部资源校验 → 实例化 JoyArm。

    每次调用返回新实例（机械臂对象有状态、可连真机，不做缓存）；任一步校验失败
    **抛携带具体原因的异常**（不返回 None），错误消息附可用型号列表（= `listArm()`
    结果）。
    """

    # ----------------------------------------------------------
    # factory 工厂方法
    # ----------------------------------------------------------
    def create(self, name: str) -> JoyArm:
        """创建指定型号的 JoyArm 实例（四步固定顺序校验）。

        :param name: 型号名（如 `"joyarm_dm"`）——= config 文件名（URDF 目录
            `robot_model/<型号>/` 同名）。
        :return: 初始化自检通过的 JoyArm 实例；四步全过 `logger.info`『创建成功』。
        :raises ValueError: ①config 缺失/解析失败、②backend 名不在 REGISTRY、
            ③robot_model 四路径缺失（消息含逐项缺失清单）、④JoyArm 构造自检失败
            （原因链保留）；均附可用型号列表。
        """
        avail = _available_models()
        # ① config：存在且解析为非空 dict（load_config 抛错消息含路径与原始解析错误）
        try:
            cfg = load_config(name)
        except Exception as e:
            raise ValueError(
                f"joyarm/__init__.py - JoyArmFactory.create：① config 校验失败：{e}；"
                f"可用型号：{avail}") from e
        # ② backend：cfg.backend.name ∈ REGISTRY（backend 段须为 dict）
        backend_cfg = cfg.get("backend")
        backend_name = backend_cfg.get("name") if isinstance(backend_cfg, dict) else None
        if not isinstance(backend_name, str) or backend_name not in REGISTRY:
            raise ValueError(
                f"joyarm/__init__.py - JoyArmFactory.create：② backend 校验失败："
                f"backend.name={backend_name!r} 不在 REGISTRY"
                f"（可用：{sorted(REGISTRY)}）；可用型号：{avail}")
        # ③ robot_model：四路径按序存在
        missing = _model_paths_missing(name)
        if missing:
            raise ValueError(
                f"joyarm/__init__.py - JoyArmFactory.create：③ robot_model 校验失败，"
                f"逐项缺失：{missing}；可用型号：{avail}")
        # ④ 构造：JoyArm(name) 类内初始化自检（异常向上抛、保留原因链）
        try:
            arm = JoyArm(name)
        except Exception as e:
            raise ValueError(
                f"joyarm/__init__.py - JoyArmFactory.create：④ 构造自检失败：{e}；"
                f"可用型号：{avail}") from e
        logger.info("joyarm/__init__.py - JoyArmFactory.create：『%s』创建成功", name)
        return arm

    __call__ = create

    def listArm(self) -> list:
        """列出可用型号名（实例方法透传模块级 :func:`listArm`）。"""
        return listArm()


joyArm = JoyArmFactory()
