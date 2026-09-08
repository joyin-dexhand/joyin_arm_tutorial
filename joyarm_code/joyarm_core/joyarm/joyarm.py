"""``JoyArm`` —— 完整机械臂类（arm + end）。

不同机械臂型号差异全部由 config 配置。
子类创建和型号新增请参考 ``joyarm/__init__.py``。
"""
from __future__ import annotations

import copy
import logging
import os
import time
from typing import List, Optional, Union

import numpy as np
import pinocchio as pin

from ..utils.limits import clamp_to_limits, joint_limits_from_model, soft_limits
from ..utils.transforms import quat_conj, quat_mul, quat_to_axis_angle
from ..utils.types import (
    ArmState,
    ControlMode,
    JointLimits,
    Pose,
    TcpLimits,
    TrajFrame,
)

from ..backend import Backend, get_backend
from ..robotics.fkine import REGISTRY as _FKINE_REGISTRY
from ..robotics.ikine import REGISTRY as _IKINE_REGISTRY
from ..robotics.jacobian import REGISTRY as _JACOBIAN_REGISTRY
from ..robotics.dynamics import REGISTRY as _DYNAMICS_REGISTRY
from ..robotics.trajectory import REGISTRY as _TRAJ_REGISTRY
from ..robotics.trajectory import cubic_traj as _cubic_traj
from ..robotics.control import REGISTRY as _CONTROL_REGISTRY

__all__ = ["JoyArm", "load_config"]

logger = logging.getLogger("joyarm_core.joyarm")


# ============================================================
# 模块级：路径常量与配置加载（load_config 为公开 API）
# ============================================================
# configs/ 目录（joyarm.py 位于 joyarm_core/joyarm/，上溯一级即包根）
_CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
)

# robot_model/ 资产目录（URDF + meshes，运行期加载）
_ROBOT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot_model"
)


def load_config(model: str, strict: bool = False) -> Optional[dict]:
    """加载 ``configs/<model>.yaml`` 型号配置。

    :param model: 型号名（与 yaml 文件名、yaml ``basic.name`` 字段一致）。
    :param strict: 严格模式（``JoyArmFactory`` 路径）：文件缺失/解析失败抛
        ``ValueError``（列出可用型号）；缺省容错返回 ``None``。
    """
    try:
        import yaml

        with open(os.path.join(_CONFIGS_DIR, f"{model}.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"joyarm.py - load_config：顶层应为映射，实际 {type(cfg).__name__}")
        return cfg
    except Exception as e:
        if not strict:
            return None
        try:
            avail = sorted(f[:-5] for f in os.listdir(_CONFIGS_DIR) if f.endswith(".yaml"))
        except OSError:
            avail = []
        raise ValueError(
            f"joyarm.py - load_config：『{model}』型号在 configs 中未找到；可用：{avail}"
        ) from e


# ============================================================
# 模块级：六域规格常量与域构建（软失败语义）
# ============================================================
# config robotics 段 "default" 别名对应的各域默认注册名
_DOMAIN_DEFAULTS = {
    "fkine": "pin",
    "ikine": "pin",
    "jacobian": "pin",
    "dynamics": "pin",
    "traj": "auto",
    "control": "auto",
}

# 域 → 注册表（六域统一字典化；config 规格可为单值或列表，全部加载、首个活动）
_DOMAIN_REGISTRIES = {
    "fkine": _FKINE_REGISTRY,
    "ikine": _IKINE_REGISTRY,
    "jacobian": _JACOBIAN_REGISTRY,
    "dynamics": _DYNAMICS_REGISTRY,
    "traj": _TRAJ_REGISTRY,
    "control": _CONTROL_REGISTRY,
}

# 运行期可设参数白名单（set_config 点路径 → 应用函数；config 文件不回写）
_SETTABLE = {
    "joyarm.joint_soft_margins": "_set_arm_soft_margin",
    "joyarm.end_soft_margins": "_set_end_soft_margin",
}


def _build_domain(domain: str, registry: dict, spec) -> dict:
    """按 config 规格实例化一域策略成员（软失败语义，架构约束）。

    :param spec: 注册名字符串 / ``{name:..., **参数}`` / 上述的**列表**（全部加载）。
    :return: 成员字典 ``{注册名: 实例}``——域未配置返回空；注册名不存在/实例化
        失败时输出警告并跳过该成员（置空，不中断创建）。
    """
    if not spec:
        return {}
    specs = list(spec) if isinstance(spec, (list, tuple)) else [spec]
    members: dict = {}
    for s in specs:
        if isinstance(s, str):
            name, params = s, {}
        elif isinstance(s, dict):
            params = dict(s)
            name = params.pop("name", None)
            if name is None:
                logger.warning("robotics.%s 规格缺 name 键，跳过该成员：%r", domain, s)
                continue
        else:
            logger.warning("robotics.%s 规格类型非法（%s），跳过", domain, type(s).__name__)
            continue
        if name == "default":
            name = _DOMAIN_DEFAULTS.get(domain, name)
        if name not in registry:
            logger.warning("『%s』注册名在 robotics.%s 中未找到；可用：%s（该成员置空）",
                           name, domain, sorted(registry))
            continue
        try:
            members[name] = registry[name](**params)
        except Exception as e:
            logger.warning("robotics.%s『%s』实例化失败：%s（该成员置空）", domain, name, e)
    return members


class JoyArm:
    """完整机械臂类（组合根，兼容带末端执行器的 6R/7R 臂，单类）。

    :param model: 型号名（与 ``configs/<model>.yaml`` 文件名、yaml ``basic.name``
        字段一致）。
    :param urdf_path: URDF 路径；缺省由 config ``basic.robot`` 解析随包资产
        （``robot_model/<robot>/urdf/<robot>.urdf``）。
    :param ee_frame_name: 末端帧名；缺省取 config ``basic.ee_frame``，再回退 ``"ee"``。
    :param config: 型号 YAML 字典（``basic``/``joyarm``/``robotics``/``backend`` 段）；
        缺省自动加载 ``configs/<model>.yaml``，一次性存入 ``self._config``。
    """

    # ----------------------------------------------------------
    # init 初始化（成员变量全量定义 → 逐步赋真实值；config 驱动构造）
    # ----------------------------------------------------------
    def __init__(self,
        model: str,
        urdf_path: Optional[str] = None,
        ee_frame_name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        # ----------------------------------------------------------
        # joyarm 成员变量（全量定义：先开辟空间并赋默认值，随后逐步赋真实值）
        # ----------------------------------------------------------
        # ---- 基本属性 ----
        self._config: dict = {}                       # config 四段快照（basic/joyarm/robotics/backend）
        self._urdf_path: str = ""                     # URDF 文件路径
        self.model: str = ""                          # 型号名
        self.n: int = 0                               # 关节数（nq）
        self.connected: bool = False                  # 真机连接状态（默认离线）
        # ---- pinocchio 构型（URDF 驱动）----
        self.pin_model: Optional[pin.Model] = None    # URDF 解析构型模型
        self.pin_data: Optional[pin.Data] = None      # 模型配套计算数据
        # ---- 末端帧 ----
        self.ee_frame_name: str = ""                  # 末端帧名
        self.ee_frame_id: int = -1                    # 末端帧索引（pinocchio frames 表）
        # ---- 限位（本体硬/软 + 末端硬/软 + TCP 空间）----
        self.joint_limits: Optional[JointLimits] = None        # 本体硬限位（URDF 解析）
        self.joint_limits_soft: Optional[JointLimits] = None   # 本体软限位（硬限位 margin 内缩）
        self.end_limits: Optional[JointLimits] = None          # 末端硬限位（无末端段为 None）
        self.end_limits_soft: Optional[JointLimits] = None     # 末端软限位（无末端为 None）
        self.tcp_limits: TcpLimits = TcpLimits()               # TCP 空间限位（默认占位）
        # ---- 特征位形 ----
        self._q_zero: np.ndarray = np.zeros(0)        # 硬件零位（编码器标零基准）
        self._q_home: np.ndarray = np.zeros(0)        # 上电初始位形
        self._q_neutral: np.ndarray = np.zeros(0)     # 数值求解默认初值
        # ---- 关节名称 ----
        self._joint_names: List[str] = []             # 本体关节名（config backend 顺序）
        # ---- 六域策略成员（config 选型，全部加载、首个活动）----
        self._fkine_solvers: dict = {}                # fkine 成员字典 {注册名: 实例}
        self._ikine_solvers: dict = {}                # ikine 成员字典
        self._jacobian_solvers: dict = {}             # jacobian 成员字典
        self._dynamics_solvers: dict = {}             # dynamics 成员字典
        self._traj_planners: dict = {}                # traj 成员字典
        self._controllers: dict = {}                  # control 成员字典
        self._active_name: dict = {}                  # 各域活动成员注册名（域 → 名）
        # ---- 整机后端 ----
        self._backend: Optional[Backend] = None       # 整机通信后端（软失败可置空）
        # ---- 轨迹桥（发布即不可变）----
        self._target_traj: Optional[List[TrajFrame]] = None    # 目标序列（应用任务写）
        self._current_frame: Optional[TrajFrame] = None        # 当前帧（规划线程写）

        # ----------------------------------------------------------
        # init 逐步赋真实值（config → URDF/末端帧 → pinocchio → 限位 →
        # 关节名/特征位形 → TCP 限位 → 后端 → 六域字典）
        # ----------------------------------------------------------
        # ---- 型号 config（工厂注入；直用时自动加载 configs/<model>.yaml）----
        if config is None:
            config = load_config(model)
            if config is None:
                logger.warning("configs/%s.yaml 未找到；以空 config 构造（离线纯 URDF 模式）",
                               model)
        cfg = config or {}
        self._config = cfg

        # ---- urdf / ee_frame：参数 > config basic 段 > 内置默认 ----
        basic = cfg.get("basic") or {}
        if urdf_path is None:
            robot = basic.get("robot")
            if robot is None:
                raise ValueError(
                    "joyarm.py - JoyArm.__init__：未指定 urdf_path，且 config basic.robot "
                    "缺失，无法解析 URDF。\n请传入 urdf_path，或在 configs yaml 配置 "
                    "basic.robot（如 joyarm_dm_fixend）。"
                )
            urdf_path = self._resolve_robot_urdf(robot)
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(
                f"joyarm.py - JoyArm.__init__：未找到 URDF 文件：{urdf_path}\n"
                f"请将正式 URDF 放入 joyarm_core/robot_model/。"
            )
        self._urdf_path = urdf_path
        ee_frame_name = ee_frame_name or basic.get("ee_frame") or "ee"

        # ---- 构建 pinocchio 模型（构型加载，URDF 驱动）----
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()

        # ---- 基本属性 ----
        self.n = self.pin_model.nq
        self.model = model

        # ---- 末端帧 ----
        self.ee_frame_name = ee_frame_name
        # pinocchio getFrameId 对未知名不抛异常而是返回 nframes，据此判缺
        self.ee_frame_id = self.pin_model.getFrameId(ee_frame_name)
        if self.ee_frame_id >= len(self.pin_model.frames):
            raise ValueError(
                f"joyarm.py - JoyArm.__init__：URDF 中找不到末端帧 '{ee_frame_name}'；"
                f"可用帧：{[f.name for f in self.pin_model.frames]}"
            )

        # ---- config joyarm 段（margin / 特征位形 / TCP 限位共用源）----
        jcfg = cfg.get("joyarm") or {}
        utils_cfg = (cfg.get("basic") or {}).get("utils") or {}
        for old_key in ("joint_limits_soft_margin", "joint_soft_margins"):
            if old_key in utils_cfg:
                logger.warning(
                    "config basic.utils.%s 已迁移至 joyarm 段（joint_soft_margins /"
                    " end_soft_margins 四键绝对余量），本次不生效；请移动该键", old_key)
        arm_soft_margins = jcfg.get("joint_soft_margins") or {}
        end_soft_margins = jcfg.get("end_soft_margins") or {}

        # ---- 关节限位（硬 + 软）----
        # 硬限位来自 URDF（经 pinocchio 解析）；软限位 = 硬限位按 config
        # joyarm.joint_soft_margins 四种逐关节绝对余量（上/下/速度/力矩）内缩，
        # 未配置时全零 margin（软=硬）
        self.joint_limits = joint_limits_from_model(self.pin_model)
        self.joint_limits_soft = soft_limits(self.joint_limits, margins=arm_soft_margins)

        # ---- 末端限位（硬限位自 backend.end.joints 逐电机解析；无末端段为 None）----
        backend_cfg = cfg.get("backend") or {}
        self.end_limits = Backend.end_limits_from_cfg(backend_cfg)
        self.end_limits_soft = (soft_limits(self.end_limits, margins=end_soft_margins)
                                if self.end_limits is not None else None)

        # ---- 关节名称（config backend.arm.joints 顺序；教学/日志按名寻址）----
        self._joint_names = [
            str(j.get("name", f"joint{i + 1}"))
            for i, j in enumerate(backend_cfg.get("arm", {}).get("joints") or [])
        ]

        # ---- 特征位形（config joyarm 段优先，缺失回退 q_zero）----
        # 长度合法时统一裁剪到硬限位；长度错误不在此拦截——软失败构造，由 check_config 报告

        def _pose(key, fallback):
            arr = np.asarray(jcfg.get(key, fallback), dtype=float).reshape(-1)
            return np.clip(arr, self.joint_limits.q_min, self.joint_limits.q_max) \
                if arr.size == self.n else arr

        self._q_zero = _pose("q_zero", np.zeros(self.n))
        self._q_home = _pose("q_home", self._q_zero)
        self._q_neutral = _pose("q_neutral", self._q_zero)

        # ---- TCP 空间限位（joyarm 段提供则覆盖默认占位）----
        if jcfg.get("tcp_limits"):
            self._apply_tcp_limits(jcfg["tcp_limits"])

        # ---- 整机通信后端：config backend 段 name 选型（软失败：无效置空+警告）----
        if backend_cfg:
            backend_cfg = dict(backend_cfg)
            backend_name = backend_cfg.pop("name", None)
            if backend_name is None:
                logger.warning("config backend 段缺少选型键 name；后端置空（离线计算仍可用）")
            else:
                try:
                    # 限位参数传递（组合根职责）：URDF 硬限位 + 两份 margin；
                    # 软限位由后端基类 __init__ 自建（限位裁剪唯一执行点，独立使用后端同样生效）
                    self._backend = get_backend(backend_name)(
                        backend_cfg,
                        joint_limits=self.joint_limits,
                        joint_soft_margins=arm_soft_margins,
                        end_soft_margins=end_soft_margins)
                except Exception as e:
                    logger.error("后端『%s』构建失败：%s（后端置空，离线计算仍可用）",
                                 backend_name, e)
                    self._backend = None

        # ---- 六域策略成员字典（config 选型，全部加载、首个为活动；软失败）----
        # 各域 REGISTRY 默认空表（仅 ABC 接口）——域未配置即跳过，教学各章实现
        # 注册后经 config 选型接入
        robotics_cfg = cfg.get("robotics") or {}
        for domain, registry in _DOMAIN_REGISTRIES.items():
            members = _build_domain(domain, registry, robotics_cfg.get(domain))
            self._members(domain).update(members)
            self._active_name[domain] = next(iter(members), None)

    # ---- init 内部：随包 URDF 解析（robot 资产加载逻辑）----
    @staticmethod
    def _resolve_robot_urdf(robot: str) -> str:
        """解析随包 URDF：``robot_model/<robot>/urdf/<robot>.urdf``。

        :raises ValueError: 型号目录/URDF 不存在（列出可用型号）。
        """
        path = os.path.join(_ROBOT_MODEL_DIR, robot, "urdf", f"{robot}.urdf")
        if os.path.isfile(path):
            return path
        try:
            avail = sorted(d for d in os.listdir(_ROBOT_MODEL_DIR)
                           if os.path.isdir(os.path.join(_ROBOT_MODEL_DIR, d)))
        except OSError:
            avail = []
        raise ValueError(
            f"joyarm.py - _resolve_robot_urdf：『{robot}』型号在 robot_model 中未找到；可用：{avail}")

    # ---- init 内部：TCP 空间限位应用（config joyarm 段）----
    def _apply_tcp_limits(self, tl: dict) -> None:
        """由 config 的 tcp_limits 段覆盖默认末端限位。

        ``workspace_box`` 兼容两种写法：``[[xmin,ymin,zmin],[xmax,ymax,zmax]]``
        （yaml 常用，min/max 两行）或每轴一行 ``[min,max]`` 的 ``(3,2)``；内部
        统一为 :class:`TcpLimits` 约定的 ``(3,2)``。
        """
        box = np.asarray(tl.get("workspace_box",
                                [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.8]]), dtype=float)
        if box.shape == (2, 3):          # [min 行, max 行] → 每轴 [min, max]
            box = box.T
        self.tcp_limits = TcpLimits(
            workspace_box=box,
            v_lin_max=float(tl.get("v_lin_max", 0.0)),
            v_ang_max=float(tl.get("v_ang_max", 0.0)),
            f_max=float(tl.get("f_max", 0.0)),
            t_max=float(tl.get("t_max", 0.0)),
        )

    # ----------------------------------------------------------
    # basic 基本属性与特征位形（只读门面；离线可用）
    # ----------------------------------------------------------
    # ---- 特征位形（只读，返回拷贝）----
    @property
    def q_zero(self) -> np.ndarray:
        """硬件零位 ``(n,)``（编码器标零基准）。"""
        return self._q_zero.copy()

    @property
    def q_home(self) -> np.ndarray:
        """上电初始位形 ``(n,)``。"""
        return self._q_home.copy()

    @property
    def q_neutral(self) -> np.ndarray:
        """数值求解默认初值 ``(n,)``（如 IK 迭代起点）。"""
        return self._q_neutral.copy()

    # ---- 关节名称（config 顺序；按名寻址）----
    @property
    def joint_names(self) -> List[str]:
        """本体关节名列表（config ``backend.arm.joints`` 顺序；无 backend 段为空）。"""
        return list(self._joint_names)

    def joint_index(self, name: str) -> int:
        """关节名 → 索引（教学/日志按名寻址）。

        :raises ValueError: 名称不在关节名列表中（消息列出可用名）。
        """
        try:
            return self._joint_names.index(str(name))
        except ValueError:
            raise ValueError(
                f"joyarm.py - joint_index：关节名 {name!r} 未找到；"
                f"可用：{self._joint_names}") from None

    # ---- 关节角采样（基于软限位；裁剪/校验统一走 utils.clamp_to_limits）----
    def rand_q(self,
        size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """在**软限位**内均匀采样关节角。"""
        rng = rng if rng is not None else np.random.default_rng()
        low = self.joint_limits_soft.q_min
        high = self.joint_limits_soft.q_max
        if size is None:
            return rng.uniform(low, high)
        return rng.uniform(low, high, size=(size, self.n))

    # ---- 打印表示 ----
    def __repr__(self) -> str:
        domains = {d: (self._active_name.get(d) or "-") for d in _DOMAIN_REGISTRIES}
        return (
            f"{type(self).__name__}(model={self.model!r}, n={self.n}, "
            f"ee_frame={self.ee_frame_name!r}, "
            f"solvers={domains}, "
            f"backend={'yes' if self._backend else 'no'}, "
            f"{'connected' if self.connected else 'offline'})"
        )

    # ----------------------------------------------------------
    # config 配置管理（读取 / 运行期设置 / 自检；架构约束：配置功能全集）
    # ----------------------------------------------------------
    # ---- 配置读取 ----
    def get_config(self) -> dict:
        """当前配置深拷贝快照（四段 ``basic``/``joyarm``/``robotics``/``backend``）。

        教学算法读取型号教学数据（如 MDH 参数 ``joyarm.arm_mdh_and_limits``）统一
        经此接口从**类内已加载的 config**获取，不重新加载 yaml 文件。
        """
        return copy.deepcopy(self._config)

    # ---- 运行期设置（白名单内即时生效；config 文件不回写）----
    def set_config(self, path: str, value) -> None:
        """运行期设置参数（白名单内即时生效；config 文件不回写，重启以 yaml 为准）。

        :param path: 点路径，如 ``"joyarm.joint_soft_margins"``。
        :raises ValueError: 路径不在白名单内（其余项请手改 configs yaml，保留注释）。
        """
        handler = _SETTABLE.get(path)
        if handler is None:
            raise ValueError(
                f"joyarm.py - set_config：path={path!r} 不在运行期可设白名单内：{sorted(_SETTABLE)}"
                f"（其余配置请直接编辑 configs yaml）"
            )
        getattr(self, handler)(value)
        # 写回内存 config 快照（尽力而为：路径中遇列表段（如多载规格）则跳过，
        # 运行时值已由 handler 生效，快照以 yaml 结构为准）
        node = self._config
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.get(k) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                node = None
                break
        if isinstance(node, dict):
            node[keys[-1]] = value

    def _set_arm_soft_margin(self, v) -> None:
        """重算本体软限位并同步后端守卫（``v`` 为四键 margin 字典）。"""
        self.joint_limits_soft = soft_limits(self.joint_limits, margins=v or {})
        if self._backend is not None:   # 后端基类守卫同步换用新软限位
            self._backend.set_arm_limits_soft(self.joint_limits_soft)

    def _set_end_soft_margin(self, v) -> None:
        """重算末端软限位并同步后端守卫；无末端执行器时不可设。"""
        if self.end_limits is None:
            raise ValueError(
                "joyarm.py - set_config：无末端执行器（config backend.end.joints），"
                "joyarm.end_soft_margins 不可设置")
        self.end_limits_soft = soft_limits(self.end_limits, margins=v or {})
        if self._backend is not None:
            self._backend.set_end_limits_soft(self.end_limits_soft)

    # ---- 自检（check_config 离线可跑 / check_hardware 需 connect）----
    def check_config(self) -> None:
        """配置最小自检（离线可跑）：basic 段齐全 / 命名链 / URDF 存在 /
        特征位形长度与限位 / backend 选型与关节数一致。

        最小检查语义：通过则静默返回；发现问题则 ``ValueError`` 携带全部
        问题一次抛出。robotics 域成员加载为软失败架构（未加载时门面调用
        显性报错），不在此检查。

        :raises ValueError: 配置存在问题时抛出，消息列出全部问题。
        """
        problems: List[str] = []
        cfg = self._config
        if "basic" not in cfg:
            problems.append("缺少配置段：basic")
        basic = cfg.get("basic") or {}
        if basic.get("name") not in (None, self.model):
            problems.append(f"basic.name={basic.get('name')!r} 与 model={self.model!r} 不一致")
        if not os.path.isfile(self._urdf_path):
            problems.append(f"URDF 文件不存在：{self._urdf_path}")
        jcfg = cfg.get("joyarm") or {}
        for key in ("q_zero", "q_home", "q_neutral"):
            v = jcfg.get(key)
            if v is None:
                continue
            arr = np.asarray(v, dtype=float)
            if arr.size != self.n:
                problems.append(f"joyarm.{key} 长度 {arr.size} ≠ 关节数 {self.n}")
            elif ((arr < self.joint_limits.q_min - 1e-9).any()
                    or (arr > self.joint_limits.q_max + 1e-9).any()):
                problems.append(f"joyarm.{key} 超出关节限位")
        bcfg = cfg.get("backend") or {}
        if bcfg:
            if self._backend is None:
                problems.append("backend 已配置但后端未加载")
            else:
                nb = len((bcfg.get("arm") or {}).get("joints") or [])
                if nb and nb != self.n:
                    problems.append(f"backend.arm.joints 数量 {nb} ≠ 关节数 {self.n}")
        if problems:
            raise ValueError(
                "joyarm.py - check_config：配置自检未通过：\n"
                + "\n".join(f"  - {p}" for p in problems))

    def check_hardware(self) -> None:
        """硬件最小自检（需 ``connect()``）：逐关节通讯/电机故障/编码器有效性，
        末端电机同构检查（通讯/故障）。

        最小检查语义：通过则静默返回；发现硬故障则 ``RuntimeError`` 携带全部
        问题一次抛出。运行期安全监控（关节过温、碰撞等）归 ROS2 节点，不在
        核心库；指令越限由后端基类限位守卫（``Backend.send_*`` 模板）兜底。

        :raises RuntimeError: 未连接真机，或存在硬故障时抛出（消息列出全部问题）。
        """
        self._require_connected()
        problems: List[str] = []
        st = self.get_arm_state()
        js = st.joint
        for i in range(self.n):
            if not bool(np.asarray(js.comm_ok)[i]):
                problems.append(f"本体关节[{i}] 通讯无应答")
            if bool(np.asarray(js.error)[i]):
                problems.append(f"本体关节[{i}] 电机故障（故障标志置位）")
            if not bool(np.asarray(js.angle_ok)[i]):
                problems.append(f"本体关节[{i}] 编码器角度无效")
        try:
            es = self._backend.read_state_end()
        except Exception as e:
            es = {}
            problems.append(f"末端状态读取失败：{e}")
        for i, ok in enumerate(es.get("comm_ok", [])):
            if not ok:
                problems.append(f"末端电机[{i}] 通讯无应答")
        for i, err in enumerate(es.get("error", [])):
            if err:
                problems.append(f"末端电机[{i}] 电机故障（故障标志置位）")
        if problems:
            raise RuntimeError(
                "joyarm.py - check_hardware：硬件自检未通过：\n"
                + "\n".join(f"  - {p}" for p in problems))

    # ----------------------------------------------------------
    # robotics 六域策略（成员字典管理 + 六域求解门面 → 活动策略成员）
    # ----------------------------------------------------------
    # ---- 成员字典访问 / 切换（六域统一）----
    def _members(self, domain: str) -> dict:
        """域 → 成员字典。"""
        return {
            "fkine": self._fkine_solvers,
            "ikine": self._ikine_solvers,
            "jacobian": self._jacobian_solvers,
            "dynamics": self._dynamics_solvers,
            "traj": self._traj_planners,
            "control": self._controllers,
        }[domain]

    def _active(self, domain: str):
        """活动成员；域未加载时抛 ``RuntimeError``（软失败在门面调用处显性化）。"""
        d = self._members(domain)
        name = self._active_name.get(domain)
        if name is None or name not in d:
            raise RuntimeError(
                f"joyarm.py - _active：robotics.{domain} 成员未加载"
                f"（config 未配置或注册名未实现）；已加载：{sorted(d) or '无'}。"
                f"教程各章实现算法并注册后，经 configs yaml 的 robotics.{domain} 段选型接入"
            )
        return d[name]

    def _pick(self, domain: str, name: Optional[str]):
        """按注册名取已加载成员（``None`` = 活动成员）。"""
        if name is None:
            return self._active(domain)
        d = self._members(domain)
        if name not in d:
            raise ValueError(
                f"joyarm.py - _pick：『{name}』未在 robotics.{domain} 已加载成员中；"
                f"已加载：{sorted(d)}"
            )
        return d[name]

    def set_solver(self, domain: str, name: str):
        """运行期切换活动成员（按注册名；域 ∈ fkine/ikine/jacobian/dynamics/traj/control）。"""
        if domain not in _DOMAIN_REGISTRIES:
            raise ValueError(
                f"joyarm.py - set_solver：未知域 {domain!r}；可用：{sorted(_DOMAIN_REGISTRIES)}")
        inst = self._pick(domain, name)   # 校验已加载
        self._active_name[domain] = name
        return inst

    def list_solvers(self, domain: str) -> list:
        """列出某域已加载成员注册名（首个为活动成员）。"""
        return sorted(self._members(domain))

    # ---- fkine 正运动学（参数排序：通用在前、特有 keyword-only 在后）----
    def fkine(self, q: np.ndarray, frame: Union[str, int], rep: str = "pose"):
        """正运动学（``frame`` 目标帧名/索引，必填；``rep`` 取 ``pose``（默认，xyz+四元数）/``T``（4×4 矩阵）/``se3``（pin.SE3））。"""
        return self._active("fkine").solve(self, q, frame=frame, rep=rep)

    # ---- ikine 逆运动学 ----
    def ikine(self, target: Pose, frame: Union[str, int], q0: np.ndarray, **kw):
        """逆运动学单解（``q0`` ``(n,)`` 必填：数值法迭代起点 / 解析法限位剔除后
        选最近解的参考；``tol``/``iters`` 等为求解器特有参数）。"""
        return self._active("ikine").solve(self, target, frame, q0, **kw)

    def ikine_all(self, target: Pose, frame: Union[str, int], **kw):
        """逆运动学全部解析解（``q`` 为 ``(K,n)``，经 ±2π 平移尽量落入限位；
        数值法实现不支持）。"""
        return self._active("ikine").solve_all(self, target, frame, **kw)

    # ---- jacobian 雅可比及衍生量 ----
    def jac(self, q: np.ndarray, frame: Union[str, int], ref: str = "base"):
        """雅可比 J(q)（``ref`` 取 ``local``/``base``：末端帧系 / 基座系）。"""
        return self._active("jacobian").jac(self, q, frame=frame, ref=ref)

    def manipulability(self, q: np.ndarray, frame: Union[str, int]) -> float:
        """Yoshikawa 可操作度（雅可比衍生量）。"""
        return self._active("jacobian").manipulability(self, q, frame=frame)

    def cond_number(self, q: np.ndarray, frame: Union[str, int]) -> float:
        """雅可比条件数（雅可比衍生量）。"""
        return self._active("jacobian").cond_number(self, q, frame=frame)

    def statics(self, q: np.ndarray, F: np.ndarray, frame: Union[str, int]):
        """静力学 τ = JᵀF（雅可比衍生量）。"""
        return self._active("jacobian").statics(self, q, F, frame=frame)

    # ---- dynamics 动力学 ----
    def idyn(self, q, dq, ddq, f_ext=None):
        """逆动力学（委托活动动力学成员）。"""
        return self._active("dynamics").idyn(self, q, dq, ddq, f_ext=f_ext)

    def mass_matrix(self, q):
        """关节空间惯量矩阵 M(q)。"""
        return self._active("dynamics").mass_matrix(self, q)

    def coriolis(self, q, dq):
        """科氏+向心项 C(q,q̇)q̇。"""
        return self._active("dynamics").coriolis(self, q, dq)

    def gravity(self, q):
        """重力项 G(q)。"""
        return self._active("dynamics").gravity(self, q)

    def cartesian_inertia(self, q, frame: Union[str, int]):
        """笛卡尔惯量 Λ=J⁻ᵀMJ⁻¹（M ⊕ ``arm.jac`` 模板）。"""
        return self._active("dynamics").cartesian_inertia(self, q, frame=frame)

    # ----------------------------------------------------------
    # traj 轨迹桥（规划线程 ⇄ 控制线程的数据管道；线程本体属 Ch5/Ch6 教学内容）
    # ----------------------------------------------------------
    # ---- 目标序列（写：应用线程低频 / 读：规划线程）----
    def set_target_traj(self, targets) -> None:
        """写入目标序列（写者：应用线程，低频）。

        :param targets: :class:`TrajFrame` 单帧或列表（单帧自动归一为列表）。
            目标有效性判别在规划求解时由 ``TrajPlanner._check_targets`` 执行。
        """
        self._target_traj = ([targets] if isinstance(targets, TrajFrame)
                             else list(targets))

    def get_target_traj(self) -> Optional[List[TrajFrame]]:
        """读取目标序列（读者：规划线程；返回当前快照引用，规划期勿由他方改动）。"""
        return self._target_traj

    # ---- 当前帧（写：规划线程 ≈控制频率 / 读：控制线程）----
    def set_current_frame(self, frame: TrajFrame) -> None:
        """写入当前轨迹帧（写者：规划线程，≈控制频率）。

        单步原子引用赋值——控制线程任意时刻读到的都是最新**完整**帧（最多
        滞后一个换帧周期）；帧对象发布后视为不可变，勿再修改。
        """
        self._current_frame = frame

    def get_current_frame(self) -> Optional[TrajFrame]:
        """读取当前轨迹帧（读者：控制线程；首帧发布前为 ``None`` 初始态）。"""
        return self._current_frame

    # ----------------------------------------------------------
    # backend 真机连接与执行（依赖 _backend；connect 后才可执行 arm_*/end_*）
    # ----------------------------------------------------------
    # ---- 连接管理与前置校验 ----
    def connect(self) -> None:
        """连接真机（整机后端：本体 + 末端）。"""
        self._require_backend()
        self._backend.connect()
        self.connected = bool(self._backend.connected)

    def disconnect(self) -> None:
        """断开真机（整机后端：本体 + 末端）。"""
        if self._backend is not None:
            self._backend.disconnect()
        self.connected = False

    def __enter__(self) -> "JoyArm":
        """上下文管理入口：未连接时自动 ``connect()``；退出时安全收尾。"""
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """退出收尾（尽力而为）：失能本体/末端 → 断开；各步失败仅告警不抛。"""
        for name, fn in (("disable_arm", self.disable_arm),
                         ("disable_end", self.disable_end),
                         ("disconnect", self.disconnect)):
            try:
                fn()
            except Exception as e:
                logger.warning("__exit__：%s 失败：%s", name, e)

    def _require_connected(self) -> None:
        """执行类方法前置：未连接真机（离线）时抛 ``RuntimeError``。"""
        if not self.connected:
            raise RuntimeError(
                f"joyarm.py - _require_connected：[{self.model}] 未连接真机（离线）；"
                f"请先 connect()。")

    def _require_backend(self) -> None:
        """后端前置：无后端（未配置/选型无效）时抛 ``RuntimeError``。"""
        if self._backend is None:
            raise RuntimeError(
                f"joyarm.py - _require_backend：[{self.model}] 无整机后端"
                f"（config backend 段缺失或选型无效），无法执行硬件操作。"
            )

    # ---- 紧急阻尼（纯后端安全操作，任意控制状态可用）----
    def damping_mode(self, kd: float = 10.0) -> None:
        """紧急阻尼模式：**任何状态**下将全部电机（本体 + 末端）切为 MIT 纯阻尼
        （``kp=q=dq=tau=0, kd``）——电机仅产生 ∝ 速度的黏滞阻力，防发疯/防坠落。

        依次执行：本体/末端切 MIT → 下发阻尼帧 → 尽力使能（故障电机不阻断其余）
        → 再发一帧（使能即生效）。各阶段 **best-effort**：单阶段失败仅记警告不
        抛出，保证尽量多的电机收到指令。

        :param kd: 阻尼系数（N·m·s/rad），默认 10.0。
        :raises RuntimeError: 未连接真机 / 无后端（无法触达电机）。
        """
        self._require_connected()
        self._require_backend()
        self._damping_frames(kd)

    def _damping_frames(self, kd: float = 10.0) -> None:
        """阻尼帧序列（无线程操作）：切 MIT → 阻尼帧 → 尽力使能 → 再发一帧。"""
        z = np.zeros(self.n)

        def _send_arm(tag: str) -> None:
            try:
                self._backend.send_mit_arm(z, z, z, kp=z, kd=np.full(self.n, kd))
            except Exception as e:
                logger.warning("damping_mode：%s阻尼指令失败：%s", tag, e)

        def _send_end(tag: str) -> None:
            try:
                n_end = len(self._backend.read_state_end().get("q", []))
                if n_end:
                    ez = np.zeros(n_end)
                    self._backend.send_mit_end(ez, ez, ez, kp=ez,
                                               kd=np.full(n_end, kd))
            except Exception as e:
                logger.warning("damping_mode：%s阻尼指令失败：%s", tag, e)

        for setter, tag in ((self._backend.set_mode_arm, "本体切 MIT"),
                            (self._backend.set_mode_end, "末端切 MIT")):
            try:
                setter(ControlMode.MIT)
            except Exception as e:
                logger.warning("damping_mode：%s失败：%s", tag, e)
        _send_arm("本体")
        _send_end("末端")
        try:
            self._backend.enable_arm()
            self._backend.enable_end()
        except Exception as e:
            logger.warning("damping_mode：使能失败（可能部分电机故障，已失能者将自由）：%s", e)
        _send_arm("本体（使能后）")
        _send_end("末端（使能后）")

    # ---- 本体执行类方法（arm_*；依赖 _backend，未连接 raise）----
    def enable_arm(self, joint: Optional[int] = None) -> None:
        """使能本体关节电机（``joint=None`` 全部）。"""
        self._require_connected()
        self._backend.enable_arm(joint)

    def disable_arm(self, joint: Optional[int] = None) -> None:
        """失能本体关节电机。"""
        self._require_connected()
        self._backend.disable_arm(joint)

    def set_zero_arm(self, joint: Optional[int] = None) -> None:
        """本体零位标定（先失能，反馈无故障后再标零）。"""
        self._require_connected()
        self._backend.set_zero_arm(joint)

    def clear_fault_arm(self, joint: Optional[int] = None) -> None:
        """本体关节故障清除（验证式复位：失能清错 → 核对 → 使能 → 核对）。

        仍存在未恢复故障时后端抛 ``RuntimeError`` 汇总（电机名 + 故障码）。

        :param joint: 关节索引，``None`` 表示全部电机。
        """
        self._require_connected()
        self._backend.clear_fault_arm(joint)

    def set_mode_arm(self, mode: ControlMode = ControlMode.POSITION,
                     joint: Optional[int] = None) -> None:
        """切换本体控制模式（收指令前必须先切到对应模式；默认位置模式，``joint=None`` 全部）。"""
        self._require_connected()
        self._backend.set_mode_arm(mode, joint)

    def read_mode_arm(self, joint: Optional[int] = None) -> Optional[ControlMode]:
        """查询本体关节当前控制模式（本地缓存，离线可查；``joint=None`` 时整臂
        各关节模式唯一才返回该模式，否则 ``None``）。"""
        self._require_backend()
        return self._backend.read_mode_arm(joint)

    def get_arm_state(self) -> ArmState:
        """读取本体状态快照（委托 ``_backend.read_state_arm()``；与 ``get_end_state`` 对应）。

        fkine 求解器已注册时同步填充 ``tcp.pose``（末端位姿）；未注册（教学
        过渡态）时跳过填充，仅返回关节原始状态。

        :raises RuntimeError: 未连接真机（``connected=False``）时抛出。
        """
        self._require_connected()
        state = self._backend.read_state_arm()
        if self._active_name.get("fkine") is not None:
            state.tcp.pose = self.fkine(state.joint.q, self.ee_frame_name)   # 默认 rep="pose" → Pose
        return state

    def set_arm_command(self,
        mode: ControlMode = ControlMode.POSITION,
        q: Optional[np.ndarray] = None,
        dq: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        joint: Optional[int] = None,
    ) -> None:
        """按控制模式下发运动指令（委托 ``_backend``；三态 POSITION/VELOCITY/MIT）。

        ``joint`` 指定时为**单关节控制**（标量参数自动升维）；``joint=None`` 全部。
        MIT 模式 ``kp/kd`` 可缺省（``None`` 透传后端回退 config 增益）；
        纯力矩不设独立模式，经 MIT（``kp=kd=0``）实现。

        限位守卫在后端基类 ``Backend.send_*_arm`` 模板内（指令路径唯一裁剪点）：
        ``q`` 裁剪到软限位、``dq``/``tau`` 裁剪到幅值上限，越界告警（节流每
        0.5s 至多一条）就近裁剪（``kp``/``kd`` 为标定增益，不裁剪）；软限位
        由后端构造时自建。

        :raises RuntimeError: 未连接真机时抛出。
        :raises ValueError: 对应模式所需参数缺失时抛出。
        """
        self._require_connected()
        if mode == ControlMode.POSITION:
            if q is None:
                raise ValueError("joyarm.py - set_arm_command：POSITION 模式需要 q")
            self._backend.send_position_arm(q, joint)
        elif mode == ControlMode.VELOCITY:
            if dq is None:
                raise ValueError("joyarm.py - set_arm_command：VELOCITY 模式需要 dq")
            self._backend.send_velocity_arm(dq, joint)
        elif mode == ControlMode.MIT:
            missing = [
                name for name, val in (("q", q), ("dq", dq), ("tau", tau)) if val is None
            ]
            if missing:
                raise ValueError(f"joyarm.py - set_arm_command：MIT 模式缺少参数：{missing}")
            self._backend.send_mit_arm(q, dq, tau, kp=kp, kd=kd, joint=joint)
        else:
            raise ValueError(f"joyarm.py - set_arm_command：未知控制模式：{mode}")

    # ---- 末端执行类方法（end_*；依赖 _backend，未连接 raise）----
    def enable_end(self, joint: Optional[int] = None) -> None:
        """使能末端执行器电机（``joint=None`` 全部）。"""
        self._require_connected()
        self._backend.enable_end(joint)

    def disable_end(self, joint: Optional[int] = None) -> None:
        """失能末端执行器电机。"""
        self._require_connected()
        self._backend.disable_end(joint)

    def set_zero_end(self, joint: Optional[int] = None) -> None:
        """末端零位标定（先失能，反馈无故障后再标零）。"""
        self._require_connected()
        self._backend.set_zero_end(joint)

    def clear_fault_end(self, joint: Optional[int] = None) -> None:
        """末端电机故障清除（语义同 :meth:`clear_fault_arm`，验证式复位）。"""
        self._require_connected()
        self._backend.clear_fault_end(joint)

    def set_mode_end(self, mode: ControlMode = ControlMode.POSITION,
                     joint: Optional[int] = None) -> None:
        """切换末端控制模式（收指令前必须先切到对应模式；默认位置模式，``joint=None`` 全部）。"""
        self._require_connected()
        self._backend.set_mode_end(mode, joint)

    def read_mode_end(self, joint: Optional[int] = None) -> Optional[ControlMode]:
        """查询末端电机当前控制模式（本地缓存，离线可查；``joint=None`` 时整个末端
        各电机模式唯一才返回该模式，否则 ``None``）。"""
        self._require_backend()
        return self._backend.read_mode_end(joint)

    def end_open(self, joint: Optional[int] = None) -> None:
        """张开末端到最大（默认行程/力度；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_action_end("open", joint)

    def end_close(self, joint: Optional[int] = None) -> None:
        """闭合末端（夹到默认力度即停；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_action_end("close", joint)

    def end_zero(self, joint: Optional[int] = None) -> None:
        """末端归零（目标 = 电机弧度 0，裁剪到行程内；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_action_end("zero", joint)

    def set_end_position(self, position, joint: Optional[int] = None) -> None:
        """末端位置控制（连续量，如夹爪电机弧度；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_position_end(position, joint)

    def set_end_force(self, force, joint: Optional[int] = None) -> None:
        """末端力度控制（如夹持力 N；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_force_end(force, joint)

    def get_end_state(self, joint: Optional[int] = None) -> dict:
        """读取末端状态（字段由后端定义；值为所选电机的逐电机序列）。

        :param joint: 末端电机索引，``None`` 表示全部。
        :raises RuntimeError: 未连接真机时抛出。
        """
        self._require_connected()
        return self._backend.read_state_end(joint)

    # ---- 电机参数读写（read/write_param_{arm,end}；未连接 raise）----
    def read_param_arm(self, key: str, joint: Optional[int] = None):
        """读本体关节电机参数（key 为参数名，语义由后端定义）。

        :param key: 参数名（如 ``"pos_kp"``）。
        :param joint: 关节索引，``None`` 表示全部电机。
        :return: 指定 ``joint`` 时返回该电机参数值（float 或 int）；
            ``joint=None`` 时返回逐电机参数值列表。
        """
        self._require_connected()
        return self._backend.read_param_arm(key, joint)

    def write_param_arm(self, key: str, value, joint: Optional[int] = None,
                        persist: bool = False) -> None:
        """写本体关节电机参数。

        :param key: 参数名。
        :param value: 参数值，标量（作用于所选全部电机）或与所选电机数一致的列表。
        :param joint: 关节索引，``None`` 表示全部电机。
        :param persist: ``True`` 时写入并持久化到非易失存储。
        """
        self._require_connected()
        return self._backend.write_param_arm(key, value, joint, persist=persist)

    def read_param_end(self, key: str, joint: Optional[int] = None):
        """读末端电机参数（key 为参数名，语义由后端定义）。"""
        self._require_connected()
        return self._backend.read_param_end(key, joint)

    def write_param_end(self, key: str, value, joint: Optional[int] = None,
                        persist: bool = False) -> None:
        """写末端电机参数。"""
        self._require_connected()
        return self._backend.write_param_end(key, value, joint, persist=persist)

    # ----------------------------------------------------------
    # 运动便利与安全层（hold/lock/move_j/safe_*；move_j 为独立功能，
    # 与「轨迹桥 → 规划器 → 控制器」常规管线并行，仅安全层与直接调用使用）
    # ----------------------------------------------------------
    # ---- 原位保持与急停锁定 ----
    def hold_position(self, kp=None, kd=None, tau=None) -> None:
        """原位保持（阻抗锁定当前姿态，含 tau 前馈补偿）。

        读当前关节角 → 切 MIT 模式 → ``q=当前, dq=0, tau_ff`` 阻抗保持。
        ``tau`` 缺省自动取重力前馈：dynamics 域已注册时 ``gravity(q_cur)``，
        未注册（Ch8 前）置零并告警（退化保持）；可用 ``tau=`` 显式覆盖。
        ``kp``/``kd`` 缺省 ``None`` 透传后端回退 config ``MIT`` 增益。

        与 :meth:`damping_mode` 区分：阻尼为零目标纯黏滞（防坠落），本方法
        为锁定**当前**姿态的位置阻抗 + 重力补偿。
        """
        self._require_connected()
        q = np.asarray(self.get_arm_state().joint.q, dtype=float).reshape(-1)
        if tau is None:
            try:
                tau = np.asarray(self.gravity(q), dtype=float).reshape(-1)
            except RuntimeError:
                tau = np.zeros(self.n)
                logger.warning("hold_position：dynamics 域未注册，tau 前馈置零（退化保持）")
        self.set_mode_arm(ControlMode.MIT)
        self.set_arm_command(ControlMode.MIT, q=q, dq=np.zeros(self.n), tau=tau,
                             kp=kp, kd=kd)

    def lock_position(self) -> None:
        """急停锁定：任意模式立刻切位置模式并维持当前关节角（q 锁定）。"""
        self._require_connected()
        q = np.asarray(self.get_arm_state().joint.q, dtype=float).reshape(-1)
        self.set_mode_arm(ControlMode.POSITION)
        self.set_arm_command(ControlMode.POSITION, q=q)

    # ---- 点到点运动（move_l / teach_mode 为占位）----
    def move_j(self, q, t=None, *, rate=None,
               wait_tol: float = 0.05, wait_timeout: float = 10.0) -> None:
        """关节空间点到点阻塞运动（三次多项式插值）——**独立功能，与规划器并行**。

        边界：本方法直接借 ``robotics.trajectory.planning`` 规划纯函数 + 位置
        指令流下发，**不经**「轨迹桥 → 规划器 → 控制器」管线；仅供安全层
        （safe_home/safe_zero）与直接调用。**常规运动**一律走
        ``set_target_traj → 规划器 → 控制器 → set_arm_command`` 管线。

        调度为精确定时器（绝对时间表 + sleep/自旋混合等待，帧间隔亚毫秒
        抖动）。注意：峰值关节速度 = ``1.5·max|q−q_cur|/t``，受 config
        ``POS_VEL.vlim`` 约束，超出时实际时长 > ``t``（由末段到位等待兜底）；
        真机使用前请先安全化 config（vlim/kp/kd 封顶）。

        :param q: 目标关节角 ``(n,)``，弧度（入口裁剪到软限位——规划与到位
            判定均以裁剪后目标为准，后端守卫再裁为幂等空操作）。
        :param t: 总时长（秒）；缺省 ``max|q−q_cur|``（隐含峰值 1.5 rad/s）；
            ``t ≤ 0`` 直发目标。
        :param rate: 发送率（Hz）；缺省 config ``backend.arm.control_rate``
            （回退 100），钳制 ≤1000（DM 控制帧间隔 ≥1ms）。
        :param wait_tol: 末段到位容差（rad，经 :meth:`is_in_position`）。
        :param wait_timeout: 到位等待超时（秒），超时 ``RuntimeError``。
        :raises ValueError: 目标维度与 ``n`` 不符。
        :raises RuntimeError: 未连接 / 到位超时。
        """
        self._require_connected()
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.n:
            raise ValueError(
                f"joyarm.py - move_j：目标维度 {q.shape[0]} 与关节数 {self.n} 不符")
        # 入口裁软限位：规划、下发、到位判定统一用裁剪后目标，消除
        # 「后端裁剪到位 ≠ 判定目标」的失配超时（后端守卫再裁为幂等空操作）
        q_c = clamp_to_limits(q, self.joint_limits_soft)
        if not np.array_equal(q, q_c):
            logger.warning("move_j：目标关节角越软限位，已裁剪 %s → %s"
                           "（规划与到位判定以裁剪后为准）",
                           np.round(q, 4).tolist(), np.round(q_c, 4).tolist())
        q = q_c
        q0 = np.asarray(self.get_arm_state().joint.q, dtype=float).reshape(-1)
        if rate is None:
            rate = float(((self._config.get("backend") or {}).get("arm") or {})
                         .get("control_rate", 100.0))
        rate = min(float(rate), 1000.0)
        if t is None:
            t = float(np.max(np.abs(q - q0)))
        ts, qs, _ = _cubic_traj(q0, q, float(t), rate)
        self.set_mode_arm(ControlMode.POSITION)          # 非位置模式先切
        start = time.perf_counter()
        spin_margin = 0.001                               # 先 sleep 到 deadline−margin 再自旋
        for ti, qi in zip(ts, qs):
            deadline = start + float(ti)
            now = time.perf_counter()
            if deadline - now > spin_margin:
                time.sleep(deadline - now - spin_margin)
            while time.perf_counter() < deadline:
                pass
            self._backend.send_position_arm(qi)
        if not self._wait_in_position(q, wait_tol, wait_timeout):
            raise RuntimeError(
                f"joyarm.py - move_j：到位超时（{wait_timeout}s 内未达容差 "
                f"{wait_tol} rad；可能受 vlim 限速，请增大 t 或检查 config）")

    def move_l(self, pose, t=None, **kw) -> None:
        """笛卡尔直线阻塞运动（**占位**）：需 ikine（Ch3）+ 笛卡尔规划（Ch5）落地后实现。

        常规运动管线：``set_target_traj → 规划器 → 控制器 → set_arm_command``。
        """
        raise NotImplementedError(
            "joyarm.py - move_l：笛卡尔直线运动需 ikine（Ch3）+ 笛卡尔规划"
            "（Ch5），尚未实现；常规运动请走 轨迹桥 → 规划器 → 控制器 管线")

    def teach_mode(self, on: bool = True) -> None:
        """拖动示教模式（**占位**）：需 dynamics 重力补偿（Ch8）落地后实现。

        目标语义：MIT 模式 + ``gravity(q_cur)`` 前馈 + 零刚度（``kp=0``）——
        臂仅余重力补偿，可徒手拖动（``on=False`` 退出恢复）；示教录制/回放
        为 Ch12 章节内容。与 :meth:`hold_position` 区分：后者锁定当前姿态
        （位置阻抗），本方法零刚度自由拖动。
        """
        raise NotImplementedError(
            "joyarm.py - teach_mode：拖动示教需 dynamics 重力补偿（Ch8）落地后"
            "实现（MIT 模式 + gravity 前馈 + 零刚度 kp=0）；示教录制/回放见 Ch12")

    # ---- 安全回位 ----
    def safe_home(self, t=None, *, wait_tol: float = 0.05,
                  wait_timeout: float = 10.0) -> None:
        """安全回 home：位置模式下由当前姿态恢复到 home 位形。

        非位置模式先切位置模式，再经 :meth:`move_j`（三次多项式）运动；
        ``t`` 缺省 ``max|q_home − q_cur|``。
        """
        self._require_connected()
        self.set_mode_arm(ControlMode.POSITION)
        self.move_j(self.q_home, t, wait_tol=wait_tol, wait_timeout=wait_timeout)

    def home_to_zero(self, t=None, *, wait_tol: float = 0.05,
                     wait_timeout: float = 10.0) -> None:
        """home → zero：**先检查当前位于 home**（软限位投影后，容差 ``wait_tol``），
        再运动到零位（与 :meth:`move_j` 的入口裁剪判定基准一致）。

        :raises RuntimeError: 当前不在 home 位形（请先 :meth:`safe_home` /
            :meth:`safe_zero`）。
        """
        self._require_connected()
        q_home_eff = clamp_to_limits(self.q_home, self.joint_limits_soft)
        if not self.is_in_position(q=q_home_eff, tol_q=wait_tol):
            raise RuntimeError(
                f"joyarm.py - home_to_zero：当前不在 home 位形（容差 {wait_tol} rad）；"
                f"请先 safe_home() 或 safe_zero()")
        self.move_j(self.q_zero, t, wait_tol=wait_tol, wait_timeout=wait_timeout)

    def safe_zero(self) -> None:
        """安全回零（组合）：急停锁定 → 回 home → home 检查 → 回 zero。

        依次执行 :meth:`lock_position` → :meth:`safe_home` →
        :meth:`home_to_zero`，任一段失败即抛出中断。末端执行器不参与
        （需要时先 :meth:`end_zero`）。
        """
        self._require_connected()
        self.lock_position()
        self.safe_home()
        self.home_to_zero()

    # ---- 到位判断 ----
    def is_in_position(self, q=None, pose=None, frame=None,
                       tol_q: float = 0.05, tol_pos: float = 1e-3,
                       tol_rot: float = 1e-2) -> bool:
        """到位判断（单入口双判断）：关节目标 ``q`` 或笛卡尔目标 ``pose`` 恰一。

        :param q: 关节目标 ``(n,)``——逐关节 ``‖Δq‖∞ ≤ tol_q``（rad）。
        :param pose: 笛卡尔目标 :class:`Pose`——位置 ``‖Δp‖ ≤ tol_pos``（m）
            且姿态四元数误差角 ≤ ``tol_rot``（rad）。
        :param frame: pose 分支参考帧名；缺省 ``ee_frame_name``。
        :raises ValueError: ``q``/``pose`` 双空或双给、维度不符。
        :raises RuntimeError: 未连接；pose 分支 fkine 域未注册（无 TCP 位姿）。
        """
        if (q is None) == (pose is None):
            raise ValueError("joyarm.py - is_in_position：q 与 pose 必须恰给其一")
        st = self.get_arm_state()                        # 未连接在此抛 RuntimeError
        if q is not None:
            q = np.asarray(q, dtype=float).reshape(-1)
            if q.shape[0] != self.n:
                raise ValueError(
                    f"joyarm.py - is_in_position：目标维度 {q.shape[0]} "
                    f"与关节数 {self.n} 不符")
            cur = np.asarray(st.joint.q, dtype=float).reshape(-1)
            return bool(np.max(np.abs(cur - q)) <= tol_q)
        frame = frame or self.ee_frame_name
        # 始终经 fkine 门面现算当前 TCP 位姿（fkine 未注册时报清晰 RuntimeError；
        # 不用 st.tcp.pose——未注册时其为恒等默认值，不可作比较基准）
        cur_pose = self.fkine(st.joint.q, frame)
        tgt = pose if isinstance(pose, Pose) else Pose()
        d_pos = float(np.linalg.norm(
            np.asarray(cur_pose.position, float) - np.asarray(tgt.position, float)))
        q_err = quat_mul(quat_conj(np.asarray(cur_pose.orientation, float)),
                         np.asarray(tgt.orientation, float))
        d_rot = float(quat_to_axis_angle(q_err)[1])
        return bool(d_pos <= tol_pos and d_rot <= tol_rot)

    def _wait_in_position(self, q, tol_q: float, timeout: float) -> bool:
        """轮询到位（50ms 周期）；超时前最后一次复查后返回结果。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_in_position(q=q, tol_q=tol_q):
                return True
            time.sleep(0.05)
        return self.is_in_position(q=q, tol_q=tol_q)
