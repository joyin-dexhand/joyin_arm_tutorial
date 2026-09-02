"""``JoyArm`` —— 完整机械臂类（arm本体 + end执行器），**组合根**，单类。

型号差异全部由配置表达：``configs/<model>.yaml`` 经 ``basic.robot`` 选 URDF 资产、
``basic.ee_frame`` 选末端帧、``backend:`` 段 ``name`` 经 backend REGISTRY 选整机
后端（型号与整机后端 1:1）、``joyarm:`` 段配特征位形与 TCP 限位、``robotics:`` 段
选各域求解器——**无型号子类**：新型号 = ``configs/<型号>.yaml`` + ``robot_model/`` 资产
+ ``backend/backend_*.py``（backend ``REGISTRY`` 一行）。

六域策略成员机制：``robotics:`` 段按注册名选型组装私有成员字典（``_fkine_solvers``
等，config 可指定单个或列表，首个为活动成员），公开门面（``fkine`` 等）委托活动
成员，运行期 ``set_solver`` 切换。各域 ``REGISTRY`` 默认仅有 ABC 接口（空表）——
具体算法为教程各章教学内容（fkine Ch2 / ikine Ch3 / jacobian Ch4 / traj Ch5 /
control Ch6 / dynamics Ch8），章节实现后注册即接入。

config 在构造时一次性加载存入 ``self._config``；教学数据（如 MDH 参数
``joyarm.arm_mdh_and_limits``）由教学算法经 ``arm.get_config()`` 从类内数据读取，
**不重新加载 yaml 文件**。

pinocchio ``pin_model``/``pin_data`` + 限位为构型加载产物（URDF 驱动）；离线
（``connected=False`` 默认）：计算类随时可用（已注册域）；执行类 ``connect()``
前抛 ``RuntimeError``。
"""
from __future__ import annotations

import copy
import logging
import os
from typing import List, Optional, Union

import numpy as np

from ..utils.limits import clamp_to_limits, joint_limits_from_model, soft_limits
from ..utils.types import (
    ArmState,
    ControlMode,
    TcpLimits,
)

try:
    import pinocchio as pin
except ImportError:
    pin = None

from ..backend import get_backend
from ..robotics.fkine import REGISTRY as _FKINE_REGISTRY
from ..robotics.ikine import REGISTRY as _IKINE_REGISTRY
from ..robotics.jacobian import REGISTRY as _JACOBIAN_REGISTRY
from ..robotics.dynamics import REGISTRY as _DYNAMICS_REGISTRY
from ..robotics.trajectory import REGISTRY as _TRAJ_REGISTRY
from ..robotics.control import REGISTRY as _CONTROL_REGISTRY

__all__ = ["JoyArm", "load_config"]

logger = logging.getLogger("joyarm_core.joyarm")

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
            raise ValueError(f"顶层应为映射，实际 {type(cfg).__name__}")
        return cfg
    except Exception as e:
        if not strict:
            return None
        try:
            avail = sorted(f[:-5] for f in os.listdir(_CONFIGS_DIR) if f.endswith(".yaml"))
        except OSError:
            avail = []
        raise ValueError(f"『{model}』型号在 configs 中未找到；可用：{avail}") from e


# config robotics 段 "default" 别名对应的各域默认注册名（教程各章实现注册后生效）
_DOMAIN_DEFAULTS = {
    "fkine": "pin",
    "ikine": "pin",
    "jacobian": "pin",
    "dynamics": "pin",
    "traj": "default",
    "control": "position",
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
    "basic.utils.joint_limits_soft_margin": "_set_soft_margin",
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
    :param mesh_dirs: mesh 搜索目录；``load_geometry``: 是否加载 visual/collision 几何。
    :param config: 型号 YAML 字典（``basic``/``joyarm``/``robotics``/``backend`` 段）；
        缺省自动加载 ``configs/<model>.yaml``，一次性存入 ``self._config``。
    """

    def __init__(self,
        model: str,
        urdf_path: Optional[str] = None,
        ee_frame_name: Optional[str] = None,
        mesh_dirs: Optional[List[str]] = None,
        load_geometry: bool = False,
        config: Optional[dict] = None,
    ):
        if pin is None:
            raise ImportError(
                "JoyArm 需要 pinocchio 才能加载 URDF 与计算运动学。请安装：\n"
                "  uv pip install pin\n"
                "  # 或：pip install pin"
            )

        # ---- 型号 config（工厂注入；直用时自动加载 configs/<model>.yaml）----
        if config is None:
            config = load_config(model)
            if config is None:
                logger.warning("configs/%s.yaml 未找到；以空 config 构造（离线纯 URDF 模式）",
                               model)
        cfg = config or {}
        self._config: dict = cfg

        # ---- urdf / ee_frame：参数 > config basic 段 > 内置默认 ----
        basic = cfg.get("basic") or {}
        if urdf_path is None:
            robot = basic.get("robot")
            if robot is None:
                raise ValueError(
                    "未指定 urdf_path，且 config basic.robot 缺失，无法解析 URDF。\n"
                    "请传入 urdf_path，或在 configs yaml 配置 basic.robot"
                    "（如 joyarm_dm_fixend）。"
                )
            urdf_path = self._resolve_robot_urdf(robot)
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(
                f"未找到 URDF 文件：{urdf_path}\n"
                f"请将正式 URDF 放入 joyarm_core/robot_model/。"
            )
        self._urdf_path: str = urdf_path
        ee_frame_name = ee_frame_name or basic.get("ee_frame") or "ee"

        # ---- 构建 pinocchio 模型（构型加载，URDF 驱动）----
        if mesh_dirs:
            self.pin_model = pin.buildModelFromUrdf(urdf_path, package_dirs=mesh_dirs)
        else:
            self.pin_model = pin.buildModelFromUrdf(urdf_path)

        self.collision_model: Optional[object] = None
        self.visual_model: Optional[object] = None
        if load_geometry:
            try:
                geo_dirs = mesh_dirs or [os.path.dirname(urdf_path)]
                self.collision_model = pin.buildGeomFromUrdf(
                    self.pin_model, urdf_path, pin.GeometryType.COLLISION, package_dirs=geo_dirs
                )
                self.visual_model = pin.buildGeomFromUrdf(
                    self.pin_model, urdf_path, pin.GeometryType.VISUAL, package_dirs=geo_dirs
                )
            except Exception as e:
                import warnings

                warnings.warn(f"几何模型加载失败：{e}")
        self.pin_data = self.pin_model.createData()

        # ---- 基本属性 ----
        self.n: int = self.pin_model.nq
        self.nv: int = self.pin_model.nv
        self.model: str = model
        self.connected: bool = False

        # ---- 末端帧 ----
        self.ee_frame_name: str = ee_frame_name
        # pinocchio getFrameId 对未知名不抛异常而是返回 nframes，据此判缺
        self.ee_frame_id: int = self.pin_model.getFrameId(ee_frame_name)
        if self.ee_frame_id >= len(self.pin_model.frames):
            raise ValueError(
                f"URDF 中找不到末端帧 '{ee_frame_name}'；"
                f"可用帧：{[f.name for f in self.pin_model.frames]}"
            )

        # ---- 关节限位（硬 + 软）----
        # margin 具体值由 yaml basic.utils.joint_limits_soft_margin 提供；未配置时取 0（软=硬）
        self.joint_limits = joint_limits_from_model(self.pin_model)
        utils_cfg = (cfg.get("basic") or {}).get("utils") or {}
        self.joint_limits_soft = soft_limits(
            self.joint_limits, margin=float(utils_cfg.get("joint_limits_soft_margin", 0.0))
        )
        self.qlow: np.ndarray = self.joint_limits_soft.q_min
        self.qhigh: np.ndarray = self.joint_limits_soft.q_max

        # ---- 特征位形（config joyarm 段优先，缺失回退 q_zero）----
        jcfg = cfg.get("joyarm") or {}
        self._q_zero: np.ndarray = np.clip(
            np.asarray(jcfg.get("q_zero", np.zeros(self.n)), dtype=float).reshape(-1),
            self.joint_limits.q_min,
            self.joint_limits.q_max,
        )
        self._q_home: np.ndarray = np.asarray(
            jcfg.get("q_home", self._q_zero), dtype=float
        ).reshape(-1)
        self._q_neutral: np.ndarray = np.asarray(
            jcfg.get("q_neutral", self._q_zero), dtype=float
        ).reshape(-1)

        # ---- TCP 空间限位：joyarm 段提供则覆盖默认占位 ----
        self.tcp_limits: TcpLimits = TcpLimits()
        if jcfg.get("tcp_limits"):
            self._apply_tcp_limits(jcfg["tcp_limits"])
        self.T_base: np.ndarray = np.eye(4)  # 基坐标系偏移

        # ---- 整机通信后端：config backend 段 name 选型（软失败：无效置空+警告）----
        self._backend = None
        backend_cfg = cfg.get("backend")
        if backend_cfg:
            backend_cfg = dict(backend_cfg)
            backend_name = backend_cfg.pop("name", None)
            if backend_name is None:
                logger.warning("config backend 段缺少选型键 name；后端置空（离线计算仍可用）")
            else:
                try:
                    self._backend = get_backend(backend_name)(backend_cfg)
                except Exception as e:
                    logger.error("后端『%s』构建失败：%s（后端置空，离线计算仍可用）",
                                 backend_name, e)
                    self._backend = None

        # ---- 六域策略成员字典（config 选型，全部加载、首个为活动；软失败）----
        # 各域 REGISTRY 默认空表（仅 ABC 接口）——域未配置即跳过，教学各章实现
        # 注册后经 config 选型接入
        self._fkine_solvers: dict = {}
        self._ikine_solvers: dict = {}
        self._jacobian_solvers: dict = {}
        self._dynamics_solvers: dict = {}
        self._traj_planners: dict = {}
        self._controllers: dict = {}
        self._active_name: dict = {}
        robotics_cfg = cfg.get("robotics") or {}
        for domain, registry in _DOMAIN_REGISTRIES.items():
            members = _build_domain(domain, registry, robotics_cfg.get(domain))
            self._members(domain).update(members)
            self._active_name[domain] = next(iter(members), None)

    # ----------------------------------------------------------
    # 成员字典访问 / 切换（六域统一）
    # ----------------------------------------------------------
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
                f"robotics.{domain} 成员未加载（config 未配置或注册名未实现）；"
                f"已加载：{sorted(d) or '无'}。教程各章实现算法并注册后，"
                f"经 configs yaml 的 robotics.{domain} 段选型接入"
            )
        return d[name]

    def _pick(self, domain: str, name: Optional[str]):
        """按注册名取已加载成员（``None`` = 活动成员）。"""
        if name is None:
            return self._active(domain)
        d = self._members(domain)
        if name not in d:
            raise ValueError(
                f"『{name}』未在 robotics.{domain} 已加载成员中；已加载：{sorted(d)}"
            )
        return d[name]

    def set_solver(self, domain: str, name: str):
        """运行期切换活动成员（按注册名；域 ∈ fkine/ikine/jacobian/dynamics/traj/control）。"""
        if domain not in _DOMAIN_REGISTRIES:
            raise ValueError(f"未知域 {domain!r}；可用：{sorted(_DOMAIN_REGISTRIES)}")
        inst = self._pick(domain, name)   # 校验已加载
        self._active_name[domain] = name
        return inst

    def set_controller(self, name: str):
        """运行期切换控制律（``set_solver("control", name)`` 的惯用别名）。"""
        return self.set_solver("control", name)

    def list_solvers(self, domain: str) -> list:
        """列出某域已加载成员注册名（首个为活动成员）。"""
        return sorted(self._members(domain))

    # ----------------------------------------------------------
    # 特征位形（只读，返回拷贝）
    # ----------------------------------------------------------
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

    @property
    def state(self) -> Optional[ArmState]:
        """最新状态快照：已连接现读一次 ``get_arm_state()``，离线 ``None``。"""
        if self.connected:
            return self.get_arm_state()
        return None

    # ----------------------------------------------------------
    # 真机连接（connect 后才可执行 arm_*/end_*）
    # ----------------------------------------------------------
    def connect(self) -> None:
        """连接真机（整机后端：本体 + 末端）。"""
        self._require_backend()
        self._backend.connect()
        self.connected = True

    def disconnect(self) -> None:
        """断开真机（整机后端：本体 + 末端）。"""
        if self._backend is not None:
            self._backend.disconnect()
        self.connected = False

    def _require_connected(self) -> None:
        """执行类方法前置：未连接真机（离线）时抛 ``RuntimeError``。"""
        if not self.connected:
            raise RuntimeError(f"[{self.model}] 未连接真机（离线）；请先 connect()。")

    def _require_backend(self) -> None:
        """后端前置：无后端（未配置/选型无效）时抛 ``RuntimeError``。"""
        if self._backend is None:
            raise RuntimeError(
                f"[{self.model}] 无整机后端（config backend 段缺失或选型无效），"
                f"无法执行硬件操作。"
            )

    # ----------------------------------------------------------
    # 配置：读取 / 运行期设置 / 自检（架构约束：配置功能全集）
    # ----------------------------------------------------------
    def get_config(self) -> dict:
        """当前配置深拷贝快照（四段 ``basic``/``joyarm``/``robotics``/``backend``）。

        教学算法读取型号教学数据（如 MDH 参数 ``joyarm.arm_mdh_and_limits``）统一
        经此接口从**类内已加载的 config**获取，不重新加载 yaml 文件。
        """
        return copy.deepcopy(self._config)

    def set_config(self, path: str, value) -> None:
        """运行期设置参数（白名单内即时生效；config 文件不回写，重启以 yaml 为准）。

        :param path: 点路径，如 ``"basic.utils.joint_limits_soft_margin"``。
        :raises ValueError: 路径不在白名单内（其余项请手改 configs yaml，保留注释）。
        """
        handler = _SETTABLE.get(path)
        if handler is None:
            raise ValueError(
                f"path={path!r} 不在运行期可设白名单内：{sorted(_SETTABLE)}"
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

    def _set_soft_margin(self, v) -> None:
        self.joint_limits_soft = soft_limits(self.joint_limits, float(v))
        self.qlow = self.joint_limits_soft.q_min
        self.qhigh = self.joint_limits_soft.q_max

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
                "配置自检未通过：\n" + "\n".join(f"  - {p}" for p in problems))

    def check_hardware(self) -> None:
        """硬件最小自检（需 ``connect()``）：逐关节通讯/电机故障/编码器有效性，
        末端电机同构检查（通讯/故障）。

        最小检查语义：通过则静默返回；发现硬故障则 ``RuntimeError`` 携带全部
        问题一次抛出。运行期安全监控（关节过温、碰撞等）归 ROS2 节点，不在
        核心库；指令越限由指令路径的限位裁剪（``clamp_to_limits``）兜底。

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
        except Exception:
            es = {}
        for i, ok in enumerate(es.get("comm_ok", [])):
            if not ok:
                problems.append(f"末端电机[{i}] 通讯无应答")
        for i, err in enumerate(es.get("error", [])):
            if err:
                problems.append(f"末端电机[{i}] 电机故障（故障标志置位）")
        if problems:
            raise RuntimeError(
                "硬件自检未通过：\n" + "\n".join(f"  - {p}" for p in problems))

    # ----------------------------------------------------------
    # 内部：随包 URDF 解析（robot 资产加载逻辑）
    # ----------------------------------------------------------
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
        raise ValueError(f"『{robot}』型号在 robot_model 中未找到；可用：{avail}")

    # ----------------------------------------------------------
    # 打印表示
    # ----------------------------------------------------------
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
    # 关节角采样 / 裁剪 / 校验（基于软限位）
    # ----------------------------------------------------------
    def rand_q(self,
        size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """在**软限位**内均匀采样关节角。"""
        rng = rng if rng is not None else np.random.default_rng()
        low, high = self.qlow, self.qhigh
        if size is None:
            return rng.uniform(low, high)
        return rng.uniform(low, high, size=(size, self.n))

    def clamp_q(self, q: np.ndarray) -> np.ndarray:
        """将关节角裁剪到**软限位**内（薄委托 :func:`joyarm_core.utils.limits.clamp_to_limits`）。"""
        return clamp_to_limits(q, self.joint_limits_soft)

    def is_q_valid(self, q: np.ndarray) -> bool:
        """关节角是否在**软限位**内（单点）。"""
        q = np.asarray(q, dtype=float).reshape(-1)
        return bool(np.all(q >= self.qlow - 1e-9) and np.all(q <= self.qhigh + 1e-9))

    # ----------------------------------------------------------
    # robotics 求解算法（门面 → 活动策略成员；参数排序：通用在前、特有 keyword-only 在后）
    # ----------------------------------------------------------
    def fkine(self, q: np.ndarray, frame: Union[str, int], rep: str = "pose"):
        """正运动学（``frame`` 目标帧名/索引，必填；``rep`` 取 ``pose``（默认，xyz+四元数）/``T``（4×4 矩阵）/``se3``（pin.SE3））。"""
        return self._active("fkine").solve(self, q, frame=frame, rep=rep)

    def ikine(self, T_target, frame=None, *, q0=None, **kw):
        """逆运动学（``T_target`` 通用；``q0``/``tol``/``iters`` 等为求解器特有参数）。"""
        return self._active("ikine").solve(self, T_target, frame=frame, q0=q0, **kw)

    def ikine_constrained(self, T_target, frame=None, *, q0=None, **kw):
        """带关节软限位约束的逆运动学（失败随机重启）。"""
        return self._active("ikine").solve_constrained(self, T_target, frame=frame, q0=q0, **kw)

    def jac(self, q: np.ndarray, frame: Optional[Union[str, int]] = None, ref: str = "local"):
        """雅可比 J(q)（``ref`` 取 ``local``/``base``：末端帧系 / 基座系）。"""
        return self._active("jacobian").jac(self, q, frame=frame, ref=ref)

    def manipulability(self, q: np.ndarray, frame: Optional[Union[str, int]] = None) -> float:
        """Yoshikawa 可操作度（雅可比衍生量）。"""
        return self._active("jacobian").manipulability(self, q, frame=frame)

    def cond_number(self, q: np.ndarray, frame: Optional[Union[str, int]] = None) -> float:
        """雅可比条件数（雅可比衍生量）。"""
        return self._active("jacobian").cond_number(self, q, frame=frame)

    def statics(self, q: np.ndarray, F: np.ndarray, frame: Optional[Union[str, int]] = None):
        """静力学 τ = JᵀF（雅可比衍生量）。"""
        return self._active("jacobian").statics(self, q, F, frame=frame)

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

    def cartesian_inertia(self, q, frame=None):
        """笛卡尔惯量 Λ=J⁻ᵀMJ⁻¹（M ⊕ ``arm.jac`` 模板）。"""
        return self._active("dynamics").cartesian_inertia(self, q, frame=frame)

    # ----------------------------------------------------------
    # 紧急阻尼（纯后端安全操作，任意状态可用）
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # 本体执行类方法（依赖 _backend；未连接 raise；arm_* 与 end_* 对应）
    # ----------------------------------------------------------
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

        :raises RuntimeError: 未连接真机时抛出。
        :raises ValueError: 对应模式所需参数缺失时抛出。
        """

        def _vec(x):
            if x is None:
                return None
            return np.atleast_1d(np.asarray(x, dtype=float))

        self._require_connected()
        if mode == ControlMode.POSITION:
            if q is None:
                raise ValueError("POSITION 模式需要 q")
            self._backend.send_position_arm(_vec(q), joint)
        elif mode == ControlMode.VELOCITY:
            if dq is None:
                raise ValueError("VELOCITY 模式需要 dq")
            self._backend.send_velocity_arm(_vec(dq), joint)
        elif mode == ControlMode.MIT:
            missing = [
                name for name, val in (("q", q), ("dq", dq), ("tau", tau)) if val is None
            ]
            if missing:
                raise ValueError(f"MIT 模式缺少参数：{missing}")
            self._backend.send_mit_arm(_vec(q), _vec(dq), _vec(tau),
                                       kp=_vec(kp) if kp is not None else None,
                                       kd=_vec(kd) if kd is not None else None,
                                       joint=joint)
        else:
            raise ValueError(f"未知控制模式：{mode}")

    # ----------------------------------------------------------
    # 电机参数读写（依赖 _backend；未连接 raise）
    # ----------------------------------------------------------
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
    # 末端执行类方法（依赖 _backend；未连接 raise）
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # 内部：TCP 空间限位应用（config joyarm 段）
    # ----------------------------------------------------------
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
