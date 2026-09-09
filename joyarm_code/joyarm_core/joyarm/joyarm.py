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

from ..utils.limits import clamp_to_limits, limits_from_joint_cfgs, rand_within_limits, soft_limits_from_cfg
from ..utils.transforms import quat_conj, quat_mul, quat_to_axis_angle
from ..utils.types import ArmState, ControlMode, JointLimits, Pose, TcpLimits, TrajFrame
from ..utils.interpolation import cubic_traj as _cubic_traj

from ..backend import Backend, get_backend
from ..robotics.fkine import REGISTRY as _FKINE_REGISTRY
from ..robotics.ikine import REGISTRY as _IKINE_REGISTRY
from ..robotics.jacobian import REGISTRY as _JACOBIAN_REGISTRY
from ..robotics.dynamics import REGISTRY as _DYNAMICS_REGISTRY
from ..robotics.trajectory import REGISTRY as _TRAJ_REGISTRY
from ..robotics.control import REGISTRY as _CONTROL_REGISTRY

__all__ = ["JoyArm", "load_config"]

logger = logging.getLogger("joyarm_core.joyarm")


# ============================================================
# 模块级：路径常量与配置加载（load_config 为公开 API）
# 工厂生产流程：① 传入型号 → ② 加载对应 config 文件
# （configs/<型号>.yaml）→ ③ 按 config 的 robot_model（URDF）、
# robotics（求解器）、backend（通信）等配置参数初始化并创建
# JoyArm 对象（初始化后默认不连接 backend）
# ============================================================
# configs/ 目录（joyarm.py 位于 joyarm_core/joyarm/，上溯一级即包根）
_CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
)

# robot_model/ 资产目录（URDF + meshes，运行期加载）
_ROBOT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot_model"
)

# arm 关节名约定：URDF 中仅 name 为 joint1~joint9 的关节被识别为本体关节，其余不参与关节数校验
_ARM_JOINT_NAMES = {f"joint{i}" for i in range(1, 10)}


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
# 模块级：六域规格常量与域构建（硬失败语义：配置的成员必须全部创建成功）
# ============================================================
# 域 → 注册表（六域统一字典化；config 规格可为单值或列表，全部加载、首个激活）
# 注册名约定：与求解器类名对应（类名小写 + 下划线，如 FkinePin → "fkine_pin"）
_DOMAIN_REGISTRIES = {
    "fkine": _FKINE_REGISTRY,
    "ikine": _IKINE_REGISTRY,
    "jacobian": _JACOBIAN_REGISTRY,
    "dynamics": _DYNAMICS_REGISTRY,
    "traj": _TRAJ_REGISTRY,
    "control": _CONTROL_REGISTRY,
}

def _build_domain(domain: str, registry: dict, spec) -> dict:
    """按 config 规格实例化一域策略成员（硬失败语义）。

    :param spec: 注册名字符串 / ``{name:..., **参数}`` / 上述的**列表**（全部加载）。
    :return: 成员字典 ``{注册名: 实例}``——域未配置返回空；**任一成员规格非法、
        注册名不存在或实例化失败即抛 ``ValueError``**（全部创建成功 + 首个激活
        才通过）。
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
                raise ValueError(
                    f"joyarm.py - _build_domain：robotics.{domain} 规格缺 name 键：{s!r}")
        else:
            raise ValueError(
                f"joyarm.py - _build_domain：robotics.{domain} 规格类型非法"
                f"（{type(s).__name__}），需为注册名字符串或 {{name:..., **参数}} 字典")
        if name not in registry:
            raise ValueError(
                f"joyarm.py - _build_domain：注册名『{name}』在 robotics.{domain} 中"
                f"未找到；可用：{sorted(registry) or '无（尚未实现注册）'}")
        try:
            members[name] = registry[name](**params)
        except Exception as e:
            raise ValueError(
                f"joyarm.py - _build_domain：robotics.{domain}『{name}』实例化失败：{e}"
            ) from e
    return members


class JoyArm:
    """完整机械臂类（组合根，兼容带末端执行器的 6R/7R 臂，单类）。

    :param model: 型号名（product model，区别于构型模型 ``pin_model``；与
        ``configs/<model>.yaml`` 文件名、yaml ``basic.name`` 字段一致）。
    :param config: 型号 YAML 字典（``basic``/``joyarm``/``robotics``/``backend``
        四段——内部指定 robot_model（URDF 资产）、robotics 求解器、backend
        通信等配置）；缺省自动加载 ``configs/<model>.yaml``。**config 必需**：
        无法加载或 :meth:`check_config` 自检不通过即构造失败。
    """

    # ----------------------------------------------------------
    # init 初始化（成员变量全量定义 → 逐步赋真实值；config 驱动构造）
    # ----------------------------------------------------------
    def __init__(self, model: str, config: Optional[dict] = None):
        # ---- config 解析（工厂注入；直用时自动加载 configs/<model>.yaml；必需）----
        if config is None:
            config = load_config(model, strict=True)   # 缺失/解析失败 → ValueError
        # 深拷贝隔离外部引用：调用方后续改动不影响类内快照（get_config 另返回深拷贝）
        cfg = copy.deepcopy(config)
        # 初始化前先自检 config，自检通过才赋值并进行初始化
        self.check_config(model, cfg)

        basic = cfg.get("basic") or {}
        jcfg = cfg.get("joyarm") or {}
        backend_cfg = cfg.get("backend") or {}
        arm_joint_cfgs = (backend_cfg.get("arm") or {}).get("joints") or []
        end_joint_cfgs = (backend_cfg.get("end") or {}).get("joints") or []

        # ----------------------------------------------------------
        # joyarm 成员变量（全量定义：可直接赋值的直接赋值，其余先赋默认值）
        # ----------------------------------------------------------
        # ---- 基本属性 ----
        self.model: str = model                          # 型号名（product model）
        self._config: dict = cfg                         # config 四段快照（深拷贝隔离）
        self._urdf_path: str = ""                        # URDF 文件路径
        self.connected: bool = False                     # 真机连接状态（初始化后默认不连接）
        self.ee_frame_name: str = str(basic.get("ee_frame") or "ee")  # 末端帧名
        # ---- pinocchio 构型（URDF 驱动）----
        self.pin_model: Optional[pin.Model] = None       # URDF 解析构型模型
        self.pin_data: Optional[pin.Data] = None         # 模型配套计算数据
        self.ee_frame_id: int = -1                       # 末端帧索引（pinocchio frames 表）
        # ---- 本体（arm）----
        self.n_arm: int = 0                              # 本体关节数（config arm.joints 数）
        self.arm_limits: Optional[JointLimits] = None    # 本体硬限位（config arm.joints 四键；初始化后固定）
        self.arm_limits_soft: Optional[JointLimits] = None   # 本体软限位（config joyarm.arm_soft_limits 直配，上层状态判断用）
        self._arm_joint_names: List[str] = []            # 本体关节名（config backend.arm.joints 顺序）
        self._arm_zero: np.ndarray = np.zeros(0)         # 本体零位（按自由度全零）
        self._arm_home: np.ndarray = np.zeros(0)         # 本体上电初始位形（config joyarm.arm_home）
        self._arm_neutral: np.ndarray = np.zeros(0)      # 本体数值求解默认初值（按自由度全零）
        # ---- 末端（end）----
        self.n_end: int = 0                              # 末端电机数（config end.joints 数；无末端 0）
        self.end_limits: Optional[JointLimits] = None    # 末端硬限位（config end.joints 四键，电机行程）
        self.end_limits_soft: Optional[JointLimits] = None   # 末端软限位（config joyarm.end_soft_limits 直配，上层状态判断用）
        self._end_joint_names: List[str] = []            # 末端电机名（config backend.end.joints 顺序）
        self._end_zero: np.ndarray = np.zeros(0)         # 末端零位（按自由度全零）
        self._end_home: np.ndarray = np.zeros(0)         # 末端初始位形（config joyarm.end_home，电机空间）
        self._end_neutral: np.ndarray = np.zeros(0)      # 末端数值求解默认初值（按自由度全零）
        # ---- TCP 空间限位 ----
        self.tcp_limits: TcpLimits = TcpLimits()         # TCP 空间限位（默认占位）
        # ---- 六域策略成员（config 选型，全部加载、首个激活）----
        self._fkine_solvers: dict = {}                   # fkine 成员字典 {注册名: 实例}
        self._ikine_solvers: dict = {}                   # ikine 成员字典
        self._jacobian_solvers: dict = {}                # jacobian 成员字典
        self._dynamics_solvers: dict = {}                # dynamics 成员字典
        self._traj_planners: dict = {}                   # traj 成员字典
        self._controllers: dict = {}                     # control 成员字典
        self._active_name: dict = {}                     # 各域激活成员注册名（域 → 名，有且仅一个）
        # ---- 整机后端（config backend 段选型，必须构建成功）----
        self._backend: Optional[Backend] = None          # 整机通信后端
        # ---- 轨迹桥（发布即不可变；写入口深拷贝隔离）----
        self._target_traj: Optional[List[TrajFrame]] = None    # 目标序列（应用任务写）
        self._current_frame: Optional[TrajFrame] = None        # 当前帧（规划线程写）

        # ----------------------------------------------------------
        # init 逐步赋真实值（robot_model URDF → 关节数/名称 → 限位（硬/软）→
        # 特征位形 → TCP 限位 → backend → 六域字典）
        # ----------------------------------------------------------
        # ---- robot_model：构建 pinocchio 构型（basic.robot → 随包 URDF 资产）----
        self._urdf_path = self._resolve_robot_urdf(str(basic["robot"]))
        self.pin_model = pin.buildModelFromUrdf(self._urdf_path)
        self.pin_data = self.pin_model.createData()

        # ---- 末端帧（pinocchio getFrameId 对未知名不抛异常而是返回 nframes，据此判缺）----
        self.ee_frame_id = self.pin_model.getFrameId(self.ee_frame_name)
        if self.ee_frame_id >= len(self.pin_model.frames):
            raise ValueError(
                f"joyarm.py - JoyArm.__init__：URDF 中找不到末端帧 '{self.ee_frame_name}'；"
                f"可用帧：{[f.name for f in self.pin_model.frames]}"
            )

        # ---- 关节数与关节名----
        self._arm_joint_names = [
            str(j.get("name", f"joint{i + 1}"))
            for i, j in enumerate(arm_joint_cfgs)
        ]
        self._end_joint_names = [
            str(j.get("name", f"end_joint{i + 1}"))
            for i, j in enumerate(end_joint_cfgs)
        ]
        self.n_arm = len(self._arm_joint_names)
        self.n_end = len(self._end_joint_names)
        arm_nq_urdf = self._urdf_arm_nq(self.pin_model)
        if self.n_arm != arm_nq_urdf:
            raise ValueError(
                f"joyarm.py - JoyArm.__init__：backend.arm.joints 数量 {self.n_arm} "
                f"≠ URDF arm 关节数 {arm_nq_urdf}（仅支持 arm 的关节数比对校验（config 与 urdf），"
                f"name 为 joint1~joint9 的 arm 的 joint 才会被识别；end 关节不参与校验）")

        # ---- 硬限位（config backend.*.joints 四键；初始化后即固定）----
        # arm 四键数值须与 URDF limit 标定保持一致（维护约定）；backend 下发指令只裁硬限位
        self.arm_limits = limits_from_joint_cfgs(arm_joint_cfgs)
        self.end_limits = limits_from_joint_cfgs(end_joint_cfgs)
        self._check_arm_limits_vs_urdf(self._urdf_path, arm_joint_cfgs)

        # ---- 软限位（config joyarm.arm_soft_limits / end_soft_limits 四键直值，
        # arm/end 分开配置；未配置时软=硬）。仅加载备用：上层"超软限位→状态异常
        # →急停恢复"后续实现，不参与指令裁剪（下发只裁硬限位）----
        self.arm_limits_soft = (
            soft_limits_from_cfg(jcfg["arm_soft_limits"], self.n_arm)
            if jcfg.get("arm_soft_limits") is not None
            else copy.deepcopy(self.arm_limits))
        self.end_limits_soft = (
            soft_limits_from_cfg(jcfg["end_soft_limits"], self.n_end)
            if (self.end_limits is not None and jcfg.get("end_soft_limits") is not None)
            else copy.deepcopy(self.end_limits))
        # 软限位须位于对应硬限位内（越界为配置错误，硬失败）
        for tag, soft, hard in (("arm", self.arm_limits_soft, self.arm_limits),
                                ("end", self.end_limits_soft, self.end_limits)):
            if soft is None or hard is None:
                continue
            beyond = ((np.asarray(soft.q_min) < np.asarray(hard.q_min) - 1e-9).any()
                      or (np.asarray(soft.q_max) > np.asarray(hard.q_max) + 1e-9).any()
                      or (np.asarray(soft.dq_max) > np.asarray(hard.dq_max) + 1e-9).any()
                      or (np.asarray(soft.tau_max) > np.asarray(hard.tau_max) + 1e-9).any())
            if beyond:
                raise ValueError(
                    f"joyarm.py - JoyArm.__init__：joyarm.{tag}_soft_limits 超出对应"
                    f"硬限位（软限位须位于硬限位内，供上层状态判断）")

        # ---- 特征位形（zero/neutral 按自由度全零；home 自 config，越限裁剪并告警）----

        def _clip_pose(arr, limits, key):
            arr = np.asarray(arr, dtype=float).reshape(-1)
            if limits is None:
                return arr
            clipped = np.clip(arr, limits.q_min, limits.q_max)
            if not np.array_equal(arr, clipped):
                logger.warning("config joyarm.%s 越限位，已裁剪 %s → %s（以裁剪后为准）",
                               key, np.round(arr, 4).tolist(), np.round(clipped, 4).tolist())
            return clipped

        self._arm_zero = np.zeros(self.n_arm)
        self._arm_neutral = np.zeros(self.n_arm)
        self._arm_home = _clip_pose(
            jcfg.get("arm_home") if jcfg.get("arm_home") is not None else np.zeros(self.n_arm),
            self.arm_limits, "arm_home")
        self._end_zero = np.zeros(self.n_end)
        self._end_neutral = np.zeros(self.n_end)
        self._end_home = _clip_pose(
            jcfg.get("end_home") if jcfg.get("end_home") is not None else np.zeros(self.n_end),
            self.end_limits, "end_home")

        # ---- TCP 空间限位（joyarm 段提供则覆盖默认占位）----
        if jcfg.get("tcp_limits"):
            self._apply_tcp_limits(jcfg["tcp_limits"])

        # ---- 整机通信后端（config backend 段 name 选型；必须成功，失败即构造失败）----
        # 硬限位由 backend 自 cfg 四键解析（arm/end 同构，与 JoyArm 侧同源同值）；
        # backend 守卫只裁硬限位（限位裁剪唯一执行点，独立使用后端同样生效）
        backend_name = str(backend_cfg["name"])          # check_config 已确保存在
        bcfg_rest = dict(backend_cfg)
        bcfg_rest.pop("name", None)
        self._backend = get_backend(backend_name)(bcfg_rest)

        # ---- 六域策略成员字典（config 选型；任一成员创建失败即构造失败；首个激活）----
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

    @staticmethod
    def _urdf_arm_nq(pin_model) -> int:
        """统计 pin 模型中 arm 关节（name 为 joint1~joint9）的自由度之和。

        索引 0 为 universe 跳过；mimic 及 end/手指等其余关节名不在约定集合内、
        不计入（关节数校验只比对 arm，end 不校验）。

        :param pin_model: pinocchio 模型（``buildModelFromUrdf`` 产物）。
        :return: arm 关节自由度之和。
        """
        nq = 0
        for i in range(1, len(pin_model.names)):
            if pin_model.names[i] in _ARM_JOINT_NAMES:
                nq += pin_model.joints[i].nq
        return nq

    # ---- init 内部：URDF 关节限位解析 + config 一致性自检（仅告警）----
    @staticmethod
    def _urdf_joint_limits(urdf_path: str) -> dict:
        """解析 URDF 活动关节的限位与 mimic 标记（一致性自检的 URDF 侧数据源）。

        仅收录有限位的活动关节（revolute / continuous / prismatic；fixed 等
        无限位关节不参与检查）；``<limit>`` 属性映射为四键——``lower``→``q_min``、
        ``upper``→``q_max``、``velocity``→``dq_max``、``effort``→``tau_max``
        （缺属性为 ``None``，如 continuous 关节无位置上下限属正常）。

        :param urdf_path: URDF 文件路径。
        :return: ``{关节名: {q_min, q_max, dq_max, tau_max, mimic}}``，
            前四值为 float 或 ``None``，``mimic`` 为是否含 ``<mimic>`` 标签。
        """
        import xml.etree.ElementTree as ET

        def _num(lim, key: str):
            v = None if lim is None else lim.get(key)
            return None if v is None else float(v)

        joints: dict = {}
        for j in ET.parse(urdf_path).getroot().iterfind("joint"):
            if j.get("type") not in ("revolute", "continuous", "prismatic"):
                continue
            lim = j.find("limit")
            joints[j.get("name")] = {
                "q_min": _num(lim, "lower"), "q_max": _num(lim, "upper"),
                "dq_max": _num(lim, "velocity"), "tau_max": _num(lim, "effort"),
                "mimic": j.find("mimic") is not None,
            }
        return joints

    @staticmethod
    def _check_arm_limits_vs_urdf(urdf_path: str, arm_joint_cfgs: list) -> None:
        """config ``backend.arm.joints`` 四键 ↔ URDF ``limit`` 一致性自检（仅告警）。

        维护约定：arm 硬限位以 config 四键为唯一生效源（守卫/规划均消费），
        URDF ``limit`` 为标定基准——不一致仅 ``logger.warning`` 提示核对，
        不阻断初始化（数值以 config 为准）。三类检查：

        ① config 关节在 URDF 中不存在（限位无法与标定核对）；
        ② 同名关节四键数值与 URDF 不一致（容差 1e-6，逐键比对）；
        ③ URDF 存在未被 config 定义的 arm 活动关节（name 为 joint1~joint9 的
        非 mimic 关节应与 config 一一对应，mimic 从动关节随主动关节定义，不
        单独配置；end/手指等其余关节不校验、不告警）。

        末端为电机空间行程（URDF 通常不含末端关节），不参与比对。

        :param urdf_path: URDF 文件路径（``basic.robot`` 解析产物）。
        :param arm_joint_cfgs: config ``backend.arm.joints`` 条目列表。
        """
        urdf = JoyArm._urdf_joint_limits(urdf_path)
        for j in arm_joint_cfgs or []:
            name = str(j.get("name", ""))
            u = urdf.get(name)
            if u is None:
                logger.warning(
                    "URDF 限位自检：config 关节 %r 在 URDF 中不存在（限位无法与 URDF 标定核对）",
                    name)
                continue
            for key in ("q_min", "q_max", "dq_max", "tau_max"):
                cfg_v, urdf_v = j.get(key), u[key]
                if cfg_v is None or urdf_v is None:
                    continue     # config 缺键由 check_config 把关；URDF 缺属性不比对
                if abs(float(cfg_v) - urdf_v) > 1e-6:
                    logger.warning(
                        "URDF 限位自检：关节 %r 的 %s 不一致——config=%s，URDF=%s"
                        "（数值以 config 为准，请核对标定）", name, key, cfg_v, urdf_v)
        defined = {str(j.get("name", "")) for j in arm_joint_cfgs or []}
        for name, u in urdf.items():
            if name not in _ARM_JOINT_NAMES:
                continue    # 非 arm 关节（end/手指等）不校验
            if name not in defined and not u["mimic"]:
                logger.warning(
                    "URDF 限位自检：URDF 活动关节 %r 未在 config backend.arm.joints 中定义",
                    name)

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
    # ---- 特征位形（arm/end 各三个；只读，返回拷贝）----
    @property
    def arm_zero(self) -> np.ndarray:
        """本体零位 ``(n_arm,)``（按自由度全零，编码器标零基准）。"""
        return self._arm_zero.copy()

    @property
    def arm_home(self) -> np.ndarray:
        """本体上电初始位形 ``(n_arm,)``（config ``joyarm.arm_home``）。"""
        return self._arm_home.copy()

    @property
    def arm_neutral(self) -> np.ndarray:
        """本体数值求解默认初值 ``(n_arm,)``（全零，如 IK 迭代起点）。"""
        return self._arm_neutral.copy()

    @property
    def end_zero(self) -> np.ndarray:
        """末端零位 ``(n_end,)``（按自由度全零；无末端为空数组）。"""
        return self._end_zero.copy()

    @property
    def end_home(self) -> np.ndarray:
        """末端初始位形 ``(n_end,)``（config ``joyarm.end_home``，电机空间；缺省全零）。"""
        return self._end_home.copy()

    @property
    def end_neutral(self) -> np.ndarray:
        """末端数值求解默认初值 ``(n_end,)``（全零；无末端为空数组）。"""
        return self._end_neutral.copy()

    # ---- 关节名称（config 顺序；按名寻址）----
    @property
    def joint_names_arm(self) -> List[str]:
        """本体关节名列表（config ``backend.arm.joints`` 顺序）。"""
        return list(self._arm_joint_names)

    @property
    def joint_names_end(self) -> List[str]:
        """末端电机名列表（config ``backend.end.joints`` 顺序；无末端为空）。"""
        return list(self._end_joint_names)

    def joint_index_arm(self, name: str) -> int:
        """本体关节名 → 索引（教学/日志按名寻址）。

        :raises ValueError: 名称不在本体关节名列表中（消息列出可用名）。
        """
        try:
            return self._arm_joint_names.index(str(name))
        except ValueError:
            raise ValueError(
                f"joyarm.py - joint_index_arm：关节名 {name!r} 未找到；"
                f"可用：{self._arm_joint_names}") from None

    def joint_index_end(self, name: str) -> int:
        """末端电机名 → 索引。

        :raises ValueError: 名称不在末端电机名列表中（消息列出可用名）。
        """
        try:
            return self._end_joint_names.index(str(name))
        except ValueError:
            raise ValueError(
                f"joyarm.py - joint_index_end：电机名 {name!r} 未找到；"
                f"可用：{self._end_joint_names}") from None

    # ---- 关节角采样（arm 硬限位内；采样实现在 utils.rand_within_limits）----
    def rand_q_arm(self,
        size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """在**本体硬限位**内均匀采样关节角（``(n_arm,)``；``size`` 给 ``(size, n_arm)``）。"""
        return rand_within_limits(self.arm_limits, size=size, rng=rng)

    # ---- 打印表示 ----
    def __repr__(self) -> str:
        domains = {d: (self._active_name.get(d) or "-") for d in _DOMAIN_REGISTRIES}
        return (
            f"{type(self).__name__}(model={self.model!r}, n_arm={self.n_arm}, "
            f"n_end={self.n_end}, ee_frame={self.ee_frame_name!r}, "
            f"solvers={domains}, "
            f"{'connected' if self.connected else 'offline'})"
        )

    # ----------------------------------------------------------
    # config 配置管理（读取 / 自检；架构约束：配置功能全集）
    # ----------------------------------------------------------
    # ---- 配置读取 ----
    def get_config(self) -> dict:
        """当前配置深拷贝快照（四段 ``basic``/``joyarm``/``robotics``/``backend``）。

        外部算法读取型号数据（如 MDH 参数 ``joyarm.arm_mdh_and_limits``）统一
        经此接口从**类内已加载的 config**获取，不重新加载 yaml 文件。
        """
        return copy.deepcopy(self._config)

    # ---- 自检（check_config 离线静态 / check_hardware 临时连接硬件）----
    @staticmethod
    def check_config(model: str, config: dict) -> None:
        """config 静态自检（**初始化前**调用；自检通过才赋值并进行初始化）。

        检查项：basic 段齐全 / 命名链 / URDF 资产存在 / backend 必配且 arm
        关节四键限位齐全 / 特征位形（``arm_home``/``end_home``）长度与关节数
        一致 / robotics 段结构合法（注册名是否存在由 ``_build_domain`` 硬失败
        兜底）。

        :raises ValueError: 配置存在问题时抛出，消息列出全部问题。
        """
        problems: List[str] = []
        cfg = config or {}
        if "basic" not in cfg:
            problems.append("缺少配置段：basic")
        basic = cfg.get("basic") or {}
        if basic.get("name") not in (None, model):
            problems.append(f"basic.name={basic.get('name')!r} 与 model={model!r} 不一致")
        if not basic.get("robot"):
            problems.append("basic.robot 缺失（robot_model URDF 索引，无法解析 URDF）")
        else:
            try:
                JoyArm._resolve_robot_urdf(str(basic["robot"]))
            except ValueError as e:
                problems.append(str(e))
        if not basic.get("ee_frame"):
            problems.append("basic.ee_frame 缺失（末端帧名）")
        if "backend" not in cfg:
            problems.append("缺少配置段：backend（必配：不配置则无法指定通信与执行）")
        bcfg = cfg.get("backend") or {}
        if not bcfg.get("name"):
            problems.append("backend.name 缺失（整机后端选型键）")
        for i, j in enumerate((bcfg.get("arm") or {}).get("joints") or []):
            if not isinstance(j, dict) or not j.get("name"):
                problems.append(f"backend.arm.joints[{i}] 缺少 name 键")
            else:
                missing = [k for k in ("q_min", "q_max", "dq_max", "tau_max")
                           if j.get(k) is None]
                if missing:
                    problems.append(
                        f"backend.arm.joints[{i}]（{j.get('name')}）缺少限位键 {missing}"
                        f"（四键数值须与 URDF limit 标定保持一致）")
        for part in ("arm", "end"):
            joints = (bcfg.get(part) or {}).get("joints") or []
            if part == "arm" and not joints:
                problems.append("backend.arm.joints 为空（本体电机列表必配）")
            for i, j in enumerate(joints):
                if not isinstance(j, dict) or not j.get("name"):
                    if part == "end":
                        problems.append(f"backend.end.joints[{i}] 缺少 name 键")
        n_arm_cfg = len((bcfg.get("arm") or {}).get("joints") or [])
        n_end_cfg = len((bcfg.get("end") or {}).get("joints") or [])
        jcfg = cfg.get("joyarm") or {}
        for key, n_ref in (("arm_home", n_arm_cfg), ("end_home", n_end_cfg)):
            v = jcfg.get(key)
            if v is None:
                continue
            size = np.asarray(v, dtype=float).size
            if size != n_ref:
                problems.append(f"joyarm.{key} 长度 {size} ≠ 关节数 {n_ref}")
        for key, n_ref in (("arm_soft_limits", n_arm_cfg), ("end_soft_limits", n_end_cfg)):
            soft = jcfg.get(key)
            if soft is None:
                continue
            if not isinstance(soft, dict):
                problems.append(f"joyarm.{key} 需为四键字典（q_min/q_max/dq_max/tau_max）")
                continue
            for k, v in soft.items():
                if k in ("q_min", "q_max", "dq_max", "tau_max"):
                    if np.asarray(v, dtype=float).ndim > 0 and \
                            np.asarray(v, dtype=float).reshape(-1).shape[0] != n_ref:
                        problems.append(
                            f"joyarm.{key}.{k} 长度 "
                            f"{np.asarray(v, dtype=float).reshape(-1).shape[0]} ≠ 关节数 {n_ref}")
        robotics_cfg = cfg.get("robotics") or {}
        for domain in _DOMAIN_REGISTRIES:
            spec = robotics_cfg.get(domain)
            if spec is None:
                continue
            specs = spec if isinstance(spec, (list, tuple)) else [spec]
            for s in specs:
                if isinstance(s, str):
                    continue
                if isinstance(s, dict):
                    if not s.get("name"):
                        problems.append(f"robotics.{domain} 规格缺 name 键：{s!r}")
                else:
                    problems.append(
                        f"robotics.{domain} 规格类型非法（{type(s).__name__}）")
        if problems:
            raise ValueError(
                "joyarm.py - check_config：配置自检未通过：\n"
                + "\n".join(f"  - {p}" for p in problems))

    def check_hardware(self) -> None:
        """硬件自检（使用时**手动**调用；初始化后不自检硬件）。

        自检时**先临时连接硬件**，在电机保持失能（disable）状态下完成全部
        自检，完成后再 ``disconnect`` 断开；若本已连接则沿用现有连接、检完
        不断开。检查项：本体逐关节通讯/电机故障/编码器有效性，末端电机
        同构检查（通讯/故障）。

        通过则静默返回；发现硬故障则 ``RuntimeError`` 携带全部问题一次抛出。
        运行期安全监控（关节过温、碰撞等）归 ROS2 节点，不在核心库；指令
        越限由后端基类限位守卫（``Backend.send_*`` 模板）兜底。

        :raises RuntimeError: 连接失败或存在硬故障时抛出（消息列出全部问题）。
        """
        own_conn = not self.connected
        if own_conn:
            self.connect()                       # 临时连接（不使能电机）
        try:
            problems: List[str] = []
            st = self.get_arm_state()
            js = st.joint
            for i in range(self.n_arm):
                if not bool(np.asarray(js.comm_ok)[i]):
                    problems.append(f"本体关节[{i}] 通讯无应答")
                if bool(np.asarray(js.error)[i]):
                    problems.append(f"本体关节[{i}] 电机故障（故障标志置位）")
                if not bool(np.asarray(js.angle_ok)[i]):
                    problems.append(f"本体关节[{i}] 编码器角度无效")
            try:
                es = self.get_end_state()
            except Exception as e:
                es = {}
                problems.append(f"末端状态读取失败：{e}")
            for i, ok in enumerate(es.get("comm_ok", [])):
                if not ok:
                    problems.append(f"末端电机[{i}] 通讯无应答")
            for i, err in enumerate(es.get("error", [])):
                if err:
                    problems.append(f"末端电机[{i}] 电机故障（故障标志置位）")
        finally:
            if own_conn:
                self.disconnect()
        if problems:
            raise RuntimeError(
                "joyarm.py - check_hardware：硬件自检未通过：\n"
                + "\n".join(f"  - {p}" for p in problems))

    # ----------------------------------------------------------
    # robotics 六域策略（成员字典管理 + 六域求解门面 → 激活策略成员）
    # ----------------------------------------------------------
    # ---- 成员字典访问 / 切换（六域统一；有且仅有一个激活）----
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
        """激活成员；域未配置时抛 ``RuntimeError``。"""
        d = self._members(domain)
        name = self._active_name.get(domain)
        if name is None or name not in d:
            raise RuntimeError(
                f"joyarm.py - _active：robotics.{domain} 成员未加载"
                f"（config 未配置或注册名未实现）；已加载：{sorted(d) or '无'}。"
            )
        return d[name]

    def _pick(self, domain: str, name: Optional[str]):
        """按注册名取已加载成员（``None`` = 激活成员）。"""
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
        """运行期切换激活成员（按注册名；域 ∈ fkine/ikine/jacobian/dynamics/traj/control）。

        每域**有且仅有一个**激活成员：config 加载后首个为激活，此后经本方法切换。
        """
        if domain not in _DOMAIN_REGISTRIES:
            raise ValueError(
                f"joyarm.py - set_solver：未知域 {domain!r}；可用：{sorted(_DOMAIN_REGISTRIES)}")
        inst = self._pick(domain, name)   # 校验已加载
        self._active_name[domain] = name
        return inst

    def list_solvers(self, domain: str) -> list:
        """列出某域已加载成员注册名（首个为激活成员）。"""
        return sorted(self._members(domain))

    # ---- fkine 正运动学（参数排序：通用在前、特有 keyword-only 在后）----
    def fkine(self, q: np.ndarray, frame: Union[str, int], rep: str = "pose"):
        """正运动学（``frame`` 目标帧名/索引，必填；``rep`` 取 ``pose``（默认，xyz+四元数）/``T``（4×4 矩阵）/``se3``（pin.SE3））。"""
        return self._active("fkine").solve(self, q, frame=frame, rep=rep)

    # ---- ikine 逆运动学 ----
    def ikine(self, target: Pose, frame: Union[str, int], q0: np.ndarray, **kw):
        """逆运动学单解（``q0`` ``(n_arm,)`` 必填：数值法迭代起点 / 解析法限位
        剔除后选最近解的参考；``tol``/``iters`` 等为求解器特有参数）。"""
        return self._active("ikine").solve(self, target, frame, q0, **kw)

    def ikine_all(self, target: Pose, frame: Union[str, int], **kw):
        """逆运动学全部解析解（``q`` 为 ``(K, n_arm)``，经 ±2π 平移尽量落入限位；
        数值法实现不支持）。"""
        return self._active("ikine").solve_all(self, target, frame, **kw)

    # ---- jacobian 雅可比及衍生量 ----
    def jac(self, q: np.ndarray, frame: Union[str, int], ref: str = "base"):
        """雅可比 J(q)（``ref`` 取 ``local``/``base``：末端帧系 / 基座系）。"""
        return self._active("jacobian").jac(self, q, frame=frame, ref=ref)

    def fkine_vel(self, q: np.ndarray, dq: np.ndarray,
                  frame: Union[str, int], ref: str = "base"):
        """速度正解 ``V = J(q)·q̇``（微分运动学正解）。

        :param q: 关节角 ``(n_arm,)``；``dq`` 关节速度 ``(n_arm,)``（rad/s）。
        :return: ``(6,)`` 末端速度旋量（线速度 m/s + 角速度 rad/s，参考系同 ``ref``）。
        """
        return self._active("jacobian").fkine_vel(self, q, dq, frame, ref=ref)

    def ikine_vel(self, q: np.ndarray, V: np.ndarray,
                  frame: Union[str, int], ref: str = "base",
                  damping: float = 1e-3) -> np.ndarray:
        """速度逆解（微分逆解）``q̇ = J*·V``（阻尼最小二乘：冗余臂最小范数解，
        奇异附近阻尼正则化）。

        :param q: 关节角 ``(n_arm,)``；``V`` 末端速度旋量 ``(6,)``（参考系同 ``ref``）。
        :param damping: DLS 阻尼 λ（默认 ``1e-3``）。
        :return: ``(n_arm,)`` 关节速度（rad/s）。
        """
        return self._active("jacobian").ikine_vel(
            self, q, V, frame, ref=ref, damping=damping)

    def manipulability(self, q: np.ndarray, frame: Union[str, int]) -> float:
        """Yoshikawa 可操作度（雅可比衍生量）。"""
        return self._active("jacobian").manipulability(self, q, frame=frame)

    def statics(self, q: np.ndarray, F: np.ndarray, frame: Union[str, int]):
        """静力学 ``τ = JᵀF``：``F`` 为 ``(6,)`` 末端六维力旋量（力 N + 力矩
        N·m），返回 ``τ ∈ R^n_arm`` 关节力矩（雅可比衍生量）。"""
        return self._active("jacobian").statics(self, q, F, frame=frame)

    # ---- dynamics 动力学 ----
    def idyn(self, q, dq, ddq, f_ext=None):
        """逆动力学（委托激活动力学成员）。"""
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
    # traj 轨迹桥（规划线程 ⇄ 控制线程的数据管道）
    # ----------------------------------------------------------
    # ---- 目标序列（写：应用线程低频 / 读：规划线程）----
    def set_target_traj(self, targets) -> None:
        """写入目标序列（写者：应用线程，低频；**深拷贝隔离**，调用方后续
        修改不影响桥内数据）。

        :param targets: :class:`TrajFrame` 单帧或列表（单帧自动归一为列表）。
            目标有效性判别在规划求解时由 ``TrajPlanner._check_targets`` 执行。
        """
        frames = [targets] if isinstance(targets, TrajFrame) else list(targets)
        self._target_traj = copy.deepcopy(frames)

    def get_target_traj(self) -> Optional[List[TrajFrame]]:
        """读取目标序列（读者：规划线程；返回当前快照引用，规划期勿由他方改动）。"""
        return self._target_traj

    # ---- 当前帧（写：规划线程 ≈控制频率 / 读：控制线程）----
    def set_current_frame(self, frame: TrajFrame) -> None:
        """写入当前轨迹帧（写者：规划线程，≈控制频率；**深拷贝隔离**）。

        单步原子引用赋值——控制线程任意时刻读到的都是最新**完整**帧（最多
        滞后一个换帧周期）；帧对象发布后视为不可变，勿再修改。
        """
        self._current_frame = copy.deepcopy(frame)

    def get_current_frame(self) -> Optional[TrajFrame]:
        """读取当前轨迹帧（读者：控制线程；首帧发布前为 ``None`` 初始态）。"""
        return self._current_frame

    # ----------------------------------------------------------
    # backend 真机连接与执行（依赖 _backend；connect 后才可执行 arm_*/end_*）
    # ----------------------------------------------------------
    # ---- 连接管理与前置校验（connected / enabled / mode 三级前置）----
    def connect(self) -> None:
        """连接真机（整机后端：本体 + 末端；连接**不使能**电机）。"""
        self._backend.connect()
        self.connected = bool(self._backend.connected)

    def disconnect(self) -> None:
        """断开真机（整机后端先失能全部电机再断开总线）。"""
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

    def _require_enabled_arm(self) -> None:
        """前置：本体关节电机未全部使能时抛 ``RuntimeError``（在线查询使能位）。

        适用于运动/指令下发前（如 ``move_j``、安全起停、``hold_position``）；
        急停类安全操作（``damping_mode``/``lock_position``）不做此前置。
        """
        enabled = np.asarray(self.get_arm_state().joint.enabled, dtype=bool).reshape(-1)
        if not bool(enabled.all()):
            bad = np.where(~enabled)[0].tolist()
            raise RuntimeError(
                f"joyarm.py - _require_enabled_arm：本体关节 {bad} 未使能；请先 enable_arm()")

    def _require_enabled_end(self) -> None:
        """前置：末端电机未全部使能时抛 ``RuntimeError``；无末端（``n_end=0``）跳过。"""
        if self.n_end == 0:
            return
        enabled = list(self.get_end_state().get("enabled", []))
        if not all(enabled):
            bad = [i for i, ok in enumerate(enabled) if not ok]
            raise RuntimeError(
                f"joyarm.py - _require_enabled_end：末端电机 {bad} 未使能；请先 enable_end()")

    def _require_mode_arm(self, mode: ControlMode, joint: Optional[int] = None) -> None:
        """前置：本体当前控制模式 ≠ ``mode`` 时抛 ``RuntimeError``（模式缓存，离线可查）。

        适用于下发控制指令前（``set_arm_command`` 及运动安全层）——收指令前
        必须已处于对应控制模式；``None`` 表示未设置或各关节模式不一致。
        """
        cur = self.read_mode_arm(joint)
        if cur != mode:
            raise RuntimeError(
                f"joyarm.py - _require_mode_arm：本体当前控制模式为 {cur}"
                f"（None=未设置/各关节不一致），与指令模式 {mode} 不符；"
                f"请先 set_mode_arm({mode})")

    # ---- 紧急阻尼（纯后端安全操作，任意控制状态可用）----
    def damping_mode(self, kd: float = 10.0) -> None:
        """紧急阻尼模式：**任何状态**下将全部电机（本体 + 末端）切为 MIT 纯阻尼
        （``kp=q=dq=tau=0, kd``）——电机仅产生 ∝ 速度的黏滞阻力，防发疯/防坠落。

        依次执行：本体/末端切 MIT → 下发阻尼帧 → 尽力使能（故障电机不阻断其余）
        → 再发一帧（使能即生效）。各阶段 **best-effort**：单阶段失败仅记警告不
        抛出，保证尽量多的电机收到指令。

        :param kd: 阻尼系数（N·m·s/rad），默认 10.0。
        :raises RuntimeError: 未连接真机（无法触达电机）。
        """
        self._require_connected()
        self._damping_frames(kd)

    def _damping_frames(self, kd: float = 10.0) -> None:
        """阻尼帧序列（无线程操作）：切 MIT → 阻尼帧 → 尽力使能 → 再发一帧。"""
        z = np.zeros(self.n_arm)

        def _send_arm(tag: str) -> None:
            try:
                self._backend.send_mit_arm(z, z, z, kp=z, kd=np.full(self.n_arm, kd))
            except Exception as e:
                logger.warning("damping_mode：%s阻尼指令失败：%s", tag, e)

        def _send_end(tag: str) -> None:
            try:
                if self.n_end:
                    ez = np.zeros(self.n_end)
                    self._backend.send_mit_end(ez, ez, ez, kp=ez,
                                               kd=np.full(self.n_end, kd))
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
        """查询本体关节当前控制模式（backend 本地缓存，离线可查；``joint=None`` 时整臂
        各关节模式唯一才返回该模式，否则 ``None``）。"""
        return self._backend.read_mode_arm(joint)

    def get_arm_state(self) -> ArmState:
        """读取本体状态快照（委托 ``_backend.read_state_arm()``；与 ``get_end_state`` 对应）。

        fkine 域已配置激活成员时同步填充 ``tcp.pose``（末端位姿）；未配置时
        跳过填充，仅返回关节原始状态。

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

        限位守卫在后端基类 ``Backend.send_*_arm`` 模板内（硬限位裁剪唯一执行点）：
        ``q`` 裁剪到硬限位、``dq``/``tau`` 裁剪到幅值上限，越限告警（节流每
        0.5s 至多一条）就近裁剪（``kp``/``kd`` 为标定增益，不裁剪）；软限位
        不参与指令裁剪（JoyArm 直配加载，供上层状态判断）。

        :raises RuntimeError: 未连接真机 / 当前控制模式与 ``mode`` 不符
            （需先 :meth:`set_mode_arm`）。
        :raises ValueError: 对应模式所需参数缺失。
        """
        self._require_connected()
        self._require_mode_arm(mode, joint)           # 收指令前必须已切到对应模式
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
        """查询末端电机当前控制模式（backend 本地缓存，离线可查；``joint=None`` 时整个末端
        各电机模式唯一才返回该模式，否则 ``None``）。"""
        return self._backend.read_mode_end(joint)

    def set_end_open(self, joint: Optional[int] = None) -> None:
        """张开末端到最大（默认行程/力度；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_action_end("open", joint)

    def set_end_close(self, joint: Optional[int] = None) -> None:
        """闭合末端（夹到默认力度即停；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_action_end("close", joint)

    def set_end_zero(self, joint: Optional[int] = None) -> None:
        """末端归零（目标 = 电机弧度 0，裁剪到行程内；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_action_end("zero", joint)

    def set_end_position(self, position, joint: Optional[int] = None) -> None:
        """末端位置控制（连续量，如夹爪电机弧度；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_position_end(position, joint)

    def set_end_tau(self, tau, joint: Optional[int] = None) -> None:
        """末端力矩控制（直接前馈电机力矩 N·m；``joint=None`` 全部末端电机）。"""
        self._require_connected()
        self._backend.send_tau_end(tau, joint)

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
    # motion 运动便利与安全层（hold/lock/move_j；move_j 为独立功能，
    # 与「轨迹桥 → 规划器 → 控制器」常规管线并行，仅直接调用使用）
    # ----------------------------------------------------------
    # ---- 原位保持与急停锁定 ----
    def hold_position(self, kp=None, kd=None, tau=None) -> None:
        """原位保持（阻抗锁定当前姿态，含 tau 前馈补偿）。

        读当前关节角 → 切 MIT 模式 → ``q=当前, dq=0, tau_ff`` 阻抗保持。
        ``tau`` 缺省自动取重力前馈：dynamics 域已配置激活成员时
        ``gravity(q_cur)``，未配置置零并告警（退化保持）；可用 ``tau=`` 显式
        覆盖。``kp``/``kd`` 缺省 ``None`` 透传后端回退 config ``MIT`` 增益。

        与 :meth:`damping_mode` 区分：阻尼为零目标纯黏滞（防坠落），本方法
        为锁定**当前**姿态的位置阻抗 + 重力补偿。
        """
        self._require_connected()
        self._require_enabled_arm()
        q = np.asarray(self.get_arm_state().joint.q, dtype=float).reshape(-1)
        if tau is None:
            try:
                tau = np.asarray(self.gravity(q), dtype=float).reshape(-1)
            except RuntimeError:
                tau = np.zeros(self.n_arm)
                logger.warning("hold_position：dynamics 域未配置，tau 前馈置零（退化保持）")
        self.set_mode_arm(ControlMode.MIT)
        self.set_arm_command(ControlMode.MIT, q=q, dq=np.zeros(self.n_arm), tau=tau,
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
        """关节空间点到点阻塞运动（三次多项式插值，位置模式指令流）——
        **独立功能，与规划器并行**。

        边界：本方法直接借 ``utils.interpolation`` 插值原语 + 位置
        指令流下发，**不经**「轨迹桥 → 规划器 → 控制器」管线，仅供直接调用。
        **常规运动**一律走 ``set_target_traj → 规划器 → 控制器 → set_arm_command``
        管线；安全回位（``safe_*``）走 MIT 阻抗模式（见 :meth:`safe_home`）。

        调度为精确定时器（绝对时间表 + sleep/自旋混合等待，帧间隔亚毫秒
        抖动）。注意：峰值关节速度 = ``1.5·max|q−q_cur|/t``，受 config
        ``POS_VEL.vlim`` 约束，超出时实际时长 > ``t``（由末段到位等待兜底）；
        真机使用前请先安全化 config（vlim/kp/kd 封顶）。

        :param q: 目标关节角 ``(n_arm,)``，弧度（入口裁剪到硬限位——规划与
            到位判定均以裁剪后目标为准，后端守卫再裁为幂等空操作）。
        :param t: 总时长（秒）；缺省 ``max|q−q_cur|``（隐含峰值 1.5 rad/s）；
            ``t ≤ 0`` 直发目标。
        :param rate: 发送率（Hz）；缺省 config ``backend.arm.control_rate``
            （回退 100），钳制 ≤1000（DM 控制帧间隔 ≥1ms）。
        :param wait_tol: 末段到位容差（rad，经 :meth:`is_in_position`）。
        :param wait_timeout: 到位等待超时（秒），超时 ``RuntimeError``。
        :raises ValueError: 目标维度与 ``n_arm`` 不符。
        :raises RuntimeError: 未连接 / 关节未使能 / 到位超时。
        """
        self._require_connected()
        self._require_enabled_arm()
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.n_arm:
            raise ValueError(
                f"joyarm.py - move_j：目标维度 {q.shape[0]} 与关节数 {self.n_arm} 不符")
        # 入口裁硬限位：规划、下发、到位判定统一用裁剪后目标，消除
        # 「后端裁剪到位 ≠ 判定目标」的失配超时（后端守卫再裁为幂等空操作）
        q_c = clamp_to_limits(q, self.arm_limits)
        if not np.array_equal(q, q_c):
            logger.warning("move_j：目标关节角越硬限位，已裁剪 %s → %s"
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
        """笛卡尔直线阻塞运动（**占位**）：需 ikine 与笛卡尔规划实现后落地。

        常规运动管线：``set_target_traj → 规划器 → 控制器 → set_arm_command``。
        """
        raise NotImplementedError(
            "joyarm.py - move_l：笛卡尔直线运动需 ikine + 笛卡尔规划，尚未实现；"
            "常规运动请走 轨迹桥 → 规划器 → 控制器 管线")

    def teach_mode(self, on: bool = True) -> None:
        """拖动示教模式（**占位**）：需 dynamics 重力补偿实现后落地。

        目标语义：MIT 模式 + ``gravity(q_cur)`` 前馈 + 零刚度（``kp=0``）——
        臂仅余重力补偿，可徒手拖动（``on=False`` 退出恢复）；示教录制/回放
        另行实现。与 :meth:`hold_position` 区分：后者锁定当前姿态（位置阻抗），
        本方法零刚度自由拖动。
        """
        raise NotImplementedError(
            "joyarm.py - teach_mode：拖动示教需 dynamics 重力补偿实现"
            "（MIT 模式 + gravity 前馈 + 零刚度 kp=0）")

    # ---- 安全起停（arm + end 均执行；关节空间运动，本体 MIT 阻抗模式）----
    def safe_home(self, t=None, *, wait_tol: float = 0.05,
                  wait_timeout: float = 10.0) -> None:
        """安全回 home（arm + end）：关节空间运动，本体经 MIT 阻抗模式三次
        多项式插值（q/dq 跟踪，kp/kd 回退 config MIT 增益），末端位置模式
        运动到 ``end_home``；``t`` 缺省 ``max|arm_home − q_cur|``。
        """
        self._require_connected()
        self._require_enabled_arm()
        self._safe_move(self.arm_home, self.end_home, t,
                        wait_tol=wait_tol, wait_timeout=wait_timeout)

    def home_to_zero(self, t=None, *, wait_tol: float = 0.05,
                     wait_timeout: float = 10.0) -> None:
        """home → zero（arm + end）：**先检查当前位于 home**（arm 于 ``arm_home``
        且 end 于 ``end_home``，硬限位投影后容差 ``wait_tol``），再运动到零位
        （MIT 阻抗，同 :meth:`safe_home`）。

        :raises RuntimeError: 当前不在 home 位形（请先 :meth:`safe_home` /
            :meth:`safe_zero`）。
        """
        self._require_connected()
        arm_ok = self.is_in_position(
            q=clamp_to_limits(self.arm_home, self.arm_limits), tol_q=wait_tol)
        end_ok = self._is_end_in_position(
            clamp_to_limits(self.end_home, self.end_limits), tol_q=wait_tol) \
            if self.n_end else True
        if not (arm_ok and end_ok):
            raise RuntimeError(
                f"joyarm.py - home_to_zero：当前不在 home 位形（容差 {wait_tol} rad）；"
                f"请先 safe_home() 或 safe_zero()")
        self._safe_move(self.arm_zero, self.end_zero, t,
                        wait_tol=wait_tol, wait_timeout=wait_timeout)

    def safe_zero(self, t=None, *, wait_tol: float = 0.05,
                  wait_timeout: float = 10.0) -> None:
        """安全回零（组合，arm + end）：先 :meth:`safe_home` 回 home，
        home 后再 :meth:`home_to_zero` 回零；任一段失败即抛出中断。
        """
        self._require_connected()
        self.safe_home(t, wait_tol=wait_tol, wait_timeout=wait_timeout)
        self.home_to_zero(t, wait_tol=wait_tol, wait_timeout=wait_timeout)

    def _safe_move(self, q_arm, q_end, t=None, *, rate=None,
                   wait_tol: float = 0.05, wait_timeout: float = 10.0) -> None:
        """安全运动内核：本体 MIT 阻抗模式三次多项式流（q/dq 跟踪 + config
        MIT 增益 + gravity 前馈可用则用），末端位置模式运动；末段到位等待。

        末端不适用 MIT 阻抗（config 末端 MIT 增益标定为 0），故经位置模式
        运动到目标（电机空间）。
        """
        # 入口裁硬限位（与 move_j 判定基准一致；软限位归上层状态判断）
        q_arm = clamp_to_limits(np.asarray(q_arm, dtype=float).reshape(-1),
                                self.arm_limits)
        q0 = np.asarray(self.get_arm_state().joint.q, dtype=float).reshape(-1)
        if rate is None:
            rate = float(((self._config.get("backend") or {}).get("arm") or {})
                         .get("control_rate", 100.0))
        rate = min(float(rate), 1000.0)
        if t is None:
            t = float(np.max(np.abs(q_arm - q0)))
        ts, qs, dqs = _cubic_traj(q0, q_arm, float(t), rate)
        # 末端：位置模式单目标（电机空间）
        if self.n_end:
            q_end = clamp_to_limits(np.asarray(q_end, dtype=float).reshape(-1),
                                    self.end_limits)
            self._backend.set_mode_end(ControlMode.POSITION)
            self._backend.send_position_end(q_end)
        # 本体：MIT 阻抗流（kp/kd 透传 None → 后端回退 config MIT 增益；
        # tau 取起点重力前馈常量，dynamics 未配置时置零）
        try:
            tau_ff = np.asarray(self.gravity(q0), dtype=float).reshape(-1)
        except RuntimeError:
            tau_ff = np.zeros(self.n_arm)
        self.set_mode_arm(ControlMode.MIT)
        start = time.perf_counter()
        spin_margin = 0.001                               # 先 sleep 到 deadline−margin 再自旋
        for ti, qi, dqi in zip(ts, qs, dqs):
            deadline = start + float(ti)
            now = time.perf_counter()
            if deadline - now > spin_margin:
                time.sleep(deadline - now - spin_margin)
            while time.perf_counter() < deadline:
                pass
            self._backend.send_mit_arm(qi, dqi, tau_ff)
        if not self._wait_in_position(q_arm, wait_tol, wait_timeout):
            raise RuntimeError(
                f"joyarm.py - _safe_move：到位超时（{wait_timeout}s 内未达容差 "
                f"{wait_tol} rad；请增大 t 或检查 config 增益/限速）")

    # ---- 到位判断 ----
    def is_in_position(self, q=None, pose=None, frame=None,
                       tol_q: float = 0.05, tol_pos: float = 1e-3,
                       tol_rot: float = 1e-2) -> bool:
        """到位判断（单入口双判断）：关节目标 ``q`` 或笛卡尔目标 ``pose`` 恰一。

        :param q: 关节目标 ``(n_arm,)``——逐关节 ``‖Δq‖∞ ≤ tol_q``（rad）。
        :param pose: 笛卡尔目标 :class:`Pose`——位置 ``‖Δp‖ ≤ tol_pos``（m）
            且姿态四元数误差角 ≤ ``tol_rot``（rad）。
        :param frame: pose 分支参考帧名；缺省 ``ee_frame_name``。
        :raises ValueError: ``q``/``pose`` 双空或双给、维度不符。
        :raises RuntimeError: 未连接；pose 分支 fkine 域未配置（无 TCP 位姿）。
        """
        if (q is None) == (pose is None):
            raise ValueError("joyarm.py - is_in_position：q 与 pose 必须恰给其一")
        st = self.get_arm_state()                        # 未连接在此抛 RuntimeError
        if q is not None:
            q = np.asarray(q, dtype=float).reshape(-1)
            if q.shape[0] != self.n_arm:
                raise ValueError(
                    f"joyarm.py - is_in_position：目标维度 {q.shape[0]} "
                    f"与关节数 {self.n_arm} 不符")
            cur = np.asarray(st.joint.q, dtype=float).reshape(-1)
            return bool(np.max(np.abs(cur - q)) <= tol_q)
        frame = frame or self.ee_frame_name
        # 始终经 fkine 门面现算当前 TCP 位姿（fkine 未配置时报清晰 RuntimeError；
        # 不用 st.tcp.pose——未配置时其为恒等默认值，不可作比较基准）
        cur_pose = self.fkine(st.joint.q, frame)
        tgt = pose if isinstance(pose, Pose) else Pose()
        d_pos = float(np.linalg.norm(
            np.asarray(cur_pose.position, float) - np.asarray(tgt.position, float)))
        q_err = quat_mul(quat_conj(np.asarray(cur_pose.orientation, float)),
                         np.asarray(tgt.orientation, float))
        d_rot = float(quat_to_axis_angle(q_err)[1])
        return bool(d_pos <= tol_pos and d_rot <= tol_rot)

    def _is_end_in_position(self, q_end, tol_q: float = 0.05) -> bool:
        """末端到位判断（逐电机 ``‖Δq‖∞ ≤ tol_q``；无末端恒 ``True``）。"""
        if self.n_end == 0:
            return True
        cur = np.asarray(self.get_end_state().get("q", []), dtype=float).reshape(-1)
        if cur.size == 0:
            return False
        q_end = np.asarray(q_end, dtype=float).reshape(-1)
        return bool(np.max(np.abs(cur - q_end)) <= tol_q)

    def _wait_in_position(self, q, tol_q: float, timeout: float) -> bool:
        """轮询到位（50ms 周期）；超时前最后一次复查后返回结果。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_in_position(q=q, tol_q=tol_q):
                return True
            time.sleep(0.05)
        return self.is_in_position(q=q, tol_q=tol_q)
