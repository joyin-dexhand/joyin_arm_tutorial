"""``JoyArmDM`` —— JoyArm 6 自由度机械臂型号预设（完整臂 = 本体 + 夹爪，Ch2 即用）。

固化 joyarm_dm 硬件平台（达妙 DM 电机，CAN）默认值：URDF 路径、末端帧、MDH 参考
表、home 位形；绑定 ``backend_arm_cls = BackendArmDM`` / ``backend_end_cls =
BackendEndGripper``，构造时按 ``configs/joyarm_dm.yaml`` 的对应段实例化。可调参数
（末端帧/home/MDH/末端限位/后端参数）config 优先、缺失回退类常量（无 YAML 也能
实例化）；``MDH_TABLE`` 仅教学参考，FK 走 pinocchio，数值以第七章 URDF 导出为准。
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .joyarm import JoyArm
from ..utils.types import TcpLimits
from ..backends.backend_arm_dm import BackendArmDM
from ..backends.backend_end_gripper import BackendEndGripper

__all__ = ["JoyArmDM"]


class JoyArmDM(JoyArm):
    """JoyArm（joyarm_dm）6 自由度机械臂型号预设（完整臂 = 本体 + 夹爪）。

    ``urdf_path`` / ``ee_frame_name`` 缺省时**优先取 ``configs/joyarm_dm.yaml``**，
    配置缺失再回退到类常量与随包默认 URDF。

    .. note::

        当前随包 URDF 为过渡型号 ``joyarm_dm_fixend``（纯 6 转动臂，
        ``nq=6``）；第七章导出正式版 ``joyarm1.urdf`` 后替换 ``DEFAULT_URDF``
        与对应 config。

    :param name: 名称。
    :param urdf_path: URDF 文件路径；缺省解析随包资源（config > 类常量）。
    :param ee_frame_name: 末端帧名；缺省用 config / 类常量 ``EE_FRAME``。
    :param mesh_dirs: mesh 搜索目录列表。
    :param load_geometry: 是否加载 visual/collision 几何；缺省 ``False``。
    """

    # ---- 类常量（joyarm_dm 预设；config 缺失时的兜底）----
    CONFIG_NAME: str = "joyarm_dm"       # 对应 configs/<CONFIG_NAME>.yaml
    DEFAULT_URDF: str = "robots/joyarm_dm_fixend/urdf/joyarm_dm_fixend.urdf"
    EE_FRAME: str = "end_link"                 # 末端帧在 URDF 里的名字

    # ---- 两个后端类：子类在类体绑定（构造时按 config 实例化）----
    backend_arm_cls = BackendArmDM        # 多轴本体后端
    backend_end_cls = BackendEndGripper     # 末端执行器后端

    # MDH 参数：(α_{i-1}, a_{i-1}, d_i, θ_offset_i)，6 行——教学参考常量
    MDH_TABLE: np.ndarray = np.zeros((6, 4))
    # link6 → end_link 固定位姿（URDF end_joint，fixed）：MDH 白盒递推后的固定尾巴
    T_LINK6_END: np.ndarray = np.eye(4)

    # home 位形（实际 home，未必等于中性位形）；第七章标定后核对
    Q_HOME: np.ndarray = np.zeros(6)

    def __init__(
        self,
        name: str = "JoyArmDM",
        urdf_path: Optional[str] = None,
        ee_frame_name: Optional[str] = None,
        mesh_dirs: Optional[List[str]] = None,
        load_geometry: bool = False,
    ):
        # ---- 加载型号 YAML 配置（可选：缺 PyYAML/文件时返回 None，回退类常量）----
        self.config: Optional[dict] = self._load_config(self.CONFIG_NAME)
        cfg = self.config or {}

        # ---- urdf / ee_frame：参数 > config > 类常量 ----
        if urdf_path is None:
            urdf_path = self._resolve_default_urdf(cfg.get("urdf"))
        ee = ee_frame_name or cfg.get("ee_frame") or self.EE_FRAME
        # 传 config 给 JoyArm：由基类按 backend_arm/backend_end 段实例化两个后端
        super().__init__(
            name=name,
            urdf_path=urdf_path,
            ee_frame_name=ee,
            mesh_dirs=mesh_dirs,
            load_geometry=load_geometry,
            config=cfg,
        )

        # ---- config 驱动的属性（回退到类常量）----
        self._q_home: np.ndarray = np.asarray(
            cfg.get("q_home", self.Q_HOME), dtype=float
        ).reshape(-1)
        if cfg.get("mdh") is not None:
            self.MDH_TABLE = np.asarray(cfg["mdh"], dtype=float).reshape(6, 4)
        if cfg.get("T_link6_end") is not None:
            self.T_LINK6_END = np.asarray(cfg["T_link6_end"], dtype=float).reshape(4, 4)

        # ---- 末端限位：config 提供则覆盖 JoyArm 的占位 TcpLimits ----
        if cfg.get("tcp_limits"):
            self._apply_tcp_limits(cfg["tcp_limits"])

    # ----------------------------------------------------------
    # 教学参考
    # ----------------------------------------------------------
    @property
    def mdh_table(self) -> np.ndarray:
        """返回 MDH 参数 ``(6,4)`` ndarray（教学查阅用）。"""
        return self.MDH_TABLE.copy()

    @property
    def T_link6_end(self) -> np.ndarray:
        """返回 link6 → end_link 固定位姿 ``(4,4)`` ndarray（教学查阅用）。"""
        return self.T_LINK6_END.copy()

    @property
    def q_home(self) -> np.ndarray:
        """实际 home 位形 ``(6,)``（优先取自 config）。"""
        return self._q_home.copy()

    # ----------------------------------------------------------
    # 内部：配置加载
    # ----------------------------------------------------------
    @staticmethod
    def _load_config(name: str) -> Optional[dict]:
        """按型号名加载 ``configs/<name>.yaml``。

        :return: 配置字典；未装 PyYAML 或文件缺失时返回 ``None``（回退类常量）。
        """
        try:
            import yaml  # PyYAML（可选依赖：缺失时回退类常量，不阻断实例化）
        except ImportError:
            return None
        try:
            from importlib.resources import files

            ref = files("joyarm_core").joinpath(f"configs/{name}.yaml")
            path = str(ref)
            if not os.path.isfile(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    def _apply_tcp_limits(self, tl: dict) -> None:
        """由 config 的 tcp_limits 段覆盖默认末端限位。"""
        self.tcp_limits = TcpLimits(
            workspace_box=np.asarray(
                tl.get("workspace_box", [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.8]]),
                dtype=float,
            ),
            v_lin_max=float(tl.get("v_lin_max", 0.0)),
            v_ang_max=float(tl.get("v_ang_max", 0.0)),
            f_max=float(tl.get("f_max", 0.0)),
            t_max=float(tl.get("t_max", 0.0)),
        )

    # ----------------------------------------------------------
    # 内部：解析随包 URDF
    # ----------------------------------------------------------
    @classmethod
    def _resolve_default_urdf(cls, cfg_urdf: Optional[str] = None) -> str:
        """用 importlib.resources 解析随包 URDF 路径（优先 config 的 urdf 段）。"""
        rel = cfg_urdf or cls.DEFAULT_URDF
        try:
            from importlib.resources import files

            # 以顶层 joyarm_core/ 包为根（robots/ 在 joyarm_core/robots/）
            ref = files("joyarm_core").joinpath(rel)
            path = str(ref)
            if os.path.isfile(path):
                return path
        except Exception:
            pass

        # 回退：相对顶层 joyarm_core/ 包目录解析（本文件在 joyarm_core/arms/，需上溯一级）
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(here, rel)
