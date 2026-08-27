"""``JoyArmDM`` —— JoyArm 6 自由度机械臂型号预设（完整臂 = 本体 + 夹爪，Ch2 即用）。

固化 joyarm_dm 硬件平台默认值：随包 URDF（``robot`` 键）、末端帧、MDH 参考表；
整机后端不绑类——由基类按 ``configs/joyarm_dm.yaml`` 的 ``backend:`` 段
``name`` 选型构建（``backend_dm`` 与本型号 1:1 对应）。可调参数（末端帧/MDH/
TCP 空间限位）config 优先、缺失回退类常量（无 YAML 也能实例化）；``MDH_TABLE``/
``MDH_LIMITS`` 仅供 MDH 白盒/手写运动学链路（可选数据源，不覆盖 URDF/pin 限位），
FK 默认走 pinocchio（URDF 链路），数值以第七章 URDF 导出为准。
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .joyarm import JoyArm, load_config
from ..utils.types import TcpLimits

__all__ = ["JoyArmDM"]


class JoyArmDM(JoyArm):
    """JoyArm（joyarm_dm）6 自由度机械臂型号预设（完整臂 = 本体 + 夹爪）。

    ``urdf_path`` / ``ee_frame_name`` 缺省时**优先取 ``configs/joyarm_dm.yaml``**，
    配置缺失再回退到类常量与随包默认 URDF；``config`` 由 ``JoyArmFactory`` 注入
    （已加载并校验命名链），直用时自动加载。

    .. note::

        当前随包 URDF 为过渡型号 ``joyarm_dm_fixend``（纯 6 转动臂，
        ``nq=6``）；第七章导出正式版 ``joyarm1.urdf`` 后替换 ``ROBOT``
        与对应 config。

    :param name: 名称。
    :param urdf_path: URDF 文件路径；缺省解析随包资源（config ``robot`` 键 > 类常量）。
    :param ee_frame_name: 末端帧名；缺省用 config / 类常量 ``EE_FRAME``。
    :param mesh_dirs / load_geometry: 同 :class:`JoyArm`。
    :param config: 已加载的型号 YAML 字典（工厂注入）；缺省自动加载（容错回退类常量）。
    """

    # ---- 类常量（joyarm_dm 预设；config 缺失时的兜底）----
    CONFIG_NAME: str = "joyarm_dm"        # 对应 configs/<CONFIG_NAME>.yaml
    ROBOT: str = "joyarm_dm_fixend"       # 对应 robots/<ROBOT>/urdf/<ROBOT>.urdf
    EE_FRAME: str = "end_link"            # 末端帧在 URDF 里的名字

    # MDH 链路（可选数据源，不覆盖 URDF/pin 限位）：
    #   MDH_TABLE：(α_{i-1}, a_{i-1}, d_i, θ_offset_i)，6 行——教学参考常量
    #   MDH_LIMITS：(q_min, q_max, dq_max, tau_max, soft_margin)，6 行 × 5 列
    MDH_TABLE: np.ndarray = np.zeros((6, 4))
    MDH_LIMITS: np.ndarray = np.zeros((6, 5))
    # 末link系 → end系固定位姿（URDF end_joint，fixed）：MDH 白盒递推后的固定尾巴
    T_LINKN_END: np.ndarray = np.eye(4)

    def __init__(
        self,
        name: str = "JoyArmDM",
        urdf_path: Optional[str] = None,
        ee_frame_name: Optional[str] = None,
        mesh_dirs: Optional[List[str]] = None,
        load_geometry: bool = False,
        config: Optional[dict] = None,
    ):
        # ---- 型号 YAML（工厂注入；直用时容错加载，缺失回退类常量）----
        self.config: Optional[dict] = (
            config if config is not None else load_config(self.CONFIG_NAME)
        )
        cfg = self.config or {}

        # ---- urdf / ee_frame：参数 > config > 类常量 ----
        if urdf_path is None:
            urdf_path = self._resolve_default_urdf(cfg.get("robot"))
        ee = ee_frame_name or cfg.get("ee_frame") or self.EE_FRAME
        # 传 config 给 JoyArm：由基类按 backend 段 name 选型构建整机后端
        super().__init__(
            name=name,
            urdf_path=urdf_path,
            ee_frame_name=ee,
            mesh_dirs=mesh_dirs,
            load_geometry=load_geometry,
            config=cfg,
        )

        # ---- config 驱动的属性（回退到类常量）----
        # MDH 链路（可选）：arm_mdh_and_limits 6×9 → 前 4 列 MDH + 后 5 列限位
        if cfg.get("arm_mdh_and_limits") is not None:
            arr = np.asarray(cfg["arm_mdh_and_limits"], dtype=float).reshape(6, -1)
            self.MDH_TABLE = arr[:, :4]
            self.MDH_LIMITS = arr[:, 4:]
        if cfg.get("T_linkn_end") is not None:
            self.T_LINKN_END = np.asarray(cfg["T_linkn_end"], dtype=float).reshape(4, 4)

        # ---- TCP 空间限位：config 提供则覆盖 JoyArm 的占位 TcpLimits ----
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
    def mdh_limits(self) -> np.ndarray:
        """返回 MDH 链路限位 ``(6,5)`` ndarray：[q_min, q_max, dq_max, tau_max, soft_margin]。"""
        return self.MDH_LIMITS.copy()

    @property
    def T_linkn_end(self) -> np.ndarray:
        """返回末link系 → end系固定位姿 ``(4,4)`` ndarray（教学查阅用）。"""
        return self.T_LINKN_END.copy()

    # ----------------------------------------------------------
    # 内部：配置应用
    # ----------------------------------------------------------
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
    def _resolve_default_urdf(cls, robot: Optional[str] = None) -> str:
        """解析随包 URDF 路径：``robots/<robot>/urdf/<robot>.urdf``（robot 取 config > 类常量）。"""
        robot = robot or cls.ROBOT
        pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # joyarm_core/
        path = os.path.join(pkg, "robots", robot, "urdf", f"{robot}.urdf")
        if os.path.isfile(path):
            return path
        robots_dir = os.path.join(pkg, "robots")
        avail = sorted(
            d for d in os.listdir(robots_dir) if os.path.isdir(os.path.join(robots_dir, d))
        )
        raise ValueError(f"『{robot}』型号在 robots 中未找到；可用：{avail}")
