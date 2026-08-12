"""``JoyArmRebotDM`` —— JoyArm 6 自由度机械臂（运动学模型层）。

固化 JoyArm（reBot-DevArm 硬件平台）特有默认值（URDF 路径、末端帧、
MDH 参考表、home 位形）。

型号与通信正交
--------------

- 不同型号（``joyarm_rebot_dm``…）= :class:`~joyarm.arm.Arm` 子类
  （各自 URDF/限位/home）。未来新增型号只需继承 ``JoyArmRebotDM`` 并覆盖
  类常量，再在 :func:`joyarm.load_arm` 注册表登记一行。
- 通信 = backend 实例。
- 二者在 :func:`joyarm.load_arm` 工厂正交组合（见 ``__init__.py``）。

MDH 参数说明
------------

``MDH_TABLE`` 为教学参考常量（改进 Denavit-Hartenberg 参数，6 行
``(α, a, d, θ_offset)``），供教学查阅；FK 计算仍走 pinocchio（黑盒，
``method="auto"``）。确切数值以第七章 URDF 导出后核对为准。

对应章节：Ch2.2 即用（joyarm_rebot_dm 预设）。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .arm import Arm
from ..backends.arm_backend import ArmBackend

__all__ = ["JoyArmRebotDM"]


class JoyArmRebotDM(Arm):
    """JoyArm（reBot-DevArm）6 自由度机械臂型号预设。

    ``urdf_path`` 缺省时用 :mod:`importlib.resources` 解析随包
    ``DEFAULT_URDF``（``robots/reBot-DevArm_fixend/urdf/reBot-DevArm_fixend.urdf``，
    随包已提供）。

    .. note::

        当前随包 URDF 为过渡型号 ``reBot-DevArm_fixend``（纯 6 转动臂，
        ``nq=6``）；第七章导出正式版 ``joyarm1.urdf`` 后替换 ``DEFAULT_URDF``。

    :param urdf_path: URDF 文件路径；缺省解析随包资源。
    :param ee_frame_name: 末端帧名；缺省用类常量 ``EE_FRAME``。
    :param mesh_dirs: mesh 搜索目录列表。
    :param backend: 真机后端实例；``None`` = 离线模式。
    :param load_geometry: 是否加载 visual/collision 几何（meshcat 可视化需 ``True``）；
                          缺省 ``False``（仅运动学计算，更快、无需 mesh）。
    :param name: 名称。
    """

    # ---- 类常量（joyarm_rebot_dm 预设）----
    # 这些是 JoyArm 机械臂的"出厂参数"；未来新增型号可继承覆盖
    # 过渡型号：reBot-DevArm_fixend（纯 6 转动臂，nq=6）；正式 URDF 待第七章导出
    DEFAULT_URDF: str = "robots/reBot-DevArm_fixend/urdf/reBot-DevArm_fixend.urdf"
    EE_FRAME: str = "end_link"                # 末端帧在 URDF 里的名字
    N_DOF: int = 6                            # 自由度数

    # MDH 参数：(α_{i-1}, a_{i-1}, d_i, θ_offset_i)，6 行
    # 教学参考常量；确切值以第七章 URDF 导出后核对为准
    # 注意：FK 实际走 pinocchio（黑盒），这张表只是给学生查原理用的
    MDH_TABLE: np.ndarray = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    # home 位形（实际 home，未必等于中性位形）；第七章标定后核对
    Q_HOME: np.ndarray = np.zeros(6)

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        ee_frame_name: Optional[str] = None,
        mesh_dirs: Optional[List[str]] = None,
        backend: Optional[ArmBackend] = None,
        load_geometry: bool = False,
        name: str = "JoyArmRebotDM",
    ):
        # urdf_path 未指定时，自动解析随包默认 URDF（见 _resolve_default_urdf）
        if urdf_path is None:
            urdf_path = self._resolve_default_urdf()
        # 调用父类 Arm 的初始化，完成 pinocchio 模型加载、限位构建等
        super().__init__(
            urdf_path=urdf_path,
            backend=backend,
            ee_frame_name=ee_frame_name or self.EE_FRAME,   # None 时用类常量
            mesh_dirs=mesh_dirs,
            load_geometry=load_geometry,
            name=name,
        )
        # 末端工具偏移（可选，默认单位阵）
        # 例如装了夹爪/相机后，末端参考点会从 end_link 偏移到工具尖端
        self.tcp_offset: np.ndarray = np.eye(4)

    # ----------------------------------------------------------
    # 教学参考
    # ----------------------------------------------------------
    @property
    def mdh_table(self) -> np.ndarray:
        """返回 MDH 参数 ``(6,4)`` ndarray（教学查阅用）。"""
        return self.MDH_TABLE.copy()           # copy 防止外部修改类常量

    @property
    def q_home(self) -> np.ndarray:
        """实际 home 位形 ``(6,)``。"""
        return self.Q_HOME.copy()              # copy 防止外部修改类常量

    # ----------------------------------------------------------
    # 内部：解析随包 URDF
    # ----------------------------------------------------------
    @classmethod
    def _resolve_default_urdf(cls) -> str:
        """用 importlib.resources 解析随包 ``DEFAULT_URDF`` 路径。

        若随包资源不存在（用户尚未放入 URDF），回退为相对代码库的路径，
        由 :class:`Arm.__init__` 抛出带定位信息的 ``FileNotFoundError``。
        """
        import os

        try:
            # importlib.resources：按"包名 + 相对路径"定位随包资源
            # 好处是即使包被打包成 zip 也能正确解析，比硬编码绝对路径更健壮
            from importlib.resources import files

            # DEFAULT_URDF 以顶层 joyarm/ 包为根（robots/ 在 joyarm/robots/），
            # 不能用 __package__（本模块在 joyarm.arm 子包里，会解析到 joyarm/arm/robots/）
            ref = files("joyarm").joinpath(cls.DEFAULT_URDF)
            # str(Path) 兼容 importlib.resources 的 Traversable
            path = str(ref)
            if os.path.isfile(path):
                return path
        except Exception:
            pass                              # 解析失败就走下面的回退路径

        # 回退：相对顶层 joyarm/ 包目录解析（本文件在 joyarm/arm/，需上溯一级）
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(here, cls.DEFAULT_URDF)
