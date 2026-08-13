"""``Arm`` —— 完整机械臂（含末端执行器）= Mas + End。

``Arm(Mas)`` 继承多轴本体的全部能力（``model``/``fkine``/``command``/…），并组合
一个末端执行器 ``self.end``（:class:`joyarm.arms.end.End`）。子类绑定两个后端**类**：
``backend_mas_cls``（本体后端类，定义于 :class:`Mas`）与 ``backend_end_cls``（末端后端类，
定义于本类），构造时分别按 ``configs/*.yaml`` 的 ``backend_mas`` / ``backend_end`` 段实例化。
"""
from __future__ import annotations

from typing import List, Optional

from .mas import Mas
from .end import End

__all__ = ["Arm"]


class Arm(Mas):
    """完整机械臂 = 多轴本体（Mas）+ 末端执行器（End）。

    :param urdf_path: URDF 文件路径。
    :param ee_frame_name: 末端参考帧名（默认 ``"ee"``）。
    :param mesh_dirs: URDF mesh 搜索目录列表。
    :param load_geometry: 是否加载 visual/collision 几何。
    :param name: 名称。
    :param config: 型号 YAML 配置字典（含 ``backend_mas`` / ``backend_end`` 等段）。
    """

    # 末端后端**类**（本体后端类 ``backend_mas_cls`` 定义于 Mas，由本类继承）
    backend_end_cls: Optional[type] = None

    def __init__(
        self,
        urdf_path: str,
        ee_frame_name: str = "ee",
        mesh_dirs: Optional[List[str]] = None,
        load_geometry: bool = False,
        name: str = "Arm",
        config: Optional[dict] = None,
    ):
        super().__init__(
            urdf_path=urdf_path,
            ee_frame_name=ee_frame_name,
            mesh_dirs=mesh_dirs,
            load_geometry=load_geometry,
            name=name,
            config=config,
        )
        cfg = config or {}
        # 组合末端执行器：按 config['backend_end'] 段实例化 backend_end_cls
        self.end: Optional[End] = (
            End(backend_cls=self.backend_end_cls, backend_params=cfg.get("backend_end"))
            if self.backend_end_cls is not None
            else None
        )

    # ----------------------------------------------------------
    # 连接（本体 + 末端）
    # ----------------------------------------------------------
    def connect(self) -> None:
        """连接本体 + 末端真机。"""
        super().connect()                       # Mas.connect() 连 backend_mas
        if self.end is not None:
            self.end.connect()

    def disconnect(self) -> None:
        """断开本体 + 末端真机。"""
        super().disconnect()
        if self.end is not None:
            self.end.disconnect()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, n={self.n}, "
            f"ee_frame={self.ee_frame_name!r}, end={'yes' if self.end else 'no'}, "
            f"{'connected' if self.connected else 'offline'})"
        )
