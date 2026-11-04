"""MeshCat 教学演示共用的模型加载与简单标记工具。"""
from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

MODEL_NAME = "rebot_arm_b601_dm"
PROJECT_DIR = Path(__file__).resolve().parents[2]


def model_urdf_path(model_name: str = MODEL_NAME) -> Path:
    """返回指定型号的 URDF 路径。"""
    return (
        PROJECT_DIR
        / "joyarm_core"
        / "robot_model"
        / model_name
        / "urdf"
        / f"{model_name}.urdf"
    )


def load_visual_models(model_name: str = MODEL_NAME):
    """加载模型并锁定末端执行器，只保留本体关节。"""
    urdf_path = model_urdf_path(model_name)
    tree = ET.parse(urdf_path)
    for mesh in tree.getroot().iter("mesh"):
        filename = mesh.get("filename")
        if filename and not Path(filename).is_absolute():
            mesh.set("filename", str((urdf_path.parent / filename).resolve()))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf") as temporary:
        tree.write(temporary, encoding="unicode", xml_declaration=True)
        temporary.flush()
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            temporary.name
        )

    arm_names = {f"joint{index}" for index in range(1, 7)}
    locked_ids = [
        joint_id
        for joint_id in range(1, model.njoints)
        if model.names[joint_id] not in arm_names
    ]
    model, geometries = pin.buildReducedModel(
        model,
        [collision_model, visual_model],
        locked_ids,
        pin.neutral(model),
    )
    return model, geometries[0], geometries[1]


def create_viewer(model_name: str = MODEL_NAME, *, open_browser: bool = True):
    """创建并加载 MeshCat 可视化器。"""
    model, collision_model, visual_model = load_visual_models(model_name)
    viewer = MeshcatVisualizer(model, collision_model, visual_model)
    viewer.initViewer(open=open_browser)
    viewer.loadViewerModel(rootNodeName=model_name)
    return viewer


def add_pose_marker(viewer, node_name: str, pose, color: int, size: float = 0.07):
    """在 MeshCat 中添加坐标系和目标点。"""
    node = viewer.viewer[node_name]
    node["frame"].set_object(mg.triad(size))
    node["frame"].set_transform(pose.T)
    node["point"].set_object(
        mg.Sphere(size * 0.13),
        mg.MeshLambertMaterial(color=color),
    )
    node["point"].set_transform(pose.T)


def set_arm_color(viewer, color) -> None:
    """设置机械臂可视化几何的颜色。"""
    rgba = np.asarray(color, dtype=float)
    for geometry_object in viewer.visual_model.geometryObjects:
        viewer.loadViewerGeometryObject(
            geometry_object,
            pin.GeometryType.VISUAL,
            color=rgba,
        )


def add_path(viewer, node_name: str, points: np.ndarray, color: int) -> None:
    """在 MeshCat 中绘制 TCP 折线路径。"""
    geometry = mg.PointsGeometry(np.asarray(points, dtype=float).T)
    material = mg.LineBasicMaterial(color=color, linewidth=3.0)
    viewer.viewer[node_name].set_object(mg.Line(geometry, material))
