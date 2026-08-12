"""视觉（适配器层，Ch14 占位，可选）。

提供相机标定（内参 / 手眼）与检测 / 位姿估计接口。

.. note::

    UI / 语音 / NFC / UWB / 体感：归入 ``joyarm_code/quickstart/`` 应用层
    （架构设计第 6 条交互方式）；视觉算法占位留库内。视觉部分**先不实现**，
    仅保留接口签名 + ``raise NotImplementedError("Ch14 实现")``。

对应章节：Ch14（``chapter4_4.md`` 第十四章 用户接口）。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "intrinsic_calibrate",
    "hand_eye_calibrate",
    "Detector",
    "PoseEstimator",
]

# 【给新手的话】本文件是"占位文件"——函数体目前都是 raise NotImplementedError。
# 这不是 bug，而是教学安排：第十四章会真正实现视觉相关功能（且为可选模块）。
# 视觉让机械臂"看得见"：通过相机标定建立"像素 ↔ 物体三维位置"的对应关系，
# 再用检测/位姿估计找到目标，从而实现"看到什么就抓什么"的智能操作。


def intrinsic_calibrate(
    img_pts: List[np.ndarray],
    obj_pts: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """14.3 内参标定（Zhang 法）。

    :param img_pts: 各图像检测到的角点像素坐标列表，每个 ``(N_i,2)``。
    :param obj_pts: 标定板对应世界坐标列表，每个 ``(N_i,3)``。
    :return: ``(K, dist)`` —— ``(3,3)`` 内参矩阵与畸变系数向量。
    """
    # 占位：Ch14 实现——用棋盘格多角度拍照，解出相机内参（焦距/中心）和镜头畸变。
    raise NotImplementedError("intrinsic_calibrate 待 Ch14 实现")


def hand_eye_calibrate(
    gripper_base: List[np.ndarray],
    cam_target: List[np.ndarray],
    mode: str = "eye_in_hand",
) -> np.ndarray:
    """14.3 手眼标定。

    :param gripper_base: 机械臂基坐标系下末端位姿序列 ``[(4,4), ...]``。
    :param cam_target: 相机看到标定板位姿序列 ``[(4,4), ...]``。
    :param mode: ``"eye_in_hand"``（相机装在末端）或 ``"eye_to_hand"``
                 （相机固定外部）。
    :return: ``(4,4)`` 手眼变换矩阵 ``T_gripper_camera`` 或 ``T_base_camera``。
    """
    # 占位：Ch14 实现——求解"相机坐标系 ↔ 末端（或基座）坐标系"的固定变换，
    # 这样相机看到的物体位置才能换算成机械臂要去抓的位置。
    raise NotImplementedError("hand_eye_calibrate 待 Ch14 实现")


class Detector:
    """14.3 目标检测接口（占位）。"""

    def detect(self, image: np.ndarray) -> List[dict]:
        """检测图像中的目标。

        :param image: ``(H,W,3)`` BGR 或 RGB 图像。
        :return: 检测结果列表（每个含类别、置信度、bbox 等）。
        """
        # 占位：Ch14 实现——在图像中找出目标物体，返回位置框和类别（可接深度学习模型）。
        raise NotImplementedError("Detector.detect 待 Ch14 实现")


class PoseEstimator:
    """14.3 6D 位姿估计接口（占位）。"""

    def estimate(self, image: np.ndarray, detection: Optional[dict] = None) -> np.ndarray:
        """估计目标的 6D 位姿。

        :return: ``(4,4)`` 目标在相机坐标系下的位姿。
        """
        # 占位：Ch14 实现——不仅找到物体，还估计它的三维位置和朝向（6 自由度位姿）。
        raise NotImplementedError("PoseEstimator.estimate 待 Ch14 实现")
