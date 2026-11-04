"""JointDragTeachController —— 关节 MIT 拖动示教控制器。

1. MIT 拖动示教——重力补偿 tau=G(q) + 阻尼 kd + 刚度 kp 维持平衡点防漂移，平衡点按阈值/步进率缓释跟随（拖得动、松手停）；逐关节独立：|q − q_eq| > threshold 的关节按 step_rate 比例缓慢拖移；
2. 构造注入 kd/kp/threshold/step_rate（cfg `robotics.control` 条目透传）：构造时数值校验（kd/kp/threshold 正有限、step_rate ∈(0,1]）；
3. joint_state.q 缺失或含非有限值 → warn 一次本拍不下发；dyn 域未配置 → warn 一次退化为无重力补偿（纯阻尼+刚度保持）。
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

from . import Controller
from ...utils.types import ControlMode, JointState, TrajFrame

__all__ = ["JointDragTeachController"]

logger = logging.getLogger("joyarm_core.controller_joint_drag_teach")

# 数值参数（标量或一维序列；首拍广播为 (n_arm,)）
_Num = Union[float, np.ndarray]


def _check(name: str, val, hi: Optional[float] = None) -> None:
    """构造参数数值校验：标量或一维序列，逐元素有限且 > 0（hi 非 None 另须 ≤ hi）。

    :param name: 参数名（异常消息用）。
    :param val: 待校验值（标量或一维序列；维度校验延迟到首拍）。
    :param hi: 上界（闭区间端点；None = 不设上界）。
    :raises ValueError: 非数值、非标量/一维序列、含非有限值或越界。
    """
    try:
        arr = np.asarray(val, dtype=float)
    except (TypeError, ValueError):
        arr = None
    ok = arr is not None and arr.ndim <= 1 and bool(np.all(np.isfinite(arr))) \
        and bool(np.all(arr > 0.0)) and (hi is None or bool(np.all(arr <= hi)))
    if not ok:
        expect = f"有限数 ∈(0,{hi:g}]" if hi is not None else "正有限数"
        raise ValueError(
            "controller_joint_drag_teach.py - JointDragTeachController.__init__："
            f"{name}={val} 非法（{expect}，标量或 n_arm 元列表）")


def _broadcast(name: str, val: _Num, n: int) -> np.ndarray:
    """首拍参数广播：标量 → (n,) 常值数组；一维序列 → 长度须等于 n。

    :param name: 参数名（异常消息用）。
    :param val: 已通过 `_check` 数值校验的参数（标量或一维序列）。
    :param n: 关节数 n_arm（首拍取 `arm.n_arm`）。
    :return: (n,) float 数组（新分配，与入参不共享内存）。
    :raises ValueError: 序列长度 ≠ n。
    """
    if np.ndim(val) == 0:
        return np.full(n, float(val))
    arr = np.asarray(val, dtype=float).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(
            "controller_joint_drag_teach.py - JointDragTeachController.step："
            f"{name} 长度 {arr.shape[0]} ≠ n_arm {n}（须标量或 n_arm 元列表）")
    return arr.copy()


class JointDragTeachController(Controller):
    """拖动示教控制律：MIT 保持平衡点（重力补偿+阻尼+刚度），平衡点按阈值/步进率缓释跟随人手拖动。"""

    MODE = ControlMode.MIT                               # 电机模式：MIT（显式声明，与基类缺省一致）

    def __init__(self, kd: _Num = 1.0, kp: _Num = 10.0, threshold: _Num = 0.05, step_rate: _Num = 0.1):
        """构造：数值校验并暂存四参数（n_arm 广播与维度校验延迟到首拍）。

        :param kd: MIT 速度增益（阻尼）；正有限数，标量或 n_arm 元列表。
        :param kp: MIT 位置增益（刚度）；正有限数，标量或 n_arm 元列表。
        :param threshold: 平衡点跟随触发阈值 rad；正有限数，标量或 n_arm 元列表。
        :param step_rate: 平衡点跟随步进率（每拍缓释比例）；有限数 ∈(0,1]，标量或 n_arm 元列表。
        :raises ValueError: 任一参数非法（非有限、越界或非标量/一维序列）。
        """
        _check("kd", kd)
        _check("kp", kp)
        _check("threshold", threshold)
        _check("step_rate", step_rate, 1.0)
        self._kd: _Num = kd                              # MIT 速度增益（阻尼；首拍广播为 (n,)）
        self._kp: _Num = kp                              # MIT 位置增益（刚度；首拍广播为 (n,)）
        self._threshold: _Num = threshold                # 平衡点跟随触发阈值 rad（逐关节独立判越阈）
        self._step_rate: _Num = step_rate                # 平衡点跟随步进率（每拍缓释比例 ∈(0,1]）
        self._q_eq: Optional[np.ndarray] = None          # 平衡点（None = 未初始化，首拍锚定当前 q）
        self._state_warned: bool = False                 # 状态无效告警去重（有效状态到达后复位）
        self._grav_warned: bool = False                  # 重力退化告警去重（dyn 域未配置仅告警一次）

    def step(self, arm, target: TrajFrame, joint_state: JointState) -> bool:
        """步进指令：平衡点缓释跟随 + MIT 重力补偿保持（拖动示教；target 轨迹帧忽略）。

        :param arm: 机械臂对象。
        :param target: 目标轨迹帧（本控制律忽略——拖动示教不跟踪轨迹参考）。
        :param joint_state: 当前关节状态（取 q；缺失或含非有限值则本拍不下发）。
        :return: 是否已成功计算并下发本拍指令。

        **实现**：首拍将四参数广播为 (n_arm,) 并锚定平衡点 = 当前 q 拷贝；逐拍逐关节独立——
            |q − q_eq| > threshold 的关节平衡点按 step_rate 比例缓释跟随（其余关节不动），
            重力补偿 tau 经公开门面 `arm.get_gravity(q)` 求解，随后 `send_mit_arm` 整组下发。
        """
        q = None if joint_state is None else joint_state.q
        if q is not None:
            q = np.asarray(q, dtype=float).reshape(-1)
        if q is None or not bool(np.all(np.isfinite(q))):
            if not self._state_warned:
                logger.warning(
                    "controller_joint_drag_teach.py - JointDragTeachController.step："
                    "joint_state.q 缺失或含非有限值（NaN/inf），本拍不下发 MIT 保持指令")
                self._state_warned = True
            return False
        self._state_warned = False      # 有效状态到达，复位告警去重
        if self._q_eq is None:          # 首拍：四参数广播 + 平衡点锚定（构造时未知维数）
            n = arm.n_arm
            self._kd = _broadcast("kd", self._kd, n)
            self._kp = _broadcast("kp", self._kp, n)
            self._threshold = _broadcast("threshold", self._threshold, n)
            self._step_rate = _broadcast("step_rate", self._step_rate, n)
            self._q_eq = q.copy()
        diff = q - self._q_eq
        mask = np.abs(diff) > self._threshold    # 越阈关节：平衡点缓释跟随（其余关节不动）
        self._q_eq[mask] += self._step_rate[mask] * diff[mask]
        try:
            tau = arm.get_gravity(q)    # 公开门面：dyn 域激活求解器算 G(q)
        except RuntimeError:
            if not self._grav_warned:
                logger.warning(
                    "controller_joint_drag_teach.py - JointDragTeachController.step："
                    "dyn 域未配置（get_gravity 不可用），已退化为无重力补偿（纯阻尼+刚度保持）")
                self._grav_warned = True
            tau = np.zeros_like(self._q_eq)
        arm.send_mit_arm(q=self._q_eq, dq=np.zeros_like(self._q_eq), kp=self._kp, kd=self._kd, tau=tau)
        return True
