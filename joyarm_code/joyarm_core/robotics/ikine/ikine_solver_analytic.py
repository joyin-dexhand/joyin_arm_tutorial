"""JoyArm 解析逆运动学：枚举肩、腕、肘分支，并通过 MDH 正运动学回代验证。

适用于第2/3/4轴平行、第5/6轴相交的六转动关节构型。
MDH 正运动学与单节变换复用 fkine_solver_mdh 模块。
"""

from __future__ import annotations

import numpy as np

from ..fkine.fkine_solver_mdh import _fkine_mdh, _mdh_transform
from .ikine_solver import IkineSolver
from ...utils.types import IKResult, Pose

__all__ = ["AnalyticIkineSolver"]


def _target_link6(target: Pose, T_e6: np.ndarray) -> np.ndarray:
    """将目标 TCP 位姿转换为 MDH 第6系在基座系下的位姿。

    T_0e = T_06 @ T_6e，因此 T_06 = T_0e @ inv(T_6e)。
    T_0e 由 target.T 给出；T_e6 为预先计算的 inv(T_6e)。
    返回 (4,4) 齐次变换矩阵。
    """
    return target.T @ T_e6


def _solve_q1(mdh: np.ndarray, p_w: np.ndarray) -> list[float]:
    """根据腕心位置求解两个肩部分支，返回 q1（弧度）列表。

    lateral = d4 + d5*cos(alpha4)
    radial = ±sqrt(x_w**2 + y_w**2 - lateral**2)
    theta1 = atan2(y_w, x_w) - atan2(lateral, radial)
    q1 = theta1 - theta10

    根号内为负时，腕心无法满足固定侧向偏置，返回空列表。
    """
    x_w, y_w = p_w[0], p_w[1]

    d4 = mdh[3, 2]
    alpha4 = mdh[4, 0]
    d5 = mdh[4, 2]

    lateral = d4 + d5 * np.cos(alpha4)
    theta10 = mdh[0, 3]

    rho_sq = x_w**2 + y_w**2
    radial_sq = rho_sq - lateral**2
    if radial_sq < 0:
        return []

    radial = np.sqrt(radial_sq)
    azimuth = np.arctan2(y_w, x_w)

    solutions = []

    for r in (radial, -radial):
        theta1 = azimuth - np.arctan2(lateral, r)
        q1 = theta1 - theta10
        solutions.append(float(q1))

    return solutions

def _rot_x(angle: float) -> np.ndarray:
    """返回绕 X 轴旋转 angle 弧度的 (3,3) 旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def _rot_z(angle: float) -> np.ndarray:
    """返回绕 Z 轴旋转 angle 弧度的 (3,3) 旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)

    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _reduced_orientation(mdh: np.ndarray, R_06: np.ndarray, q1: float) -> np.ndarray:
    """消除第1关节姿态，得到只含 phi、theta5、theta6 的约化姿态。

    theta1 = q1 + theta10
    B = Rz(theta1) @ Rx(-pi/2)
    Q = B.T @ R_06
    Q = Rz(phi) @ Rx(alpha4) @ Rz(theta5) @ Rx(alpha5) @ Rz(theta6)
    其中 phi = -theta2 + theta3 + theta4，返回 Q 的尺寸为 (3,3)。
    """
    theta1 = q1 + mdh[0, 3]

    B = _rot_z(theta1) @ _rot_x(-np.pi / 2.0)
    Q = B.T @ R_06
    return Q

def _solve_phi_q56(
    mdh: np.ndarray,
    Q: np.ndarray,
    q6_ref: float = 0.0,
    singular_tol: float = 1e-9,
) -> list[tuple[float, float, float]]:
    """从约化姿态 Q 求解两个腕部分支，返回 (phi, q5, q6) 列表。

    cos(theta5) = -Q[2,2] / sin(alpha4)
    sin(theta5) = ±sqrt(1 - cos(theta5)**2)
    phi = atan2(Q[1,2], Q[0,2])
          - atan2(-cos(alpha4)*cos(theta5), sin(theta5))

    非奇异时，移除第6关节之前的旋转后，由 R_6 = Rz(theta6) 提取 theta6。
    q5、q6 需减去 MDH 零位偏置，phi 为平面合角，所有角度单位为弧度。
    当 abs(sin(theta5)) < singular_tol 时，theta5 为 0 或 pi，phi 与
    theta6 只能确定组合角。此时以 q6_ref 作为 q6 的代表值，再由组合角
    求 phi，返回一个可复现的奇异位形解。
    """
    alpha4, alpha5 = mdh[4, 0], mdh[5, 0]
    theta50, theta60 = mdh[4, 3], mdh[5, 3]
    sa4, ca4 = np.sin(alpha4), np.cos(alpha4)

    cos_theta5 = np.clip(-Q[2, 2] / sa4, -1.0, 1.0)
    sin_abs = np.sqrt(max(0.0, 1.0 - cos_theta5**2))
    # 奇异分支：theta5 = 0 或 pi，phi 与 theta6 耦合，无法单独求解。
    if sin_abs < singular_tol:
        # q6_ref 是关节角 q6；theta6 还要加 MDH 固定偏置 theta60。
        theta6 = q6_ref + theta60

        if cos_theta5 >= 0.0:
            # theta5 = 0：phi + theta6 = gamma
            theta5 = 0.0
            gamma = np.arctan2(Q[1, 0], Q[0, 0])
            phi = gamma - theta6
        else:
            # theta5 = pi：phi - theta6 = gamma
            theta5 = np.pi
            gamma = np.arctan2(-Q[1, 0], -Q[0, 0])
            phi = gamma + theta6

        # 归一化到 [-pi, pi]。
        phi = np.arctan2(np.sin(phi), np.cos(phi))

        return [(
            float(phi),
            float(theta5 - theta50),
            float(theta6 - theta60),
        )]
    solutions = []
    for sin_theta5 in (sin_abs, -sin_abs):
        theta5 = np.arctan2(sin_theta5, cos_theta5)
        phi = (
            np.arctan2(Q[1, 2], Q[0, 2])
            - np.arctan2(-ca4 * cos_theta5, sin_theta5)
        )
        R_before_6 = (
            _rot_z(phi)
            @ _rot_x(alpha4)
            @ _rot_z(theta5)
            @ _rot_x(alpha5)
        )
        R_6 = R_before_6.T @ Q
        theta6 = np.arctan2(R_6[1, 0], R_6[0, 0])
        solutions.append((
            float(phi),
            float(theta5 - theta50),
            float(theta6 - theta60),
        ))
    return solutions


def _solve_q234(
    mdh: np.ndarray,
    p_w: np.ndarray,
    q1: float,
    phi: float,
    reach_tol: float = 1e-10,
) -> list[tuple[float, float, float]]:
    """已知 q1 和平面合角 phi，求肘上、肘下两组 (q2, q3, q4)。

    先将腕心变换到第1系：p_1 = R_01.T @ (p_w - p_01)。
    移除肩部偏置和由 phi 确定的腕部投影后，剩余平面二连杆满足：

    cos(theta3) = (x**2 + z**2 - a2**2 - a3**2) / (2*a2*a3)
    theta3 = ±acos(cos(theta3))
    theta2 = atan2(z, x) + atan2(a3*sin(theta3), a2 + a3*cos(theta3))
    theta4 = phi + theta2 - theta3

    各 theta 减去对应 MDH 零位偏置后得到关节角 q（弧度）。
    余弦值超出 [-1,1] 且超过 reach_tol 时判定不可达，返回空列表。
    """
    T_01 = _mdh_transform(mdh[0], q1)
    p_1 = T_01[:3, :3].T @ (p_w - T_01[:3, 3])

    a1, a2, a3, a4 = mdh[1, 1], mdh[2, 1], mdh[3, 1], mdh[4, 1]
    alpha4, d5 = mdh[4, 0], mdh[4, 2]
    theta20, theta30, theta40 = mdh[1, 3], mdh[2, 3], mdh[3, 3]


    # 移除肩部偏置和已知腕部投影，得到标准平面二连杆的目标坐标。
    wrist_x = a4 * np.cos(phi) + np.sin(alpha4) * d5 * np.sin(phi)
    wrist_z = -a4 * np.sin(phi) + np.sin(alpha4) * d5 * np.cos(phi)
    x = p_1[0] - a1 - wrist_x
    z = p_1[2] - wrist_z

    cosine3 = (x * x + z * z - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
    if cosine3 < -1.0 - reach_tol or cosine3 > 1.0 + reach_tol:
        return []
    cosine3 = float(np.clip(cosine3, -1.0, 1.0))

    solutions = []
    for theta3 in (np.arccos(cosine3), -np.arccos(cosine3)):
        theta2 = (
            np.arctan2(z, x)
            + np.arctan2(a3 * np.sin(theta3), a2 + a3 * np.cos(theta3))
        )
        theta4 = phi + theta2 - theta3
        solutions.append((
            float(theta2 - theta20),
            float(theta3 - theta30),
            float(theta4 - theta40),
        ))
    return solutions


def _pose_error(T_actual: np.ndarray, T_target: np.ndarray) -> float:
    """计算 FK 回代残差：位置误差范数与姿态误差角之和。

    e_p = norm(p_actual - p_target)
    R_error = R_target @ R_actual.T
    e_R = acos((trace(R_error) - 1) / 2)
    返回 e_p + e_R；这是用于筛选候选解的混合残差。
    """
    position = np.linalg.norm(T_actual[:3, 3] - T_target[:3, 3])
    R_error = T_target[:3, :3] @ T_actual[:3, :3].T
    angle = np.arccos(np.clip((np.trace(R_error) - 1.0) / 2.0, -1.0, 1.0))
    return float(position + angle)


class AnalyticIkineSolver(IkineSolver):

    """JoyArm 六轴解析逆解：第2/3/4轴平行，第5/6轴相交。

    枚举 2 个肩部 × 2 个腕部 × 2 个肘部分支，最多生成 8 组候选解。
    用 MDH 正运动学回代验证，腕部奇异时以 q6 参考角选取代表解。
    固定参数在首次求解时准备；同一 arm 连续求解复用，切换实例时重新准备。
    配置遵循机械臂实例生命周期内不变、修改后重建实例的约定。
    """

    def __init__(self):
        """仅缓存最近使用的 arm 及其固定参数，不缓存目标或关节限位。"""
        self._prepared = None

    def _prepare(self, arm) -> tuple[np.ndarray, np.ndarray]:
        """通过公开配置快照准备 MDH 和 TCP 逆变换，按 arm 对象身份复用。

        :param arm: 提供 get_config() 的机械臂实例。
        :return: MDH 参数数组和 TCP 到第6系的固定变换。
        """
        prepared = self._prepared
        if prepared is None or prepared[0] is not arm:
            config = arm.get_config()["joyarm"]
            mdh = np.asarray(config["arm_mdh"], dtype=float)
            T_e6 = np.linalg.inv(np.asarray(config["T_linkn_endtcp"], dtype=float))
            mdh.setflags(write=False)
            T_e6.setflags(write=False)
            # 完整准备成功后再替换；局部快照保证本次求解使用同一组参数。
            prepared = (arm, mdh, T_e6)
            self._prepared = prepared
        return prepared[1], prepared[2]

    def solve(
        self,
        arm,
        target: Pose,
        frame,
        q0: np.ndarray,
        *,
        tol: float = 1e-6,
        iters: int = 0,
        **kwargs,
    ) -> IKResult:
        """剔除超出软限位的候选解，返回最接近 q0 的单组解。

        先调用 solve_all，再最小化各关节角与 q0 的偏差平方和。
        :param target: TCP 目标位姿。
        :param frame: 配置的末端帧名称或索引。
        :param q0: (6,) 参考关节角，单位为弧度。
        :param tol: FK 回代残差阈值，传给 solve_all。
        :param iters: 为兼容统一接口保留，解析法不使用。
        :return: IKResult；无可行解时 success=False。
        """
        result = self.solve_all(arm, target, frame, tol=tol,q6_ref=float(np.asarray(q0, dtype=float)[5]), **kwargs)
        if not result.success:
            return result
        limits = arm.arm_limits_soft
        return self._select_nearest(
            result.q,
            np.asarray(q0, dtype=float),
            limits.q_min,
            limits.q_max,
        )

    def solve_all(
        self,
        arm,
        target: Pose,
        frame,
        *,
        tol: float = 1e-6,
        **kwargs,
    ) -> IKResult:
        """枚举全部非奇异候选解，用 MDH FK 回代剔除伪解并去重。

        计算顺序：移除 TCP 固定变换 → q1 → (phi, q5, q6) → (q2, q3, q4)。
        逐关节加减整圈，尽量将角度移入硬限位；本方法不剔除越限解。
        :param target: TCP 目标位姿。
        :param frame: 配置的末端帧名称或索引，其他帧不支持。
        :param tol: 位置误差范数与姿态误差角之和的允许上限。
        :return: IKResult，q 为 (K,6)，K <= 8；无解时为 (0,6)。
        :raises ValueError: 请求的帧不是配置的末端帧。
        """
        q6_ref = float(kwargs.pop("q6_ref", 0.0))
        model = arm.pin_model
        end_id = model.getFrameId(arm.ee_frame_name)
        frame_id = int(frame) if isinstance(frame, (int, np.integer)) \
            else model.getFrameId(str(frame))
        if frame_id != end_id:
            raise ValueError(
                "ikine_solver_analytic.py - AnalyticIkineSolver："
                f"仅支持配置的末端帧 {arm.ee_frame_name!r}，实际为 {frame!r}"
            )

        mdh, T_e6 = self._prepare(arm)
        T_06 = _target_link6(target, T_e6)
        raw = []
        for q1 in _solve_q1(mdh, T_06[:3, 3]):
            Q = _reduced_orientation(mdh, T_06[:3, :3], q1)
            for phi, q5, q6 in _solve_phi_q56(mdh, Q, q6_ref=q6_ref):
                for q2, q3, q4 in _solve_q234(mdh, T_06[:3, 3], q1, phi):
                    raw.append(np.array([q1, q2, q3, q4, q5, q6]))

        limits = arm.arm_limits
        solutions = []
        errors = []
        for q in raw:
            q = self._shift_2pi(q, limits.q_min, limits.q_max)
            err = _pose_error(_fkine_mdh(mdh, q), T_06)
            if err > tol:
                continue
            # 用模 2π 的角差去重，合并相差整圈的等价构型。
            if any(np.max(np.abs(np.arctan2(np.sin(q - old), np.cos(q - old)))) < 1e-8
                   for old in solutions):
                continue
            solutions.append(q)
            errors.append(err)

        if not solutions:
            return IKResult(q=np.zeros((0, 6)), success=False, err=float("inf"), n_iter=0)
        return IKResult(
            q=np.vstack(solutions),
            success=True,
            err=float(max(errors)),
            n_iter=0,
        )
