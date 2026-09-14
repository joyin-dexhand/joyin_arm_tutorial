"""PinIkineSolver —— 基于 pinocchio 的数值逆运动学默认实现。

Levenberg-Marquardt 迭代：世界系六维位姿误差 ``[Δp; log3(R_ref·R_curᵀ)]``
经 ``computeFrameJacobian``（LOCAL_WORLD_ALIGNED）的阻尼伪逆修正关节角，
步被拒（误差不降）则增大阻尼、被接受则衰减（自适应正则化），逐迭代钳硬限位；
单起点失败时限位内随机重启（默认 3 次）规避局部极小。适用任意构型（5/6/7 轴通用）。
config ``robotics.ikine`` 段写注册名 ``pin_ikine_solver`` 即选用。
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from . import IkineSolver
from ...utils.types import IKResult, Pose

__all__ = ["PinIkineSolver"]

_LAMBDA_MAX = 1e4    # 阻尼上限（超出视为该起点失败）


def _frame_id(model, frame) -> int:
    """帧名/索引 → 帧索引；未找到抛错（getFrameId 静默返回哨兵，须拦截）。"""
    fid = int(frame) if isinstance(frame, (int, np.integer)) \
        else model.getFrameId(str(frame))
    if not 0 <= fid < model.nframes:   # 拒越界与负索引
        raise ValueError(
            f"ikine_solver_pin.py - PinIkineSolver：帧 {frame!r} 在模型中未找到；"
            f"可用帧：{[f.name for f in model.frames]}")
    return fid


class PinIkineSolver(IkineSolver):
    """逆运动学默认实现：LM 数值迭代 + 多起点重启（pinocchio FK/雅可比）。"""

    def __init__(self, damping: float = 1e-3, step_max: float = 0.5,
                 restarts: int = 3):
        """求解参数经 config ``robotics.ikine`` 的 ``**params`` 注入。

        :param damping: LM 初始阻尼 λ（自适应升降：步接受 ×0.5、被拒 ×4）。
        :param step_max: 单次迭代关节角修正上限（rad，防单步跨跳过冲）。
        :param restarts: 单起点失败后的限位内随机重启次数（0 = 不重启）；
            重启采样确定性（固定种子），同输入同结果。
        """
        self.damping = float(damping)
        self.step_max = float(step_max)
        self.restarts = int(restarts)

    def solve(self, arm, target: Pose, frame, q0: np.ndarray, *,
              tol: float = 1e-3, iters: int = 200, **kw) -> IKResult:
        """数值求解：LM 迭代至位姿误差 < ``tol``（位置 m 范数 + 姿态 rad
        范数之和）；``q0`` 为首选起点，失败时限位内随机重启。

        不可达目标**不抛异常**——返回 ``IKResult(success=False, err=最优残差)``。
        """
        model = arm.pin_model
        data = pin.Data(model)      # 整个求解一份私有 Data（迭代内复用、不共享）
        fid = _frame_id(model, frame)
        limits = arm.arm_limits
        T_ref = target.T
        starts = [np.asarray(q0, dtype=float).reshape(-1)]
        if limits is not None:                              # 无限位声明则不重启
            rng = np.random.default_rng(0)                  # 重启确定性
            for _ in range(self.restarts):
                starts.append(rng.uniform(limits.q_min, limits.q_max))

        best = IKResult(success=False, err=float("inf"))
        total = 0
        for s, q_start in enumerate(starts):
            q, used = self._lm(model, data, fid, limits, q_start, T_ref,
                               tol, iters)
            total += used
            err = self._err(model, data, fid, q, T_ref)
            if err < best.err:
                best = IKResult(q=q, success=err < tol, err=err, n_iter=total)
            if best.success:
                break                                        # 首个成功即返回
        return best

    # ---- 内部：单起点 LM 迭代 ----
    def _lm(self, model, data, fid, limits, q0, T_ref,
            tol: float, iters: int):
        """单起点 Levenberg-Marquardt：返回 (q, 实际迭代数)。"""
        p_ref, R_ref = T_ref[:3, 3], T_ref[:3, :3]
        q = q0.copy()
        lam = self.damping
        err = self._err(model, data, fid, q, T_ref)
        for i in range(int(iters)):
            if err < tol:
                return q, i
            pin.forwardKinematics(model, data, q)
            M = pin.updateFramePlacement(model, data, fid)
            e = np.hstack([p_ref - M.translation,              # 位置误差（世界系）
                           pin.log3(R_ref @ M.rotation.T)])     # 姿态误差旋量（世界系）
            J = pin.computeFrameJacobian(model, data, q, fid,
                                         pin.LOCAL_WORLD_ALIGNED)
            U, sv, Vt = np.linalg.svd(J, full_matrices=False)
            for _ in range(20):                                # 内环：调 λ 至误差下降
                dq = (Vt.T * (sv / (sv ** 2 + lam ** 2))) @ U.T @ e
                n = np.linalg.norm(dq)
                if n > self.step_max:
                    dq *= self.step_max / n
                q_new = np.clip(q + dq, limits.q_min, limits.q_max) \
                    if limits is not None else q + dq
                err_new = self._err(model, data, fid, q_new, T_ref)
                if err_new < err:
                    lam = max(lam * 0.5, 1e-6)
                    q, err = q_new, err_new
                    break
                lam *= 4
                if lam > _LAMBDA_MAX:
                    return q, i + 1
        return q, int(iters)

    @staticmethod
    def _err(model, data, fid, q, T_ref) -> float:
        """位姿残差：位置误差范数 + 姿态误差角（世界系）。"""
        pin.forwardKinematics(model, data, q)
        M = pin.updateFramePlacement(model, data, fid)
        return float(np.linalg.norm(T_ref[:3, 3] - M.translation)
                     + np.linalg.norm(pin.log3(T_ref[:3, :3] @ M.rotation.T)))
