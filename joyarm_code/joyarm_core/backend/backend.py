"""``Backend`` —— joyarm 硬件后端抽象基类。

负责接入具体型号机械臂硬件（总线、协议），向 joyarm 提供硬件无关的统一接口。

代码主要分三层：
  1. 公开方法（供上层调用）；
  2. 共用实现 ``_*_impl``；
  3. 抽象方法（``@abstractmethod``，子类按型号实现；多数为单关节操作，连接/断开/末端离散动作为整组操作。

主要约定：
- 失败处理：抽象方法失败可上抛异常或返回 ``False``/``None``，基类统一转为 warn 日志；
- 生命周期标志 ``is_connected_*`` / ``is_abled_*``：``None`` 未知、``True``/``False`` 已确定；
        操作执行前置 ``None``，成功后确定，中途意外停留 ``None``；
- 发送指令自动做模式检查、维度/空值检查与越限裁剪（就近裁剪 + 限频告警）；
- 状态码统一 ``0``=失能、``1``=使能（均正常）、``≥2``=故障（子类负责映射厂商原始码）。

子类：实现全部 ``@abstractmethod``。
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from ..utils.limits import limits_from_joint_cfgs
from ..utils.types import ControlMode, JointLimits, JointState

__all__ = ["Backend"]

logger = logging.getLogger("joyarm_core.backend")
logger.setLevel(logging.INFO)  # 教学库：放行 INFO，成功信息默认可见（不依赖应用层日志配置）

# 三个可选运行参数的默认值（cfg ``backend:`` 段顶层同名键可覆盖，缺省或非正数沿用默认并 warn）
_WARN_INTERVAL_DEFAULT = 0.5   # warn_interval（秒）：越限告警限频间隔
_WRITE_SETTLE_DEFAULT = 0.1    # write_settle（秒）：写参数后的静置等待
_REFRESH_HZ_DEFAULT = 10.0     # state_refresh_hz（Hz）：低频状态刷新频率


class Backend(ABC):
    """硬件后端抽象基类（arm + end 一体）。
    """

    def __init__(self, cfg: dict) -> None:
        """初始化: cfg 解析成员并启动低频状态刷新线程。

        :param cfg: config 配置 ``backend:`` 段整体。
        """
        # ============================================================
        # 成员变量定义（先赋初值，具体值经下方初始化过程自 cfg 解析）
        # ============================================================
        self._cfg = cfg                                       # config 的 backend 段
        self._name = ""                                       # backend 名（cfg.name）
        self._channel_arm = None                              # arm 总线通道（cfg.arm.channel）
        self._channel_end = None                              # end 总线通道（cfg.end.channel）
        self._baud_arm = None                                 # arm 总线波特率（cfg.arm.baud_rate）
        self._baud_end = None                                 # end 总线波特率（cfg.end.baud_rate）
        self._protocol_arm = ""                               # arm 协议（cfg.arm.protocol）
        self._protocol_end = ""                               # end 协议（cfg.end.protocol）
        self._jointscfg_arm: list[dict] = []                  # arm 各 joint 配置（cfg.arm.joints）
        self._jointscfg_end: list[dict] = []                  # end 各 joint 配置（cfg.end.joints）
        self._n_joints_arm: int = 0                           # arm 关节数（len(cfg.arm.joints)）
        self._n_joints_end: int = 0                           # end 电机数（len(cfg.end.joints)）
        self._joint_motor_id_arm: list = []                   # arm 各电机总线地址（cfg.arm.joints[].motor_id）
        self._joint_motor_id_end: list = []                   # end 各电机总线地址（cfg.end.joints[].motor_id）
        self._joint_feedback_id_arm: list = []                # arm 各关节应答帧标识（cfg.arm.joints[].feedback_id）
        self._joint_feedback_id_end: list = []                # end 各电机应答帧标识（cfg.end.joints[].feedback_id）
        self._joint_model_arm: list = []                      # arm 各电机型号（cfg.arm.joints[].model）
        self._joint_model_end: list = []                      # end 各电机型号（cfg.end.joints[].model）
        self._joint_limits_arm: Optional[JointLimits] = None  # arm 硬限位（发送越限裁剪的依据）
        self._joint_limits_end: Optional[JointLimits] = None  # end 硬限位（发送越限裁剪的依据）
        self._kp_mit_default_arm: Optional[np.ndarray] = None # arm MIT 位置增益发送默认（cfg MIT.kp → (n,)）
        self._kd_mit_default_arm: Optional[np.ndarray] = None # arm MIT 速度阻尼发送默认（cfg MIT.kd → (n,)）
        self._kp_mit_default_end: Optional[np.ndarray] = None # end MIT 位置增益发送默认（cfg MIT.kp → (n,)）
        self._kd_mit_default_end: Optional[np.ndarray] = None # end MIT 速度阻尼发送默认（cfg MIT.kd → (n,)）
        self._vlim_default_arm: Optional[np.ndarray] = None   # arm 位置指令限速发送默认（cfg POS_VEL.vlim → (n,)）
        self._flim_default_arm: Optional[np.ndarray] = None   # arm 归一化限流发送默认 0~1（cfg POS_VEL.flim → (n,)）
        self._vlim_default_end: Optional[np.ndarray] = None   # end 位置指令限速发送默认（cfg POS_VEL.vlim → (n,)）
        self._flim_default_end: Optional[np.ndarray] = None   # end 归一化限流发送默认 0~1（cfg POS_VEL.flim → (n,)）
        self._pos_kp_arm: Optional[np.ndarray] = None         # arm 电机寄存器位置环 kp 增益（cfg POS_VEL.pos_kp → (n,)，设模式时子类直取）
        self._pos_ki_arm: Optional[np.ndarray] = None         # arm 电机寄存器位置环 ki 增益（cfg POS_VEL.pos_ki → (n,)）
        self._vel_kp_arm: Optional[np.ndarray] = None         # arm 电机寄存器速度环 kp 增益（cfg POS_VEL.vel_kp → (n,)）
        self._vel_ki_arm: Optional[np.ndarray] = None         # arm 电机寄存器速度环 ki 增益（cfg POS_VEL.vel_ki → (n,)）
        self._pos_kp_end: Optional[np.ndarray] = None         # end 电机寄存器位置环 kp 增益（cfg POS_VEL.pos_kp → (n,)）
        self._pos_ki_end: Optional[np.ndarray] = None         # end 电机寄存器位置环 ki 增益（cfg POS_VEL.pos_ki → (n,)）
        self._vel_kp_end: Optional[np.ndarray] = None         # end 电机寄存器速度环 kp 增益（cfg POS_VEL.vel_kp → (n,)）
        self._vel_ki_end: Optional[np.ndarray] = None         # end 电机寄存器速度环 ki 增益（cfg POS_VEL.vel_ki → (n,)）
        self._is_connected_arm: Optional[bool] = None         # arm 连接状态（True 已连接 / False 未连接 / None 未知）
        self._is_connected_end: Optional[bool] = None         # end 连接状态（True 已连接 / False 未连接 / None 未知）
        self._is_abled_arm: Optional[bool] = None             # arm 使能状态（True 已使能 / False 已失能 / None 未知）
        self._is_abled_end: Optional[bool] = None             # end 使能状态（True 已使能 / False 已失能 / None 未知）
        self._mode_arm: Optional[ControlMode] = None          # arm 全组一致的控制模式（get_mode 读齐且一致时更新，None=未读取）
        self._mode_end: Optional[ControlMode] = None          # end 全组一致的控制模式（get_mode 读齐且一致时更新，None=未读取）
        self._joint_mode_arm: list = []                       # arm 逐关节控制模式（None=未读取；get_mode 读齐后整体赋值）
        self._joint_mode_end: list = []                       # end 逐关节控制模式（None=未读取；get_mode 读齐后整体赋值）
        self._joint_state_arm = JointState()                  # arm 整组实时关节状态
        self._joint_state_end = JointState()                  # end 整组实时关节状态
        self._warn_lock = threading.Lock()                    # 告警限频锁（指令可能从控制线程与应用线程并发发来）
        self._warn_last = 0.0                                 # 上次限频告警的 monotonic 时刻
        self._warn_interval = _WARN_INTERVAL_DEFAULT          # 越限告警限频间隔（秒；cfg warn_interval 可覆盖）
        self._write_settle = _WRITE_SETTLE_DEFAULT            # 写参数后的静置等待（秒；cfg write_settle 可覆盖）
        self._refresh_hz = _REFRESH_HZ_DEFAULT                # 低频状态刷新频率（Hz；cfg state_refresh_hz 可覆盖）
        self._refresh_stop = threading.Event()                # 刷新线程停止标志
        self._refresh_thread: Optional[threading.Thread] = None  # 刷新线程（daemon）

        # ============================================================
        # 初始化过程（自 cfg 解析具体值并启动低频刷新线程）
        # ============================================================
        cfg = dict(cfg or {})
        arm_cfg = dict(cfg.get("arm") or {})
        end_cfg = dict(cfg.get("end") or {})
        self._name = str(cfg.get("name", ""))
        self._channel_arm = arm_cfg.get("channel")
        self._channel_end = end_cfg.get("channel")
        self._baud_arm = arm_cfg.get("baud_rate")
        self._baud_end = end_cfg.get("baud_rate")
        self._protocol_arm = str(arm_cfg.get("protocol", ""))
        self._protocol_end = str(end_cfg.get("protocol", ""))
        self._jointscfg_arm = list(arm_cfg.get("joints") or [])
        self._jointscfg_end = list(end_cfg.get("joints") or [])
        # 逐关节协议标识三键（所有子类通用，缺键 None 占位、校验留给子类协议层）
        self._joint_motor_id_arm = [j.get("motor_id") for j in self._jointscfg_arm]
        self._joint_motor_id_end = [j.get("motor_id") for j in self._jointscfg_end]
        self._joint_feedback_id_arm = [j.get("feedback_id") for j in self._jointscfg_arm]
        self._joint_feedback_id_end = [j.get("feedback_id") for j in self._jointscfg_end]
        self._joint_model_arm = [j.get("model") for j in self._jointscfg_arm]
        self._joint_model_end = [j.get("model") for j in self._jointscfg_end]
        self._joint_limits_arm = limits_from_joint_cfgs(self._jointscfg_arm)
        self._joint_limits_end = limits_from_joint_cfgs(self._jointscfg_end)
        self._n_joints_arm, self._n_joints_end = len(self._jointscfg_arm), len(self._jointscfg_end)
        # 状态按关节数定维：t 填 0、物理量填 NaN、error 填 -1，表示未读取；逐关节模式缓存填 None（未读取）
        self._joint_state_arm = JointState(t=0.0,
                                           q=np.full(self._n_joints_arm, np.nan),
                                           dq=np.full(self._n_joints_arm, np.nan),
                                           tau=np.full(self._n_joints_arm, np.nan),
                                           temp_mos=np.full(self._n_joints_arm, np.nan),
                                           temp_rotor=np.full(self._n_joints_arm, np.nan),
                                           error=np.full(self._n_joints_arm, -1))
        self._joint_state_end = JointState(t=0.0,
                                           q=np.full(self._n_joints_end, np.nan),
                                           dq=np.full(self._n_joints_end, np.nan),
                                           tau=np.full(self._n_joints_end, np.nan),
                                           temp_mos=np.full(self._n_joints_end, np.nan),
                                           temp_rotor=np.full(self._n_joints_end, np.nan),
                                           error=np.full(self._n_joints_end, -1))
        self._joint_mode_arm = [None] * self._n_joints_arm
        self._joint_mode_end = [None] * self._n_joints_end
        # 提取发送默认值与 POS_VEL 增益为 (n,) 成员：缺配置的关节以 NaN 占位并 warn；
        self._kp_mit_default_arm = self._extract_joint_values("arm", "MIT", "kp")
        self._kd_mit_default_arm = self._extract_joint_values("arm", "MIT", "kd")
        self._vlim_default_arm = self._extract_joint_values("arm", "POS_VEL", "vlim")
        self._flim_default_arm = self._extract_joint_values("arm", "POS_VEL", "flim")
        self._pos_kp_arm = self._extract_joint_values("arm", "POS_VEL", "pos_kp")
        self._pos_ki_arm = self._extract_joint_values("arm", "POS_VEL", "pos_ki")
        self._vel_kp_arm = self._extract_joint_values("arm", "POS_VEL", "vel_kp")
        self._vel_ki_arm = self._extract_joint_values("arm", "POS_VEL", "vel_ki")
        self._kp_mit_default_end = self._extract_joint_values("end", "MIT", "kp")
        self._kd_mit_default_end = self._extract_joint_values("end", "MIT", "kd")
        self._vlim_default_end = self._extract_joint_values("end", "POS_VEL", "vlim")
        self._flim_default_end = self._extract_joint_values("end", "POS_VEL", "flim")
        self._pos_kp_end = self._extract_joint_values("end", "POS_VEL", "pos_kp")
        self._pos_ki_end = self._extract_joint_values("end", "POS_VEL", "pos_ki")
        self._vel_kp_end = self._extract_joint_values("end", "POS_VEL", "vel_kp")
        self._vel_ki_end = self._extract_joint_values("end", "POS_VEL", "vel_ki")
        # 可选运行参数（cfg 顶层键）：缺省或非正数沿用默认并 warn
        val = float(cfg.get("state_refresh_hz", _REFRESH_HZ_DEFAULT))
        if val <= 0:
            logger.warning("backend.py - Backend.__init__：state_refresh_hz=%s 非法（须为正数），沿用默认 %s Hz",
                           val, _REFRESH_HZ_DEFAULT)
        else:
            self._refresh_hz = val
        val = float(cfg.get("warn_interval", _WARN_INTERVAL_DEFAULT))
        if val <= 0:
            logger.warning("backend.py - Backend.__init__：warn_interval=%s 非法（须为正数），沿用默认 %s 秒",
                           val, _WARN_INTERVAL_DEFAULT)
        else:
            self._warn_interval = val
        val = float(cfg.get("write_settle", _WRITE_SETTLE_DEFAULT))
        if val <= 0:
            logger.warning("backend.py - Backend.__init__：write_settle=%s 非法（须为正数），沿用默认 %s 秒",
                           val, _WRITE_SETTLE_DEFAULT)
        else:
            self._write_settle = val
        self._refresh_thread = threading.Thread(target=self._refresh_loop,
                                                name="backend-state-refresh", daemon=True)
        self._refresh_thread.start()

    # ============================================================
    # 生命周期管理（连接与使能）
    # ============================================================
    # =============== 连接（connect / disconnect） ===============
    def connect_arm(self, channel: Optional[str] = None, protocol: Optional[str] = None) -> None:
        """连接 arm 硬件总线
        
        执行前 ``_is_connected_arm`` 置 ``None``；连接成功置 ``True``。

        :param channel: 总线通道；默认取成员 ``_channel_arm``，显式传入则同步更新该成员。
        :param protocol: 协议标识；默认取成员 ``_protocol_arm``，显式传入则同步更新。
        """
        self._connect_impl("arm", channel, protocol)

    def connect_end(self, channel: Optional[str] = None, protocol: Optional[str] = None) -> None:
        """连接 end 硬件总线

        执行前 ``_is_connected_end`` 置 ``None``；连接成功置 ``True``。

        :param channel: 总线通道；默认取成员 ``_channel_end``，显式传入则同步更新该成员。
        :param protocol: 协议标识；默认取成员 ``_protocol_end``，显式传入则同步更新。
        """
        self._connect_impl("end", channel, protocol)

    def disconnect_arm(self) -> None:
        """断开 arm 硬件总线
        
        执行前 ``_is_connected_arm`` 置 ``None``；断开成功置 ``False``。
        """
        self._disconnect_impl("arm")

    def disconnect_end(self) -> None:
        """断开 end 硬件总线
        
        执行前 ``_is_connected_end`` 置 ``None``；断开成功置 ``False``。
        """
        self._disconnect_impl("end")

    # ================= 使能（enable / disable） =================
    def enable_arm(self) -> int:
        """使能 arm 全部关节电机（仅指令发送无回读验证）

        逐关节调用子类实现；执行前 ``_is_abled_arm`` 置 ``None``，全部成功置 ``True``。

        :return: 全部使能成功返回 1；任一失败（warn 提示）返回 0。
        """
        return self._enable_impl("arm")

    def enable_end(self) -> int:
        """使能 end 全部关节电机（仅指令发送无回读验证）

        逐关节调用子类实现；执行前 ``_is_abled_end`` 置 ``None``，全部成功置 ``True``。

        :return: 全部使能成功返回 1；任一失败（warn 提示）返回 0。
        """
        return self._enable_impl("end")

    def disable_arm(self) -> int:
        """失能 arm 全部关节电机（仅指令发送无回读验证）

        逐关节调用子类实现；执行前 ``_is_abled_arm`` 置 ``None``，全部成功置 ``False``。

        :return: 全部失能成功返回 1；任一失败（warn 提示）返回 0。
        """
        return self._disable_impl("arm")

    def disable_end(self) -> int:
        """失能 end 全部关节电机（仅指令发送无回读验证）

        逐关节调用子类实现；执行前 ``_is_abled_end`` 置 ``None``，全部成功置 ``False``。

        :return: 全部失能成功返回 1；任一失败（warn 提示）返回 0。
        """
        return self._disable_impl("end")

    # ==================== 共用实现（arm / end） ====================
    def _connect_impl(self, family: str, channel: Optional[str], protocol: Optional[str]) -> None:
        """连接的共用实现：同步 channel/protocol 成员 → 置 ``None`` → 调用子类实现 → 成功置 ``True``。"""
        if self._skip_empty_family(family, f"connect_{family}"):
            return
        if channel is not None:
            setattr(self, f"_channel_{family}", channel)
        if protocol is not None:
            setattr(self, f"_protocol_{family}", protocol)
        setattr(self, f"_is_connected_{family}", None)  # 执行前置 None；中途意外则停留于此
        try:
            ok = getattr(self, f"_connect_{family}")(getattr(self, f"_channel_{family}"),
                                                     getattr(self, f"_protocol_{family}"))
        except Exception as e:
            logger.warning("backend.py - Backend.connect_%s：连接内核异常：%s", family, e)
            return
        if ok:
            setattr(self, f"_is_connected_{family}", True)
            logger.info("backend.py - Backend.connect_%s：连接成功（channel=%s, protocol=%s）",
                        family, getattr(self, f"_channel_{family}"), getattr(self, f"_protocol_{family}"))
            if self._set_mode_impl(family, ControlMode.MIT) != 1:  # 连接成功后自动尝试设默认 MIT 模式
                logger.warning("backend.py - Backend.connect_%s：默认模式（MIT）设置失败", family)
        else:
            logger.warning("backend.py - Backend.connect_%s：连接失败（channel=%s, protocol=%s）",
                           family, getattr(self, f"_channel_{family}"), getattr(self, f"_protocol_{family}"))

    def _disconnect_impl(self, family: str) -> None:
        """断开连接的共用实现：先尽力失能 → 置 ``None`` → 调用子类实现 → 成功置 ``False``。"""
        if self._skip_empty_family(family, f"disconnect_{family}"):
            return
        if getattr(self, f"_is_connected_{family}") is False:
            return
        if getattr(self, f"_is_abled_{family}") is not False:  # 先失能：未确知已失能才发帧；失败仅提示，不阻断断连
            self._disable_impl(family)
            sleep_time = getattr(self, f"_write_settle") or _WRITE_SETTLE_DEFAULT
            time.sleep(sleep_time)
        setattr(self, f"_is_connected_{family}", None)  # 执行前置 None；中途意外则停留于此
        setattr(self, f"_is_abled_{family}", None)      # 连接已断，使能状态无法核实
        try:
            getattr(self, f"_disconnect_{family}")()
        except Exception as e:
            logger.warning("backend.py - Backend.disconnect_%s：断连内核异常：%s", family, e)
            return
        setattr(self, f"_is_connected_{family}", False)
        setattr(self, f"_mode_{family}", None)  # 模式无法核实，清空防止过期模式放行错误指令
        setattr(self, f"_joint_mode_{family}", [None] * self._n(family))
        logger.info("backend.py - Backend.disconnect_%s：断连成功", family)

    def _enable_impl(self, family: str) -> int:
        """使能的共用实现: 置 ``None`` → 逐关节调用子类实现 → 全部成功置 ``True``。"""
        if self._skip_empty_family(family, f"enable_{family}"):
            return 1
        if not self._require_connected(family, f"enable_{family}"):
            return 0
        setattr(self, f"_is_abled_{family}", None)  # 执行前置 None；中途意外则停留于此
        ok = True
        for i in range(self._n(family)):
            try:
                if not getattr(self, f"_enable_joint_{family}")(i):
                    ok = False
                    logger.warning("backend.py - Backend.enable_%s：joint『%s』使能失败",
                                   family, self._jname(family, i))
            except Exception as e:
                ok = False
                logger.warning("backend.py - Backend.enable_%s：joint『%s』使能异常：%s",
                               family, self._jname(family, i), e)
        if ok:
            setattr(self, f"_is_abled_{family}", True)
            logger.info("backend.py - Backend.enable_%s：全部 joint 使能成功", family)
        return 1 if ok else 0

    def _disable_impl(self, family: str) -> int:
        """失能的共用实现（仅指令发送无回读验证）: 置 ``None`` → 逐关节调用子类实现 → 全部成功置 ``False``。"""
        if self._skip_empty_family(family, f"disable_{family}"):
            return 1
        if not self._require_connected(family, f"disable_{family}"):
            return 0
        setattr(self, f"_is_abled_{family}", None)  # 执行前置 None；中途意外则停留于此
        ok = True
        for i in range(self._n(family)):
            try:
                if not getattr(self, f"_disable_joint_{family}")(i):
                    ok = False
                    logger.warning("backend.py - Backend.disable_%s：joint『%s』失能失败",
                                   family, self._jname(family, i))
            except Exception as e:
                ok = False
                logger.warning("backend.py - Backend.disable_%s：joint『%s』失能异常：%s",
                               family, self._jname(family, i), e)
        if ok:
            setattr(self, f"_is_abled_{family}", False)
            logger.info("backend.py - Backend.disable_%s：全部 joint 失能成功", family)
        return 1 if ok else 0

    # =================== 抽象方法（子类实现） ===================
    @abstractmethod
    def _connect_arm(self, channel, protocol) -> bool:
        """arm 连接抽象方法（子类实现）
        
        打开 ``channel`` 总线（协议按 ``protocol`` 识别；子类协议固定时也可用自带默认）。

        :return: 成功 ``True``；失败返回 ``False`` 或上抛异常。
        """

    @abstractmethod
    def _connect_end(self, channel, protocol) -> bool:
        """end 连接抽象方法（子类实现）
        
        打开 ``channel`` 总线（协议按 ``protocol`` 识别；与 arm 同 channel 时共享总线）。

        :return: 成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
    
    @abstractmethod
    def _disconnect_arm(self) -> None:
        """arm 断开连接抽象方法（子类实现）
        
        关闭子类协议层自建的资源（总线接收线程、串口/CAN 句柄等）。
        """

    @abstractmethod
    def _disconnect_end(self) -> None:
        """end 断开连接抽象方法（子类实现）
        
        关闭子类协议层自建的资源（总线接收线程、串口/CAN 句柄等）。
        """

    @abstractmethod
    def _enable_joint_arm(self, i: int) -> bool:
        """arm 单关节使能抽象方法（子类实现）, 失败上抛或返回 ``False``。"""

    @abstractmethod
    def _enable_joint_end(self, i: int) -> bool:
        """end 单电机的使能抽象方法（子类实现）, 失败上抛或返回 ``False``。"""

    @abstractmethod
    def _disable_joint_arm(self, i: int) -> bool:
        """arm 单关节失能抽象方法（子类实现）。失败上抛或返回 ``False``。"""

    @abstractmethod
    def _disable_joint_end(self, i: int) -> bool:
        """end 单电机的失能抽象方法（子类实现）, 失败上抛或返回 ``False``。"""

    # ============================================================
    # 失能态读取
    # ============================================================
    def read_param_arm(self, key: str) -> Optional[list]:
        """读 arm 全部关节的电机参数（参数键表由子类定义）。

        :param key: 参数名（子类映射到厂商寄存器）。
        :return: 逐关节参数值列表；失败返回 ``None``。
        """
        return self._read_params_impl("arm", key)

    def read_param_end(self, key: str) -> Optional[list]:
        """读 end 全部电机的电机参数（参数键表由子类定义）。

        :param key: 参数名（子类映射到厂商寄存器）。
        :return: 逐电机参数值列表；失败返回 ``None``。
        """
        return self._read_params_impl("end", key)

    def get_mode_arm(self) -> Union[ControlMode, None]:
        """读 arm 全部关节的控制模式，并存到 ``_joint_mode_arm``。

        :return: 各关节模式一致：更新 ``_mode_arm``并返回该模式；不一致或读取失败：返回 ``None``。
        """
        return self._get_mode_impl("arm")

    def get_mode_end(self) -> Union[ControlMode, None]:
        """读 end 全部电机的控制模式，并存到 ``_joint_mode_end``。

        :return: 各电机模式一致：更新 ``_mode_end``并返回该模式；不一致或读取失败：返回 ``None``。
        """
        return self._get_mode_impl("end")

    def get_state_arm(self) -> Optional[JointState]:
        """读 arm 全部关节的状态，整体写入并记时间戳。

        子类未提供的量（返回的字典缺该键）该字段整组置 ``None``。

        :return: 成功返回 ``_joint_state_arm``（本次装配并原子换入的对象，``t`` 为本次更新时刻）；任一关节读取失败、
            或已提供的字段存在空值（数据异常）时，跳过本轮更新（限频 warn）并返回 ``None``。
        """
        return self._get_state_impl("arm")

    def get_state_end(self) -> Optional[JointState]:
        """读 end 全部电机的状态，已提供的字段全部读齐后整体写入并记录时间戳。

        子类未提供的量（返回的字典缺该键）该字段整组置 ``None``。

        :return: 成功返回 ``_joint_state_end``（本次装配并原子换入的对象，``t`` 为本次更新时刻）；任一电机读取失败、
            或已提供的字段存在空值（数据异常）时，跳过本轮更新（限频 warn）并返回 ``None``。
        """
        return self._get_state_impl("end")

    # ==================== 共用实现（arm / end） ====================
    def _read_params_impl(self, family: str, key: str) -> Optional[list]:
        """读参数的共用实现：逐关节读寄存器 → 值列表（任一失败则返回 ``None``）。"""
        if self._skip_empty_family(family, f"read_param_{family}"):
            return []
        if not self._require_connected(family, f"read_param_{family}"):
            return None
        vals = []
        for i in range(self._n(family)):
            try:
                vals.append(getattr(self, f"_read_joint_param_{family}")(i, key))
            except Exception as e:
                logger.warning("backend.py - Backend.read_param_%s：joint『%s』读『%s』失败：%s",
                               family, self._jname(family, i), key, e)
                return None
        if len(vals) != self._n(family):  # 返回值维度校验（防子类误返回序列而非单值）
            logger.warning("backend.py - Backend.read_param_%s：返回维度 %d ≠ 关节数 %d",
                           family, len(vals), self._n(family))
            return None
        return vals

    def _get_mode_impl(self, family: str) -> Union[ControlMode, None]:
        """读模式的共用实现：先全部读齐再整体赋值。

        全组一致时更新 ``_mode_{family}``并返回该模式；不一致或读取失败返回 ``None``（不赋值、不更新成员）。
        """
        if self._skip_empty_family(family, f"get_mode_{family}"):
            return None
        if not self._require_connected(family, f"get_mode_{family}"):
            return None
        modes = []
        for i in range(self._n(family)):
            try:
                modes.append(getattr(self, f"_read_joint_mode_{family}")(i))
            except Exception as e:
                logger.warning("backend.py - Backend.get_mode_%s：joint『%s』读模式失败：%s",
                               family, self._jname(family, i), e)
                return None
        if any(m is None for m in modes):
            logger.warning("backend.py - Backend.get_mode_%s：存在未读到模式的 joint，整族不更新", family)
            return None
        setattr(self, f"_joint_mode_{family}", list(modes))  # 逐关节写入各自模式
        if len(set(modes)) == 1:
            setattr(self, f"_mode_{family}", modes[0])
            return modes[0]
        logger.warning("backend.py - Backend.get_mode_%s：各 joint 模式不一致（%s），族模式成员不更新",
                       family, modes)
        return None

    def _get_state_impl(self, family: str) -> Optional[JointState]:
        """读状态的共用实现：已提供的字段全部读齐后构造完整状态对象，单一原子赋值换入。

        子类返回的字典**缺某个键** = 子类/硬件不提供该量，该字段整组置 ``None``（合法的缺席）；
        键**存在但值为 ``None`` 或不可解析** = 数据异常，跳过本轮赋值并限频 warn，返回 ``None``。

        return: 成功返回 ``_joint_state_{family}``（本次装配并原子换入的对象，``t`` 为本次更新时刻）；任一关节读取失败、
            或已提供的字段存在空值（数据异常）时，跳过本轮更新（限频 warn）并返回 ``None``。
        """
        if self._skip_empty_family(family, f"get_state_{family}"):
            return None
        if not self._require_connected(family, f"get_state_{family}"):
            return None
        snaps = []
        for i in range(self._n(family)):
            try:
                snaps.append(getattr(self, f"_read_joint_state_{family}")(i) or {})
            except Exception as e:
                logger.warning("backend.py - Backend.get_state_%s：joint『%s』读状态失败：%s",
                               family, self._jname(family, i), e)
                return None
        names = ("q", "dq", "tau", "temp_mos", "temp_rotor", "error")
        absent = {name for name in names if any(name not in snap for snap in snaps)}  # 子类未提供该量 → 字段置 None
        bad = {name for name in names if name not in absent
               and any(snap.get(name) is None for snap in snaps)}                    # 键存在但值为空 → 数据异常
        if bad:
            self._warn_throttled(
                "backend.py - Backend.get_state_%s：在场子值 %s 存在空值（数据异常），本轮状态跳过更新（保持上次快照）",
                family, sorted(bad))
            return None
        present = tuple(name for name in names if name not in absent)
        try:  # 先解析后提交：任一不可解析按数据异常跳过本轮，不产生部分赋值
            assembled = {name: np.asarray([snap[name] for snap in snaps], dtype=int if name == "error" else float)
                         for name in present}
        except (TypeError, ValueError) as e:
            self._warn_throttled("backend.py - Backend.get_state_%s：子值解析失败（数据异常）：%s", family, e)
            return None
        # 全部字段备齐后构造完整状态对象，单一赋值换入
        fields = {name: None for name in absent}
        fields.update(assembled)  # error 整数异常码，其余物理量 float
        new_state = JointState(t=time.time(), **fields)  # t=本次更新时刻
        setattr(self, f"_joint_state_{family}", new_state)
        return new_state

    # ======================= 低频状态刷新 =======================
    def _refresh_loop(self) -> None:
        """低频状态刷新线程体（daemon，``__init__`` 启动，进程退出自动结束）。

        每周期（``1/_refresh_hz`` 秒）检查 arm/end 两组：未连接（含未知）跳过；``t`` 仍新鲜
        （距上次更新不足半周期，说明有更高频的读取在更新）跳过；否则刷新状态——仅当逐关节
        模式缓存 ``_joint_mode_{family}`` 还有未读取的关节才先 ``get_mode_*``，随后 ``get_state_*``。
        """
        period = 1.0 / self._refresh_hz
        fresh = 0.5 * period  # 新鲜阈值：更新滞后不足半周期即视为正被更高频的读取更新
        while not self._refresh_stop.wait(period):
            now = time.time()
            for family in ("arm", "end"):
                if not getattr(self, f"_is_connected_{family}"):
                    continue
                if now - getattr(self, f"_joint_state_{family}").t < fresh:
                    continue
                if any(m is None for m in getattr(self, f"_joint_mode_{family}")):
                    getattr(self, f"get_mode_{family}")()
                getattr(self, f"get_state_{family}")()

    # =================== 抽象方法（子类实现） ===================
    @abstractmethod
    def _read_joint_param_arm(self, i: int, key: str):
        """arm 单关节读参数抽象方法（子类实现）：读该电机 ``key`` 对应寄存器，返回参数值。失败上抛。"""

    @abstractmethod
    def _read_joint_param_end(self, i: int, key: str):
        """end 单电机读参数抽象方法（子类实现）：读该电机 ``key`` 对应寄存器，返回参数值。失败上抛。"""

    @abstractmethod
    def _read_joint_mode_arm(self, i: int) -> Optional[ControlMode]:
        """arm 单关节读模式抽象方法（子类实现）：读该电机当前控制模式。读不到返回 ``None``，失败上抛。"""

    @abstractmethod
    def _read_joint_mode_end(self, i: int) -> Optional[ControlMode]:
        """end 单电机读模式抽象方法（子类实现）：读该电机当前控制模式。读不到返回 ``None``，失败上抛。"""

    @abstractmethod
    def _read_joint_state_arm(self, i: int) -> dict:
        """arm 单关节读状态抽象方法（子类实现）：返回该关节当前状态量的字典。

        键可含 ``q`` / ``dq`` / ``tau`` / ``temp_mos`` / ``temp_rotor`` / ``error``（rad / rad/s / N·m / ℃ / ℃ / 状态码）。
        硬件不提供的量**直接缺键**（字典不含该键）；键存在但值为 ``None`` 属数据获取异常。
        ``error`` 全库约定：``0``=失能、``1``=使能（均正常）、``≥2``=故障码（子类负责映射厂商原始码）。

        return: 成功返回字典（部分缺值置none）；异常上抛。
        """

    @abstractmethod
    def _read_joint_state_end(self, i: int) -> dict:
        """end 单电机读状态抽象方法（子类实现）：返回该电机当前状态量的字典。

        键可含 ``q`` / ``dq`` / ``tau`` / ``temp_mos`` / ``temp_rotor`` / ``error``（rad / rad/s / N·m / ℃ / ℃ / 状态码）。
        硬件不提供的量**直接缺键**（字典不含该键）；键存在但值为 ``None`` 属数据获取异常。
        ``error`` 全库约定：``0``=失能、``1``=使能（均正常）、``≥2``=故障码（子类负责映射厂商原始码）。

        return: 成功返回字典（部分缺值置none）；异常上抛。
        """

    # ============================================================
    # 失能态写入
    # ============================================================
    def write_param_arm(self, key: str, values) -> int:
        """写 arm 全部关节的电机参数（参数键表由子类定义）。

        :param key: 参数名（子类映射到厂商寄存器）。
        :param values: 参数值列表（长度 = 关节数）。
        :return: 成功返回 1；维度不符（warn 并跳过本次）或任一失败返回 0。
        """
        return self._write_param_impl("arm", key, values)

    def write_param_end(self, key: str, values) -> int:
        """写 end 全部电机的电机参数（参数键表由子类定义）。

        :param key: 参数名（子类映射到厂商寄存器）。
        :param values: 参数值列表（长度 = 电机关节数）。
        :return: 成功返回 1；维度不符（warn 并跳过本次）或任一失败返回 0。
        """
        return self._write_param_impl("end", key, values)

    def set_zero_arm(self) -> int:
        """arm 逐关节设置位置零点。

        :return: 全部成功返回 1；任一失败（warn 提示）返回 0。
        """
        return self._set_zero_impl("arm")

    def set_zero_end(self) -> int:
        """end 逐电机设置位置零点。

        :return: 全部成功返回 1；任一失败（warn 提示）返回 0。
        """
        return self._set_zero_impl("end")

    def set_mode_arm(self, mode: ControlMode = ControlMode.MIT) -> int:
        """arm 逐关节设置控制模式。

        :param mode: 目标控制模式，默认 :class:`ControlMode.MIT`（默认模式）。
        :return: 成功返回 1；否则（warn 提示）返回 0。
        """
        return self._set_mode_impl("arm", mode)

    def set_mode_end(self, mode: ControlMode = ControlMode.MIT) -> int:
        """end 逐电机设置控制模式。

        :param mode: 目标控制模式，默认 :class:`ControlMode.MIT`（默认模式）。
        :return: 成功返回 1；否则（warn 提示）返回 0。
        """
        return self._set_mode_impl("end", mode)

    # ==================== 共用实现（arm / end） ====================
    def _write_param_impl(self, family: str, key: str, values) -> int:
        """写参数的共用实现：维度自检 → 逐关节全部写入。
        """
        n = self._n(family)
        vals = self._to_float_arr(values, f"write_param_{family}")
        if vals is None:
            return 0
        if vals.shape[0] != n:
            logger.warning("backend.py - Backend.write_param_%s：『%s』值列表维度 %d ≠ 关节数 %d，本次写入跳过",
                           family, key, vals.shape[0], n)
            return 0
        if self._skip_empty_family(family, f"write_param_{family}"):
            return 1
        if not self._require_connected(family, f"write_param_{family}"):
            return 0
        for i in range(n):
            try:
                getattr(self, f"_write_joint_param_{family}")(i, key, vals[i])
            except Exception as e:
                logger.warning("backend.py - Backend.write_param_%s：joint『%s』写『%s』失败：%s",
                               family, self._jname(family, i), key, e)
                return 0
        return 1

    def _set_zero_impl(self, family: str) -> int:
        """设零点的共用实现：逐关节设置位置零点，任一失败即返回 0。"""
        if self._skip_empty_family(family, f"set_zero_{family}"):
            return 1
        if not self._require_connected(family, f"set_zero_{family}"):
            return 0
        for i in range(self._n(family)):
            try:
                if not getattr(self, f"_set_joint_zero_{family}")(i):
                    logger.warning("backend.py - Backend.set_zero_%s：joint『%s』设零失败",
                                   family, self._jname(family, i))
                    return 0
            except Exception as e:
                logger.warning("backend.py - Backend.set_zero_%s：joint『%s』设零异常：%s",
                               family, self._jname(family, i), e)
                return 0
        logger.info("backend.py - Backend.set_zero_%s：全部 joint 设零成功", family)
        return 1

    def _set_mode_impl(self, family: str, mode: ControlMode) -> int:
        """设模式的共用实现：增益检查（缺配置仅 warn，改用电机内部增益）→ 逐关节设置 → 读回全部核对（经 ``get_mode_*``）。"""
        if self._skip_empty_family(family, f"set_mode_{family}"):
            return 1
        if not self._require_connected(family, f"set_mode_{family}"):
            return 0
        for i in range(self._n(family)):
            try:
                getattr(self, f"_set_joint_mode_{family}")(i, mode)
            except Exception as e:
                logger.warning("backend.py - Backend.set_mode_%s：joint『%s』设置失败：%s",
                               family, self._jname(family, i), e)
                return 0
        return 1

    # =================== 抽象方法（子类实现） ===================
    @abstractmethod
    def _write_joint_param_arm(self, i: int, key: str, value: float) -> None:
        """arm 单关节写参数抽象方法（子类实现）：写该电机 ``key`` 寄存器为 ``value``。失败上抛。"""

    @abstractmethod
    def _write_joint_param_end(self, i: int, key: str, value: float) -> None:
        """end 单电机写参数抽象方法（子类实现）：写该电机 ``key`` 寄存器为 ``value``。失败上抛。"""

    @abstractmethod
    def _set_joint_mode_arm(self, i: int, mode: ControlMode) -> None:
        """arm 单关节设模式抽象方法（子类实现）：写该电机的模式寄存器，并配置所需的增益寄存器。失败上抛。

        切到 POSITION / VELOCITY 时须一并写入 cfg ``POS_VEL`` 的四个增益寄存器
        （``pos_kp``/``pos_ki``/``vel_kp``/``vel_ki``，取值自基类成员 ``_pos_kp_arm`` 等）。
        """

    @abstractmethod
    def _set_joint_mode_end(self, i: int, mode: ControlMode) -> None:
        """end 单电机设模式抽象方法（子类实现）：写该电机的模式寄存器，并配置所需的增益寄存器。失败上抛。

        切到 POSITION / VELOCITY 时须一并写入 cfg ``POS_VEL`` 的四个增益寄存器
        （``pos_kp``/``pos_ki``/``vel_kp``/``vel_ki``，取值自基类成员 ``_pos_kp_end`` 等）。
        """

    @abstractmethod
    def _set_joint_zero_arm(self, i: int) -> bool:
        """arm 单关节设零抽象方法（子类实现）：将当前位置记为零点。成功 ``True``，失败 ``False`` 或上抛。"""

    @abstractmethod
    def _set_joint_zero_end(self, i: int) -> bool:
        """end 单电机设零抽象方法（子类实现）：将当前位置记为零点。成功 ``True``，失败 ``False`` 或上抛。"""

    # ============================================================
    # 使能态指令发送
    # ============================================================
    def send_mit_arm(self, tau, q, dq, kp=None, kd=None) -> None:
        """arm MIT 指令（电机内部 ``τ = tau + kp·(q_d−q) + kd·(dq_d−dq)``），整组下发。

        :param tau: 前馈力矩 ``(n,)``，N·m。
        :param q: 位置目标 ``(n,)``，rad。
        :param dq: 速度目标 ``(n,)``，rad/s。
        :param kp: 位置增益 ``(n,)``；缺省用 cfg ``MIT.kp`` 提取的默认值。
        :param kd: 速度阻尼 ``(n,)``；缺省用 cfg ``MIT.kd`` 提取的默认值。
        """
        if kp is None:
            kp = self._kp_mit_default_arm
        if kd is None:
            kd = self._kd_mit_default_arm
        self._send_mit_impl("arm", tau, q, dq, kp, kd)

    def send_position_arm(self, q, vlim=None, flim=None) -> None:
        """arm 位置指令整组下发。

        :param q: 位置目标 ``(n,)``，rad。
        :param vlim: 速度上限 ``(n,)``，rad/s；缺省用 cfg ``POS_VEL.vlim`` 提取的默认值。
        :param flim: 归一化力矩电流上限 ``(n,)``（0~1）；缺省用 cfg ``POS_VEL.flim`` 提取的默认值。
        """
        if vlim is None:
            vlim = self._vlim_default_arm
        if flim is None:
            flim = self._flim_default_arm
        self._send_position_impl("arm", q, vlim, flim)

    def send_vel_arm(self, dq) -> None:
        """arm 速度指令整组下发。

        :param dq: 速度目标 ``(n,)``，rad/s。
        """
        self._send_vel_impl("arm", dq)

    def send_mit_end(self, tau, q, dq, kp=None, kd=None) -> None:
        """end MIT 指令（电机内部 ``τ = tau + kp·(q_d−q) + kd·(dq_d−dq)``），整组下发。

        :param tau: 前馈力矩 ``(n,)``，N·m。
        :param q: 位置目标 ``(n,)``，rad。
        :param dq: 速度目标 ``(n,)``，rad/s。
        :param kp: 位置增益 ``(n,)``；缺省用 cfg ``MIT.kp`` 提取的默认值。
        :param kd: 速度阻尼 ``(n,)``；缺省用 cfg ``MIT.kd`` 提取的默认值。
        """
        if kp is None:
            kp = self._kp_mit_default_end
        if kd is None:
            kd = self._kd_mit_default_end
        self._send_mit_impl("end", tau, q, dq, kp, kd)

    def send_position_end(self, q, vlim=None, flim=None) -> None:
        """end 位置指令整组下发。

        :param q: 位置目标 ``(n,)``，rad。
        :param vlim: 速度上限 ``(n,)``，rad/s；缺省用 cfg ``POS_VEL.vlim`` 提取的默认值。
        :param flim: 归一化力矩电流上限 ``(n,)``（0~1）；缺省用 cfg ``POS_VEL.flim`` 提取的默认值。
        """
        if vlim is None:
            vlim = self._vlim_default_end
        if flim is None:
            flim = self._flim_default_end
        self._send_position_impl("end", q, vlim, flim)

    def send_vel_end(self, dq) -> None:
        """end 速度指令整组下发。

        :param dq: 速度目标 ``(n,)``，rad/s。
        """
        self._send_vel_impl("end", dq)

    def send_action_end(self, action: str, **kwargs) -> None:
        """end 离散动作（子类定义）。

        :param action: 动作名，常见 ``"open"`` / ``"close"`` / ``"home"``。
        :param kwargs: 动作的可选配置（由子类解释）。
        """
        if self._skip_empty_family("end", "send_action_end"):
            return
        if not self._require_connected("end", "send_action_end"):
            return
        self._call("send_action_end", self._send_action_end, action, **kwargs)

    # ==================== 共用实现（arm / end） ====================
    def _send_mit_impl(self, family: str, tau, q, dq, kp, kd) -> None:
        """MIT 发送的共用实现：①维度校验 → ②空值检查+越限裁剪 → ③连接/模式检查 → ④逐关节调用子类实现下发。"""
        if self._skip_empty_family(family, f"send_mit_{family}"):
            return
        # ① 维度校验（不符 → warn 跳过本次发送）
        n = self._n(family)
        arrs = [self._to_float_arr(x, f"send_mit_{family}") for x in (tau, q, dq, kp, kd)]
        if any(a is None for a in arrs):
            return
        tau, q, dq, kp, kd = arrs
        if len(tau) != n or len(q) != n or len(dq) != n or len(kp) != n or len(kd) != n:
            self._warn_throttled(
                "backend.py - Backend.send_mit_%s：指令维度（tau=%d, q=%d, dq=%d, kp=%d, kd=%d）≠ 关节数 %d，本次发送跳过",
                family, len(tau), len(q), len(dq), len(kp), len(kd), n)
            return
        # ② 空值检查（有空值 → warn 不发送）+ 裁剪（越限 → 就近裁剪 + 限频告警）
        if not self._require_present(f"send_mit_{family}", tau=tau, q=q, dq=dq, kp=kp, kd=kd):
            return
        lim = getattr(self, f"_joint_limits_{family}")  # 空组已在前跳过，限位必已解析
        clipped = np.clip(tau, -lim.tau_max, lim.tau_max)
        if not np.array_equal(tau, clipped):
            self._warn_throttled("backend.py - Backend.send_mit_%s：tau 指令越限，已就近裁剪 %s → %s",
                                 family, np.round(tau, 4).tolist(), np.round(clipped, 4).tolist())
            tau = clipped
        clipped = np.clip(q, lim.q_min, lim.q_max)
        if not np.array_equal(q, clipped):
            self._warn_throttled("backend.py - Backend.send_mit_%s：q 指令越限，已就近裁剪 %s → %s",
                                 family, np.round(q, 4).tolist(), np.round(clipped, 4).tolist())
            q = clipped
        clipped = np.clip(dq, -lim.dq_max, lim.dq_max)
        if not np.array_equal(dq, clipped):
            self._warn_throttled("backend.py - Backend.send_mit_%s：dq 指令越限，已就近裁剪 %s → %s",
                                 family, np.round(dq, 4).tolist(), np.round(clipped, 4).tolist())
            dq = clipped
        # ③ 连接/模式前置检查 → ④ 逐关节调用子类实现下发
        if not self._require_connected(family, f"send_mit_{family}"):
            return
        if not self._require_mode(family, ControlMode.MIT, f"send_mit_{family}"):
            return
        for i in range(n):
            self._call(f"send_mit_{family}（{self._jname(family, i)}）",
                       getattr(self, f"_send_joint_mit_{family}"),
                       i, tau[i], q[i], dq[i], kp[i], kd[i])

    def _send_position_impl(self, family: str, q, vlim, flim) -> None:
        """位置发送的共用实现：①维度校验 → ②空值检查+限幅裁剪 → ③连接/模式检查 → ④逐关节调用子类实现下发。"""
        if self._skip_empty_family(family, f"send_position_{family}"):
            return
        # ① 维度校验（不符 → warn 跳过本次发送）
        n = self._n(family)
        arrs = [self._to_float_arr(x, f"send_position_{family}") for x in (q, vlim, flim)]
        if any(a is None for a in arrs):
            return
        q, vlim, flim = arrs
        if len(q) != n or len(vlim) != n or len(flim) != n:
            self._warn_throttled(
                "backend.py - Backend.send_position_%s：指令维度（q=%d, vlim=%d, flim=%d）≠ 关节数 %d，本次发送跳过",
                family, len(q), len(vlim), len(flim), n)
            return
        # ② 空值检查（有空值 → warn 不发送）+ 裁剪（越限 → 就近裁剪 + 限频告警）
        if not self._require_present(f"send_position_{family}", q=q, vlim=vlim, flim=flim):
            return
        lim = getattr(self, f"_joint_limits_{family}")  # 空组已在前跳过，限位必已解析
        clipped = np.clip(q, lim.q_min, lim.q_max)
        if not np.array_equal(q, clipped):
            self._warn_throttled("backend.py - Backend.send_position_%s：q 指令越限，已就近裁剪 %s → %s",
                                 family, np.round(q, 4).tolist(), np.round(clipped, 4).tolist())
            q = clipped
        clipped = np.clip(vlim, 0.0, lim.dq_max)
        if not np.array_equal(vlim, clipped):
            self._warn_throttled("backend.py - Backend.send_position_%s：vlim 超出 [0, dq_max]，已就近裁剪 %s → %s",
                                 family, np.round(vlim, 4).tolist(), np.round(clipped, 4).tolist())
            vlim = clipped
        clipped = np.clip(flim, 0.0, 1.0)
        if not np.array_equal(flim, clipped):
            self._warn_throttled("backend.py - Backend.send_position_%s：flim 超出归一化范围 [0, 1]，已就近裁剪 %s → %s",
                                 family, np.round(flim, 4).tolist(), np.round(clipped, 4).tolist())
            flim = clipped
        # ③ 连接/模式前置检查 → ④ 逐关节调用子类实现下发
        if not self._require_connected(family, f"send_position_{family}"):
            return
        if not self._require_mode(family, ControlMode.POSITION, f"send_position_{family}"):
            return
        for i in range(n):
            self._call(f"send_position_{family}（{self._jname(family, i)}）",
                       getattr(self, f"_send_joint_position_{family}"),
                       i, q[i], vlim[i], flim[i])

    def _send_vel_impl(self, family: str, dq) -> None:
        """速度发送的共用实现：①维度校验 → ②空值检查+幅值裁剪 → ③连接/模式检查 → ④逐关节调用子类实现下发。"""
        if self._skip_empty_family(family, f"send_vel_{family}"):
            return
        # ① 维度校验（不符 → warn 跳过本次发送）
        n = self._n(family)
        dq = self._to_float_arr(dq, f"send_vel_{family}")
        if dq is None:
            return
        if len(dq) != n:
            self._warn_throttled("backend.py - Backend.send_vel_%s：指令维度（dq=%d）≠ 关节数 %d，本次发送跳过",
                                 family, len(dq), n)
            return
        # ② 空值检查（有空值 → warn 不发送）+ 裁剪（越限 → 就近裁剪 + 限频告警）
        if not self._require_present(f"send_vel_{family}", dq=dq):
            return
        lim = getattr(self, f"_joint_limits_{family}")  # 空组已在前跳过，限位必已解析
        clipped = np.clip(dq, -lim.dq_max, lim.dq_max)
        if not np.array_equal(dq, clipped):
            self._warn_throttled("backend.py - Backend.send_vel_%s：dq 指令越限，已就近裁剪 %s → %s",
                                 family, np.round(dq, 4).tolist(), np.round(clipped, 4).tolist())
            dq = clipped
        # ③ 连接/模式前置检查 → ④ 逐关节调用子类实现下发
        if not self._require_connected(family, f"send_vel_{family}"):
            return
        if not self._require_mode(family, ControlMode.VELOCITY, f"send_vel_{family}"):
            return
        for i in range(n):
            self._call(f"send_vel_{family}（{self._jname(family, i)}）",
                       getattr(self, f"_send_joint_vel_{family}"), i, dq[i])

    # =================== 抽象方法（子类实现） ===================
    @abstractmethod
    def _send_joint_mit_arm(self, i: int, tau: float, q: float, dq: float,
                            kp: float, kd: float) -> None:
        """arm 单关节 MIT 发送抽象方法（子类实现）：发一帧 MIT 指令，入参已经过检查和越限裁剪。失败上抛。"""

    @abstractmethod
    def _send_joint_mit_end(self, i: int, tau: float, q: float, dq: float,
                            kp: float, kd: float) -> None:
        """end 单电机 MIT 发送抽象方法（子类实现）：发一帧 MIT 指令，入参已经过检查和越限裁剪。失败上抛。"""

    @abstractmethod
    def _send_joint_position_arm(self, i: int, q: float, vlim: float, flim: float) -> None:
        """arm 单关节位置发送抽象方法（子类实现）：发一帧位置指令，入参已经过检查和限幅裁剪。失败上抛。"""

    @abstractmethod
    def _send_joint_position_end(self, i: int, q: float, vlim: float, flim: float) -> None:
        """end 单电机位置发送抽象方法（子类实现）：发一帧位置指令，入参已经过检查和限幅裁剪。失败上抛。"""

    @abstractmethod
    def _send_joint_vel_arm(self, i: int, dq: float) -> None:
        """arm 单关节速度发送抽象方法（子类实现）：发一帧速度指令，入参已经过检查和幅值裁剪。失败上抛。"""

    @abstractmethod
    def _send_joint_vel_end(self, i: int, dq: float) -> None:
        """end 单电机速度发送抽象方法（子类实现）：发一帧速度指令，入参已经过检查和幅值裁剪。失败上抛。"""

    @abstractmethod
    def _send_action_end(self, action: str, **kwargs) -> None:
        """end 离散动作抽象方法（子类实现）：按 ``action`` 发送对应指令（动作映射与配置由子类定义）。失败上抛。"""

    # ============================================================
    # 错误与恢复（get_error_* 读码 → check_error_* 判正常 → clear_error_* 失能态清错）
    # ============================================================
    def get_error_arm(self) -> Optional[list]:
        """获取 arm 各关节状态码。

        状态码 0=失能、1=使能，均属正常；其他码为故障（含义随固件/型号定）。

        :return: 逐关节状态码 list（关节按序对应；无关节为空 list）；未连接、状态快照过期或 ``error`` 段缺失返回 ``None``。
        """
        return self._get_error_impl("arm")

    def get_error_end(self) -> Optional[list]:
        """获取 end 各电机状态码。

        状态码 0=失能、1=使能，均属正常；其他码为故障（含义随固件/型号定）。

        :return: 逐电机状态码 list（电机按序对应；无电机为空 list）；未连接、状态快照过期或 ``error`` 段缺失返回 ``None``。
        """
        return self._get_error_impl("end")

    def check_error_arm(self) -> bool:
        """检查 arm 各关节状态码（基于 ``get_error_arm``）。

        状态码 0=失能、1=使能，均属正常；存在其他码故障。

        :return: 状态码正常返回 ``True``；存在故障码、未连接或状态快照过期返回 ``False``。
        """
        return self._check_error_impl("arm")

    def check_error_end(self) -> bool:
        """检查 end 各电机状态码（基于 ``get_error_end``）。

        状态码 0=失能、1=使能，均属正常；存在其他码故障。

        :return: 状态码正常返回 ``True``；存在故障码、未连接或状态快照过期返回 ``False``。
        """
        return self._check_error_impl("end")

    def clear_error_arm(self) -> bool:
        """清除 arm 各关节硬件错误（仅发清错指令，失能状态下执行）。

        :return: 成功返回 ``True``；门禁未过或任一发送失败（warn 提示）返回 ``False``。
        """
        return self._clear_error_impl("arm")

    def clear_error_end(self) -> bool:
        """清除 end 各电机硬件错误（仅发清错指令，失能状态下执行）。

        :return: 成功返回 ``True``；门禁未过或任一发送失败（warn 提示）返回 ``False``。
        """
        return self._clear_error_impl("end")

    # ==================== 共用实现（arm / end） ====================
    def _get_error_impl(self, family: str) -> Optional[list]:
        """读码的共用实现：``_joint_state_{family}`` 的 ``error`` 段。"""
        if self._skip_empty_family(family, f"get_error_{family}"):
            return []
        if not self._require_connected(family, f"get_error_{family}"):
            return None
        st = getattr(self, f"_joint_state_{family}")
        if time.time() - st.t > 2.0 / self._refresh_hz:  # 快照过期
            logger.warning("backend.py - Backend.get_error_%s：状态快照过期（先 get_state_%s 或等刷新），无法获取",
                           family, family)
            return None
        err = st.error
        if err is None:
            logger.warning("backend.py - Backend.get_error_%s：error 段缺失，无法获取", family)
            return None
        return err.tolist()

    def _check_error_impl(self, family: str) -> bool:
        """检查的共用实现：基于读码结果判断，状态码 0=失能 / 1=使能 均正常，出现其他码即故障。"""
        errs = self._get_error_impl(family)
        if errs is None:
            return False
        if any(e not in (0, 1) for e in errs):
            logger.warning("backend.py - Backend.check_error_%s：存在故障状态码（%s）", family, errs)
            return False
        logger.info("backend.py - Backend.check_error_%s：状态码正常（%s）", family, errs)
        return True

    def _clear_error_impl(self, family: str) -> bool:
        """清错的共用实现：逐关节发送清错指令（无静置、无读回验证）。"""
        if self._skip_empty_family(family, f"clear_error_{family}"):
            return True
        if not self._require_connected(family, f"clear_error_{family}"):
            return False
        abled = getattr(self, f"_is_abled_{family}")
        if abled is not False:
            logger.warning("backend.py - Backend.clear_error_%s：%s，清错须在失能状态下执行（先 disable_%s），操作跳过",
                           family, "已使能" if abled is True else "使能状态未知", family)
            return False
        for i in range(self._n(family)):
            jn = self._jname(family, i)
            try:
                if not getattr(self, f"_clear_joint_error_{family}")(i):
                    logger.warning("backend.py - Backend.clear_error_%s：joint『%s』清错失败", family, jn)
                    return False
            except Exception as e:
                logger.warning("backend.py - Backend.clear_error_%s：joint『%s』清错异常：%s", family, jn, e)
                return False
        return True

    # =================== 抽象方法（子类实现） ===================
    @abstractmethod
    def _clear_joint_error_arm(self, i: int) -> bool:
        """arm 单关节清错抽象方法（子类实现）：发送该电机错误清除指令。成功 ``True``，失败 ``False`` 或上抛。"""

    @abstractmethod
    def _clear_joint_error_end(self, i: int) -> bool:
        """end 单电机清错抽象方法（子类实现）：发送该电机错误清除指令。成功 ``True``，失败 ``False`` 或上抛。"""

    # ============================================================
    # 外部属性访问（供 joyarm 上层读取内部成员，返回实时引用）
    # ============================================================
    @property
    def cfg(self) -> dict:
        """config 的 ``backend`` 段整体（含 ``name``；实时引用）。"""
        return self._cfg

    @property
    def name(self) -> str:
        """后端名（cfg ``name`` 选型键）。"""
        return self._name

    @property
    def n_joints_arm(self) -> int:
        """arm 关节数。"""
        return self._n_joints_arm

    @property
    def n_joints_end(self) -> int:
        """end 电机关节数（无 end 段为 0）。"""
        return self._n_joints_end

    @property
    def is_connected_arm(self) -> Optional[bool]:
        """arm 连接状态（True 已连接 / False 未连接 / None 未知）。"""
        return self._is_connected_arm

    @property
    def is_connected_end(self) -> Optional[bool]:
        """end 连接状态（True 已连接 / False 未连接 / None 未知）。"""
        return self._is_connected_end

    @property
    def is_abled_arm(self) -> Optional[bool]:
        """arm 使能状态（True 已使能 / False 已失能 / None 未知）。"""
        return self._is_abled_arm

    @property
    def is_abled_end(self) -> Optional[bool]:
        """end 使能状态（True 已使能 / False 已失能 / None 未知）。"""
        return self._is_abled_end

    @property
    def mode_arm(self) -> Optional[ControlMode]:
        """arm 全组一致的控制模式（get_mode 读齐且一致时更新；None=未读取）。"""
        return self._mode_arm

    @property
    def mode_end(self) -> Optional[ControlMode]:
        """end 全组一致的控制模式（get_mode 读齐且一致时更新；None=未读取）。"""
        return self._mode_end

    @property
    def joint_state_arm(self) -> JointState:
        """arm 整组实时关节状态。"""
        return self._joint_state_arm

    @property
    def joint_state_end(self) -> JointState:
        """end 整组实时关节状态。"""
        return self._joint_state_end

    @property
    def joint_limits_arm(self) -> Optional[JointLimits]:
        """arm 硬限位（发送越限裁剪的唯一依据；无 arm 段为 ``None``）。"""
        return self._joint_limits_arm

    @property
    def joint_limits_end(self) -> Optional[JointLimits]:
        """end 硬限位（无 end 段为 ``None``）。"""
        return self._joint_limits_end

    # ============================================================
    # 内部助手（仅基类使用）
    # ============================================================
    def _n(self, family: str) -> int:
        """组（``"arm"`` / ``"end"``）的关节数（读 ``_n_joints_{family}`` 成员）。"""
        return getattr(self, f"_n_joints_{family}")

    def _skip_empty_family(self, family: str, caller: str) -> bool:
        """空组（关节数 0）跳过判定：视为成功直接返回，同时限频 warn 提示未配置任何关节。"""
        if self._n(family) != 0:
            return False
        self._warn_throttled("backend.py - Backend.%s：%s 未配置任何 joint（n=0），本次操作视为成功跳过",
                             caller, family)
        return True

    def _jname(self, family: str, i: int):
        """joint 显示名（取 cfg ``name`` 键，缺省用索引），用于日志。"""
        return getattr(self, f"_jointscfg_{family}")[i].get("name", f"joint{i}")

    def _extract_joint_values(self, family: str, section: str, key: str) -> np.ndarray:
        """逐关节提取 cfg 段键值（``MIT.kp`` / ``POS_VEL.pos_kp`` 等）→ ``(n,)`` 数组。

        缺段/缺键/非数值 → 该关节以 NaN 占位（空值）并 warn；NaN 的发送默认值会在发送时
        被空值检查拦下（不发送），NaN 的增益在 ``set_mode`` 非 MIT 时仅提示改用电机内部增益。
        """
        vals, holes = [], []
        for i, jcfg in enumerate(getattr(self, f"_jointscfg_{family}")):
            try:
                v = float((jcfg.get(section) or {}).get(key))
            except (TypeError, ValueError):
                v = np.nan
            if not np.isfinite(v):
                holes.append(self._jname(family, i))
                v = np.nan
            vals.append(v)
        if holes:
            logger.warning("backend.py - Backend.__init__：%s『%s.%s』存在空值（%s），使用处将被相应门禁拦截",
                           family, section, key, "、".join(holes))
        return np.asarray(vals, dtype=float)

    def _require_connected(self, family: str, caller: str) -> bool:
        """连接前置检查：未连接或状态未知（``None``）时 warn 并返回 ``False``。"""
        if not getattr(self, f"_is_connected_{family}"):
            logger.warning("backend.py - Backend.%s：%s 未连接或状态未知，操作跳过", caller, family)
            return False
        return True

    def _require_mode(self, family: str, mode: ControlMode, caller: str) -> bool:
        """模式前置检查：组模式成员 ``_mode_{family}`` 等于所需模式才放行，否则限频 warn。

        ``_mode_{family}`` 与逐关节模式缓存 ``_joint_mode_{family}`` 同步更新
        （``get_mode`` 读齐且一致时），为 ``None`` 即模式未读取。
        """
        cur = getattr(self, f"_mode_{family}")
        if cur == mode:
            return True
        self._warn_throttled("backend.py - Backend.%s：%s 当前模式 %s ≠ 所需 %s（先 set_mode/get_mode）",
                             caller, family, cur if cur is not None else "未读取", mode)
        return False

    def _require_present(self, caller: str, **vals) -> bool:
        """发送前空值检查：各参数须全不为空（非 ``None`` 且无 NaN），否则限频 warn 并不发送。"""
        for name, v in vals.items():
            if v is None or np.isnan(np.asarray(v, dtype=float)).any():
                self._warn_throttled("backend.py - Backend.%s：参数『%s』存在空值（未传入或 cfg 未配置），本次不发送",
                                     caller, name)
                return False
        return True

    def _to_float_arr(self, x, caller: str) -> Optional[np.ndarray]:
        """指令/参数值转 ``(≥1,)`` float 数组：含 None 等非数值时限频 warn 并返回 ``None``（不向调用者抛）。"""
        try:
            return np.atleast_1d(np.asarray(x, dtype=float))
        except (TypeError, ValueError) as e:
            self._warn_throttled("backend.py - Backend.%s：参数存在非数值（%s），本次操作跳过", caller, e)
            return None

    def _warn_throttled(self, fmt: str, *args) -> None:
        """限频告警：至多每 ``self._warn_interval`` 秒一条（指令可能从控制线程与应用线程并发发来）。"""
        now = time.monotonic()
        with self._warn_lock:
            if now - self._warn_last < self._warn_interval:
                return
            self._warn_last = now
        logger.warning(fmt, *args)

    def _call(self, caller: str, fn, *args, **kwargs) -> None:
        """安全调用子类实现：异常统一转为 warn，不向调用者抛。"""
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.warning("backend.py - Backend.%s：内核异常：%s", caller, e)
