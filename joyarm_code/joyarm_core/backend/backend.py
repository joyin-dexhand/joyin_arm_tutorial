"""``Backend`` —— joyarm 硬件后端抽象基类（只定义各型号通用的功能）。

负责**接入具体型号的机械臂硬件**（总线、电机协议、末端执行器），对joyarm 上层暴露**硬件无关**接口：
基类实现状态维护、逐 joint 循环编排、维度校验、限位守卫、低频状态刷新与属性访问等通用功能；
子类按型号实现各**抽象内核**（单 joint / 单族的具体协议操作）。

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

# 越限告警节流间隔（秒）：持续越限至多每 0.5 s（2 Hz）告警一条；裁剪本身不受节流影响、始终执行
_WARN_INTERVAL = 0.5
# write_param 写后至同键读回验证的静置间隔（秒）
_WRITE_SETTLE = 0.1
# 低频状态刷新默认频率（Hz），cfg 键 state_refresh_hz 可覆盖
_REFRESH_HZ_DEFAULT = 10.0


class Backend(ABC):
    """joyarm 硬件后端抽象基类（arm 本体 + end 末端执行器一体）。

    九大功能：成员变量定义 / 初始化过程 / 生命周期管理（连接与使能）/
    失能态读取 / 失能态写入 / 使能态指令发送 / 错误与恢复 / 外部属性访问 / 内部助手。
    状态标志（``is_connected_arm/end`` / ``is_abled_arm/end``）：``None`` 未知、``True``、``False``。

    :param cfg: config 配置 ``backend:`` 段整体
    """

    def __init__(self, cfg: dict) -> None:
        """初始化 backend：自 cfg 解析成员并启动低频状态刷新线程。

        :param cfg: config 配置 ``backend:`` 段整体。
        """
        # ============================================================
        # 成员变量定义（先赋初值，具体值经下方初始化过程自 cfg 解析）
        # ============================================================
        self._cfg = cfg                                       # config 的 backend 段整体
        self._name = ""                                       # backend 名（cfg.name）
        self._channel_arm = None                              # arm 总线通道（cfg.arm.channel）
        self._channel_end = None                              # end 总线通道（cfg.end.channel）
        self._baud_arm = None                                 # arm 总线波特率（cfg.arm.baud_rate，协议相关键）
        self._baud_end = None                                 # end 总线波特率（cfg.end.baud_rate，协议相关键）
        self._protocol_arm = ""                               # arm 协议标识（cfg.arm.protocol）
        self._protocol_end = ""                               # end 协议标识（cfg.end.protocol）
        self._jointscfg_arm: list[dict] = []                  # arm 各 joint 配置列表（cfg.arm.joints）
        self._jointscfg_end: list[dict] = []                  # end 各 joint 配置列表（cfg.end.joints）
        self._joint_limits_arm: Optional[JointLimits] = None  # arm 硬限位（守卫依据）
        self._joint_limits_end: Optional[JointLimits] = None  # end 硬限位（守卫依据）
        self._kp_mit_default_arm: Optional[np.ndarray] = None    # arm MIT 位置增益发送默认（cfg MIT.kp → (n,)）
        self._kd_mit_default_arm: Optional[np.ndarray] = None    # arm MIT 速度阻尼发送默认（cfg MIT.kd → (n,)）
        self._kp_mit_default_end: Optional[np.ndarray] = None    # end MIT 位置增益发送默认（cfg MIT.kp → (n,)）
        self._kd_mit_default_end: Optional[np.ndarray] = None    # end MIT 速度阻尼发送默认（cfg MIT.kd → (n,)）
        self._vlim_default_arm: Optional[np.ndarray] = None      # arm 位置指令限速发送默认（cfg POS_VEL.vlim → (n,)）
        self._flim_default_arm: Optional[np.ndarray] = None      # arm 归一化限流发送默认 0~1（cfg POS_VEL.flim → (n,)）
        self._vlim_default_end: Optional[np.ndarray] = None      # end 位置指令限速发送默认（cfg POS_VEL.vlim → (n,)）
        self._flim_default_end: Optional[np.ndarray] = None      # end 归一化限流发送默认 0~1（cfg POS_VEL.flim → (n,)）
        self._pos_kp_arm: Optional[np.ndarray] = None        # arm 位置环 kp 增益（cfg POS_VEL.pos_kp → (n,)，set_mode 内核直取）
        self._pos_ki_arm: Optional[np.ndarray] = None        # arm 位置环 ki 增益（cfg POS_VEL.pos_ki → (n,)）
        self._vel_kp_arm: Optional[np.ndarray] = None        # arm 速度环 kp 增益（cfg POS_VEL.vel_kp → (n,)）
        self._vel_ki_arm: Optional[np.ndarray] = None        # arm 速度环 ki 增益（cfg POS_VEL.vel_ki → (n,)）
        self._pos_kp_end: Optional[np.ndarray] = None        # end 位置环 kp 增益（cfg POS_VEL.pos_kp → (n,)）
        self._pos_ki_end: Optional[np.ndarray] = None        # end 位置环 ki 增益（cfg POS_VEL.pos_ki → (n,)）
        self._vel_kp_end: Optional[np.ndarray] = None        # end 速度环 kp 增益（cfg POS_VEL.vel_kp → (n,)）
        self._vel_ki_end: Optional[np.ndarray] = None        # end 速度环 ki 增益（cfg POS_VEL.vel_ki → (n,)）
        self.is_connected_arm: Optional[bool] = None          # arm 连接状态（True 已连接 / False 未连接 / None 未知）
        self.is_connected_end: Optional[bool] = None          # end 连接状态（True 已连接 / False 未连接 / None 未知）
        self.is_abled_arm: Optional[bool] = None              # arm 使能状态（True 已使能 / False 已失能 / None 未知）
        self.is_abled_end: Optional[bool] = None              # end 使能状态（True 已使能 / False 已失能 / None 未知）
        self.mode_arm: Optional[ControlMode] = None           # arm 族一致控制模式（公开；get_mode 读齐且一致时更新，None=未读取）
        self.mode_end: Optional[ControlMode] = None           # end 族一致控制模式（公开；语义同 mode_arm）
        self._joint_mode_arm: list = []                       # arm 逐 joint 模式缓存（None=未读取；get_mode 读齐整体赋值）
        self._joint_mode_end: list = []                       # end 逐 joint 模式缓存（语义同 _joint_mode_arm）
        self.joint_state_arm = JointState()      # arm 整族状态槽（公开实时引用）
        self.joint_state_end = JointState()      # end 整族状态槽（公开实时引用）
        self._warn_lock = threading.Lock()                    # 告警节流锁（指令可能从控制线程与应用线程并发）
        self._warn_last = 0.0                                 # 上次节流告警的 monotonic 时刻
        self._refresh_hz = _REFRESH_HZ_DEFAULT                # 低频状态刷新频率（cfg.state_refresh_hz 可覆盖）
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
        self._joint_limits_arm = limits_from_joint_cfgs(self._jointscfg_arm)
        self._joint_limits_end = limits_from_joint_cfgs(self._jointscfg_end)
        n_arm, n_end = len(self._jointscfg_arm), len(self._jointscfg_end)
        # 状态槽定维：各子值维度 = 关节数（物理量 NaN 占位、error 整数异常码 -1 占位 = 未读取）；模式缓存逐 joint None
        self.joint_state_arm = JointState(q=np.full(n_arm, np.nan), dq=np.full(n_arm, np.nan),
                                          tau=np.full(n_arm, np.nan), temp_mos=np.full(n_arm, np.nan),
                                          temp_rotor=np.full(n_arm, np.nan), error=np.full(n_arm, -1))
        self.joint_state_end = JointState(q=np.full(n_end, np.nan), dq=np.full(n_end, np.nan),
                                          tau=np.full(n_end, np.nan), temp_mos=np.full(n_end, np.nan),
                                          temp_rotor=np.full(n_end, np.nan), error=np.full(n_end, -1))
        self._joint_mode_arm = [None] * n_arm
        self._joint_mode_end = [None] * n_end
        # 发送默认值与 POS_VEL 增益提取：逐关节提取为 (n,) 成员（空值 NaN 占位 + warn；
        # 发送默认值被发送空值门禁拦截，增益被 set_mode 非 MIT 门禁拦截）
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
        hz = float(cfg.get("state_refresh_hz", _REFRESH_HZ_DEFAULT))
        if hz <= 0:
            logger.warning("backend.py - Backend.__init__：state_refresh_hz=%s 非法，沿用默认 %s Hz",
                            hz, _REFRESH_HZ_DEFAULT)
        else:
            self._refresh_hz = hz
        self._refresh_thread = threading.Thread(target=self._refresh_loop,
                                                name="backend-state-refresh", daemon=True)
        self._refresh_thread.start()

    # ============================================================
    # 生命周期管理（连接与使能）
    # ============================================================
    # =============== 连接（connect / disconnect） ===============
    def connect_arm(self, channel: Optional[str] = None, protocol: Optional[str] = None) -> None:
        """连接 arm 硬件总线
        
        is_connected_arm 执行前置 ``None``，成功置 ``True``，意外保持 ``None``。

        :param channel: 总线通道；默认取成员 ``_channel_arm``（cfg 解析），显式传入则同步更新该成员。
        :param protocol: 协议标识；默认取成员 ``_protocol_arm``（cfg 解析），显式传入则同步更新。
        """
        self._connect_impl("arm", channel, protocol)

    def connect_end(self, channel: Optional[str] = None, protocol: Optional[str] = None) -> None:
        """连接 end 硬件总线

        is_connected_end 执行前置 ``None``，成功置 ``True``，意外保持 ``None``。

        :param channel: 总线通道；默认取成员 ``_channel_end``（cfg 解析），显式传入则同步更新该成员。
        :param protocol: 协议标识；默认取成员 ``_protocol_end``（cfg 解析），显式传入则同步更新。
        """
        self._connect_impl("end", channel, protocol)

    def disconnect_arm(self) -> None:
        """断开 arm 硬件总线
        
        is_connected_arm 执行前置 ``None``，成功置 ``False``，意外保持 ``None``。
        """
        self._disconnect_impl("arm")

    def disconnect_end(self) -> None:
        """断开 end 硬件总线
        
        is_connected_end 执行前置 ``None``，成功置 ``False``，意外保持 ``None``。
        """
        self._disconnect_impl("end")

    # ================= 使能（enable / disable） =================
    def enable_arm(self) -> int:
        """使能 arm 全部关节电机

        逐 joint 调内核；执行前 ``is_abled_arm`` 置 ``None``，全成功置 ``True``，意外保持 ``None``。

        :return: 全部使能成功返回 1，任一失败（warn 提示）返回 0。
        """
        return self._enable_impl("arm")

    def enable_end(self) -> int:
        """使能 end 全部关节电机
        
        逐 joint 调内核；执行前 ``is_abled_end`` 置 ``None``，全成功置 ``True``，意外保持 ``None``。
        """
        return self._enable_impl("end")

    def disable_arm(self) -> int:
        """失能 arm 全部关节电机

        逐 joint 调内核；执行前 ``is_abled_arm`` 置 ``None``，全成功置 ``False``，意外保持 ``None``。

        :return: 全部失能成功返回 1，任一失败（warn 提示）返回 0。
        """
        return self._disable_impl("arm")

    def disable_end(self) -> int:
        """失能 end 全部关节电机

        逐 joint 调内核；执行前 ``is_abled_end`` 置 ``None``，全成功置 ``False``，意外保持 ``None``。

        :return: 全部失能成功返回 1，任一失败（warn 提示）返回 0。
        """
        return self._disable_impl("end")

    # ======================== 族通用实现 ========================
    def _connect_impl(self, family: str, channel: Optional[str], protocol: Optional[str]) -> None:
        """连接族通用实现：同步 channel/protocol 成员 → 置 None → 调内核 → 成功置 True。"""
        if self._skip_empty_family(family, f"connect_{family}"):
            return
        if channel is not None:
            setattr(self, f"_channel_{family}", channel)
        if protocol is not None:
            setattr(self, f"_protocol_{family}", protocol)
        setattr(self, f"is_connected_{family}", None)  # 执行中未知态；任何意外即停留于此
        try:
            ok = getattr(self, f"_connect_{family}")(getattr(self, f"_channel_{family}"),
                                                     getattr(self, f"_protocol_{family}"))
        except Exception as e:
            logger.warning("backend.py - Backend.connect_%s：连接内核异常：%s", family, e)
            return
        if ok:
            setattr(self, f"is_connected_{family}", True)
            logger.info("backend.py - Backend.connect_%s：连接成功（channel=%s, protocol=%s）",
                        family, getattr(self, f"_channel_{family}"),
                        getattr(self, f"_protocol_{family}"))
            if self._set_mode_impl(family, ControlMode.MIT) != 1:  # 自动设 MIT 默认模式（best-effort）
                logger.warning("backend.py - Backend.connect_%s：默认模式（MIT）设置失败", family)
        else:
            logger.warning("backend.py - Backend.connect_%s：连接失败（channel=%s, protocol=%s）",
                           family, getattr(self, f"_channel_{family}"),
                           getattr(self, f"_protocol_{family}"))

    def _disconnect_impl(self, family: str) -> None:
        """断连族通用实现：先失能（best-effort）→ 置 None → 调内核 → 成功置 False。

        仅 ``is_connected is False``（确知未连接）时为空操作；``None``（未知）也尝试断连，确保清理半开链路；
        ``is_abled`` 断连后恒 ``None``（链路断开，使能态不可核实）；模式成员与逐 joint 模式缓存同步清空
        （防重连时设模式失败后，陈旧模式经发送门禁放行错误指令）；``joint_state`` 最后快照保留供上层查历史。
        """
        if self._skip_empty_family(family, f"disconnect_{family}"):
            return
        if getattr(self, f"is_connected_{family}") is False:
            return
        if getattr(self, f"is_abled_{family}") is not False:  # 先失能（未确知已失能才发帧；失败仅 warn 不阻断断连）
            self._disable_impl(family)
        setattr(self, f"is_connected_{family}", None)  # 执行中未知态；任何意外即停留于此
        setattr(self, f"is_abled_{family}", None)      # 链路断开，使能态不可核实
        try:
            getattr(self, f"_disconnect_{family}")()
        except Exception as e:
            logger.warning("backend.py - Backend.disconnect_%s：断连内核异常：%s", family, e)
            return
        setattr(self, f"is_connected_{family}", False)
        setattr(self, f"mode_{family}", None)  # 模式不可核实，清空防陈旧放行
        setattr(self, f"_joint_mode_{family}", [None] * self._n(family))
        logger.info("backend.py - Backend.disconnect_%s：断连成功", family)

    def _enable_impl(self, family: str) -> int:
        """使能族通用实现：置 None → 逐 joint 调内核 → 全成功置 True。"""
        if self._skip_empty_family(family, f"enable_{family}"):
            return 1
        if not self._require_connected(family, f"enable_{family}"):
            return 0
        setattr(self, f"is_abled_{family}", None)  # 执行中未知态；任何意外即停留于此
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
            setattr(self, f"is_abled_{family}", True)
            logger.info("backend.py - Backend.enable_%s：全部 joint 使能成功", family)
        return 1 if ok else 0

    def _disable_impl(self, family: str) -> int:
        """失能族通用实现：置 None → 逐 joint 调内核 → 全成功置 False。"""
        if self._skip_empty_family(family, f"disable_{family}"):
            return 1
        if not self._require_connected(family, f"disable_{family}"):
            return 0
        setattr(self, f"is_abled_{family}", None)  # 执行中未知态；任何意外即停留于此
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
            setattr(self, f"is_abled_{family}", False)
            logger.info("backend.py - Backend.disable_%s：全部 joint 失能成功", family)
        return 1 if ok else 0

    # =================== 抽象内核（子类实现） ===================
    @abstractmethod
    def _connect_arm(self, channel, protocol) -> bool:
        """arm 连接内核：打开 ``channel`` 总线（协议按 ``protocol`` 识别或子类默认直接给定）。

        :return: 成功 ``True``；失败 ``False`` 或上抛异常。
        """

    @abstractmethod
    def _connect_end(self, channel, protocol) -> bool:
        """end 连接内核：打开 ``channel`` 总线（协议按 ``protocol`` 识别或子类默认直接给定；与 arm 同 channel 时共享总线）。

        :return: 成功 ``True``；失败 ``False`` 或上抛异常。
        """
    
    @abstractmethod
    def _disconnect_arm(self) -> None:
        """arm 断连内核：关闭子类协议层自建资源（总线接收线程、串口/CAN 句柄等）。

        电机失能由基类断连编排先行完成（先失能再断连），内核只负责关资源。
        """

    @abstractmethod
    def _disconnect_end(self) -> None:
        """end 断连内核：关闭子类协议层自建资源（总线接收线程、串口/CAN 句柄等）。

        电机失能由基类断连编排先行完成（先失能再断连），内核只负责关资源。
        """

    @abstractmethod
    def _enable_joint_arm(self, i: int) -> bool:
        """arm 单 joint 使能内核：``i`` 为 ``_jointscfg_arm`` 下标。失败上抛或返回 ``False``。"""

    @abstractmethod
    def _enable_joint_end(self, i: int) -> bool:
        """end 单电机使能内核：``i`` 为 ``_jointscfg_end`` 下标。失败上抛或返回 ``False``。"""

    @abstractmethod
    def _disable_joint_arm(self, i: int) -> bool:
        """arm 单 joint 失能内核。失败上抛或返回 ``False``。"""

    @abstractmethod
    def _disable_joint_end(self, i: int) -> bool:
        """end 单电机失能内核：``i`` 为 ``_jointscfg_end`` 下标。失败上抛或返回 ``False``。"""

    # ============================================================
    # 失能态读取
    # ============================================================
    def read_param_arm(self, key: str) -> Optional[list]:
        """读 arm 全部 joint 的电机参数（键值表由子类定义）。

        :param key: 参数名（子类映射到厂商寄存器，如 DM 的 ``"pos_kp"``）。
        :return: 逐 joint 参数值列表（长度 = 关节数）；未连接或任一失败返回 ``None``。
        """
        return self._read_params_impl("arm", key)

    def read_param_end(self, key: str) -> Optional[list]:
        """读 end 全部电机的电机参数（键值表由子类定义）。

        :param key: 参数名（子类映射到厂商寄存器，如 DM 的 ``"pos_kp"``）。
        :return: 逐电机参数值列表（长度 = 电机关节数）；未连接或任一失败返回 ``None``。
        """
        return self._read_params_impl("end", key)

    def get_mode_arm(self) -> Union[ControlMode, bool, None]:
        """读 arm 全部 joint 控制模式并逐 joint 缓存到 ``_joint_mode_arm``。

        :return: 各 joint 模式一致：更新成员 ``mode_arm``（info）并返回该模式；不一致：warn 并返回
            ``False``（逐 joint 模式仍保留在缓存）；任一读取失败：返回 ``None`` 。
        """
        return self._get_mode_impl("arm")

    def get_mode_end(self) -> Union[ControlMode, bool, None]:
        """读 end 全部电机控制模式并逐电机缓存到 ``_joint_mode_end``。

        :return: 各电机模式一致：更新成员 ``mode_end``（info）并返回该模式；不一致：warn 并返回
            ``False``（逐电机模式仍保留在缓存）；任一读取失败：返回 ``None``。
        """
        return self._get_mode_impl("end")

    def get_state_arm(self) -> Optional[JointState]:
        """读 arm 全部 joint 运动状态，在场子值齐备才整体装配并打时间戳。

        子类不提供的量（内核字典缺键）该字段整体置 ``None``。

        :return: 成功返回 ``joint_state_arm``（实时引用，``t`` 为本次更新时刻）；任一 joint 读取失败、
            在场子值为空值（数据异常）时，跳过本轮赋值（节流 warn）并返回 ``None``。
        """
        return self._get_state_impl("arm")

    def get_state_end(self) -> Optional[JointState]:
        """读 end 全部电机运动状态，在场子值齐备才整体装配并打时间戳。

        子类不提供的量（内核字典缺键）该字段整体置 ``None``。

        :return: 成功返回 ``joint_state_end``（实时引用，``t`` 为本次更新时刻）；任一电机读取失败、
            在场子值为空值（数据异常）时，跳过本轮赋值（节流 warn）并返回 ``None``。
        """
        return self._get_state_impl("end")

    # ======================== 族通用实现 ========================
    def _read_params_impl(self, family: str, key: str) -> Optional[list]:
        """读参数族通用实现：逐 joint 读寄存器 → 值列表（任一失败整族作废返回 ``None``）。"""
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
        if len(vals) != self._n(family):  # 返回值维度校验（防御内核误返回序列）
            logger.warning("backend.py - Backend.read_param_%s：返回维度 %d ≠ 关节数 %d",
                           family, len(vals), self._n(family))
            return None
        return vals

    def _get_mode_impl(self, family: str) -> Union[ControlMode, bool, None]:
        """读模式族通用实现：先全部读齐再整体赋值（任一失败不赋值，族内同步，不更新 ``t``）。

        全族一致时更新成员 ``mode_{family}``（info）并返回该模式；不一致 warn 返回 ``False``；
        任一读取失败返回 ``None``（不赋值、不更新成员）。
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
        setattr(self, f"_joint_mode_{family}", list(modes))  # 逐 joint 写入各自模式
        if len(set(modes)) == 1:
            setattr(self, f"mode_{family}", modes[0])
            logger.info("backend.py - Backend.get_mode_%s：全族模式一致（%s）", family, modes[0])
            return modes[0]
        logger.warning("backend.py - Backend.get_mode_%s：各 joint 模式不一致（%s），族模式成员不更新",
                       family, modes)
        return False

    def _get_state_impl(self, family: str) -> Optional[JointState]:
        """读状态族通用实现：在场子值齐备才整体赋值并打时间戳（与模式读取解耦）。

        内核字典**缺键**说明子类/硬件不提供该量，该字段整体置 ``None``（结构性缺席，合法）；
        键**在场但值为 ``None``/不可解析**（数据异常），则跳过本轮赋值并节流 warn，返回 ``None``。
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
        state = getattr(self, f"joint_state_{family}")
        names = ("q", "dq", "tau", "temp_mos", "temp_rotor", "error")
        absent = {name for name in names if any(name not in snap for snap in snaps)}  # 子类不含该量 → 字段置 None
        bad = {name for name in names if name not in absent
               and any(snap.get(name) is None for snap in snaps)}                    # 键在值空 → 数据异常
        if bad:
            self._warn_throttled(
                "backend.py - Backend.get_state_%s：在场子值 %s 存在空值（数据异常），本轮状态跳过更新（保持上次快照）",
                family, sorted(bad))
            return None
        present = tuple(name for name in names if name not in absent)
        try:  # 先解析后提交：任一不可解析按数据异常整轮跳过，不产生部分赋值
            assembled = {name: np.asarray([snap[name] for snap in snaps],
                                           dtype=int if name == "error" else float)
                         for name in present}
        except (TypeError, ValueError) as e:
            self._warn_throttled("backend.py - Backend.get_state_%s：子值解析失败（数据异常）：%s", family, e)
            return None
        for name in absent:
            setattr(state, name, None)
        for name, arr in assembled.items():
            setattr(state, name, arr)  # error 整数异常码，其余物理量 float
        state.t = time.time()  # 打时间戳：低频保活线程据此跳过新鲜族
        return state

    # ======================= 低频状态保活 =======================
    def _refresh_loop(self) -> None:
        """低频状态保活线程体（daemon，``__init__`` 启动，进程退出自动结束）。

        每周期（``1/_refresh_hz`` 秒）遍历 arm/end：未连接（含未知态）跳过；``t`` 新鲜（滞后不足半周期）跳过保活；
        ``t`` 陈旧时刷新状态槽——仅当模式缓存 ``_joint_mode_{family}`` 存在未读取 joint 才先 ``get_mode_*``，
        随后 ``get_state_*``。
        """
        period = 1.0 / self._refresh_hz
        fresh = 0.5 * period  # 新鲜度阈值：滞后 < 半周期视为已被高频途径更新
        while not self._refresh_stop.wait(period):
            now = time.time()
            for family in ("arm", "end"):
                if not getattr(self, f"is_connected_{family}"):
                    continue
                if now - getattr(self, f"joint_state_{family}").t < fresh:
                    continue
                if any(m is None for m in getattr(self, f"_joint_mode_{family}")):
                    getattr(self, f"get_mode_{family}")()
                getattr(self, f"get_state_{family}")()

    # =================== 抽象内核（子类实现） ===================
    @abstractmethod
    def _read_joint_param_arm(self, i: int, key: str):
        """arm 单 joint 读参数内核：读该电机 ``key`` 对应寄存器，返回参数值。失败上抛。"""

    @abstractmethod
    def _read_joint_param_end(self, i: int, key: str):
        """end 单电机读参数内核：读该电机 ``key`` 对应寄存器，返回参数值。失败上抛。"""

    @abstractmethod
    def _read_joint_mode_arm(self, i: int) -> Optional[ControlMode]:
        """arm 单 joint 读模式内核：读该电机当前控制模式。读不到返回 ``None``，失败上抛。"""

    @abstractmethod
    def _read_joint_mode_end(self, i: int) -> Optional[ControlMode]:
        """end 单电机读模式内核：读该电机当前控制模式。读不到返回 ``None``，失败上抛。"""

    @abstractmethod
    def _read_joint_state_arm(self, i: int) -> dict:
        """arm 单 joint 读状态内核：返回该 joint 的可用量字典。

        键可含 ``q`` / ``dq`` / ``tau`` / ``temp_mos`` / ``temp_rotor`` / ``error`` （rad / rad/s / N·m / ℃ / ℃ / 异常码）。
        硬件不提供的量**缺键**（该字段不被包含）；键在场但值为 ``None`` 属数据获取异常。
        ``error`` 全库约定：``0``=失能、``1``=使能正常、``≥2``=故障码（子类负责映射厂商原始码）。
        """

    @abstractmethod
    def _read_joint_state_end(self, i: int) -> dict:
        """end 单电机读状态内核：返回该电机的可用量字典。

        键可含 ``q`` / ``dq`` / ``tau`` / ``temp_mos`` / ``temp_rotor`` / ``error`` （rad / rad/s / N·m / ℃ / ℃ / 异常码）。
        硬件不提供的量**缺键**（该字段不被包含）；键在场但值为 ``None`` 属数据获取异常。
        ``error`` 全库约定：``0``=失能、``1``=使能正常、``≥2``=故障码（子类负责映射厂商原始码）。
        """

    # ============================================================
    # 失能态写入
    # ============================================================
    def write_param_arm(self, key: str, values) -> int:
        """写 arm 全部 joint 的电机参数并读回验证（键值表由子类定义）。

        逐 joint 全部写入 → 静置 ``0.1 s`` → 同键逐 joint 读回 → 与传入值**逐元素相等**才算成功。

        :param key: 参数名（子类映射到厂商寄存器）。
        :param values: 参数值列表（长度 = 关节数）。
        :return: 写入并验证成功返回 1；值列表维度不符（warn 并跳过本次）或任一失败返回 0。
        """
        return self._write_param_impl("arm", key, values)

    def write_param_end(self, key: str, values) -> int:
        """写 end 全部电机的电机参数并读回验证（键值表由子类定义）。

        逐电机全部写入 → 静置 ``0.1 s`` → 同键逐电机读回 → 与传入值**逐元素相等**才算成功。

        :param key: 参数名（子类映射到厂商寄存器）。
        :param values: 参数值列表（长度 = 电机关节数）。
        :return: 写入并验证成功返回 1；值列表维度不符（warn 并跳过本次）或任一失败返回 0。
        """
        return self._write_param_impl("end", key, values)

    def set_zero_arm(self) -> int:
        """arm 逐 joint 设置位置零点。

        :return: 全部成功返回 1，任一失败（warn 提示）返回 0。
        """
        return self._set_zero_impl("arm")

    def set_zero_end(self) -> int:
        """end 逐电机设置位置零点。

        :return: 全部成功返回 1，任一失败（warn 提示）返回 0。
        """
        return self._set_zero_impl("end")

    def set_mode_arm(self, mode: ControlMode = ControlMode.MIT) -> int:
        """arm 逐 joint 设置控制模式，设置后整族读回核对。

        :param mode: 目标控制模式，默认 :class:`ControlMode.MIT`（默认模式）。
        :return: 读回与目标一致返回 1（模式同步写入状态槽）；否则（warn 提示）返回 0。
        """
        return self._set_mode_impl("arm", mode)

    def set_mode_end(self, mode: ControlMode = ControlMode.MIT) -> int:
        """end 逐电机设置控制模式，设置后整族读回核对。

        :param mode: 目标控制模式，默认 :class:`ControlMode.MIT`（默认模式）。
        :return: 读回与目标一致返回 1（模式同步写入状态槽）；否则（warn 提示）返回 0。
        """
        return self._set_mode_impl("end", mode)

    # ======================== 族通用实现 ========================
    def _write_param_impl(self, family: str, key: str, values) -> int:
        """写参数族通用实现：维度自检 → 逐 joint 全部写入 → 静置 → 同键读回逐元素核对。

        值列表维度 ≠ 关节数时 warn 具体原因并跳过本次写入（返回 0）。
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
        time.sleep(_WRITE_SETTLE)  # 写入生效静置，再读回验证
        back = []
        for i in range(n):
            try:
                back.append(getattr(self, f"_read_joint_param_{family}")(i, key))
            except Exception as e:
                logger.warning("backend.py - Backend.write_param_%s：joint『%s』回读『%s』失败：%s",
                               family, self._jname(family, i), key, e)
                return 0
        if not np.allclose(np.asarray(back, dtype=float), vals, rtol=1e-5, atol=1e-9):
            # 容差比较：寄存器常为 float32，读回经 float32→float64 转换后与写入值必有低位差异，严格相等会误判失败
            logger.warning("backend.py - Backend.write_param_%s：『%s』写入验证失败：传入 %s ≠ 回读 %s",
                           family, key, vals.tolist(), back)
            return 0
        logger.info("backend.py - Backend.write_param_%s：参数『%s』写入并验证成功", family, key)
        return 1

    def _set_zero_impl(self, family: str) -> int:
        """设零点族通用实现：逐 joint 设置位置零点，任一失败即返回 0。"""
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
        """设模式族通用实现：增益校验（空值仅 warn）→ 逐 joint 设置 → 整族读回核对（复用 :meth:`get_mode`）。"""
        if self._skip_empty_family(family, f"set_mode_{family}"):
            return 1
        if not self._require_connected(family, f"set_mode_{family}"):
            return 0
        if mode is not ControlMode.MIT:  # 非 MIT 模式依赖 POS_VEL 增益寄存器，先校验增益成员无空值
            holes = [k for k in ("pos_kp", "pos_ki", "vel_kp", "vel_ki")
                     if np.isnan(getattr(self, f"_{k}_{family}")).any()]
            if holes:
                logger.warning("backend.py - Backend.set_mode_%s：POS_VEL 增益『%s』存在空值（cfg 未配置），采用电机内部增益",
                               family, "、".join(holes))
        for i in range(self._n(family)):
            try:
                getattr(self, f"_set_joint_mode_{family}")(i, mode)
            except Exception as e:
                logger.warning("backend.py - Backend.set_mode_%s：joint『%s』设置失败：%s",
                               family, self._jname(family, i), e)
                return 0
        time.sleep(_WRITE_SETTLE)  # 写入生效静置，再读回验证
        back = getattr(self, f"get_mode_{family}")()
        if back != mode:
            logger.warning("backend.py - Backend.set_mode_%s：读回核对不一致（期望 %s，实际 %s）",
                           family, mode, back)
            return 0
        logger.info("backend.py - Backend.set_mode_%s：模式切换成功（%s）", family, mode)
        return 1

    # =================== 抽象内核（子类实现） ===================
    @abstractmethod
    def _write_joint_param_arm(self, i: int, key: str, value: float) -> None:
        """arm 单 joint 写参数内核：写该电机 ``key`` 寄存器为 ``value``。失败上抛。"""

    @abstractmethod
    def _write_joint_param_end(self, i: int, key: str, value: float) -> None:
        """end 单电机写参数内核：写该电机 ``key`` 寄存器为 ``value``。失败上抛。"""

    @abstractmethod
    def _set_joint_mode_arm(self, i: int, mode: ControlMode) -> None:
        """arm 单 joint 设模式内核（含电机寄存器增益配置：增益自基类成员直取）。失败上抛。"""

    @abstractmethod
    def _set_joint_mode_end(self, i: int, mode: ControlMode) -> None:
        """end 单电机设模式内核（含电机寄存器增益配置：增益自基类成员直取）。失败上抛。"""

    @abstractmethod
    def _set_joint_zero_arm(self, i: int) -> bool:
        """arm 单 joint 设零内核：当前位置记为零点。成功 ``True``，失败 ``False`` 或上抛。"""

    @abstractmethod
    def _set_joint_zero_end(self, i: int) -> bool:
        """end 单电机设零内核：当前位置记为零点。成功 ``True``，失败 ``False`` 或上抛。"""

    # ============================================================
    # 使能态指令发送
    # ============================================================
    def send_mit_arm(self, tau, q, dq, kp=None, kd=None) -> None:
        """arm MIT 指令（电机内部 `` τ = tau + kp·(q_d−q) + kd·(dq_d−dq)``），整组下发。

        ``q``/``dq``/``tau`` 越限就近裁剪并节流告警。
        模式门禁：需先处于 ``MIT`` 模式（不符/未读取 warn 不发）；
        内核发送前校验各参数非空（含 NaN），空值 warn 不发送。

        :param tau: 前馈力矩 ``(n,)``，N·m。
        :param q: 位置目标 ``(n,)``，rad。
        :param dq: 速度目标 ``(n,)``，rad/s。
        :param kp: 位置增益 ``(n,)``；缺省赋 cfg ``MIT.kp`` 成员。
        :param kd: 速度阻尼 ``(n,)``；缺省赋 cfg ``MIT.kd`` 成员。
        """
        if kp is None:
            kp = self._kp_mit_default_arm  # 缺省 → cfg 提取的成员默认值
        if kd is None:
            kd = self._kd_mit_default_arm
        self._send_mit_impl("arm", tau, q, dq, kp, kd)

    def send_position_arm(self, q, vlim=None, flim=None) -> None:
        """arm 位置指令整组下发。

        :param q: 位置目标 ``(n,)``，rad。
        :param vlim: 速度上限 ``(n,)``，rad/s；缺省赋 cfg ``POS_VEL.vlim`` 成员。
        :param flim: 归一化力矩电流上限 ``(n,)``；缺省赋 cfg ``POS_VEL.flim`` 成员。
        """
        if vlim is None:
            vlim = self._vlim_default_arm  # 缺省 → cfg 提取的成员默认值
        if flim is None:
            flim = self._flim_default_arm
        self._send_position_impl("arm", q, vlim, flim)

    def send_vel_arm(self, dq) -> None:
        """arm 速度指令整组下发。

        :param dq: 速度目标 ``(n,)``，rad/s；越限裁剪到 ``±dq_max``。
        """
        self._send_vel_impl("arm", dq)

    def send_mit_end(self, tau, q, dq, kp=None, kd=None) -> None:
        """end MIT 指令（电机内部 `` τ = tau + kp·(q_d−q) + kd·(dq_d−dq)``），整组下发。

        ``q``/``dq``/``tau`` 越限就近裁剪并节流告警。
        模式门禁：需先处于 ``MIT`` 模式；
        内核发送前校验各参数非空（含 NaN）。

        :param tau: 前馈力矩 ``(n,)``，N·m。
        :param q: 位置目标 ``(n,)``，rad。
        :param dq: 速度目标 ``(n,)``，rad/s。
        :param kp: 位置增益 ``(n,)``；缺省赋 cfg ``MIT.kp`` 成员。
        :param kd: 速度阻尼 ``(n,)``；缺省赋 cfg ``MIT.kd`` 成员。
        """
        if kp is None:
            kp = self._kp_mit_default_end  # 缺省 → cfg 提取的成员默认值
        if kd is None:
            kd = self._kd_mit_default_end
        self._send_mit_impl("end", tau, q, dq, kp, kd)

    def send_position_end(self, q, vlim=None, flim=None) -> None:
        """end 位置指令整组下发。

        模式门禁：需先处于 ``POSITION`` 模式；
        内核发送前校验各参数非空（含 NaN）。

        :param q: 位置目标 ``(n,)``，rad。
        :param vlim: 速度上限 ``(n,)``，rad/s；缺省赋 cfg ``POS_VEL.vlim`` 成员。
        :param flim: 归一化力矩电流上限 ``(n,)``；缺省赋 cfg ``POS_VEL.flim`` 成员。
        """
        if vlim is None:
            vlim = self._vlim_default_end  # 缺省 → cfg 提取的成员默认值
        if flim is None:
            flim = self._flim_default_end
        self._send_position_impl("end", q, vlim, flim)

    def send_vel_end(self, dq) -> None:
        """end 速度指令整组下发。

        模式门禁：需先处于 ``VELOCITY`` 模式；
        内核发送前校验参数非空（含 NaN）。

        :param dq: 速度目标 ``(n,)``，rad/s；越限裁剪到 ``±dq_max``。
        """
        self._send_vel_impl("end", dq)

    def send_action_end(self, action: str, **kwargs) -> None:
        """end 离散动作（动作语义与可选配置由子类定义）。

        :param action: 动作名，常见 ``"open"`` / ``"close"`` / ``"home"``。
        :param kwargs: 动作的可选配置（由子类解释）。
        """
        if self._skip_empty_family("end", "send_action_end"):
            return
        if not self._require_connected("end", "send_action_end"):
            return
        self._call("send_action_end", self._send_action_end, action, **kwargs)

    # ======================== 族通用实现 ========================
    def _send_mit_impl(self, family: str, tau, q, dq, kp, kd) -> None:
        """MIT 族通用实现：维度校验 → 空值门禁+限位裁剪 → 连接/模式门禁 → 逐 joint 内核下发。"""
        if self._skip_empty_family(family, f"send_mit_{family}"):
            return
        # ① 全部量维度校验（不符 → warn 跳过本次发送）
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
        # ② 空值门禁（空值 → warn 不发送）+ 限位裁剪（越限 → 就近裁剪 + 节流 warn）
        if not self._require_present(f"send_mit_{family}", tau=tau, q=q, dq=dq, kp=kp, kd=kd):
            return
        lim = getattr(self, f"_joint_limits_{family}")  # 空族已跳过，限位必已解析
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
        # ③ 连接/模式门禁 → ④ 逐 joint 内核下发
        if not self._require_connected(family, f"send_mit_{family}"):
            return
        if not self._require_mode(family, ControlMode.MIT, f"send_mit_{family}"):
            return
        for i in range(n):
            self._call(f"send_mit_{family}（{self._jname(family, i)}）",
                       getattr(self, f"_send_joint_mit_{family}"),
                       i, tau[i], q[i], dq[i], kp[i], kd[i])

    def _send_position_impl(self, family: str, q, vlim, flim) -> None:
        """位置族通用实现：维度校验 → 空值门禁+限位裁剪 → 连接/模式门禁 → 逐 joint 内核下发。"""
        if self._skip_empty_family(family, f"send_position_{family}"):
            return
        # ① 全部量维度校验（不符 → warn 跳过本次发送）
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
        # ② 空值门禁（空值 → warn 不发送）+ 限位裁剪（越限 → 就近裁剪 + 节流 warn）
        if not self._require_present(f"send_position_{family}", q=q, vlim=vlim, flim=flim):
            return
        lim = getattr(self, f"_joint_limits_{family}")  # 空族已跳过，限位必已解析
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
        # ③ 连接/模式门禁 → ④ 逐 joint 内核下发
        if not self._require_connected(family, f"send_position_{family}"):
            return
        if not self._require_mode(family, ControlMode.POSITION, f"send_position_{family}"):
            return
        for i in range(n):
            self._call(f"send_position_{family}（{self._jname(family, i)}）",
                       getattr(self, f"_send_joint_position_{family}"),
                       i, q[i], vlim[i], flim[i])

    def _send_vel_impl(self, family: str, dq) -> None:
        """速度族通用实现：维度校验 → 空值门禁+限位裁剪 → 连接/模式门禁 → 逐 joint 内核下发。"""
        if self._skip_empty_family(family, f"send_vel_{family}"):
            return
        # ① 全部量维度校验（不符 → warn 跳过本次发送）
        n = self._n(family)
        dq = self._to_float_arr(dq, f"send_vel_{family}")
        if dq is None:
            return
        if len(dq) != n:
            self._warn_throttled("backend.py - Backend.send_vel_%s：指令维度（dq=%d）≠ 关节数 %d，本次发送跳过",
                                 family, len(dq), n)
            return
        # ② 空值门禁（空值 → warn 不发送）+ 限位裁剪（越限 → 就近裁剪 + 节流 warn）
        if not self._require_present(f"send_vel_{family}", dq=dq):
            return
        lim = getattr(self, f"_joint_limits_{family}")  # 空族已跳过，限位必已解析
        clipped = np.clip(dq, -lim.dq_max, lim.dq_max)
        if not np.array_equal(dq, clipped):
            self._warn_throttled("backend.py - Backend.send_vel_%s：dq 指令越限，已就近裁剪 %s → %s",
                                 family, np.round(dq, 4).tolist(), np.round(clipped, 4).tolist())
            dq = clipped
        # ③ 连接/模式门禁 → ④ 逐 joint 内核下发
        if not self._require_connected(family, f"send_vel_{family}"):
            return
        if not self._require_mode(family, ControlMode.VELOCITY, f"send_vel_{family}"):
            return
        for i in range(n):
            self._call(f"send_vel_{family}（{self._jname(family, i)}）",
                       getattr(self, f"_send_joint_vel_{family}"), i, dq[i])

    # =================== 发送内核（子类实现） ===================
    @abstractmethod
    def _send_joint_mit_arm(self, i: int, tau: float, q: float, dq: float,
                            kp: float, kd: float) -> None:
        """arm 单 joint MIT 发送内核（一帧，入参已限位裁剪）。失败上抛。"""

    @abstractmethod
    def _send_joint_mit_end(self, i: int, tau: float, q: float, dq: float,
                            kp: float, kd: float) -> None:
        """end 单电机 MIT 发送内核（一帧，入参已限位裁剪）。失败上抛。"""

    @abstractmethod
    def _send_joint_position_arm(self, i: int, q: float, vlim: float, flim: float) -> None:
        """arm 单 joint 位置发送内核（一帧，入参已限幅裁剪）。失败上抛。"""

    @abstractmethod
    def _send_joint_position_end(self, i: int, q: float, vlim: float, flim: float) -> None:
        """end 单电机位置发送内核（一帧，入参已限幅裁剪）。失败上抛。"""

    @abstractmethod
    def _send_joint_vel_arm(self, i: int, dq: float) -> None:
        """arm 单 joint 速度发送内核（一帧，入参已幅值裁剪）。失败上抛。"""

    @abstractmethod
    def _send_joint_vel_end(self, i: int, dq: float) -> None:
        """end 单电机速度发送内核（一帧，入参已幅值裁剪）。失败上抛。"""

    @abstractmethod
    def _send_action_end(self, action: str, **kwargs) -> None:
        """end 离散动作发送内核（动作映射与配置由子类定义）。失败上抛。"""

    # ============================================================
    # 错误与恢复
    # ============================================================
    def check_error_arm(self) -> bool:
        """检查 arm 各 joint 状态码（读一次整族状态）。

        状态码 0=失能、1=使能，均属正常：全为 0/1 → info 状态码并返回 ``True``；
        存在其他码（故障）→ warn 状态码并返回 ``False``。

        :return: 状态码正常返回 ``True``；存在故障码或读取失败返回 ``False``。
        """
        return self._check_error_impl("arm")

    def check_error_end(self) -> bool:
        """检查 end 各电机状态码（读一次整族状态）。

        状态码 0=失能、1=使能，均属正常：全为 0/1 → info 状态码并返回 ``True``；
        存在其他码（故障）→ warn 状态码并返回 ``False``。

        :return: 状态码正常返回 ``True``；存在故障码或读取失败返回 ``False``。
        """
        return self._check_error_impl("end")

    def clear_error_arm(self) -> bool:
        """清除 arm 各 joint 硬件错误并验证。

        先逐 joint 清错 → 静置等待 → :meth:`check_error_arm` 验证。

        :return: 清除并验证成功返回 ``True``；任一步失败 warn 并返回 ``False``。
        """
        return self._clear_error_impl("arm")

    def clear_error_end(self) -> bool:
        """清除 end 各电机硬件错误并验证。

        先逐电机清错 → 静置等待 → :meth:`check_error_end` 验证。

        :return: 清除并验证成功返回 ``True``；任一步失败 warn 并返回 ``False``。
        """
        return self._clear_error_impl("end")

    # ======================== 族通用实现 ========================
    def _check_error_impl(self, family: str) -> bool:
        """检查族通用实现：读一次整族状态，状态码 0=失能 / 1=使能 均正常，其他码为故障。"""
        if self._skip_empty_family(family, f"check_error_{family}"):
            return True
        if not self._require_connected(family, f"check_error_{family}"):
            return False
        st = getattr(self, f"get_state_{family}")()
        if st is None:
            logger.warning("backend.py - Backend.check_error_%s：状态读取失败，无法检查", family)
            return False
        err = st.error
        if err is None:
            logger.warning("backend.py - Backend.check_error_%s：error 字段缺失，无法检查", family)
            return False
        if ((err != 0) & (err != 1)).any():
            logger.warning("backend.py - Backend.check_error_%s：存在故障状态码（%s）", family, err.tolist())
            return False
        logger.info("backend.py - Backend.check_error_%s：状态码正常（%s）", family, err.tolist())
        return True

    def _clear_error_impl(self, family: str) -> bool:
        """清错族通用实现：逐 joint 清错 → 静置 → check_error 验证。"""
        if self._skip_empty_family(family, f"clear_error_{family}"):
            return True
        if not self._require_connected(family, f"clear_error_{family}"):
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
        time.sleep(_WRITE_SETTLE)
        return self._check_error_impl(family)

    # =================== 抽象内核（子类实现） ===================
    @abstractmethod
    def _clear_joint_error_arm(self, i: int) -> bool:
        """arm 单 joint 清错内核：发送该电机错误清除指令。成功 ``True``，失败 ``False`` 或上抛。"""

    @abstractmethod
    def _clear_joint_error_end(self, i: int) -> bool:
        """end 单电机清错内核：发送该电机错误清除指令。成功 ``True``，失败 ``False`` 或上抛。"""

    # ============================================================
    # 外部属性访问（joyarm 经属性打点访问私有成员，返回实时引用）
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
        return len(self._jointscfg_arm)

    @property
    def n_joints_end(self) -> int:
        """end 电机关节数（无 end 段为 0）。"""
        return len(self._jointscfg_end)

    @property
    def joint_limits_arm(self) -> Optional[JointLimits]:
        """arm 硬限位（守卫裁剪唯一依据；无 arm 段为 ``None``）。"""
        return self._joint_limits_arm

    @property
    def joint_limits_end(self) -> Optional[JointLimits]:
        """end 硬限位（无 end 段为 ``None``）。"""
        return self._joint_limits_end

    # ============================================================
    # 内部助手（仅基类使用）
    # ============================================================
    def _n(self, family: str) -> int:
        """族（``"arm"`` / ``"end"``）关节数。"""
        return len(getattr(self, f"_jointscfg_{family}"))

    def _skip_empty_family(self, family: str, caller: str) -> bool:
        """空族（n=0）跳过判定：视为成功直接返回，但节流 warn 提示未配置任何 joint。"""
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

        缺段/缺键/非数值 → 该关节 NaN 占位（空值）并 warn；发送默认值被空值门禁拦截，
        增益空值在 ``set_mode`` 非 MIT 时 warn（采用电机内部增益）。
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
        """连接门禁：未连接或状态未知（``None``）时 warn 并返回 ``False``。"""
        if not getattr(self, f"is_connected_{family}"):
            logger.warning("backend.py - Backend.%s：%s 未连接或状态未知，操作跳过", caller, family)
            return False
        return True

    def _require_mode(self, family: str, mode: ControlMode, caller: str) -> bool:
        """模式门禁：族模式成员 ``mode_{family}`` 等于所需模式才放行，否则节流 warn。

        ``mode_{family}`` 与逐 joint 模式缓存 ``_joint_mode_{family}`` 同步更新
        （``get_mode`` 读齐且一致时），为 ``None`` 即模式未读取。
        """
        cur = getattr(self, f"mode_{family}")
        if cur == mode:
            return True
        self._warn_throttled("backend.py - Backend.%s：%s 当前模式 %s ≠ 所需 %s（先 set_mode/get_mode）",
                             caller, family, cur if cur is not None else "未读取", mode)
        return False

    def _require_present(self, caller: str, **vals) -> bool:
        """发送空值门禁：各参数须全不为空（非 ``None`` 且无 NaN），否则节流 warn 不发送。"""
        for name, v in vals.items():
            if v is None or np.isnan(np.asarray(v, dtype=float)).any():
                self._warn_throttled("backend.py - Backend.%s：参数『%s』存在空值（未传入或 cfg 未配置），本次不发送",
                                     caller, name)
                return False
        return True

    def _to_float_arr(self, x, caller: str) -> Optional[np.ndarray]:
        """指令/参数值转 ``(≥1,)`` float 数组：含 None 等非数值时节流 warn 并返回 ``None``（不向用户抛）。"""
        try:
            return np.atleast_1d(np.asarray(x, dtype=float))
        except (TypeError, ValueError) as e:
            self._warn_throttled("backend.py - Backend.%s：参数存在非数值（%s），本次操作跳过", caller, e)
            return None

    def _warn_throttled(self, fmt: str, *args) -> None:
        """节流告警：至多每 ``_WARN_INTERVAL`` 秒一条（指令可能从控制线程与应用线程并发）。"""
        now = time.monotonic()
        with self._warn_lock:
            if now - self._warn_last < _WARN_INTERVAL:
                return
            self._warn_last = now
        logger.warning(fmt, *args)

    def _call(self, caller: str, fn, *args, **kwargs) -> None:
        """安全内核调用：异常统一降级为 warn，不向用户抛。"""
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.warning("backend.py - Backend.%s：内核异常：%s", caller, e)
