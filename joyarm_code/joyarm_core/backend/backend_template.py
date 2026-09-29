"""``BackendTemplate`` —— Backend 子类实现模板（复制本文件为 ``backend_<型号>.py`` 后填写）。

与 ``config/joyarm_template.yaml`` 配套使用，六步接入新型号：
1. 复制本文件为 ``backend_<型号>.py``，类名为``Backend<型号>``，全文替换 ``BackendTemplate`` / logger 名；
2. 复制 ``config/joyarm_template.yaml`` 为 ``config/joyarm_<型号>.yaml``，填好 ``backend:`` 段；
3. 按本文件各方法的注释实现全部 29 个抽象方法（最小可用子集见下方说明），可分步实现、逐步联调；
4. 在 ``backend/__init__.py`` 顶部导入子类，并在 ``REGISTRY`` 加一行注册；
5. ``joyarm_<型号>.yaml`` 的 ``backend.name`` 填子类 backend 的注册名；
6. 子类实现后更新注释 docstring：由「子类型号模板和实现约束」改为「子类型号的介绍说明」—— 包括文件头 docstring、类 docstring、各方法 docstring。

参考实现：``backend_dm.py`` 的 ``BackendDM`` 。

架构约束：
- 子类**不得新增公开方法/属性**：JoyArm 上层只面向 ``Backend`` 基类编程，子型号专属功能一律放私有层；
- 基类负责全部编排（逐关节循环、维度/空值检查、越限裁剪、连接与模式前置检查、限频日志、异常降级），
  子类只做单关节/单总线的协议操作——**不要在子类重复基类已做的检查与裁剪**；
- 失败约定：抽象方法失败上抛异常或返回 ``False``/``None`` 即可，基类统一转限频 warn。

最小可用子集（29 个抽象方法须全部**定义**以满足抽象方法强制；未实现的能力**保留本模板的 ``NotImplementedError``**）：
- 必实现 14 个：``_connect/_disconnect_arm/end`` / ``_enable/_disable_joint_arm/end`` / ``_read_joint_state_arm/end`` / ``_read/_set_joint_mode_arm/end``
    ——任何电机/舵机型号的可用底线（协议无模式回读时 ``_read_joint_mode_*`` 须用本地镜像，恒返 ``None`` 会使 ``get_mode_*`` 核实永远失败）；
- 按需实现 15 个：``_read/_write_joint_param_arm/end``（协议支持寄存器读写才需要）、``_set_joint_zero_arm/end``、
  ``_send_joint_mit/position/vel_arm/end``（型号支持哪种模式实现哪种，如舵机通常仅位置）、
  ``_send_action_end``（仅离散动作末端）、``_clear_joint_error_arm/end``（固件支持清错才需要）。

联调自检清单（实现完成后逐项核对，每项对应一条基类保障）：
1. 空 cfg ``{}`` 即可构造成功（基类 ``__init__`` 对缺段/缺键全部容错，构造不做任何总线 I/O）；
2. ``connect_*`` 成功后 ``is_connected_*`` 为 ``True``；无 MIT 的型号在 yaml 配 ``default_mode: position`` 后连接无 warn；
3. ``get_state_*`` 返回 ``JointState``：硬件提供的字段为 ``(n,)`` 数组、硬件不提供的字段为 ``None``，``t`` 为本次读取时刻（每次成功调用都更新）；
4. ``set_mode_*`` 成功后 ``mode_*`` 即为所设模式（乐观更新缓存，发送门禁立即放行）；``get_mode_*`` 能读回同一模式（读回以硬件真值纠正缓存）；
5. 发送越限指令被基类就近裁剪并限频 warn（内核收到的入参必在硬限位内，子类无需自检）；
6. ``read_param_*`` 返回逐关节**标量** list（子类返回非标量会被基类拦截、整组作废返 ``None``）；
7. ``disconnect_*`` 后重连可恢复（断连清空模式缓存，重连自动设默认模式并乐观回填）。
"""
from __future__ import annotations

import logging
from typing import Optional

from ..utils.types import ControlMode
from .backend import Backend

__all__ = ["BackendTemplate"]

logger = logging.getLogger("joyarm_core.backend_<型号>")  # 改型号时同步改名（如 ...backend_**）


class BackendTemplate(Backend):
    """Backend<型号> 硬件后端模板。

    ── 子类自建成员（协议层私有资源，``__init__`` 内初始化，全部 ``_`` 前缀）按协议需要自建，典型三件：
        电机句柄表（下标对齐 ``_jointscfg_arm/end``）、总线封装（``dict[channel → bus]``，arm/end 同 channel 则共享一条）、接收线程 + 发送锁 + 请求-应答同步 Event。

    ── 线程模型（并发调用方，总线访问务必线程安全）──
    - 应用线程：上层直接调用公开方法（send_* 高频路径在控制线程）；
    - 基类低频刷新线程（daemon）：周期调用 ``_read_joint_mode_*`` / ``_read_joint_state_*``；
    - 因此同一总线会被并发读写：收发同步方案（全双工三件套 / 半双工总线锁）见下方差异总览。

    ── 两类硬件的典型差异（总览，具体实施因型号而异）──
    - 总线/收发：应答式关节电机用「接收线程 + 发送锁 + 请求-应答 Event 同步」；半双工轮询舵机总线锁包住「发帧 + 等应答」整个事务；
    - 模式：关节电机通常有模式寄存器、可回读；舵机通常无模式概念 → ``_read_joint_mode_*`` 返回本地镜像，yaml 配 ``default_mode: position``；
    - 发送面：关节电机类通常 MIT / 位置 / 速度三模式全支持；舵机按硬件能力实现（通常位置模式，部分型号有速度/力矩模式可映射 VELOCITY/MIT），
      未实现的模式保留 ``NotImplementedError`` 桩 = 基类转 warn 优雅降级；
    - 状态：关节电机类反馈帧含 q/dq/tau/error（协议帧不含温度则 ``temp_*`` 缺键）；舵机类轮询读位置/负载/温度，硬件没有的量直接缺键；
    - 参数：关节电机类通常为寄存器 RID 表读写；舵机通常为 RAM 表读写，键名由子类定义。

    混合硬件属常态：同一子类可 arm 用关节电机 + end 用舵机——两族 ``channel`` 不同即各开各的独立总线，相同时才共享一条。
    无 end 的型号：yaml ``end:`` 段留空（关节数 0）即可，end 系列方法仍须定义（保留桩）以满足抽象方法强制。
    """

    def __init__(self, cfg: dict) -> None:
        """子类构造：先 ``super().__init__(cfg)``，再初始化协议层私有资源。

        :param cfg: yaml ``backend:`` 段整体。
        """
        super().__init__(cfg)
        # ---- 基类已初始化、子类可直接消费的成员（勿重复解析，缺省为 None）----
        # self._cfg                                  yaml backend 段整体（型号自定义顶层键在此取）
        # self._name                                 backend 注册名（cfg.name）
        # self._channel_arm / _channel_end           总线通道（cfg.arm/end.channel）
        # self._baud_arm / _baud_end                 总线波特率（cfg.arm/end.baud_rate）
        # self._protocol_arm / _protocol_end         协议标识（cfg.arm/end.protocol）
        # self._jointscfg_arm[i] / _jointscfg_end[i]     逐关节配置字典
        # self._joint_motor_id_arm[i] / _joint_motor_id_end[i]        电机总线地址（joints[].motor_id）
        # self._joint_feedback_id_arm[i] / _joint_feedback_id_end[i]  应答帧标识（joints[].feedback_id）
        # self._joint_model_arm[i] / _joint_model_end[i]              电机型号（joints[].model）
        # self._n_joints_arm / _n_joints_end         arm/end 关节数
        # self._joint_limits_arm / _joint_limits_end 硬限位 JointLimits（q_min/q_max/dq_max/tau_max 四个(n,) 向量，±inf=无限制）
        # self._pos_kp/_pos_ki/_vel_kp/_vel_ki × arm/end（8 个 (n,)）  POS_VEL 增益，设模式写增益寄存器时直取（缺配置 NaN）
        # self._joint_state_arm / _joint_state_end   私有状态对象（属性 joint_state_* 对外只读）

        # ---- 其余基类私有成员（基类自管理：按需借用，勿重复解析 cfg；状态缓存类只读、勿直接改写）----
        # self._kp/_kd_mit_default × arm/end（4 个 (n,)）   MIT 发送默认（cfg MIT.kp/kd）
        # self._vlim/_flim_default × arm/end（4 个 (n,)）   位置指令限速/限流默认（cfg POS_VEL.vlim/flim，flim 归一化 0~1）
        # self._is_connected_arm/end、_is_abled_arm/end     连接/使能三值缓存（None=未知；同名只读属性对外）
        # self._mode_arm/end、_joint_mode_arm/end           模式缓存（get_mode 读齐且一致时更新 / set_mode 成功时乐观更新）
        # self._default_mode_arm/end                        连接后基类自动设的模式（cfg default_mode）
        # self._write_settle / _refresh_hz                  指令帧生效等待（秒）/ 低频刷新频率（Hz）（cfg 可覆盖）
        # self._warn_interval / _info_interval              限频日志间隔（秒）
        # 勿动：_warn_last/_info_last（限频计时字典）、_refresh_stop/_refresh_thread（基类刷新线程）——
        # 子类接收线程自建停止标志（断连即停），勿复用 _refresh_stop。
        # 型号自定义键只从 _cfg 顶层与 _jointscfg_*[i] 取，其余键值勿重复解析。

        # ---- 子类私有成员示例（按型号删改；约定：全部 _ 前缀，不新增公开成员）----
        # self._motors_arm = self._build_motors("arm")  # arm 电机句柄表（下标 = _jointscfg_arm 下标）
        # self._motors_end = self._build_motors("end")  # end 电机句柄表
        # self._buses = {}                              # channel → 总线封装（arm/end 同 channel 则共享一条）

        # ---- 私有助手 ``_build_motors(family)`` 的契约（子类自建，基类无此方法）----
        # 按 ``self._joint_model_{family}[i]`` 选择型号量程参数表（位域打包用 PMAX/VMAX/TMAX 等，逐关节区分），创建电机句柄对象，下标严格对齐 ``_jointscfg_{family}``；
        # 只建对象、不做任何总线 I/O（真正打开总线在 ``_connect_*``）。

        # ---- 模块级常量三件套（参数寄存器组织范式，详见 ``_read_joint_param_*`` 注释）----
        # _PARAM_RIDS: dict[str, int]    参数名 → 厂商寄存器 RID（arm/end 共表时统一一张）
        # _READONLY_KEYS: frozenset[str] 只读参数集（序列号/硬件版本等，写侧上抛拒绝）
        # _UINT_RIDS: frozenset[int]     整型（uint32）寄存器 RID 集，其余按 float32 解析

    # ============================================================
    # 第一部分：生命周期——连接与使能（8 个抽象方法）
    # ============================================================
    # =============== 连接（connect / disconnect） ===============
    def _connect_arm(self, channel, protocol) -> bool:
        """连接 arm 总线：打开 ``channel``，按需创建接收线程与发送锁。

        实现要点：
        - 波特率取 ``self._baud_arm``；``protocol`` 通常为信息性标识，协议固定时可直接忽略、用子类自带默认；
        - arm 与 end 的 ``channel`` 相同时共享一条总线：先连的一方创建、后连的一方复用（只打开总线、不做任何电机操作）；
        - 幂等：基类不设「已连接即返回」守卫（保留强制重连恢复路径），本方法须自行处理「总线已打开」——直接返回 ``True``；
        - 全双工协议：创建接收线程，按 ``self._joint_feedback_id_arm[i]`` 把应答帧分发到对应电机句柄；
          半双工舵机：无接收线程，只开句柄 + 一把总线锁（「发帧 + 等应答」在锁内成对进行）；
        - 返回 ``True`` 后基类**立即**自动设默认模式并静置 ``write_settle``——返回 ``True`` 时总线必须已可收发指令帧。

        :param channel: 总线通道。
        :param protocol: 协议标识。
        :return: ``True`` / ``False`` 或直接上抛异常。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _connect_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _connect_end(self, channel, protocol) -> bool:
        """连接 end 总线：打开 ``channel``，按需创建接收线程与发送锁。

        实现要点：
        - 波特率取 ``self._baud_end``；``protocol`` 通常为信息性标识；
        - 与 arm 的 ``channel`` 相同时**必须复用 arm 已打开的总线**，不同时才另开独立总线（只打开总线、不做任何电机操作）；
        - 幂等：总线已打开直接返回 ``True``（基类不设「已连接即返回」守卫）；
        - 全双工协议按 ``self._joint_feedback_id_end[i]`` 分发应答帧；半双工舵机只开句柄 + 一把总线锁；
        - 返回 ``True`` 后基类**立即**自动设默认模式并静置 ``write_settle``——返回 ``True`` 时总线必须已可收发指令帧。

        :param channel: 总线通道。
        :param protocol: 协议标识。
        :return: 连接成功 ``True``；失败返回 ``False`` 或直接上抛异常。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _connect_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _disconnect_arm(self) -> None:
        """断开 arm 总线：只关闭子类自建的协议层资源。

        实现要点：
        - 电机失能已由基类在调用本方法前完成（先失能再断连），本方法不用处理电机；
        - 需要关闭的典型资源：接收线程（置停止标志 → join（带超时）→ 关句柄）、串口/CAN 句柄（取锁/关闭同样带超时，防与卡住的总线读互等）；
        - **共享总线的关断要谨慎**：断开 arm 时若 end 仍连接着同一条总线（``is_connected_end`` 不为 ``False``），只解除 arm 的关联、保留总线本体，待最后一组断开时才真正关闭；
        - 重复调用应安全（幂等）。

        :raises Exception: 断开失败上抛，基类转 warn（``is_connected_arm`` 停留 ``None``）。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _disconnect_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _disconnect_end(self) -> None:
        """断开 end 总线：只关闭子类自建的协议层资源。

        实现要点：
        - 电机失能已由基类在调用本方法前完成（先失能再断连），本方法不用处理电机；
        - 需要关闭的典型资源：接收线程（置停止标志 → join（带超时）→ 关句柄）、串口/CAN 句柄（取锁/关闭同样带超时，防与卡住的总线读互等）；
        - **共享总线的关断要谨慎**：断开 end 时若 arm 仍连接着同一条总线（``is_connected_arm`` 不为 ``False``），只解除 end 的关联、保留总线本体，待最后一组断开时才真正关闭；
        - 重复调用应安全（幂等）。

        :raises Exception: 断开失败上抛，基类转 warn（``is_connected_end`` 停留 ``None``）。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _disconnect_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    # ==================== 使能（enable / disable） ====================
    def _enable_joint_arm(self, i: int) -> bool:
        """使能 arm 第 ``i`` 个关节电机（上电进入闭环，开始响应指令）。

        实现要点：
        - 电机总线地址取 ``self._joint_motor_id_arm[i]``；

        :param i: arm 关节下标（``0 ≤ i < self._n_joints_arm``）。
        :return: 使能成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _enable_joint_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _enable_joint_end(self, i: int) -> bool:
        """使能 end 第 ``i`` 个电机（上电进入闭环）。

        实现要点：
        - 电机总线地址取 ``self._joint_motor_id_end[i]``；

        :param i: end 电机下标（``0 ≤ i < self._n_joints_end``）。
        :return: 使能成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _enable_joint_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _disable_joint_arm(self, i: int) -> bool:
        """失能 arm 第 ``i`` 个关节电机（退出闭环，电机归零）。

        实现要点：
        - 电机总线地址取 ``self._joint_motor_id_arm[i]``；

        :param i: arm 关节下标（``0 ≤ i < self._n_joints_arm``）。
        :return: 失能成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _disable_joint_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _disable_joint_end(self, i: int) -> bool:
        """失能 end 第 ``i`` 个电机（退出闭环，电机归零）。

        实现要点：
        - 电机总线地址取 ``self._joint_motor_id_end[i]``；

        :param i: end 电机下标（``0 ≤ i < self._n_joints_end``）。
        :return: 失能成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _disable_joint_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    # ============================================================
    # 第二部分：读取（6 个抽象方法）
    # ============================================================
    def _read_joint_param_arm(self, i: int, key: str):
        """读 arm 第 ``i`` 个关节的电机参数寄存器，返回参数值。

        实现要点：
        - ``key`` 为参数名，映射到厂商寄存器由子类定义——建议维护模块级常量表 ``_PARAM_RIDS_arm: dict[str, int]``（参数名 → 寄存器 RID，如 ``"pos_kp"`` → 27）；若 arm 与 end 共用同一套寄存器表，统一为 ``_PARAM_RIDS``；
          配套 ``_READONLY_KEYS``（只读参数集）与 ``_UINT_RIDS``（uint32 寄存器集，其余按 float32 解析）两个模块级常量，读写两侧共用（舵机类对应 RAM 表，组织方式相同）；
        - 寄存器值分整型（uint32）与浮点（float32）两类，按厂商手册解析；float32 读回与写入值必有低位差异，读回核对（含容差比较）由上层 JoyArm 负责，子类无需处理；
        - 请求-应答式协议建议发读帧后按 RID 等应答（Event 同步），失败/超时上抛；超时约 0.1s、重试 2~3 次为宜——本方法会被低频刷新线程间接调用，过长重试会拖慢刷新周期；
        - 半双工舵机在总线锁内完成「发读帧 + 等应答 + 解析」整个事务。

        :param i: arm 关节下标（``0 ≤ i < self._n_joints_arm``）。
        :param key: 参数名（str，经 ``_PARAM_RIDS`` 映射到厂商寄存器）。
        :return: 该参数的当前值（int 或 float，**必须是标量**）。
        :raises Exception: 读取失败/超时上抛，基类让整组读取作废返回 ``None``。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _read_joint_param_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _read_joint_param_end(self, i: int, key: str):
        """读 end 第 ``i`` 个电机的参数寄存器，返回参数值。

        实现要点：
        - ``key`` 为参数名，映射到厂商寄存器由子类定义——建议维护模块级常量表 ``_PARAM_RIDS_end: dict[str, int]``（参数名 → 寄存器 RID，如 ``"pos_kp"`` → 27）；若 arm 与 end 共用同一套寄存器表，统一为 ``_PARAM_RIDS``；
          配套 ``_READONLY_KEYS``（只读参数集）与 ``_UINT_RIDS``（uint32 寄存器集，其余按 float32 解析）两个模块级常量，读写两侧共用（舵机类对应 RAM 表，组织方式相同）；
        - 寄存器值分整型（uint32）与浮点（float32）两类，按厂商手册解析；float32 读回与写入值必有低位差异，读回核对（含容差比较）由上层 JoyArm 负责，子类无需处理；
        - 请求-应答式协议建议发读帧后按 RID 等应答（Event 同步），失败/超时上抛；超时约 0.1s、重试 2~3 次为宜——本方法会被低频刷新线程间接调用，过长重试会拖慢刷新周期；
        - 半双工舵机在总线锁内完成「发读帧 + 等应答 + 解析」整个事务。

        :param i: end 电机下标（``0 ≤ i < self._n_joints_end``）。
        :param key: 参数名（str，经 ``_PARAM_RIDS`` 映射到厂商寄存器）。
        :return: 该参数的当前值（int 或 float，**必须是标量**）。
        :raises Exception: 读取失败/超时上抛，基类让整组读取作废返回 ``None``。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _read_joint_param_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _read_joint_mode_arm(self, i: int) -> Optional[ControlMode]:
        """读 arm 第 ``i`` 个关节电机当前的控制模式。

        实现要点：
        - 返回 :class:`ControlMode`（MIT / POSITION / VELOCITY），厂商模式码需映射（如：1→MIT、2→POSITION、3→VELOCITY）；
        - 若模式存储于电机内部寄存器，本方法通常会调用 _read_joint_param_arm 方法；
        - 本方法会被基类低频刷新线程周期调用（逐关节缓存未读齐时），与发送路径共用总线；
        - **协议无模式回读时用本地镜像**：记录 ``_set_joint_mode_arm`` 最近设置的模式并原样返回（恒返 ``None`` 会使 ``get_mode_arm`` 核实永远失败）；
        - 舵机通常无模式概念：本地镜像恒返 ``ControlMode.POSITION``（yaml 配 ``default_mode: position``）。

        :param i: arm 关节下标（``0 ≤ i < self._n_joints_arm``）。
        :return: 三分支——:class:`ControlMode` 枚举=读到（**必须枚举**）；``None``=无回读能力或本地镜像未初始化；读取失败上抛（见 :raises:）。
        :raises Exception: 读取失败上抛，基类让本次 ``get_mode_arm`` 返回 ``None``。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _read_joint_mode_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _read_joint_mode_end(self, i: int) -> Optional[ControlMode]:
        """读 end 第 ``i`` 个电机当前的控制模式。

        实现要点：
        - 返回 :class:`ControlMode`（MIT / POSITION / VELOCITY），厂商模式码需映射（如：1→MIT、2→POSITION、3→VELOCITY）；
        - 若模式存储于电机内部寄存器，本方法通常会调用 _read_joint_param_end 方法；
        - 本方法会被基类低频刷新线程周期调用（逐关节缓存未读齐时），与发送路径共用总线；
        - **协议无模式回读时用本地镜像**：记录 ``_set_joint_mode_end`` 最近设置的模式并原样返回（恒返 ``None`` 会使 ``get_mode_end`` 核实永远失败）；
        - 舵机通常无模式概念：本地镜像恒返 ``ControlMode.POSITION``（yaml 配 ``default_mode: position``）。

        :param i: end 电机下标（``0 ≤ i < self._n_joints_end``）。
        :return: 三分支——:class:`ControlMode` 枚举=读到（**必须枚举**）；``None``=无回读能力或本地镜像未初始化；读取失败上抛（见 :raises:）。
        :raises Exception: 读取失败上抛，基类让本次 ``get_mode_end`` 返回 ``None``。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _read_joint_mode_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _read_joint_state_arm(self, i: int) -> dict:
        """读 arm 第 ``i`` 个关节电机当前状态，返回可用量字典。

        实现要点：
        - 返回硬件的 dict，基类按硬件能力从 ``q/dq/tau`` / ``temp_mos/temp_rotor`` / ``error`` 中选取（``rad / rad/s / N·m`` / ``℃ / ℃`` / ``状态码``）；
        - **缺键 = 该量硬件/协议不提供**（合法）；**键存在但值为 ``None`` = 可读但数据获取异常**；
        - ``error`` 必须归一化为全库约定：``0``=失能、``1``=使能（均正常）、``≥2``=故障码；厂商原始码由本方法负责映射；
        - 若协议有状态流（接收线程持续更新缓存），本方法可直接返回缓存值（不发总线帧），前提是缓存由接收线程维护且能判断新鲜度——**缓存过期须按数据异常处理**（键值置 ``None`` 或上抛，不得原样返回陈旧值：基类 ``t`` 为装配时刻）；
        - 半双工舵机（轮询式）在总线锁内发读帧等应答、解析后返回（每次调用产生总线流量）；
        - 单温度传感器硬件把温度映射到 ``temp_rotor``，``temp_mos`` 键缺省即可。

        :param i: arm 关节下标（``0 ≤ i < self._n_joints_arm``）。
        :return: 该关节当前可用量的字典（每值均为**标量**）。
        :raises Exception: 读取失败上抛。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _read_joint_state_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _read_joint_state_end(self, i: int) -> dict:
        """读 end 第 ``i`` 个电机当前状态，返回可用量字典。

        实现要点：
        - 返回硬件的 dict，基类按硬件能力从 ``q/dq/tau`` / ``temp_mos/temp_rotor`` / ``error`` 中选取；
        - **缺键 = 该量硬件/协议不提供**（合法）；**键存在但值为 ``None`` = 可读但数据获取异常**；
        - ``error`` 必须归一化为全库约定：``0``=失能、``1``=使能（均正常）、``≥2``=故障码；厂商原始码由本方法负责映射；
        - 若协议有状态流（接收线程持续更新缓存），本方法可直接返回缓存值（不发总线帧），前提是缓存由接收线程维护且能判断新鲜度——**缓存过期须按数据异常处理**（键值置 ``None`` 或上抛，不得原样返回陈旧值：基类 ``t`` 为装配时刻）；
        - 半双工舵机（轮询式）在总线锁内发读帧等应答、解析后返回（每次调用产生总线流量）；
        - 单温度传感器硬件把温度映射到 ``temp_rotor``，``temp_mos`` 键缺省即可。

        :param i: end 电机下标（``0 ≤ i < self._n_joints_end``）。
        :return: 该电机当前可用量的字典（每值均为**标量**）。
        :raises Exception: 读取失败上抛。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _read_joint_state_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    # ============================================================
    # 第三部分：写入（6 个抽象方法）
    # ============================================================
    def _write_joint_param_arm(self, i: int, key: str, value: float) -> None:
        """写 arm 第 ``i`` 个关节的电机参数寄存器为 ``value``。

        实现要点：
        - ``key`` 映射与读参数共用寄存器映射表；只读参数（如序列号/硬件版本）应上抛拒绝；
        - ``value`` 为 float 标量；请求-应答式协议建议等应答确认写成功再返回；

        :param i: arm 关节下标（``0 ≤ i < self._n_joints_arm``）。
        :param key: 参数名（str，映射同读参数）。
        :param value: 目标值（float 标量）。
        :raises Exception: 写入失败/超时上抛。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _write_joint_param_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _write_joint_param_end(self, i: int, key: str, value: float) -> None:
        """写 end 第 ``i`` 个电机的参数寄存器为 ``value``。

        实现要点：
        - ``key`` 映射与读参数共用寄存器映射表；只读参数（如序列号/硬件版本）应上抛拒绝；
        - ``value`` 为 float 标量；请求-应答式协议建议等应答确认写成功再返回；

        :param i: end 电机下标（``0 ≤ i < self._n_joints_end``）。
        :param key: 参数名（str，映射同读参数）。
        :param value: 目标值（float 标量）。
        :raises Exception: 写入失败/超时上抛。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _write_joint_param_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _set_joint_mode_arm(self, i: int, mode: ControlMode) -> None:
        """设置 arm 第 ``i`` 个关节电机的控制模式（写模式寄存器 + 配置增益寄存器）。

        实现要点：
        - 写模式寄存器（寄存器位置 ↔ 控制模式码 ↔ MIT/POSITION/VELOCITY）；
        - 若模式存储于电机内部寄存器，本方法通常会调用 _write_joint_param_arm 方法；
        - **切到 POSITION / VELOCITY 时须一并写入 cfg ``POS_VEL`` 的电机内部增益寄存器**（``pos_kp``/``pos_ki``/``vel_kp``/``vel_ki``），
          取值直接用基类成员 ``self._pos_kp_arm[i]`` / ``self._pos_ki_arm[i]`` / ``self._vel_kp_arm[i]`` / ``self._vel_ki_arm[i]``
          （已自 cfg 逐关节解析为 ``(n,)``；缺配置为 NaN 的增益不写该寄存器、沿用电机内部值——写前按需判断）；**切到 MIT 不用写增益**；
        - 舵机通常无模式寄存器：按需实现 POSITION / VELOCITY 分支和 MIT 分支（MIT 常映射为力矩/电流模式），实现的同时更新 ``_read_joint_mode_arm`` 用的本地镜像；
        - 基类写入成功后乐观更新模式缓存并返回（无静置无读回——经 ``get_mode_arm`` 读回纠正），失败上抛即可。

        :param i: arm 关节下标（``0 ≤ i < self._n_joints_arm``）。
        :param mode: 目标控制模式（:class:`ControlMode` 枚举，基类已做类型检查）。
        :raises Exception: 设置失败上抛。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _set_joint_mode_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _set_joint_mode_end(self, i: int, mode: ControlMode) -> None:
        """设置 end 第 ``i`` 个电机的控制模式（写模式寄存器 + 配置增益寄存器）。

        实现要点：
        - 写模式寄存器（寄存器位置 ↔ 控制模式码 ↔ MIT/POSITION/VELOCITY）；
        - 若模式存储于电机内部寄存器，本方法通常会调用 _write_joint_param_end 方法；
        - **切到 POSITION / VELOCITY 时须一并写入 cfg ``POS_VEL`` 的电机内部增益寄存器**（``pos_kp``/``pos_ki``/``vel_kp``/``vel_ki``），
          取值直接用基类成员 ``self._pos_kp_end[i]`` / ``self._pos_ki_end[i]`` / ``self._vel_kp_end[i]`` / ``self._vel_ki_end[i]``
          （已自 cfg 逐关节解析为 ``(n,)``；缺配置为 NaN 的增益不写该寄存器、沿用电机内部值——写前按需判断）；**切到 MIT 不用写增益**；
        - 舵机通常无模式寄存器：按需实现 POSITION / VELOCITY 分支和 MIT 分支（MIT 常映射为力矩/电流模式），实现的同时更新 ``_read_joint_mode_end`` 用的本地镜像；
        - 基类写入成功后乐观更新模式缓存并返回（无静置无读回——经 ``get_mode_end`` 读回纠正），失败上抛即可。

        :param i: end 电机下标（``0 ≤ i < self._n_joints_end``）。
        :param mode: 目标控制模式（:class:`ControlMode` 枚举，基类已做类型检查）。
        :raises Exception: 设置失败上抛。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _set_joint_mode_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _set_joint_zero_arm(self, i: int) -> bool:
        """把 arm 第 ``i`` 个关节的当前位置记为零点。

        实现要点：
        - 本方法直接发设零帧；本方法通常会调用 _write_joint_param_arm 方法。

        :param i: arm 关节下标（``0 ≤ i < self._n_joints_arm``）。
        :return: 设零成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _set_joint_zero_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _set_joint_zero_end(self, i: int) -> bool:
        """把 end 第 ``i`` 个电机的当前位置记为零点。

        实现要点：
        - 本方法直接发设零帧；本方法通常会调用 _write_joint_param_end 方法。

        :param i: end 电机下标（``0 ≤ i < self._n_joints_end``）。
        :return: 设零成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _set_joint_zero_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    # ============================================================
    # 第四部分：发送（7 个抽象方法，入参均已由基类校验裁剪）
    # ============================================================
    def _send_joint_mit_arm(self, i: int, tau: float, q: float, dq: float, kp: float, kd: float) -> None:
        """向 arm 第 ``i`` 个关节电机发一帧 MIT 指令。

        实现要点：
        - 电机内部执行 ``τ = tau + kp·(q_d−q) + kd·(dq_d−dq)``，五个标量按协议位域打包；
          位域量程由型号决定（按 ``self._joint_model_arm[i]`` 查型号量程表，同族多型号共存时逐关节区分），超量程时按型号就近钳位；
        - **入参已经过基类维度/空值检查与硬限位裁剪**（取值范围见 :param: 标注），本方法无需再检；
        - 高频路径（控制环逐周期调用）：只组帧 + 发送，不要做日志、等应答等重操作；在发送锁内写帧；
        - 失败上抛异常，基类转 warn（单关节失败不阻断其余关节的本帧发送）。

        :param i: arm 关节下标（``0 ≤ i < self._n_joints_arm``）。
        :param tau: 前馈力矩标量（N·m，已裁至 ±tau_max）。
        :param q: 位置目标标量（rad，已裁至 [q_min, q_max]）。
        :param dq: 速度目标标量（rad/s，已裁至 ±dq_max）。
        :param kp: 位置增益标量（已过空值检查、未裁剪）。
        :param kd: 速度阻尼标量（已过空值检查、未裁剪）。
        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _send_joint_mit_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _send_joint_mit_end(self, i: int, tau: float, q: float, dq: float, kp: float, kd: float) -> None:
        """向 end 第 ``i`` 个电机发一帧 MIT 指令。

        实现要点：
        - 五个标量按协议位域打包（量程按 ``self._joint_model_end[i]`` 查型号量程表），超量程按型号就近钳位；
        - 入参已经过基类校验与裁剪（取值范围见 :param: 标注），无需再检；
        - 高频路径：只组帧 + 发送（发送锁内），失败上抛。

        :param i: end 电机下标（``0 ≤ i < self._n_joints_end``）。
        :param tau: 前馈力矩标量（N·m，已裁至 ±tau_max）。
        :param q: 位置目标标量（rad，已裁至 [q_min, q_max]）。
        :param dq: 速度目标标量（rad/s，已裁至 ±dq_max）。
        :param kp: 位置增益标量（已过空值检查、未裁剪）。
        :param kd: 速度阻尼标量（已过空值检查、未裁剪）。
        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _send_joint_mit_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _send_joint_position_arm(self, i: int, q: float, vlim: float, flim: float) -> None:
        """向 arm 第 ``i`` 个关节电机发一帧位置模式指令。

        实现要点：
        - ``flim`` 需按协议换算为原始单位（如标幺 0~10000），``vlim`` 同理按协议速度原始单位换算（如舵机常用 0.1°/s 或步/s）；
        - 入参已经过基类校验与裁剪，无需再检；本方法直接发送；失败上抛。

        :param i: arm 关节下标（``0 ≤ i < self._n_joints_arm``）。
        :param q: 目标位置标量（rad，已裁至 [q_min, q_max]）。
        :param vlim: 速度上限标量（rad/s，已裁至 [0, dq_max]）。
        :param flim: 归一化力矩电流上限标量（0~1，已裁）。
        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _send_joint_position_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _send_joint_position_end(self, i: int, q: float, vlim: float, flim: float) -> None:
        """向 end 第 ``i`` 个电机发一帧位置模式指令。

        实现要点：
        - ``flim`` / ``vlim`` 按协议换算为原始单位（如标幺 0~10000 / 0.1°/s 或步/s）；
        - 入参已经过基类校验与裁剪，无需再检；本方法直接发送；失败上抛。

        :param i: end 电机下标（``0 ≤ i < self._n_joints_end``）。
        :param q: 目标位置标量（rad，已裁至 [q_min, q_max]）。
        :param vlim: 速度上限标量（rad/s，已裁至 [0, dq_max]）。
        :param flim: 归一化力矩电流上限标量（0~1，已裁）。
        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _send_joint_position_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _send_joint_vel_arm(self, i: int, dq: float) -> None:
        """向 arm 第 ``i`` 个关节电机发一帧速度模式指令。

        实现要点：
        - 模式前置检查已由基类完成，本方法直接发送；失败上抛。

        :param i: arm 关节下标（``0 ≤ i < self._n_joints_arm``）。
        :param dq: 目标速度标量（rad/s，已裁至 ±dq_max）。
        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _send_joint_vel_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _send_joint_vel_end(self, i: int, dq: float) -> None:
        """向 end 第 ``i`` 个电机发一帧速度模式指令。

        实现要点：
        - 模式前置检查已由基类完成，本方法直接发送；失败上抛。

        :param i: end 电机下标（``0 ≤ i < self._n_joints_end``）。
        :param dq: 目标速度标量（rad/s，已裁至 ±dq_max）。
        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _send_joint_vel_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _send_action_end(self, action: str, **kwargs) -> None:
        """向 end 执行器发送离散动作指令（整组粒度，无单关节语义）。

        实现要点：
        - ``action`` 常见 ``"open"`` / ``"close"`` / ``"home"``，动作集与 ``kwargs`` 可选配置由子类定义（建议在本文件头或型号文档中列明）；
        - 动作的可调参数（限速/力矩/行程等）可取 ``self._cfg`` 的型号自定义顶层键或 ``self._jointscfg_end[i]`` 的自定义键（yaml 中配置，基类解析时原样保留），
          行程目标也可直取硬限位成员 ``self._joint_limits_end``（如夹爪 open=q_min / close=q_max）；
        - 本子类方法实现内部自行保证安全（如模式前置检查与自动切换，自定义参数配置限速和限流）；
        - 夹爪类动作通常：模式检查 → 切到相应模式（若不一致） → 发动作帧；失败上抛异常，基类转 warn。

        :param action: 动作名。
        :param kwargs: 动作的可选配置（由子类解释）。
        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _send_action_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    # ============================================================
    # 第五部分：清错（2 个抽象方法）
    # ============================================================
    def _clear_joint_error_arm(self, i: int) -> bool:
        """清除 arm 第 ``i`` 个关节电机的硬件错误。

        实现要点：
        - 本方法只发清除帧；部分电机清错后自动失能，属正常（清错后保持失能态，何时重新使能由上层决定）；
        - 舵机通常无清错命令：保留 ``NotImplementedError`` 桩即优雅降级。

        :param i: arm 关节下标（``0 ≤ i < self._n_joints_arm``）。
        :return: 清除指令发送成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _clear_joint_error_arm：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")

    def _clear_joint_error_end(self, i: int) -> bool:
        """清除 end 第 ``i`` 个电机的硬件错误（仅发错误清除指令）。

        实现要点：
        - 本方法只发清除帧；部分电机清错后自动失能，属正常（清错后保持失能态，何时重新使能由上层决定）；
        - 舵机通常无清错命令：保留 ``NotImplementedError`` 桩即优雅降级。

        :param i: end 电机下标（``0 ≤ i < self._n_joints_end``）。
        :return: 清除指令发送成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError(f"{self._name or 'backend_<型号>.py'} - _clear_joint_error_end：功能缺失：子类内核方法未实现，若需实现请参照backend子类实现模板约束实现。")
