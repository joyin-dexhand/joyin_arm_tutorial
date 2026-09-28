"""``BackendTemplate`` —— Backend 子类实现模板（复制本文件为 ``backend_<型号>.py`` 后填写）。

与 ``config/joyarm_template.yaml`` 配套使用，五步接入新型号：
1. 复制本文件为 ``joyarm_core/backend/backend_<型号>.py``，类名 ``BackendTemplate`` 改为
   ``Backend<型号>``（文件名小写、类名驼峰，如 ``backend_dm.py`` ↔ ``BackendDm``）；
2. 复制 ``config/joyarm_template.yaml`` 为 ``config/<型号>.yaml``，填好 ``backend:`` 段；
3. 按本文件各方法的注释实现全部 29 个抽象方法，可分步实现、逐步联调；
4. 在 ``backend/__init__.py`` 顶部导入子类，并在 ``REGISTRY`` 加一行注册；
5. ``<型号>.yaml`` 的 ``backend.name`` 填子类 backend 的注册名。

参考实现：最小可运行子类见 ``backend_dm.py`` 的 ``BackendDM``；
  各抽象方法的完整契约见 ``backend.py`` 对应分节的 docstring。

架构约束：
- 子类**不得新增公开方法/属性**：JoyArm 上层只面向 ``Backend`` 基类编程，
  型号专属功能一律放 ``_`` 前缀私有层（协议类、常量表、助手函数均不导出）；
- 基类负责全部编排（逐关节循环、维度/空值检查、越限裁剪、连接与模式前置检查、限频日志、
  异常降级），子类只做单关节/单总线的协议操作——**不要在子类重复基类已做的检查与裁剪**；
- 失败约定：抽象方法失败上抛异常或返回 ``False``/``None`` 即可，基类统一转限频 warn
  （运行期日志按语义通道限频，机制见基类 ``_warn_throttled``/``_info_throttled``）。
"""
from __future__ import annotations

import logging
from typing import Optional

from ..utils.types import ControlMode
from .backend import Backend

__all__ = ["BackendTemplate"]

logger = logging.getLogger("joyarm_core.backend_template")  # 改型号时同步改名（如 ...backend_**）


class BackendTemplate(Backend):
    """<型号名> 硬件后端模板（复制后改名为 ``Backend<型号>``）。

    ── 子类自建成员（协议层私有资源，``__init__`` 内初始化，全部 ``_`` 前缀）按协议需要自建，典型三件：
        电机句柄表（下标对齐 ``_jointscfg_*``）、总线封装（``dict[channel → bus]``，arm/end 同 channel 共享一条）、接收线程 + 发送锁 + 请求-应答同步 Event。
    基类内部助手 ``_n()`` / ``_jname()``（关节数 / 日志用关节名）可随时调用。

    ── 线程模型（并发调用方，总线访问务必线程安全）──
    - 应用线程：上层直接调用公开方法（send_* 高频路径在控制线程）；
    - 基类低频刷新线程（daemon）：周期调用 ``_read_joint_mode_*`` / ``_read_joint_state_*``；
    - 因此同一总线会被并发读写：发送加锁、收帧独立线程、请求-应答用 Event 同步。

    无 end 的型号：yaml ``end:`` 段留空（关节数 0）即可，end 系列方法必须存在实现以满足抽象方法强制。
    """

    def __init__(self, cfg: dict) -> None:
        """子类构造：先 ``super().__init__(cfg)``（不可省），再初始化协议层私有资源。

        基类 ``__init__`` 已完成：cfg 解析（总线参数/关节配置/硬限位/发送默认/POS_VEL 增益/
        可选运行参数）、状态对象定维、低频刷新线程启动。子类在此**只建协议层对象，不做任何实际总线 I/O**——
        真正打开总线在 ``_connect_*`` 内完成，构造永远成功（无硬件也能实例化，离线可用）。

        :param cfg: yaml ``backend:`` 段整体（含/不含 ``name`` 均可，基类自取存 ``self._name``）。
        """
        super().__init__(cfg)
        # ---- 基类已初始化、子类直接消费的成员（勿重复解析，逐关节键缺省为 None）----
        # self._cfg                                  yaml backend 段整体（型号自定义顶层键在此取）
        # self._channel_arm / _channel_end           总线通道（cfg.arm/end.channel）
        # self._baud_arm / _baud_end                 总线波特率（cfg.arm/end.baud_rate）
        # self._protocol_arm / _protocol_end         协议标识（cfg.arm/end.protocol）
        # self._jointscfg_arm[i] / _jointscfg_end[i]     逐关节配置字典（下标即入参 i；其余型号自定义键在此自取）
        # self._joint_motor_id_arm[i] / _joint_motor_id_end[i]        电机总线地址（joints[].motor_id）
        # self._joint_feedback_id_arm[i] / _joint_feedback_id_end[i]  应答帧标识（joints[].feedback_id）
        # self._joint_model_arm[i] / _joint_model_end[i]              电机型号（joints[].model）
        # self._pos_kp/_pos_ki/_vel_kp/_vel_ki × arm/end（8 个 (n,)）  POS_VEL 增益，设模式写增益寄存器时直取（缺配置 NaN）
        # self._joint_state_arm / _joint_state_end   私有状态对象（属性 joint_state_* 对外只读）

        # ---- 子类私有成员示例（按型号删改；约定：全部 _ 前缀，不新增公开成员）----
        # self._motors_arm = self._build_motors("arm")  # arm 电机句柄表（下标 = _jointscfg_arm 下标）
        # self._motors_end = self._build_motors("end")  # end 电机句柄表
        # self._buses = {}                              # channel → 总线封装（arm/end 同 channel 共享一条）

    # ============================================================
    # 第一部分：生命周期——连接与使能（8 个抽象方法）
    # ============================================================
    # =============== 连接（connect / disconnect） ===============
    def _connect_arm(self, channel, protocol) -> bool:
        """连接 arm 总线：打开 ``channel``（串口设备名 / CAN 接口名），按需创建接收线程与发送锁。

        实现要点：
        - 波特率取 ``self._baud_arm``（cfg ``arm.baud_rate``）；``protocol`` 通常为信息性标识，协议固定时可直接忽略、用子类自带默认；
        - arm 与 end 的 ``channel`` 相同时共享一条总线：先连的一方创建、后连的一方复用（只打开总线、不做任何电机操作）
          （建议用 ``dict[channel → 总线封装]`` 管理，避免重复打开；两组连接顺序由上层调用决定）。

        :return: 连接成功 ``True``；失败返回 ``False`` 或直接上抛异常（基类统一转 warn，
            ``is_connected_arm`` 停留 ``None`` 表示未知态）。
        """
        raise NotImplementedError("backend_template.py - _connect_arm：模板方法未实现，请按注释实现")

    def _connect_end(self, channel, protocol) -> bool:
        """连接 end 总线：打开 ``channel``，按需创建接收线程与发送锁。

        实现要点：
        - 波特率取 ``self._baud_end``；``protocol`` 为信息性标识；
        - 与 arm 的 ``channel`` 相同时**必须复用 arm 已打开的总线**（一个串口句柄 + 一个接收
          线程 + 一把发送锁），不同时才另开一条独立总线（只打开总线、不做任何电机操作）。

        :return: 连接成功 ``True``；失败返回 ``False`` 或直接上抛异常。
        """
        raise NotImplementedError("backend_template.py - _connect_end：模板方法未实现，请按注释实现")

    def _disconnect_arm(self) -> None:
        """断开 arm 总线：只关闭子类自建的协议层资源。

        实现要点：
        - 电机失能已由基类在调用本方法前完成（先失能再断连），本方法不用管电机；
        - 需要关闭的典型资源：接收线程（置停止标志 → join → 关句柄）、串口/CAN 句柄；
        - **共享总线的关断要谨慎**：断开 arm 时若 end 仍连接着同一条总线（``is_connected_end``
          不为 ``False``），只解除 arm 的关联、保留总线本体，待最后一组断开时才真正关闭；
        - 重复调用应安全（幂等）；失败上抛，基类转 warn。
        """
        raise NotImplementedError("backend_template.py - _disconnect_arm：模板方法未实现，请按注释实现")

    def _disconnect_end(self) -> None:
        """断开 end 总线：只关闭子类自建的协议层资源。

        实现要点：
        - 电机失能已由基类在调用本方法前完成（先失能再断连），本方法不用管电机；
        - 需要关闭的典型资源：接收线程（置停止标志 → join → 关句柄）、串口/CAN 句柄；
        - **共享总线的关断要谨慎**：断开 end 时若 arm 仍连接着同一条总线（``is_connected_arm``
          不为 ``False``），只解除 end 的关联、保留总线本体，待最后一组断开时才真正关闭；
        - 重复调用应安全（幂等）；失败上抛，基类转 warn。
        """
        raise NotImplementedError("backend_template.py - _disconnect_end：模板方法未实现，请按注释实现")

    # ==================== 使能（enable / disable） ====================
    def _enable_joint_arm(self, i: int) -> bool:
        """使能 arm 第 ``i`` 个关节电机（上电进入闭环，开始响应指令）。

        实现要点：
        - ``i`` 为 ``self._jointscfg_arm`` 的下标；电机总线地址取 ``self._joint_motor_id_arm[i]``；
        - 请求-应答式协议建议发使能帧后等应答并核对状态码（``error`` 变为 1=使能），超时或应答异常按失败处理；
        - 基类逐关节调用并汇总（单个失败不阻断其余关节）。

        :return: 使能成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError("backend_template.py - _enable_joint_arm：模板方法未实现，请按注释实现")

    def _enable_joint_end(self, i: int) -> bool:
        """使能 end 第 ``i`` 个电机（上电进入闭环）。

        实现要点：
        - ``i`` 为 ``self._jointscfg_end`` 的下标；电机总线地址取 ``self._joint_motor_id_end[i]``；
        - 请求-应答式协议发使能帧后等应答并核对状态码（``error`` 变为 1=使能），超时按失败处理；
        - 基类逐电机调用并汇总（单个失败不阻断其余电机）。

        :return: 使能成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError("backend_template.py - _enable_joint_end：模板方法未实现，请按注释实现")

    def _disable_joint_arm(self, i: int) -> bool:
        """失能 arm 第 ``i`` 个关节电机（退出闭环，电机出力归零）。

        实现要点：
        - 失能后电机无保持力（重力负载会下坠），安全性由上层负责，本方法只发协议帧；
        - 请求-应答式协议可等应答核对（``error`` 变为 0=失能），超时按失败处理。

        :return: 失能成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError("backend_template.py - _disable_joint_arm：模板方法未实现，请按注释实现")

    def _disable_joint_end(self, i: int) -> bool:
        """失能 end 第 ``i`` 个电机（退出闭环，电机出力归零）。

        实现要点：
        - 失能后末端无保持力（夹爪可能松开），安全性由上层负责；
        - 请求-应答式协议可等应答核对状态码（``error`` 变为 0=失能），超时按失败处理。

        :return: 失能成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError("backend_template.py - _disable_joint_end：模板方法未实现，请按注释实现")

    # ============================================================
    # 第二部分：读取（6 个抽象方法）
    # ============================================================
    def _read_joint_param_arm(self, i: int, key: str):
        """读 arm 第 ``i`` 个关节的电机参数寄存器，返回参数值。

        实现要点：
        - ``key`` 为参数名，映射到厂商寄存器由子类定义——建议维护模块级常量表
          ``_PARAM_RIDS_arm: dict[str, int]``（参数名 → 寄存器 RID，如 ``"pos_kp"`` → 27）；
          若 arm 与 end 共用同一套寄存器表，统一为 ``_PARAM_RIDS``；
        - 寄存器值分整型（uint32）与浮点（float32）两类，按厂商手册解析；
          float32 读回与写入值必有低位差异，基类核对已用容差，子类无需处理；
        - 请求-应答式协议建议发读帧后按 RID 等应答（Event 同步），失败/超时上抛。

        :return: 该参数的当前值（int 或 float）。
        :raises Exception: 读取失败/超时上抛，基类让整组读取作废返回 ``None``。
        """
        raise NotImplementedError("backend_template.py - _read_joint_param_arm：模板方法未实现，请按注释实现")

    def _read_joint_param_end(self, i: int, key: str):
        """读 end 第 ``i`` 个电机的参数寄存器，返回参数值。

        实现要点：
        - ``key`` 为参数名，映射到厂商寄存器由子类定义——建议维护模块级常量表
          ``_PARAM_RIDS_end: dict[str, int]``（参数名 → 寄存器 RID，如 ``"pos_kp"`` → 27）；
          若 end 与 arm 共用同一套寄存器表，统一为 ``_PARAM_RIDS``；
        - 寄存器值分整型（uint32）与浮点（float32）两类，按厂商手册解析；
          float32 读回与写入值必有低位差异，基类核对已用容差，子类无需处理；
        - 请求-应答式协议建议发读帧后按 RID 等应答（Event 同步），失败/超时上抛。

        :return: 该参数的当前值（int 或 float）。
        :raises Exception: 读取失败/超时上抛。
        """
        raise NotImplementedError("backend_template.py - _read_joint_param_end：模板方法未实现，请按注释实现")

    def _read_joint_mode_arm(self, i: int) -> Optional[ControlMode]:
        """读 arm 第 ``i`` 个关节电机当前的控制模式。

        实现要点：
        - 返回 :class:`ControlMode`（MIT / POSITION / VELOCITY），厂商模式码需映射
          （如：1→MIT、2→POSITION、3→VELOCITY）；
        - 若模式存储于电机内部寄存器，本方法通常会调用 _read_joint_param_arm 方法；
        - 本方法会被基类低频刷新线程周期调用（逐关节缓存未读齐时），与发送路径共用总线。

        :return: 当前控制模式；读不到返回 ``None``。
        :raises Exception: 读取失败上抛。
        """
        raise NotImplementedError("backend_template.py - _read_joint_mode_arm：模板方法未实现，请按注释实现")

    def _read_joint_mode_end(self, i: int) -> Optional[ControlMode]:
        """读 end 第 ``i`` 个电机当前的控制模式。

        实现要点：
        - 返回 :class:`ControlMode`（MIT / POSITION / VELOCITY），厂商模式码需映射
          （如：1→MIT、2→POSITION、3→VELOCITY）；
        - 若模式存储于电机内部寄存器，本方法通常会调用 _read_joint_param_end 方法；
        - 本方法会被基类低频刷新线程周期调用（逐关节缓存未读齐时），与发送路径共用总线。

        :return: 当前控制模式；读不到返回 ``None``。
        :raises Exception: 读取失败上抛。
        """
        raise NotImplementedError("backend_template.py - _read_joint_mode_end：模板方法未实现，请按注释实现")

    def _read_joint_state_arm(self, i: int) -> dict:
        """读 arm 第 ``i`` 个关节电机当前状态，返回可用量字典。

        实现要点：
        - 返回硬件的 dict，基类按硬件能力从 ``q`` / ``dq`` / ``tau`` / ``temp_mos`` / ``temp_rotor`` /
          ``error`` 中选取（单位： rad / rad/s / N·m / ℃ / ℃ / 状态码）；
        - **缺键 = 该量硬件/协议不提供**（合法：基类把该字段整组置 ``None``）；
          **键存在但值为 ``None`` = 可读但数据获取异常**（基类跳过本轮并限频 warn）；
        - ``error`` 必须归一化为全库约定：``0``=失能、``1``=使能（均正常）、``≥2``=故障码；厂商原始码由本方法负责映射；
        - 若协议有状态流（接收线程持续更新缓存），本方法可直接返回缓存值（不发总线帧），前提是缓存由接收线程维护且能判断新鲜度。

        :return: 该关节当前可用量的字典。
        :raises Exception: 读取失败上抛。
        """
        raise NotImplementedError("backend_template.py - _read_joint_state_arm：模板方法未实现，请按注释实现")

    def _read_joint_state_end(self, i: int) -> dict:
        """读 end 第 ``i`` 个电机当前状态，返回可用量字典。

        实现要点：
        - 返回硬件的 dict，基类按硬件能力从 ``q`` / ``dq`` / ``tau`` / ``temp_mos`` / ``temp_rotor`` /
          ``error`` 中选取（单位： rad / rad/s / N·m / ℃ / ℃ / 状态码）；
        - **缺键 = 该量硬件/协议不提供**（合法：基类把该字段整组置 ``None``）；
          **键存在但值为 ``None`` = 可读但数据获取异常**（基类跳过本轮并限频 warn）；
        - ``error`` 必须归一化为全库约定：``0``=失能、``1``=使能（均正常）、``≥2``=故障码；厂商原始码由本方法负责映射；
        - 若协议有状态流（接收线程持续更新缓存），本方法可直接返回缓存值（不发总线帧），前提是缓存由接收线程维护且能判断新鲜度。

        :return: 该电机当前可用量的字典。
        :raises Exception: 读取失败上抛。
        """
        raise NotImplementedError("backend_template.py - _read_joint_state_end：模板方法未实现，请按注释实现")

    # ============================================================
    # 第三部分：写入（6 个抽象方法）
    # ============================================================
    def _write_joint_param_arm(self, i: int, key: str, value: float) -> None:
        """写 arm 第 ``i`` 个关节的电机参数寄存器为 ``value``。

        实现要点：
        - ``key`` 映射与读参数共用寄存器映射表；只读参数（如序列号/硬件版本）应上抛拒绝；
        - 基类负责维度校验、写入后静置与同键读回核对（容差比较），本方法只发写入帧；
          请求-应答式协议建议等应答确认写成功再返回；
        - 部分参数要求失能状态写入（本方法在"失能态写入"语境下被调用，时序天然合规）。

        :raises Exception: 写入失败/超时上抛。
        """
        raise NotImplementedError("backend_template.py - _write_joint_param_arm：模板方法未实现，请按注释实现")

    def _write_joint_param_end(self, i: int, key: str, value: float) -> None:
        """写 end 第 ``i`` 个电机的参数寄存器为 ``value``。

        实现要点：
        - ``key`` 映射与读参数共用寄存器映射表；只读参数（如序列号/硬件版本）应上抛拒绝；
        - 基类负责维度校验、写入后静置与同键读回核对（容差比较），本方法只发写入帧；
          请求-应答式协议建议等应答确认写成功再返回；
        - 部分参数要求失能状态写入（本方法在"失能态写入"语境下被调用，时序天然合规）。

        :raises Exception: 写入失败/超时上抛。
        """
        raise NotImplementedError("backend_template.py - _write_joint_param_end：模板方法未实现，请按注释实现")

    def _set_joint_mode_arm(self, i: int, mode: ControlMode) -> None:
        """设置 arm 第 ``i`` 个关节电机的控制模式（写模式寄存器 + 配置增益寄存器）。

        实现要点：
        - 写模式寄存器（寄存器位置 ↔ 控制模式码 1/2/3 ↔ MIT/POSITION/VELOCITY）；
        - 若模式存储于电机内部寄存器，本方法通常会调用 _write_joint_param_arm 方法；
        - **切到 POSITION / VELOCITY 时必须一并写入 cfg ``POS_VEL`` 的增益寄存器**
          （``pos_kp``/``pos_ki``/``vel_kp``/``vel_ki``），取值直接用基类成员
          ``self._pos_kp_arm[i]`` / ``self._pos_ki_arm[i]`` / ``self._vel_kp_arm[i]`` /
          ``self._vel_ki_arm[i]``（已自 cfg 逐关节解析为 ``(n,)``，缺配置为 NaN，不更改增益寄存器
          ——写前按需判断）；**切到 MIT 不用写增益**；
        - 基类设完全组后会静置并经 ``get_mode_arm`` 读回核对，本方法失败上抛即可。

        :raises Exception: 设置失败上抛。
        """
        raise NotImplementedError("backend_template.py - _set_joint_mode_arm：模板方法未实现，请按注释实现")

    def _set_joint_mode_end(self, i: int, mode: ControlMode) -> None:
        """设置 end 第 ``i`` 个电机的控制模式（写模式寄存器 + 配置增益寄存器）。

        实现要点：
        - 写模式寄存器（寄存器位置 ↔ 控制模式码 1/2/3 ↔ MIT/POSITION/VELOCITY）；
        - 若模式存储于电机内部寄存器，本方法通常会调用 _write_joint_param_end 方法；
        - **切到 POSITION / VELOCITY 时必须一并写入 cfg ``POS_VEL`` 的增益寄存器**
          （``pos_kp``/``pos_ki``/``vel_kp``/``vel_ki``），取值直接用基类成员
          ``self._pos_kp_end[i]`` / ``self._pos_ki_end[i]`` / ``self._vel_kp_end[i]`` /
          ``self._vel_ki_end[i]``（已自 cfg 逐关节解析为 ``(n,)``，缺配置为 NaN，不更改增益寄存器
          ——写前按需判断）；**切到 MIT 不用写增益**；
        - 基类设完全组后会静置并经 ``get_mode_end`` 读回核对，本方法失败上抛即可。

        :raises Exception: 设置失败上抛。
        """
        raise NotImplementedError("backend_template.py - _set_joint_mode_end：模板方法未实现，请按注释实现")

    def _set_joint_zero_arm(self, i: int) -> bool:
        """把 arm 第 ``i`` 个关节的当前位置记为零点。

        实现要点：
        - 部分电机要求失能且无故障才能设零；本方法在"失能态写入"语境下被调用，一般可直接发设零帧；

        :return: 设零成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError("backend_template.py - _set_joint_zero_arm：模板方法未实现，请按注释实现")

    def _set_joint_zero_end(self, i: int) -> bool:
        """把 end 第 ``i`` 个电机的当前位置记为零点。

        实现要点：
        - 部分电机要求失能且无故障才能设零；本方法在"失能态写入"语境下被调用，一般可直接发设零帧；

        :return: 设零成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError("backend_template.py - _set_joint_zero_end：模板方法未实现，请按注释实现")

    # ============================================================
    # 第四部分：发送（7 个抽象方法，入参均已由基类校验裁剪）
    # ============================================================
    def _send_joint_mit_arm(self, i: int, tau: float, q: float, dq: float,
                            kp: float, kd: float) -> None:
        """向 arm 第 ``i`` 个关节电机发一帧 MIT 指令。

        实现要点：
        - 电机内部执行 ``τ = tau + kp·(q_d−q) + kd·(dq_d−dq)``，五个标量按协议位域打包；
          位域量程由型号决定，超量程时按型号就近钳位（kp/kd 无硬限位，基类不裁）；
        - **入参已经过基类维度/空值检查与硬限位裁剪**（tau/q/dq 已在限位内），本方法无需再检；
        - 高频路径（控制环逐周期调用）：只组帧 + 发送，不要做日志、等应答等重操作；在发送锁内写帧；
        - 失败上抛异常，基类转 warn（单关节失败不阻断其余关节的本帧发送）。

        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError("backend_template.py - _send_joint_mit_arm：模板方法未实现，请按注释实现")

    def _send_joint_mit_end(self, i: int, tau: float, q: float, dq: float,
                            kp: float, kd: float) -> None:
        """向 end 第 ``i`` 个电机发一帧 MIT 指令。

        实现要点：
        - 五个标量按协议位域打包，超量程按型号就近钳位；入参已经过基类校验与裁剪，无需再检；
        - 高频路径：只组帧 + 发送（发送锁内），失败上抛。

        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError("backend_template.py - _send_joint_mit_end：模板方法未实现，请按注释实现")

    def _send_joint_position_arm(self, i: int, q: float, vlim: float, flim: float) -> None:
        """向 arm 第 ``i`` 个关节电机发一帧位置模式指令。

        实现要点：
        - ``q`` 目标位置（rad）、``vlim`` 速度上限（rad/s）、``flim`` 归一化力矩电流上限（0~1）；
          ``flim`` 需按协议换算为原始单位（如标幺 0~10000）；
        - 入参已经过基类校验与裁剪，无需再检；模式前置检查已由基类完成，本方法直接发送；失败上抛。

        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError("backend_template.py - _send_joint_position_arm：模板方法未实现，请按注释实现")

    def _send_joint_position_end(self, i: int, q: float, vlim: float, flim: float) -> None:
        """向 end 第 ``i`` 个电机发一帧位置模式指令。

        实现要点：
        - ``q`` 目标位置（rad）、``vlim`` 速度上限（rad/s）、``flim`` 归一化力矩电流上限（0~1）；
          ``flim`` 需按协议换算为原始单位（如标幺 0~10000）；
        - 入参已经过基类校验与裁剪，无需再检；模式前置检查已由基类完成，本方法直接发送；失败上抛。

        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError("backend_template.py - _send_joint_position_end：模板方法未实现，请按注释实现")

    def _send_joint_vel_arm(self, i: int, dq: float) -> None:
        """向 arm 第 ``i`` 个关节电机发一帧速度模式指令。

        实现要点：
        - ``dq`` 目标速度（rad/s），入参已经过基类校验与裁剪（±dq_max 内），无需再检；
        - 模式前置检查已由基类完成（须处于 VELOCITY 模式），直接发送；失败上抛。

        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError("backend_template.py - _send_joint_vel_arm：模板方法未实现，请按注释实现")

    def _send_joint_vel_end(self, i: int, dq: float) -> None:
        """向 end 第 ``i`` 个电机发一帧速度模式指令。

        实现要点：
        - ``dq`` 目标速度（rad/s），入参已经过基类校验与裁剪（±dq_max 内），无需再检；
        - 模式前置检查已由基类完成（须处于 VELOCITY 模式），直接发送；失败上抛。

        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError("backend_template.py - _send_joint_vel_end：模板方法未实现，请按注释实现")

    def _send_action_end(self, action: str, **kwargs) -> None:
        """向 end 执行器发送离散动作指令（整组粒度，无单关节语义）。

        实现要点：
        - ``action`` 常见 ``"open"`` / ``"close"`` / ``"home"``，动作集与 ``kwargs`` 可选配置
          由子类定义（建议在本文件头或型号文档中列明）；
        - 本方法不设模式前置检查（离散动作无对应 ControlMode），由子类实现内部自行保证安全
          （如模式前置检查与切换，自定义参数配置限速限流）；
        - 夹爪类动作通常：模式检查 → 切到相应模式 → 发动作帧；
        - 失败上抛异常，基类转 warn。

        :param action: 动作名。
        :param kwargs: 动作的可选配置（由子类解释）。
        :raises Exception: 发送失败上抛。
        """
        raise NotImplementedError("backend_template.py - _send_action_end：模板方法未实现，请按注释实现")

    # ============================================================
    # 第五部分：清错（2 个抽象方法）
    # ============================================================
    def _clear_joint_error_arm(self, i: int) -> bool:
        """清除 arm 第 ``i`` 个关节电机的硬件错误（发错误清除指令）。

        实现要点：
        - 基类流程为：失能门禁（须 ``is_abled`` 为 ``False``）→ 逐关节发清错帧；无静置、无读回验证，
          清错结果由上层 ``get_error_arm`` / ``check_error_arm`` 确认；本方法只发清除帧；
        - 部分电机清错后自动失能，属正常（清错后保持失能态，何时重新使能由上层决定）。

        :return: 清除指令发送成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError("backend_template.py - _clear_joint_error_arm：模板方法未实现，请按注释实现")

    def _clear_joint_error_end(self, i: int) -> bool:
        """清除 end 第 ``i`` 个电机的硬件错误（发错误清除指令）。

        实现要点：
        - 基类流程为：失能门禁（须 ``is_abled`` 为 ``False``）→ 逐电机发清错帧；无静置、无读回验证，
          清错结果由上层 ``get_error_end`` / ``check_error_end`` 确认；本方法只发清除帧；
        - 部分电机清错后自动失能，属正常（清错后保持失能态，何时重新使能由上层决定）。

        :return: 清除指令发送成功 ``True``；失败返回 ``False`` 或上抛异常。
        """
        raise NotImplementedError("backend_template.py - _clear_joint_error_end：模板方法未实现，请按注释实现")
