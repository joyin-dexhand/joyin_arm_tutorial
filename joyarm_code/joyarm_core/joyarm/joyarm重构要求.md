# joyarm 重构要求

基于 `joyarm_core/` 的最新状态，重构 `joyarm_core/joyarm/` 。joyarm的重构务必严格遵循如下要求（若存在疑问、歧义、不清晰的地方或有更好更优雅的方案，务必询问我决策，不要自以为是地擅自替我决定）：

## 1 总体要求

1. 任务执行前务必充分探索和了解 `joyarm_core`的功能和代码情况，然后进行规划和执行；
2. 本次重构任务属于复杂任务，务必使用`子agent`和`工作流`执行，发挥并行开发的优势；
3. 本次重构任务可参考`joyarm.py`但无需受其现况的制约（原实现已过时）；

## 2 包初始化和工厂功能

1. 通过`joyArm`方法作为工厂方法生产创建joyarm对象，传入`name：str`参数（joyarm的型号）。流程为：
   1. 验证`config/`中存在`<name>.yaml`配置文件，且文件非空，否则抛出具体错误原因并构造失败。同时给出可用型号列表（见下面的`listArm`）。
   2. 验证`config/<name>.yaml`配置文件中名为`backend.name`的硬件后端在`backend/`中存在（即`backend/__init__.py`的REGISTRY注册表中存在`backend.name`键），否则抛出具体错误原因并构造失败。
   3. 验证`robot_model/`下存在名为`<name>`的文件夹，且存在`robot_model/<name>/meshes/`和`robot_model/<name>/urdf/`文件夹和`robot_model/<name>/urdf/<name>.urdf`文件，否则抛出具体错误原因并构造失败。
   4. 均通过后则创建出joyarm对象并info创建成功。创建失败则列出所有可行型号（见下面的`listArm`）。
2. 通过`listArm`方法列出所有可行的joyarm型号，可作为函数被调用打印。可行的型号需要同时满足：
   1. `config/`中存在`<name>.yaml`配置文件，且文件非空，文件内包含joyarm、robotics、backend三键。此时的`<name>`即为机械臂型号名。
   2. `config/<name>.yaml`配置文件中名为`backend.name`的硬件后端在`backend/`中存在（即`backend/__init__.py`的REGISTRY注册表中存在`backend.name`键）。
   3. `robot_model/`下存在名为`<name>`的文件夹，且存在`robot_model/<name>/meshes/`和`robot_model/<name>/urdf/`文件夹和`robot_model/<name>/urdf/<name>.urdf`文件。

## 3 joyarm 功能

1. JoyArm 类的注释分两级，第一级为三行（三行banner，等号对齐），第二级为一行（一行banner，等号对齐），注释规则可参考backend基类和joyarm的现状；
2. 一级功能区 —— 类和对象的初始化 `__init__(name:str)`（类内成员全为私有成员）（初始化和校验失败则直接抛出具体错误原因并构造失败）；
   1. 先定义私有成员变量并赋默认值，再基于_cfg进行成员对象的初始化和成员变量的赋值（包括但不限于）：
      1. _cfg（按 name 加载的<config/name.yaml>配置文件）：校验joyarm、robotics、backend三字段均存在，校验joyarm.home_arm/end和joyarm.joint_limits_soft_arm/end维度等于backend.arm/end的自由度数，校验robotics六类算法下的各个子求解器均存在（与robotics各子包内的算法注册表对比）且plan/ctrl求解器均配置了plan/ctrl_hz参数（均为整数值且至少为1），cfg.backend与urdf对比校验：cfg的link和joint命名顺序（joint/link0～n，joint/link_end0～n） + 名称与urdf一致、cfg.backend的各关节限位=urdf的joint的限位（容差1e-4）、cfg.backend的各关节限位q_min ≤ q_max、cfg.backend关节名无重复、urdf的joint和link名无重复，joyarm.joint_limits_soft_arm/end需位于backend.arm/end 各关节的硬限位内，joyarm.tcp_limits.workspace_box为二行三列，第一行逐元素小于第二行且v_lin/ang_max和f/t_max均非空且大于0，需要类外访问`@property`；
      2. _pin_model(机械臂pin模型，仅arm的joint，end的joint锁定在零位（需适配所有机械臂型号，非dm单型号），link需全部加载（保证整臂动力学质量惯量的正确）。先cfg.backend + urdf 自检通过再构建，并验证cfg.joyarm.frame_end在pin_model存在。需要类外访问`@property`)；
      3. _zero_arm/end(按自由度数进行全零的创建，需要类外访问`@property`)；
      4. _neutral_arm/end（按软限位逐joint取中间值，需要类外访问`@property`）；
      5. _home_arm/end（加载自_cfg.joyarm.home_arm/end，需校验在cfg.joyarm.joint_limits_soft_arm/end内，需要类外访问`@property`）；
      6. _plan/ctrl_hz（默认10hz/100hz，由cfg.robotics的plan/ctrl求解器在set_solver执行前按子求解器的plan/ctrl_hz参数进行设置）；
      7. joint_limits_hard_arm/end(加载自cfg.backend.arm/end.joints，需要类外访问`@property`)；
      8. joint_limits_soft_arm/end(加载自cfg.joyarm.joint_limits_soft_arm/end，需要类外访问`@property`)；
      9. _tcp_limits（加载自cfg.joyarm.tcp_limits）；
      10. _backend(基于cfg.backend.name创建对应型号的硬件后端对象)；
      11. _n_arm/end（需要类外访问`@property`，需校验）；
      12. _joint_names_arm/end（加载自cfg.backend.arm/end.joints各名）；
      13. _joint_index_arm/end（键为joint_name，值为pin_model的index）
      14. _robotics_solvers（加载自cfg.robotics，双层字典（第一键选类型，第二键选对象名，值为各求解器对象））；
      15. _activated_solver（字典，指向_robotics_solvers子求解器对象，六键六值，调用示例：_activated_solver(“fkine”).solve(q)）；
      16. _target_trajframe(规划器计算时传入的目标轨迹单帧/序列)，配套类外接口get/set_*公共方法；
   2. 需要类外设置的成员变量或功能通过`set_*`或`set_*_arm/end`进行设置或执行
   3. 初始化和自检完毕后info：“初始化和自检完毕！自检仅为必要的基础校验，自检通过不代表配置和程序无误！” + joyarm的当前配置信息情况，如：自由度数、关节名及范围限位等（具体你来自行补充）。
3. 一级功能区 —— 硬件后端的连接和使能（委托后端）；
   1. connect/disconnect_arm/end 四方法，直接委托self._backend.connect/disconnect_arm/end,并返回连接结果。
   2. enable/disable_arm/end 四方法，和“connect”类方法同实现方式。
   3. is_connected/enabled_arm/end 四属性方法,`@property` 委托返回 self._backend.is_connected/enabled_arm/end 。
4. 一级功能区 —— 硬件后端失能状态（部分可兼容使能）的读取和设置（委托后端）；
   1. read/write_param_arm/end
   2. get/set_mode_arm/end
   3. get_state_arm/end
   4. set_zero_arm/end
   5. get/check/clear_error_arm/end
5. 一级功能区 —— 硬件后端使能状态的关节指令下发（委托后端）；
   1. send_mit_arm/end
   2. send_position_arm/end
   3. send_velocity_arm/end
   4. send_action_end
6. 一级功能区 —— arm/end 基础运动功能；
   1. set/get_activated_solver("六类","子求解器对象名")：设置当前激活的求解器_activated_solver(设置plan求解器时先执行一次plan再替换_activated_solver)，名称无效则warn / 获取当前激活的求解器；当设置的是plan/ctrl对象时，还需按cfg.robotics的配置更新成员变量_plan/ctrl_hz；同时，cfg中六求解器对象的参数在调用求解时也需能够成功设置，比如{ name: pin_ikine_solver, damping: 1.0e-3, step_max: 0.5, restarts: 3 }。
   2. set/get_target_trajframe()：设置目标轨迹单点或序列至_target_trajframe（原子赋值）/ 读取当前的_target_trajframe（@property）；
   3. _start/stop_motion_thread：先启动规划器线程（启动plan前先执行一次plan），再启动控制器线程，要求线程定时器的hz能随set_activated_solver按cfg中plan/ctrl子求解器的plan_hz/ctrl_hz动态更新（每次set_activated_solver设置plan/ctrl子求解器对象时，均会更新成员变量_plan/ctrl_hz）；
   4. set_end_action("action",**kw)：backend子类中自定义的动作，传入动作名及可缺省的参数；
7. 一级功能区 —— arm 的运动状态机与安全事件
   1. 运动状态机的六状态（枚举类型，按需定义到utils/，状态机跑在一个独立的线程，执行频率 = 默认+可切换规划/控制器时可动态配置，前置要求：初始化和自检通过、connect）（各个状态需用到的参数配置在cfg.joyarm.MotionFSM.*）
      1. disable 失能状态：在此状态下，若arm的各关节error码不全为失能（error=0），则执行“失能-清错-失能”；若arm关节电机全为失能状态，则空执行；
      2. zero 零位状态：若当前arm各关节 is_in_q_arm(zero_arm)，则（若arm的各关节error码不全为失能（error=0），则执行“失能-清错-失能”；若arm关节电机全为失能状态，则空执行）；否则若当前arm各关节 not is_in_q_arm(zero_arm)，则（若arm全部电机存在未使能则先使能，若arm电机get_mode后不全为position则先将电机模式全部设为position，然后执行发送位置零位指令以运动到zero，运动到zero位后再在zero位失能（后续维持零位失能态））；
      3. home 空闲状态: 若arm全部电机存在未使能则先使能，若arm电机get_mode后不全为position则先将电机模式全部设为position，最后执行发送位置home_arm位指令以持续保持home;
      4. motion 正常运动状态：若arm全部电机存在未使能则先使能，若arm电机get_mode后不全为当前控制器（robotics/control/下的各控制器对象）要求的控制模式则先切换至控制器要求的控制模式，若规划线程和控制线程存在未启动则（先停止并清理线程，然后启动plan线程（启动前先plan一次）再启动ctrl线程）（两线程需能够：若motion状态时发生了求解器的切换则需要保证plan求解器切换前先plan一次+求解器参数也随其切换而切换（如plan/ctrl_hz））;motion状态退出时则先停止控制线程再停止规划线程再清理相关缓存；
      5. hold 原位锁定状态：若arm全部电机存在未使能则先使能，若arm电机get_mode后不全为position则先将电机模式全部设为position，然后发送arm的锁定位置指令（首次进入该状态时由self._backend.get_joint_state获取要锁定到的q，状态内始终下发该锁定位置值，锁定模式退出时清空该值缓存）以保持该位置的锁定;
      6. damping 阻尼状态：若arm全部电机存在未使能则先使能，若arm电机get_mode后不全为mit则先将电机模式全部设为mit，最后执行发送mit阻尼指令（kd取cfg.joyarm.motionFSM.mit_kd_damping,若未配置或校验失败时则在joyarm init时warn并取backend的joint mit kd系数）；
   2. 运动状态机的状态切换过程：路线表见文末的附件，from左侧列的状态（起点）to上方行的状态（终点）；
   3. 安全事件结构（枚举类型六个状态，危险程度递增，按需定义到utils/）;
      1. Normal（运行正常），对应切换的状态机状态（终点目标状态）为"motion";
      2. Holding（原位锁定），对应切换的状态机状态（终点目标状态）为"hold";
      3. homing（回退锁定），对应切换的状态机状态（终点目标状态）为"home";
      4. zeroing（回零失能），对应切换的状态机状态（终点目标状态）为"zero";
      5. damping（阻尼防护），对应切换的状态机状态（终点目标状态）为"damp";
      6. disable（电机失能），对应切换的状态机状态（终点目标状态）为"disable";
   4. 状态机与安全事件两者结合过程（“安全事件”这种叫法标准码？标准做法是什么？）（若有更优雅的方案则不必严格遵循）
      1. _current_motion_state(带 get_current_motion_state方法)
      2. _target_motion_state(带 get_target_motion_state方法、_set_target_motion_state方法)
      3. 状态机状态切换时，有可能存在中间状态，需优雅处理。状态切换时需在合适时机清理旧状态的缓存数据。（exit_current -> enter_target -> current=target -> current循环？）
      4. 安全事件中，set/get_event（私有成员变量_safe_event）。set时：_safe_event=X；_target_motion_state=X;
   5. start/stop_MotionFSM()：启动/关闭运动有限状态机
      1. start：
         1. 前置要求：arm需要已连接。
         2. 清理旧缓存（若未清理）
         3. 先进入disable状态，再home状态，再motion状态
      2. stop：
         1. 前置要求：arm需要已连接。
         2. 先home状态，再zero状态
         3. disable所有电机
         4. 清理缓存，退出状态机
   6. joyarm对象销毁和关闭前的安全退出机制：需先stop_MotionFSM
   7. 相关的参考函数（若有更优雅的方案，不必严格遵循）
      1. is_in_p/pose_arm：末端位置/位姿到位判断
      2. is_in_q_arm/end：关节到位判断
8. 一级功能区 —— joyarm 的类外访问接口（私有成员变量或对象的外部set/get方法，不仅限于如下几个，可按需增加）；
   1. get_config：返回self._cfg深拷贝；
   2. solve_fkine/ikine/ikine_all,get_jac/jac_pinv_damped/idyn/mass_matrix/coriolis/gravity：委托当前激活的算法求解器求解，参数默认取`_backend.get_state_arm`
   3. get_joint_names_arm/end(joint:str="all"),返回全部（默认）或单关节（传入的序号）
   4. get_joint_index_arm/end(name:str="all"),返回全部（默认）或单关节（传入的关节名）
   5. rand_q_arm/end（joint_limits:JointLimits=self._joint_limits_soft_arm/end）
   6. connect/disconnect_arm/end(),enable/disable_arm/end
9. 一级功能区 —— 六类算法求解器的管理；
    1. get_solvers_all（类型）,默认返回六类，传参返回传入的类的全部子求解器
    2. get_solver_activated（类型）
    3. `_robotics_solvers` 双层字典：第一层选类别（共六类），第二层选该类的具体求解器对象，字典值为cfg.robotics配置的各求解器对象（先cfg自检存在后再创建）；
    4. `_activated_solvers` 字典：当前启用的求解器，键为类别（六类），值为子求解器名称（注册名，默认为cfg.robotics.* 的首个子求解器）；
    5. plan类 求解器切换前需要先plan一次再set_solver()，避免控制器取到旧缓存
10. 一级功能区 —— safe_monitor 安全事件监视器功能
    1. start/stop_monitor方法启动或关闭监视器线程
    2. 通过监测运行状态并判断和set安全事件进行各类急停和恢复（需状态机非阻塞，能立即响应安全事件）
11. 一级功能区 —— 运行状态可视化（线程，纯监控可视化界面：左侧mesh可视化，右侧joyarm各状态数据实时显示）
12. 一级功能区 —— joyarm 私有助手；
    1. `_pin_model_name2id(name:str) -> id:int`（返回值类型是int吗？我不确定），pin_model由name返回id。
    2. 其他

## 附件

### 运动状态机状态转移路线表

| from\\to | disable | zero | home | motion | hold | damping |
| --- | --- | --- | --- | --- | --- | --- |
| disable | - | 1->3->2 | 1->3 | 1->3->4 | 1->5 | 1->6 |
| zero | 2->1 | - | 2->3 | 2->3->4 | 2->5 | 2->6 |
| home | 3->1 | 3->2 | - | 3->4 | 3->5 | 3->6 |
| motion | 4->1 | 4->3->2 | 4->3 | - | 4->5 | 4->6 |
| hold | 5->1 | 5->3->2 | 5->3 | 5->3->4 | - | 5->6 |
| damping | 6->1 | 6->3->2 | 6->3 | 6->3->4 | 6->5 | - |
