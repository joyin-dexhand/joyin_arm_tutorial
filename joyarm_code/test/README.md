# test/ —— 测试套件（占位待重建）

本目录为测试套件占位。旧套件（离线回归 + 真机全覆盖，15 个 `test_*.py`）已整批删除（git 历史 `35b765f`），将随教程各章实现重建：

- **离线回归**：六域默认求解器（pin 对拍/五次边界）、`utils`（limits/transforms/types）、`FakeArm`（arm 协议符合性 + JoyArm 对拍）、配置静态自检（`check_config` 全部校验项正负例）；
- **真机测试**：DM 后端分层（协议编解码 → 总线 → 整机 → JoyArm 门面），风险递增、逐项确认。

FakeArm 即为测试标准替身（`joyarm_core.joyarm.fakearm`，鸭子类型满足 arm 协议，无需硬件）。
