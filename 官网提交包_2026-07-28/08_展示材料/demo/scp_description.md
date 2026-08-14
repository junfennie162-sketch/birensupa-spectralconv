# SCP 作品简介 · 翻斗花园

在壁仞 GPU 上以 **混合路线**实现 FNO 核心 Spectral Convolution：方式一 SUPA/C++ 验证 `spectral_mul`，方式二 PyTorch Extension 组装双角全链路（CPU FFT + SUPA 乘），并搭建 **≥4 层** FNO 完成二维涡度场单卡前向与可视化。

- 正确性：相对误差 ≤ 1e-4（实测 ~1e-7）
- 性能：64/128/256 全链路耗时见 `results.md`
- Agent：Cursor；日志见 `development_log.md`（≥5 段）
- Skill：`skill.md`
