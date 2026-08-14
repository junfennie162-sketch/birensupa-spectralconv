# Agent / Skill 交互日志（官方抽查页）

> **官方必须项**（赛道评分「Agent 开发」约 15%；缺失或不足 **视为不合格**）。  
> 本页按选手手册模板撰写，供评委 **直接抽查**。全文轨迹见 [`development_log.md`](development_log.md)（共 **43** 段编号记录）。  
> 工具：Cursor Agent（SSH · 壁仞竞赛 Docker · SDK `1.11.0.0.rc2`）。  
> 现行主报：公开 NS64 L2 **0.035012**（`spec_ref_r2` · v10）；Spectral idle **3.797 / 8.037 / 29.295 ms**（08-14 复测）。

## 场景覆盖（≥3 类 · 本页 6 类全覆盖）

| # | 官方场景 | 本页记录 | 证据产物 |
|---|---------|----------|----------|
| 1 | 算子 kernel 设计/调试/优化 | 记录 A | fused suFFT + SUPA mul；`spectral_conv_ext.su` |
| 2 | 性能瓶颈分析与定位 | 记录 B | Host↔Device 剖析；idle 冻结板 |
| 3 | 模型架构选型与超参搜索 | 记录 C、E | gate 纪律；Spectral-Refiner → v10 |
| 4 | 数据预处理与特征工程 | 记录 D | 公开 NS64 1000/128 与自建 v2 分栏 |
| 5 | 结果分析与可视化 | 记录 F | Autopsy 三图；Pred/GT |
| 6 | BIREN GPU 平台适配 | 记录 B、环境段 | `device=supa`；单卡；`brsmi` 快照 |

运行日志/截图：[`demo/media/brsmi_snapshot.txt`](demo/media/brsmi_snapshot.txt) · 正确性/性能复测见 `results/run_logs/`。

---

## Agent 交互记录 A · 算子 kernel（fused SpectralConv）

- 工具 / Agent：Cursor Agent（SSH）
- 场景标签：算子 kernel 设计/调试/优化
- 目标：把 2D Spectral Convolution 从「CPU FFT + 整谱回 Host 再乘」改成设备常驻 fused 路径，并保持相对误差 ≤ 1e-4
- 关键提示词或交互摘要：按深度优化计划连续迭代；GPU 串行；改完必须测完。用户要求核心计算走 SUPA / Extension，禁止纯原生 PyTorch 交差
- Agent 建议：分段 profile 显示 64×64 上 bridge_mul ≈ 26 ms，远大于 rfft；应把 suFFT R2C → SUPA 复数乘 → C2R 串在设备上；v1 CPU-FFT 路径保留作对照与可微训练
- 采纳的修改：实现 `spectral_conv2d_fused`（`spectral_conv_ext.su` / `.cpp` / `spectral_conv_ops.py`）；双角 reference 对齐；FNO 层复用同一 API
- 验证结果：正确性 worst rel ≈ **2.17×10⁻⁷** ≪ 1e-4；08-14 idle 复测 **3.797 / 8.037 / 29.295 ms** @64/128/256（07-31 板 3.811/8.054/29.560，噪声内）
- 未采纳内容及原因：未把 `torch.fft` 直接跑在 `device=supa`（该路径会偏出 1e-4）；未在 SUPA 上 `torch.cat` 后立刻调自定义 kernel（已知写坏内存）

---

## Agent 交互记录 B · 性能瓶颈（Host↔Device / C2R 墙）

- 工具 / Agent：Cursor Agent（SSH）
- 场景标签：性能瓶颈分析与定位 · BIREN GPU 平台适配
- 目标：解释小分辨率为何更慢，并决定 fused 阈值与缓存策略
- 关键提示词或交互摘要：用户要求继续优化算子耗时；询问为何 64/128 一度比直觉慢
- Agent 建议：主因是反复 H2D/D2H 与每次分配频域 buffer，不是 FMA 不够；应用 `_OUT_FREQ_CACHE` / `_HOST_OUT_CACHE`；fused 阈值 256→128→64 必须靠 sweep，不能拍脑袋
- 采纳的修改：`profile_segments.py` 分段计时；buffer 复用；`test_perf.py` 争用守卫（64 ms>12 不写 formal）；单卡禁止与 `ai4s-n` 并发
- 验证结果：相对本机官网 CPU 参考约 **19.5× / 11.1× / 10.0×**；formal 板 2026-07-31 idle 复测后冻结。曾把 128 强行 fused 得到 40 ms > v1 的 22 ms，**立即回退**
- 未采纳内容及原因：Plan2d / `torch.fft@SUPA` / 半精度权重（SDK 不导出或破坏 1e-4）；日常默认重跑 `test_perf` 覆写主表（仅在用户明确要求复测时另出报告）

---

## Agent 交互记录 C · 超参与精度门禁（freeze → dualview）

- 工具 / Agent：Cursor Agent（SSH）
- 场景标签：模型架构选型与超参搜索
- 目标：在公开 NS64（1000/128，10→1）上压相对 L2，且 **未破 gate 不 promote**
- 关键提示词或交互摘要：用户多次要求「继续冲精度 / promote」；同时强调不要把未过线的 sidecar 写进正式 demo
- Agent 建议：gate = 上一正式 L2 − 1e-4；探针 `nohup` + `--stop-on-gate`；禁止同构 sched 再 deepen；modes=20 / width=48 已 KILL
- 采纳的修改：`promote_guard.py`；v8 `freeze_r9`（0.035302）；v9 `dualview_r2`（0.035115，末层解冻 + 双视图一致性）
- 验证结果：v9 独立复评 **0.035114976112**；autochain 曾把未破 gate 的 0.035252 写入 demo，**已回滚 v8** 并加硬门禁
- 未采纳内容及原因：未降 gate 强行 promote；未把 soup / PF 近失（0.035216）编入正式 v 号

---

## Agent 交互记录 D · 数据口径（公开 NS64 vs 自建 v2）

- 工具 / Agent：Cursor Agent（SSH）
- 场景标签：数据预处理与特征工程
- 目标：把公开 NS64 接进 loader，并防止合成集 0.005 被误报成公开成绩
- 关键提示词或交互摘要：对照官网进阶 C 与手册「须注明数据来源」；用户要求提交以公开集为准
- Agent 建议：`dataset.py` 优先读非 `ns_like` 的 `.pt`；划分锁死 1000/128、seed 20260722；`data_disclosure.md` 分栏
- 采纳的修改：接入 `navier_stokes_v1e-3_N1200_T20.pt`；自建 v2 continue3 降为旁注 ckpt `fno_ns_demo.pt`
- 验证结果：公开集从 0.041835（v1）优化到 **0.035012**（v10）；合成集 0.005144 **从未进入** `summary.fno_ns.public_ns64`
- 未采纳内容及原因：未在评测容器联网下载 HDF5（离线可复现优先）

---

## Agent 交互记录 E · Spectral-Refiner → v10（现行主报）

- 工具 / Agent：Cursor Agent（SSH）
- 场景标签：模型架构选型与超参搜索 · 算子 kernel（仅微调 spectral 权重）
- 目标：在 v9 plateau 上寻找**新机制**过 gate（0.035015），而不是再加 epoch
- 关键提示词或交互摘要：用户同意「试文献机制」并要求「按照计划继续冲」；过线后整理 submit / skill
- Agent 建议：H1 空间梯度损失作对照；主路径用 Spectral-Refiner lite——冻结非 spectral，损失混合 rel-L2 与频域 H⁻¹ 加权 `(α+|k|²)⁻¹`
- 采纳的修改：`train_public_spectral_refiner_probe.py`；`spec_ref_r1` 到 0.035027；wave4 `spec_ref_r2` epoch7 `stop_on_gate`；`promote_public_ckpt.py --tag spec_ref_r2`
- 验证结果：clean reeval **0.035011906177** < gate **0.035014976**；相对 v9 **+0.29%**；H1 与 soup 均未超过单模型
- 未采纳内容及原因：未自动 promote（先人工确认）；未把 2.9G tar 推 GitHub（超 100MB）；未解冻 Spectral formal ms

---

## Agent 交互记录 F · 可视化与误差解剖

- 工具 / Agent：Cursor Agent（SSH）
- 场景标签：结果分析与可视化代码生成
- 目标：给答辩提供 Pred/GT/error 与「误差从哪来」的可证伪图，而不是只报一个 L2
- 关键提示词或交互摘要：用户要评委 3 分钟路径；确认执行 Offline Error Autopsy（epochs=0 只读）
- Agent 建议：共享色标 Pred/GT；best/median/worst sample strip；用 e1 与时间增量 q_t 的相关决定下一步机制
- 采纳的修改：`visualize.py`；`diagnose_public_error_autopsy.py`；三图进 `demo/media/`
- 验证结果：ρ(e1, q_t)≈0.80 → 后续 Δ-match / q_t 过采样 / Refiner；频谱未到必须涨 modes 的程度（modes=16 保持）
- 未采纳内容及原因：未把 Autopsy 近失数字编进评测报告 v 号；未拍视频（手册为建议项，storyboard 已覆盖）

---

## 如何复核本页

```bash
# 日志段数（全文 ≥5；本页抽查 6 段）
grep -c '^## Agent 交互记录' development_log.md

# 场景标签
grep -E '场景标签' AGENT_OFFICIAL.md

# 单卡运行快照
cat demo/media/brsmi_snapshot.txt
```

对应 skill：提交根 [`skill.md`](skill.md)（必须项）。完整字段版 41 段：[`development_log.md`](development_log.md)。
