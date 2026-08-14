# 对照实验矩阵 · KEEP / KILL / ABORT

> 用途：答辩/SCP「失败也展示」证据链。主报数字不得混入旁注行。  
> 主报：公开 NS64 L2=**0.035725**（`sched_samp_r5`）；Spectral idle=**3.811 / 8.054 / 29.560 ms**。  
> 方针：[`run_logs/OPT_ROUND3_PLAN_2026-08-02.md`](run_logs/OPT_ROUND3_PLAN_2026-08-02.md)

## 创新点卡片（KEEP · 主卖点）

| 卡片 | 内容 | 证据 |
|------|------|------|
| 双角 fused | suFFT R2C → SUPA gather/scatter mul → C2R；频谱常驻设备 | `spectral_conv_ops.py`；idle 主表 |
| residual + hf + aug | 公开 NS 难例：残差头 + 频域加权 + 周期 roll | `train_public_ns64_boost.py`；boost→sq3b |
| multistep TF + soft | 训练多步 teacher-forcing + 轻量能量/谱 soft | L2 0.036576 历史主报 |
| **sched-sampling multistep** | 缓升 p_ar 混入 pred 帧；eval 仍 step-1 | `fno_public_sched_samp_r5_summary.json`；L2 **0.035725** |
| 口径闸 | 公开 1000/128 与自建 v2 严格分栏；禁伪官方 | `data_disclosure.md`；OPT_ROUND2 |

## Spectral / 算子线

| 实验 | 结果 | 裁决 | 备注 |
|------|------|------|------|
| v1 CPU-FFT + SUPA mul | 正确性 PASS；perf 慢 | KEEP 对照 | 可微训练后备 |
| suFFT + CPU bridge mul | 64 上 bridge≈26ms | KILL 热路径 | 见 profile_segments |
| fused suFFT+mul | 正式热路径 | **KEEP** | formal 主表 |
| Plan2d / PlanMany | SDK 无导出 | ABORT | 封死 |
| torch.fft@SUPA | 正确性风险 | ABORT | 封死 |
| dual_scatter / packed trunc / P8b | idle 3.811 板 | **KEEP** | ms 冻结 |
| Backward spectral_mul | worst grad ~6e-8 | **KEEP** 扩展 | `test_backward.py` |
| SpectralConv3d | 2/2 PASS，四角 worst≈1.19e-7 | **KEEP** 扩展 | 非 FNO-3D |
| irregular shapes | 9/9 PASS，worst≈3.20e-7 | **KEEP** 扩展 | `spectral_irregular_2026-08-02.md` |
| SOL-ExecBench / sol_proxy | 队内 proxy | 旁注 | **禁得分句** |

## FNO / 精度线

| 实验 | 结果 | 裁决 | 备注 |
|------|------|------|------|
| 公开 scratch+continue | L2=0.041835 | 历史基线 | 已被超越 |
| boostA/B/C | →0.037820 | KEEP 轨迹 | chain_final |
| squeeze sq1–sq3 | →0.037520 | KEEP 历史 | tag `sq3b_freeze` |
| multistep TF+soft | →0.036576 | KEEP 历史 | tag `multistep_probe` |
| sched-sampling r2 | →0.036092 | KEEP 历史 | tag `sched_samp_r2` |
| sched-sampling r3 | →0.035855 | KEEP 历史 | tag `sched_samp_r3` |
| sched-sampling r4 | best≈0.035812 | ABORT promote | 未破 gate；快路径停 |
| **sched-sampling r5** | →**0.035725** | **KEEP 当前主报** | tag `sched_samp_r5`；相对 r3 +0.36% |
| weight soup | best≈0.036705 | ABORT promote | 仍差于 demo |
| sq4a 同构 continue | Δ=0 | ABORT 续烧 | 平台停 |
| sched-sampling r7 | best=0.035683 | ABORT | 无提升 early_stop；停精度 |
| soft / geom / width48→KD | skipped | SKIP | r7 NO_SIGNAL 后停 |
| F-FNO 主报替换 | 损搭建衔接分 | No-Go 主报 | 仅对照 |
| TTA 计主报 L2 | 协议灰区 | No-Go | 旁注可 |
| 自建 v2 continue3 | 0.005144 | 旁注 | **非公开分** |

## 决策纪律

1. Promote：仅公开 1000/128 test L2 严格更优。  
2. 失败实验保留日志，不污染 `fno_ns_public_demo.pt`。  
3. Spectral formal ms 冻结；新数字进 `run_logs` / `optimization.*` 旁注。
