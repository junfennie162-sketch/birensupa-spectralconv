# 对照实验矩阵 · KEEP / KILL / ABORT

> 用途：答辩/SCP「失败也展示」证据链。主报数字不得混入旁注行。  
> 主报：公开 NS64 L2=**0.035115**（`dualview_r2` · v9）；Spectral idle=**3.811 / 8.054 / 29.560 ms**。  
> 现行指针：[`run_logs/CURRENT.md`](run_logs/CURRENT.md) · 流程：[`../skills/operator_opt_loop/LOOP_PROCESS.md`](../skills/operator_opt_loop/LOOP_PROCESS.md) · 计划卡：[`run_logs/OPT_WAVE_MULTIAGENT_PLAN_2026-08-03.md`](run_logs/OPT_WAVE_MULTIAGENT_PLAN_2026-08-03.md)

## 创新点卡片（KEEP · 主卖点）

| 卡片 | 内容 | 证据 |
|------|------|------|
| 双角 fused | suFFT R2C → SUPA gather/scatter mul → C2R；频谱常驻设备 | `spectral_conv_ops.py`；idle 主表 |
| residual + hf + aug | 公开 NS 难例：残差头 + 频域加权 + 周期 roll | `train_public_ns64_boost.py`；boost→sq3b |
| multistep TF + soft | 训练多步 teacher-forcing + 轻量能量/谱 soft | L2 0.036576 历史主报 |
| sched → soft → freeze → **thaw+dualview** | 末 2 层解冻 + 双视图一致性 | `dualview_r2`；L2 **0.035115** · v9 |
| 口径闸 | 公开 1000/128 与自建 v2 严格分栏；禁伪官方 | `data_disclosure.md`；WAVE plan |

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
| sched-sampling r5 | →0.035725 | KEEP 历史 · v7 | tag `sched_samp_r5` |
| soft_r8 | →0.035623 | KEEP 历史台阶 | 曾 live；未单独编报告 v |
| freeze_r9 | →**0.035302** | KEEP 历史 · v8 | 冻 spectral；相对 v7 +1.18% |
| **dualview_r2** | →**0.035115** | **KEEP 当前主报 · v9** | long_push；相对 v8 +0.53%；已 promote |
| freeze_r10（手跑） | →0.035287 | ABORT promote | 近失 gate 0.035202；NO_SIGNAL |
| freeze_r10（autochain） | →0.035252 | ABORT promote / **已回滚 v8** | 未破 gate；`promote=true` 违规写入后 demote |
| soft_r10 | =0.035302 | ABORT | Δ=0 early_stop；停精度 |
| weight soup | best≈0.035669 | ABORT promote | 未破 gate |
| modes20 / width48 / hybrid | 远差 / 无提升 | ABORT / **KILL** | ROUND9 + R11 复证；禁再开 |
| freeze_r11 | 链中止 | ABORT | OPT_WAVE KILL；SIGTERM 停卡 |
| hard_reweight_a1 | =0.035302 Δ=0 | ABORT / NO_SIGNAL | 难例重加权短探针；未破 gate；停精度 |
| **error_autopsy_D** | epochs=0 · ρ(e1,g)=0.798 · ∩=10 | **CONDITIONAL_PF_ALLOWED** · 已开 PF | 只读五联；r10=`INCUBATE`；modes 仍封存 |
| **pf_clean_r1** | →**0.035216**（Δ≈8.6e−5） | **ABORT promote / NO_SIGNAL** · `INCUBATE` | clean-anchor PF·冻 spectral·4ep；差 gate≈1.4e−5；精度永久停 |
| **delta_match_r1** | →**0.035209**（Δ≈1.5e−5 vs freeze_r11） | **NO_SIGNAL** | Autopsy q_t→Δ-match；gate 0.035123 未破；demo 未动 |
| soup_near3 | →**0.035203** | **NO_SIGNAL** | demo+pf+delta 均匀/末二；差 gate≈8e−5 |
| **pf_delta_r1** | →**0.035192** | **NO_SIGNAL** | PF+Δ hybrid 自 soup；Δ≈3.2e−5 vs demo；未破 1e−4 gate |
| long_push w1 | →**0.035160** dualview_r1 | 近失 | qt过采样→thaw1→dualview |
| **last_thaw_r2** | →**0.035116** | **SIGNAL** | 解冻末 2 层 spectral；复评一致 |
| dualview_r2（过程） | →**0.035115** | **已 promote → v9** | 见 KEEP 卡片；backup=`demo_pre_dualview_r2.pt` |
| geom / 部分解冻 | 未开 | SKIP | A1 后不再开条件精度探针 |
| F-FNO 主报替换 | 损搭建衔接分 | No-Go 主报 | 仅对照 |
| TTA 计主报 L2 | 协议灰区 | No-Go | 旁注可 |
| 自建 v2 continue3 | 0.005144 | 旁注 | **非公开分** |

## 决策纪律

1. Promote：公开 1000/128 test L2 须破声明 gate（baseline−1e-4）且人工确认；禁止仅「优于 live」自动 promote。  
2. 失败实验保留日志，不污染 `fno_ns_public_demo.pt`。  
3. Spectral formal ms 冻结；新数字进 `run_logs` / `optimization.*` 旁注。
