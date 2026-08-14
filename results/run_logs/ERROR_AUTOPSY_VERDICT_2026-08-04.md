# Error Autopsy D · 裁决一页卡（2026-08-04）

> epochs=0 只读五联诊断。主报未动：**0.035302 · freeze_r9 · v8**。  
> 明细目录：[`error_autopsy_20260804/`](error_autopsy_20260804/) · 脚本：`fno_ns/diagnose_public_error_autopsy.py`

## 裁决

| 项 | 值 |
|----|-----|
| **VERDICT** | **`CONDITIONAL_PF_ALLOWED`** |
| 本轮是否开训 | **否**（需另开人口头 Go） |
| 主报 / 报告 v | 不变 · **不编新 v** |
| 近失 0.035252 | `INCUBATE_WEAK_SIGNAL`（不进版本链） |

**一句话**：AR exposure gap 与正式单步难例显著相关（ρ≈0.80，worst-16 重合 10/16），故未来可条件讨论带 clean-anchor 的 PF；本轮仍按纪律不开训。频谱误差未形成单一集中箱（worst16 max C_b≈0.25 @\[4,8)）→ **modes 扩容仍永久封存**。

## AR 预注册（全过）

| 条件 | 结果 |
|------|------|
| median(g)>0 且 bootstrap CI 下界>0 | PASS · median_g=0.0111 · CI≈\[0.0116, 0.0135\] |
| Spearman ρ(e1,g)≥0.4 | PASS · ρ=**0.798** |
| e1 worst-16 ∩ g worst-16 ≥8 | PASS · overlap=**10** |
| half-split 冲突 | 无 |

对照量（freeze_r9）：mean e1=**0.035302** · e2_TF=0.0390 · e2_AR=0.0515 · mean g=0.0125。

## 难例 / 近失旁证

- 与 e1 最相关动力学特征：**q_t**（时间增量，ρ≈**0.823**）——难例偏「变化大」的步，而非单纯可标量重加权。
- r10 vs r9：mean d≈**−5.0e−5**，improve_frac≈**82.8%**，CI 全负 → 弱但可信改善；未破 gate(0.035202) → **INCUBATE**，不 promote。

## 答辩引用（三图）

| 图 | 路径 |
|----|------|
| 协议不可横比 | [`demo/media/protocol_vs_fno_paper_0128.png`](../../demo/media/protocol_vs_fno_paper_0128.png) |
| e1 / TF / AR / g | [`demo/media/error_decomp_e1_tf_ar.png`](../../demo/media/error_decomp_e1_tf_ar.png) |
| 频谱 + 涡结构条带 | [`demo/media/spectrum_best_median_worst.png`](../../demo/media/spectrum_best_median_worst.png) · heatmaps 同前缀 |

可嵌 PPT：[`PPT答辩冻结稿_2026-08-04.md`](../PPT答辩冻结稿_2026-08-04.md)；评委入口：[`JUDGE_3MIN_PACK_2026-08-04.md`](JUDGE_3MIN_PACK_2026-08-04.md)。

## 纪律提醒

1. 正式 gate **1e−4 不降**；INCUBATE ≠ v9。  
2. 禁盲开 STLW / modes20 / width48 / 解冻 Spectral / `test_perf`。  
3. 若开 PF：须带 **clean-anchor**，epochs≤4，`--stop-on-gate`，破 0.035202 才可谈 promote。  
