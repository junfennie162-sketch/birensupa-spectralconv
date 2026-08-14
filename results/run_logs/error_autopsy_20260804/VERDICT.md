# Error Autopsy VERDICT · 2026-08-04

## 裁决

**`CONDITIONAL_PF_ALLOWED`**

AR preregister conditions ALL passed; PF may be considered in a future Go — NOT auto-started this round.

| 检查项 | 结果 |
|--------|------|
| cond1 median(g)>0 且 CI_lo>0 | `True` (median_g=0.0111059, CI=[0.0116296,0.0134633]) |
| cond2 ρ(e1,g)≥0.4 | `True` (ρ=0.7976) |
| cond3 worst-16 重合≥8 | `True` (overlap=10) |
| half-split conflict | `False` |
| spectrum concentrated | `False` (max [4,8) C_b=0.251) |
| near-miss label | `INCUBATE_WEAK_SIGNAL` |

## 主报（未改）

- 公开 NS64 L2 **0.035302** · `freeze_r9` · 评测报告 **v8**
- 本轮 epochs=0；未 promote；未跑 `test_perf`

## 关键数字（freeze_r9）

| 指标 | 值 |
|------|-----|
| mean e1 | 0.035302 |
| mean e2_TF | 0.039016 |
| mean e2_AR | 0.051538 |
| mean / median g | 0.012522 / 0.011106 |
| top feature vs e1 | q_t (ρ=0.823) |
| r10−r9 mean d | -5.03926e-05 · improve_frac=0.828 |

## 产物

- JSON（本目录）: `d0_protocol.json` … `d4_paired_near_miss.json`, `per_sample_*.json`, `verdict.json`
- 图（去重后只留一处）: `results/figures/` 与 `demo/media/` 同名 PNG（含 heatmaps）

## 纪律

- Autopsy 当时不开训；后续人口头 Go 已跑 `pf_clean_r1` → **NO_SIGNAL**（见 `PF_FOLLOWUP_2026-08-04.md`）。
- `INCUBATE_WEAK_SIGNAL` / PF 近失 **不进**评测报告 v 链。
