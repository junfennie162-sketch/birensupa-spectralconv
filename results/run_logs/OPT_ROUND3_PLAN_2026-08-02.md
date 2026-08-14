# OPT_ROUND3 · 2026-08-02 下一轮优化总方针

> **历史 · 已完成**。下链 [`OPT_ROUND4_PLAN_2026-08-02.md`](OPT_ROUND4_PLAN_2026-08-02.md)。  
> **现行**见 [`CURRENT.md`](CURRENT.md) · [`OPT_ROUND7_PLAN_2026-08-02.md`](OPT_ROUND7_PLAN_2026-08-02.md)。  
> 历史接棒自 [`OPT_ROUND2_PLAN_2026-08-02.md`](OPT_ROUND2_PLAN_2026-08-02.md)。  
> 工作区：`/workspace/ai4s-f` · 合入：`/workspace/ai4s/submission` · **勿改 `ai4s-n`**  
> 评测报告戳（当时）：`2026-08-02_110958`（仅历史）

## 0. 基线（滚动）

| 角色 | 指标 | 数值 |
|------|------|------|
| FNO 精度 | public NS64 rel-L2 | **0.035855**（`sched_samp_r3` promote；已破 gate） |
| Spectral 性能 | idle 64/128/256 | **3.811 / 8.054 / 29.560 ms**（冻结） |
| Spectral 正确性 | worst rel | ~2.17e-7 |

## 1. 主攻序

```text
Wave-R3-0  零 GPU：口径闸 + irregular 门禁 + 叙事/分镜
Wave-R3-1  精度：sched延长 → soft → roll+geom（gate 0.035992）
Wave-R3-2  官网级 3D 四角（低 GPU）
Wave-R3-3  maintain + pack/sync ai4s
```

## 2. Wave 状态（滚动）

| Wave | 状态 | 备注 |
|------|------|------|
| R3-0 | **done** | 危急脚本与协议 / irregular 门禁 / 叙事分镜 |
| R3-1 | **done / promoted** | ep5 best=**0.035855** → promote；跳过 soft/geom |
| R3-2 | **done** | 四角 2/2 PASS，worst ≈1.19e-7 |
| R3-3 | **done** | maintain PASS；包 `20260802_110310`；评测报告 `110530` |

## 3. No-Go

同构 squeeze；解冻 formal ms；Plan2d/真融合；F-FNO 换主报；TTA 主报；v2 伪官方；SOL 得分句；fused 长训；width48→KD 默认不做；B 站成片默认不做。
