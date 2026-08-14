# OPT_ROUND5 · 快路径方针

> **历史 · 已完成** → [`OPT_ROUND6_PLAN_2026-08-02.md`](OPT_ROUND6_PLAN_2026-08-02.md)  
> **现行**见 [`CURRENT.md`](CURRENT.md) · Round7。接棒自 [`OPT_ROUND4_PLAN_2026-08-02.md`](OPT_ROUND4_PLAN_2026-08-02.md)。  
> **纪律**：默认快路径；破 gate 即停；近失/变慢立即收口；不空等；评测报告只留一份。

## 基线

| 项 | 值 |
|----|-----|
| 正式主报 | L2 **0.035725**（`sched_samp_r5`） |
| r4 近失 ckpt | L2 **0.035812**（未 promote） |
| gate | **&lt; 0.035755** |
| Spectral | formal **冻结** |

## 主线（唯一）

```text
R5-1  自 r4_best 续跑 ≤4ep + stop-on-gate + patience=2
R5-2  SIGNAL → promote+visualize；NO_SIGNAL → 停精度（不做 soft/geom）
R5-3  maintain + pack + 唯一评测报告换戳
```

## Wave

| Wave | 状态 | 备注 |
|------|------|------|
| R5-1 | **done / SIGNAL** | ep3 best=0.035725 stop_on_gate |
| R5-2 | **done** | promote `sched_samp_r5` |
| R5-3 | **done** | 包 `20260802_123445`；报告 `123706` |

## No-Go

解冻 formal；soft/geom 本轮默认不做；长训 >4ep；改 `ai4s-n`；空等 Await。
