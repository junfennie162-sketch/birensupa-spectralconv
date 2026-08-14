# Wave-R2 条件项 · 跳过说明（2026-08-02）

R2-1 scheduled sampling 已破门槛并 promote（L2 **0.036092** < gate 0.036476）。

| 条件项 | 裁决 | 原因 |
|--------|------|------|
| R2-0 weight soup | **done / no promote** | 最佳≈0.036705，未破 gate |
| R2-2 geom+noise | **skip** | 精度线已有 promote |
| R2-3 width48→KD | **skip** | 无需扩容重训 |
| 同构 squeeze / formal ms | **No-Go** | 继承红线 |

若日后公开榜竞争需要再开 R2-2/R2-3，按 OPT_ROUND2 早停纪律执行。
