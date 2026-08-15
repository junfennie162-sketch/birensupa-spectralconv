# pruned 续测（2026-08-15）

> 旁注。**未写** `summary.json` / 正式 idle。协议对齐官网：正确性三案 + 性能 warmup=10/iters=100。

## 正确性

| 路径 | tiny 8×8 | small 32×32 | target 64×64 |
|------|----------:|------------:|-------------:|
| default_pruned | 4.675e-08 PASS | 1.137e-07 PASS | 7.162e-06 PASS |
| sufft_trunc | 4.675e-08 PASS | 1.137e-07 PASS | 2.170e-07 PASS |

## 非正式计时（未 promote）

| 路径 | 64 ms | 128 ms | 256 ms |
|------|------:|-------:|-------:|
| default_pruned | 0.989 | 2.331 | 7.877 |
| sufft_trunc | 3.653 | 8.542 | 29.149 |

all_ok: **true**
