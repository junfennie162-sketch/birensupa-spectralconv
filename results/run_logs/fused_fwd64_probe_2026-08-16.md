# fused mixed-radix forward 64 探针（2026-08-16）

> 旁注。**未写** `summary.json` / 正式 idle。`SPECTRAL_FUSED_FWD64` **KEEP 默认开**。

- 官方三案 fused/twokernel：PASS
- modes=16 @64 vs CPU rel=1.630e-06 vs two-kernel rel=**0**
- 隔离 forward：two **0.128** ms / fused **0.094** ms / match rel=**0**
- 首轮 e2e 64：two 0.800 / fused 0.776 / 主表 0.961
- 复测 4 轮 64：two ≈ **0.789–0.819** / fused ≈ **0.750**（约 5%）
- KEEP：默认 `SPECTRAL_FUSED_FWD64=1`；可用 `=0` 退回两 kernel

## e2e CPU-in（首轮）

| 路径 | 64 | 128 | 256 |
|------|---:|----:|----:|
| twokernel | 0.800 | 2.269 | 7.682 |
| fused | 0.776 | 2.379 | 7.597 |

## 64 复测（同一输入）

| 轮 | two | fused | Δ ms | Δ % |
|---:|----:|------:|-----:|----:|
| 0 | 0.819 | 0.750 | 0.069 | 8.5 |
| 1 | 0.789 | 0.750 | 0.039 | 4.9 |
| 2 | 0.789 | 0.750 | 0.039 | 4.9 |
| 3 | 0.789 | 0.750 | 0.039 | 5.0 |
