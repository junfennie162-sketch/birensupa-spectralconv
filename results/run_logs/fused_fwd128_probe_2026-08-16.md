# fused mixed-radix forward 128 探针（2026-08-16）

> 旁注。**未写** `summary.json` / 正式 idle。`SPECTRAL_FUSED_FWD128` **KEEP 默认开**。

- modes=16 @128 vs CPU rel=1.468e-06 vs two-kernel rel=**0**
- 隔离 forward（env 在圈外切换）：two **0.221–0.224** ms / fused **0.162** ms
- e2e 128 复测：two ≈ **2.18** / fused ≈ **2.10–2.14**（均值约 3%）
- 64 默认路径未回退：约 **0.755 ms**
- KEEP：默认 `SPECTRAL_FUSED_FWD128=1`；可用 `=0` 退回两 kernel
- smem：`Z[16][16][8]` + `row[128][16]` ≈ 32 KB，占用仍够（未踩 256 fft_h 32KB 那种慢）

## 128 复测（同一输入）

| 轮 | two | fused | Δ % |
|---:|----:|------:|----:|
| 0 | 2.181 | 2.124 | 2.6 |
| 1 | 2.176 | 2.101 | 3.4 |
| 2 | 2.192 | 2.107 | 3.9 |
| 3 | 2.180 | 2.145 | 1.6 |
