# fused mixed-radix inverse 256 探针（2026-08-16）

> 旁注。**未写** `summary.json` / 正式 idle。KEEP 仍是两 kernel 逆。

- 官方三案 fused/twokernel：PASS
- modes=16 @256 vs CPU rel=1.380e-06 vs two-kernel rel=0.000e+00
- 隔离 inverse：two 1.334 ms / fused 1.362 ms / match rel=0.000e+00
- e2e 256：two 7.339 / fused 7.218 / KEEP 7.870
- promote_256：False

## 分段 @256（two-kernel）

| 段 | ms |
|----|---:|
| rfft2_pruned_trunc_ms | 0.608 |
| mul_ms | 0.078 |
| ifft_h_ms | 0.304 |
| irfft_w_ms | 0.952 |
| inv_two_ms | 1.334 |
| inv_fused_ms | 1.362 |

## e2e CPU-in

| 路径 | 64 | 128 | 256 |
|------|---:|----:|----:|
| twokernel | 0.970 | 2.222 | 7.339 |
| fused | 1.075 | 2.235 | 7.218 |
