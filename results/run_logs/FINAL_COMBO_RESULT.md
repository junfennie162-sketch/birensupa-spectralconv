# 最终结合版结果

> 状态：**DONE** · 更新 2026-08-14T19:38:35Z · 已跑 120.2 / 120.0 分钟

只看这一页。不写正式 `summary.json` idle ms。

结合：官网双角合同 + 官方 shape；咱们 fused suFFT、列截断、scatter、缓存。

## 正确性（vs 官网 reference，门槛 1e-4）

| 路径 | 64 rel | 128 rel | 256 rel | 过线 |
|------|--------:|--------:|--------:|:----:|
| official_cpu | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| fused_trunc_auto | 2.548e-07 | 2.724e-07 | 2.828e-07 | PASS |
| fused_trunc_on | 2.548e-07 | 2.724e-07 | 2.828e-07 | PASS |
| fused_trunc_off | 9.055e+00 | 1.128e+00 | 3.375e+00 | FAIL |
| fused_r10_scatter | 2.548e-07 | 2.724e-07 | 2.828e-07 | PASS |
| v1_official_fft_supa_mul | 2.089e-07 | 2.123e-07 | 2.116e-07 | PASS |
| pruned_official_corners | 6.029e-06 | — | — | PASS |

## 性能中位数（ms，CPU 入 → CPU 出）

| 路径 | 64 | 128 | 256 | 样本数 |
|------|---:|----:|----:|------:|
| official_cpu | 70.164 | 146.989 | 303.379 | 204 |
| fused_trunc_auto | 3.826 | 8.360 | 29.285 | 203 |
| fused_trunc_on | 3.819 | 8.362 | 29.275 | 203 |
| fused_trunc_off | — | — | — | 0 |
| fused_r10_scatter | 3.878 | 8.475 | 29.413 | 203 |
| v1_official_fft_supa_mul | 12.809 | 13.508 | 203.365 | 203 |
| pruned_official_corners | 49.621 | — | — | 203 |

## 最终版（过线路径里、三档几何平均最快）

- 选定：**`fused_trunc_on`**
- 64/128/256 ms：3.819 / 8.362 / 29.275
- 相对现行 fused_trunc_auto：64: 0.2%; 128: -0.0%; 256: 0.0%
- 是否改正式热路径：**否**（本脚本只给结论；要 promote 另说）

正式主报仍是公开 NS64 L2 **0.035012**、idle **3.797 / 8.037 / 29.295 ms**，直到有人明确 promote。

## 后续（2026-08-15 上午）

`fused_trunc_off` 在**新进程**里也 FAIL（rel≈1.06/1.09/1.23），不是同进程缓存串了。整谱 `rfft2_sufft` / `irfft2_sufft` 是坏的；官方档一直走列截断所以没暴露。

已改：`SPECTRAL_TRUNC_COL=0` 也走同一套过线的 packed trunc（C2C 宽度 = 全 `Wf`）。复测 64/128/256 均 PASS（rel≈2.6e-7）。非正式 64 ms：auto **3.766**，trunc_off **4.808**。默认仍是 auto（只算 `modes2` 列），**不 promote** 正式 ms。
