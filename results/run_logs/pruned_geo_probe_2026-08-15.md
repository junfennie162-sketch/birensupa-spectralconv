# 几何 A/B 探针（2026-08-15）

> 旁注。**未 promote** 正式 idle。只学对方公开几何，不抄 kernel。

## 结论

| 决策 | 项 | 原因 |
|------|----|------|
| KEEP | 高度两角一次扫描（dual） | 一列扫一遍同时写顶/底角；64 档 1.464→**1.303** |
| KEEP | ifft 每线程两行 | 仅 256 及非 64/128 偶数高；64/128 已改混合基 |
| KEEP | 融合逆仅 `width>=256` | 256：~14.6 vs 两核 ~16.3 ms |
| KEEP | 混合基 ifft_h（64 `16×4` / 128 `16×8` / 256 `32×8`） | ifft_h **0.11 / 0.15 / 0.29**；256 `16×16` 仍 No-Go |
| KEEP | 混合基前向 rfft_w（128 `16×8` / 256 `16×16` smem） | 前向 trunc **0.156 / 0.221 / 0.569** |
| KEEP | 混合基前向 64（rfft `16×4` + fft_h `16×4` smem） | 64 fwd **0.126**（原 0.156） |
| No-Go | 256 irfft smem 共享 16 bin | 正确；irfft **1.081 vs 0.958** |
| KEEP | 混合基 irfft（64 `16×4`；128/256 float4 加载） | irfft_w 隔离 **0.23 / 0.28 / 0.84** |
| No-Go | 256 irfft 双 n1（线程内两次 dft16） | rel≈1、30 ms（同 256 ifft `16×16`） |
| No-Go | 256 ifft_h `64×4` | 正确；0.36 vs 0.29，占用上去但每线程工作更碎 |
| No-Go | 256 irfft launch block=128 | 正确；0.90 vs 0.84 |
| No-Go | 256 fft_h 16 m2 / 32KB smem | 正确；fwd 0.58 vs 0.52 |
| No-Go | 256 rfft 4 行/块 | 正确；fwd 复测 0.56 vs 0.52 |
| KEEP（已被取代） | 64 档逆每线程 2 像素 | 已被 64 `16×4` 取代；x2 仍留在 `pruned_fft.su` |
| KEEP（正确，默认关） | smem warp 32 点 packed FFT | CPU 蝶形=numpy；GPU shuffle 版 rel≈5；smem 版 PASS 但略慢于 DFT |
| No-Go | 16 线程/行协作 smem DFT | 正确；64 档 1.86 vs 标量 1.66，默认关 `SPECTRAL_COOP` |
| No-Go | `__shfl_xor_sync` 32 点 FFT | 算法对、Biren shuffle 错 |
| No-Go | `br[16]` vec4 逆 / 4 宽全展开 DFT | 256 档回落到 ~35 ms |

## 短探针（本轮，含 64 双像素逆）

| 路径 | 64 | 128 | 256 | 64 正确性 |
|------|--:|----:|----:|-----------|
| spec_dft | 1.546 | 4.652 | 16.263 | PASS |
| fused256（默认） | **1.538** | 4.665 | **14.690** | PASS |
| pack32 smem | 1.579 | 4.660 | 16.309 | PASS 7.08e-6 |

## 官网协议

见 `pruned_continue_test_2026-08-15.md`：默认 **1.303 / 4.314 / 14.522**。相对 qw 0.714 / 1.851 / 6.400 约 **1.8× / 2.3× / 2.3×**。
