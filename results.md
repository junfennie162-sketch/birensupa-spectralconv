# Results

This file does two things: **compare** (same official protocol, reference / start vs our implementation) and **name the code changes**. We did not swap eval data.

Raw terminal output lives under [`results/run_logs/`](results/run_logs/). Agent chat originals are in `agent_logs/`.

## 1. Comparisons

### 1.1 Mandatory operator: official reference vs our implementation

Spectral convolution is scored as the operator problem. It does not read Navier–Stokes data. Correctness is vs the official-style PyTorch dual-corner reference. Performance on the **formal** row is vs the official CPU reference script run on this machine.

| 项目 | 官网 / 参考 | 我们优化后（SUPA + Extension） | 说明 |
|------|-------------|-------------------------------|------|
| 正确性（最差相对误差） | 门槛 ≤ `1e-4` | **7.162×10⁻⁶**（3/3 PASS） | 默认裁剪路径官方三案；suFFT 三案曾报 2.170×10⁻⁷ |
| 64×64 前向 | 74.142 ms（CPU 参考） | **0.599 ms** | 约 123.8× · 摊还 pinned 输入 |
| 128×128 前向 | 89.000 ms（CPU 参考） | **1.405 ms** | 约 63.3× · 摊还 pinned 输入 |
| 256×256 前向 | 295.983 ms（CPU 参考） | **5.099 ms** | 约 58.0× · 摊还 pinned 输入 |

对应原始 log：`results/run_logs/promote_pinned_src_2026-08-16.md`（`pinned_src_r1` · v13）。上一主表 0.764 / 1.827 / 6.504。

Reproduce with `bash scripts/validate.sh`. Do not run `test_perf.py` unless you intend to rewrite `summary.json` on an idle card.

### 1.2 Advanced model: same official dataset, before vs after

Data is always the official public NS64 file `navier_stokes_v1e-3_N1200_T20.pt`. We did not edit that file. The comparison is code and training on a locked split.

| Item | Official setting | Our result on that file |
|------|------------------|-------------------------|
| Data | `navier_stokes_v1e-3_N1200_T20.pt` | **same file, unmodified** |
| Split | train 1000 / test 128, seed `20260722` | same |
| Task | 10 frames → frame 11, 64×64, ≥4 FNO layers | 4 layers, width=32, modes=16 |
| Relative L2 (formal) | lower is better | **0.035012** |
| Same-data optimization | 0.041835 when first attached | **0.035012** (~16.3% relative drop) |
| Weights | — | `fno_ns/checkpoints/fno_ns_public_demo.pt` |

Reproduce: build the operator, then `python3 render_official_demo.py` in `fno_ns/` (bring your own official `.pt`).

## 2. What we changed

### 2.1 Operator: from Host round-trips to on-device paths

现在默认热路径是**裁剪 DFT**（只变换保留的低频双角），不是整幅厂商 FFT：

1. **宽度混合基 rFFT + 高度两角 DFT**（只算 kept modes）。
2. **自研 SUPA kernel** 做官网同款双角复数乘。
3. **裁剪 iFFT** 变回空间域；CPU 入走 `_SPATIAL_OUT_CACHE`。源码：`spectral_conv/pruned_*.su`、`spectral_conv_ext.cpp`、`spectral_conv_ops.py`。

`SPECTRAL_PRUNED_FFT=0 SPECTRAL_PRUNED_INV=0` 可退回 suFFT fused（上一板 idle 3.797 / 8.037 / 29.295 ms）。

### 2.2 FNO: reuse the operator, squeeze L2 on official data

性能对比对象是本机跑出来的**官网 CPU 参考脚本**：74.142 / 89.000 / 295.983 ms → **0.599 / 1.405 / 5.099 ms**。

Figures come from `fno_ns/render_official_demo.py` on that official test set, then `visualize.py`.
