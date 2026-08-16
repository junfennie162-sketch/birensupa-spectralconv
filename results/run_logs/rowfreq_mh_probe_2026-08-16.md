# 256 中间谱 [B,C,M2,H]（2026-08-16）

> 旁注。独立 TU，`SPECTRAL_ROWFREQ_MH` 默认关。**No-Go**（rFFT 写出变跨步，隔离更慢）。**未写** `summary.json`。

- vs CPU 1.390e-06 · vs KEEP 0.000e+00
- 隔离 fwd B=4 KEEP 0.593 / MH 0.805
- 隔离 fwd B=1 KEEP 0.160 / MH 0.270
- e2e 三轮中位 KEEP 6.193 / MH 6.405（-3.4%）
