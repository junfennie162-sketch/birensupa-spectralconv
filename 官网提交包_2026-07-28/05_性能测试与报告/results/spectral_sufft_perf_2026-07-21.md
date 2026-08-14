# SpectralConv suFFT vs v1 performance

- time_utc: 2026-07-21T14:32:30Z
- config: B=4, C_in=32, C_out=64, modes=16x16, iters=100

| 分辨率 | v1 (ms) | suFFT (ms) | v1 显存 (MB) | suFFT 显存 (MB) |
|---|---:|---:|---:|---:|
| 64x64 | 14.006 | 49.089 | 4.8 | 32.2 |
| 128x128 | 18.08 | 48.995 | 4.8 | 128.4 |
| 256x256 | 248.148 | 119.916 | 4.8 | 512.8 |

summary: `/workspace/ai4s-f/submission/results/summary.json`
