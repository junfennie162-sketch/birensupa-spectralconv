# SpectralConv suFFT vs v1 performance

- time_utc: 2026-07-23T14:30:52Z
- config: B=4, C_in=32, C_out=64, modes=16x16, iters=100

| 分辨率 | v1 (ms) | suFFT (ms) | v1 显存 (MB) | suFFT 显存 (MB) |
|---|---:|---:|---:|---:|
| 64x64 | 16.05 | 36.72 | 4.8 | 37.2 |
| 128x128 | 17.964 | 37.833 | 13.1 | 141.5 |
| 256x256 | 215.916 | 74.788 | 33.4 | 546.1 |

summary: `/workspace/ai4s-f/submission/results/summary.json`
