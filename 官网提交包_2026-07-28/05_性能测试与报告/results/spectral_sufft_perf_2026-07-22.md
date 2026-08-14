# SpectralConv suFFT vs v1 performance

- time_utc: 2026-07-22T10:29:53Z
- config: B=4, C_in=32, C_out=64, modes=16x16, iters=100

| 分辨率 | v1 (ms) | suFFT (ms) | v1 显存 (MB) | suFFT 显存 (MB) |
|---|---:|---:|---:|---:|
| 64x64 | 15.918 | 6.252 | 4.8 | 36.9 |
| 128x128 | 82.0 | 15.944 | 4.8 | 133.1 |
| 256x256 | 254.928 | 87.605 | 4.8 | 517.5 |

summary: `/workspace/ai4s-f/submission/results/summary.json`
