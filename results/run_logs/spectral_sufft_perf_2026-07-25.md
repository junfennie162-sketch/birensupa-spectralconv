# SpectralConv suFFT vs v1 performance

- time_utc: 2026-07-25T12:23:09Z
- config: B=4, C_in=32, C_out=64, modes=16x16, iters=100

| 分辨率 | v1 (ms) | suFFT (ms) | v1 显存 (MB) | suFFT 显存 (MB) |
|---|---:|---:|---:|---:|
| 64x64 | 12.966 | 5.331 | 9.5 | 41.7 |
| 128x128 | 14.95 | 13.641 | 22.6 | 150.0 |
| 256x256 | 200.039 | 52.502 | 46.9 | 558.7 |

summary: `/workspace/ai4s-f/submission/results/summary.json`
