# SpectralConv performance (SUPA Extension)

- time_utc: 2026-07-23T15:01:39Z
- path: cpu_fft + SUPA spectral_mul + cpu_ifft
- config: B=4, C_in=32, C_out=64, modes=16x16, warmup=10, iters=100

| 分辨率 | 前向耗时 (ms) | 显存 (MB) |
|---|---:|---:|
| 64x64 | 12.059 | 50.9 |
| 128x128 | 15.902 | 50.9 |
| 256x256 | 75.474 | 563.7 |

```json
[
  {
    "resolution": "64x64",
    "forward_time_ms": 12.059,
    "memory_MB": 50.9,
    "batch": 4,
    "channels_in": 32,
    "channels_out": 64,
    "modes": "16x16",
    "warmup": 10,
    "iters": 100,
    "path": "cpu_fft_supa_mul_cpu_ifft"
  },
  {
    "resolution": "128x128",
    "forward_time_ms": 15.902,
    "memory_MB": 50.9,
    "batch": 4,
    "channels_in": 32,
    "channels_out": 64,
    "modes": "16x16",
    "warmup": 10,
    "iters": 100,
    "path": "cpu_fft_supa_mul_cpu_ifft"
  },
  {
    "resolution": "256x256",
    "forward_time_ms": 75.474,
    "memory_MB": 563.7,
    "batch": 4,
    "channels_in": 32,
    "channels_out": 64,
    "modes": "16x16",
    "warmup": 10,
    "iters": 100,
    "path": "cpu_fft_supa_mul_cpu_ifft"
  }
]
```

summary: `/workspace/ai4s-f/submission/results/summary.json`
