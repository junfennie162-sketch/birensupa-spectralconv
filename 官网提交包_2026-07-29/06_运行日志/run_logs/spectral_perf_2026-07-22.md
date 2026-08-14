# SpectralConv performance (SUPA Extension)

- time_utc: 2026-07-22T10:30:28Z
- path: cpu_fft + SUPA spectral_mul + cpu_ifft
- config: B=4, C_in=32, C_out=64, modes=16x16, warmup=10, iters=100

| 分辨率 | 前向耗时 (ms) | 显存 (MB) |
|---|---:|---:|
| 64x64 | 14.965 | 4.8 |
| 128x128 | 19.964 | 4.8 |
| 256x256 | 223.113 | 4.8 |

```json
[
  {
    "resolution": "64x64",
    "forward_time_ms": 14.965,
    "memory_MB": 4.8,
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
    "forward_time_ms": 19.964,
    "memory_MB": 4.8,
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
    "forward_time_ms": 223.113,
    "memory_MB": 4.8,
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
