# SpectralConv performance (SUPA Extension)

- time_utc: 2026-07-25T12:24:30Z
- path: cpu_fft + SUPA spectral_mul + cpu_ifft
- config: B=4, C_in=32, C_out=64, modes=16x16, warmup=10, iters=100

| 分辨率 | 前向耗时 (ms) | 显存 (MB) |
|---|---:|---:|
| 64x64 | 5.337 | 146.6 |
| 128x128 | 13.737 | 238.5 |
| 256x256 | 52.692 | 582.7 |

```json
[
  {
    "resolution": "64x64",
    "forward_time_ms": 5.337,
    "memory_MB": 146.6,
    "batch": 4,
    "channels_in": 32,
    "channels_out": 64,
    "modes": "16x16",
    "warmup": 10,
    "iters": 100,
    "path": "auto"
  },
  {
    "resolution": "128x128",
    "forward_time_ms": 13.737,
    "memory_MB": 238.5,
    "batch": 4,
    "channels_in": 32,
    "channels_out": 64,
    "modes": "16x16",
    "warmup": 10,
    "iters": 100,
    "path": "auto"
  },
  {
    "resolution": "256x256",
    "forward_time_ms": 52.692,
    "memory_MB": 582.7,
    "batch": 4,
    "channels_in": 32,
    "channels_out": 64,
    "modes": "16x16",
    "warmup": 10,
    "iters": 100,
    "path": "auto"
  }
]
```

summary: `/workspace/ai4s-f/submission/results/summary.json`
