# SpectralConv performance (SUPA Extension)

- time_utc: 2026-07-31T02:54:15Z
- path: auto (loads tune_results.json; FNO to_cpu=False forces fused)
- config: B=4, C_in=32, C_out=64, modes=16x16, warmup=10, iters=100

| 分辨率 | 前向耗时 (ms) | 显存 (MB) |
|---|---:|---:|
| 64x64 | 3.811 | 225.3 |
| 128x128 | 8.054 | 253.3 |
| 256x256 | 29.56 | 353.3 |

```json
[
  {
    "resolution": "64x64",
    "forward_time_ms": 3.811,
    "memory_MB": 225.3,
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
    "forward_time_ms": 8.054,
    "memory_MB": 253.3,
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
    "forward_time_ms": 29.56,
    "memory_MB": 353.3,
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
