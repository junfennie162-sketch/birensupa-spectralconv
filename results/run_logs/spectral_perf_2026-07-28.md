# SpectralConv performance (SUPA Extension)

- time_utc: 2026-07-28T15:27:46Z
- path: auto (loads tune_results.json; FNO to_cpu=False forces fused)
- config: B=4, C_in=32, C_out=64, modes=16x16, warmup=10, iters=100

| 分辨率 | 前向耗时 (ms) | 显存 (MB) |
|---|---:|---:|
| 64x64 | 32.002 | 146.6 |
| 128x128 | 66.138 | 238.5 |
| 256x256 | 100.999 | 582.7 |

```json
[
  {
    "resolution": "64x64",
    "forward_time_ms": 32.002,
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
    "forward_time_ms": 66.138,
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
    "forward_time_ms": 100.999,
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
