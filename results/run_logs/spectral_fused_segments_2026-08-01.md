# Spectral fused segment profile（旁注）

- time_utc: 2026-08-01T15:02:50Z
- config: B=4 Cin=32 Cout=64 modes=16 warmup=8 iters=30
- **不写** `summary.spectral_conv.perf`（formal 主表冻结）

## Formal idle 主表（只读）

```json
[
  {
    "resolution": "64x64",
    "forward_time_ms": 3.811,
    "memory_MB": 225.3
  },
  {
    "resolution": "128x128",
    "forward_time_ms": 8.054,
    "memory_MB": 253.3
  },
  {
    "resolution": "256x256",
    "forward_time_ms": 29.56,
    "memory_MB": 353.3
  }
]
```

## vs 官网 CPU 加速比

- 64: CPU 74.142 ms / fused 3.811 ms → **19.5×**（相对官网 CPU 参考，非竞品 GPU）
- 128: CPU 89.000 ms / fused 8.054 ms → **11.1×**（相对官网 CPU 参考，非竞品 GPU）
- 256: CPU 295.983 ms / fused 29.560 ms → **10.0×**（相对官网 CPU 参考，非竞品 GPU）

## Fused 分段（本脚本）

```json
[
  {
    "path": "fused_formal",
    "resolution": "64x64",
    "e2e_ms": 3.84,
    "h2d_ms": 0.275,
    "r2c_ms": 1.062,
    "mul_scatter_ms": 0.228,
    "c2r_ms": 2.126,
    "d2h_ms": 0.181,
    "trunc_col": true,
    "note": "\u65c1\u6ce8\u5206\u6bb5\uff1b\u4e0d\u5f97\u8986\u76d6 summary.spectral_conv.perf"
  },
  {
    "path": "fused_formal",
    "resolution": "128x128",
    "e2e_ms": 8.887,
    "h2d_ms": 0.936,
    "r2c_ms": 2.156,
    "mul_scatter_ms": 0.251,
    "c2r_ms": 4.562,
    "d2h_ms": 0.603,
    "trunc_col": true,
    "note": "\u65c1\u6ce8\u5206\u6bb5\uff1b\u4e0d\u5f97\u8986\u76d6 summary.spectral_conv.perf"
  },
  {
    "path": "fused_formal",
    "resolution": "256x256",
    "e2e_ms": 29.312,
    "h2d_ms": 3.779,
    "r2c_ms": 6.745,
    "mul_scatter_ms": 0.255,
    "c2r_ms": 16.537,
    "d2h_ms": 2.297,
    "trunc_col": true,
    "note": "\u65c1\u6ce8\u5206\u6bb5\uff1b\u4e0d\u5f97\u8986\u76d6 summary.spectral_conv.perf"
  }
]
```

## C2R 墙占比（设备段）

- 64x64: C2R 2.126 ms / (R2C+mul+C2R)=3.416 → share≈62.2%
- 128x128: C2R 4.562 ms / (R2C+mul+C2R)=6.969 → share≈65.5%
- 256x256: C2R 16.537 ms / (R2C+mul+C2R)=23.537 → share≈70.3%

## 结论

- mul/scatter 已非主耗时；墙在 C2R（irfft）与端到端同步边界。
- 加速比叙事锚定 official_baseline CPU，禁止写成官方 GPU/SOL 榜。
