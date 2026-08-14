# Optimization baseline (P0)

- time_utc: 2026-07-22T10:07:59Z
- config: B=4, Cin=32, Cout=64, modes=16, iters=30

## Frozen end-to-end (from summary.json)

- v1 rows: [{"resolution": "64x64", "forward_time_ms": 14.064, "memory_MB": 4.8}, {"resolution": "128x128", "forward_time_ms": 23.033, "memory_MB": 4.8}, {"resolution": "256x256", "forward_time_ms": 214.949, "memory_MB": 4.8}]
- sufft rows: null

## Segment timing (this profile)

```json
[
  {
    "path": "v1_cpu_fft",
    "resolution": "64x64",
    "fft_ms": 0.081,
    "h2d_mul_d2h_ms": 1.005,
    "ifft_ms": 3.446,
    "note": "h2d_mul_d2h includes spectral_mul_supa full bridge"
  },
  {
    "path": "sufft_with_cpu_bridge",
    "resolution": "64x64",
    "rfft_ms": 1.291,
    "bridge_mul_ms": 26.242,
    "irfft_ms": 17.367,
    "note": "bridge_mul = spectrum D2H + mul H2D/D2H + spectrum H2D"
  },
  {
    "path": "v1_cpu_fft",
    "resolution": "128x128",
    "fft_ms": 3.179,
    "h2d_mul_d2h_ms": 4.168,
    "ifft_ms": 10.07,
    "note": "h2d_mul_d2h includes spectral_mul_supa full bridge"
  },
  {
    "path": "sufft_with_cpu_bridge",
    "resolution": "128x128",
    "rfft_ms": 11.4,
    "bridge_mul_ms": 14.139,
    "irfft_ms": 12.865,
    "note": "bridge_mul = spectrum D2H + mul H2D/D2H + spectrum H2D"
  },
  {
    "path": "v1_cpu_fft",
    "resolution": "256x256",
    "fft_ms": 57.832,
    "h2d_mul_d2h_ms": 14.023,
    "ifft_ms": 161.429,
    "note": "h2d_mul_d2h includes spectral_mul_supa full bridge"
  },
  {
    "path": "sufft_with_cpu_bridge",
    "resolution": "256x256",
    "rfft_ms": 12.669,
    "bridge_mul_ms": 54.311,
    "irfft_ms": 28.93,
    "note": "bridge_mul = spectrum D2H + mul H2D/D2H + spectrum H2D"
  }
]
```

## Conclusion

- Expected: H2D/D2H around `spectral_mul_supa` dominates small/medium resolutions.
- P1 target: keep spectrum on SUPA; mul without CPU bridge.
