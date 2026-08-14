# FNO batch=16 benchmark

- time_utc: 2026-07-29T02:58:37Z
- device: BIREN single card (Biren106B / supa)
- data: generated_ns_like_v2 (self-generated NS-like v2; not public NS64)
- config: B=16, H=W=64, warmup=10, iters=50
- chain_consistency_rel: 5.03177652717568e-05

| scope | grid_points/s | samples/s | ms/sample | ms/batch | peak MB |
|---|---:|---:|---:|---:|---:|
| pure forward | 1359951.091 | 332.019 | 3.011873 | 48.190 | 178.1 |
| with DataLoader | 1322612.069 | 322.903 | 3.096902 | 49.550 | 178.1 |
