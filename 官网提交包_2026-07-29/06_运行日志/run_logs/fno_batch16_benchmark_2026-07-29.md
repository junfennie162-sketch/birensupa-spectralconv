# FNO batch=16 benchmark

- time_utc: 2026-07-29T02:32:52Z
- device: BIREN single card (Biren106B / supa)
- data: generated_ns_like_v2 (self-generated NS-like v2; not public NS64)
- config: B=16, H=W=64, warmup=10, iters=50
- chain_consistency_rel: 5.03177652717568e-05

| scope | grid_points/s | samples/s | ms/sample | ms/batch | peak MB |
|---|---:|---:|---:|---:|---:|
| pure forward | 1375885.508 | 335.910 | 2.976992 | 47.632 | 169.6 |
| with DataLoader | 1343384.073 | 327.975 | 3.049016 | 48.784 | 169.6 |
