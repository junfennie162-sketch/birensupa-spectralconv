# FNO training throughput (bonus)

- measured_at: 2026-07-25T13:06:25Z
- scope: one optimizer step = forward + relative-L2 loss + backward + Adam step
- device/path: cpu / use_supa=False (matches submitted checkpoint training path)
- config: B=8, 64x64, warmup=10, iters=50, seed=20260722

| metric | value |
|---|---:|
| grid_points/s | 34711.585 |
| samples/s | 8.475 |
| ms/sample | 118.001 |
| ms/batch (step) | 944.008 |

Not the inference batch=16 main table.
