# R11 · full-spectrum gather + corner scatter · 2026-07-29

## What
- `spectral_mul_gather_scatter_kernel`: read corners from full `x_freq` with strides, write `out_freq` corners.
- `spectral_mul_dual_full_scatter_out` pybind; default hot path (disable with `SPECTRAL_FULL_SCATTER=0`).
- Skip Python `.contiguous()` corner extracts; skip `out_freq.zero_` on cache hit (modes in buffer key).

## Metrics
| item | R10 | R11 |
|------|-----|-----|
| accuracy | PASS | PASS 2.17e-7 |
| A/B 64/128/256 | 5.356/13.640/52.695 | **5.284/13.573/52.563** |
| formal perf | 5.308/13.723/52.746 | **5.281/13.559/52.245** |
| batch16 gps | 1.376M | **1.359M** (48.22 ms/batch) |
| chain ckpt | PASS | PASS 5.09e-5 |

## Decision
**KEEP** as default.
