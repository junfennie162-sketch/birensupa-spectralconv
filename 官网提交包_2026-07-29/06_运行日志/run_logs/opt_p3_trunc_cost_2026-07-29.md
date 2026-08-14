# P3 · trunc-col cost cut (narrow-before-permute) · 2026-07-29

## Change

- `rfft2_sufft_trunc`: narrow `modes2` **before** permute+contiguous; stage-cache row/col/out pads; one `zero_` on full pad.
- `irfft2_sufft_trunc`: single pad buffer + stage cache (no double `zeros`).
- `SPECTRAL_TRUNC_COL=auto`: enable when `modes2/width_freq <= 0.50` (was 0.30; 64 now on).

## Micro A/B (B=4, Cin=32, Cout=64, modes=16, force on vs off)

| H | fused off ms | fused on ms | Δ |
|---|---:|---:|---:|
| 64 | 4.172 | 3.668 | +12.1% |
| 128 | 10.472 | 6.289 | +39.9% |
| 256 | 40.628 | 20.464 | +49.6% |

Corner / sparse irfft relative error vs full: **0**.

## Formal (`test_accuracy` / `test_perf`, auto)

- accuracy: PASS, worst rel `2.170e-7`
- perf: **4.563 / 9.607 / 32.024** ms

Vs P2 formal (5.297 / 12.032 / 43.678): about **−14% / −20% / −27%**.

## Verdict

**KEEP**. No ai4s merge/package this round.
