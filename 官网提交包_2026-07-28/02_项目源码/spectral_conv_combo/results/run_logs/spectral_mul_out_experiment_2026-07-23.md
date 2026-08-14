# `spectral_mul_out` + buffer cache experiment — 2026-07-23

## Hypothesis

The fused path allocates `(B, Cout, M1, M2, 2)` SUPA output twice per
call (one per corner). SUPA caching allocator might re-use freed blocks
in-loop, but the implicit `torch::empty()` in `launch_spectral_mul` costs
~5-10 ms per call. Re-using a pre-allocated buffer via a new
`spectral_mul_out` Python wrapper should win 5-15 ms total at 256.

## Implementation

1. `spectral_conv_ext.cpp`: added `spectral_mul_out(x, w, out)` that calls
   the same kernel with `out.data_ptr()` instead of `torch::empty()`.
2. `spectral_conv_ops.py`: added `_y_freq_buffer` cache and switched
   fused path to `spectral_mul_supa_out`.
3. Rebuilt `.so` via `./build.sh` (21 s, clean compile).

## Result

| res | pre (spectral_mul) | post (spectral_mul_out) | delta |
|-----|--------------------|--------------------------|-------|
| 64  | 5.31 ms            | 5.31 ms                  | ≈0    |
| 128 | 13.69 ms           | 13.65 ms                 | −0.04 |
| 256 | 52.59 ms           | 52.41 ms                 | −0.18 |

**Negligible.** The SUPA caching allocator already reuses the freed blocks
in-loop, so `torch::empty()` is essentially free. The cache adds complexity
without measurable benefit.

## Decision

- `spectral_mul_out` exposed in `spectral_conv_ext.so` for future use
  (e.g. fused-with-output-reuse in a custom kernel chain).
- `_y_freq_buffer` retained in `spectral_conv_ops.py` as dead code with
  a comment, so future experiments can flip it on.
- Fused path reverted to plain `spectral_mul_supa_device` calls.

## Lesson

**Profiling before coding** matters. The `profile_segments_v2.py` sweep
showed a 20 ms "residual" at 256 — I assumed that was allocator. It
wasn't; it was the irfft + D2H + sync overhead, all of which need a real
kernel rewrite (not a Python trick) to optimise.