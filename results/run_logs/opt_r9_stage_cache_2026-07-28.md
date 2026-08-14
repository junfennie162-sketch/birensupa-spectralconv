# R9 · suFFT stage-cache experiment (discarded) + dual_scatter temp cache

## Attempt
Cache `rfft2_sufft` / `irfft2_sufft` intermediates via `copy_(permute)` into process-local buffers.

## Result
- Accuracy still PASS (worst_rel ~2.17e-7).
- Idle-looking wall times became **~32 / 66 / 101 ms** — **invalid** while official 100-ep CPU train saturates host cores (GPU process list empty; launch latency inflated).
- Also confirmed on this SDK: strided `copy_(permute)` is slower than `permute().contiguous()` (same lesson as Python corner-slice notes).

## Kept
- `spectral_mul_dual_scatter_out` reuses cached `y_top`/`y_bot` stage buffers (fixes prior per-call `empty()` churn). Hot path still `dual_out` until clean A/B after train.

## Formal numbers
Do **not** update summary from benches concurrent with the 100-ep train. Restored R8 formal 5.336/13.784/52.797 ms.
