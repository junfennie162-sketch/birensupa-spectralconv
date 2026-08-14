# R8 · dual_scatter fix + einsum P0 fallback · 2026-07-28

## What

1. **Fixed `spectral_mul_dual_scatter_out`** (was broken: wrong axis `Wf-M1` on height dim; temps never copied back into `out_freq`).
2. **Wired scatter into fused path** (`spectral_conv2d_fused`): one pybind for zero + dual mul + corner scatter; skip Python `_out_freq_buffer.zero_` when scatter zeroes.
3. **P0**: `EinsumConv1x1` + `FNO2d.enable_einsum_skip_fallback()` for SUDNN `nn.Conv2d` crash bypass (not default — call when ErrorCode 6/719).

## Verify (this machine)

| test | result |
|------|--------|
| `test_accuracy.py` | ok, worst_rel **2.170e-7** |
| `test_perf.py` | 64/128/256 → **5.336 / 13.784 / 52.797 ms** |
| `test_chain_cpu_supa_consistency.py` | ckpt rel **4.824e-5** PASS |
| `test_supa_chain.py` | ok, 7/7 intermediates on SUPA |

## Note vs R7 baseline

R7 formal idle was 5.302/13.670/52.480 ms. R8 is within noise; main value is correctness of unused scatter path + P0 fallback ready. Host-seeded D2D for suFFT (P1 Python workaround) unchanged — SDK still needs provenance-safe storage.

## Plan status (OPERATOR_OPT_TODO)

| item | status |
|------|--------|
| P0 SUDNN crash | env currently OK; **einsum fallback landed** |
| P1 rfft2_sufft C++ | keep R7 host-seeded D2D (C++ ABI/provenance limit) |
| P2 Path B fusion | still skip (ROI) |
