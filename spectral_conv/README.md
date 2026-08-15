# Spectral Convolution (mandatory)

FNO's core 2D Spectral Convolution. Default hot path is a **pruned DFT** (kept dual-corner modes only), not a full-frame vendor FFT.

```text
x: [B, C_in, H, W]
  → width mixed-radix rFFT (modes2 bins)
  → height dual-corner DFT (top/bottom modes1)
  → SUPA complex multiply
  → inverse height + inverse width pruned iFFT
y: [B, C_out, H, W]
```

`SPECTRAL_PRUNED_FFT=0 SPECTRAL_PRUNED_INV=0` falls back to suFFT truncation.

## Acceptance

- Relative error ≤ `1e-4` vs `reference_pytorch.py`
- Official 3-case accuracy + 64/128/256 perf (`B=4, Cin=32, Cout=64, modes=16`)

## One-command check (does not write formal idle)

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
bash scripts/validate.sh
```

Formal idle stays in `results/summary.json`. Only an idle exclusive GPU should run `python3 test_perf.py`.

## Files

| File | Role |
|------|------|
| `spectral_conv_ops.py` | Assemble pruned DFT / suFFT fallback + SUPA mul |
| `spectral_conv_ext.cpp` / `.su` | pybind, workspace, complex multiply |
| `pruned_*.su` | Hot-path kernels (mixed-radix rfft / fft_h / ifft_h / irfft) |
| `build.sh` | Compile and link `spectral_conv_ext*.so` |
| `reference_pytorch.py` | CPU gold |
| `test_accuracy.py` | Official 3-case accuracy |
| `test_perf.py` | Formal 64/128/256 idle (**writes** summary) |
| `probe_pruned_continue.py` | Unofficial reproduce: 3-case + warmup=10/iters=100 |
| `test_backward.py` / `test_3d_accuracy.py` / `test_irregular_shapes.py` | Bonus |
| `tune.py` | `auto` route (not a scoring sentence) |

## Status

- Reported: **0.961 / 2.207 / 7.870 ms** (pruned DFT CPU-in KEEP, 2026-08-15)
- Previous suFFT idle: 3.797 / 8.037 / 29.295 ms (2026-08-14)
- Worst rel (official 3-case): ≈ **2.17e-7**
