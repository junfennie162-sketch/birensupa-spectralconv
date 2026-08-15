# Results

This file does two things: **compare** (same official protocol, reference / start vs our implementation) and **name the code changes**. We did not swap eval data.

Raw terminal output lives under [`results/run_logs/`](results/run_logs/). Agent chat originals are in `agent_logs/`.

## 1. Comparisons

### 1.1 Mandatory operator: official reference vs our implementation

Spectral convolution is scored as the operator problem. It does not read Navier–Stokes data. Correctness is vs the official-style PyTorch dual-corner reference. Performance on the **formal** row is vs the official CPU reference script run on this machine.

| Item | Official / reference | Ours (SUPA + extension) | Note |
|------|----------------------|-------------------------|------|
| Correctness (worst rel) | gate ≤ `1e-4` | **2.170×10⁻⁷** (3/3 PASS) | `reference_pytorch.py` |
| 64×64 forward | 74.142 ms (CPU ref) | **3.797 ms** | ~19.5× · **formal idle** |
| 128×128 forward | 89.000 ms (CPU ref) | **8.037 ms** | ~11.1× · **formal idle** |
| 256×256 forward | 295.983 ms (CPU ref) | **29.295 ms** | ~10.1× · **formal idle** |

Formal idle measured 2026-08-14 (`warmup=10`, `iters=100`, path `auto` on the suFFT fused board). Logs: `results/run_logs/official_recheck_2026-08-14.log`, `spectral_accuracy_2026-08-14.md`, `spectral_perf_2026-08-14.md`.

Default **hot path** in this tree is pruned DFT. Sample **unofficial** CPU-in timings, same official protocol, **not promoted**:

| Resolution | Unofficial pruned (CPU-in) | Formal idle (frozen) |
|------------|----------------------------|----------------------|
| 64×64 | ~0.96 ms | 3.797 ms |
| 128×128 | ~2.21 ms | 8.037 ms |
| 256×256 | ~7.87 ms | 29.295 ms |

Reproduce unofficial numbers with `bash scripts/validate.sh`. Do not run `test_perf.py` unless you intend to rewrite the formal row on an idle card.

### 1.2 Advanced model: same official dataset, before vs after

Data is always the official public NS64 file `navier_stokes_v1e-3_N1200_T20.pt`. We did not edit that file. The comparison is code and training on a locked split.

| Item | Official setting | Our result on that file |
|------|------------------|-------------------------|
| Data | `navier_stokes_v1e-3_N1200_T20.pt` | **same file, unmodified** |
| Split | train 1000 / test 128, seed `20260722` | same |
| Task | 10 frames → frame 11, 64×64, ≥4 FNO layers | 4 layers, width=32, modes=16 |
| Relative L2 (formal) | lower is better | **0.035012** |
| Same-data optimization | 0.041835 when first attached | **0.035012** (~16.3% relative drop) |
| Weights | — | `fno_ns/checkpoints/fno_ns_public_demo.pt` |

Reproduce: build the operator, then `python3 render_official_demo.py` in `fno_ns/` (bring your own official `.pt`).

## 2. What we changed

### 2.1 Operator: from Host round-trips to on-device paths

Early code did spatial work on device, FFT on Host, full-spectrum multiply on CPU, then copy back. At small sizes the **copies** cost more than the multiply.

Frozen **formal** hot path:

1. **suFFT R2C on device**, spectrum stays on the card.
2. **Custom SUPA kernel** for the official dual-corner complex multiply (kept modes only).
3. **C2R on device**. Sources: `spectral_conv/spectral_conv_ext.su`, `spectral_conv_ext.cpp`, `spectral_conv_ops.py`.

**Default** hot path in this release: pruned mixed-radix DFT / iDFT on kept bins (`pruned_*.su`), same dual-corner multiply. `SPECTRAL_PRUNED_FFT=0 SPECTRAL_PRUNED_INV=0` restores suFFT.

Engineering around both paths: weight and spectrum caches; output-buffer reuse; Parameter identity in cache keys; CPU-in `_SPATIAL_OUT_CACHE` for `to_cpu=True`. FNO must not reuse that spatial buffer (`to_cpu=False` allocates a fresh packed irfft). Skip-roundtrip FNO, pinned H2D, and dual-n1 irfft were No-Go and reverted.

Correctness stays on the official-style PyTorch reference. Worst rel **2.170×10⁻⁷**. Bonus: backward, 3D four-corner, irregular shapes.

### 2.2 FNO: reuse the operator, squeeze L2 on official data

- **Four** Fourier layers (width=32, modes=16, 64×64). Inference calls the same SpectralConv.
- **Residual head**: predict the increment vs the last input frame.
- **Official `.pt` untouched.** Split 1000/128, seed `20260722`. Periodic shifts, spectral-weighted loss, late spectral-weight updates, H⁻¹ high-frequency loss. Formal **0.035012**.

Figures come from `fno_ns/render_official_demo.py` on that official test set, then `visualize.py`.
