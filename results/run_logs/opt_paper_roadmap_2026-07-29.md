# Paper-inspired next opts · 2026-07-29

## Already harvested (TurboFNO-style)
- Column-FFT truncation to `modes2` (P2–P5 packed)
- Gather-scatter spectral mul (R11)
- Stage / plan / workspace caches

## Checked tonight — blocked on Biren
| Idea | Result |
|------|--------|
| `torch.fft` on SUPA | **Wrong** (rel≈1 vs CPU); cannot replace suFFT |
| `sufftBuildPlan2d` / `PlanMany` | In **header only**; `libsufft.so` exports **only `BuildPlan1d`** |
| Hybrid torch.fft + SUPA mul | Incorrect spectra |
| Strided pack `copy_` | Biren CopyD2D crash |

## Still open (need retrain / arch)
1. **Factorized FNO (F-FNO)** — 1D spectral along H then W; often similar L2, different FFT mix
2. **modes=12** (model default already 12; official table uses 16) — ~10% layer time @64; must retrain
3. **Wider/shallower or U-FNO / AFNO** — larger change
4. **Train schedule / data** — more epochs, lr, augmentation for L2 (not spectral ms)
5. **True FFT–GEMM–iFFT fusion** — needs device FFTDx-like API we do not have

## Bottleneck reality
@256 fused: **irfft/C2R dominates** (~16 ms); mul ~0.3 ms. Further trunc/mul micro-opts ≪ FFT wall.
