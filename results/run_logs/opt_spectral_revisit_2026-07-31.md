# Spectral revisit · 2026-07-31

## Idle recheck (valid)
- Accuracy: **PASS** (worst_rel ≈ 2.17e-7)
- Perf: **3.811 / 8.054 / 29.560 ms** @64/128/256
- Prior P8b confirm: 3.807 / 8.001 / 29.162 ms — within noise → **platform**

## Invalid (discarded)
- Concurrent with FNO CPU train: 24.923 / 44.999 / 100.001 ms
- `test_perf.py` now skips summary write if ms64 > 12 (contention guard)

## Remaining ROI
No high-ROI compliant micro-opts left (irfft/C2R wall). SDK blocks: torch.fft@SUPA, Plan2d, strided pack, fusion API.

## Verdict
SpectralConv **stop digging** unless new SDK API appears.
