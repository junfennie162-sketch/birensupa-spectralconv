# P2 · column-FFT truncation (modes2) · 2026-07-29

## Idea (from TurboFNO)
FNO corners only need width-freq bins `[0, modes2)`. After row R2C, run column C2C only on those bins; mirror on iFFT.

## P1 feasibility
- CPU oracle: trunc col-FFT corners match full FFT corners at rel **0**.
- `modes2/Wf`: 64→0.48, 128→0.25, 256→0.12.

## P2 results (B=4, Cin=32, Cout=64, modes=16)
| res | full ms | trunc ms | note |
|-----|---------|----------|------|
| 64 | ~4.0 | ~4.7 | **slower** (narrow/zero/copy overhead) |
| 128 | ~10.3 | ~9.5 | ~+7% |
| 256 | ~40.3 | ~31.5 | ~+22% |

Accuracy: trunc vs full rel **0**; `test_accuracy` PASS with trunc on/off.

## Policy
Default `SPECTRAL_TRUNC_COL=auto`: enable when `modes2/width_freq <= 0.30` (so 64 stays full; 128/256 trunc).
Force: `=1` / `=0`.

## Verdict
**KEEP as auto** (not forced on). Formal 64 not regress; large res gains.
Measured_at: 2026-07-29T10:41:36.930037+00:00
