# FNO L2 · 2026-07-31

## Current demo (official 1000/128)
**0.005143815** (continue3 @ep33) — FINAL for same-arch squeeze

Trajectory: 0.005488 → freeze2 0.005470 → global-continue 0.005268 → freeze3 0.005255 → continue2 0.005178 → freeze4 0.005171 → **continue3 0.005144**

## Succeeded
| Step | L2 | Note |
|------|-----|------|
| freeze2 | 0.005470 | freeze spectral |
| global-continue | 0.005268 | lr=2e-5, 100ep |
| freeze3 | 0.005255 | lr=3e-6, 80ep |
| continue2 | 0.005178 | lr=1e-5, 80ep |
| freeze4 | 0.005171 | lr=2e-6, 60ep |
| continue3 | **0.005144** | lr=5e-6, 50ep |

## Failed / aborted
- multiwin, matched-loss, modes=12, width48, modes20, naive wavg

## Stop
Same-arch plateau; next only if new data/arch/SDK lever.
