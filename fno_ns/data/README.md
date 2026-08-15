# FNO-NS data

Official public NS64 is **not** in this repository (GitHub 100 MB file limit).

Place the unmodified official tensor here:

| File | Split | Seed |
|------|--------|------|
| `navier_stokes_v1e-3_N1200_T20.pt` | train 1000 / test 128 | `20260722` |

The loader prefers a non-`ns_like*` filename. Synthetic `ns_like_v2_*.pt` caches are engineering side notes and must not be reported as the public score.

```bash
# huggingface-cli download abelsr1710/navier-stokes-2d-fno \
#   navier_stokes_v1e-3_N1200_T20.pt --local-dir ./
```
