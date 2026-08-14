# R12 · FFT transpose / stage-cache · 2026-07-29 · ROLLBACK

## Attempts
1. **R12**: custom `transpose_pab2` + stage buffers replacing `permute().contiguous()` in rfft/irfft.
2. **R12b**: keep permute contig, only cache `col_out` empty buffer.

## Results
| variant | 64/128/256 ms | notes |
|---------|---------------|-------|
| R11 baseline | 5.281 / 13.559 / 52.245 | KEEP |
| R12 transpose | ~7.4 / 15.7 / 56.9 (micro) | **FAIL** slower; FFT path regress |
| R12b col_out cache | 5.404 / 13.482 / 52.257 | **ROLLBACK** 64 regress + peak mem 145→272 MB |

## Lessons
- Biren: `permute().contiguous()` (Memcpy2D) beats hand-written element transpose kernel.
- Caching FFT staging that is immediately `.contiguous()`-copied out mainly inflates memory.
- Profile: at B16×C32×64, cost is ~rfft 4.9 + irfft 4.8 + mul 0.6 + D2D 0.3 ms; D2D/stream overlap not the lever.

## Code
- Left `launch_transpose_pab2` in `.su` unused (dead) — can delete later; hot path restored to R11.
