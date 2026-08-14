# Irregular shape stress — 2026-07-23

Tool: `test_irregular_shapes.py` (new file).

## Coverage

9 shapes beyond §3.2 powers-of-two:
- non-square: 64×40, 40×64, 256×64
- odd / non-power-of-2: 48, 72, 96, 160, 192, 100

## Result

| shape      | B_Cin_Cout | rel        | peak_mb | ok  |
|------------|------------|------------|---------|-----|
| 64x40      | 4_32_64    | 2.759e-07  | 20.2    | OK  |
| 40x64      | 4_32_64    | 2.938e-07  | 20.2    | OK  |
| 48x48      | 4_32_64    | 2.533e-07  | 20.4    | OK  |
| 72x72      | 4_32_64    | 2.730e-07  | 83.9    | OK  |
| 96x96      | 4_32_64    | 2.934e-07  | 125.0   | OK  |
| 160x160    | 4_32_64    | 3.920e-07  | 254.7   | OK  |
| 192x192    | 4_32_64    | 3.531e-07  | 350.1   | OK  |
| 256x64     | 4_32_64    | 3.227e-07  | 146.2   | OK  |
| 100x100    | 2_16_64    | 3.673e-07  | 103.9   | OK  |
| **worst**  |            | 3.92e-07   | 350.1   | 9/9 |

All worst rel well below 1e-4 threshold. Memory peaks are linear in shape
volume as expected. No surprises: the fused suFFT path handles irregular
shapes out of the box, and v1 path's `torch.fft.rfft2` is shape-agnostic.

## Implication

The auto-resolver's `min(H, W) >= 64 → fused` decision still holds for
all these shapes (smallest dim is 40, but 40 falls back to v1). The fused
path therefore triggers on 96, 160, 192, 100 — all pass.