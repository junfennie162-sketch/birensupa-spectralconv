# FNO chain layer-by-layer profile — 2026-07-23

Tool: `fno_ns/profile_chain.py` (new).

## Setup

```
B=4, H=W=64, width=32, modes=16, layers=4
spectral_conv2d_fused(to_cpu=False)
```

## Numbers

| stage | ms |
|---|---|
| full chain (4 layers + lift + proj) | **15.49** |
| L1 (spectral + conv + IN + gelu) | 3.40 |
| L2 | 3.38 |
| L3 | 3.40 |
| L4 | 3.38 |
| sum of layers | 13.55 |
| proj | 0.49 |
| lift (included in full above) | ~1.45 |
| per-call overhead (full − sum) | ~1.94 |

Standalone `spectral_conv2d_supa(x, w1, w2, 16, 16, 'auto')` at the same
shape (B=4, Cin=32, Cout=32, 64×64) = **3.67 ms** (with proper warmup).

## Conclusion

Each FNO layer is **3.4 ms**, of which the spectral portion is ~3.4 ms
(the conv 1×1 + InstanceNorm + GELU are < 0.1 ms combined because they
operate on small width=32 channels). The chain runs at ~spectral speed;
no Python-level optimisations can shave time because:

- `_OUT_FREQ_CACHE` already reuses spectrum buffers across calls (same
  shape every layer, since width=32 is constant).
- `_WEIGHT_SUPA_CACHE` already avoids per-layer H2D of weights.
- `forward_supa_chain` runs fully async on the SUPA stream (no per-layer
  `synchronize()`).
- The two `spectral_mul_supa_device` calls per layer are *already*
  serialised by the kernel's nullptr-stream semantics; no further
  concurrency available without rebuilding `.so`.

## Where the time *actually* goes (per `profile_segments_v2.py`)

At 64 the irfft is ~2.5 ms out of 5.3 ms spectral total. That's the
single biggest kernel within each layer. There is no Python-side win —
the only further reduction requires fusing rfft + mul + irfft into one
SUPA kernel, which is a `.cu` rewrite outside tonight's scope.

## 256 irfft alternative (rejected)

Profile: `profile_irfft_alt.py` at 256×256.

| path | ms |
|---|---|
| SUPA irfft | 27.1 |
| D2H + CPU irfft | 199.9 |
| D2H only | 30.8 |
| CPU irfft only | 199.1 |

CPU irfft is **7× slower** than SUPA irfft at 256. Don't switch.

## Net

- FNO chain stays at 15.5 ms / 4 layers (3.4 ms/layer). No code change.
- irfft stays on SUPA. No code change.
- Both findings documented for future kernel-rewrite work.