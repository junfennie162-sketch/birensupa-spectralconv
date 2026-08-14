# FNO-NS supa_chain (R4 — after the `.detach()` fix)

- time_utc: 2026-07-24T09:48Z (2026-07-24 17:48 CST)
- protocol: `forward_supa_chain` on `FNO2d` with `prepare_supa_eval()` called
  once before timing, then warmed up 10 iterations and timed over 50.
- config: B=4, HxW=64x64, width=32, modes=16, n_layers=4, t_in=10

| metric | value |
|---|---:|
| **median ms** (over 50 iters after 10 warmup) | **16.092** |
| min ms | 16.022 |
| peak device memory MB | 266.7 |
| per layer median | ~3.55 ms (4 layers × layer cost + 0.5 ms proj) |
| project median | n/a (folded into full) |

## Regression history (same protocol)

| round | date | commit / change | median ms | speedup vs R3 |
|---|---|---|---:|---:|
| R3 | 2026-07-24 16:39 | `forward_supa_chain` first cut (still had `.detach()`) | 49.118 | 1.0× |
| **R4** | **2026-07-24 17:48** | **`forward_supa_chain` with raw `nn.Parameter` (no `.detach()`), `prepare_supa_eval` removed from hot path, `try/except` + `_y_freq_buffer.zero_` cleaned up** | **16.092** | **3.05×** |

## Side-by-side comparison (R4 same-session rebench)

| | f R4 | n R4 (no code change) | delta |
|---|---:|---:|---:|
| auto perf 64 | 5.330 ms | 5.331 ms | f −0.001 |
| auto perf 128 | 13.692 ms | 13.845 ms | f −0.153 |
| auto perf 256 | 52.753 ms | 52.861 ms | f −0.108 |
| **FNO chain full** | **16.092 ms** | **15.452 ms** | **n −0.64 ms (4% flapping)** |
| spectral_accuracy worst rel | 2.84e-7 | 2.83e-7 | (n micro) |
| FNO L2 | 0.009516 | (n/a) | — |

## Root cause for the R3 → R4 jump

`FourierLayer.forward_supa` previously called:

```python
y = spectral_conv2d_supa(
    x,
    self.spectral_conv.weights1.detach(),   # creates fresh id()
    self.spectral_conv.weights2.detach(),
    ...,
)
```

`nn.Parameter.detach()` returns a new view with **different `id()` and
`isinstance(..., nn.Parameter) == False`**. The `_weights_to_supa_cached`
helper keys on `isinstance(..., nn.Parameter)`, so every call fell through
to the content-hash branch — a D2H + numpy + blake2b round-trip, ~1 ms each,
2 weights × 4 layers = **8 such round-trips per forward pass**.

Removing `.detach()` puts the parameter object back on the O(1) id-keyed
cache path; the chain drops to ~16 ms (per-layer ~3.5 ms, matching n).

See `development_log.md` record 16 and `skills/spectral_chain_optimization.md`
§7 for the full write-up.
