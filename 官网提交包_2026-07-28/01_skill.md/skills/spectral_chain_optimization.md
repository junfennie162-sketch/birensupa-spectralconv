# Optimization Notes — SpectralConv + FNO on BIREN SUPA

> Captured 2026-07-24 16:30 by Cursor Agent during the **catch-up round** vs `ai4s-n`.
> Goal: codify the lessons that pushed f and n onto identical perf plates
> (5.3 / 13.7 / 52.7 ms at 64/128/256), plus the FNO chain gap that remains.

---

## 1. "auto" path threshold = the real bottleneck, not the kernel

When `use_sufft="auto"` is wired and `>=64 → fused` (or `>=128 → fused`), the
**fused** suFFT path on BIREN delivers the same performance regardless of
which side of the threshold we land on for 64/128/256. With proper warmup +
`nn.Parameter`-cached weights, the auto-bound fused path on a single Biren106B
hits the same ~5.3 ms (64) / ~13.7 ms (128) / ~52.7 ms (256) for **both**
f's `spectral_conv_ops.py` and n's. The 256 gap from f's old 79 ms baseline
was almost entirely the `cpu_fft + bridge_mul + cpu_ifft` path being left
on for too long — switching the threshold flipped it to fused and we landed
within 0.2 ms of n.

**Take-away**: Don't ship a v1-only path on official sizes; the FFT kernel
floor is the same on both sides, and the auto rule costs you nothing.

---

## 2. FNO chain device-resident — the only *real* gap to n

`SpectralConv2d.forward(use_supa=True)` defaults to `to_cpu=True` because the
`cpu_fft` branch always returns CPU. The chain then does a per-layer D2H
which costs ~8 ms × 4 layers ≈ ~32 ms before any kernel even starts.

n's `forward_supa_chain` adds three things:

1. **`to_cpu=False`** piped end-to-end through `spectral_conv2d_supa(..., use_sufft="auto", to_cpu=False)`.
2. **`prepare_supa_eval()`** that moves `nn.InstanceNorm2d.running_mean` /
   `running_var` / `num_batches_tracked` onto SUPA explicitly. (`nn.Module.to()`
   does NOT recurse into IN running stats.)
3. A chain-level forward that stays on `x.device` all the way to the final
   `self.project(x).cpu()`.

f now mirrors this in `FNO2d.forward_supa_chain` + `FourierLayer.forward_supa`.
Empirically: even with the device-resident path, f's per-layer spectral
latency at 64×64 / width=32 / supa input+weights is ~11 ms (≈ 3× n's
~3.4 ms). The remaining gap is **not** layer overhead — it's the suFFT plan
cache being cold for cross-layer shapes and the per-call allocator's
re-touching of `_OUT_FREQ_CACHE` between layers. Future improvement:
ship one pooled spectrum buffer across all 4 layers so plans stay warm.

---

## 3. SDK API inventory (BIREN 1.11.0.0.rc2) — what *actually* works

| API | Exported? | Wired by f? | Note |
|---|---|---|---|
| `sufftCreatePlan` / `BuildPlan1d` / `ExecR2C` / `ExecC2C` / `ExecC2R` | yes | yes | foundation |
| `sufftSetWorkArea` + `sufftGetSize1d` | yes | **yes (2026-07-24)** | pin per-shape scratch buffer |
| `sufftSetStream` | yes | no | needs `<torch_br/csrc/core/supa/SUPAStream.h>` which pulls in `sutlass.h` — not on the build path |
| `sufftDestroy` | yes | no | plan cache leaks; harmless because plans are reused |
| `sufftBuildPlan2d` / `3d` / `Many` | **NO (header-only stubs)** | no | n already tried `2d` and reverted (commit `14b7205`) |
| `sufftSetCallback*` | NO | no | BIREN FFT doesn't support cuFFT-style callbacks |
| `spectral_mul_out` (ext.) | N/A | **yes (2026-07-24)** | in-place variant, paired with `_y_freq_buffer` |

**Take-away**: roughly half the headers in `sufft.h` are aspirational. The
only safe play is what's exported by `nm -D libsufft.so.0.7.0`.

---

## 4. `_y_freq_buffer` corner-id pitfall

When caching the per-corner `(B, Cout, M1, M2, 2)` SUPA buffer, **always
include `corner_id` (or equivalent) in the cache key**. C1's first cut
shared one buffer across both corner1 and corner2: `spectral_mul_out` writes
into it twice in sequence and the second write silently overlaps the
first's strided view — correctness drops to rel = 1.0 (zero output).
n's `_y_freq_buffer` adds `corner_id`; we now do too.

---

## 5. `InstanceNorm2d.to(device)` doesn't move running stats

`nn.InstanceNorm2d.to("supa")` moves `weight` and `bias` but leaves
`running_mean` / `running_var` / `num_batches_tracked` on their original
device. If you ever want IN-on-SUPA at eval, you have to move them
manually in `prepare_supa_eval()` (or fall back to GroupNorm(1, C) which
is parameter-free and works on any device).

---

## 6. Path to a 3× FNO chain speedup (open)

R3: f's chain went 47 ms (full D2H) → 49 ms with device-resident path. We
moved the bottleneck, not removed it. Real wins to chase:

1. **Plan-cache hit between layers**: warm all 4 layer plans up front in
   `prepare_supa_eval`, not lazily on first call.
2. **Drop one permute**: fused rfft does 1× `permute().contiguous()` between
   the two 1D stages; the columns could be C2C'd directly out of the R2C
   interleaved buffer's strides — saves ~2-3 ms @ 64.
3. **Persistent suFFT workspace**: swap each plan's `sufftSetWorkArea` from
   per-plan to **shared across all plans**, reducing peak workspace by ~4×.
4. **cuFFT callback**: not available on BIREN. Skip.

Each of these is 2-3 ms @ 64, all cumulative → 49 → 30 ms, with maybe 5 ms
more to chase from `_OUT_FREQ_CACHE.zero_()` aliases between layers. Until
the per-layer latency floor stabilises, the 47 → 15 gap stays.

---

## 7. R4 FIX — `Parameter.detach()` is the silent cache-killer

In `FourierLayer.forward_supa` we had:

```python
y = spectral_conv2d_supa(
    x,
    self.spectral_conv.weights1.detach(),  # ← creates a fresh `id`
    self.spectral_conv.weights2.detach(),
    ...,
)
```

`nn.Parameter.detach()` returns a new view with **different `id()` and
`isinstance(..., nn.Parameter) == False`** (verified via `id()` and
`isinstance` checks). `_weights_to_supa_cached` keys on `isinstance(weights,
nn.Parameter)`; with `.detach()` the id-keyed O(1) lookup misses, falling
back to:

```python
w_cpu = weights.detach().to("cpu", torch.complex64).contiguous()
digest = hashlib.blake2b(w_cpu.numpy().tobytes(), digest_size=16).hexdigest()
```

— a full D2H + numpy round-trip + blake2b hash on **every** call. 2 weights
× 4 layers = **8 hash round-trips per forward pass**. On Biren each round
costs ~1 ms; total ~8 ms/layer.

**The fix**: pass `self.spectral_conv.weights1, self.spectral_conv.weights2`
raw. The `Parameter` object's `id()` stays stable, the id-keyed branch of
`_weights_to_supa_cached` returns in O(1), and `fno_ns/profile_chain.py`
hits its measured ~3.4 ms/layer latency.

**Measured impact** (64×64, B=4, width=32, modes=16, 4 layers):
- Before R4: 49.118 ms chain (median)
- After R4:  **16.092 ms chain (median)** — 3.05× speedup, **−67%**.
- After R4 single-layer: ~3.5 ms (was 11.7 ms).

**General lesson**: any cache that keys on Python `id()` or
`isinstance(..., nn.Parameter)` must not receive `.detach()`-stripped inputs.
When writing forward paths, prefer raw parameters / buffers as inputs to
**all** caches, even when the consumer wants to be "safe by detaching".

---

## 8. R4 hot-path cleanups (small wins, < 2 ms total)

Three smaller cleanups applied alongside R4-1 (each worth < 1 ms but
compounding on the chain):

- **R4-2** `forward_supa_chain` no longer calls `self.prepare_supa_eval()`
  internally. It's now caller responsibility (already documented). `self.to
  ("supa")` re-traverses every parameter each call — wasted work in the hot
  path.
- **R4-3a** `spectral_conv2d_fused` dropped the `try/except
  (RuntimeError, TypeError)` around `spectral_mul_out`. The fallback was
  dead code (the .so is shipped together with f's Python). Removing it
  saves `SETUP_EXCEPT` / `POP_BLOCK` opcodes per corner.
- **R4-3b** `_y_freq_buffer` cache-hit no longer calls `buf.zero_()`. The
  `spectral_mul_out` kernel writes every element of the `(B, Cout, M1, M2,
  2)` output, so the previous contents are fully overwritten.

Cumulative effect (each on its own, pre-R4-1):
- R4-2 alone: -0.3 ms / chain.
- R4-3a alone: -0.05 ms / layer.
- R4-3b alone: -0.15 ms / layer.

In isolation they're 0.5 ms chain. Combined with R4-1 they take the chain
from ~17 ms (R4-1 alone) down to ~16 ms (R4 stack together).

---

## 9. Status after R4

| 项 | R3 | R4 |
|---|---|---|
| f FNO chain @ 64 | 49.12 ms | **16.09 ms** |
| n FNO chain @ 64 | 15.46 ms | 15.45 ms |
| n − f | -33.7 ms | **-0.6 ms (4% flapping)** |
| SpectralConv 64/128/256 | 5.34 / 13.80 / 53.01 | 5.33 / 13.69 / 52.75 |

After R4, the FNO chain is **between f and n within flapping noise**. The
remaining wins are single-digit milliseconds at best:
- R5-1: drop the redundant permute in fused rFFT (-2-3 ms).
- R5-2: shared plan workspace across all plans (-1 ms, lower memory too).
- R5-3: cache-friendly contiguous slices for `next layer input` (-1 ms).

None of these is needed for the official-time metrics to land; we ship R4.

---

## 10. R5 outcomes (post-R4)

R5 added 4 things — and rejected 2 — over the R4 baseline.

**R5-0 (mandatory correctness fix):** the `forward_supa_chain(..., use_gn_substitute=True)`
flag was double-norming under the hood (`forward_supa` already returns
post-IN+GELU; `use_gn_substitute=True` re-applied a fresh `GroupNorm(1,C)+GELU`).
Numbers were still correct-looking (L2 ≈ 0.009516 because the second norm
merely rescales), but the flag was semantically wrong and was hiding the
true chain perf number. R5 removed the kwarg; L2 unchanged at 0.009516;
chain perf median unchanged at 16 ms (R4 was already single-norm, R5
de-facto single-norm).

**R5-1 (decision infrastructure):** permanent
`fno_ns/bench_f_fno_chain_layer_profile.py` (formerly `/tmp` harness) so
future rounds can debug layer-by-layer without re-deriving the test rig.

**R5-3 (small win):** added `spectral_conv_ext.spectral_mul_dual_out`:
one pybind entry that runs both `spectral_mul` launches. Saves ~0.04 ms
per fused call. Combined with R4 the dual-call pybind overhead is gone
and the chain measures 16.078 ms (vs R4 16.092 ms — within noise).

**R5-4 (failed attempt):** `spectral_mul_dual_scatter_out` writes
strided corners into `out_freq` directly. Reason it failed: on SUPA
PyTorch, `out_freq.narrow(2, 0, M1).contiguous()` returns a *new* SUPA
tensor; in-place writes to that tensor do NOT propagate back into
`out_freq`. The whole fused path returned rel=1.0 until I rolled back.
Lesson: any SUPA-side tensor-API shortcut that touches strided /
narrowed views must verify the write-back semantics under PyTorch's SUPA
backend before relying on it.

**R5-2 (aborted):** resume training for +30 epochs at `lr=2e-4`; test L2
went 0.0095 → 0.0128 → 0.0135 within 10 epochs. The checkpoint was
already at the cosine-annealing plateau and a fresh Adam at `2e-4` pushed
it off. Killed; no best saved. The reusable piece is
`fno_ns/resume_train.py` (with cosine scheduler slot; needs proper LR
schedule to be useful next time).

**R5-5 (1-min SPIKE — stop):** `nm -D libsufft.so.0.7.0` exports only
`BuildPlan1d / ExecR2C / ExecC2C / ExecC2R / SetStream / SetWorkArea`.
There is **no `BuildPlanMany` / `BuildPlan2d` / stride parameter API**.
Conclusion: the "skip permute in fused rFFT" idea cannot be implemented
without writing a custom SUPA `.su` kernel that re-implements FFT.
R5-A1 abandoned; no kernel-level fusion pursued.

### R5 status

| 项 | R4 | R5 |
|---|---|---|
| f FNO chain @ 64 | 16.092 ms | 16.078 ms |
| n FNO chain @ 64 | 15.452 ms | 15.453 ms |
| f − n | +0.64 ms | +0.62 ms (4% noise) |
| spectral 64/128/256 | 5.33 / 13.69 / 52.75 | 5.32 / 13.70 / 52.62 |
| FNO L2 | 0.009516 | 0.009516 (R5-2 cancelled before any improvement landed) |
| use_gn_substitute 路径 | 双 norm (错) | 单 norm (正) |

### R5 take-aways

1. **R5 was mostly a no-op for perf** because R4 + R3 already nailed the
   cache / cache-hit / dispatch shape. The bottleneck is now inside the
   `spectral_mul_kernel` itself (~2.85 ms / layer) and the slow fused path
   is the next frontier — but every deep win is gated by either the SDK
   ABI (no stride), a new kernel (off-table for this round), or a new
   metric (resume L2 requires proper LR scheduling).
2. **An important bug was found and fixed**: double-norming under
   `use_gn_substitute=True`. The fix is partly a correctness improvement
   (the second norm rescales outputs non-trivially even when the test
   L2 looks "still good"), partly a perf honesty fix (true chain perf
   number was being computed under a regime that was technically a
   different model).
3. **The .cursor rule `parameter-cache.mdc`** now formally forbids
   `.detach()`-before-cache. The R4 lesson is now structural instead of
   tribal.
