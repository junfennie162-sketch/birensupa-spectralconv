---
name: spectral-conv-fno-ns-biren
description: >
  2D Spectral Convolution (SUPA, pruned-DFT hot path) and a four-layer FNO-NS
  on one BIREN GPU. Reproduce operator accuracy/perf, public NS64 relative L2,
  and the OPT-loop dry-run.
version: 1.0.0
team: FanDou Garden · North University of China · Track 5 · Biren Flying Cup
hardware: BIREN GPU · SDK 1.11.0.0.rc2 · device=supa
---

# Skill: SpectralConv + FNO-NS (FanDou Garden)

**Please Star the repo first.** If this write-up helps you ship an operator on a domestic GPU, train a neural operator, or run an Agent-backed performance loop, a Star makes it findable:

https://github.com/junfennie162-sketch/birensupa-spectralconv

(Team mirror: https://github.com/Aafff623/fandou-ai4s)

This is **one** Skill. Operator work, FNO experiments, and the optimization loop live in this file. Read top to bottom.

---

## 1. What this Skill is for

On one **BIREN** card, implement FNO's core 2D spectral convolution with **SUPA / `torch.utils.cpp_extension`**, assemble at least four FNO layers, run vorticity forward on the **unmodified official Navier–Stokes public set**, and report relative L2.

| | |
|--|--|
| **Purpose** | Let a judge or a later engineer reproduce: how to build, how to match the official reference, which numbers are formal, and which numbers may enter the scoreboard after a code change. |
| **Value** | Heterogeneous / scientific kernels: the official reference may run on CPU; the submission must run on `supa`. Neural operators: how we split the public set, residual head, and spectral loss. Agent-backed contests: freeze the performance table; isolate probes. |
| **Who** | Track 5 judges; people writing SUPA operators; people training FNO; captains who keep an OPT loop honest. |

Formal board (`results/summary.json`):

| 项 | 值 |
|----|----|
| SpectralConv 前向 64 / 128 / 256 | **0.764 / 1.827 / 6.504 ms**（`pipe_b_r1` · v12） |
| 正确性最差相对误差 | 默认裁剪路径 **7.162×10⁻⁶**（门槛 1×10⁻⁴） |
| FNO 公开 NS64 相对 L2 | **0.035012**（刚接上官方集时是 0.041835） |

This is not a renamed official PyTorch reference, and it is not a pretty L2 on a home-grown vorticity field.

The **default hot path** in this tree is a pruned DFT (only the kept dual-corner modes). Reported CPU-in timings: **0.961 / 2.207 / 7.870 ms**. Previous suFFT idle: 3.797 / 8.037 / 29.295 ms.

---

## 2. Project approach

1. **Make the mandatory operator correct first.** The handbook gives a CPU/CUDA reference, not a ready SUPA kernel. Acceptance: relative error vs that reference ≤ `1e-4`.
2. **Small resolutions lose to copies.** Early FFT round-trips to Host made the multiply cheaper than the memcpy. Keep the spectrum on device: suFFT R2C → custom SUPA dual-corner mul → suFFT C2R, then a pruned DFT that never transforms the dropped bins.
3. **FNO only assembles.** All four Fourier layers call the same `spectral_conv2d_supa`. Accuracy work stays on the official `.pt`: residual head, periodic shifts, spectral-weight updates, H⁻¹ loss on high frequencies.
4. **Close every round with rules.** Read the formal board, then probe. Promote only on a real gain. Write formal ms only on an idle card. SOL / autotune / synthetic sets are side notes.

---

## 3. What we actually learned

Docs teach APIs. Judgment comes from the failures.

- The reference may be CPU; the submission must be SUPA. The official perf script hard-codes `cuda`. A machine without NVIDIA is not “SUPA is wrong”.
- Leave headroom on accuracy (we sit at `2×10⁻⁷`) so fusion, truncation, and caches have room to miss.
- A “slow” 64×64 is usually Host↔Device and C2R, not a weak multiply.
- Formal score only counts the official filename and split. A synthetic set, however pretty, does not enter the board.
- With an Agent: a run that finishes is not a run that may be promoted. Gate it in a script; do not remember numbers from chat.
- One card. Two jobs at once → ErrorCode 719 → milliseconds are void.

---

## 4. Operator development

### 4.1 Background

Spectral convolution is the heavy block in one FNO layer: FFT the spatial field, keep the low-frequency dual corner `modes1 × modes2`, complex-multiply in frequency, iFFT back. Sources: `spectral_conv/spectral_conv_ext.su`, `.cpp`, `pruned_*.su`, `spectral_conv_ops.py`. Build: `cd spectral_conv && ./build.sh`.

Every new shell:

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
```

Device name is `supa`. `torch.cuda.is_available()` being false is normal.

### 4.2 What we squeezed on the official GPU

| Idea | What we did |
|------|-------------|
| Device-resident fused path | suFFT R2C → custom dual-corner mul → suFFT C2R; no full-spectrum Host round-trip |
| Pruned DFT (default hot path) | Mixed-radix rFFT / height DFT / inverse only on kept modes; suFFT remains the fallback |
| Multiply only kept modes | Same dual-corner truncation as the official reference |
| Caches | Reuse weights, frequency buffers, Host staging; fewer H2D and mallocs |
| CPU-in output cache | `_SPATIAL_OUT_CACHE` + packed irfft for `to_cpu=True` |
| Resolution routing | Eval sizes 64/128/256 on the fused/pruned path; `use_sufft="auto"` |
| Cache keys include corners | Two corners, two buffers, so the second write cannot clobber the first |
| Weights keep Parameter identity | Do not `detach()` before the call or the cache always misses |
| Idle card before formal ms | A busy GPU does not get to rewrite the scoreboard |

Versus the official CPU reference run on this machine: about **19.5× / 11.1× / 10.1×** on the **formal idle** board. Backward, 3D four-corner, and irregular shapes are bonus; they do not change the mandatory protocol. Autotune only picks the `auto` route; **the tune median is not a scoring sentence**.

We do not: run `torch.fft` directly on `supa`; call a 2D plan the headers do not export; pretend an in-house bandwidth proxy is the official split.

### 4.3 Techniques, lessons, problems

Used / learned: `.su` kernels bound through a PyTorch extension; suFFT's usable surface is 1D plans, so 2D is 1D plus permute; correctness is always the same dual-corner PyTorch reference.

| Problem | What we did |
|---------|-------------|
| Official perf script says no CUDA | That script times the reference; our operator uses `supa` |
| SUDNN ErrorCode 6, even 1×1 conv | Treat as driver/library; stop and log; do not randomly edit the operator |
| Relative error jumps to ~1 | `torch.cat` on SUPA fed straight into a custom kernel; or two corners sharing one buffer |
| `torch.fft` on `supa` | Produces numbers; relative error ~`5×10⁻³`; fails `1e-4` |
| Small fused graphs slower than CPU FFT | Missing warmup, Parameter cache, oversized threshold; then drop the fused cutoff to ≥64 |
| Returning a cached spatial buffer into FNO | Next GELU / add / InstanceNorm aliases the storage; skip-roundtrip FNO is a No-Go |
| Pinned H2D / dual-n1 irfft | Slower or wrong on this card; reverted |

```bash
cd spectral_conv
./build.sh
python3 test_accuracy.py
python3 probe_pruned_continue.py   # unofficial; does not write formal idle
# python3 test_perf.py             # idle exclusive GPU only
```

---

## 5. FNO experiments

### 5.1 Background

Four Fourier layers, width=32, modes=16, 64×64. Task: first 10 frames predict the next frame. Data must be the official file `navier_stokes_v1e-3_N1200_T20.pt`, train 1000 / test 128, seed `20260722`. The pack does not include that ~376 MB file. Synthetic sets are engineering side notes only.

Inference calls the same spectral convolution. Weights: `fno_ns/checkpoints/fno_ns_public_demo.pt`.

### 5.2 What we squeezed

| Idea | What we did |
|------|-------------|
| Reuse the mandatory operator | Advanced score only counts if the operator actually sits in the model |
| Device-resident chain | Four layers with `to_cpu=False`, one D2H at the end |
| Warmup | `prepare_supa_eval()`: FFT plans; InstanceNorm running stats must be moved to `supa` explicitly |
| Residual head | Predict the increment vs the last input frame |
| Do not edit the official `.pt` | Periodic shifts, spectral-weighted loss, late-stage spectral-weight updates, H⁻¹ on high frequencies |
| Demo matches the score | Forward on the official test set; pick the sample whose L2 is closest to 0.035012 |

Same official data: relative L2 **0.041835 → 0.035012**. Long training is CPU, `use_supa=False`. Do not bind SUPA mul into every epoch.

### 5.3 Techniques, lessons, problems

Used / learned: a neural operator swaps the layer's spectral conv; it is not “a CNN on a domestic card”. Public-set reports need filename, split, and seed. Cover figures must not be relative-error heatmaps.

| Problem | What we did |
|---------|-------------|
| Synthetic L2 looks great | Never write it as the formal score |
| Default forward script may use a generated cache | Official demo is `render_official_demo.py` + official `.pt` |
| InstanceNorm wrong after `model.to("supa")` | `running_mean` / `running_var` do not follow |
| Training suddenly crawls | Every layer bound to SUPA mul; Host↔Device storm |

```bash
cd fno_ns
python3 render_official_demo.py   # needs official .pt
python3 visualize.py
```

---

## 6. Optimization loop

### 6.1 Background

There is one card and a pile of required artifacts. If an Agent trains and re-runs `test_perf.py` in the same breath, the formal board fills with noise. The loop does not invent kernels; it says **when code may change** and **which numbers may enter the board**. Default is dry-run: no retrain, no formal perf write.

### 6.2 The discipline is part of protecting GPU numbers

| Step | Name | Must |
|------|------|------|
| P0 | Env and one card | Source the SDK; never two GPU jobs |
| P1 | Read the board | Only public L2 and idle ms in `summary.json` |
| P2 | Accuracy probe | Background; stop on gate; do not hang the chat for hours |
| P3 | Promote? | Switch official weights and demo figures only if better |
| P4 | Guardrail | Accuracy first; do not default-rerun `test_perf.py` |
| P5 | Materials | Checklist, Agent log, one current eval report |
| P6 | Pack | Pack after a real promote |

SOL, tune, and synthetic sets may be side notes. They are not scoring sentences.

```bash
python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
```

### 6.3 Techniques, lessons, problems

Used / learned: encode the process so a miss `exit 1`s; an accuracy line that stops only allows a new mechanism, not the same schedule again.

| Problem | What we did |
|---------|-------------|
| Re-run perf after every edit | Contention overwrites frozen idle ms |
| Sidecar / synthetic set in the demo | Judges cannot match public NS64 |
| Training waited on in the chat | Session dies and the card stays busy; background + short poll |

---

## 7. How to run it (for reproduction)

Input: `x: [B, C_in, H, W]` with `modes1/modes2`; FNO takes multi-frame vorticity `[B, T_in, H, W]`.

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2

bash scripts/validate.sh
```

Outputs land in `results/summary.json`, `results/run_logs/`, and the flow-field figures. Agent audit page: `AGENT_OFFICIAL.md`.

Capability boundary: default hot path is pruned DFT; suFFT fused is the frozen formal idle board; v1 (CPU FFT) is a differentiable training / contrast path. Do not share the GPU with another job.

---

If this helped, please Star: https://github.com/junfennie162-sketch/birensupa-spectralconv
