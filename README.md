# FanDou Garden · Spectral Convolution + FNO-NS

<p align="left">
  <b>Shusheng Guozhi Science Challenge</b> · Biren Flying Cup · Track 5 (Models &amp; Operators)<br/>
  Team: <b>FanDou Garden</b> · North University of China<br/>
  Submission repo: <a href="https://github.com/junfennie162-sketch/birensupa-spectralconv">junfennie162-sketch/birensupa-spectralconv</a><br/>
  Team mirror: <a href="https://github.com/Aafff623/fandou-ai4s">Aafff623/fandou-ai4s</a>
</p>

If this tree helps you write operators on a domestic GPU, train a neural operator, or run an Agent-backed performance loop, please **Star** the repo so the next person can find it:

https://github.com/junfennie162-sketch/birensupa-spectralconv

On one **Biren106B** card we implement FNO's core 2D **Spectral Convolution** with **SUPA + a PyTorch extension** (contest Route 2), then assemble **four** Fourier layers for public-set vorticity prediction on 2D incompressible Navier–Stokes. Cursor / Biren Agent logs are part of the submission.

---

## Reported scores

Numbers below are the **formal** board in [`results/summary.json`](results/summary.json). Lower L2 and lower ms are better.

| Item | Formal value |
|------|----------------|
| Public NS64 relative L2 (10 → 1) | **0.035012** · tag `spec_ref_r2` · **v10** |
| Spectral idle 64 / 128 / 256 | **3.797 / 8.037 / 29.295 ms** (2026-08-14 idle recheck) |
| Spectral worst relative error | **2.170×10⁻⁷** (threshold `1e-4`) |
| Checkpoint | `fno_ns/checkpoints/fno_ns_public_demo.pt` |
| Phase | `submit_gate` done |

**Default hot path** in this tree is a **pruned DFT** (transform only the kept dual-corner modes). That path is **not** the formal idle row. Sample unofficial CPU-in timings under the official protocol (warmup=10, iters=100): about **0.96 / 2.21 / 7.87 ms**. They are **not promoted**. Reproduce them with `bash scripts/validate.sh`; do **not** run `test_perf.py` unless the GPU is idle and you intend to rewrite the formal table.

---

## Visuals (official public NS64, same checkpoint)

These two figures are rendered from the official test split with `fno_ns/render_official_demo.py`. They are not sketches and not synthetic data.

| Typical sample (L2 closest to 0.035012) | Best / typical / worst |
|---|---|
| [![Typical sample: last input, ground truth, prediction](demo/media/01_typical_sample_pred_vs_gt.png)](demo/media/01_typical_sample_pred_vs_gt.png) | [![Best, typical, and worst test samples](demo/media/02_best_typical_worst.png)](demo/media/02_best_typical_worst.png) |

---

## Quick validation on BIREN

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
bash scripts/validate.sh
```

That command sources the SDK, builds the extension, runs the official 3-case accuracy test, times the unofficial pruned path, and dry-runs the OPT-loop gates. It **does not** call `test_perf.py`.

Public-set FNO evaluation needs the official tensor (not shipped; ~376 MB):

```bash
# place navier_stokes_v1e-3_N1200_T20.pt under fno_ns/data/
cd fno_ns && python3 render_official_demo.py
```

Downloadable pack (English only): [`contest_submit/`](contest_submit/).

---

## Repository layout

We keep contest names `spectral_conv/` and `fno_ns/` rather than renaming the tree to `project/`. Clone root **is** the submission root.

| Path | What it is |
|------|------------|
| [`skill.md`](skill.md) | One-file Skill (start here for the method) |
| [`AGENT_OFFICIAL.md`](AGENT_OFFICIAL.md) | Agent audit page (contest scoring item) |
| [`development_log.md`](development_log.md) | Full Agent log (original Chinese; English banner at top) |
| [`results.md`](results.md) | Same-protocol comparison + what we changed |
| `spectral_conv/` | Mandatory operator: pruned-DFT kernels, suFFT fallback, tests |
| `fno_ns/` | Four-layer FNO-NS; reuses the same operator |
| `scripts/validate.sh` | One-command BIREN check |
| `demo/media/` | Cover figures + `brsmi` snapshot |
| `results/summary.json` | Frozen formal numbers |
| `contest_submit/` | Single English tarball |

---

## Environment

| Item | Value |
|------|--------|
| GPU | Biren106B, **one card** |
| SDK | `1.11.0.0.rc2` |
| Device name | `device="supa"` (import `torch_br` first) |
| `torch.cuda.is_available()` | `False` is expected on this platform |

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
brsmi   # confirm the card is free before any GPU job
```

Do not run two SUPA jobs at once (ErrorCode 719; timings become junk).

---

## Operator (mandatory Spectral Convolution)

Input `x: [B, C_in, H, W]` with configurable `modes1/modes2`. Default path:

```text
width mixed-radix rFFT (kept bins only)
  → height dual-corner DFT (top/bottom modes1)
  → SUPA complex multiply (same dual-corner truncation as the official reference)
  → inverse height + inverse width pruned iFFT
→ y: [B, C_out, H, W]
```

`SPECTRAL_PRUNED_FFT=0 SPECTRAL_PRUNED_INV=0` falls back to device-resident suFFT R2C → SUPA mul → C2R (the frozen formal idle board).

Correctness is always against `spectral_conv/reference_pytorch.py` (CPU dual-corner reference). Bonus tests: backward, 3D four-corner, irregular shapes. They do not change the mandatory protocol.

Build: `cd spectral_conv && ./build.sh`.

---

## FNO-NS (advanced)

Four Fourier layers, `width=32`, `modes=16`, 64×64. Task: **10 input frames → frame 11** on the **unmodified** official file `navier_stokes_v1e-3_N1200_T20.pt` (train 1000 / test 128, seed `20260722`). Inference calls the same SpectralConv; we do not ship a second FFT.

On that official split: **0.041835 → 0.035012**. Residual head, periodic shifts, spectral-weight fine-tune, Sobolev H⁻¹ high-frequency loss. Long training stays on CPU (`use_supa=False`).

---

## Agent / Skill

Contest scoring includes Agent development (~15%). Open [`AGENT_OFFICIAL.md`](AGENT_OFFICIAL.md) first, then [`skill.md`](skill.md). The OPT loop is `python3 skills/operator_opt_loop/run_loop.py --dry-run --strict`: probes may run; formal idle stays frozen until an explicit idle recheck.

---

## Limits

- Formal score is public NS64 **single-step** L2, not a 10-step rollout and not a synthetic vorticity set.
- Unofficial pruned milliseconds are labeled unofficial until a promote.
- No multi-GPU / distributed path.
- Official NS `.pt` is not in git (GitHub 100 MB file limit).
