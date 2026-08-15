# Agent / Skill log (official audit page)

> **Required contest item** (Agent development ≈ 15% of the track score; missing or thin logs fail).  
> Written to the handbook template so a judge can sample it directly. Full trail: [`development_log.md`](development_log.md) (**75+** numbered entries; original Chinese).  
> Tool: Cursor Agent (SSH · Biren contest Docker · SDK `1.11.0.0.rc2`).  
> Formal board: public NS64 L2 **0.035012** (`spec_ref_r2` · v10); Spectral **0.961 / 2.207 / 7.870 ms** (pruned DFT CPU-in KEEP, 2026-08-15).

## Scene coverage (≥3 types · this page covers all 6)

| # | Official scene | This page | Evidence |
|---|----------------|-----------|----------|
| 1 | Operator kernel design / debug / opt | Record A | fused suFFT + SUPA mul; later pruned DFT |
| 2 | Perf bottleneck analysis | Record B | Host↔Device; idle freeze |
| 3 | Architecture / hyper-parameters | Records C, E | gate discipline; Spectral-Refiner → v10 |
| 4 | Data / features | Record D | public NS64 1000/128 vs synthetic v2 |
| 5 | Analysis and visualization | Record F | Autopsy figures; Pred/GT |
| 6 | BIREN platform | Records B, env | `device=supa`; one card; `brsmi` |

Run snapshot: [`demo/media/brsmi_snapshot.txt`](demo/media/brsmi_snapshot.txt). Accuracy / perf recheck: `results/run_logs/`.

---

## Agent record A · operator kernel (fused SpectralConv)

- Tool / Agent: Cursor Agent (SSH)
- Scene: operator kernel design / debug / optimization
- Goal: move 2D Spectral Convolution from “CPU FFT + full-spectrum Host multiply” to a device-resident fused path, keep relative error ≤ `1e-4`
- Prompt / intent: iterate the deep-opt plan; serialize the GPU; measure after every edit. Core compute must be SUPA / extension; no vanilla PyTorch as the submission
- Agent advice: 64×64 profile showed `bridge_mul` ≈ 26 ms, far above rfft. Chain suFFT R2C → SUPA mul → C2R on device. Keep v1 CPU-FFT for contrast and differentiable training
- Adopted: `spectral_conv2d_fused` (`spectral_conv_ext.su` / `.cpp` / `spectral_conv_ops.py`); dual-corner reference; FNO layers share the API
- Verified: worst rel ≈ **2.17×10⁻⁷** ≪ `1e-4`; 2026-08-14 idle **3.797 / 8.037 / 29.295 ms** @64/128/256
- Rejected: `torch.fft` on `device=supa` (misses `1e-4`); `torch.cat` on SUPA fed straight into a custom kernel (known memory smash)

---

## Agent record B · bottleneck (Host↔Device / C2R wall)

- Tool / Agent: Cursor Agent (SSH)
- Scene: perf bottleneck · BIREN platform
- Goal: explain why small sizes were slow; set fused threshold and cache policy
- Prompt / intent: keep cutting operator time; ask why 64/128 felt slower than intuition
- Agent advice: repeated H2D/D2H and per-step frequency-buffer alloc, not missing FMA. Use `_OUT_FREQ_CACHE` / `_HOST_OUT_CACHE`. Fused cutoff 256→128→64 must be a sweep, not a guess
- Adopted: `profile_segments.py`; buffer reuse; `test_perf.py` contention guard (do not write formal if 64 ms > 12); no concurrent job with the partner workspace
- Verified: vs official CPU reference ~**19.5× / 11.1× / 10.0×**; formal board frozen after the 2026-07-31 idle recheck, confirmed 2026-08-14. Forcing 128 fused at 40 ms > v1 22 ms was **rolled back immediately**
- Rejected: Plan2d / `torch.fft@SUPA` / half-precision weights (SDK gap or breaks `1e-4`); daily default rerun of `test_perf.py`

---

## Agent record C · hyper-parameters and accuracy gate (freeze → dualview)

- Tool / Agent: Cursor Agent (SSH)
- Scene: architecture / hyper-parameter search
- Goal: cut public NS64 (1000/128, 10→1) relative L2; **no promote if the gate does not break**
- Prompt / intent: “keep pushing accuracy / promote”; do not write a failing sidecar into the official demo
- Agent advice: gate = previous formal L2 − `1e-4`; probe with `nohup` + `--stop-on-gate`; do not deepen the same schedule; modes=20 / width=48 already KILL
- Adopted: `promote_guard.py`; v8 `freeze_r9` (0.035302); v9 `dualview_r2` (0.035115, last layer unfrozen + dual-view consistency)
- Verified: v9 independent reeval **0.035114976112**; an autochain once wrote 0.035252 into demo **without** breaking gate — rolled back to v8 and the guard became hard
- Rejected: lowering the gate to force a promote; soup / PF near-miss (0.035216) as a formal version

---

## Agent record D · data protocol (public NS64 vs synthetic v2)

- Tool / Agent: Cursor Agent (SSH)
- Scene: data preprocessing / features
- Goal: attach public NS64 to the loader; stop a synthetic 0.005 L2 from being reported as the public score
- Prompt / intent: handbook “state the data source”; submission must be the public set
- Agent advice: `dataset.py` prefers non-`ns_like` `.pt`; lock 1000/128, seed 20260722; split columns in `data_disclosure.md`
- Adopted: `navier_stokes_v1e-3_N1200_T20.pt`; synthetic v2 continue3 demoted to side-note ckpt `fno_ns_demo.pt`
- Verified: public set 0.041835 (v1) → **0.035012** (v10); synthetic 0.005144 **never** entered `summary.fno_ns.public_ns64`
- Rejected: downloading HDF5 in the eval container (offline reproducibility first)

---

## Agent record E · Spectral-Refiner → v10 (current formal L2)

- Tool / Agent: Cursor Agent (SSH)
- Scene: architecture / hyper-parameters · operator (spectral weights only)
- Goal: on the v9 plateau, find a **new mechanism** that breaks gate 0.035015, not more epochs of the same run
- Prompt / intent: try a literature mechanism; after a pass, tidy submit / skill
- Agent advice: H1 spatial-gradient loss as contrast; main path Spectral-Refiner lite — freeze non-spectral weights; mix rel-L2 with frequency H⁻¹ weights `(α+|k|²)⁻¹`
- Adopted: `train_public_spectral_refiner_probe.py`; `spec_ref_r1` to 0.035027; wave4 `spec_ref_r2` epoch 7 `stop_on_gate`; `promote_public_ckpt.py --tag spec_ref_r2`
- Verified: clean reeval **0.035011906177** < gate **0.035014976**; vs v9 **+0.29%**; H1 and soup did not beat the single model
- Rejected: auto-promote (human confirm first); pushing a 2.9 GB tar to GitHub; unfreezing Spectral formal ms

---

## Agent record F · visualization and error autopsy

- Tool / Agent: Cursor Agent (SSH)
- Scene: analysis and visualization
- Goal: Pred/GT/error figures a judge can falsify, not a lone L2
- Prompt / intent: 3-minute judge path; Offline Error Autopsy with `epochs=0` (read-only)
- Agent advice: shared colormap Pred/GT; best/median/worst strip; use correlation of e1 vs time increment `q_t` to pick the next mechanism
- Adopted: `visualize.py`; autopsy figures into `demo/media/`
- Verified: ρ(e1, `q_t`)≈0.80 → later Δ-match / `q_t` oversample / Refiner; spectrum did not force a modes bump (stay at 16)
- Rejected: putting autopsy near-miss numbers into the eval-report version chain; video (handbook optional)

---

## How to check this page

```bash
grep -c '^## Agent' development_log.md
grep -E 'Scene:|scene:' AGENT_OFFICIAL.md || grep -E 'Scene' AGENT_OFFICIAL.md
cat demo/media/brsmi_snapshot.txt
```

Skill: [`skill.md`](skill.md). Full field-complete log: [`development_log.md`](development_log.md).
