#!/usr/bin/env python3
"""SOL-ExecBench-inspired performance harness for SpectralConv on SUPA.

Reference methodology (adapted to Biren, not NVIDIA B200/SOLAR):
  https://github.com/nvidia/sol-execbench
  arXiv:2603.19173

What we adopt:
  - correctness before timing
  - warmup=10, iters=50, trials=3 (SOL defaults), report mean latency
  - synchronize around the timed region
  - clone / reshuffle inputs each timed iter (anti state-cache)
  - score relative to a fixed scoring baseline (team v1 path), not only absolute ms
  - report gap to a fixed "reference narrative" latency (handbook PyTorch-on-SUPA)

What we cannot port 1:1:
  - SOLAR analytical SOL bound on B200
  - nvidia-smi clock lock / CUDA Event / 256MB L2 clear
  - Full SOL-Score = f(baseline, SOL_bound); we report a proxy gap score instead
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

from reference_pytorch import make_random_weights, spectral_conv2d
from spectral_conv_ops import spectral_conv2d_supa

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"

# Official contest shapes (§3.2)
PERF_RESOLUTIONS = [(64, 64), (128, 128), (256, 256)]
PERF_B, PERF_C_IN, PERF_C_OUT = 4, 32, 64
PERF_MODES1, PERF_MODES2 = 16, 16

# SOL-ExecBench-like timing config (README defaults)
WARMUP_RUNS = 10
TIMED_ITERS = 50
TRIALS = 3
SEED = 200
REL_TOL = 1.0e-4

# Fixed narrative reference (handbook / early contest PyTorch-on-SUPA numbers)
REF_NARRATIVE_MS = {
    "64x64": 5.3,
    "128x128": 8.7,
    "256x256": 27.7,
}


def _rel_err(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    diff = torch.linalg.norm((prediction - reference).reshape(-1))
    ref = torch.linalg.norm(reference.reshape(-1)).clamp_min(1.0e-12)
    return float((diff / ref).item())


def make_input(height: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.empty(
        (PERF_B, PERF_C_IN, height, width),
        dtype=torch.float32,
        device="cpu",
    )
    x.uniform_(-0.5, 0.5, generator=generator)
    return x.contiguous()


def check_correctness(height: int, width: int) -> dict:
    x = make_input(height, width, SEED + height)
    w1 = make_random_weights(
        PERF_C_IN, PERF_C_OUT, PERF_MODES1, PERF_MODES2, SEED + width
    )
    ref = spectral_conv2d(x, w1, PERF_MODES1, PERF_MODES2)
    pred = spectral_conv2d_supa(
        x, w1, PERF_MODES1, PERF_MODES2, use_sufft=True
    )
    rel = _rel_err(pred, ref)
    return {
        "resolution": f"{height}x{width}",
        "rel_error": rel,
        "ok": rel <= REL_TOL,
    }


def _time_path(use_sufft, height: int, width: int, trial_index: int) -> float:
    """One trial: warmup + timed iters with fresh clones each iter.

    use_sufft: True / False / \"auto\"
    """
    base_x = make_input(height, width, SEED + height + 17 * trial_index)
    weights = make_random_weights(
        PERF_C_IN,
        PERF_C_OUT,
        PERF_MODES1,
        PERF_MODES2,
        SEED + width + 31 * trial_index,
    )

    for step in range(WARMUP_RUNS):
        x = base_x.clone()
        x.add_(0.001 * (step + 1))
        _ = spectral_conv2d_supa(
            x, weights, PERF_MODES1, PERF_MODES2, use_sufft=use_sufft
        )
    torch.supa.synchronize()

    start = time.perf_counter()
    for step in range(TIMED_ITERS):
        x = base_x.clone()
        x.add_(0.001 * (step + 1))
        _ = spectral_conv2d_supa(
            x, weights, PERF_MODES1, PERF_MODES2, use_sufft=use_sufft
        )
    torch.supa.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / TIMED_ITERS * 1000.0


def gap_score(candidate_ms: float, baseline_ms: float, ref_ms: float) -> float:
    """Proxy for SOL-Score without hardware SOL bound.

    Interprets ``ref_ms`` as a stand-in lower target (handbook narrative).
    0.5 ≈ match scoring baseline (v1); 1.0 ≈ match ref target.
    Clamped to [0, 1].
    """
    if baseline_ms <= ref_ms:
        # baseline already at/below ref; score by how close candidate is to ref
        if candidate_ms <= ref_ms:
            return 1.0
        return max(0.0, min(1.0, ref_ms / candidate_ms))
    # linear close of gap from baseline -> ref
    closed = (baseline_ms - candidate_ms) / (baseline_ms - ref_ms)
    return float(max(0.0, min(1.0, 0.5 + 0.5 * closed)))


def benchmark_resolution(height: int, width: int) -> dict:
    key = f"{height}x{width}"
    corr = check_correctness(height, width)
    if not corr["ok"]:
        raise RuntimeError(f"correctness failed before timing: {corr}")

    v1_trials = [_time_path(False, height, width, t) for t in range(TRIALS)]
    fused_trials = [_time_path(True, height, width, t) for t in range(TRIALS)]
    auto_trials = [_time_path("auto", height, width, t) for t in range(TRIALS)]
    v1_ms = sorted(v1_trials)[len(v1_trials) // 2]
    fused_ms = sorted(fused_trials)[len(fused_trials) // 2]
    auto_ms = sorted(auto_trials)[len(auto_trials) // 2]
    # also keep means for transparency
    v1_mean = sum(v1_trials) / len(v1_trials)
    fused_mean = sum(fused_trials) / len(fused_trials)
    auto_mean = sum(auto_trials) / len(auto_trials)
    ref_ms = REF_NARRATIVE_MS[key]

    torch.supa.reset_peak_memory_stats()
    _ = spectral_conv2d_supa(
        make_input(height, width, SEED),
        make_random_weights(
            PERF_C_IN, PERF_C_OUT, PERF_MODES1, PERF_MODES2, SEED + 1
        ),
        PERF_MODES1,
        PERF_MODES2,
        use_sufft="auto",
    )
    torch.supa.synchronize()
    memory_mb = torch.supa.max_memory_allocated() / 1024**2

    row = {
        "resolution": key,
        "correctness_rel": corr["rel_error"],
        "v1_mean_ms": round(v1_mean, 3),
        "v1_median_ms": round(v1_ms, 3),
        "v1_trials_ms": [round(v, 3) for v in v1_trials],
        "fused_mean_ms": round(fused_mean, 3),
        "fused_median_ms": round(fused_ms, 3),
        "fused_trials_ms": [round(v, 3) for v in fused_trials],
        "auto_mean_ms": round(auto_mean, 3),
        "auto_median_ms": round(auto_ms, 3),
        "auto_trials_ms": [round(v, 3) for v in auto_trials],
        "speedup_auto_vs_v1": round(v1_ms / max(auto_ms, 1e-9), 3),
        "ref_narrative_ms": ref_ms,
        "gap_auto_to_ref_ms": round(auto_ms - ref_ms, 3),
        "proxy_sol_score_auto": round(gap_score(auto_ms, v1_ms, ref_ms), 4),
        "memory_MB": round(memory_mb, 1),
        "config": {
            "warmup": WARMUP_RUNS,
            "iters": TIMED_ITERS,
            "trials": TRIALS,
            "clone_inputs_each_iter": True,
            "method": "sol_execbench_adapted_biren",
            "formal_path": "auto_v1_lt256_fused_ge256",
        },
    }
    print(row)
    return row


def main() -> None:
    if torch.supa.device_count() < 1:
        raise RuntimeError("no SUPA device")
    from spectral_conv_ops import warmup_spectral_plans

    print(
        {
            "task": "spectral_conv_sol_style_perf",
            "ref": "https://github.com/nvidia/sol-execbench",
            "warmup": WARMUP_RUNS,
            "iters": TIMED_ITERS,
            "trials": TRIALS,
        }
    )
    # Settle suFFT plans before timed trials (SOL-style: exclude one-time setup)
    warmup_spectral_plans(64, 64)
    warmup_spectral_plans(128, 128)
    warmup_spectral_plans(256, 256)

    rows = [benchmark_resolution(h, w) for h, w in PERF_RESOLUTIONS]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = datetime.now().strftime("%Y-%m-%d")

    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    summary.setdefault("meta", {})["updated_at"] = stamp
    spectral = summary.setdefault("spectral_conv", {})
    spectral["sol_style_perf"] = {
        "status": "measured",
        "reference": "https://github.com/nvidia/sol-execbench",
        "note": (
            "Adapted harness: correctness-first, warmup/iters/trials, "
            "input clone per iter, proxy gap score vs v1 baseline and "
            "handbook narrative ref (no Biren SOLAR bound)."
        ),
        "rows": rows,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / f"spectral_sol_style_perf_{day}.md"
    lines = [
        "# SpectralConv SOL-ExecBench-style perf (Biren adapted)",
        "",
        f"- time_utc: {stamp}",
        f"- reference: https://github.com/nvidia/sol-execbench",
        f"- warmup/iters/trials: {WARMUP_RUNS}/{TIMED_ITERS}/{TRIALS}",
        "- scoring_baseline: team v1 (cpu fft + bridged mul)",
        "- ref_target: handbook PyTorch-on-SUPA narrative ms",
        "",
        "| res | v1_med | fused_med | auto_med | speedup_auto | ref_ms | gap_auto | proxy_sol | mem_MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['resolution']} | {row['v1_median_ms']} | {row['fused_median_ms']} | "
            f"{row['auto_median_ms']} | {row['speedup_auto_vs_v1']} | {row['ref_narrative_ms']} | "
            f"{row['gap_auto_to_ref_ms']} | {row['proxy_sol_score_auto']} | {row['memory_MB']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Official contest table in `results.md` still uses §3.2 iters=100 wall clock;",
        "  this log is the rigorous cross-check for optimization decisions (O2).",
        "- Full NVIDIA SOL-Score needs SOLAR bounds on B200; Biren uses proxy only.",
        "",
    ]
    log_path.write_text("\n".join(lines))
    print({"summary": str(SUMMARY_PATH), "run_log": str(log_path), "ok": True})


if __name__ == "__main__":
    main()
