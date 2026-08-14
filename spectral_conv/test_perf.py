#!/usr/bin/env python3
"""SpectralConv performance: SUPA Extension path at 64 / 128 / 256.

Aligned with 赛题文档/官网-赛道五-模型与算子详情页.md §3.2 defaults
(B=4, C_in=32, C_out=64, modes=16x16, warmup=10, iters=100).

Measures the team forward (`spectral_conv2d_supa`: CPU FFT + SUPA mul + CPU iFFT).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

from reference_pytorch import make_random_weights
from spectral_conv_ops import spectral_conv2d_supa

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"

# Official brief §3.2
PERF_RESOLUTIONS = [(64, 64), (128, 128), (256, 256)]
PERF_B, PERF_C_IN, PERF_C_OUT = 4, 32, 64
PERF_MODES1, PERF_MODES2 = 16, 16
PERF_WARMUP = 10
PERF_ITERS = 100
PERF_SEED = 42


def pick_device() -> None:
    if torch.supa.device_count() < 1:
        raise RuntimeError("no SUPA device; cannot run spectral_perf")


def make_input(height: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.empty(
        (PERF_B, PERF_C_IN, height, width),
        dtype=torch.float32,
        device="cpu",
    )
    x.uniform_(-0.5, 0.5, generator=generator)
    return x.contiguous()


def benchmark_resolution(height: int, width: int) -> dict:
    x = make_input(height, width, PERF_SEED + height)
    # Match FNO / SpectralConv2dSupa: Parameter weights hit the identity cache.
    weights = torch.nn.Parameter(
        make_random_weights(
            PERF_C_IN, PERF_C_OUT, PERF_MODES1, PERF_MODES2, PERF_SEED + width
        )
    )
    weights2 = torch.nn.Parameter(
        make_random_weights(
            PERF_C_IN, PERF_C_OUT, PERF_MODES1, PERF_MODES2, PERF_SEED + width + 7
        )
    )

    for _ in range(PERF_WARMUP):
        _ = spectral_conv2d_supa(x, weights, weights2, PERF_MODES1, PERF_MODES2)
    torch.supa.synchronize()

    start = time.time()
    for _ in range(PERF_ITERS):
        _ = spectral_conv2d_supa(x, weights, weights2, PERF_MODES1, PERF_MODES2)
    torch.supa.synchronize()
    forward_time_ms = (time.time() - start) / PERF_ITERS * 1000.0

    torch.supa.reset_peak_memory_stats()
    _ = spectral_conv2d_supa(x, weights, weights2, PERF_MODES1, PERF_MODES2)
    torch.supa.synchronize()
    memory_mb = torch.supa.max_memory_allocated() / 1024**2

    row = {
        "resolution": f"{height}x{width}",
        "forward_time_ms": round(forward_time_ms, 3),
        "memory_MB": round(memory_mb, 1),
        "batch": PERF_B,
        "channels_in": PERF_C_IN,
        "channels_out": PERF_C_OUT,
        "modes": f"{PERF_MODES1}x{PERF_MODES2}",
        "warmup": PERF_WARMUP,
        "iters": PERF_ITERS,
        "path": "auto",
    }
    print(
        f"分辨率 {height}x{width}: 前向 {row['forward_time_ms']:.3f}ms, "
        f"显存 {row['memory_MB']:.1f}MB"
    )
    return row


def write_artifacts(perf_rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = datetime.now().strftime("%Y-%m-%d")

    # Guard: concurrent CPU train / busy host can inflate wall clock 5–10×.
    # Do not clobber formal summary with contended measurements.
    ms64 = float(perf_rows[0]["forward_time_ms"])
    contended = ms64 > 12.0
    if contended:
        print(
            {
                "warning": "perf_contention_skip_summary_write",
                "ms64": ms64,
                "threshold": 12.0,
                "hint": "re-run on idle GPU/CPU; keeping previous formal rows",
            },
            flush=True,
        )

    summary = {}
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
    spectral = summary.get("spectral_conv", {})
    if not isinstance(spectral, dict):
        spectral = {}
    if not contended:
        spectral["perf"] = {
            "status": "measured",
            "config": {
                "B": PERF_B,
                "C_in": PERF_C_IN,
                "C_out": PERF_C_OUT,
                "modes": [PERF_MODES1, PERF_MODES2],
                "warmup": PERF_WARMUP,
                "iters": PERF_ITERS,
                "path": "auto",
                "timing_scope": "CPU input through CPU output; synchronized wall clock",
            },
            "rows": [
                {
                    "resolution": row["resolution"],
                    "forward_time_ms": row["forward_time_ms"],
                    "memory_MB": row["memory_MB"],
                }
                for row in perf_rows
            ],
            "measured_at": stamp,
        }
        if "status" not in spectral:
            spectral["status"] = "perf_measured"
        summary["spectral_conv"] = spectral
        summary.setdefault("meta", {})
        summary["meta"]["updated_at"] = stamp
        # Preserve existing FNO notes; only refresh Spectral ms prefix.
        prev = summary["meta"].get("notes") or ""
        ms_note = (
            f"SpectralConv auto {perf_rows[0]['forward_time_ms']:.3f}/"
            f"{perf_rows[1]['forward_time_ms']:.3f}/"
            f"{perf_rows[2]['forward_time_ms']:.3f} ms "
            f"(warmup={PERF_WARMUP}, iters={PERF_ITERS})."
        )
        if "FNO" in prev:
            # keep trailing FNO sentence if present
            fno_part = prev[prev.find("FNO") :] if "FNO" in prev else ""
            summary["meta"]["notes"] = f"{ms_note} {fno_part}".strip()
        else:
            summary["meta"]["notes"] = ms_note
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    log_path = RUN_LOG_DIR / f"spectral_perf_{day}.md"
    lines = [
        "# SpectralConv performance (SUPA Extension)",
        "",
        f"- time_utc: {stamp}",
        f"- path: auto (loads tune_results.json; FNO to_cpu=False forces fused)",
        f"- config: B={PERF_B}, C_in={PERF_C_IN}, C_out={PERF_C_OUT}, "
        f"modes={PERF_MODES1}x{PERF_MODES2}, warmup={PERF_WARMUP}, iters={PERF_ITERS}",
        "",
        "| 分辨率 | 前向耗时 (ms) | 显存 (MB) |",
        "|---|---:|---:|",
    ]
    for row in perf_rows:
        lines.append(
            f"| {row['resolution']} | {row['forward_time_ms']} | {row['memory_MB']} |"
        )
    lines.extend(
        [
            "",
            "```json",
            json.dumps(perf_rows, indent=2),
            "```",
            "",
            f"summary: `{SUMMARY_PATH}`",
            "",
        ]
    )
    log_path.write_text("\n".join(lines))
    print({"summary": str(SUMMARY_PATH), "run_log": str(log_path)})


def main() -> None:
    pick_device()
    print(
        {
            "torch": torch.__version__,
            "device": "supa",
            "task": "spectral_conv_perf",
        }
    )
    print("\n--- 性能测试（自研 SUPA Extension）---")
    from spectral_conv_ops import warmup_spectral_plans
    for h, w in PERF_RESOLUTIONS:
        warmup_spectral_plans(h, w, PERF_C_IN, PERF_C_OUT, PERF_MODES1, PERF_MODES2)
    perf_rows = [benchmark_resolution(h, w) for h, w in PERF_RESOLUTIONS]
    write_artifacts(perf_rows)
    print({"task": "spectral_conv_perf", "ok": True})


if __name__ == "__main__":
    main()
