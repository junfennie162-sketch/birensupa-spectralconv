#!/usr/bin/env python3
"""suFFT SpectralConv performance vs v1 (CPU FFT) at 64 / 128 / 256."""

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

PERF_RESOLUTIONS = [(64, 64), (128, 128), (256, 256)]
PERF_B, PERF_C_IN, PERF_C_OUT = 4, 32, 64
PERF_MODES1, PERF_MODES2 = 16, 16
PERF_WARMUP = 10
PERF_ITERS = 100
PERF_SEED = 42


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


def benchmark_path(height: int, width: int, use_sufft: bool) -> dict:
    x = make_input(height, width, PERF_SEED + height)
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
    path_name = "sufft_fft_supa_mul" if use_sufft else "cpu_fft_supa_mul"

    for _ in range(PERF_WARMUP):
        _ = spectral_conv2d_supa(
            x, weights, weights2, PERF_MODES1, PERF_MODES2, use_sufft=use_sufft
        )
    torch.supa.synchronize()

    start = time.time()
    for _ in range(PERF_ITERS):
        _ = spectral_conv2d_supa(
            x, weights, weights2, PERF_MODES1, PERF_MODES2, use_sufft=use_sufft
        )
    torch.supa.synchronize()
    forward_time_ms = (time.time() - start) / PERF_ITERS * 1000.0

    torch.supa.reset_peak_memory_stats()
    _ = spectral_conv2d_supa(
        x, weights, weights2, PERF_MODES1, PERF_MODES2, use_sufft=use_sufft
    )
    torch.supa.synchronize()
    memory_mb = torch.supa.max_memory_allocated() / 1024**2

    row = {
        "resolution": f"{height}x{width}",
        "path": path_name,
        "forward_time_ms": round(forward_time_ms, 3),
        "memory_MB": round(memory_mb, 1),
    }
    print(
        f"[{path_name}] {height}x{width}: {row['forward_time_ms']:.3f}ms, "
        f"{row['memory_MB']:.1f}MB"
    )
    return row


def main() -> None:
    if torch.supa.device_count() < 1:
        raise RuntimeError("no SUPA device")
    print({"task": "spectral_conv_sufft_perf", "iters": PERF_ITERS})

    v1_rows = []
    sufft_rows = []
    for height, width in PERF_RESOLUTIONS:
        v1_rows.append(benchmark_path(height, width, use_sufft=False))
        sufft_rows.append(benchmark_path(height, width, use_sufft=True))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = datetime.now().strftime("%Y-%m-%d")
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    spectral = summary.setdefault("spectral_conv", {})
    spectral.setdefault("sufft", {})
    spectral["sufft"]["perf"] = {
        "status": "measured",
        "config": {
            "B": PERF_B,
            "C_in": PERF_C_IN,
            "C_out": PERF_C_OUT,
            "modes": [PERF_MODES1, PERF_MODES2],
            "warmup": PERF_WARMUP,
            "iters": PERF_ITERS,
            "note": "batched 1D suFFT + plan cache; compared with v1 CPU FFT",
        },
        "rows_v1": v1_rows,
        "rows_sufft": sufft_rows,
    }
    summary.setdefault("meta", {})["updated_at"] = stamp
    summary["meta"]["notes"] = (
        "strengthen: suFFT accuracy+perf measured vs v1; formal pack still deferred."
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / f"spectral_sufft_perf_{day}.md"
    lines = [
        "# SpectralConv suFFT vs v1 performance",
        "",
        f"- time_utc: {stamp}",
        f"- config: B={PERF_B}, C_in={PERF_C_IN}, C_out={PERF_C_OUT}, "
        f"modes={PERF_MODES1}x{PERF_MODES2}, iters={PERF_ITERS}",
        "",
        "| 分辨率 | v1 (ms) | suFFT (ms) | v1 显存 (MB) | suFFT 显存 (MB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for left, right in zip(v1_rows, sufft_rows):
        lines.append(
            f"| {left['resolution']} | {left['forward_time_ms']} | "
            f"{right['forward_time_ms']} | {left['memory_MB']} | {right['memory_MB']} |"
        )
    lines.extend(["", f"summary: `{SUMMARY_PATH}`", ""])
    log_path.write_text("\n".join(lines))
    print({"run_log": str(log_path), "ok": True})


if __name__ == "__main__":
    main()
