#!/usr/bin/env python3
"""Auto-tune the effective SpectralConv path and buffer-cache limit.

The sweep uses the same CPU-input to CPU-output scope as the official operator
benchmark. Only knobs consumed by the current implementation are scanned.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

import spectral_conv_ops as ops

THIS_DIR = Path(__file__).resolve().parent
RESULT_PATH = THIS_DIR / "tune_results.json"
PATH_CHOICES = ("v1", "fused")
BUFFER_CHOICES = (2, 4, 8)
DEFAULT_SHAPES = (64, 128, 256)
BATCH_SIZE = 4
CHANNELS_IN = 32
CHANNELS_OUT = 64
MODES = 16
SEED = 20260725


def reset_peak_memory() -> None:
    if hasattr(torch_br.supa, "reset_peak_memory_stats"):
        torch_br.supa.reset_peak_memory_stats()
    elif hasattr(torch.supa, "reset_peak_memory_stats"):
        torch.supa.reset_peak_memory_stats()


def peak_memory_mb() -> float:
    if hasattr(torch_br.supa, "max_memory_allocated"):
        return float(torch_br.supa.max_memory_allocated()) / 1024**2
    return float(torch.supa.max_memory_allocated()) / 1024**2


def make_inputs(resolution: int):
    generator = torch.Generator(device="cpu").manual_seed(SEED + resolution)
    input_tensor = torch.randn(
        BATCH_SIZE,
        CHANNELS_IN,
        resolution,
        resolution,
        generator=generator,
        dtype=torch.float32,
    )
    scale = 1.0 / (CHANNELS_IN * CHANNELS_OUT)
    weights1 = torch.nn.Parameter(
        scale
        * torch.rand(
            CHANNELS_IN,
            CHANNELS_OUT,
            MODES,
            MODES,
            generator=generator,
            dtype=torch.cfloat,
        )
    )
    weights2 = torch.nn.Parameter(
        scale
        * torch.rand(
            CHANNELS_IN,
            CHANNELS_OUT,
            MODES,
            MODES,
            generator=generator,
            dtype=torch.cfloat,
        )
    )
    return input_tensor, weights1, weights2


def benchmark(
    resolution: int,
    path: str,
    buffer_max: int,
    warmup: int,
    iters: int,
) -> dict:
    assert path in PATH_CHOICES
    assert buffer_max in BUFFER_CHOICES
    ops.clear_weight_supa_cache()
    ops._AUTO_TUNE_TABLE.clear()
    ops._AUTO_TUNE_TABLE["__global__"] = {"buffer_max": buffer_max}
    input_tensor, weights1, weights2 = make_inputs(resolution)
    use_sufft = path == "fused"

    def call() -> torch.Tensor:
        return ops.spectral_conv2d_supa(
            input_tensor,
            weights1,
            weights2,
            MODES,
            MODES,
            use_sufft=use_sufft,
            to_cpu=True,
        )

    try:
        for _ in range(warmup):
            call()
        torch_br.supa.synchronize()
        reset_peak_memory()
        samples: list[float] = []
        for _ in range(iters):
            torch_br.supa.synchronize()
            start = time.perf_counter()
            call()
            torch_br.supa.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
        return {
            "path": path,
            "buffer_max": buffer_max,
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples),
            "peak_memory_MB": peak_memory_mb(),
            "ok": True,
        }
    except Exception as exception:
        return {
            "path": path,
            "buffer_max": buffer_max,
            "median_ms": None,
            "mean_ms": None,
            "peak_memory_MB": None,
            "ok": False,
            "error": str(exception),
        }


def _is_stable(row: dict, max_mean_median_ratio: float = 2.0) -> bool:
    median = float(row["median_ms"])
    mean = float(row["mean_ms"])
    if median <= 0.0:
        return False
    return (mean / median) <= max_mean_median_ratio


def pick_best(rows: list[dict]) -> dict:
    successful_rows = [row for row in rows if row["ok"]]
    if not successful_rows:
        raise RuntimeError("all auto-tune configurations failed")
    # Drop unstable rows first (v1 often shows median≈1.5 ms with mean≈13–20 ms).
    stable_rows = [row for row in successful_rows if _is_stable(row)]
    candidates = stable_rows or successful_rows
    return min(
        candidates,
        key=lambda row: (row["mean_ms"], row["median_ms"], row["peak_memory_MB"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", nargs="+", type=int, default=list(DEFAULT_SHAPES))
    parser.add_argument("--quick", action="store_true", help="3 warmup and 3 measured calls")
    parser.add_argument("--dry-run", action="store_true", help="do not apply decisions in this process")
    parser.add_argument("--out", type=Path, default=RESULT_PATH)
    arguments = parser.parse_args()
    warmup, iters = (3, 3) if arguments.quick else (10, 30)

    sweep: list[dict] = []
    table: dict[int, dict] = {}
    for resolution in arguments.shape:
        rows = [
            benchmark(resolution, path, buffer_max, warmup, iters)
            for path in PATH_CHOICES
            for buffer_max in BUFFER_CHOICES
        ]
        best = pick_best(rows)
        sweep.append({"resolution": resolution, "rows": rows, "best": best})
        table[resolution] = {
            "use_sufft": best["path"] == "fused",
            "buffer_max": best["buffer_max"],
        }
        print(
            {
                "resolution": resolution,
                "best_path": best["path"],
                "buffer_max": best["buffer_max"],
                "median_ms": best["median_ms"],
                "peak_memory_MB": best["peak_memory_MB"],
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "CPU input through CPU output; synchronized wall clock",
        "config": {
            "B": BATCH_SIZE,
            "C_in": CHANNELS_IN,
            "C_out": CHANNELS_OUT,
            "modes": [MODES, MODES],
            "warmup": warmup,
            "iters": iters,
            "seed": SEED,
        },
        "sweep": sweep,
        "table": table,
    }
    arguments.out.write_text(json.dumps(payload, indent=2) + "\n")
    if not arguments.dry_run:
        ops._AUTO_TUNE_TABLE.clear()
        ops._AUTO_TUNE_TABLE.update(table)
    print({"result": str(arguments.out), "applied_entries": 0 if arguments.dry_run else len(table)})


if __name__ == "__main__":
    main()
