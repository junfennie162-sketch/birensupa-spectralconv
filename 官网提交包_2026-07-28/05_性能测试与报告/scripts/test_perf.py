#!/usr/bin/env python3
"""Full SpectralConv perf at 64/128/256 (official §3.2: time + memory)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

from spectral_conv_ops import clear_weight_supa_cache, spectral_conv2d_supa, warmup_spectral_plans

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"

RESOLUTIONS = [(64, 64), (128, 128), (256, 256)]
B, CIN, COUT = 4, 32, 64
MODES1, MODES2 = 16, 16
WARMUP, ITERS = 10, 50


def _peak_memory_mb() -> float:
    try:
        if hasattr(torch_br.supa, "max_memory_allocated"):
            return float(torch_br.supa.max_memory_allocated()) / (1024.0 * 1024.0)
        if hasattr(torch.supa, "max_memory_allocated"):
            return float(torch.supa.max_memory_allocated()) / (1024.0 * 1024.0)
    except Exception:
        pass
    return -1.0


def _reset_peak_memory() -> None:
    try:
        if hasattr(torch_br.supa, "reset_peak_memory_stats"):
            torch_br.supa.reset_peak_memory_stats()
        elif hasattr(torch.supa, "reset_peak_memory_stats"):
            torch.supa.reset_peak_memory_stats()
    except Exception:
        pass


def main():
    rows = []
    torch.manual_seed(0)
    scale = 1.0 / (CIN * COUT)
    w1 = torch.nn.Parameter(
        (scale * torch.rand(CIN, COUT, MODES1, MODES2, dtype=torch.cfloat)).contiguous()
    )
    w2 = torch.nn.Parameter(
        (scale * torch.rand(CIN, COUT, MODES1, MODES2, dtype=torch.cfloat)).contiguous()
    )

    for height, width in RESOLUTIONS:
        warmup_spectral_plans(height, width, CIN, COUT, MODES1, MODES2)

    for height, width in RESOLUTIONS:
        x = torch.randn(B, CIN, height, width, dtype=torch.float32)
        clear_weight_supa_cache()
        for _ in range(WARMUP):
            _ = spectral_conv2d_supa(x, w1, w2, MODES1, MODES2, use_sufft="auto")
        torch_br.supa.synchronize()
        _reset_peak_memory()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            y = spectral_conv2d_supa(x, w1, w2, MODES1, MODES2, use_sufft="auto")
        torch_br.supa.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / ITERS
        mem = _peak_memory_mb()
        row = {
            "resolution": f"{height}x{width}",
            "forward_time_ms": f"{ms:.3f}",
            "forward_ms_raw": ms,
            "memory_MB": f"{mem:.1f}" if mem >= 0 else "n/a",
            "memory_mb_raw": mem,
            "output_shape": list(y.shape),
            "path": "auto",
        }
        rows.append(row)
        print(row)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    summary.setdefault("meta", {})["updated_at"] = stamp
    sc = summary.setdefault("spectral_conv_combo", {})
    sc["perf"] = rows
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    log_path = RUN_LOG_DIR / f"spectral_combo_perf_{datetime.now().strftime('%Y-%m-%d')}.md"
    lines = [
        "# SpectralConv combo perf (use_sufft=auto + warmup)",
        "",
        f"- time_utc: {stamp}",
        f"- warmup/iters: {WARMUP}/{ITERS}",
        "",
        "| resolution | forward_ms | memory_MB |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['resolution']} | {row['forward_time_ms']} | {row['memory_MB']} |"
        )
    log_path.write_text("\n".join(lines) + "\n")
    print({"task": "spectral_conv_combo_perf", "ok": True})


if __name__ == "__main__":
    main()
