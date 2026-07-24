#!/usr/bin/env python3
"""Full SpectralConv perf at 64/128/256 (CPU FFT + SUPA mul)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

from spectral_conv_ops import spectral_conv2d_supa

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"

RESOLUTIONS = [(64, 64), (128, 128), (256, 256)]
B, CIN, COUT = 4, 32, 64
MODES1, MODES2 = 16, 16
WARMUP, ITERS = 5, 20


def main():
    rows = []
    torch.manual_seed(0)
    scale = 1.0 / (CIN * COUT)
    w1 = (scale * torch.rand(CIN, COUT, MODES1, MODES2, dtype=torch.cfloat)).contiguous()
    w2 = (scale * torch.rand(CIN, COUT, MODES1, MODES2, dtype=torch.cfloat)).contiguous()

    for height, width in RESOLUTIONS:
        x = torch.randn(B, CIN, height, width, dtype=torch.float32)
        for _ in range(WARMUP):
            _ = spectral_conv2d_supa(x, w1, w2, MODES1, MODES2)
        torch_br.supa.synchronize()
        t0 = time.time()
        for _ in range(ITERS):
            y = spectral_conv2d_supa(x, w1, w2, MODES1, MODES2)
        torch_br.supa.synchronize()
        ms = (time.time() - t0) * 1000.0 / ITERS
        row = {
            "resolution": f"{height}x{width}",
            "forward_time_ms": f"{ms:.3f}",
            "forward_ms_raw": ms,
            "output_shape": list(y.shape),
        }
        rows.append(row)
        print(row)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    summary.setdefault("meta", {})["updated_at"] = stamp
    sc = summary.setdefault("spectral_conv", {})
    sc["perf"] = rows
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    log_path = RUN_LOG_DIR / f"spectral_perf_{datetime.now().strftime('%Y-%m-%d')}.md"
    lines = [
        "# SpectralConv full-pipeline perf (CPU FFT + SUPA mul)",
        "",
        f"- time_utc: {stamp}",
        "",
        "| resolution | forward_ms |",
        "|---|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['resolution']} | {row['forward_time_ms']} |")
    log_path.write_text("\n".join(lines) + "\n")
    print({"task": "spectral_conv_perf", "ok": True})


if __name__ == "__main__":
    main()
