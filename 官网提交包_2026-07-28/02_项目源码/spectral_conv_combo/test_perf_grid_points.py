#!/usr/bin/env python3
"""SpectralConv performance in official 'grid_points/s' metric.

Official metric (per user-provided screenshot):
    grid_points/s = Σ(H·W·batch) / Σt_warmup
    with warmup=10, statistics=50, batch_size=16.

Run:
    cd spectral_conv_combo && python3 test_perf_grid_points.py
    # → results/run_logs/spectral_grid_points_<date>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "fno_ns"))

import torch  # noqa: E402

# Reuse combo API
from spectral_conv_ops import spectral_conv2d_supa  # noqa: E402


def measure_grid_points(resolution: int, batch_size: int = 16, modes: int = 16,
                       in_channels: int = 32, out_channels: int = 64,
                       warmup: int = 10, iters: int = 50) -> dict:
    H = W = resolution
    x = torch.randn(batch_size, in_channels, H, W)
    w1 = torch.randn(in_channels, out_channels, modes, modes)
    w2 = torch.randn(in_channels, out_channels, modes, modes)
    # warmup
    for _ in range(warmup):
        _ = spectral_conv2d_supa(x, w1, w2, modes, modes)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    # measure
    times = []
    t_total = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = spectral_conv2d_supa(x, w1, w2, modes, modes)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_total += time.perf_counter() - t0
        times.append((time.perf_counter() - t0) * 1000.0)
    gp_per_s = (batch_size * H * W) / (t_total / iters)
    return {
        "resolution": f"{H}x{W}",
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        "mean_ms": round(sum(times) / iters, 4),
        "median_ms": round(sorted(times)[iters // 2], 4),
        "min_ms": round(min(times), 4),
        "max_ms": round(max(times), 4),
        "grid_points_per_s": int(gp_per_s),
    }


def main():
    rows = []
    for res in (64, 128, 256):
        rows.append(measure_grid_points(res, batch_size=16))
    print(json.dumps(rows, indent=2))
    out_dir = ROOT.parent / "results" / "run_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "spectral_grid_points.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()