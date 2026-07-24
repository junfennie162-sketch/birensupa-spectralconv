#!/usr/bin/env python3
"""Fresh rebench: f side, spectral_conv2d_supa with use_sufft='auto', iters=50.

Mirrors protocol of partner's combo test_perf.py so numbers are directly
comparable. Does NOT modify f business code.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(Path("/workspace/ai4s-f/submission/spectral_conv")))

import torch
import torch_br  # noqa: F401

from spectral_conv_ops import clear_weight_supa_cache, spectral_conv2d_supa, warmup_spectral_plans

RES = [(64, 64), (128, 128), (256, 256)]
B, CIN, COUT = 4, 32, 64
M1, M2 = 16, 16
WARMUP, ITERS = 10, 50


def peak_mb():
    try:
        return float(torch.supa.max_memory_allocated()) / 1024 / 1024
    except Exception:
        try:
            return float(torch_br.supa.max_memory_allocated()) / 1024 / 1024
        except Exception:
            return -1.0


def reset_peak():
    try:
        torch.supa.reset_peak_memory_stats()
    except Exception:
        try:
            torch_br.supa.reset_peak_memory_stats()
        except Exception:
            pass


def main():
    rows = []
    torch.manual_seed(0)
    scale = 1.0 / (CIN * COUT)
    w1 = torch.nn.Parameter(
        (scale * torch.rand(CIN, COUT, M1, M2, dtype=torch.cfloat)).contiguous()
    )
    w2 = torch.nn.Parameter(
        (scale * torch.rand(CIN, COUT, M1, M2, dtype=torch.cfloat)).contiguous()
    )

    for h, w in RES:
        warmup_spectral_plans(h, w, CIN, COUT, M1, M2)
    for h, w in RES:
        x = torch.randn(B, CIN, h, w, dtype=torch.float32)
        clear_weight_supa_cache()
        for _ in range(WARMUP):
            _ = spectral_conv2d_supa(x, w1, w2, M1, M2, use_sufft="auto")
        torch_br.supa.synchronize()
        reset_peak()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            y = spectral_conv2d_supa(x, w1, w2, M1, M2, use_sufft="auto")
        torch_br.supa.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / ITERS
        mem = peak_mb()
        row = {
            "resolution": f"{h}x{w}",
            "forward_time_ms": f"{ms:.3f}",
            "forward_ms_raw": ms,
            "memory_MB": f"{mem:.1f}" if mem >= 0 else "n/a",
            "memory_mb_raw": mem,
            "output_shape": list(y.shape),
            "path": "auto",
            "iters": ITERS,
            "warmup": WARMUP,
        }
        rows.append(row)
        print(row)
    out_dir = Path("/tmp/ai4s-rebench-2026-07-24")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "f_auto_perf.json").write_text(__import__("json").dumps(rows, indent=2))
    print({"task": "f_spectral_auto_perf", "ok": True, "iters": ITERS})


if __name__ == "__main__":
    main()
