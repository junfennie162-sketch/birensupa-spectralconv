#!/usr/bin/env python3
"""Single-forward validation of FNO-NS (≥4 layers) with SUPA SpectralConv."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "spectral_conv_combo"))

from model import FNO2d  # noqa: E402

RESULTS_DIR = ROOT.parent / "results"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"
SUMMARY_PATH = RESULTS_DIR / "summary.json"


def make_synthetic_batch(batch_size=2, t_in=10, resolution=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Synthetic vorticity-like field in [-1, 1]
    return torch.randn(batch_size, t_in, resolution, resolution, generator=g)


def main():
    import torch_br  # noqa: F401

    device_probe = str(torch.empty((1,), device="supa").device)
    model = FNO2d(modes1=8, modes2=8, width=16, n_layers=4, in_channels=10, out_channels=1)
    model.eval()
    x = make_synthetic_batch()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 64, 64), y.shape
    n_fourier = len(model.fourier_layers)
    assert n_fourier >= 4

    import time

    # Baseline submission path timing (CPU-dwelling).
    for _ in range(3):
        _ = model(x)
    torch_br.supa.synchronize()
    t0 = time.perf_counter()
    fast_iters = 20
    for _ in range(fast_iters):
        _ = model(x)
    torch_br.supa.synchronize()
    baseline_ms = (time.perf_counter() - t0) * 1000.0 / fast_iters

    # SUPA fast-path benchmark (does not change `y`).
    model_cpu = FNO2d(modes1=8, modes2=8, width=16, n_layers=4, in_channels=10, out_channels=1)
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()
    model_cpu.prepare_supa_eval()
    x_supa = x.to("supa").contiguous()
    for _ in range(3):
        _ = model_cpu.forward_supa_chain(x_supa)
    torch_br.supa.synchronize()
    t0 = time.perf_counter()
    for _ in range(fast_iters):
        _ = model_cpu.forward_supa_chain(x_supa)
    torch_br.supa.synchronize()
    fast_ms = (time.perf_counter() - t0) * 1000.0 / fast_iters

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {
        "task": "fno_ns_forward",
        "ok": True,
        "supa_probe": device_probe,
        "n_fourier_layers": n_fourier,
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "output_min": float(y.min()),
        "output_max": float(y.max()),
        "forward_ms_baseline": round(baseline_ms, 3),
        "forward_ms_supa_chain": round(fast_ms, 3),
        "speedup": round(baseline_ms / fast_ms, 3) if fast_ms > 0 else 1.0,
        "time_utc": stamp,
    }
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    summary.setdefault("meta", {})["updated_at"] = stamp
    summary["fno_ns"] = {"status": "forward_pass", "report": report}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    (RUN_LOG_DIR / f"fno_forward_{datetime.now().strftime('%Y-%m-%d')}.md").write_text(
        "# FNO-NS forward (chain & baseline)\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n"
    )


if __name__ == "__main__":
    main()
