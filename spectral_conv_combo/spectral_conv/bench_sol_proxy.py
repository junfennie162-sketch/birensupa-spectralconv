#!/usr/bin/env python3
"""SOL-style gap proxy for SpectralConv + FNO batch16 (NOT official hardware SOL).

Reports wall time, peak memory, crude arithmetic intensity / GB/s proxies, and
the gap versus a simple machine upper-bound estimate. Explicitly labelled as a
local analysis aid per 赛道验收清单 §3.3 SOL-Score / SOL-ExecBench 思路.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "fno_ns"))

from reference_pytorch import make_random_weights  # noqa: E402
from spectral_conv_ops import clear_weight_supa_cache, spectral_conv2d_supa  # noqa: E402

# Conservative Biren106B analysis anchors (not vendor SOL; for gap storytelling).
PEAK_TFLOPS_FP32_PROXY = 40.0  # rough device-class proxy
PEAK_HBM_TB_S_PROXY = 1.0  # TB/s class proxy


def _sync() -> None:
    torch_br.supa.synchronize()


def _bench_spectral(h: int, warmup: int = 10, iters: int = 50) -> dict:
    b, cin, cout, modes = 4, 32, 64, 16
    clear_weight_supa_cache()
    g = torch.Generator().manual_seed(42 + h)
    x = torch.empty(b, cin, h, h)
    x.uniform_(-0.5, 0.5, generator=g)
    w1 = make_random_weights(cin, cout, modes, modes, 100 + h)
    w2 = make_random_weights(cin, cout, modes, modes, 200 + h)
    for _ in range(warmup):
        spectral_conv2d_supa(x, w1, w2, modes, modes, use_sufft="auto")
    if hasattr(torch_br.supa, "reset_peak_memory_stats"):
        torch_br.supa.reset_peak_memory_stats()
    _sync()
    xs = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        spectral_conv2d_supa(x, w1, w2, modes, modes, use_sufft="auto")
        _sync()
        xs.append(time.perf_counter() - t0)
    median_s = statistics.median(xs)
    # Traffic proxy: x + 2*weights + y (float32) + spectrum-ish (rough).
    bytes_rw = (
        b * cin * h * h * 4
        + 2 * (cin * cout * modes * modes * 8)  # complex64
        + b * cout * h * h * 4
    )
    # Flops proxy: 2 corners * B * Cout * M1 * M2 * Cin * 6 (complex mul-add-ish)
    flops = 2 * b * cout * modes * modes * cin * 6
    gb_s = (bytes_rw / median_s) / 1e9
    tflops = (flops / median_s) / 1e12
    return {
        "resolution": f"{h}x{h}",
        "median_ms": median_s * 1000,
        "peak_memory_MB": float(torch_br.supa.max_memory_allocated() / (1024**2))
        if hasattr(torch_br.supa, "max_memory_allocated")
        else None,
        "bytes_rw_proxy": bytes_rw,
        "flops_proxy": flops,
        "GB_s_proxy": gb_s,
        "TFLOPS_proxy": tflops,
        "sol_time_vs_peak_compute": tflops / PEAK_TFLOPS_FP32_PROXY,
        "sol_bw_vs_peak_hbm": gb_s / (PEAK_HBM_TB_S_PROXY * 1000),
        "bottleneck_hint": "memory" if (gb_s / (PEAK_HBM_TB_S_PROXY * 1000)) > (tflops / PEAK_TFLOPS_FP32_PROXY) else "compute_or_launch",
    }


def _bench_fno_batch16(warmup: int = 10, iters: int = 50) -> dict:
    from model import FNO2d

    torch.manual_seed(20260722)
    model = FNO2d(16, 16, 32, 4, 10, 1).eval()
    model.prepare_supa_eval()
    x = torch.randn(16, 10, 64, 64, device="supa")
    with torch.no_grad():
        for _ in range(warmup):
            model.forward_supa_chain(x, use_sufft="auto")
        if hasattr(torch_br.supa, "reset_peak_memory_stats"):
            torch_br.supa.reset_peak_memory_stats()
        _sync()
        xs = []
        for _ in range(iters):
            _sync()
            t0 = time.perf_counter()
            model.forward_supa_chain(x, use_sufft="auto")
            _sync()
            xs.append(time.perf_counter() - t0)
    median_s = statistics.median(xs)
    grid_points = iters * 16 * 64 * 64  # for reference; rate uses median*iters? use 1/median
    gps = (16 * 64 * 64) / median_s
    # Dominant cost narrative from R6/R7 skills: 4× (safe D2D + suFFT + mul + IN/GELU)
    return {
        "batch_size": 16,
        "median_ms_per_batch": median_s * 1000,
        "grid_points_per_second": gps,
        "peak_memory_MB": float(torch_br.supa.max_memory_allocated() / (1024**2))
        if hasattr(torch_br.supa, "max_memory_allocated")
        else None,
        "dominant_cost_note": "per-layer host-seeded D2D + suFFT permute/plan + spectral_mul + IN/GELU/1x1",
    }


def main() -> int:
    spectral = [_bench_spectral(h) for h in (64, 128, 256)]
    fno = _bench_fno_batch16()
    report = {
        "status": "measured",
        "disclaimer": (
            "Local SOL-style proxy only. Not official Biren SOL / SOL-ExecBench. "
            "Peak TFLOPS/HBM anchors are conservative analysis placeholders."
        ),
        "anchors": {
            "peak_tflops_fp32_proxy": PEAK_TFLOPS_FP32_PROXY,
            "peak_hbm_tb_s_proxy": PEAK_HBM_TB_S_PROXY,
        },
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": "Biren106B / supa",
        "spectral_conv_auto": spectral,
        "fno_batch16_random": fno,
        "takeaways": [
            "Spectral formal path is CPU-origin → no R7 safe-buffer tax; dominated by suFFT + mul.",
            "FNO chain still pays per-layer host-seeded D2D before suFFT (correctness).",
            "mul kernel R7 float2/unroll is ~noise vs prior; bigger FNO wins came from R6→R7 materialization.",
        ],
    }
    out_json = RESULTS / "sol_proxy_r7.json"
    out_md = RESULTS / "run_logs" / "sol_proxy_r7_2026-07-25.md"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "run_logs").mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# SOL-style gap proxy (R7)",
        "",
        f"- measured_at: {report['measured_at']}",
        f"- disclaimer: {report['disclaimer']}",
        "",
        "## SpectralConv auto",
        "",
        "| res | median ms | GB/s proxy | TFLOPS proxy | bottleneck hint |",
        "|---|---:|---:|---:|---|",
    ]
    for row in spectral:
        lines.append(
            f"| {row['resolution']} | {row['median_ms']:.3f} | {row['GB_s_proxy']:.2f} | "
            f"{row['TFLOPS_proxy']:.4f} | {row['bottleneck_hint']} |"
        )
    lines += [
        "",
        "## FNO batch16 (random model)",
        "",
        f"- median_ms/batch: {fno['median_ms_per_batch']:.3f}",
        f"- grid_points/s: {fno['grid_points_per_second']:.0f}",
        f"- peak_MB: {fno['peak_memory_MB']}",
        f"- note: {fno['dominant_cost_note']}",
        "",
        "## Takeaways",
        "",
    ]
    for t in report["takeaways"]:
        lines.append(f"- {t}")
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    print({"wrote": str(out_json), "md": str(out_md)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
