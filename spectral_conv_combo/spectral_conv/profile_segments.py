#!/usr/bin/env python3
"""P0: segment timing profile for v1 vs suFFT SpectralConv paths."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

import spectral_conv_ext
from reference_pytorch import make_random_weights
from spectral_conv_ops import (
    _complex_to_interleaved,
    _interleaved_to_complex,
    spectral_mul_supa,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"
SUMMARY_PATH = RESULTS_DIR / "summary.json"

B, CIN, COUT = 4, 32, 64
MODES = 16
ITERS = 30
WARMUP = 5


def sync():
    torch.supa.synchronize()


def profile_v1(height: int, width: int) -> dict:
    x = torch.randn(B, CIN, height, width)
    w = make_random_weights(CIN, COUT, MODES, MODES, 0)
    # warmup
    for _ in range(WARMUP):
        xf = torch.fft.rfft2(x)
        _ = spectral_mul_supa(xf[:, :, :MODES, :MODES], w)
        sync()

    t_fft = t_h2d_mul_d2h = t_ifft = 0.0
    for _ in range(ITERS):
        sync()
        t0 = time.perf_counter()
        xf = torch.fft.rfft2(x)
        sync()
        t_fft += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = spectral_mul_supa(xf[:, :, :MODES, :MODES], w)
        sync()
        t_h2d_mul_d2h += time.perf_counter() - t0

        out_f = torch.zeros(B, COUT, height, width // 2 + 1, dtype=torch.complex64)
        # mul already timed; rebuild cheap for ifft shape
        out_f[:, :, :MODES, :MODES] = spectral_mul_supa(xf[:, :, :MODES, :MODES], w)
        sync()
        t0 = time.perf_counter()
        _ = torch.fft.irfft2(out_f, s=(height, width))
        sync()
        t_ifft += time.perf_counter() - t0

    scale = 1000.0 / ITERS
    return {
        "path": "v1_cpu_fft",
        "resolution": f"{height}x{width}",
        "fft_ms": round(t_fft * scale, 3),
        "h2d_mul_d2h_ms": round(t_h2d_mul_d2h * scale, 3),
        "ifft_ms": round(t_ifft * scale, 3),
        "note": "h2d_mul_d2h includes spectral_mul_supa full bridge",
    }


def profile_sufft(height: int, width: int) -> dict:
    x = torch.randn(B, CIN, height, width)
    w = make_random_weights(CIN, COUT, MODES, MODES, 0)
    x_supa = x.to("supa").contiguous()
    for _ in range(WARMUP):
        xf = spectral_conv_ext.rfft2_sufft(x_supa)
        sync()
        _ = spectral_mul_supa(_interleaved_to_complex(xf.cpu())[:, :, :MODES, :MODES], w)
        sync()

    t_rfft = t_bridge_mul = t_irfft = 0.0
    for _ in range(ITERS):
        sync()
        t0 = time.perf_counter()
        xf = spectral_conv_ext.rfft2_sufft(x_supa)
        sync()
        t_rfft += time.perf_counter() - t0

        t0 = time.perf_counter()
        x_cpu = _interleaved_to_complex(xf.cpu())
        y_modes = spectral_mul_supa(x_cpu[:, :, :MODES, :MODES], w)
        out_f = torch.zeros(B, COUT, height, width // 2 + 1, dtype=torch.complex64)
        out_f[:, :, :MODES, :MODES] = y_modes
        out_i = _complex_to_interleaved(out_f).to("supa")
        sync()
        t_bridge_mul += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = spectral_conv_ext.irfft2_sufft(out_i, height, width)
        sync()
        t_irfft += time.perf_counter() - t0

    scale = 1000.0 / ITERS
    return {
        "path": "sufft_with_cpu_bridge",
        "resolution": f"{height}x{width}",
        "rfft_ms": round(t_rfft * scale, 3),
        "bridge_mul_ms": round(t_bridge_mul * scale, 3),
        "irfft_ms": round(t_irfft * scale, 3),
        "note": "bridge_mul = spectrum D2H + mul H2D/D2H + spectrum H2D",
    }


def main() -> None:
    rows = []
    for h, w in [(64, 64), (128, 128), (256, 256)]:
        print(f"profile {h}x{w} v1...")
        rows.append(profile_v1(h, w))
        print(rows[-1])
        print(f"profile {h}x{w} sufft...")
        rows.append(profile_sufft(h, w))
        print(rows[-1])

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = datetime.now().strftime("%Y-%m-%d")
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / f"opt_baseline_{day}.md"

    # freeze known summary perf into baseline section
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    v1_perf = (summary.get("spectral_conv") or {}).get("perf") or {}
    sufft_perf = ((summary.get("spectral_conv") or {}).get("sufft") or {}).get("perf") or {}

    lines = [
        "# Optimization baseline (P0)",
        "",
        f"- time_utc: {stamp}",
        f"- config: B={B}, Cin={CIN}, Cout={COUT}, modes={MODES}, iters={ITERS}",
        "",
        "## Frozen end-to-end (from summary.json)",
        "",
        f"- v1 rows: {json.dumps(v1_perf.get('rows'), ensure_ascii=False)}",
        f"- sufft rows: {json.dumps(sufft_perf.get('rows_sufft'), ensure_ascii=False)}",
        "",
        "## Segment timing (this profile)",
        "",
        "```json",
        json.dumps(rows, indent=2),
        "```",
        "",
        "## Conclusion",
        "",
        "- Expected: H2D/D2H around `spectral_mul_supa` dominates small/medium resolutions.",
        "- P1 target: keep spectrum on SUPA; mul without CPU bridge.",
        "",
    ]
    log_path.write_text("\n".join(lines))

    summary.setdefault("meta", {})["updated_at"] = stamp
    summary.setdefault("optimization", {})["p0_baseline"] = {
        "status": "done",
        "run_log": str(log_path),
        "segment_rows": rows,
        "frozen_v1": v1_perf.get("rows"),
        "frozen_sufft": sufft_perf.get("rows_sufft"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print({"ok": True, "run_log": str(log_path)})


if __name__ == "__main__":
    main()
