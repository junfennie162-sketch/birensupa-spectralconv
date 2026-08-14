#!/usr/bin/env python3
"""Segment timing for adaptive / fused / v1 paths (delivery narrative)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

from reference_pytorch import make_random_weights
from spectral_conv_ops import (
    _weights_to_supa_cached,
    spectral_conv2d_fused,
    spectral_conv2d_supa,
    spectral_mul_supa,
    spectral_mul_supa_device,
)
import spectral_conv_ext

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"
B, CIN, COUT, M = 4, 32, 64, 16


def _ms(fn) -> float:
    torch.supa.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.supa.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def profile_fused(h: int, w: int) -> dict:
    x = torch.randn(B, CIN, h, w)
    weights = make_random_weights(CIN, COUT, M, M, 7)
    # warmup
    _ = spectral_conv2d_fused(x, weights, M, M)
    x_supa = x.to("supa").contiguous()
    w_supa = _weights_to_supa_cached(weights)

    def do_rfft():
        return spectral_conv_ext.rfft2_sufft(x_supa)

    rfft_ms = _ms(do_rfft)
    x_freq = spectral_conv_ext.rfft2_sufft(x_supa)

    def do_mul():
        c = x_freq[:, :, :M, :M, :].contiguous()
        return spectral_mul_supa_device(c, w_supa, synchronize=True)

    mul_ms = _ms(do_mul)
    out = torch.zeros(B, COUT, h, w // 2 + 1, 2, device="supa")
    out[:, :, :M, :M, :] = spectral_mul_supa_device(
        x_freq[:, :, :M, :M, :].contiguous(), w_supa, synchronize=True
    )

    def do_irfft():
        return spectral_conv_ext.irfft2_sufft(out.contiguous(), h, w)

    irfft_ms = _ms(do_irfft)
    return {
        "path": "fused",
        "resolution": f"{h}x{w}",
        "rfft_ms": round(rfft_ms, 3),
        "mul_ms": round(mul_ms, 3),
        "irfft_ms": round(irfft_ms, 3),
    }


def profile_v1(h: int, w: int) -> dict:
    x = torch.randn(B, CIN, h, w)
    weights = make_random_weights(CIN, COUT, M, M, 8)
    _ = spectral_conv2d_supa(x, weights, M, M, use_sufft=False)

    def do_fft():
        return torch.fft.rfft2(x)

    fft_ms = _ms(do_fft)
    xf = torch.fft.rfft2(x)

    def do_bridge_mul():
        return spectral_mul_supa(xf[:, :, :M, :M], weights)

    bridge_ms = _ms(do_bridge_mul)
    out_f = torch.zeros(B, COUT, h, w // 2 + 1, dtype=torch.cfloat)
    out_f[:, :, :M, :M] = spectral_mul_supa(xf[:, :, :M, :M], weights)

    def do_ifft():
        return torch.fft.irfft2(out_f, s=(h, w))

    ifft_ms = _ms(do_ifft)
    return {
        "path": "v1",
        "resolution": f"{h}x{w}",
        "fft_ms": round(fft_ms, 3),
        "bridge_mul_ms": round(bridge_ms, 3),
        "ifft_ms": round(ifft_ms, 3),
    }


def main() -> None:
    rows = []
    for h, w in [(64, 64), (128, 128), (256, 256)]:
        rows.append(profile_v1(h, w))
        rows.append(profile_fused(h, w))
        print(rows[-2])
        print(rows[-1])
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y-%m-%d")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = RUN_LOG_DIR / f"opt_segments_round2_{day}.md"
    lines = [
        "# Segment timing round-2 (delivery narrative)",
        "",
        f"- time_utc: {stamp}",
        "- formal_path: auto (v1 if min(H,W)<256 else fused)",
        "",
    ]
    for row in rows:
        lines.append(f"- {row}")
    lines.append("")
    path.write_text("\n".join(lines))
    print({"run_log": str(path)})


if __name__ == "__main__":
    main()
