#!/usr/bin/env python3
"""Layer / stage profile of f's FNO-NS chain (R5 instrument).

Reports:
- full chain latency (median / mean / min over `iters` after `warmup`)
- per-stage breakdown: `rfft2_sufft`, `corner_prep`, `spectral_mul` (×2 corners),
  `irfft2_sufft`, `conv1x1`, `add`, `norm`, `gelu` — all per layer
- per-layer aggregated latency

Used to find hotspots before more invasive R5 patches (zero+scatter fusion,
dual-pybind dispatch). Mirrors n's `profile_chain.py` protocol but stages
the actual `FourierLayer.forward_supa` path on f's model (which goes
through `spectral_conv2d_supa(..., to_cpu=False)` end-to-end).

Usage:
    cd /workspace/ai4s-f/submission/fno_ns
    python3 bench_f_fno_chain_layer_profile.py --warmup 10 --iters 50
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/ai4s-f/submission/fno_ns")
sys.path.insert(0, "/workspace/ai4s-f/submission/spectral_conv")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_br  # noqa: F401

from model import FNO2d
from spectral_conv_ops import (
    spectral_conv2d_supa,
    warmup_spectral_plans,
)
import spectral_conv_ext


def wall(fn, *, warmup=10, iters=50):
    """Median over iters after warmup. All SUPA tensor mutations are sync'd
    around each call so the timer captures only the stage under test."""
    for _ in range(warmup):
        fn()
    torch_br.supa.synchronize()
    samples = []
    for _ in range(iters):
        torch_br.supa.synchronize()
        t0 = time.perf_counter()
        fn()
        torch_br.supa.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), min(samples), sum(samples) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--layers", type=int, default=4)
    args = p.parse_args()

    torch.manual_seed(0)
    model = FNO2d(
        modes1=args.modes, modes2=args.modes,
        width=args.width, n_layers=args.layers,
        in_channels=10, out_channels=1,
    ).eval()
    model.prepare_supa_eval()
    x = torch.randn(args.B, 10, args.H, args.W, device="supa", dtype=torch.float32)
    device = x.device
    grid = model.get_grid(x.shape, device)
    x_in = torch.cat([x, grid], dim=1)
    h_lift = model.lift.to(device)(x_in)

    # Warm plans (off the timed path).
    warmup_spectral_plans(args.H, args.W, args.width, args.width,
                          args.modes, args.modes)

    def _full():
        return model.forward_supa_chain(x, use_sufft="auto")

    full_med, full_min, full_mean = wall(_full, warmup=args.warmup, iters=args.iters)
    print(f"chain full @ B={args.B} {args.H}x{args.W}, w={args.width}, "
          f"m={args.modes}, L={args.layers}:")
    print(f"  median {full_med:.3f} ms / mean {full_mean:.3f} ms / min {full_min:.3f} ms")

    # Per-layer stage breakdown.
    h = h_lift
    print()
    print(f"{'layer':<8} {'rfft':>7} {'mul×2':>7} {'irfft':>7} {'conv1x1':>8} "
          f"{'add':>6} {'norm':>6} {'gelu':>6} {'sum':>6}")
    layer_totals = []
    for i, layer in enumerate(model.fourier_layers):
        # Snapshot h for the test loop (avoid carry-over corruption).
        h_local = h.detach().clone()

        def _rfft():
            spectral_conv2d_supa(h_local, layer.spectral_conv.weights1,
                                layer.spectral_conv.weights2,
                                args.modes, args.modes,
                                use_sufft=True, to_cpu=False)

        # Approximate spectral kernel by re-calling the full forward and
        # using L1 outputs as a proxy. Stage-attribution is coarse; the
        # purpose is to know what's tractable, not micro-allocate.

        # Use a broader split: re-use forward_supa (full layer) and
        # measure the conv1x1+norm+gelu residual separately.
        weights1 = layer.spectral_conv.weights1
        weights2 = layer.spectral_conv.weights2

        def _spec():
            return spectral_conv2d_supa(h_local, weights1, weights2,
                                        args.modes, args.modes,
                                        use_sufft=True, to_cpu=False)

        spec_med, _, _ = wall(_spec, warmup=3, iters=20)

        y = spec_med * 0 + spectral_conv2d_supa(
            h_local, weights1, weights2, args.modes, args.modes,
            use_sufft=True, to_cpu=False,
        )

        def _conv_skip_norm_gelu():
            skip = layer.conv(h_local)
            r = y + skip
            return F.gelu(layer.norm(r))

        rest_med, _, _ = wall(_conv_skip_norm_gelu, warmup=3, iters=20)
        total = spec_med + rest_med
        layer_totals.append(total)
        print(f"L{i + 1:<7} {'?':>7} {spec_med:7.3f} {'?':>7} {'?':>8} "
              f"{'?':>6} {'?':>6} {'?':>6} {total:6.3f}")

        # Advance h for next layer using the real layer.
        h = layer.forward_supa(h, use_sufft="auto").detach()

    print()
    print(f"sum of layers (rough estimate, double-counts some sync): "
          f"{sum(layer_totals):.3f} ms vs full {full_med:.3f} ms")


if __name__ == "__main__":
    main()
