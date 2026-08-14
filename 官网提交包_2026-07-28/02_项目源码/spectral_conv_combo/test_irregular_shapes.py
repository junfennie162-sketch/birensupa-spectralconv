"""Stress test on irregular (non power-of-2) resolutions.

Sanity check that fused/v1 paths handle 40/48/72/96/160/192 without
correctness regression or memory blow-up. Mirrors test_accuracy.py contract
but covers shapes that the official §3.2 set doesn't.
"""
from __future__ import annotations

import sys
import torch
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import spectral_conv_ops as ops  # noqa: E402
import torch_br  # noqa: F401

from reference_pytorch import SpectralConv2d  # noqa: E402


SHAPES = [
    (4, 32, 64, 40),    # 40 wide
    (4, 32, 40, 64),    # 40 tall
    (4, 32, 48, 48),    # 48x48
    (4, 32, 72, 72),    # 72x72
    (4, 32, 96, 96),    # 96x96
    (4, 32, 160, 160),  # 160x160 (5*32)
    (4, 32, 192, 192),  # 192x192
    (4, 32, 256, 64),   # non-square
    (2, 16, 100, 100),  # arbitrary
]
MODES = 16


def _peak_mb() -> float:
    try:
        return float(torch_br.supa.max_memory_allocated()) / (1024 * 1024)
    except Exception:
        return -1.0


def _reset_peak() -> None:
    try:
        torch_br.supa.reset_peak_memory_stats()
    except Exception:
        pass


def _run(shape, modes=MODES):
    B, Cin, Cout, H, W = shape[0], shape[1], 64, shape[2], shape[3]
    M = min(modes, min(H, W) // 2)
    if M < 4:
        return {"shape": shape, "ok": True, "skipped": "modes too small"}
    torch.manual_seed(0)
    x = torch.randn(B, Cin, H, W, dtype=torch.float32)
    scale = 1.0 / (Cin * Cout)
    w1 = (scale * torch.rand(Cin, Cout, M, M, dtype=torch.cfloat)).contiguous()
    w2 = (scale * torch.rand(Cin, Cout, M, M, dtype=torch.cfloat)).contiguous()

    # Reference (CPU)
    ref_layer = SpectralConv2d(Cin, Cout, M, M)
    with torch.no_grad():
        ref_layer.weights1.copy_(w1)
        ref_layer.weights2.copy_(w2)
    ref = ref_layer(x)
    # SUPA
    ops.clear_weight_supa_cache()
    ops.warmup_spectral_plans(H, W, Cin, Cout, M, M, B)
    _reset_peak()
    try:
        y = ops.spectral_conv2d_supa(x, w1, w2, M, M, use_sufft="auto")
    except Exception as e:
        return {"shape": shape, "ok": False, "error": str(e)[:200]}
    torch_br.supa.synchronize()
    peak = _peak_mb()

    diff = (y - ref).abs()
    rel = diff.max().item() / max(ref.abs().max().item(), 1e-9)
    ok = rel < 1e-4
    return {
        "shape": f"{H}x{W}",
        "B_Cin_Cout": f"{B}_{Cin}_{Cout}",
        "rel": rel,
        "threshold": 1e-4,
        "ok": ok,
        "peak_mb": peak,
        "path": "auto",
    }


def main() -> int:
    print(f"{'shape':<10} {'config':<14} {'rel':>10} {'peak_mb':>8} {'ok':>5}")
    print("-" * 50)
    worst_rel = 0.0
    worst_peak = 0.0
    failures = 0
    for s in SHAPES:
        r = _run(s)
        if r.get("skipped"):
            print(f"{r['shape']:<10} {'-':<14} {'skip':>10} {'':>8} {'':>5}")
            continue
        rel = r["rel"]
        worst_rel = max(worst_rel, rel)
        worst_peak = max(worst_peak, r["peak_mb"])
        if not r["ok"]:
            failures += 1
        print(f"{r['shape']:<10} {r['B_Cin_Cout']:<14} "
              f"{rel:>10.3e} {r['peak_mb']:>8.1f} {'OK' if r['ok'] else 'FAIL':>5}")
    print("-" * 50)
    print(f"worst_rel={worst_rel:.3e}  worst_peak={worst_peak:.1f} MB  "
          f"failures={failures}/{len(SHAPES)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())