#!/usr/bin/env python3
"""f side accuracy 5-case battery, matching n's table (8/32/64/128/256)."""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, "/workspace/ai4s-f/submission/spectral_conv")
import torch, torch_br  # noqa: F401
from reference_pytorch import make_random_weights, spectral_conv2d
from spectral_conv_ops import spectral_conv2d_supa


def rel(a, b):
    d = float((a - b).norm())
    r = float(a.norm())
    return d / r if r > 1e-12 else d


def main():
    seeds = 100
    rows = []
    cases = [
        ("tiny_8x8",       2,  2,  3,   8,   8,  2,  2),
        ("small_32x32",    2,  4,  4,  32,  32,  8,  8),
        ("target_64x64",   2,  4,  4,  64,  64, 12, 12),
        ("official_128",   4, 32, 64, 128, 128, 16, 16),
        ("official_256_fused", 4, 32, 64, 256, 256, 16, 16),
    ]
    for name, B, Cin, Cout, H, W, m1, m2 in cases:
        seed = seeds
        seeds += 50
        x = torch.empty((B, Cin, H, W), dtype=torch.float32).uniform_(-0.5, 0.5)
        w1 = make_random_weights(Cin, Cout, m1, m2, seed + 17)
        w2 = make_random_weights(Cin, Cout, m1, m2, seed + 23)
        torch.manual_seed(seed)
        y_ref = spectral_conv2d(x, w1, m1, m2, weights2=w2)
        y_act = spectral_conv2d_supa(x, w1, w2, m1, m2, use_sufft="auto")
        rr = rel(y_ref, y_act)
        ok = rr <= 1e-4
        rows.append({"case": name, "shape": f"B{B}_Cin{Cin}_Cout{Cout}_{H}x{W}",
                     "modes": f"{m1}x{m2}", "rel": rr, "ok": ok})
        print({"case": name, "rel": rr, "ok": ok})
    worst = max(r["rel"] for r in rows)
    Path("/tmp/ai4s-rebench-2026-07-24/f_accuracy_5case.json").write_text(
        json.dumps({"rows": rows, "worst_rel": worst, "all_ok": all(r["ok"] for r in rows)}, indent=2)
    )


if __name__ == "__main__":
    main()
