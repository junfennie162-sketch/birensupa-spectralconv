#!/usr/bin/env python3
"""9-shape defensive validation for SpectralConv (f-side).

Mirrors `ai4s-n`'s `test_irregular_shapes.py` so both partners share the same
defensive coverage. Each shape runs the SUPA `auto` path against a CPU
reference and reports worst rel-error; threshold 1e-4.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch_br  # noqa: F401

from reference_pytorch import spectral_conv2d
from spectral_conv_ops import spectral_conv2d_supa

THRESH = 1.0e-4
# 9 defensive shapes — mostly non-power-of-2 / non-square.
SHAPES = [
    # (name,                  B, Cin, Cout, H,  W,  m1, m2)
    ("aspym_40x64",           4, 8,  16,   40, 64,  8,  8),
    ("square_72x72",          4, 8,  16,   72, 72,  8,  8),
    ("square_96x96",          4, 8,  16,   96, 96, 12, 12),
    ("square_100x100",        4, 8,  16,  100,100, 12, 12),
    ("square_160x160",        4, 8,  16,  160,160, 16, 16),
    ("square_192x192",        2, 4,   8,  192,192, 16, 16),
    ("wider_256x64",          4, 8,  16,  256, 64, 16,  8),
    ("almost_2pow_48x48",     4, 8,  16,   48, 48,  8,  8),
    ("non2pow_28x28",         4, 8,  16,   28, 28,  6,  6),
]


def rel(a, b):
    d = float((a - b).norm())
    r = float(a.norm())
    return d / r if r > 1e-12 else d


def main() -> None:
    rows = []
    for name, B, Cin, Cout, H, W, m1, m2 in SHAPES:
        torch.manual_seed(B * 31 + H + W)
        x = torch.randn(B, Cin, H, W, dtype=torch.float32)
        scale = 1.0 / (Cin * Cout)
        w1 = (scale * torch.rand(Cin, Cout, m1, m2, dtype=torch.cfloat)).contiguous()
        w2 = (scale * torch.rand(Cin, Cout, m1, m2, dtype=torch.cfloat)).contiguous()
        y_ref = spectral_conv2d(x, w1, m1, m2, weights2=w2)
        try:
            y_act = spectral_conv2d_supa(x, w1, w2, m1, m2, use_sufft="auto")
        except Exception as exc:
            rows.append((name, f"raised: {exc!r}", False))
            print(f"{name}: raised {exc!r}")
            continue
        r = rel(y_ref, y_act)
        ok = r <= THRESH
        rows.append((name, r, ok))
        print({"shape": name, "rel": r, "ok": ok, "threshold": THRESH})

    passed = sum(1 for _, _, o in rows if o)
    failed = [n for n, _, o in rows if not o]
    print({"task": "test_irregular_shapes", "passed": passed, "total": len(rows),
           "failed": failed})
    if failed:
        raise AssertionError(f"failed shapes: {failed}")


if __name__ == "__main__":
    main()
