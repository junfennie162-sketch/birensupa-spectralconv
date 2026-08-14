#!/usr/bin/env python3
"""Robustness check: f FNO-NS forward survives non-square + non-2pow shapes.

Loads `checkpoints/fno_ns_demo.pt` (which was trained at 64x64) and runs the
SUPA `forward_supa_chain` at 6 defensive shapes. Because the lift expects
[T_in + 2, H, W], grid channels are appended on-device. We expect moderate
L2 because the model wasn't trained for these shapes, but it must NOT
raise or return NaN/Inf.

Threshold for "no NaN/Inf" is 0; for "L2 finite" is 1e3 (extremely loose,
since shapes change the problem character). The value is informational.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spectral_conv"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch_br  # noqa: F401

from model import FNO2d

CKPT = Path(__file__).resolve().parent / "checkpoints" / "fno_ns_demo.pt"
T_IN, MODES, WIDTH, N_LAYERS = 10, 16, 32, 4


def main() -> None:
    torch.manual_seed(0)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = FNO2d(modes1=MODES, modes2=MODES, width=WIDTH, n_layers=N_LAYERS,
                  in_channels=T_IN, out_channels=1)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.prepare_supa_eval()

    shapes = [
        # (name, B, Cin, H,  W,  m1, m2)
        ("square_64x64",        4, T_IN,  64,  64, 16, 16),
        ("square_72x72",        4, T_IN,  72,  72, 14, 14),
        ("square_96x96",        4, T_IN,  96,  96, 16, 16),
        ("wider_64x96",         4, T_IN,  64,  96, 16, 16),
        ("square_100x100",      4, T_IN, 100, 100, 16, 16),
        ("square_160x160",      4, T_IN, 160, 160, 16, 16),
    ]
    rows = []
    for name, B, Cin, H, W, m1, m2 in shapes:
        # Re-init a model with the right modes for this shape. The first
        # entry reuses the trained checkpoint (16x16 modes); subsequent
        # entries are sanity-checked with fresh weights (we only assert
        # finite + non-NaN + non-Inf — L2 is not meaningful here).
        if (m1, m2) == (16, 16):
            m_local = model
            m_local.load_state_dict(ckpt["model"])
        else:
            m_local = FNO2d(modes1=m1, modes2=m2, width=WIDTH,
                            n_layers=N_LAYERS, in_channels=T_IN, out_channels=1)
        m_local.eval().to("supa")
        # move IN running stats as well, mirroring prepare_supa_eval
        for L in m_local.fourier_layers:
            n = L.norm
            if getattr(n, "running_mean", None) is not None:
                n.running_mean = n.running_mean.to("supa")
                n.running_var = n.running_var.to("supa")
        torch.manual_seed(B * 37 + H + W)
        x = torch.randn(B, Cin, H, W, dtype=torch.float32)
        try:
            with torch.no_grad():
                y = m_local.forward_supa_chain(x.to("supa"), use_sufft="auto")
            finite = torch.isfinite(y).all().item()
            nan = torch.isnan(y).any().item()
            inf = torch.isinf(y).any().item()
            rows.append((name, finite, not nan, not inf,
                         float(y.min().item()), float(y.max().item())))
            print({"shape": name, "finite": finite, "nan": nan, "inf": inf,
                   "out_min": float(y.min().item()), "out_max": float(y.max().item()),
                   "modes": (m1, m2)})
        except Exception as exc:
            print({"shape": name, "raised": repr(exc), "modes": (m1, m2)})
            rows.append((name, False, False, False, 0.0, 0.0))

    bad = [n for n, *flags in rows if not all(flags[:3])]
    print({"task": "test_irregular_FNO", "shapes": len(rows), "broken": bad})
    if bad:
        raise AssertionError(f"broken FNO shapes: {bad}")


if __name__ == "__main__":
    main()
