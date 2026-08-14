#!/usr/bin/env python3
"""Device-resident SUPA chain assertion test.

Runs `FNO2d.forward_supa_chain` at the official 64x64 config and asserts that
every intermediate tensor stays on SUPA; only the final output is on CPU.
A regression here means per-layer D2H crept back into the chain.

Usage:
    cd /workspace/ai4s-f/submission/fno_ns
    python3 test_supa_chain.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure spectral_conv_ops is importable (sibling dir).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spectral_conv"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch_br  # noqa: F401
from model import FNO2d  # noqa: E402


def main() -> None:
    H = W = 64
    M = 16
    width = 32
    B = 4
    in_ch = 10
    n_layers = 4

    torch.manual_seed(0)
    m = FNO2d(
        modes1=M, modes2=M, width=width, n_layers=n_layers,
        in_channels=in_ch, out_channels=1,
    ).eval()
    m.prepare_supa_eval()
    x = torch.randn(B, in_ch, H, W, device="supa", dtype=torch.float32)

    # R5: `use_gn_substitute` was removed (it was double-norming under the
    # hood). Use the canonical IN path.
    out = m.forward_supa_chain(x, use_sufft="auto")

    # Replay manually to inspect intermediates.
    m.prepare_supa_eval()
    device = x.device
    grid = m.get_grid(x.shape, device)
    h = m.lift.to(device)(torch.cat([x, grid], dim=1))
    devices: list[tuple[str, str]] = [("post_lift", h.device.type)]
    for i, layer in enumerate(m.fourier_layers):
        y = layer.forward_supa(h, use_sufft="auto")
        devices.append((f"L{i + 1}.forward_supa_out", y.device.type))
        h = y
    devices.append(("post_chain_pre_project", h.device.type))
    proj = m.project.to(device)(h)
    devices.append(("project_out", proj.device.type))

    print({"devices": devices, "final_out_device": out.device.type})

    fail = []
    for tag, dev in devices[:-1]:  # all but final are SUPA
        if dev != "supa":
            fail.append((tag, dev))
    if out.device.type != "cpu":
        fail.append(("final_out", out.device.type))

    if fail:
        raise AssertionError(f"SUPA-chain broke at: {fail}")
    print({"task": "test_supa_chain", "ok": True,
           "intermediates": len(devices), "all_on_supa": True})


if __name__ == "__main__":
    main()
