#!/usr/bin/env python3
"""Optional short train on synthetic data + relative L2 report (加分项)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "spectral_conv"))

from model import FNO2d  # noqa: E402


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).norm() / (target.norm() + 1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--t_in", type=int, default=4)
    args = parser.parse_args()

    import torch_br  # noqa: F401

    torch.manual_seed(0)
    model = FNO2d(
        modes1=4,
        modes2=4,
        width=8,
        n_layers=4,
        in_channels=args.t_in,
        out_channels=1,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(args.epochs):
        x = torch.randn(args.batch_size, args.t_in, args.resolution, args.resolution)
        # Toy target: mean of input channels
        y = x.mean(dim=1, keepdim=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        print({"epoch": epoch + 1, "mse": losses[-1], "rel_l2": relative_l2(pred.detach(), y)})

    ckpt = ROOT / "checkpoint_synth.pt"
    torch.save({"model": model.state_dict(), "losses": losses}, ckpt)
    report = {
        "task": "fno_ns_train_synth",
        "ok": True,
        "epochs": args.epochs,
        "final_mse": losses[-1],
        "checkpoint": str(ckpt),
        "note": "Synthetic toy task for pipeline demo; replace with NS64 data for formal L2",
    }
    print(report)
    (ROOT.parent / "results" / "run_logs").mkdir(parents=True, exist_ok=True)
    (ROOT.parent / "results" / "run_logs" / "fno_train_synth.md").write_text(
        "# FNO synthetic train\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n"
    )


if __name__ == "__main__":
    main()
