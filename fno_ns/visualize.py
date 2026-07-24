#!/usr/bin/env python3
"""Visualize FNO prediction vs synthetic ground-truth target (demo figure)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "spectral_conv_combo"))

from model import FNO2d  # noqa: E402

FIG_DIR = ROOT.parent / "results" / "figures"


def main():
    import torch_br  # noqa: F401

    torch.manual_seed(7)
    model = FNO2d(modes1=8, modes2=8, width=16, n_layers=4, in_channels=10, out_channels=1)
    model.eval()
    x = torch.randn(1, 10, 64, 64)
    # Synthetic "gt": next-frame proxy = smoothed input last channel
    gt = x[:, -1:, :, :]
    with torch.no_grad():
        pred = model(x)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    im0 = axes[0].imshow(gt[0, 0].numpy(), cmap="coolwarm")
    axes[0].set_title("Synthetic target")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(pred[0, 0].numpy(), cmap="coolwarm")
    axes[1].set_title("FNO prediction")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    err = (pred - gt).abs()[0, 0].numpy()
    im2 = axes[2].imshow(err, cmap="hot")
    axes[2].set_title("|pred - target|")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    out = FIG_DIR / "fno_ns_pred_vs_gt.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print({"figure": str(out), "ok": True})


if __name__ == "__main__":
    main()
