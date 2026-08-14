#!/usr/bin/env python3
"""Official-split polish with multi-window train augmentation.

Train: sliding windows over T=30 → many (x,y) pairs per sample.
Test: unchanged official first-window protocol (SequenceVorticityDataset).
Start from current best; freeze spectral by default; promote only if test improves.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d, count_parameters
from test_forward import relative_l2

THIS_DIR = Path(__file__).resolve().parent
BEST_CKPT = THIS_DIR / "checkpoints" / "fno_ns_official_best.pt"
DEMO_CKPT = THIS_DIR / "checkpoints" / "fno_ns_demo.pt"
OUT_CKPT = THIS_DIR / "checkpoints" / "fno_ns_multiwin_candidate.pt"
OUT_META = THIS_DIR / "checkpoints" / "fno_ns_multiwin_meta.json"
SEED = 20260722


class MultiWindowVorticityDataset(Dataset):
    """All valid (t_in → t_out) windows inside each trajectory."""

    def __init__(self, data: torch.Tensor, t_in: int = 10, t_out: int = 1):
        if data.dim() != 4:
            raise ValueError(f"expect [N,T,H,W], got {tuple(data.shape)}")
        self.data = data.contiguous()
        self.t_in = t_in
        self.t_out = t_out
        t = self.data.shape[1]
        self.n_win = t - t_in - t_out + 1
        if self.n_win <= 0:
            raise ValueError("not enough time steps for multi-window")
        self._len = self.data.shape[0] * self.n_win

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int):
        sample_index = index // self.n_win
        start = index % self.n_win
        sample = self.data[sample_index]
        x = sample[start : start + self.t_in]
        y = sample[start + self.t_in : start + self.t_in + self.t_out]
        return x, y


def batch_loss(pred, target):
    diff = torch.norm(pred - target, dim=(-2, -1))
    ref = torch.norm(target, dim=(-2, -1)).clamp_min(1e-12)
    return (diff / ref).mean()


def evaluate(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in loader:
            scores.append(relative_l2(model(x, use_supa=False), y))
    return sum(scores) / max(len(scores), 1)


def freeze_spectral(model: FNO2d) -> int:
    n = 0
    for name, p in model.named_parameters():
        if "spectral_conv" in name:
            p.requires_grad_(False)
            n += p.numel()
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--init", type=str, default=str(BEST_CKPT))
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    data, src = load_or_build_ns_like(
        n_samples=args.n_train + args.n_test,
        resolution=64,
        n_times=30,
        seed=SEED,
        version="v2",
    )
    train_data, test_data = split_train_test(
        data, args.n_train, args.n_test, seed=SEED
    )
    train_loader = DataLoader(
        MultiWindowVorticityDataset(train_data, 10, 1),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        SequenceVorticityDataset(test_data, 10, 1),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = FNO2d(modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1)
    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    frozen = 0 if args.no_freeze else freeze_spectral(model)
    baseline = evaluate(model, test_loader)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        {
            "task": "fno_ns_multiwin_polish",
            "init": args.init,
            "data_source": src,
            "train_windows": len(train_loader.dataset),
            "steps_per_epoch": len(train_loader),
            "baseline_test_l2": baseline,
            "frozen_spectral": not args.no_freeze,
            "frozen_params": frozen,
            "trainable_params": sum(p.numel() for p in trainable),
            "epochs": args.epochs,
            "lr": args.lr,
            "params": count_parameters(model),
        },
        flush=True,
    )

    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 3))
    best = baseline
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    hist = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        s, n = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = batch_loss(model(x, use_supa=False), y)
            loss.backward()
            opt.step()
            s += float(loss.item())
            n += 1
        sch.step()
        test_l2 = evaluate(model, test_loader)
        row = {
            "epoch": epoch,
            "train_rel_l2": s / max(n, 1),
            "test_rel_l2": test_l2,
            "lr": sch.get_last_lr()[0],
        }
        if test_l2 < best:
            best = test_l2
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model": best_state,
                    "test_l2": best,
                    "epoch": epoch,
                    "data_source": src,
                    "multiwin": {
                        "lr": args.lr,
                        "baseline": baseline,
                        "frozen_spectral": not args.no_freeze,
                    },
                },
                OUT_CKPT,
            )
            row["saved_best"] = True
        hist.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs or row.get("saved_best"):
            print(row, flush=True)

    if not OUT_CKPT.exists():
        torch.save({"model": best_state, "test_l2": best}, OUT_CKPT)

    improved = best < baseline - 1e-9
    meta = {
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline_test_l2": baseline,
        "best_test_l2": best,
        "improved": improved,
        "delta": baseline - best,
        "epochs": args.epochs,
        "lr": args.lr,
        "elapsed_sec": round(time.time() - t0, 1),
        "history": hist,
        "checkpoint": str(OUT_CKPT),
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(meta, flush=True)

    if improved and args.promote:
        import shutil

        shutil.copy2(DEMO_CKPT, DEMO_CKPT.with_suffix(".pt.pre_multiwin_backup"))
        shutil.copy2(OUT_CKPT, DEMO_CKPT)
        shutil.copy2(OUT_CKPT, BEST_CKPT)
        print({"promoted": True, "new_l2": best}, flush=True)
    else:
        print({"promoted": False, "improved": improved}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
