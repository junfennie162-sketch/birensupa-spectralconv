#!/usr/bin/env python3
"""Continue official-split training from a checkpoint with matched per-sample L2 loss.

Uses n_train=1000, n_test=128, bs=16. Saves best-on-test to
checkpoints/fno_ns_continue_best.pt. Optionally --promote into demo/official_best.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d, count_parameters
from test_forward import relative_l2

THIS_DIR = Path(__file__).resolve().parent
OUT_CKPT = THIS_DIR / "checkpoints" / "fno_ns_continue_best.pt"
DEMO_CKPT = THIS_DIR / "checkpoints" / "fno_ns_demo.pt"
BEST_CKPT = THIS_DIR / "checkpoints" / "fno_ns_official_best.pt"
SEED = 20260722


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--init", type=str, required=True)
    p.add_argument("--promote", action="store_true")
    args = p.parse_args()

    torch.manual_seed(SEED)
    data, src = load_or_build_ns_like(
        n_samples=args.n_train + args.n_test,
        resolution=64,
        n_times=30,
        seed=SEED,
        version="v2",
    )
    tr, te = split_train_test(data, args.n_train, args.n_test, seed=SEED)
    train_loader = DataLoader(
        SequenceVorticityDataset(tr, 10, 1), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        SequenceVorticityDataset(te, 10, 1), batch_size=args.batch_size, shuffle=False
    )

    model = FNO2d(modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1)
    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    baseline = evaluate(model, test_loader)
    print(
        {
            "task": "fno_ns_continue",
            "init": args.init,
            "baseline_test_l2": baseline,
            "epochs": args.epochs,
            "lr": args.lr,
            "params": count_parameters(model),
            "steps_per_epoch": len(train_loader),
        },
        flush=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
                    "continue": {"lr": args.lr, "baseline": baseline, "from": args.init},
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
    (THIS_DIR / "checkpoints" / "fno_ns_continue_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(meta, flush=True)
    if improved and args.promote:
        import shutil

        shutil.copy2(DEMO_CKPT, DEMO_CKPT.with_suffix(".pt.pre_continue_backup"))
        shutil.copy2(OUT_CKPT, DEMO_CKPT)
        shutil.copy2(OUT_CKPT, BEST_CKPT)
        print({"promoted": True, "new_l2": best}, flush=True)
    else:
        print({"promoted": False, "improved": improved}, flush=True)


if __name__ == "__main__":
    main()
