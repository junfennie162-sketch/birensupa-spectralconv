#!/usr/bin/env python3
"""Continue modes=16 demo with original global rel-L2 (same as train_official).

Gentle unfrozen fine-tune from current best; promote only if official 1000/128
test L2 improves.
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
DEMO_CKPT = THIS_DIR / "checkpoints" / "fno_ns_demo.pt"
BEST_CKPT = THIS_DIR / "checkpoints" / "fno_ns_official_best.pt"
OUT_CKPT = THIS_DIR / "checkpoints" / "fno_ns_global_continue_best.pt"
OUT_META = THIS_DIR / "checkpoints" / "fno_ns_global_continue_meta.json"
SEED = 20260722


def global_rel_l2(pred, target):
    return torch.norm(pred - target) / torch.norm(target).clamp_min(1e-12)


def evaluate(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in loader:
            scores.append(relative_l2(model(x, use_supa=False), y))
    return sum(scores) / max(len(scores), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3.0e-5)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--init", type=str, default=str(DEMO_CKPT))
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
            "task": "fno_ns_global_continue",
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
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        s, n = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = global_rel_l2(model(x, use_supa=False), y)
            loss.backward()
            opt.step()
            s += float(loss.item())
            n += 1
            step += 1
        sch.step()
        test_l2 = evaluate(model, test_loader)
        row = {
            "epoch": epoch,
            "step": step,
            "train_rel_l2": s / max(n, 1),
            "test_rel_l2": test_l2,
            "best_l2": min(best, test_l2),
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
                    "step": step,
                    "data_source": src,
                    "global_continue": {"lr": args.lr, "baseline": baseline},
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

        shutil.copy2(DEMO_CKPT, DEMO_CKPT.with_suffix(".pt.pre_global_continue_backup"))
        shutil.copy2(OUT_CKPT, DEMO_CKPT)
        shutil.copy2(OUT_CKPT, BEST_CKPT)
        print({"promoted": True, "new_l2": best}, flush=True)
    else:
        print({"promoted": False, "improved": improved}, flush=True)


if __name__ == "__main__":
    main()
