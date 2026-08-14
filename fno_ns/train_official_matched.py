#!/usr/bin/env python3
"""Official-gate FNO train with eval-matched per-sample relative L2 loss.

- Split: n_train=1000, n_test=128, bs=16 (same as contest gate)
- Loss: mean per-sample rel-L2 (matches evaluate)
- Writes best to checkpoints/fno_ns_matched_best.pt (does NOT touch demo
  unless --promote and best < current demo on this split)
- Optional --init to warm-start from an existing ckpt
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
OUT_CKPT = THIS_DIR / "checkpoints" / "fno_ns_matched_best.pt"
DEMO_CKPT = THIS_DIR / "checkpoints" / "fno_ns_demo.pt"
BEST_CKPT = THIS_DIR / "checkpoints" / "fno_ns_official_best.pt"
SEED = 20260722


def batch_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = torch.norm(pred - target, dim=(-2, -1))
    ref = torch.norm(target, dim=(-2, -1)).clamp_min(1.0e-12)
    return (diff / ref).mean()


def evaluate(model: FNO2d, loader: DataLoader) -> float:
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in loader:
            scores.append(relative_l2(model(x, use_supa=False), y))
    return sum(scores) / max(len(scores), 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--init", type=str, default="")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data, src = load_or_build_ns_like(
        n_samples=args.n_train + args.n_test,
        resolution=64,
        n_times=30,
        seed=args.seed,
        version="v2",
    )
    tr, te = split_train_test(data, args.n_train, args.n_test, seed=args.seed)
    train_loader = DataLoader(
        SequenceVorticityDataset(tr, 10, 1), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        SequenceVorticityDataset(te, 10, 1), batch_size=args.batch_size, shuffle=False
    )

    model = FNO2d(modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1)
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print({"loaded_init": args.init}, flush=True)

    # Gate: must beat current demo on this split to promote later
    demo_l2 = float("inf")
    if DEMO_CKPT.exists():
        demo = FNO2d(modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1)
        demo.load_state_dict(
            torch.load(DEMO_CKPT, map_location="cpu", weights_only=False)["model"]
        )
        demo_l2 = evaluate(demo, test_loader)

    start_l2 = evaluate(model, test_loader)
    steps_per_epoch = len(train_loader)
    print(
        {
            "task": "fno_ns_matched_official",
            "data_source": src,
            "n_train": args.n_train,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch * args.epochs,
            "start_test_l2": start_l2,
            "demo_test_l2": demo_l2,
            "params": count_parameters(model),
        },
        flush=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 3))
    best = start_l2
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    hist = []
    t0 = time.time()
    step = 0

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
                    "matched": {"lr": args.lr, "init": args.init or None},
                },
                OUT_CKPT,
            )
            row["saved_best"] = True
        hist.append(row)
        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == args.epochs
            or row.get("saved_best")
            or step in (500, 2000, 6000)
        ):
            print(row, flush=True)
            if step in (500, 2000, 6000) or step == steps_per_epoch * args.epochs:
                tag = "晋阶" if step <= 500 else ("有效" if step <= 2000 else "推荐")
                print({"gate": tag, "step": step, "best_l2": best}, flush=True)

    if not OUT_CKPT.exists():
        torch.save({"model": best_state, "test_l2": best, "epoch": args.epochs}, OUT_CKPT)

    improved_vs_demo = best < demo_l2 - 1e-9
    meta = {
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": "1000/128",
        "start_test_l2": start_l2,
        "demo_test_l2": demo_l2,
        "best_test_l2": best,
        "improved_vs_demo": improved_vs_demo,
        "delta_vs_demo": (demo_l2 - best) if demo_l2 < float("inf") else None,
        "epochs": args.epochs,
        "lr": args.lr,
        "total_steps": step,
        "elapsed_sec": round(time.time() - t0, 1),
        "history": hist,
        "checkpoint": str(OUT_CKPT),
    }
    (THIS_DIR / "checkpoints" / "fno_ns_matched_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(meta, flush=True)

    if improved_vs_demo and args.promote:
        import shutil

        shutil.copy2(DEMO_CKPT, DEMO_CKPT.with_suffix(".pt.pre_matched_backup"))
        shutil.copy2(OUT_CKPT, DEMO_CKPT)
        shutil.copy2(OUT_CKPT, BEST_CKPT)
        print({"promoted": True, "new_l2": best}, flush=True)
    else:
        print(
            {
                "promoted": False,
                "reason": (
                    "not better than demo"
                    if not improved_vs_demo
                    else "pass --promote"
                ),
            },
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
