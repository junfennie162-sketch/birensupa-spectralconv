#!/usr/bin/env python3
"""From-scratch official-split FNO with larger width (default 48).

Same global rel-L2 recipe as train_official.py / modes retrain.
Promote only if test L2 beats current modes=16 width=32 demo on 1000/128.
Note: wider ckpt cannot drop into demo without updating loaders/defaults.
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
SEED = 20260722


def global_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred - target) / torch.norm(target).clamp_min(1.0e-12)


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
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    out_ckpt = THIS_DIR / "checkpoints" / f"fno_ns_w{args.width}_best.pt"
    out_meta = THIS_DIR / "checkpoints" / f"fno_ns_w{args.width}_meta.json"

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

    demo_l2 = float("inf")
    if DEMO_CKPT.exists():
        demo = FNO2d(modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1)
        demo.load_state_dict(
            torch.load(DEMO_CKPT, map_location="cpu", weights_only=False)["model"]
        )
        demo_l2 = evaluate(demo, test_loader)

    model = FNO2d(
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        n_layers=4,
        in_channels=10,
        out_channels=1,
    )
    start_l2 = evaluate(model, test_loader)
    print(
        {
            "task": "fno_ns_width_retrain",
            "modes": args.modes,
            "width": args.width,
            "params": count_parameters(model),
            "data_source": src,
            "epochs": args.epochs,
            "lr": args.lr,
            "start_test_l2": start_l2,
            "demo_test_l2_w32": demo_l2,
            "steps_per_epoch": len(train_loader),
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
                    "modes": args.modes,
                    "width": args.width,
                    "data_source": src,
                },
                out_ckpt,
            )
            row["saved_best"] = True
        hist.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs or row.get("saved_best"):
            print(row, flush=True)

    if not out_ckpt.exists():
        torch.save(
            {"model": best_state, "test_l2": best, "modes": args.modes, "width": args.width},
            out_ckpt,
        )

    improved = best < demo_l2 - 1e-9
    meta = {
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modes": args.modes,
        "width": args.width,
        "demo_test_l2_w32": demo_l2,
        "best_test_l2": best,
        "improved_vs_demo": improved,
        "delta_vs_demo": (demo_l2 - best) if demo_l2 < float("inf") else None,
        "epochs": args.epochs,
        "lr": args.lr,
        "elapsed_sec": round(time.time() - t0, 1),
        "history": hist,
        "checkpoint": str(out_ckpt),
        "note": "width!=32 changes all layer shapes; demo loaders must match before promote",
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n")
    print(meta, flush=True)

    if improved and args.promote:
        import shutil

        shutil.copy2(DEMO_CKPT, DEMO_CKPT.with_suffix(f".pt.pre_w{args.width}_backup"))
        shutil.copy2(out_ckpt, DEMO_CKPT)
        shutil.copy2(out_ckpt, BEST_CKPT)
        print({"promoted": True, "new_l2": best, "width": args.width}, flush=True)
    else:
        print({"promoted": False, "improved_vs_demo": improved}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
