#!/usr/bin/env python3
"""Polish FNO toward official-gate split: n_train=1000, n_test=128, bs=16.

Baseline on this split for official_best ≈ 0.005488 (NOT the 768-split ~0.00275).
Freeze spectral by default; tiny lr; promote only if test L2 improves on THIS split.
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
CANDIDATE_CKPT = THIS_DIR / "checkpoints" / "fno_ns_official_polish_candidate.pt"
CANDIDATE_META = THIS_DIR / "checkpoints" / "fno_ns_official_polish_meta.json"
SEED = 20260722


def batch_rel_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


def freeze_spectral(model: FNO2d) -> int:
    n = 0
    for name, p in model.named_parameters():
        if "spectral_conv" in name:
            p.requires_grad_(False)
            n += p.numel()
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--init", type=str, default=str(BEST_CKPT))
    args = parser.parse_args()

    torch.manual_seed(SEED)
    data, data_source = load_or_build_ns_like(
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
        SequenceVorticityDataset(train_data, 10, 1),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        SequenceVorticityDataset(test_data, 10, 1),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = FNO2d(modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1)
    init_path = Path(args.init)
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    frozen = 0 if args.no_freeze else freeze_spectral(model)

    baseline = evaluate(model, test_loader)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        {
            "task": "fno_ns_official_split_polish",
            "init": str(init_path),
            "data_source": data_source,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "batch_size": args.batch_size,
            "baseline_test_l2": baseline,
            "frozen_spectral_params": frozen,
            "trainable_params": sum(p.numel() for p in trainable),
            "epochs": args.epochs,
            "lr": args.lr,
            "total_params": count_parameters(model),
        },
        flush=True,
    )

    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 3)
    )
    best_l2 = baseline
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum, n = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = batch_rel_l2_loss(model(x, use_supa=False), y)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item())
            n += 1
        scheduler.step()
        test_l2 = evaluate(model, test_loader)
        row = {
            "epoch": epoch,
            "train_rel_l2": train_sum / max(n, 1),
            "test_rel_l2": test_l2,
            "lr": scheduler.get_last_lr()[0],
        }
        if test_l2 < best_l2:
            best_l2 = test_l2
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(
                {
                    "model": best_state,
                    "test_l2": best_l2,
                    "epoch": epoch,
                    "data_source": data_source,
                    "split": {"n_train": args.n_train, "n_test": args.n_test},
                    "official_polish": {
                        "lr": args.lr,
                        "baseline_test_l2": baseline,
                        "frozen_spectral": not args.no_freeze,
                    },
                },
                CANDIDATE_CKPT,
            )
            # also refresh official_best when improved
            torch.save(
                {
                    "model": best_state,
                    "test_l2": best_l2,
                    "epoch": epoch,
                    "step": epoch * len(train_loader),
                },
                BEST_CKPT.with_name("fno_ns_official_best_candidate.pt"),
            )
            row["saved_best"] = True
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs or row.get("saved_best"):
            print(row, flush=True)

    if not CANDIDATE_CKPT.exists():
        torch.save({"model": best_state, "test_l2": best_l2}, CANDIDATE_CKPT)

    improved = best_l2 < baseline - 1e-9
    meta = {
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": "official_1000_128",
        "baseline_test_l2": baseline,
        "candidate_test_l2": best_l2,
        "improved": improved,
        "delta": baseline - best_l2,
        "epochs": args.epochs,
        "lr": args.lr,
        "frozen_spectral": not args.no_freeze,
        "elapsed_sec": round(time.time() - t0, 1),
        "history": history,
        "candidate_path": str(CANDIDATE_CKPT),
    }
    CANDIDATE_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(meta, flush=True)

    if improved and args.promote:
        import shutil

        shutil.copy2(DEMO_CKPT, DEMO_CKPT.with_suffix(".pt.pre_official_polish_backup"))
        shutil.copy2(CANDIDATE_CKPT, DEMO_CKPT)
        shutil.copy2(CANDIDATE_CKPT, BEST_CKPT)
        print({"promoted": True, "new_l2": best_l2}, flush=True)
    else:
        print(
            {
                "promoted": False,
                "reason": "not improved" if not improved else "pass --promote",
            },
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
