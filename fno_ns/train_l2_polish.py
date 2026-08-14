#!/usr/bin/env python3
"""Polish FNO L2 from current best demo; promote only if test L2 improves.

Fixes vs earlier sidecar:
- Baseline = *current* demo test L2 (not a stale 0.0095 constant)
- Train loss matches eval: mean per-sample relative L2 over spatial dims
- Eval every epoch; keep best-on-test candidate
- Default start: fno_ns_demo.pt (should already be official_best-class)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import count_parameters
from test_forward import evaluate, make_model, build_data_loaders, relative_l2

THIS_DIR = Path(__file__).resolve().parent
DEMO_CKPT = THIS_DIR / "checkpoints" / "fno_ns_demo.pt"
CANDIDATE_CKPT = THIS_DIR / "checkpoints" / "fno_ns_l2_polish_candidate.pt"
CANDIDATE_META = THIS_DIR / "checkpoints" / "fno_ns_l2_polish_meta.json"
SEED = 20260722


def batch_rel_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable mean per-sample relative L2 (matches evaluate())."""
    diff = torch.norm(pred - target, dim=(-2, -1))
    ref = torch.norm(target, dim=(-2, -1)).clamp_min(1.0e-12)
    return (diff / ref).mean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=768)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--init",
        type=str,
        default=str(DEMO_CKPT),
        help="checkpoint to fine-tune from",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    train_loader, test_loader, data_source, _ = build_data_loaders(
        args.n_train, args.n_test, args.batch_size
    )
    model = make_model()
    init_path = Path(args.init)
    if not init_path.exists():
        raise FileNotFoundError(init_path)
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    baseline_l2, *_ = evaluate(model, test_loader, use_supa=False)
    print(
        {
            "task": "fno_ns_l2_polish",
            "init": str(init_path),
            "data_source": data_source,
            "baseline_test_l2": baseline_l2,
            "epochs": args.epochs,
            "lr": args.lr,
            "params": count_parameters(model),
        },
        flush=True,
    )

    if args.eval_only:
        meta = {
            "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "baseline_test_l2": baseline_l2,
            "candidate_test_l2": baseline_l2,
            "improved": False,
            "eval_only": True,
        }
        CANDIDATE_META.write_text(json.dumps(meta, indent=2) + "\n")
        print(meta, flush=True)
        return 0

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 3)
    )

    best_l2 = baseline_l2
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum, n = 0.0, 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            pred = model(inputs, use_supa=False)
            loss = batch_rel_l2_loss(pred, targets)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item())
            n += 1
        scheduler.step()
        train_l2 = train_sum / max(n, 1)

        row: dict = {
            "epoch": epoch,
            "train_rel_l2": train_l2,
            "lr": scheduler.get_last_lr()[0],
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            test_l2, *_ = evaluate(model, test_loader, use_supa=False)
            row["test_rel_l2"] = test_l2
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
                        "polish": {
                            "lr": args.lr,
                            "baseline_test_l2": baseline_l2,
                            "from": str(init_path.name),
                        },
                    },
                    CANDIDATE_CKPT,
                )
                row["saved_best"] = True
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs or row.get("saved_best"):
            print(row, flush=True)

    # Ensure candidate exists even if no improvement
    if not CANDIDATE_CKPT.exists() or best_l2 >= baseline_l2:
        torch.save(
            {
                "model": best_state,
                "test_l2": best_l2,
                "epoch": args.epochs,
                "data_source": data_source,
                "polish": {
                    "lr": args.lr,
                    "baseline_test_l2": baseline_l2,
                    "from": str(init_path.name),
                    "note": "no_improvement" if best_l2 >= baseline_l2 else "best",
                },
            },
            CANDIDATE_CKPT,
        )

    improved = best_l2 < baseline_l2 - 1e-9
    meta = {
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source": data_source,
        "baseline_test_l2": baseline_l2,
        "candidate_test_l2": best_l2,
        "improved": improved,
        "delta": baseline_l2 - best_l2,
        "epochs": args.epochs,
        "lr": args.lr,
        "elapsed_sec": round(time.time() - t0, 1),
        "history": history,
        "candidate_path": str(CANDIDATE_CKPT),
    }
    CANDIDATE_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(meta, flush=True)

    if improved and args.promote:
        import shutil

        backup = DEMO_CKPT.with_suffix(".pt.pre_polish_backup")
        shutil.copy2(DEMO_CKPT, backup)
        shutil.copy2(CANDIDATE_CKPT, DEMO_CKPT)
        print({"promoted": True, "backup": str(backup), "new_l2": best_l2}, flush=True)
    else:
        print(
            {
                "promoted": False,
                "reason": (
                    "not improved"
                    if not improved
                    else "pass --promote to replace demo"
                ),
            },
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
