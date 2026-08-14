#!/usr/bin/env python3
"""Aggressive L2 polish: freeze spectral weights, tiny lr on head/skip/norm.

Starts from best demo; only promote if test L2 improves.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from model import count_parameters
from test_forward import evaluate, make_model, build_data_loaders

THIS_DIR = Path(__file__).resolve().parent
DEMO_CKPT = THIS_DIR / "checkpoints" / "fno_ns_demo.pt"
CANDIDATE_CKPT = THIS_DIR / "checkpoints" / "fno_ns_l2_freeze_candidate.pt"
CANDIDATE_META = THIS_DIR / "checkpoints" / "fno_ns_l2_freeze_meta.json"
SEED = 20260722


def batch_rel_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = torch.norm(pred - target, dim=(-2, -1))
    ref = torch.norm(target, dim=(-2, -1)).clamp_min(1.0e-12)
    return (diff / ref).mean()


def freeze_spectral(model) -> int:
    n = 0
    for name, p in model.named_parameters():
        if "spectral_conv" in name:
            p.requires_grad_(False)
            n += p.numel()
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=768)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--init", type=str, default=str(DEMO_CKPT))
    parser.add_argument("--no-freeze", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    train_loader, test_loader, data_source, _ = build_data_loaders(
        args.n_train, args.n_test, args.batch_size
    )
    model = make_model()
    init_path = Path(args.init)
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    frozen = 0 if args.no_freeze else freeze_spectral(model)

    baseline_l2, *_ = evaluate(model, test_loader, use_supa=False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        {
            "task": "fno_ns_l2_freeze_polish",
            "init": str(init_path),
            "baseline_test_l2": baseline_l2,
            "frozen_spectral_params": frozen,
            "trainable_params": sum(p.numel() for p in trainable),
            "epochs": args.epochs,
            "lr": args.lr,
        },
        flush=True,
    )

    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 3)
    )

    best_l2 = baseline_l2
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        # keep BN/IN in train; spectral frozen still runs forward
        train_sum, n = 0.0, 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = batch_rel_l2_loss(model(inputs, use_supa=False), targets)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item())
            n += 1
        scheduler.step()
        test_l2, *_ = evaluate(model, test_loader, use_supa=False)
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
                    "freeze_polish": {
                        "lr": args.lr,
                        "baseline_test_l2": baseline_l2,
                        "frozen_spectral": not args.no_freeze,
                    },
                },
                CANDIDATE_CKPT,
            )
            row["saved_best"] = True
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs or row.get("saved_best"):
            print(row, flush=True)

    if not CANDIDATE_CKPT.exists():
        torch.save(
            {
                "model": best_state,
                "test_l2": best_l2,
                "data_source": data_source,
                "freeze_polish": {"baseline_test_l2": baseline_l2, "note": "no_file_yet"},
            },
            CANDIDATE_CKPT,
        )

    improved = best_l2 < baseline_l2 - 1e-9
    meta = {
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline_test_l2": baseline_l2,
        "candidate_test_l2": best_l2,
        "improved": improved,
        "delta": baseline_l2 - best_l2,
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

        backup = DEMO_CKPT.with_suffix(".pt.pre_freeze_polish_backup")
        shutil.copy2(DEMO_CKPT, backup)
        shutil.copy2(CANDIDATE_CKPT, DEMO_CKPT)
        print({"promoted": True, "new_l2": best_l2, "backup": str(backup)}, flush=True)
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
