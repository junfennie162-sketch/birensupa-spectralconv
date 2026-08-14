#!/usr/bin/env python3
"""R7 L2 sidecar: careful fine-tune from demo ckpt; replace only if test L2 improves.

Does NOT overwrite fno_ns_demo.pt unless --promote and candidate beats baseline.
Data remains generated_ns_like_v2 (not public NS64).
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d, count_parameters
from test_forward import relative_l2, evaluate, make_model, build_data_loaders

THIS_DIR = Path(__file__).resolve().parent
DEMO_CKPT = THIS_DIR / "checkpoints" / "fno_ns_demo.pt"
CANDIDATE_CKPT = THIS_DIR / "checkpoints" / "fno_ns_r7_candidate.pt"
CANDIDATE_META = THIS_DIR / "checkpoints" / "fno_ns_r7_candidate_meta.json"
BASELINE_L2 = 0.00951623497530818
SEED = 20260722


def fine_tune(
    model: FNO2d,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    *,
    prior_history: list[float] | None = None,
    data_source: str = "generated_ns_like_v2",
    save_every: int = 10,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[float] = []
    prior = list(prior_history or [])
    model.train()
    for epoch_index in range(epochs):
        total = 0.0
        n = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            pred = model(inputs, use_supa=False)
            loss = torch.norm(pred - targets) / torch.norm(targets).clamp_min(1e-12)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n += 1
        scheduler.step()
        epoch_l2 = total / max(n, 1)
        history.append(epoch_l2)
        done = epoch_index + 1
        if done == 1 or done % 5 == 0 or done == epochs:
            print(
                {"epoch": done, "train_relative_l2": epoch_l2, "lr": scheduler.get_last_lr()[0]},
                flush=True,
            )
        if save_every > 0 and (done % save_every == 0 or done == epochs):
            CANDIDATE_CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "history": prior + history,
                    "data_source": data_source,
                    "r7_sidecar": {
                        "epochs_done": done,
                        "epochs_target": epochs,
                        "lr": lr,
                        "from_checkpoint": str(DEMO_CKPT.name),
                    },
                },
                CANDIDATE_CKPT,
            )
            print({"checkpoint_saved": str(CANDIDATE_CKPT), "epochs_done": done}, flush=True)
    return history


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=768)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--promote", action="store_true", help="replace demo ckpt if better")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    train_loader, test_loader, data_source, _data = build_data_loaders(
        args.n_train, args.n_test, args.batch_size
    )
    model = make_model()
    if not DEMO_CKPT.exists():
        raise FileNotFoundError(DEMO_CKPT)
    ckpt = torch.load(DEMO_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    prior_history = list(ckpt.get("history", []))

    if not args.eval_only:
        print(
            {
                "task": "fno_ns_r7_sidecar",
                "data_source": data_source,
                "epochs": args.epochs,
                "lr": args.lr,
                "baseline_l2": BASELINE_L2,
                "note": "self-generated NS-like; not public NS64",
            }
        )
        new_hist = fine_tune(
            model,
            train_loader,
            args.epochs,
            args.lr,
            args.weight_decay,
            prior_history=prior_history,
            data_source=data_source,
            save_every=10,
        )
        history = prior_history + new_hist
    else:
        cand = torch.load(CANDIDATE_CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(cand["model"])
        history = list(cand.get("history", []))

    # Compare against *current* demo on disk when present (avoid stale BASELINE).
    demo_l2 = BASELINE_L2
    if DEMO_CKPT.exists():
        demo_model = make_model()
        demo_ckpt = torch.load(DEMO_CKPT, map_location="cpu", weights_only=False)
        demo_model.load_state_dict(demo_ckpt["model"])
        demo_l2, *_ = evaluate(demo_model, test_loader, use_supa=False)
    rel_torch, *_ = evaluate(model, test_loader, use_supa=False)
    improved = rel_torch < demo_l2
    meta = {
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source": data_source,
        "data_disclosure": "self-generated NS-like v2; not public NS64",
        "baseline_rel_l2": demo_l2,
        "legacy_baseline_rel_l2": BASELINE_L2,
        "candidate_rel_l2_torch": rel_torch,
        "improved": improved,
        "delta": demo_l2 - rel_torch,
        "epochs_added": args.epochs,
        "lr": args.lr,
        "parameters": count_parameters(model),
        "history_len": len(history),
        "candidate_path": str(CANDIDATE_CKPT),
    }
    CANDIDATE_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(meta)

    if improved and args.promote:
        import shutil

        backup = DEMO_CKPT.with_suffix(".pt.pre_r7_backup")
        shutil.copy2(DEMO_CKPT, backup)
        shutil.copy2(CANDIDATE_CKPT, DEMO_CKPT)
        print({"promoted": True, "backup": str(backup)})
    else:
        print(
            {
                "promoted": False,
                "reason": "not improved" if not improved else "pass --promote to replace demo",
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
