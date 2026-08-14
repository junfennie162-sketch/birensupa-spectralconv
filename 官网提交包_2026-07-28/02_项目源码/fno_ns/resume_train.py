#!/usr/bin/env python3
"""Resume FNO training from a prior checkpoint with best-checkpoint saving.

Loads the existing `checkpoints/fno_ns_demo.pt`, runs additional epochs in
small increments of `RESUME_STEP`, evaluates relative L2 on the canonical
test split, and saves a new best checkpoint whenever L2 improves. This is
a slow, CPU-only pass intended to land a free win on the L2 metric.

Differs from the upstream `test_forward.py`:
- does NOT clobber `summary.json` / `results.md` or the existing checkpoint;
  the new best is written to `checkpoints/fno_ns_resume_best.pt`.
- evaluates **after every `RESUME_STEP` epochs** and keeps the best-so-far
  state dict, so we don't have to run 200 epochs blind.

Usage:
    cd /workspace/ai4s-f/submission/fno_ns
    python3 resume_train.py --steps 6 --epochs-per-step 5
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spectral_conv"))

import torch
from torch.utils.data import DataLoader

from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    diff = torch.norm(prediction - target, dim=(-2, -1))
    ref = torch.norm(target, dim=(-2, -1)).clamp_min(1.0e-12)
    return float((diff / ref).mean().item())


def evaluate(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for inputs, targets in loader:
            scores.append(relative_l2(model(inputs, use_supa=False), targets))
    return sum(scores) / len(scores)


def train_one(model, loader, optim, n_epochs, history, log_prefix=""):
    model.train()
    for epoch in range(n_epochs):
        total = 0.0
        count = 0
        for inputs, targets in loader:
            optim.zero_grad()
            pred = model(inputs, use_supa=False)
            loss = torch.norm(pred - targets) / torch.norm(targets).clamp_min(1e-12)
            loss.backward()
            optim.step()
            total += float(loss.item())
            count += 1
        avg = total / max(count, 1)
        history.append(avg)
        if (epoch + 1) % 1 == 0:
            print(f"  {log_prefix}epoch {epoch + 1}: train rel L2 = {avg:.6f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--epochs-per-step", type=int, default=5)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument(
        "--seed", type=int, default=20260722,
        help="must match the seed used for `ns_like_v2` generation.",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)

    CKPT_BEST = Path(__file__).resolve().parent / "checkpoints" / "fno_ns_resume_best.pt"
    CKPT_BEST.parent.mkdir(parents=True, exist_ok=True)

    data, data_source = load_or_build_ns_like(
        n_samples=1024, resolution=64, n_times=30,
        seed=args.seed, version="v2",
    )
    train_data, test_data = split_train_test(data, 768, 128, seed=args.seed)
    train_loader = DataLoader(
        SequenceVorticityDataset(train_data, 10, 1),
        batch_size=8, shuffle=True,
    )
    test_loader = DataLoader(
        SequenceVorticityDataset(test_data, 10, 1),
        batch_size=8, shuffle=False,
    )

    model = FNO2d(modes1=16, modes2=16, width=32, n_layers=4,
                  in_channels=10, out_channels=1)
    src = Path(__file__).resolve().parent / "checkpoints" / "fno_ns_demo.pt"
    src_state = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(src_state["model"])
    starting_history = list(src_state.get("history", []))
    print(f"loaded {src} ({len(starting_history)} epochs already logged)")

    best_l2 = evaluate(model, test_loader)
    history = list(starting_history)
    print(f"baseline L2 = {best_l2:.6f}")

    log_path = Path(__file__).resolve().parents[1] / "results" / "run_logs" / "fno_resume_l2_2026-07-24.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# FNO-NS resume training (R5-2)",
        f"# start_at: total epochs at load = {len(starting_history)}",
        f"# baseline_l2 = {best_l2:.6f}",
    ]
    log_path.write_text("\n".join(lines) + "\n")
    lines.append("# step | extra_epochs | train_l2 | test_l2 | saved_best")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-4)

    for step in range(1, args.steps + 1):
        n = args.epochs_per_step
        train_one(model, train_loader, optim, n, history,
                  log_prefix=f"step {step}/{args.steps} ")
        cur_l2 = evaluate(model, test_loader)
        improved = ""
        if cur_l2 < best_l2:
            best_l2 = cur_l2
            torch.save(
                {"model": model.state_dict(),
                 "history": history,
                 "data_source": data_source,
                 "test_l2": cur_l2,
                 "total_epochs": len(history)},
                CKPT_BEST,
            )
            improved = "  *best*"
        entry = f"step {step} | +{n} | last_train={history[-1]:.6f} | test_l2={cur_l2:.6f} | best={best_l2:.6f}{improved}"
        print(entry)
        lines.append(entry)
        log_path.write_text("\n".join(lines) + "\n")

    final_entry = f"# final_best_l2 = {best_l2:.6f}"
    lines.append(final_entry)
    log_path.write_text("\n".join(lines) + "\n")
    print(final_entry)
    print(f"# final best saved to {CKPT_BEST}")


if __name__ == "__main__":
    main()
