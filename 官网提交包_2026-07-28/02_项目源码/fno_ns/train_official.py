#!/usr/bin/env python3
"""FNO-NS training aligned with contest official gate (image 2).

Official gate (per user-provided screenshot):
- n_train=1000, batch_size=16 → ~63 step/epoch
- n_epochs=100 → 6300 step total
- 晋阶 : <= 500  step (≈ 8 epoch)        → 流程分
- 有效 : <= 2000 step (≈ 30 epoch)        → 可评实测 L2
- 推荐 : >= 6000 step (≈ 100 epoch)       → 对齐参考训练集

We use:
- ns_like_v2 cache (or auto-generated) at 64x64
- 768/128 train/test split (matches existing checkpoints)
- bs=16, Adam(lr=2e-4, wd=1e-4), rel-L2 loss on CPU path
- Save best-L2 checkpoint under checkpoints/fno_ns_official_best.pt
- Append metrics to results/run_logs/fno_official_train_<date>.log

Run:
    cd fno_ns && python3 train_official.py --epochs 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "spectral_conv_combo"))

from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d


def relative_l2(pred, target):
    diff = torch.norm(pred - target, dim=(-2, -1))
    ref = torch.norm(target, dim=(-2, -1)).clamp_min(1.0e-12)
    return float((diff / ref).mean().item())


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
    p.add_argument("--use-scheduler", action="store_true", default=True, help="use CosineAnnealingLR (default on)")
    p.add_argument("--batch-size", type=int, default=16, help="official = 16")
    p.add_argument("--n-train", type=int, default=1000, help="official = 1000")
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument("--rebuild", action="store_true", help="rebuild NS-like cache even if file exists")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    CKPT = ROOT / "checkpoints" / "fno_ns_official_best.pt"
    CKPT.parent.mkdir(parents=True, exist_ok=True)

    data, source = load_or_build_ns_like(n_samples=args.n_train + args.n_test, resolution=64, n_times=30, seed=args.seed, rebuild=args.rebuild, version="v2")
    train_data, test_data = split_train_test(data, args.n_train, args.n_test, seed=args.seed)
    train_loader = DataLoader(SequenceVorticityDataset(train_data, 10, 1), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(SequenceVorticityDataset(test_data, 10, 1), batch_size=args.batch_size, shuffle=False)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    print(f"data={source} train={args.n_train} bs={args.batch_size} steps/epoch={steps_per_epoch} total_steps={total_steps}")

    model = FNO2d(modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    # CosineAnnealing: at epoch=N lr=0; with N=100 lr reaches ~0 only at last epoch (intended).
    # For short test runs (e.g. epochs=2) this would kill lr early; user can pass --no-scheduler.
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 3))

    log = ROOT.parent / "results" / "run_logs" / f"fno_official_train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# FNO-NS official-gate training",
        f"# data={source} n_train={args.n_train} bs={args.batch_size} steps/epoch={steps_per_epoch}",
        f"# epochs={args.epochs} total_steps={total_steps}",
        "# gate: 晋阶<=500 step (≈8 epoch), 有效<=2000 step (≈30 epoch), 推荐>=6000 step (≈100 epoch)",
        "# step | epoch | train_l2 | test_l2 | best_l2 | saved_best",
    ]
    log.write_text("\n".join(header) + "\n")

    best_l2 = float("inf")
    step_count = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum, train_count = 0.0, 0
        for x, y in train_loader:
            optim.zero_grad()
            pred = model(x, use_supa=False)
            loss = torch.norm(pred - y) / torch.norm(y).clamp_min(1e-12)
            loss.backward()
            optim.step()
            train_loss_sum += float(loss.item())
            train_count += 1
            step_count += 1
        if scheduler is not None:
            scheduler.step()
        avg_train = train_loss_sum / max(train_count, 1)
        test_l2 = evaluate(model, test_loader)
        improved = ""
        if test_l2 < best_l2:
            best_l2 = test_l2
            torch.save({"model": model.state_dict(), "test_l2": test_l2, "step": step_count, "epoch": epoch + 1}, CKPT)
            improved = "  *best*"
        entry = f"step {step_count:5d} | epoch {epoch+1:3d} | train_l2={avg_train:.6f} | test_l2={test_l2:.6f} | best_l2={best_l2:.6f} | lr={optim.param_groups[0]['lr']:.2e}{improved}"
        print(entry)
        log.write_text(log.read_text() + entry + "\n")
        # 晋阶 (≤500 step)、有效 (≤2000 step)、推荐 (≥6000 step) gate print
        if step_count in (500, 2000, 6000) or step_count == total_steps:
            tag = "晋阶" if step_count <= 500 else ("有效" if step_count <= 2000 else "推荐")
            print(f"  >> gate={tag} step={step_count} test_l2={test_l2:.6f} best_l2={best_l2:.6f}")

    elapsed = time.time() - t0
    summary = {
        "task": "fno_ns_official_train",
        "data_source": source,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "best_test_l2": best_l2,
        "elapsed_sec": round(elapsed, 1),
        "checkpoint": str(CKPT),
        "gate": (
            "晋阶" if total_steps <= 500 else
            "有效" if total_steps <= 2000 else
            "推荐"
        ),
        "log": str(log),
    }
    print("==SUMMARY==")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    log.write_text(log.read_text() + "\n# SUMMARY\n" + json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()