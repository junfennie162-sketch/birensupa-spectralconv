#!/usr/bin/env python3
"""Train FNO-NS on the public NS64 viscosity dataset (HF mirror pull).

Does NOT overwrite v2 demo / official_best. Writes:
  checkpoints/fno_ns_public_ns64_best.pt
  results/run_logs/fno_public_ns64_train_*.log

Protocol: n_train=1000, n_test=128, bs=16, T_in=10, T_out=1, 100 epochs.
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
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument(
        "--init-from",
        type=str,
        default="",
        help="optional warm-start ckpt (state_dict under key 'model')",
    )
    p.add_argument(
        "--ckpt-name",
        type=str,
        default="fno_ns_public_ns64_best.pt",
        help="checkpoint filename under checkpoints/",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ckpt_path = ROOT / "checkpoints" / args.ckpt_name
    meta_path = ckpt_path.with_name(ckpt_path.stem + "_meta.json")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    data, source = load_or_build_ns_like(
        n_samples=args.n_train + args.n_test,
        resolution=64,
        n_times=20,
        seed=args.seed,
        version="v2",
    )
    if not str(source).startswith("file:navier_stokes"):
        raise SystemExit(
            f"expected public navier_stokes*.pt, got source={source!r}. "
            "Put the HF file under fno_ns/data/ first."
        )

    train_data, test_data = split_train_test(
        data, args.n_train, args.n_test, seed=args.seed
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

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    print(
        f"data={source} shape={tuple(data.shape)} train={args.n_train} "
        f"bs={args.batch_size} steps/epoch={steps_per_epoch} total_steps={total_steps}",
        flush=True,
    )

    model = FNO2d(
        modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1
    )
    if args.init_from:
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
        print(f"warm_start={args.init_from}", flush=True)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(args.epochs, 3)
    )

    log = (
        ROOT.parent
        / "results"
        / "run_logs"
        / f"fno_public_ns64_train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "\n".join(
            [
                "# FNO-NS PUBLIC NS64 training",
                f"# data={source} shape={tuple(data.shape)}",
                f"# n_train={args.n_train} n_test={args.n_test} bs={args.batch_size}",
                f"# epochs={args.epochs} total_steps={total_steps} lr={args.lr}",
                f"# init_from={args.init_from or 'scratch'}",
                "# step | epoch | train_l2 | test_l2 | best_l2 | lr",
                "",
            ]
        )
    )

    step_count = 0
    t0 = time.time()
    baseline = evaluate(model, test_loader)
    # Warm-start / resume: never overwrite an existing better checkpoint with a worse epoch.
    best_l2 = baseline if args.init_from else float("inf")
    if ckpt_path.exists() and args.init_from:
        try:
            prev = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            best_l2 = min(best_l2, float(prev.get("test_l2", best_l2)))
        except Exception:
            pass
    print(f"baseline_test_l2={baseline:.6f} track_best={best_l2:.6f}", flush=True)
    log.write_text(
        log.read_text()
        + f"# baseline_test_l2={baseline:.6f} track_best={best_l2:.6f}\n"
    )
    if args.init_from and baseline <= best_l2 + 1e-15:
        torch.save(
            {
                "model": model.state_dict(),
                "test_l2": baseline,
                "step": 0,
                "epoch": 0,
                "data_source": source,
                "split": {
                    "n_train": args.n_train,
                    "n_test": args.n_test,
                    "seed": args.seed,
                },
                "promoted_tag": "public_ns64_train",
                "init_from": args.init_from or "scratch",
            },
            ckpt_path,
        )
        meta_path.write_text(
            json.dumps(
                {
                    "best_test_l2": baseline,
                    "epoch": 0,
                    "step": 0,
                    "data_source": source,
                    "checkpoint": str(ckpt_path),
                },
                indent=2,
            )
            + "\n"
        )
        best_l2 = baseline

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
        scheduler.step()
        avg_train = train_loss_sum / max(train_count, 1)
        test_l2 = evaluate(model, test_loader)
        improved = ""
        if test_l2 < best_l2:
            best_l2 = test_l2
            torch.save(
                {
                    "model": model.state_dict(),
                    "test_l2": test_l2,
                    "step": step_count,
                    "epoch": epoch + 1,
                    "data_source": source,
                    "split": {
                        "n_train": args.n_train,
                        "n_test": args.n_test,
                        "seed": args.seed,
                    },
                    "promoted_tag": "public_ns64_train",
                    "init_from": args.init_from or "scratch",
                },
                ckpt_path,
            )
            meta_path.write_text(
                json.dumps(
                    {
                        "best_test_l2": best_l2,
                        "epoch": epoch + 1,
                        "step": step_count,
                        "data_source": source,
                        "checkpoint": str(ckpt_path),
                    },
                    indent=2,
                )
                + "\n"
            )
            improved = "  *best*"
        entry = (
            f"step {step_count:5d} | epoch {epoch+1:3d} | train_l2={avg_train:.6f} | "
            f"test_l2={test_l2:.6f} | best_l2={best_l2:.6f} | "
            f"lr={optim.param_groups[0]['lr']:.2e}{improved}"
        )
        print(entry, flush=True)
        log.write_text(log.read_text() + entry + "\n")
        if step_count in (500, 2000, 6000) or step_count == total_steps:
            tag = (
                "晋阶"
                if step_count <= 500
                else ("有效" if step_count <= 2000 else "推荐")
            )
            print(
                f"  >> gate={tag} step={step_count} test_l2={test_l2:.6f} best_l2={best_l2:.6f}",
                flush=True,
            )

    elapsed = time.time() - t0
    summary = {
        "task": "fno_ns_public_ns64_train",
        "data_source": source,
        "data_shape": list(data.shape),
        "n_train": args.n_train,
        "n_test": args.n_test,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "baseline_test_l2": baseline,
        "best_test_l2": best_l2,
        "elapsed_sec": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "init_from": args.init_from or "scratch",
        "log": str(log),
    }
    print("==SUMMARY==", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    (ROOT.parent / "results" / "run_logs" / "fno_public_ns64_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    log.write_text(
        log.read_text()
        + "\n# SUMMARY\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
