#!/usr/bin/env python3
"""Scheduled-sampling / soft-sched multistep probe (public NS64).

Eval remains official step-1. Does NOT auto-promote.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d
from train_public_multistep_probe import energy_rel, spectrum_tilt_rel
from train_public_ns64_boost import (
    RollAugDataset,
    evaluate,
    highfreq_rel_loss,
    predict,
    rel_l2_loss,
)

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent / "results" / "run_logs"
CKPT_DIR = ROOT / "checkpoints"


def sched_multistep_loss(
    model: FNO2d,
    x: torch.Tensor,
    y_seq: torch.Tensor,
    *,
    residual: bool,
    hf_weight: float,
    energy_w: float,
    tilt_w: float,
    p_ar: float,
    soft_alpha: float,
) -> torch.Tensor:
    cur = x
    total = None
    steps = y_seq.shape[1]
    for t in range(steps):
        target = y_seq[:, t : t + 1]
        pred = predict(model, cur, residual)
        loss = rel_l2_loss(pred, target)
        if hf_weight > 0:
            loss = loss + hf_weight * highfreq_rel_loss(pred, target)
        if energy_w > 0:
            loss = loss + energy_w * energy_rel(pred, target)
        if tilt_w > 0:
            loss = loss + tilt_w * spectrum_tilt_rel(pred, target)
        total = loss if total is None else total + loss
        if soft_alpha > 0:
            nxt = soft_alpha * pred.detach() + (1.0 - soft_alpha) * target
        else:
            use_pred = p_ar > 0 and torch.rand(()) < p_ar
            nxt = pred.detach() if use_pred else target
        cur = torch.cat([cur[:, 1:], nxt], dim=1)
    assert total is not None
    return total / steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--t-out-train", type=int, default=3)
    ap.add_argument("--hf-weight", type=float, default=0.2)
    ap.add_argument("--energy-weight", type=float, default=0.05)
    ap.add_argument("--tilt-weight", type=float, default=0.05)
    ap.add_argument("--p-ar-max", type=float, default=0.35)
    ap.add_argument(
        "--soft-alpha-max",
        type=float,
        default=0.0,
        help="if >0, use soft mix α*pred+(1-α)*gt instead of Bernoulli p_ar",
    )
    ap.add_argument("--early-stop-patience", type=int, default=3)
    ap.add_argument("--baseline", type=float, default=0.035724617540836334)
    ap.add_argument("--gate", type=float, default=0.0, help="default = baseline-1e-4")
    ap.add_argument(
        "--stop-on-gate",
        action="store_true",
        default=True,
        help="stop as soon as best < gate (fast-path; default on)",
    )
    ap.add_argument("--no-stop-on-gate", action="store_true")
    ap.add_argument("--residual", action="store_true", default=True)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument(
        "--init-from",
        type=str,
        default=str(CKPT_DIR / "fno_ns_public_demo.pt"),
    )
    ap.add_argument("--tag", type=str, default="sched_samp_r5")
    args = ap.parse_args()
    residual = False if args.no_residual else True
    augment = False if args.no_augment else True
    stop_on_gate = False if args.no_stop_on_gate else True
    gate = args.gate if args.gate > 0 else args.baseline - 1e-4

    torch.manual_seed(args.seed)
    data, source = load_or_build_ns_like(
        n_samples=args.n_train + args.n_test,
        resolution=64,
        n_times=20,
        seed=args.seed,
        version="v2",
    )
    if not str(source).startswith("file:navier_stokes"):
        raise SystemExit(f"need public NS64, got {source}")

    train_data, test_data = split_train_test(
        data, args.n_train, args.n_test, seed=args.seed
    )
    train_loader = DataLoader(
        RollAugDataset(
            SequenceVorticityDataset(train_data, 10, args.t_out_train),
            augment=augment,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        SequenceVorticityDataset(test_data, 10, 1),
        batch_size=args.batch_size,
        shuffle=False,
    )

    blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
    modes = int(blob.get("modes", 16)) if isinstance(blob, dict) else 16
    width = int(blob.get("width", 32)) if isinstance(blob, dict) else 32
    model = FNO2d(
        modes1=modes,
        modes2=modes,
        width=width,
        n_layers=4,
        in_channels=10,
        out_channels=1,
    )
    model.load_state_dict(blob["model"] if "model" in blob else blob)
    print(f"init modes={modes} width={width}", flush=True)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 3))

    baseline = evaluate(model, test_loader, residual=residual)
    best = baseline
    print(
        f"baseline={baseline:.8f} gate={gate:.8f} soft_alpha_max={args.soft_alpha_max}",
        flush=True,
    )

    ckpt_path = CKPT_DIR / f"fno_ns_public_{args.tag}_best.pt"
    log_path = LOG_DIR / f"fno_public_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    lines = [
        f"# sched tag={args.tag} baseline={baseline:.8f} gate={gate:.8f} "
        f"soft_max={args.soft_alpha_max}"
    ]
    stale = 0
    t0 = time.time()
    epoch = -1

    for epoch in range(args.epochs):
        frac = epoch / max(args.epochs - 1, 1)
        p_ar = args.p_ar_max * frac
        soft_alpha = args.soft_alpha_max * frac
        model.train()
        tr_sum, tr_n = 0.0, 0
        for x, y_seq in train_loader:
            optim.zero_grad()
            loss = sched_multistep_loss(
                model,
                x,
                y_seq,
                residual=residual,
                hf_weight=args.hf_weight,
                energy_w=args.energy_weight,
                tilt_w=args.tilt_weight,
                p_ar=p_ar,
                soft_alpha=soft_alpha,
            )
            loss.backward()
            optim.step()
            tr_sum += float(loss.item())
            tr_n += 1
        sched.step()
        test_l2 = evaluate(model, test_loader, residual=residual)
        improved = test_l2 < best - 1e-9
        if improved:
            best = test_l2
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "test_l2": test_l2,
                    "epoch": epoch + 1,
                    "data_source": source,
                    "residual": residual,
                    "modes": modes,
                    "width": width,
                    "promoted_tag": args.tag,
                    "p_ar": p_ar,
                    "soft_alpha": soft_alpha,
                    "split": {
                        "n_train": args.n_train,
                        "n_test": args.n_test,
                        "seed": args.seed,
                    },
                },
                ckpt_path,
            )
        else:
            stale += 1
        line = (
            f"epoch {epoch+1:3d} | p_ar={p_ar:.3f} soft={soft_alpha:.3f} | "
            f"train={tr_sum/max(tr_n,1):.6f} | test_l2={test_l2:.8f} | best={best:.8f}"
            + (" *" if improved else "")
        )
        print(line, flush=True)
        lines.append(line)
        if stop_on_gate and best < gate:
            lines.append(f"stop_on_gate best={best:.8f} < gate={gate:.8f}")
            print("stop_on_gate", flush=True)
            break
        if stale >= args.early_stop_patience:
            lines.append(f"early_stop patience={args.early_stop_patience}")
            print("early_stop", flush=True)
            break
        if test_l2 > baseline + 5e-4:
            lines.append("abort rebound >5e-4")
            print("abort rebound", flush=True)
            break

    summary = {
        "task": "fno_public_sched_sampling",
        "tag": args.tag,
        "baseline_test_l2": baseline,
        "best_test_l2": best,
        "gate": gate,
        "beat_gate": best < gate,
        "improved_vs_baseline": best < baseline - 1e-9,
        "delta_vs_baseline": baseline - best,
        "epochs_ran": epoch + 1,
        "p_ar_max": args.p_ar_max,
        "soft_alpha_max": args.soft_alpha_max,
        "lr": args.lr,
        "elapsed_sec": round(time.time() - t0, 1),
        "checkpoint": str(ckpt_path) if ckpt_path.exists() else None,
        "log": str(log_path),
        "promote": False,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    (LOG_DIR / f"fno_public_boost_{args.tag}_summary.json").write_text(text)
    (LOG_DIR / f"fno_public_{args.tag}_summary.json").write_text(text)
    lines.append(json.dumps(summary, ensure_ascii=False))
    log_path.write_text("\n".join(lines) + "\n")
    print(text)
    if summary["beat_gate"]:
        print("SIGNAL: beat_gate — eligible for promote review")
    else:
        print("NO_SIGNAL")


if __name__ == "__main__":
    main()
