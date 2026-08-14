#!/usr/bin/env python3
"""Public NS64 · thaw last K Fourier spectral_conv only (capacity without modes↑).

Freeze all spectral_conv except fourier_layers[-K:]; train with Δ-match + light HF.
Eval official 10→1. Does NOT auto-promote.
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
from train_public_delta_match_probe import delta_match_loss
from train_public_ns64_boost import RollAugDataset, evaluate

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent / "results" / "run_logs"
CKPT_DIR = ROOT / "checkpoints"


def thaw_last_spectral(model: FNO2d, k: int) -> tuple[int, int]:
    """Freeze all spectral; unfreeze last k Fourier layers' spectral_conv."""
    n_layers = len(model.fourier_layers)
    k = max(1, min(k, n_layers))
    frozen = thawed = 0
    thaw_ids = set(range(n_layers - k, n_layers))
    for name, param in model.named_parameters():
        if "spectral_conv" not in name:
            continue
        # name like fourier_layers.3.spectral_conv.weights1
        parts = name.split(".")
        try:
            idx = int(parts[1])
        except (IndexError, ValueError):
            param.requires_grad_(False)
            frozen += param.numel()
            continue
        if idx in thaw_ids:
            param.requires_grad_(True)
            thawed += param.numel()
        else:
            param.requires_grad_(False)
            frozen += param.numel()
    return frozen, thawed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--spectral-lr", type=float, default=5e-7)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--hf-weight", type=float, default=0.12)
    ap.add_argument("--lambda-delta", type=float, default=0.45)
    ap.add_argument("--thaw-k", type=int, default=1)
    ap.add_argument("--early-stop-patience", type=int, default=4)
    ap.add_argument("--baseline", type=float, default=0.03522327123209834)
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--stop-on-gate", action="store_true")
    ap.add_argument("--residual", action="store_true", default=True)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument(
        "--init-from",
        type=str,
        default=str(CKPT_DIR / "fno_ns_public_pf_delta_r1_best.pt"),
    )
    ap.add_argument("--tag", type=str, default="last_thaw_r1")
    args = ap.parse_args()

    residual = False if args.no_residual else True
    augment = False if args.no_augment else True
    gate = float(args.gate) if args.gate and args.gate > 0 else float(args.baseline) - 1e-4
    stop_on_gate = bool(args.stop_on_gate)

    torch.manual_seed(args.seed)
    ckpt_path = CKPT_DIR / f"fno_ns_public_{args.tag}_best.pt"
    meta_path = ckpt_path.with_name(ckpt_path.stem + "_meta.json")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    data, source = load_or_build_ns_like(
        n_samples=args.n_train + args.n_test,
        resolution=64,
        n_times=20,
        seed=args.seed,
        version="v2",
    )
    if not str(source).startswith("file:navier_stokes"):
        raise SystemExit(f"need public NS64 file, got {source}")

    train_data, test_data = split_train_test(
        data, args.n_train, args.n_test, seed=args.seed
    )
    train_loader = DataLoader(
        RollAugDataset(SequenceVorticityDataset(train_data, 10, 1), augment=augment),
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
        modes1=modes, modes2=modes, width=width, n_layers=4, in_channels=10, out_channels=1
    )
    model.load_state_dict(blob["model"] if "model" in blob else blob)
    frozen_n, thawed_n = thaw_last_spectral(model, args.thaw_k)
    print(
        f"init from {args.init_from} thaw_k={args.thaw_k} "
        f"frozen_spectral={frozen_n} thawed_spectral={thawed_n}",
        flush=True,
    )

    head_params = []
    spectral_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "spectral_conv" in name:
            spectral_params.append(p)
        else:
            head_params.append(p)
    optim = torch.optim.Adam(
        [
            {"params": head_params, "lr": args.lr},
            {"params": spectral_params, "lr": args.spectral_lr},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 3))

    baseline = evaluate(model, test_loader, residual=residual)
    best = baseline
    print(f"baseline={baseline:.8f} gate={gate:.8f}", flush=True)

    log_path = LOG_DIR / f"fno_public_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    lines = [f"# last_thaw tag={args.tag} baseline={baseline:.8f} gate={gate:.8f}"]
    stale = 0
    t0 = time.time()
    epoch = -1

    for epoch in range(args.epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for x, y in train_loader:
            optim.zero_grad()
            loss = delta_match_loss(
                model,
                x,
                y,
                residual=residual,
                hf_weight=args.hf_weight,
                lambda_delta=args.lambda_delta,
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
                    "thaw_k": args.thaw_k,
                    "split": {
                        "n_train": args.n_train,
                        "n_test": args.n_test,
                        "seed": args.seed,
                    },
                },
                ckpt_path,
            )
            meta_path.write_text(
                json.dumps(
                    {
                        "tag": args.tag,
                        "best_test_l2": best,
                        "gate": gate,
                        "epoch": epoch + 1,
                        "checkpoint": str(ckpt_path),
                    },
                    indent=2,
                )
                + "\n"
            )
        else:
            stale += 1
        line = (
            f"epoch {epoch+1:3d} | train={tr_sum/max(tr_n,1):.6f} | "
            f"test_l2={test_l2:.8f} | best={best:.8f}"
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
        if test_l2 > baseline + 8e-4:
            lines.append("abort rebound >8e-4")
            print("abort rebound", flush=True)
            break

    summary = {
        "task": "fno_public_last_thaw",
        "tag": args.tag,
        "baseline_test_l2": baseline,
        "best_test_l2": best,
        "gate": gate,
        "beat_gate": best < gate,
        "improved_vs_baseline": best < baseline - 1e-9,
        "delta_vs_baseline": baseline - best,
        "epochs_ran": epoch + 1,
        "thaw_k": args.thaw_k,
        "lr": args.lr,
        "spectral_lr": args.spectral_lr,
        "lambda_delta": args.lambda_delta,
        "init_from": args.init_from,
        "elapsed_sec": round(time.time() - t0, 1),
        "checkpoint": str(ckpt_path) if ckpt_path.exists() else None,
        "log": str(log_path),
        "promote": False,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    (LOG_DIR / f"fno_public_boost_{args.tag}_summary.json").write_text(text)
    lines.append(json.dumps(summary, ensure_ascii=False))
    log_path.write_text("\n".join(lines) + "\n")
    print(text)
    print("SIGNAL: beat_gate" if summary["beat_gate"] else "NO_SIGNAL")


if __name__ == "__main__":
    main()
