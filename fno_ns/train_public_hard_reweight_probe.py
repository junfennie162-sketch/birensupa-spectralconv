#!/usr/bin/env python3
"""A1 · hard-example reweight probe on public NS64.

Upsamples high per-sample relative-L2 examples in the train loss (focal-style).
Does NOT auto-promote. Use stop-on-gate + early-stop-patience.
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
from train_public_ns64_boost import (
    RollAugDataset,
    evaluate,
    highfreq_rel_loss,
    predict,
)

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent / "results" / "run_logs"
CKPT_DIR = ROOT / "checkpoints"


def per_sample_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    b = pred.shape[0]
    diff = torch.norm((pred - target).reshape(b, -1), dim=1)
    ref = torch.norm(target.reshape(b, -1), dim=1).clamp_min(1e-12)
    return diff / ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--hf-weight", type=float, default=0.15)
    ap.add_argument("--hard-gamma", type=float, default=1.5)
    ap.add_argument("--hard-clip", type=float, default=4.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--early-stop-patience", type=int, default=2)
    ap.add_argument("--baseline", type=float, default=0.03530218452215195)
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--stop-on-gate", action="store_true")
    ap.add_argument("--residual", action="store_true", default=True)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--freeze-spectral", action="store_true", default=True)
    ap.add_argument("--no-freeze-spectral", action="store_true")
    ap.add_argument(
        "--init-from",
        type=str,
        default=str(CKPT_DIR / "fno_ns_public_demo.pt"),
    )
    ap.add_argument("--ckpt-name", type=str, default="fno_ns_public_hard_reweight_a1_best.pt")
    ap.add_argument("--tag", type=str, default="hard_reweight_a1")
    args = ap.parse_args()

    residual = False if args.no_residual else True
    augment = False if args.no_augment else True
    freeze_spectral = False if args.no_freeze_spectral else True
    gate = float(args.gate) if args.gate and args.gate > 0 else float(args.baseline) - 1e-4
    stop_on_gate = bool(args.stop_on_gate)

    torch.manual_seed(args.seed)
    ckpt_path = CKPT_DIR / args.ckpt_name
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
    modes = int(blob.get("modes", 16))
    width = int(blob.get("width", 32))
    model = FNO2d(
        modes1=modes,
        modes2=modes,
        width=width,
        n_layers=4,
        in_channels=10,
        out_channels=1,
    )
    model.load_state_dict(blob["model"])

    frozen_n = 0
    if freeze_spectral:
        for name, param in model.named_parameters():
            if "spectral_conv" in name:
                param.requires_grad_(False)
                frozen_n += param.numel()

    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"fno_public_hard_reweight_{stamp}.log"
    log_path.write_text(
        f"# hard_reweight tag={args.tag} gamma={args.hard_gamma} clip={args.hard_clip}\n"
        f"# baseline={args.baseline} gate={gate} stop_on_gate={stop_on_gate}\n"
        f"# init={args.init_from} freeze_spectral={freeze_spectral} frozen_n={frozen_n}\n"
    )

    baseline = evaluate(model, test_loader, residual=residual)
    best_l2 = baseline
    torch.save(
        {
            "model": model.state_dict(),
            "test_l2": baseline,
            "epoch": 0,
            "data_source": source,
            "residual": residual,
            "augment": augment,
            "hf_weight": args.hf_weight,
            "hard_gamma": args.hard_gamma,
            "modes": modes,
            "width": width,
            "promoted_tag": args.tag,
            "split": {"n_train": args.n_train, "n_test": args.n_test, "seed": args.seed},
        },
        ckpt_path,
    )
    print(f"baseline_test_l2={baseline:.8f} gate={gate:.8f}", flush=True)
    log_path.write_text(log_path.read_text() + f"# measured_baseline={baseline:.8f}\n")

    t0 = time.time()
    stale = 0
    stop_reason = "completed"
    for epoch in range(args.epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for x, y in train_loader:
            optim.zero_grad()
            pred = predict(model, x, residual)
            per = per_sample_rel_l2(pred, y)
            mean = per.mean().clamp_min(1e-12)
            w = (per / mean).pow(args.hard_gamma).clamp(1.0 / args.hard_clip, args.hard_clip)
            loss = (w * per).mean()
            if args.hf_weight > 0:
                loss = loss + args.hf_weight * highfreq_rel_loss(pred, y)
            loss.backward()
            optim.step()
            tr_sum += float(loss.item())
            tr_n += 1

        train_l2 = tr_sum / max(tr_n, 1)
        test_l2 = evaluate(model, test_loader, residual=residual)
        mark = ""
        if test_l2 < best_l2 - 1e-9:
            best_l2 = test_l2
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "test_l2": test_l2,
                    "epoch": epoch + 1,
                    "data_source": source,
                    "residual": residual,
                    "augment": augment,
                    "hf_weight": args.hf_weight,
                    "hard_gamma": args.hard_gamma,
                    "modes": modes,
                    "width": width,
                    "promoted_tag": args.tag,
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
                        "best_test_l2": best_l2,
                        "epoch": epoch + 1,
                        "tag": args.tag,
                        "gate": gate,
                        "checkpoint": str(ckpt_path),
                    },
                    indent=2,
                )
                + "\n"
            )
            mark = "  *best*"
        else:
            stale += 1

        line = (
            f"epoch {epoch+1:3d} | train={train_l2:.6f} | test_l2={test_l2:.6f} | "
            f"best={best_l2:.6f} | lr={optim.param_groups[0]['lr']:.2e}{mark}"
        )
        print(line, flush=True)
        log_path.write_text(log_path.read_text() + line + "\n")

        if stop_on_gate and best_l2 < gate:
            stop_reason = f"stop_on_gate best={best_l2:.8f} < gate={gate:.8f}"
            print(stop_reason, flush=True)
            log_path.write_text(log_path.read_text() + stop_reason + "\n")
            break
        if args.early_stop_patience > 0 and stale >= args.early_stop_patience:
            stop_reason = f"early_stop patience={args.early_stop_patience}"
            print(stop_reason, flush=True)
            log_path.write_text(log_path.read_text() + stop_reason + "\n")
            break

    summary = {
        "task": "fno_public_hard_reweight",
        "tag": args.tag,
        "data_source": source,
        "baseline_test_l2": baseline,
        "best_test_l2": best_l2,
        "gate": gate,
        "beat_gate": bool(best_l2 < gate),
        "improved_vs_baseline": bool(best_l2 < baseline - 1e-9),
        "delta_vs_baseline": baseline - best_l2,
        "stop_reason": stop_reason,
        "hard_gamma": args.hard_gamma,
        "hard_clip": args.hard_clip,
        "hf_weight": args.hf_weight,
        "freeze_spectral": freeze_spectral,
        "epochs": args.epochs,
        "lr": args.lr,
        "elapsed_sec": round(time.time() - t0, 1),
        "checkpoint": str(ckpt_path),
        "log": str(log_path),
        "promote": False,
        "note": "no auto-promote; human Go required if beat_gate",
    }
    out = LOG_DIR / f"fno_public_boost_{args.tag}_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
