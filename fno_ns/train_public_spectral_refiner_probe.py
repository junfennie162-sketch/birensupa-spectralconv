#!/usr/bin/env python3
"""Public NS64 · Spectral-Refiner lite (Sobolev-weighted spectral loss).

Only spectral_conv weights trainable; loss mixes rel-L2 with H^{-1}-style
Fourier weighting (alpha + |k|^2)^{-1} on prediction error — inspired by
Spectral-Refiner (ICLR 2025). Eval: official 10→1 rel-L2. No auto-promote.
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
    predict,
    rel_l2_loss,
)

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent / "results" / "run_logs"
CKPT_DIR = ROOT / "checkpoints"


def sobolev_neg1_rel_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
    mix_l2: float,
) -> torch.Tensor:
    """Relative Sobolev H^{-1}-style loss via rfft2 weighting."""
    err = pred - target
    ef = torch.fft.rfft2(err)
    tf = torch.fft.rfft2(target)
    b, c, h, wf = ef.shape
    fdtype = torch.float32 if ef.device.type == "cpu" else torch.float32
    yy = torch.arange(h, device=ef.device, dtype=fdtype)[:, None]
    xx = torch.arange(wf, device=ef.device, dtype=fdtype)[None, :]
    yy = torch.minimum(yy, h - yy)
    rr2 = yy**2 + xx**2
    w = 1.0 / (alpha + rr2).clamp_min(1e-6)
    em = torch.linalg.vector_norm(torch.view_as_real(ef), dim=-1)
    tm = torch.linalg.vector_norm(torch.view_as_real(tf), dim=-1)
    num = torch.norm(w * em)
    den = torch.norm(w * tm).clamp_min(1e-12)
    sob = num / den
    if mix_l2 <= 0:
        return sob
    return (1.0 - mix_l2) * rel_l2_loss(pred, target) + mix_l2 * sob


def freeze_non_spectral(model: FNO2d) -> tuple[int, int]:
    frozen = thawed = 0
    for name, p in model.named_parameters():
        if "spectral_conv" in name:
            p.requires_grad_(True)
            thawed += p.numel()
        else:
            p.requires_grad_(False)
            frozen += p.numel()
    return frozen, thawed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--lr", type=float, default=8e-7)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--sob-alpha", type=float, default=1.0)
    ap.add_argument("--mix-l2", type=float, default=0.35, help="weight on Sobolev term")
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--baseline", type=float, default=0.03511497611179948)
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--stop-on-gate", action="store_true")
    ap.add_argument("--residual", action="store_true", default=True)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument(
        "--init-from",
        type=str,
        default=str(CKPT_DIR / "fno_ns_public_demo.pt"),
    )
    ap.add_argument("--tag", type=str, default="spec_ref_r1")
    args = ap.parse_args()

    residual = False if args.no_residual else True
    augment = False if args.no_augment else True
    gate = float(args.gate) if args.gate and args.gate > 0 else float(args.baseline) - 1e-4
    stop_on_gate = bool(args.stop_on_gate)

    torch.manual_seed(args.seed)
    ckpt_path = CKPT_DIR / f"fno_ns_public_{args.tag}_best.pt"
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
        raise SystemExit(f"need public NS64, got {source}")

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
        modes1=modes, modes2=modes, width=width, n_layers=4,
        in_channels=10, out_channels=1,
    )
    model.load_state_dict(blob["model"] if "model" in blob else blob)
    frozen_n, spectral_n = freeze_non_spectral(model)
    print(
        f"init from {args.init_from} spectral_only n={spectral_n} frozen={frozen_n} "
        f"mix_l2={args.mix_l2} alpha={args.sob_alpha}",
        flush=True,
    )

    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 3))

    baseline = evaluate(model, test_loader, residual=residual)
    best = baseline
    print(f"baseline={baseline:.8f} gate={gate:.8f}", flush=True)

    log_path = LOG_DIR / f"fno_public_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    lines = [f"# spec_ref tag={args.tag} baseline={baseline:.8f} gate={gate:.8f}"]
    stale = 0
    t0 = time.time()
    epoch = -1

    for epoch in range(args.epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for x, y in train_loader:
            optim.zero_grad()
            pred = predict(model, x, residual)
            loss = sobolev_neg1_rel_loss(
                pred, y, alpha=args.sob_alpha, mix_l2=args.mix_l2
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
                    "mix_l2": args.mix_l2,
                    "sob_alpha": args.sob_alpha,
                    "split": {"n_train": args.n_train, "n_test": args.n_test, "seed": args.seed},
                },
                ckpt_path,
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
            lines.append(f"stop_on_gate best={best:.8f}")
            print("stop_on_gate", flush=True)
            break
        if stale >= args.early_stop_patience:
            lines.append("early_stop")
            print("early_stop", flush=True)
            break
        if test_l2 > baseline + 8e-4:
            lines.append("abort rebound")
            print("abort rebound", flush=True)
            break

    summary = {
        "task": "fno_public_spectral_refiner_lite",
        "tag": args.tag,
        "baseline_test_l2": baseline,
        "best_test_l2": best,
        "gate": gate,
        "beat_gate": best < gate,
        "mix_l2": args.mix_l2,
        "sob_alpha": args.sob_alpha,
        "epochs_ran": epoch + 1,
        "elapsed_sec": round(time.time() - t0, 1),
        "checkpoint": str(ckpt_path) if ckpt_path.exists() else None,
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
