#!/usr/bin/env python3
"""Public NS64 · q_t oversampling + PF/Δ hybrid (longer probe).

Hard cases correlate with temporal increment |y-x_last| (Autopsy q_t).
Unlike hard_reweight (loss scalar), this *resamples* high-q_t windows more often.

Eval: official clean 10→1. Does NOT auto-promote.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d
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


class QtWeightedBase(Dataset):
    """Fixed-window dataset exposing per-index q_t for sampling weights."""

    def __init__(self, data: torch.Tensor, t_in: int = 10, t_out: int = 2):
        self.base = SequenceVorticityDataset(data, t_in, t_out)
        # q_t on first future step vs last input frame
        x0 = data[:, t_in - 1]
        y0 = data[:, t_in]
        qt = (y0 - x0).reshape(data.shape[0], -1).norm(dim=1)
        self.qt = qt.clamp_min(1e-8)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        return self.base[index]


def hybrid_loss(
    model: FNO2d,
    x: torch.Tensor,
    y_seq: torch.Tensor,
    *,
    residual: bool,
    hf_weight: float,
    lambda_pf: float,
    lambda_delta: float,
) -> torch.Tensor:
    y1 = y_seq[:, 0:1]
    y2 = y_seq[:, 1:2]
    pred1 = predict(model, x, residual)
    loss = rel_l2_loss(pred1, y1)
    if hf_weight > 0:
        loss = loss + hf_weight * highfreq_rel_loss(pred1, y1)
    if lambda_delta > 0:
        xl = x[:, -1:]
        loss = loss + lambda_delta * rel_l2_loss(pred1 - xl, y1 - xl)
    if lambda_pf > 0:
        x_pf = torch.cat([x[:, 1:], pred1.detach()], dim=1)
        pred2 = predict(model, x_pf, residual)
        loss_pf = rel_l2_loss(pred2, y2)
        if hf_weight > 0:
            loss_pf = loss_pf + hf_weight * highfreq_rel_loss(pred2, y2)
        loss = loss + lambda_pf * loss_pf
    return loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--hf-weight", type=float, default=0.15)
    ap.add_argument("--lambda-pf", type=float, default=0.6)
    ap.add_argument("--lambda-delta", type=float, default=0.4)
    ap.add_argument("--qt-power", type=float, default=1.5)
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="0=len(train)")
    ap.add_argument("--early-stop-patience", type=int, default=4)
    ap.add_argument("--baseline", type=float, default=0.03522327123209834)
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
        default=str(CKPT_DIR / "fno_ns_public_pf_delta_r1_best.pt"),
    )
    ap.add_argument("--tag", type=str, default="qt_over_r1")
    args = ap.parse_args()

    residual = False if args.no_residual else True
    augment = False if args.no_augment else True
    freeze_spectral = False if args.no_freeze_spectral else True
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
    qt_base = QtWeightedBase(train_data, 10, 2)
    weights = (qt_base.qt ** args.qt_power).double()
    weights = weights / weights.mean()
    n_steps = args.steps_per_epoch if args.steps_per_epoch > 0 else len(qt_base)
    sampler = WeightedRandomSampler(weights, num_samples=n_steps, replacement=True)
    train_loader = DataLoader(
        RollAugDataset(qt_base, augment=augment),
        batch_size=args.batch_size,
        sampler=sampler,
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
    print(f"init modes={modes} width={width} from {args.init_from}", flush=True)

    frozen_n = 0
    if freeze_spectral:
        for name, param in model.named_parameters():
            if "spectral_conv" in name:
                param.requires_grad_(False)
                frozen_n += param.numel()
        print(f"freeze_spectral params={frozen_n}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 3))

    baseline = evaluate(model, test_loader, residual=residual)
    best = baseline
    print(
        f"baseline={baseline:.8f} gate={gate:.8f} qt_power={args.qt_power} "
        f"λ_pf={args.lambda_pf} λ_δ={args.lambda_delta}",
        flush=True,
    )

    log_path = LOG_DIR / f"fno_public_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    lines = [f"# qt_oversample tag={args.tag} baseline={baseline:.8f} gate={gate:.8f}"]
    stale = 0
    t0 = time.time()
    epoch = -1

    for epoch in range(args.epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for x, y_seq in train_loader:
            optim.zero_grad()
            loss = hybrid_loss(
                model,
                x,
                y_seq,
                residual=residual,
                hf_weight=args.hf_weight,
                lambda_pf=args.lambda_pf,
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
                    "qt_power": args.qt_power,
                    "freeze_spectral": freeze_spectral,
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
        "task": "fno_public_qt_oversample",
        "tag": args.tag,
        "baseline_test_l2": baseline,
        "best_test_l2": best,
        "gate": gate,
        "beat_gate": best < gate,
        "improved_vs_baseline": best < baseline - 1e-9,
        "delta_vs_baseline": baseline - best,
        "epochs_ran": epoch + 1,
        "qt_power": args.qt_power,
        "lambda_pf": args.lambda_pf,
        "lambda_delta": args.lambda_delta,
        "hf_weight": args.hf_weight,
        "lr": args.lr,
        "freeze_spectral": freeze_spectral,
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
