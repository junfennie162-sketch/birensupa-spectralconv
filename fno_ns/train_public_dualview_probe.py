#!/usr/bin/env python3
"""Public NS64 · dual-view consistency (two rolls) + supervised L2/Δ.

loss = L2(pred_a, y_a) + λ_δ L2(Δa) + λ_cons * L2(pred_a_unroll, pred_b_unroll)
where pred_*_unroll reverse the roll so both live in original coords.

Eval official 10→1. Does NOT auto-promote.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d
from train_public_ns64_boost import (
    evaluate,
    highfreq_rel_loss,
    predict,
    rel_l2_loss,
)

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent / "results" / "run_logs"
CKPT_DIR = ROOT / "checkpoints"


class DualViewDataset(Dataset):
    def __init__(self, base: SequenceVorticityDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, y = self.base[index]
        h, w = x.shape[-2], x.shape[-1]
        sh1 = int(torch.randint(0, h, (1,)).item())
        sw1 = int(torch.randint(0, w, (1,)).item())
        sh2 = int(torch.randint(0, h, (1,)).item())
        sw2 = int(torch.randint(0, w, (1,)).item())
        xa = torch.roll(x, shifts=(sh1, sw1), dims=(-2, -1))
        ya = torch.roll(y, shifts=(sh1, sw1), dims=(-2, -1))
        xb = torch.roll(x, shifts=(sh2, sw2), dims=(-2, -1))
        return xa, ya, xb, sh1, sw1, sh2, sw2


def dualview_loss(
    model: FNO2d,
    xa,
    ya,
    xb,
    sh1,
    sw1,
    sh2,
    sw2,
    *,
    residual: bool,
    hf_weight: float,
    lambda_delta: float,
    lambda_cons: float,
) -> torch.Tensor:
    pred_a = predict(model, xa, residual)
    loss = rel_l2_loss(pred_a, ya)
    if hf_weight > 0:
        loss = loss + hf_weight * highfreq_rel_loss(pred_a, ya)
    if lambda_delta > 0:
        xl = xa[:, -1:]
        loss = loss + lambda_delta * rel_l2_loss(pred_a - xl, ya - xl)

    if lambda_cons > 0:
        pred_b = predict(model, xb, residual)
        # unroll both predictions to a common (identity) frame via inverse rolls
        # batch may have different shifts — apply per-sample
        b = pred_a.shape[0]
        pa = []
        pb = []
        for i in range(b):
            pa.append(
                torch.roll(
                    pred_a[i],
                    shifts=(-int(sh1[i]), -int(sw1[i])),
                    dims=(-2, -1),
                )
            )
            pb.append(
                torch.roll(
                    pred_b[i],
                    shifts=(-int(sh2[i]), -int(sw2[i])),
                    dims=(-2, -1),
                )
            )
        pa_t = torch.stack(pa, dim=0)
        pb_t = torch.stack(pb, dim=0)
        loss = loss + lambda_cons * rel_l2_loss(pa_t, pb_t.detach())
    return loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--hf-weight", type=float, default=0.12)
    ap.add_argument("--lambda-delta", type=float, default=0.35)
    ap.add_argument("--lambda-cons", type=float, default=0.25)
    ap.add_argument("--early-stop-patience", type=int, default=4)
    ap.add_argument("--baseline", type=float, default=0.03522327123209834)
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--stop-on-gate", action="store_true")
    ap.add_argument("--residual", action="store_true", default=True)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--freeze-spectral", action="store_true", default=True)
    ap.add_argument("--no-freeze-spectral", action="store_true")
    ap.add_argument(
        "--init-from",
        type=str,
        default=str(CKPT_DIR / "fno_ns_public_pf_delta_r1_best.pt"),
    )
    ap.add_argument("--tag", type=str, default="dualview_r1")
    args = ap.parse_args()

    residual = False if args.no_residual else True
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
    train_loader = DataLoader(
        DualViewDataset(SequenceVorticityDataset(train_data, 10, 1)),
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
    if freeze_spectral:
        for name, param in model.named_parameters():
            if "spectral_conv" in name:
                param.requires_grad_(False)
    print(f"init from {args.init_from} freeze_spectral={freeze_spectral}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 3))

    baseline = evaluate(model, test_loader, residual=residual)
    best = baseline
    print(
        f"baseline={baseline:.8f} gate={gate:.8f} λ_cons={args.lambda_cons}",
        flush=True,
    )

    log_path = LOG_DIR / f"fno_public_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    lines = [f"# dualview tag={args.tag} baseline={baseline:.8f} gate={gate:.8f}"]
    stale = 0
    t0 = time.time()
    epoch = -1

    for epoch in range(args.epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for xa, ya, xb, sh1, sw1, sh2, sw2 in train_loader:
            optim.zero_grad()
            loss = dualview_loss(
                model,
                xa,
                ya,
                xb,
                sh1,
                sw1,
                sh2,
                sw2,
                residual=residual,
                hf_weight=args.hf_weight,
                lambda_delta=args.lambda_delta,
                lambda_cons=args.lambda_cons,
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
                    "lambda_cons": args.lambda_cons,
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
        "task": "fno_public_dualview",
        "tag": args.tag,
        "baseline_test_l2": baseline,
        "best_test_l2": best,
        "gate": gate,
        "beat_gate": best < gate,
        "improved_vs_baseline": best < baseline - 1e-9,
        "delta_vs_baseline": baseline - best,
        "epochs_ran": epoch + 1,
        "lambda_cons": args.lambda_cons,
        "lambda_delta": args.lambda_delta,
        "lr": args.lr,
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
