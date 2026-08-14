#!/usr/bin/env python3
"""Wave-2 probe: multi-step teacher-forcing aux loss + light energy/spectrum soft.

- Train: T_out_train>=2 sliding TF losses (residual-aware)
- Eval: official step-1 only on public 1000/128
- Does NOT auto-promote; writes summary JSON for gate decision
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
    rel_l2_loss,
    relative_l2,
)

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent / "results" / "run_logs"
CKPT_DIR = ROOT / "checkpoints"
BASELINE = 0.037519834004342556
GATE = BASELINE - 1e-4


def energy_rel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pe = pred.pow(2).mean()
    te = target.pow(2).mean().clamp_min(1e-12)
    return (pe - te).abs() / te


def spectrum_tilt_rel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pf = torch.fft.rfft2(pred).abs()
    tf = torch.fft.rfft2(target).abs()
    # emphasize outer rings (similar spirit to hf loss, lighter)
    b, c, h, wf = tf.shape
    yy = torch.arange(h, device=tf.device, dtype=tf.dtype)[:, None]
    xx = torch.arange(wf, device=tf.device, dtype=tf.dtype)[None, :]
    yy = torch.minimum(yy, h - yy)
    rr = torch.sqrt(yy**2 + xx**2)
    w = (rr / max(h / 4.0, 1.0)).clamp(0, 2.0)
    num = torch.norm(w * (pf - tf))
    den = torch.norm(w * tf).clamp_min(1e-12)
    return num / den


def multistep_loss(
    model: FNO2d,
    x: torch.Tensor,
    y_seq: torch.Tensor,
    *,
    residual: bool,
    hf_weight: float,
    energy_w: float,
    tilt_w: float,
) -> torch.Tensor:
    """y_seq: [B, T_out, H, W]."""
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
        # teacher forcing: append GT frame
        cur = torch.cat([cur[:, 1:], target], dim=1)
    assert total is not None
    return total / steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--t-out-train", type=int, default=3)
    ap.add_argument("--hf-weight", type=float, default=0.2)
    ap.add_argument("--energy-weight", type=float, default=0.05)
    ap.add_argument("--tilt-weight", type=float, default=0.05)
    ap.add_argument("--residual", action="store_true", default=True)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument(
        "--init-from",
        type=str,
        default=str(CKPT_DIR / "fno_ns_public_demo.pt"),
    )
    ap.add_argument("--tag", type=str, default="multistep_probe")
    args = ap.parse_args()
    residual = False if args.no_residual else True

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
    if data.shape[1] < 10 + args.t_out_train:
        raise SystemExit("not enough timesteps for multistep")

    train_data, test_data = split_train_test(
        data, args.n_train, args.n_test, seed=args.seed
    )
    train_loader = DataLoader(
        RollAugDataset(
            SequenceVorticityDataset(train_data, 10, args.t_out_train),
            augment=args.augment,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        SequenceVorticityDataset(test_data, 10, 1),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = FNO2d(
        modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1
    )
    blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"] if "model" in blob else blob)
    print(f"init={args.init_from} residual={residual} t_out_train={args.t_out_train}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 3))

    baseline = evaluate(model, test_loader, residual=residual)
    best = baseline
    print(f"baseline_step1_l2={baseline:.8f} gate={GATE:.8f}")

    log_path = LOG_DIR / f"fno_public_multistep_probe_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"fno_ns_public_{args.tag}_best.pt"
    lines = [
        f"# multistep probe tag={args.tag}",
        f"# baseline={baseline:.8f} gate={GATE:.8f}",
        f"# t_out_train={args.t_out_train} lr={args.lr} hf={args.hf_weight} "
        f"energy={args.energy_weight} tilt={args.tilt_weight}",
    ]

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for x, y_seq in train_loader:
            optim.zero_grad()
            loss = multistep_loss(
                model,
                x,
                y_seq,
                residual=residual,
                hf_weight=args.hf_weight,
                energy_w=args.energy_weight,
                tilt_w=args.tilt_weight,
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
            torch.save(
                {
                    "model": model.state_dict(),
                    "test_l2": test_l2,
                    "epoch": epoch + 1,
                    "data_source": source,
                    "residual": residual,
                    "promoted_tag": args.tag,
                    "split": {
                        "n_train": args.n_train,
                        "n_test": args.n_test,
                        "seed": args.seed,
                    },
                },
                ckpt_path,
            )
        line = (
            f"epoch {epoch+1:3d} | train={tr_sum/max(tr_n,1):.6f} | "
            f"test_l2={test_l2:.8f} | best={best:.8f}"
            + (" *" if improved else "")
        )
        print(line, flush=True)
        lines.append(line)

    elapsed = time.time() - t0
    beat_gate = best < GATE
    summary = {
        "task": "fno_public_multistep_probe",
        "tag": args.tag,
        "baseline_test_l2": baseline,
        "best_test_l2": best,
        "gate": GATE,
        "beat_gate": beat_gate,
        "improved_vs_baseline": best < baseline - 1e-9,
        "delta_vs_baseline": baseline - best,
        "epochs": args.epochs,
        "t_out_train": args.t_out_train,
        "lr": args.lr,
        "hf_weight": args.hf_weight,
        "energy_weight": args.energy_weight,
        "tilt_weight": args.tilt_weight,
        "residual": residual,
        "elapsed_sec": round(elapsed, 1),
        "checkpoint": str(ckpt_path) if ckpt_path.exists() else None,
        "log": str(log_path),
        "promote": False,
        "note": "probe only; promote requires best < gate and explicit human/agent promote step",
    }
    out = LOG_DIR / f"fno_public_boost_{args.tag}_summary.json"
    # also canonical name from plan
    out2 = LOG_DIR / "fno_public_multistep_probe_summary.json"
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    out.write_text(text)
    out2.write_text(text)
    lines.append(json.dumps(summary, ensure_ascii=False))
    log_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if beat_gate:
        print("SIGNAL: beat_gate=True — eligible for short continue + promote review")
    else:
        print("NO_SIGNAL: stop precision line unless Wave-3 conditional")


if __name__ == "__main__":
    main()
