#!/usr/bin/env python3
"""Geometric symmetry aug (flip/90°) + optional roll + light noise; public NS64.

Eval step-1 only. Does NOT auto-promote.
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
from train_public_multistep_probe import multistep_loss
from train_public_ns64_boost import evaluate

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent / "results" / "run_logs"
CKPT_DIR = ROOT / "checkpoints"


class GeomNoiseDataset(Dataset):
    def __init__(
        self,
        base: SequenceVorticityDataset,
        *,
        noise_std: float,
        enable_geom: bool,
        enable_roll: bool,
    ):
        self.base = base
        self.noise_std = noise_std
        self.enable_geom = enable_geom
        self.enable_roll = enable_roll

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, y = self.base[index]
        if self.enable_roll:
            h, w = x.shape[-2], x.shape[-1]
            sh = int(torch.randint(0, h, (1,)).item())
            sw = int(torch.randint(0, w, (1,)).item())
            x = torch.roll(x, shifts=(sh, sw), dims=(-2, -1))
            y = torch.roll(y, shifts=(sh, sw), dims=(-2, -1))
        if self.enable_geom:
            op = int(torch.randint(0, 4, (1,)).item())
            if op == 1:
                x = torch.flip(x, dims=(-1,))
                y = torch.flip(y, dims=(-1,))
            elif op == 2:
                x = torch.flip(x, dims=(-2,))
                y = torch.flip(y, dims=(-2,))
            elif op == 3:
                x = torch.rot90(x, k=1, dims=(-2, -1))
                y = torch.rot90(y, k=1, dims=(-2, -1))
        if self.noise_std > 0:
            x = x + self.noise_std * torch.randn_like(x)
        return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--t-out-train", type=int, default=3)
    ap.add_argument("--hf-weight", type=float, default=0.2)
    ap.add_argument("--energy-weight", type=float, default=0.05)
    ap.add_argument("--tilt-weight", type=float, default=0.05)
    ap.add_argument("--noise-std", type=float, default=0.005)
    ap.add_argument("--enable-roll", action="store_true", default=True)
    ap.add_argument("--no-roll", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=3)
    ap.add_argument("--abort-ep6-delta", type=float, default=1e-4)
    ap.add_argument("--baseline", type=float, default=0.03585514333099127)
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--residual", action="store_true", default=True)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument(
        "--init-from",
        type=str,
        default=str(CKPT_DIR / "fno_ns_public_demo.pt"),
    )
    ap.add_argument("--tag", type=str, default="geom_roll_r4")
    args = ap.parse_args()
    residual = False if args.no_residual else True
    enable_roll = False if args.no_roll else True
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
        GeomNoiseDataset(
            SequenceVorticityDataset(train_data, 10, args.t_out_train),
            noise_std=args.noise_std,
            enable_geom=True,
            enable_roll=enable_roll,
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
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 3))

    baseline = evaluate(model, test_loader, residual=residual)
    best = baseline
    print(f"baseline={baseline:.8f} gate={gate:.8f} roll={enable_roll}", flush=True)
    ckpt_path = CKPT_DIR / f"fno_ns_public_{args.tag}_best.pt"
    log_path = LOG_DIR / f"fno_public_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    lines = [f"# geom+noise tag={args.tag} baseline={baseline:.8f}"]
    stale = 0
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
            stale = 0
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
        else:
            stale += 1
        line = (
            f"epoch {epoch+1:3d} | train={tr_sum/max(tr_n,1):.6f} | "
            f"test_l2={test_l2:.8f} | best={best:.8f}"
            + (" *" if improved else "")
        )
        print(line, flush=True)
        lines.append(line)
        if epoch + 1 >= 6 and (baseline - best) < args.abort_ep6_delta:
            lines.append("abort: ep6 delta < 1e-4")
            print("abort ep6", flush=True)
            break
        if stale >= args.early_stop_patience:
            lines.append("early_stop")
            print("early_stop", flush=True)
            break

    summary = {
        "task": "fno_public_geom_noise_r2",
        "tag": args.tag,
        "baseline_test_l2": baseline,
        "best_test_l2": best,
        "gate": gate,
        "beat_gate": best < gate,
        "enable_roll": enable_roll,
        "improved_vs_baseline": best < baseline - 1e-9,
        "delta_vs_baseline": baseline - best,
        "noise_std": args.noise_std,
        "elapsed_sec": round(time.time() - t0, 1),
        "checkpoint": str(ckpt_path) if ckpt_path.exists() else None,
        "log": str(log_path),
        "promote": False,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    (LOG_DIR / "fno_public_geom_noise_r2_summary.json").write_text(text)
    (LOG_DIR / f"fno_public_boost_{args.tag}_summary.json").write_text(text)
    lines.append(json.dumps(summary, ensure_ascii=False))
    log_path.write_text("\n".join(lines) + "\n")
    print(text)


if __name__ == "__main__":
    main()
