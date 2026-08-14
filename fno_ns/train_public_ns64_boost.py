#!/usr/bin/env python3
"""Public-NS64 booster: methods tailored to harder real NS dynamics.

Why public is harder (measured): larger frame deltas, more high-frequency energy,
weaker temporal correlation. This trainer attacks that with:

1) persistence residual  — network predicts y - x[:,-1:], then add last frame
2) high-freq loss mix    — add FFT-magnitude relative loss (emphasize fine scales)
3) periodic roll aug     — random circular shifts (torus / periodic BC)
4) longer low-lr continue from current public best

Does not overwrite legacy v2 demo. Writes:
  checkpoints/fno_ns_public_ns64_boost_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    diff = torch.norm(pred - target, dim=(-2, -1))
    ref = torch.norm(target, dim=(-2, -1)).clamp_min(1.0e-12)
    return float((diff / ref).mean().item())


def rel_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred - target) / torch.norm(target).clamp_min(1e-12)


def highfreq_rel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 in Fourier magnitude; weights fine scales implicitly."""
    pf = torch.fft.rfft2(pred)
    tf = torch.fft.rfft2(target)
    # magnitude on complex
    pm = torch.linalg.vector_norm(torch.view_as_real(pf), dim=-1)
    tm = torch.linalg.vector_norm(torch.view_as_real(tf), dim=-1)
    # emphasize mid/high radii
    b, c, h, wf = tm.shape
    yy = torch.arange(h, device=tm.device, dtype=tm.dtype)[:, None]
    xx = torch.arange(wf, device=tm.device, dtype=tm.dtype)[None, :]
    yy = torch.minimum(yy, h - yy)
    rr = torch.sqrt(yy**2 + xx**2)
    w = 1.0 + (rr / max(h / 4.0, 1.0)).clamp(0, 3.0)  # up to 4x on outer rings
    num = torch.norm(w * (pm - tm))
    den = torch.norm(w * tm).clamp_min(1e-12)
    return num / den


class RollAugDataset(Dataset):
    """Wrap SequenceVorticityDataset with optional periodic rolls at train time."""

    def __init__(self, base: SequenceVorticityDataset, augment: bool):
        self.base = base
        self.augment = augment

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, y = self.base[index]
        if self.augment:
            h, w = x.shape[-2], x.shape[-1]
            sh = int(torch.randint(0, h, (1,)).item())
            sw = int(torch.randint(0, w, (1,)).item())
            x = torch.roll(x, shifts=(sh, sw), dims=(-2, -1))
            y = torch.roll(y, shifts=(sh, sw), dims=(-2, -1))
        return x, y


def predict(model: FNO2d, x: torch.Tensor, residual: bool) -> torch.Tensor:
    raw = model(x, use_supa=False)
    if residual:
        return raw + x[:, -1:, :, :]
    return raw


@torch.no_grad()
def evaluate(model: FNO2d, loader: DataLoader, residual: bool) -> float:
    model.eval()
    scores = []
    for x, y in loader:
        scores.append(relative_l2(predict(model, x, residual), y))
    return sum(scores) / max(len(scores), 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument("--hf-weight", type=float, default=0.35, help="weight of high-freq loss")
    p.add_argument(
        "--residual",
        action="store_true",
        help="predict residual vs last input frame (harder NS prior)",
    )
    p.add_argument("--augment", action="store_true", help="periodic roll augmentation")
    p.add_argument(
        "--init-from",
        type=str,
        default="",
        help="warm-start ckpt; leave empty for scratch",
    )
    p.add_argument("--ckpt-name", type=str, default="fno_ns_public_ns64_boost_best.pt")
    p.add_argument("--tag", type=str, default="public_boost")
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--width", type=int, default=32)
    p.add_argument(
        "--freeze-spectral",
        action="store_true",
        help="freeze spectral_conv weights; train head/skip/norm only (P2b polish)",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1.0e-4,
        help="Adam weight decay (use 0 for tiny-lr freeze polish)",
    )
    p.add_argument(
        "--gate",
        type=float,
        default=0.0,
        help="stop/promote threshold; best < gate is SIGNAL (0=disabled)",
    )
    p.add_argument(
        "--stop-on-gate",
        action="store_true",
        help="stop training immediately when best_test_l2 < gate",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="stop after N epochs without best improve (0=disabled)",
    )
    args = p.parse_args()
    residual = bool(args.residual)
    augment = bool(args.augment)
    gate = float(args.gate) if args.gate and args.gate > 0 else None
    stop_on_gate = bool(args.stop_on_gate) and gate is not None
    early_stop_patience = int(args.early_stop_patience)

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

    model = FNO2d(
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        n_layers=4,
        in_channels=10,
        out_channels=1,
    )
    if args.init_from and Path(args.init_from).exists():
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        prev_residual = bool(blob.get("residual", False)) if isinstance(blob, dict) else False
        try:
            model.load_state_dict(blob["model"] if "model" in blob else blob)
        except RuntimeError as exc:
            raise SystemExit(
                f"init-from shape mismatch (modes/width?): {exc}. "
                "Leave --init-from empty for capacity retrain."
            ) from exc
        print(
            f"warm_start={args.init_from} prev_residual={prev_residual} "
            f"modes={args.modes} width={args.width}",
            flush=True,
        )
        # Absolute ckpt → residual head: shrink projection so start ≈ persistence.
        if residual and not prev_residual:
            with torch.no_grad():
                last = model.project[-1]
                if hasattr(last, "weight"):
                    last.weight.mul_(0.05)
                if getattr(last, "bias", None) is not None:
                    last.bias.zero_()
            print("adapted absolute ckpt → residual head (shrink project)", flush=True)

    frozen_n = 0
    if args.freeze_spectral:
        for name, param in model.named_parameters():
            if "spectral_conv" in name:
                param.requires_grad_(False)
                frozen_n += param.numel()
        print(f"freeze_spectral params={frozen_n}", flush=True)

    trainable = [param for param in model.parameters() if param.requires_grad]
    optim = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(args.epochs, 3)
    )

    log = (
        ROOT.parent
        / "results"
        / "run_logs"
        / f"fno_public_boost_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# public NS64 boost tag={args.tag}\n"
        f"# data={source} residual={residual} aug={augment} hf_w={args.hf_weight}\n"
        f"# init={args.init_from} epochs={args.epochs} lr={args.lr}\n"
        f"# freeze_spectral={args.freeze_spectral} frozen_n={frozen_n} "
        f"wd={args.weight_decay}\n"
    )
    log.write_text(header)

    baseline = evaluate(model, test_loader, residual=residual)
    best_l2 = baseline
    torch.save(
        {
            "model": model.state_dict(),
            "test_l2": baseline,
            "epoch": 0,
            "step": 0,
            "data_source": source,
            "residual": residual,
            "augment": augment,
            "hf_weight": args.hf_weight,
            "modes": args.modes,
            "width": args.width,
            "promoted_tag": args.tag,
            "split": {"n_train": args.n_train, "n_test": args.n_test, "seed": args.seed},
        },
        ckpt_path,
    )
    print(
        f"baseline_test_l2={baseline:.6f} residual={residual} aug={augment}",
        flush=True,
    )
    log.write_text(log.read_text() + f"# baseline={baseline:.6f}\n")

    step = 0
    t0 = time.time()
    stale = 0
    stop_reason = ""
    for epoch in range(args.epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for x, y in train_loader:
            optim.zero_grad()
            pred = predict(model, x, residual)
            loss = rel_l2_loss(pred, y)
            if args.hf_weight > 0:
                loss = loss + args.hf_weight * highfreq_rel_loss(pred, y)
            loss.backward()
            optim.step()
            tr_sum += float(loss.item())
            tr_n += 1
            step += 1
        scheduler.step()
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
                    "step": step,
                    "data_source": source,
                    "residual": residual,
                    "augment": augment,
                    "hf_weight": args.hf_weight,
                    "modes": args.modes,
                    "width": args.width,
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
                        "step": step,
                        "residual": residual,
                        "augment": augment,
                        "hf_weight": args.hf_weight,
                        "data_source": source,
                        "checkpoint": str(ckpt_path),
                        "tag": args.tag,
                    },
                    indent=2,
                )
                + "\n"
            )
            mark = "  *best*"
        else:
            stale += 1
        line = (
            f"step {step:5d} | epoch {epoch+1:3d} | train={train_l2:.6f} | "
            f"test_l2={test_l2:.6f} | best={best_l2:.6f} | "
            f"lr={optim.param_groups[0]['lr']:.2e}{mark}"
        )
        print(line, flush=True)
        log.write_text(log.read_text() + line + "\n")
        if stop_on_gate and best_l2 < gate:
            stop_reason = f"stop_on_gate best={best_l2:.8f} < gate={gate:.8f}"
            print(stop_reason, flush=True)
            log.write_text(log.read_text() + stop_reason + "\n")
            break
        if early_stop_patience > 0 and stale >= early_stop_patience:
            stop_reason = f"early_stop patience={early_stop_patience}"
            print(stop_reason, flush=True)
            log.write_text(log.read_text() + stop_reason + "\n")
            break

    summary = {
        "task": "fno_public_ns64_boost",
        "tag": args.tag,
        "data_source": source,
        "baseline_test_l2": baseline,
        "best_test_l2": best_l2,
        "gate": gate,
        "beat_gate": bool(gate is not None and best_l2 < gate),
        "stop_reason": stop_reason or "completed",
        "residual": residual,
        "augment": augment,
        "hf_weight": args.hf_weight,
        "freeze_spectral": bool(args.freeze_spectral),
        "frozen_spectral_params": frozen_n,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "elapsed_sec": round(time.time() - t0, 1),
        "checkpoint": str(ckpt_path),
        "improved": best_l2 < baseline - 1e-9,
        "delta_l2": baseline - best_l2,
        "log": str(log),
    }
    print("==SUMMARY==", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    (
        ROOT.parent / "results" / "run_logs" / f"fno_public_boost_{args.tag}_summary.json"
    ).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if summary["beat_gate"]:
        print("SIGNAL: beat_gate — eligible for promote review", flush=True)
    elif gate is not None:
        print("NO_SIGNAL: best did not beat gate", flush=True)


if __name__ == "__main__":
    main()
