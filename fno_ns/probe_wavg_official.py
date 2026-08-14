#!/usr/bin/env python3
"""Weight-average existing official-split checkpoints; report test L2 on 1000/128.

Does not overwrite demo unless --promote and avg beats current demo.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d
from test_forward import relative_l2

THIS_DIR = Path(__file__).resolve().parent
DEMO_CKPT = THIS_DIR / "checkpoints" / "fno_ns_demo.pt"
BEST_CKPT = THIS_DIR / "checkpoints" / "fno_ns_official_best.pt"
OUT_CKPT = THIS_DIR / "checkpoints" / "fno_ns_wavg_official.pt"
OUT_META = THIS_DIR / "checkpoints" / "fno_ns_wavg_official_meta.json"
SEED = 20260722


def evaluate(model: FNO2d, loader: DataLoader) -> float:
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in loader:
            scores.append(relative_l2(model(x, use_supa=False), y))
    return sum(scores) / max(len(scores), 1)


def average_state_dicts(paths: list[Path]) -> dict:
    states = []
    for p in paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        states.append(ckpt["model"])
    avg = {}
    keys = states[0].keys()
    for k in keys:
        # Keep complex weights as complex; .float() would drop imag.
        stacked = torch.stack([s[k].to(dtype=torch.float32 if not s[k].is_complex() else torch.complex64) for s in states], dim=0)
        mean = stacked.mean(dim=0)
        avg[k] = mean.to(dtype=states[0][k].dtype)
    return avg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpts",
        nargs="+",
        default=[
            str(THIS_DIR / "checkpoints" / "fno_ns_freeze_win_005470.pt"),
            str(THIS_DIR / "checkpoints" / "fno_ns_freeze_win_005473.pt"),
            str(THIS_DIR / "checkpoints" / "fno_ns_official_best.pt"),
        ],
    )
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    paths = [Path(p) for p in args.ckpts if Path(p).exists()]
    if len(paths) < 2:
        raise SystemExit(f"need >=2 existing ckpts, got {paths}")

    data, src = load_or_build_ns_like(
        n_samples=args.n_train + args.n_test,
        resolution=64,
        n_times=30,
        seed=SEED,
        version="v2",
    )
    _, te = split_train_test(data, args.n_train, args.n_test, seed=SEED)
    test_loader = DataLoader(
        SequenceVorticityDataset(te, 10, 1),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = FNO2d(modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1)
    per = {}
    for p in paths:
        model.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"])
        per[str(p.name)] = evaluate(model, test_loader)

    demo_l2 = float("inf")
    if DEMO_CKPT.exists():
        model.load_state_dict(
            torch.load(DEMO_CKPT, map_location="cpu", weights_only=False)["model"]
        )
        demo_l2 = evaluate(model, test_loader)

    avg_state = average_state_dicts(paths)
    model.load_state_dict(avg_state)
    avg_l2 = evaluate(model, test_loader)

    torch.save(
        {
            "model": avg_state,
            "test_l2": avg_l2,
            "sources": [str(p) for p in paths],
            "per_source_l2": per,
            "data_source": src,
        },
        OUT_CKPT,
    )

    improved = avg_l2 < demo_l2 - 1e-9
    meta = {
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": [str(p) for p in paths],
        "per_source_l2": per,
        "demo_test_l2": demo_l2,
        "wavg_test_l2": avg_l2,
        "improved": improved,
        "delta": (demo_l2 - avg_l2) if demo_l2 < float("inf") else None,
        "checkpoint": str(OUT_CKPT),
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(meta, flush=True)

    if improved and args.promote:
        import shutil

        shutil.copy2(DEMO_CKPT, DEMO_CKPT.with_suffix(".pt.pre_wavg_backup"))
        shutil.copy2(OUT_CKPT, DEMO_CKPT)
        shutil.copy2(OUT_CKPT, BEST_CKPT)
        print({"promoted": True, "new_l2": avg_l2}, flush=True)
    else:
        print({"promoted": False, "improved": improved}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
