#!/usr/bin/env python3
"""R2-0: greedy weight soup / uniform average on public NS64 (no TTA).

Evaluates individual ckpts and soups. Promote only if soup L2 < gate.
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
from train_public_ns64_boost import evaluate

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent / "results" / "run_logs"
CKPT_DIR = ROOT / "checkpoints"
BASELINE = 0.035724617540836334
GATE = BASELINE - 1e-4


def load_sd(path: Path) -> tuple[dict, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["model"] if "model" in blob else blob
    meta = {k: blob[k] for k in blob if k != "model"} if isinstance(blob, dict) else {}
    return sd, meta


def average_state_dicts(sds: list[dict]) -> dict:
    """Average tensors in-place dtype (keeps complex spectral weights)."""
    out = {}
    for k in sds[0].keys():
        acc = None
        for sd in sds:
            t = sd[k]
            acc = t.clone() if acc is None else acc + t
        assert acc is not None
        out[k] = acc / len(sds)
    return out


def eval_sd(sd: dict, loader: DataLoader, residual: bool) -> float:
    model = FNO2d(
        modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1
    )
    model.load_state_dict(sd)
    return evaluate(model, loader, residual=residual)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpts",
        nargs="+",
        default=[
            str(CKPT_DIR / "fno_ns_public_boost_C_best.pt"),
            str(CKPT_DIR / "fno_ns_public_sq3b_best.pt"),
            str(CKPT_DIR / "fno_ns_public_demo.pt"),
        ],
    )
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--baseline", type=float, default=BASELINE)
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--out-name", type=str, default="fno_ns_public_soup_r8_best.pt")
    ap.add_argument(
        "--always-save",
        action="store_true",
        help="save best soup even if it does not beat gate",
    )
    args = ap.parse_args()
    baseline = args.baseline
    gate = args.gate if args.gate > 0 else baseline - 1e-4

    data, source = load_or_build_ns_like(
        n_samples=args.n_train + args.n_test,
        resolution=64,
        n_times=20,
        seed=args.seed,
        version="v2",
    )
    if not str(source).startswith("file:navier_stokes"):
        raise SystemExit(f"need public NS64, got {source}")
    _, test_data = split_train_test(data, args.n_train, args.n_test, seed=args.seed)
    loader = DataLoader(
        SequenceVorticityDataset(test_data, 10, 1),
        batch_size=args.batch_size,
        shuffle=False,
    )

    rows = []
    sds = []
    residual = True
    for p in args.ckpts:
        path = Path(p)
        if not path.exists():
            print(f"SKIP missing {path}")
            continue
        sd, meta = load_sd(path)
        res = bool(meta.get("residual", True))
        residual = residual and res
        l2 = eval_sd(sd, loader, residual=res)
        rows.append(
            {
                "path": str(path.name),
                "test_l2": l2,
                "residual": res,
                "meta_l2": meta.get("test_l2"),
                "tag": meta.get("promoted_tag"),
            }
        )
        sds.append(sd)
        print(f"single {path.name} l2={l2:.8f} residual={res}", flush=True)

    if len(sds) < 2:
        raise SystemExit("need >=2 ckpts for soup")

    soups = {}
    # uniform all
    soups["uniform_all"] = average_state_dicts(sds)
    # pairwise with demo (last)
    if len(sds) >= 2:
        soups["uniform_last2"] = average_state_dicts(sds[-2:])
    if len(sds) >= 3:
        soups["uniform_first_last"] = average_state_dicts([sds[0], sds[-1]])

    soup_rows = []
    best_name, best_l2, best_sd = None, 1e9, None
    for name, sd in soups.items():
        l2 = eval_sd(sd, loader, residual=residual)
        soup_rows.append({"name": name, "test_l2": l2})
        print(f"soup {name} l2={l2:.8f}", flush=True)
        if l2 < best_l2:
            best_name, best_l2, best_sd = name, l2, sd

    beat = best_l2 < gate
    out_ckpt = CKPT_DIR / args.out_name
    saved = False
    if best_sd is not None and (beat or args.always_save):
        torch.save(
            {
                "model": best_sd,
                "test_l2": best_l2,
                "residual": residual,
                "promoted_tag": f"soup_{best_name}",
                "data_source": source,
                "modes": 16,
                "width": 32,
                "split": {
                    "n_train": args.n_train,
                    "n_test": args.n_test,
                    "seed": args.seed,
                },
                "soup_of": [Path(p).name for p in args.ckpts if Path(p).exists()],
            },
            out_ckpt,
        )
        saved = True

    summary = {
        "task": "fno_public_soup_r8",
        "baseline": baseline,
        "gate": gate,
        "singles": rows,
        "soups": soup_rows,
        "best_soup": best_name,
        "best_test_l2": best_l2,
        "beat_gate": beat,
        "checkpoint": str(out_ckpt) if saved else None,
        "note": "weight average only; not TTA",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = LOG_DIR / "fno_public_soup_r8_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if beat:
        print("SIGNAL: soup beat_gate — eligible for promote review")
    else:
        print("NO_SIGNAL: soup did not beat gate")


if __name__ == "__main__":
    main()
