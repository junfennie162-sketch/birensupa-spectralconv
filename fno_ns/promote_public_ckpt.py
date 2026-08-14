#!/usr/bin/env python3
"""Promote a public NS64 checkpoint into demo + summary + demo_batch."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d
from train_public_ns64_boost import evaluate, predict, relative_l2

ROOT = Path(__file__).resolve().parent
SUB = ROOT.parent
CKPT = ROOT / "checkpoints"
SUMMARY = SUB / "results" / "summary.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260722)
    args = ap.parse_args()

    src = Path(args.src)
    blob = torch.load(src, map_location="cpu", weights_only=False)
    residual = bool(blob.get("residual", True))
    test_l2_meta = float(blob.get("test_l2", -1))

    data, source = load_or_build_ns_like(
        n_samples=args.n_train + args.n_test,
        resolution=64,
        n_times=20,
        seed=args.seed,
        version="v2",
    )
    if not str(source).startswith("file:navier_stokes"):
        raise SystemExit(source)
    train_data, test_data = split_train_test(
        data, args.n_train, args.n_test, seed=args.seed
    )
    test_loader = DataLoader(
        SequenceVorticityDataset(test_data, 10, 1), batch_size=16, shuffle=False
    )
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
    test_l2 = evaluate(model, test_loader, residual=residual)
    print(
        f"verified_test_l2={test_l2:.12f} meta={test_l2_meta} "
        f"residual={residual} modes={modes} width={width}"
    )

    demo = CKPT / "fno_ns_public_demo.pt"
    best = CKPT / "fno_ns_public_ns64_best.pt"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if demo.exists():
        shutil.copy2(demo, CKPT / f"fno_ns_public_demo.pt.pre_{args.tag}_backup")
    if best.exists():
        shutil.copy2(best, CKPT / f"fno_ns_public_ns64_best.pt.pre_{args.tag}_backup")

    payload = {
        "model": blob["model"],
        "test_l2": test_l2,
        "epoch": blob.get("epoch"),
        "data_source": source,
        "residual": residual,
        "modes": modes,
        "width": width,
        "promoted_tag": args.tag,
        "split": {"n_train": args.n_train, "n_test": args.n_test, "seed": args.seed},
        "promoted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "src": str(src),
    }
    torch.save(payload, demo)
    torch.save(payload, best)

    # rebuild demo_batch in visualize.py format: input/target/pred
    ds = SequenceVorticityDataset(test_data, 10, 1)
    model.eval()
    scores = []
    preds = []
    truths = []
    inputs = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            pred = predict(model, x, residual)
            scores.append(relative_l2(pred, y))
            if i < 16:
                inputs.append(x.squeeze(0))
                truths.append(y.squeeze(0))
                preds.append(pred.squeeze(0))
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    best_i, med_i, worst_i = order[0], order[len(order) // 2], order[-1]
    batch = {
        "input": torch.stack(inputs),
        "target": torch.stack(truths),
        "pred": torch.stack(preds),
    }
    torch.save(batch, CKPT / "demo_batch.pt")
    meta = {
        "schema_version": 1,
        "sample_index": 0,
        "sample_relative_l2": scores[0],
        "batch_mean_relative_l2": test_l2,
        "data": "public_ns64",
        "data_source": source,
        "checkpoint": "checkpoints/fno_ns_public_demo.pt",
        "residual": residual,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_time_step": 10,
        "strip": {
            "best": {"index": best_i, "sample_relative_l2": scores[best_i]},
            "median": {"index": med_i, "sample_relative_l2": scores[med_i]},
            "worst": {"index": worst_i, "sample_relative_l2": scores[worst_i]},
        },
        "promoted_tag": args.tag,
    }
    (CKPT / "demo_batch_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (CKPT / "fno_ns_public_ns64_meta.json").write_text(
        json.dumps(
            {
                "relative_l2": test_l2,
                "promoted_tag": args.tag,
                "residual": residual,
                "checkpoint": "fno_ns/checkpoints/fno_ns_public_demo.pt",
                "updated_at": meta["generated_at"],
            },
            indent=2,
        )
        + "\n"
    )

    summary = json.loads(SUMMARY.read_text())
    ts = meta["generated_at"]
    summary.setdefault("meta", {})["updated_at"] = ts
    summary["meta"]["notes"] = f"Public NS64 FNO L2={test_l2:.6f} ({args.tag})."
    fno = summary.setdefault("fno_ns", {})
    fno["relative_l2"] = test_l2
    fno["l2_note"] = f"PRIMARY public NS64 after {args.tag}"
    fno["checkpoint_primary"] = "fno_ns/checkpoints/fno_ns_public_demo.pt"
    fno["public_ns64"] = {
        "status": f"promoted_{args.tag}",
        "data_source": source,
        "relative_l2": test_l2,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "seed": args.seed,
        "checkpoint": "fno_ns/checkpoints/fno_ns_public_demo.pt",
        "protocol": "official_gate_1000_128",
        "promoted_tag": args.tag,
    }
    viz = fno.setdefault("visualization", {})
    viz["sample_relative_l2"] = scores[0]
    viz["data"] = "public_ns64"
    viz["strip"] = meta["strip"]
    viz["note"] = f"Figures after {args.tag} promote ({stamp})."
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    # mark matching probe summary promote=true
    for probe_name in (
        f"fno_public_sched_sampling_{args.tag}_summary.json",
        f"fno_public_{args.tag}_summary.json",
        f"fno_public_boost_{args.tag}_summary.json",
    ):
        probe = SUB / "results" / "run_logs" / probe_name
        if probe.exists():
            p = json.loads(probe.read_text())
            p["promote"] = True
            p["promoted_test_l2"] = test_l2
            probe.write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({"promoted": True, "tag": args.tag, "test_l2": test_l2, "strip": meta["strip"]}, indent=2))


if __name__ == "__main__":
    main()
