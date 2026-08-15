#!/usr/bin/env python3
"""Rebuild flow-field figures from the official public NS64 file + public weights.

Does not retrain. Does not rewrite summary.json L2 / Spectral ms.
Writes checkpoints/demo_batch.pt then calls visualize.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SequenceVorticityDataset, split_train_test
from model import FNO2d
from visualize import main as visualize_main

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
DATA_FILE = ROOT / "data" / "navier_stokes_v1e-3_N1200_T20.pt"
PUBLIC_CKPT = CKPT_DIR / "fno_ns_public_demo.pt"
DEMO_BATCH = CKPT_DIR / "demo_batch.pt"
DEMO_META = CKPT_DIR / "demo_batch_meta.json"
N_TRAIN, N_TEST, SEED = 1000, 128, 20260722
OFFICIAL_L2 = 0.035012


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    diff = torch.norm(pred - target, dim=(-2, -1))
    ref = torch.norm(target, dim=(-2, -1)).clamp_min(1.0e-12)
    return float((diff / ref).mean().item())


def predict(model: FNO2d, x: torch.Tensor, residual: bool) -> torch.Tensor:
    raw = model(x, use_supa=False)
    if residual:
        return raw + x[:, -1:, :, :]
    return raw


def load_official() -> torch.Tensor:
    if not DATA_FILE.exists():
        raise SystemExit(f"missing official data: {DATA_FILE}")
    payload = torch.load(DATA_FILE, map_location="cpu", weights_only=False)
    tensor = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    if tensor.dim() == 4 and tensor.shape[-1] != tensor.shape[-2]:
        if tensor.shape[1] == tensor.shape[2]:
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return tensor.to(torch.float32)


def main() -> None:
    blob = torch.load(PUBLIC_CKPT, map_location="cpu", weights_only=False)
    residual = bool(blob.get("residual", True))
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
    model.eval()

    data = load_official()
    _, test_data = split_train_test(data, N_TRAIN, N_TEST, seed=SEED)
    loader = DataLoader(
        SequenceVorticityDataset(test_data, 10, 1),
        batch_size=16,
        shuffle=False,
    )
    inputs_all = []
    targets_all = []
    preds_all = []
    with torch.no_grad():
        for inputs, targets in loader:
            preds_all.append(predict(model, inputs, residual))
            inputs_all.append(inputs)
            targets_all.append(targets)
    inputs = torch.cat(inputs_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    pred = torch.cat(preds_all, dim=0)
    torch.save({"input": inputs, "target": targets, "pred": pred}, DEMO_BATCH)

    per = []
    for i in range(pred.shape[0]):
        per.append(relative_l2(pred[i : i + 1], targets[i : i + 1]))
    typical_i = min(range(len(per)), key=lambda i: abs(per[i] - OFFICIAL_L2))
    mean_l2 = float(sum(per) / len(per))
    meta = {
        "schema_version": 1,
        "data": "public_ns64",
        "data_source": f"file:{DATA_FILE.name}",
        "checkpoint": "checkpoints/fno_ns_public_demo.pt",
        "residual": residual,
        "seed": SEED,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "sample_index": typical_i,
        "target_time_step": 10,
        "sample_relative_l2": per[typical_i],
        "test_mean_relative_l2": mean_l2,
        "official_relative_l2": OFFICIAL_L2,
        "n_forwarded": int(pred.shape[0]),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "figures from official NS64 + public_demo.pt; contest L2 is official_relative_l2",
    }
    DEMO_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        {
            "data": meta["data_source"],
            "residual": residual,
            "typical_index": typical_i,
            "typical_rel_l2": per[typical_i],
            "test_mean_relative_l2": mean_l2,
            "official_relative_l2": OFFICIAL_L2,
            "demo_batch": str(DEMO_BATCH),
        }
    )
    visualize_main()


if __name__ == "__main__":
    main()
