#!/usr/bin/env python3
"""Train or reproducibly evaluate the FNO-NS checkpoint.

The default mode is evaluation-only so submission regression never retrains or
clobbers the 110-epoch checkpoint. Pass ``--train`` explicitly to reproduce the
canonical training configuration.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d, count_parameters

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "fno_ns_demo.pt"
CHECKPOINT_META_PATH = CHECKPOINT_DIR / "checkpoint_meta.json"
DEMO_BATCH_PATH = CHECKPOINT_DIR / "demo_batch.pt"
DEMO_BATCH_META_PATH = CHECKPOINT_DIR / "demo_batch_meta.json"

RESOLUTION = 64
T_IN = 10
T_OUT = 1
MODES = 16
WIDTH = 32
N_LAYERS = 4
DEFAULT_N_TRAIN = 768
DEFAULT_N_TEST = 128
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 110
DEFAULT_LEARNING_RATE = 5.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-4
SEED = 20260722


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    difference_norm = torch.norm(prediction - target, dim=(-2, -1))
    target_norm = torch.norm(target, dim=(-2, -1)).clamp_min(1.0e-12)
    return float((difference_norm / target_norm).mean().item())


def train_cpu(
    model: FNO2d,
    data_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> list[float]:
    optimizer = torch.optim.Adam(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[float] = []
    model.train()
    for epoch_index in range(epochs):
        total_relative_l2 = 0.0
        batch_count = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            prediction = model(inputs, use_supa=False)
            loss = torch.norm(prediction - targets) / torch.norm(targets).clamp_min(1.0e-12)
            loss.backward()
            optimizer.step()
            total_relative_l2 += float(loss.item())
            batch_count += 1
        scheduler.step()
        epoch_relative_l2 = total_relative_l2 / max(batch_count, 1)
        history.append(epoch_relative_l2)
        if epoch_index == 0 or (epoch_index + 1) % 10 == 0:
            print({"epoch": epoch_index + 1, "train_relative_l2": epoch_relative_l2})
    return history


@torch.no_grad()
def evaluate(model: FNO2d, data_loader: DataLoader, use_supa: bool):
    model.eval()
    batch_scores: list[float] = []
    first_batch = None
    for inputs, targets in data_loader:
        prediction = model(inputs, use_supa=use_supa)
        batch_scores.append(relative_l2(prediction, targets))
        if first_batch is None:
            first_batch = (inputs.cpu(), targets.cpu(), prediction.cpu())
    assert first_batch is not None and batch_scores
    return (
        sum(batch_scores) / len(batch_scores),
        first_batch[0],
        first_batch[1],
        first_batch[2],
    )


def build_data_loaders(n_train: int, n_test: int, batch_size: int):
    data, data_source = load_or_build_ns_like(
        n_samples=max(1024, n_train + n_test),
        resolution=RESOLUTION,
        n_times=max(T_IN + 20, 30),
        seed=SEED,
        version="v2",
    )
    train_data, test_data = split_train_test(data, n_train, n_test, seed=SEED)
    train_loader = DataLoader(
        SequenceVorticityDataset(train_data, T_IN, T_OUT),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        SequenceVorticityDataset(test_data, T_IN, T_OUT),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader, data_source, data


def make_model() -> FNO2d:
    return FNO2d(
        modes1=MODES,
        modes2=MODES,
        width=WIDTH,
        n_layers=N_LAYERS,
        in_channels=T_IN,
        out_channels=T_OUT,
    )


def write_metadata(
    *,
    mode: str,
    data_source: str,
    n_train: int,
    n_test: int,
    batch_size: int,
    epochs: int,
    history: list[float],
    rel_l2_torch: float,
    rel_l2_supa: float,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    optimizer_steps = epochs * math.ceil(n_train / batch_size)
    checkpoint_meta = {
        "schema_version": 1,
        "mode": mode,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(RESULTS_DIR.parent)),
        "data": data_source,
        "data_disclosure": "self-generated NS-like v2; not public NS64",
        "resolution": RESOLUTION,
        "t_in": T_IN,
        "t_out": T_OUT,
        "n_train": n_train,
        "n_val": 0,
        "n_test": n_test,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "seed": SEED,
        "model": {"modes": MODES, "width": WIDTH, "fourier_layers": N_LAYERS},
        "training": {
            "optimizer": "Adam",
            "learning_rate": DEFAULT_LEARNING_RATE,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR for the canonical training stage",
            "optimizer_state_available": False,
            "scheduler_state_available": False,
            "history_points": len(history),
            "history_interval_epochs": 1,
        },
        "evaluation": {
            "rel_l2_torch": rel_l2_torch,
            "rel_l2_supa": rel_l2_supa,
            "evaluated_at": timestamp,
        },
    }
    CHECKPOINT_META_PATH.write_text(json.dumps(checkpoint_meta, indent=2) + "\n")

    sample_relative_l2 = relative_l2(predictions[0:1], targets[0:1])
    demo_meta = {
        "schema_version": 1,
        "sample_index": 0,
        "input_shape": list(inputs[0].shape),
        "target_shape": list(targets[0].shape),
        "prediction_shape": list(predictions[0].shape),
        "input_time_steps": list(range(T_IN)),
        "target_time_step": T_IN,
        "sample_relative_l2": sample_relative_l2,
        "data": data_source,
        "seed": SEED,
        "generated_at": timestamp,
    }
    DEMO_BATCH_META_PATH.write_text(json.dumps(demo_meta, indent=2) + "\n")


def update_summary(
    *,
    data_source: str,
    n_train: int,
    n_test: int,
    batch_size: int,
    epochs: int,
    history: list[float],
    rel_l2_torch: float,
    rel_l2_supa: float,
) -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    fno_summary = summary.setdefault("fno_ns", {})
    fno_summary.update(
        {
            "status": "checkpoint_evaluation_pass",
            "rel_l2": rel_l2_supa,
            "rel_l2_torch_path": rel_l2_torch,
            "fourier_layers": N_LAYERS,
            "resolution": RESOLUTION,
            "modes": MODES,
            "width": WIDTH,
            "t_in": T_IN,
            "data": data_source,
            "data_disclosure": "Self-generated NS-like v2; not public NS64. See results/data_disclosure.md.",
            "n_train": n_train,
            "n_val": 0,
            "n_test": n_test,
            "batch_size": batch_size,
            "train_epochs": epochs,
            "optimizer_steps": epochs * math.ceil(n_train / batch_size),
            "seed": SEED,
            "train_rel_l2_history": history,
            "checkpoint": str(CHECKPOINT_PATH),
            "checkpoint_meta": str(CHECKPOINT_META_PATH.relative_to(RESULTS_DIR.parent)),
        }
    )
    summary.setdefault("meta", {})["updated_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="explicitly train and replace checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.epochs > 0 and args.n_train > 0 and args.n_test > 0 and args.batch_size > 0
    torch.manual_seed(SEED)
    train_loader, test_loader, data_source, data = build_data_loaders(
        args.n_train, args.n_test, args.batch_size
    )
    model = make_model()
    history: list[float]
    mode = "train" if args.train else "evaluate_checkpoint"

    if args.train:
        history = train_cpu(
            model,
            train_loader,
            args.epochs,
            DEFAULT_LEARNING_RATE,
            DEFAULT_WEIGHT_DECAY,
        )
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "history": history, "data_source": data_source}, CHECKPOINT_PATH)
    else:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"missing {CHECKPOINT_PATH}; run with --train explicitly")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        history = list(checkpoint.get("history", []))
        if len(history) != args.epochs:
            raise AssertionError(
                f"checkpoint history has {len(history)} epochs, expected {args.epochs}; "
                "pass the matching --epochs or retrain explicitly"
            )

    print(
        {
            "task": "fno_ns_checkpoint_evaluation",
            "mode": mode,
            "data_source": data_source,
            "data_shape": list(data.shape),
            "parameters": count_parameters(model),
            "history_epochs": len(history),
        }
    )
    rel_l2_torch, inputs, targets, predictions_torch = evaluate(
        model, test_loader, use_supa=False
    )
    rel_l2_supa, inputs, targets, predictions_supa = evaluate(
        model, test_loader, use_supa=True
    )
    torch.save(
        {"input": inputs, "target": targets, "pred": predictions_supa},
        DEMO_BATCH_PATH,
    )
    write_metadata(
        mode=mode,
        data_source=data_source,
        n_train=args.n_train,
        n_test=args.n_test,
        batch_size=args.batch_size,
        epochs=args.epochs,
        history=history,
        rel_l2_torch=rel_l2_torch,
        rel_l2_supa=rel_l2_supa,
        inputs=inputs,
        targets=targets,
        predictions=predictions_supa,
    )
    update_summary(
        data_source=data_source,
        n_train=args.n_train,
        n_test=args.n_test,
        batch_size=args.batch_size,
        epochs=args.epochs,
        history=history,
        rel_l2_torch=rel_l2_torch,
        rel_l2_supa=rel_l2_supa,
    )
    print(
        {
            "rel_l2_torch": rel_l2_torch,
            "rel_l2_supa": rel_l2_supa,
            "torch_supa_l2_delta": abs(rel_l2_torch - rel_l2_supa),
            "ok": True,
        }
    )


if __name__ == "__main__":
    main()
