#!/usr/bin/env python3
"""FNO training-throughput bonus metric (same grid_points/s dimension).

Scope explicitly includes: forward + relative-L2 loss + backward + optimizer step.
Matches the training path used for the submitted checkpoint (CPU / use_supa=False).
This is a bonus report under the 2026-07-25 FNO performance protocol — not the
inference batch=16 main table.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = THIS_DIR.parent
sys.path.insert(0, str(SUBMISSION_DIR / "spectral_conv"))
sys.path.insert(0, str(THIS_DIR))

from model import FNO2d  # noqa: E402

SUMMARY_PATH = SUBMISSION_DIR / "results" / "summary.json"
RUN_LOG_DIR = SUBMISSION_DIR / "results" / "run_logs"

BATCH_SIZE = 8
HEIGHT = 64
WIDTH = 64
T_IN = 10
WARMUP = 10
ITERS = 50
SEED = 20260722
LEARNING_RATE = 5.0e-4
WEIGHT_DECAY = 1.0e-4


def train_step(
    model: FNO2d,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs, use_supa=False)
    loss = torch.norm(prediction - targets) / torch.norm(targets).clamp_min(1.0e-12)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def main() -> None:
    torch.manual_seed(SEED)
    model = FNO2d(
        modes1=16,
        modes2=16,
        width=32,
        n_layers=4,
        in_channels=T_IN,
        out_channels=1,
    )
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    inputs = torch.randn(BATCH_SIZE, T_IN, HEIGHT, WIDTH, dtype=torch.float32)
    targets = torch.randn(BATCH_SIZE, 1, HEIGHT, WIDTH, dtype=torch.float32)

    for _ in range(WARMUP):
        train_step(model, optimizer, inputs, targets)

    start = time.perf_counter()
    last_loss = 0.0
    for _ in range(ITERS):
        last_loss = train_step(model, optimizer, inputs, targets)
    elapsed = time.perf_counter() - start

    total_samples = ITERS * BATCH_SIZE
    samples_per_second = total_samples / elapsed
    report = {
        "status": "measured",
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bonus": True,
        "scope": (
            "one optimizer step = forward + relative-L2 loss + backward + Adam step"
        ),
        "device": "cpu",
        "path": "use_supa=False (matches submitted checkpoint training path)",
        "data": "synthetic_random_batches_for_throughput_only",
        "config": {
            "batch_size": BATCH_SIZE,
            "height": HEIGHT,
            "width": WIDTH,
            "t_in": T_IN,
            "warmup": WARMUP,
            "iters": ITERS,
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dtype": "float32",
        },
        "last_loss": last_loss,
        "metrics": {
            "grid_points_per_second": samples_per_second * HEIGHT * WIDTH,
            "samples_per_second": samples_per_second,
            "milliseconds_per_sample": elapsed * 1000.0 / total_samples,
            "step_milliseconds_per_batch": elapsed * 1000.0 / ITERS,
            "peak_memory_MB": None,
            "note": "CPU training path; peak device MB not applicable",
        },
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    summary.setdefault("fno_ns", {})["train_throughput"] = report
    summary.setdefault("meta", {})["updated_at"] = report["measured_at"]
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / "fno_train_throughput_2026-07-25.md"
    metrics = report["metrics"]
    log_path.write_text(
        "\n".join(
            [
                "# FNO training throughput (bonus)",
                "",
                f"- measured_at: {report['measured_at']}",
                f"- scope: {report['scope']}",
                f"- device/path: {report['device']} / {report['path']}",
                (
                    f"- config: B={BATCH_SIZE}, {HEIGHT}x{WIDTH}, warmup={WARMUP}, "
                    f"iters={ITERS}, seed={SEED}"
                ),
                "",
                "| metric | value |",
                "|---|---:|",
                f"| grid_points/s | {metrics['grid_points_per_second']:.3f} |",
                f"| samples/s | {metrics['samples_per_second']:.3f} |",
                f"| ms/sample | {metrics['milliseconds_per_sample']:.3f} |",
                f"| ms/batch (step) | {metrics['step_milliseconds_per_batch']:.3f} |",
                "",
                "Not the inference batch=16 main table.",
                "",
            ]
        )
    )
    print(json.dumps(report, indent=2))
    print({"summary": str(SUMMARY_PATH), "run_log": str(log_path)})


if __name__ == "__main__":
    main()
