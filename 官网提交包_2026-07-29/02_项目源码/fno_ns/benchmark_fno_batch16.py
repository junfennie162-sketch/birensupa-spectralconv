#!/usr/bin/env python3
"""Official-aligned FNO inference benchmark at batch_size=16."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = THIS_DIR.parent
sys.path.insert(0, str(SUBMISSION_DIR / "spectral_conv"))
sys.path.insert(0, str(THIS_DIR))

import torch_br  # noqa: F401,E402
from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test  # noqa: E402
from model import FNO2d  # noqa: E402
from test_chain_cpu_supa_consistency import compare_model  # noqa: E402

SUMMARY_PATH = SUBMISSION_DIR / "results" / "summary.json"
RUN_LOG_DIR = SUBMISSION_DIR / "results" / "run_logs"
CHECKPOINT_PATH = THIS_DIR / "checkpoints" / "fno_ns_demo.pt"
BATCH_SIZE = 16
HEIGHT = 64
WIDTH = 64
WARMUP = 10
ITERS = 50
SEED = 20260722


def make_model() -> FNO2d:
    model = FNO2d(
        modes1=16,
        modes2=16,
        width=32,
        n_layers=4,
        in_channels=10,
        out_channels=1,
    ).eval()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model


def reset_peak_memory() -> None:
    if hasattr(torch_br.supa, "reset_peak_memory_stats"):
        torch_br.supa.reset_peak_memory_stats()
    elif hasattr(torch.supa, "reset_peak_memory_stats"):
        torch.supa.reset_peak_memory_stats()


def peak_memory_mb() -> float:
    if hasattr(torch_br.supa, "max_memory_allocated"):
        return float(torch_br.supa.max_memory_allocated()) / 1024**2
    return float(torch.supa.max_memory_allocated()) / 1024**2


def metrics(elapsed_seconds: float, iterations: int, peak_mb: float) -> dict:
    total_samples = iterations * BATCH_SIZE
    samples_per_second = total_samples / elapsed_seconds
    return {
        "grid_points_per_second": samples_per_second * HEIGHT * WIDTH,
        "samples_per_second": samples_per_second,
        "milliseconds_per_sample": elapsed_seconds * 1000.0 / total_samples,
        "forward_milliseconds_per_batch": elapsed_seconds * 1000.0 / iterations,
        "peak_memory_MB": peak_mb,
    }


def benchmark_pure_forward(model: FNO2d, inputs_cpu: torch.Tensor) -> dict:
    model.prepare_supa_eval()
    inputs_supa = inputs_cpu.to("supa")
    with torch.no_grad():
        for _ in range(WARMUP):
            model.forward_supa_chain(inputs_supa, use_sufft="auto")
        torch_br.supa.synchronize()
        reset_peak_memory()
        start = time.perf_counter()
        for _ in range(ITERS):
            model.forward_supa_chain(inputs_supa, use_sufft="auto")
        torch_br.supa.synchronize()
        elapsed = time.perf_counter() - start
    return metrics(elapsed, ITERS, peak_memory_mb())


def benchmark_with_dataloader(model: FNO2d, data_loader: DataLoader) -> dict:
    iterator = iter(data_loader)
    model.prepare_supa_eval()

    def next_inputs() -> torch.Tensor:
        nonlocal iterator
        try:
            inputs, _ = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            inputs, _ = next(iterator)
        assert inputs.shape[0] == BATCH_SIZE
        return inputs

    with torch.no_grad():
        for _ in range(WARMUP):
            model.forward_supa_chain(next_inputs().to("supa"), use_sufft="auto")
        torch_br.supa.synchronize()
        reset_peak_memory()
        start = time.perf_counter()
        for _ in range(ITERS):
            model.forward_supa_chain(next_inputs().to("supa"), use_sufft="auto")
        torch_br.supa.synchronize()
        elapsed = time.perf_counter() - start
    return metrics(elapsed, ITERS, peak_memory_mb())


def main() -> None:
    torch.manual_seed(SEED)
    data, data_source = load_or_build_ns_like(
        n_samples=1024, resolution=HEIGHT, n_times=30, seed=SEED, version="v2"
    )
    _, test_data = split_train_test(data, 768, 128, seed=SEED)
    dataset = SequenceVorticityDataset(test_data, 10, 1)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    inputs_cpu, _ = next(iter(data_loader))

    gate_model = make_model()
    consistency = compare_model(gate_model, inputs_cpu)
    if not consistency["ok"]:
        raise AssertionError(f"refusing to benchmark incorrect chain: {consistency}")

    pure_forward = benchmark_pure_forward(make_model(), inputs_cpu)
    with_dataloader = benchmark_with_dataloader(make_model(), data_loader)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {
        "status": "measured",
        "measured_at": timestamp,
        "device": "Biren106B / supa",
        "data": data_source,
        "data_disclosure": "self-generated NS-like v2; not public NS64",
        "config": {
            "batch_size": BATCH_SIZE,
            "height": HEIGHT,
            "width": WIDTH,
            "warmup": WARMUP,
            "iters": ITERS,
            "seed": SEED,
            "dtype": "float32",
        },
        "chain_consistency": consistency,
        "pure_forward": pure_forward,
        "with_dataloader": with_dataloader,
    }
    print(json.dumps(report, indent=2))

    summary = json.loads(SUMMARY_PATH.read_text())
    summary.setdefault("fno_ns", {})["perf_batch16"] = report
    summary.setdefault("meta", {})["updated_at"] = timestamp
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / "fno_batch16_benchmark_2026-07-25.md"
    lines = [
        "# FNO batch=16 benchmark",
        "",
        f"- time_utc: {timestamp}",
        "- device: BIREN single card (Biren106B / supa)",
        f"- data: {data_source} (self-generated NS-like v2; not public NS64)",
        f"- config: B={BATCH_SIZE}, H=W={HEIGHT}, warmup={WARMUP}, iters={ITERS}",
        f"- chain_consistency_rel: {consistency['relative_error']}",
        "",
        "| scope | grid_points/s | samples/s | ms/sample | ms/batch | peak MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scope, row in (("pure forward", pure_forward), ("with DataLoader", with_dataloader)):
        lines.append(
            f"| {scope} | {row['grid_points_per_second']:.3f} | "
            f"{row['samples_per_second']:.3f} | {row['milliseconds_per_sample']:.6f} | "
            f"{row['forward_milliseconds_per_batch']:.3f} | {row['peak_memory_MB']:.1f} |"
        )
    lines.append("")
    log_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
