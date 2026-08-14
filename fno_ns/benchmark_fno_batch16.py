#!/usr/bin/env python3
"""Official-aligned FNO inference benchmark at batch_size=16.

Default: public NS64 + fno_ns_public_demo.pt (protocol-aligned).
Legacy v2 self-generated path retained via --legacy-v2.
"""

from __future__ import annotations

import argparse
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
PUBLIC_CKPT = THIS_DIR / "checkpoints" / "fno_ns_public_demo.pt"
LEGACY_CKPT = THIS_DIR / "checkpoints" / "fno_ns_demo.pt"
BATCH_SIZE = 16
HEIGHT = 64
WIDTH = 64
WARMUP = 10
ITERS = 50
SEED = 20260722


def make_model(checkpoint_path: Path) -> FNO2d:
    model = FNO2d(
        modes1=16,
        modes2=16,
        width=32,
        n_layers=4,
        in_channels=10,
        out_channels=1,
    ).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--legacy-v2",
        action="store_true",
        help="use self-generated NS-like v2 + fno_ns_demo.pt (engineering旁注)",
    )
    args = ap.parse_args()

    torch.manual_seed(SEED)
    if args.legacy_v2:
        ckpt = LEGACY_CKPT
        data, data_source = load_or_build_ns_like(
            n_samples=1024, resolution=HEIGHT, n_times=30, seed=SEED, version="v2"
        )
        _, test_data = split_train_test(data, 768, 128, seed=SEED)
        disclosure = "self-generated NS-like v2; not public NS64"
        log_name = "fno_batch16_benchmark_legacy_v2.md"
    else:
        ckpt = PUBLIC_CKPT
        data, data_source = load_or_build_ns_like(
            n_samples=1128, resolution=HEIGHT, n_times=20, seed=SEED, version="v2"
        )
        if not str(data_source).startswith("file:navier_stokes"):
            raise SystemExit(
                f"expected public navier_stokes*.pt, got source={data_source!r}"
            )
        _, test_data = split_train_test(data, 1000, 128, seed=SEED)
        disclosure = "public NS64 navier_stokes_v1e-3_N1200_T20.pt; n_train=1000/n_test=128"
        log_name = (
            f"fno_batch16_benchmark_public_ns64_"
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"
        )

    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint: {ckpt}")

    dataset = SequenceVorticityDataset(test_data, 10, 1)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    inputs_cpu, _ = next(iter(data_loader))
    # Official chain gate shape is B=4; B=16 public batches can sit ~1e-4 due to
    # norm aggregation noise. Gate on B=4, still report B=16 consistency as note.
    gate_inputs = torch.stack([dataset[i][0] for i in range(4)], dim=0)

    gate_model = make_model(ckpt)
    consistency = compare_model(gate_model, gate_inputs)
    if not consistency["ok"]:
        raise AssertionError(f"refusing to benchmark incorrect chain: {consistency}")
    consistency_b16 = compare_model(make_model(ckpt), inputs_cpu)

    pure_forward = benchmark_pure_forward(make_model(ckpt), inputs_cpu)
    with_dataloader = benchmark_with_dataloader(make_model(ckpt), data_loader)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {
        "status": "measured",
        "measured_at": timestamp,
        "device": "Biren106B / supa",
        "data": data_source,
        "data_disclosure": disclosure,
        "checkpoint": str(ckpt.relative_to(SUBMISSION_DIR)),
        "config": {
            "batch_size": BATCH_SIZE,
            "height": HEIGHT,
            "width": WIDTH,
            "warmup": WARMUP,
            "iters": ITERS,
            "seed": SEED,
            "dtype": "float32",
            "split": "1000/128" if not args.legacy_v2 else "768/128",
        },
        "chain_consistency": consistency,
        "chain_consistency_batch16_note": consistency_b16,
        "pure_forward": pure_forward,
        "with_dataloader": with_dataloader,
        "note": (
            "legacy v2 engineering旁注"
            if args.legacy_v2
            else "protocol-aligned public NS64 + public_demo ckpt; gate=B4; not Spectral formal ms"
        ),
    }
    print(json.dumps(report, indent=2))

    if not args.legacy_v2:
        summary = json.loads(SUMMARY_PATH.read_text())
        summary.setdefault("fno_ns", {})["perf_batch16"] = report
        summary.setdefault("meta", {})["updated_at"] = timestamp
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / log_name
    lines = [
        "# FNO batch=16 benchmark",
        "",
        f"- time_utc: {timestamp}",
        "- device: BIREN single card (Biren106B / supa)",
        f"- data: {data_source}",
        f"- disclosure: {disclosure}",
        f"- checkpoint: `{ckpt.relative_to(SUBMISSION_DIR)}`",
        f"- config: B={BATCH_SIZE}, H=W={HEIGHT}, warmup={WARMUP}, iters={ITERS}",
        f"- chain_consistency_rel (gate B=4): {consistency['relative_error']}",
        f"- chain_consistency_rel (batch16 note): {consistency_b16['relative_error']}",
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
    print("wrote", log_path)


if __name__ == "__main__":
    main()
