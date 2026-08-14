#!/usr/bin/env python3
"""SpectralConv correctness: SUPA path vs PyTorch reference (rel err <= 1e-4)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

from reference_pytorch import make_random_weights, spectral_conv2d
from spectral_conv_ops import spectral_conv2d_supa

REL_ERROR_THRESHOLD = 1.0e-4
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"


def relative_error(expected: torch.Tensor, actual: torch.Tensor) -> float:
    """Frobenius relative error ||y-ŷ||_F / ||y||_F (handbook 相对误差)."""
    diff_norm = float((expected - actual).norm().item())
    ref_norm = float(expected.norm().item())
    if ref_norm < 1.0e-12:
        return diff_norm
    return diff_norm / ref_norm


def max_abs_error(expected: torch.Tensor, actual: torch.Tensor) -> float:
    return float((expected - actual).abs().max().item())


def make_input(batch_size, channels_in, height, width, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.empty((batch_size, channels_in, height, width), dtype=torch.float32)
    x.uniform_(-0.5, 0.5, generator=generator)
    return x.contiguous()


def run_case(name, batch_size, channels_in, channels_out, height, width, modes1, modes2, seed):
    x = make_input(batch_size, channels_in, height, width, seed)
    weights = make_random_weights(channels_in, channels_out, modes1, modes2, seed + 17)
    weights2 = make_random_weights(channels_in, channels_out, modes1, modes2, seed + 23)

    expected = spectral_conv2d(x, weights, modes1, modes2, weights2=weights2)
    actual = spectral_conv2d_supa(x, weights, weights2, modes1, modes2)

    rel = relative_error(expected, actual)
    abs_err = max_abs_error(expected, actual)
    ok = rel <= REL_ERROR_THRESHOLD
    record = {
        "case": name,
        "shape": f"B{batch_size}_Cin{channels_in}_Cout{channels_out}_{height}x{width}",
        "modes": f"{modes1}x{modes2}",
        "max_abs": abs_err,
        "max_rel": rel,
        "threshold": REL_ERROR_THRESHOLD,
        "ok": ok,
    }
    print(record)
    if not ok:
        raise AssertionError(f"{name} failed max_rel={rel} (threshold {REL_ERROR_THRESHOLD})")
    return record


def write_summary(records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    worst_rel = max(item["max_rel"] for item in records)
    all_ok = all(item["ok"] for item in records)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    summary = {}
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
    summary.setdefault("meta", {})
    summary["meta"]["updated_at"] = stamp
    summary["spectral_conv"] = {
        "status": "accuracy_pass" if all_ok else "accuracy_fail",
        "rel_error": worst_rel,
        "threshold": REL_ERROR_THRESHOLD,
        "cases": records,
        "perf": summary.get("spectral_conv", {}).get("perf", {}),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    log_path = RUN_LOG_DIR / f"spectral_accuracy_{datetime.now().strftime('%Y-%m-%d')}.md"
    lines = [
        "# SpectralConv accuracy",
        "",
        f"- time_utc: {stamp}",
        f"- worst_rel: {worst_rel}",
        f"- threshold: {REL_ERROR_THRESHOLD}",
        f"- ok: {all_ok}",
        "",
        "```json",
        json.dumps(records, indent=2),
        "```",
        "",
    ]
    log_path.write_text("\n".join(lines))
    print({"summary": str(SUMMARY_PATH), "run_log": str(log_path), "worst_rel": worst_rel, "ok": all_ok})


def main():
    print(
        {
            "torch": torch.__version__,
            "supa_empty_ok": str(torch.empty((1,), device="supa").device),
            "cwd": os.getcwd(),
        }
    )
    records = []
    # First knife: small shapes
    records.append(run_case("tiny_8x8", 2, 2, 3, 8, 8, 2, 2, 100))
    records.append(run_case("small_32x32", 2, 4, 4, 32, 32, 8, 8, 200))
    records.append(run_case("target_64x64", 2, 4, 4, 64, 64, 12, 12, 300))
    write_summary(records)
    print({"task": "spectral_conv_accuracy", "ok": True})


if __name__ == "__main__":
    main()
