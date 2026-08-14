#!/usr/bin/env python3
"""Strengthen path: SpectralConv with suFFT R2C/C2R vs CPU reference."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

from reference_pytorch import make_random_weights, spectral_conv2d
from spectral_conv_ops import spectral_conv2d_supa
from test_accuracy import REL_ERROR_THRESHOLD, make_input, max_abs_error, relative_error

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"


def run_case(name, batch_size, channels_in, channels_out, height, width, modes1, modes2, seed):
    x = make_input(batch_size, channels_in, height, width, seed)
    weights = make_random_weights(channels_in, channels_out, modes1, modes2, seed + 17)
    weights2 = make_random_weights(channels_in, channels_out, modes1, modes2, seed + 23)
    expected = spectral_conv2d(x, weights, modes1, modes2, weights2=weights2)
    actual = spectral_conv2d_supa(x, weights, weights2, modes1, modes2, use_sufft=True)
    rel = relative_error(expected, actual)
    abs_err = max_abs_error(expected, actual)
    ok = rel <= REL_ERROR_THRESHOLD
    record = {
        "case": name,
        "path": "sufft_r2c_supa_mul_sufft_c2r",
        "shape": f"B{batch_size}_Cin{channels_in}_Cout{channels_out}_{height}x{width}",
        "modes": f"{modes1}x{modes2}",
        "max_abs": abs_err,
        "max_rel": rel,
        "threshold": REL_ERROR_THRESHOLD,
        "ok": ok,
    }
    print(record)
    if not ok:
        raise AssertionError(f"{name} failed max_rel={rel}")
    return record


def main() -> None:
    print({"task": "spectral_conv_sufft_accuracy", "device": "supa"})
    records = [
        run_case("tiny_8x8", 2, 2, 3, 8, 8, 2, 2, 100),
        run_case("small_32x32", 2, 4, 4, 32, 32, 8, 8, 200),
        run_case("target_64x64", 2, 4, 4, 64, 64, 12, 12, 300),
    ]
    worst = max(item["max_rel"] for item in records)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = datetime.now().strftime("%Y-%m-%d")

    summary = {}
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
    spectral = summary.setdefault("spectral_conv", {})
    spectral["sufft"] = {
        "status": "accuracy_pass",
        "rel_error": worst,
        "threshold": REL_ERROR_THRESHOLD,
        "cases": records,
    }
    summary.setdefault("meta", {})["updated_at"] = stamp
    summary["meta"]["notes"] = (
        "strengthen: suFFT R2C/C2R path accuracy_pass; formal pack still deferred."
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / f"spectral_sufft_accuracy_{day}.md"
    log_path.write_text(
        "\n".join(
            [
                "# SpectralConv suFFT accuracy",
                "",
                f"- time_utc: {stamp}",
                f"- worst_rel: {worst}",
                f"- ok: True",
                "",
                "```json",
                json.dumps(records, indent=2),
                "```",
                "",
            ]
        )
    )
    print({"worst_rel": worst, "run_log": str(log_path), "ok": True})


if __name__ == "__main__":
    main()
