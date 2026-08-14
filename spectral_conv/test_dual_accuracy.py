#!/usr/bin/env python3
"""Dual-corner SpectralConv accuracy (official FNO corners) vs CPU reference."""

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


def run_case(name, batch_size, channels_in, channels_out, height, width, modes1, modes2, seed, use_sufft):
    x = make_input(batch_size, channels_in, height, width, seed)
    w1 = make_random_weights(channels_in, channels_out, modes1, modes2, seed + 17)
    w2 = make_random_weights(channels_in, channels_out, modes1, modes2, seed + 91)
    expected = spectral_conv2d(x, w1, modes1, modes2, weights2=w2)
    actual = spectral_conv2d_supa(x, w1, w2, modes1, modes2, use_sufft=use_sufft)
    rel = relative_error(expected, actual)
    ok = rel <= REL_ERROR_THRESHOLD
    record = {
        "case": name,
        "path": "fused_dual" if use_sufft else "v1_dual",
        "max_rel": rel,
        "max_abs": max_abs_error(expected, actual),
        "ok": ok,
    }
    print(record)
    if not ok:
        raise AssertionError(f"{name} failed rel={rel}")
    return record


def main() -> None:
    records = []
    for use_sufft in (False, True):
        tag = "fused" if use_sufft else "v1"
        records.append(run_case(f"{tag}_8x8", 2, 2, 3, 8, 8, 2, 2, 100, use_sufft))
        records.append(run_case(f"{tag}_32x32", 2, 4, 4, 32, 32, 8, 8, 200, use_sufft))
        records.append(run_case(f"{tag}_64x64", 2, 4, 4, 64, 64, 12, 12, 300, use_sufft))
    worst = max(r["max_rel"] for r in records)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    spectral = summary.setdefault("spectral_conv", {})
    spectral["dual_corner"] = {
        "status": "accuracy_pass",
        "rel_error": worst,
        "threshold": REL_ERROR_THRESHOLD,
        "cases": records,
    }
    summary.setdefault("meta", {})["updated_at"] = stamp
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = RUN_LOG_DIR / f"spectral_dual_accuracy_{datetime.now().strftime('%Y-%m-%d')}.md"
    log.write_text(
        f"# Dual-corner accuracy\n\n- worst_rel: {worst}\n\n```json\n"
        + json.dumps(records, indent=2)
        + "\n```\n"
    )
    print({"ok": True, "worst_rel": worst, "run_log": str(log)})


if __name__ == "__main__":
    main()
