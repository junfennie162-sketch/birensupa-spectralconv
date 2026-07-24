#!/usr/bin/env python3
"""Full SpectralConv correctness: SUPA path vs official dual-weight CPU reference."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

from reference_pytorch import SpectralConv2d
from spectral_conv_ops import spectral_conv2d_supa

REL_ERROR_THRESHOLD = 1.0e-4
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"


def relative_error(expected: torch.Tensor, actual: torch.Tensor) -> float:
    diff_norm = float((expected - actual).norm().item())
    ref_norm = float(expected.norm().item())
    if ref_norm < 1.0e-12:
        return diff_norm
    return diff_norm / ref_norm


def run_case(name, batch, cin, cout, height, width, modes1, modes2, seed):
    torch.manual_seed(seed)
    model = SpectralConv2d(cin, cout, modes1, modes2)
    x = torch.randn(batch, cin, height, width, dtype=torch.float32)
    with torch.no_grad():
        expected = model(x)
        actual = spectral_conv2d_supa(x, model.weights1, model.weights2, modes1, modes2)
    rel = relative_error(expected, actual)
    ok = rel <= REL_ERROR_THRESHOLD
    record = {
        "case": name,
        "shape": f"B{batch}_Cin{cin}_Cout{cout}_{height}x{width}",
        "modes": f"{modes1}x{modes2}",
        "rel": rel,
        "threshold": REL_ERROR_THRESHOLD,
        "ok": ok,
    }
    print(record)
    if not ok:
        raise AssertionError(f"{name} failed rel={rel}")
    return record


def main():
    print(
        {
            "torch": torch.__version__,
            "supa": str(torch.empty((1,), device="supa").device),
        }
    )
    records = [
        run_case("tiny_8x8", 2, 2, 3, 8, 8, 2, 2, 11),
        run_case("small_32x32", 2, 4, 4, 32, 32, 8, 8, 22),
        run_case("target_64x64", 2, 4, 4, 64, 64, 12, 12, 33),
        run_case("official_128", 4, 32, 64, 128, 128, 16, 16, 44),
    ]
    worst = max(r["rel"] for r in records)
    all_ok = all(r["ok"] for r in records)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    summary.setdefault("meta", {})["updated_at"] = stamp
    summary["spectral_conv"] = {
        "status": "accuracy_pass" if all_ok else "accuracy_fail",
        "rel_error": worst,
        "threshold": REL_ERROR_THRESHOLD,
        "cases": records,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    log_path = RUN_LOG_DIR / f"spectral_accuracy_{datetime.now().strftime('%Y-%m-%d')}.md"
    log_path.write_text(
        "\n".join(
            [
                "# SpectralConv Extension accuracy (dual-corner)",
                "",
                f"- time_utc: {stamp}",
                f"- worst_rel: {worst}",
                f"- ok: {all_ok}",
                "",
                "```json",
                json.dumps(records, indent=2),
                "```",
                "",
            ]
        )
    )
    print({"task": "spectral_conv_accuracy", "ok": all_ok, "worst_rel": worst})


if __name__ == "__main__":
    main()
