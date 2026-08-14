#!/usr/bin/env python3
"""P6: SpectralConv3d accuracy vs CPU reference (single low-frequency corner)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

from reference_pytorch import make_random_weights, spectral_conv3d
from spectral_conv_ops import spectral_conv3d_supa

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"
REL_THRESHOLD = 1.0e-4


def relative_error(prediction: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    diff = prediction - reference
    max_abs = float(diff.abs().max().item())
    ref_norm = torch.linalg.norm(reference.reshape(-1)).clamp_min(1.0e-12)
    max_rel = float((torch.linalg.norm(diff.reshape(-1)) / ref_norm).item())
    return max_abs, max_rel


def run_case(
    name: str,
    batch: int,
    cin: int,
    cout: int,
    depth: int,
    height: int,
    width: int,
    modes: tuple[int, int, int],
    seed: int,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, cin, depth, height, width, generator=generator)
    # reuse 2d maker then expand last mode dim
    w2 = make_random_weights(cin, cout, modes[0], modes[1] * modes[2], seed + 1)
    weights = w2.reshape(cin, cout, modes[0], modes[1], modes[2])
    ref = spectral_conv3d(x, weights, *modes)
    pred = spectral_conv3d_supa(x, weights, *modes)
    max_abs, max_rel = relative_error(pred, ref)
    return {
        "case": name,
        "shape": f"B{batch}_Cin{cin}_Cout{cout}_{depth}x{height}x{width}",
        "modes": f"{modes[0]}x{modes[1]}x{modes[2]}",
        "max_abs": max_abs,
        "max_rel": max_rel,
        "threshold": REL_THRESHOLD,
        "ok": max_rel <= REL_THRESHOLD,
    }


def main() -> None:
    print({"task": "spectral_conv3d_accuracy", "threshold": REL_THRESHOLD})
    cases = [
        run_case("tiny_8", 2, 2, 3, 8, 8, 8, (2, 2, 2), 11),
        run_case("small_16", 2, 4, 4, 16, 16, 16, (4, 4, 4), 22),
    ]
    worst = max(c["max_rel"] for c in cases)
    ok = all(c["ok"] for c in cases)
    for case in cases:
        print(case)
    print({"worst_rel": worst, "ok": ok})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = datetime.now().strftime("%Y-%m-%d")

    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    summary.setdefault("meta", {})
    summary["meta"]["updated_at"] = stamp
    summary.setdefault("spectral_conv", {})
    summary["spectral_conv"]["conv3d"] = {
        "status": "accuracy_pass" if ok else "fail",
        "rel_error": worst,
        "threshold": REL_THRESHOLD,
        "cases": cases,
        "note": "CPU rFFT3 + SUPA spectral_mul via reshape; single low corner",
    }
    summary.setdefault("optimization", {})
    summary["optimization"]["p6_3d"] = {
        "status": "done" if ok else "fail",
        "worst_rel": worst,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    log_path = RUN_LOG_DIR / f"spectral_3d_accuracy_{day}.md"
    lines = [
        "# SpectralConv3d accuracy (P6)",
        "",
        f"- time_utc: {stamp}",
        f"- threshold: {REL_THRESHOLD}",
        f"- worst_rel: {worst}",
        f"- ok: {ok}",
        "",
    ]
    for case in cases:
        lines.append(
            f"- {case['case']}: max_rel={case['max_rel']:.3e}, ok={case['ok']}"
        )
    lines.append("")
    log_path.write_text("\n".join(lines))
    print({"summary": str(SUMMARY_PATH), "run_log": str(log_path), "ok": ok})
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
