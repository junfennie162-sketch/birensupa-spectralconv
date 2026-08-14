#!/usr/bin/env python3
"""CPU vs SUPA FNO-chain numerical consistency gate."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = THIS_DIR.parent
sys.path.insert(0, str(SUBMISSION_DIR / "spectral_conv"))
sys.path.insert(0, str(THIS_DIR))

import torch_br  # noqa: F401,E402
from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test  # noqa: E402
from model import FNO2d  # noqa: E402

SUMMARY_PATH = SUBMISSION_DIR / "results" / "summary.json"
RUN_LOG_DIR = SUBMISSION_DIR / "results" / "run_logs"
CHECKPOINT_PATH = THIS_DIR / "checkpoints" / "fno_ns_demo.pt"
RELATIVE_ERROR_THRESHOLD = 1.0e-4
SEED = 20260722


def relative_error(reference: torch.Tensor, actual: torch.Tensor) -> float:
    difference_norm = torch.linalg.norm((reference - actual).reshape(-1))
    reference_norm = torch.linalg.norm(reference.reshape(-1)).clamp_min(1.0e-12)
    return float((difference_norm / reference_norm).item())


def make_model() -> FNO2d:
    return FNO2d(
        modes1=16,
        modes2=16,
        width=32,
        n_layers=4,
        in_channels=10,
        out_channels=1,
    ).eval()


@torch.no_grad()
def compare_model(model: FNO2d, input_cpu: torch.Tensor) -> dict:
    reference = model(input_cpu, use_supa=False)
    model.prepare_supa_eval()
    actual = model.forward_supa_chain(input_cpu.to("supa"), use_sufft="auto")
    error = relative_error(reference, actual)
    return {
        "relative_error": error,
        "threshold": RELATIVE_ERROR_THRESHOLD,
        "finite": bool(torch.isfinite(actual).all().item()),
        "ok": error <= RELATIVE_ERROR_THRESHOLD and bool(torch.isfinite(actual).all().item()),
        "input_shape": list(input_cpu.shape),
    }


def checkpoint_input() -> torch.Tensor:
    data, _ = load_or_build_ns_like(
        n_samples=1024,
        resolution=64,
        n_times=30,
        seed=SEED,
        version="v2",
    )
    _, test_data = split_train_test(data, 768, 128, seed=SEED)
    dataset = SequenceVorticityDataset(test_data, 10, 1)
    return torch.stack([dataset[index][0] for index in range(4)], dim=0)


def main() -> None:
    torch.manual_seed(0)
    random_model = make_model()
    random_result = compare_model(random_model, torch.randn(4, 10, 64, 64))

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    trained_model = make_model()
    trained_model.load_state_dict(checkpoint["model"])
    checkpoint_result = compare_model(trained_model, checkpoint_input())

    all_ok = random_result["ok"] and checkpoint_result["ok"]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {
        "status": "pass" if all_ok else "fail",
        "threshold": RELATIVE_ERROR_THRESHOLD,
        "random_model": random_result,
        "checkpoint_model": checkpoint_result,
        "fallback": "SUPA-resident fused input D2D-copied into host-seeded SUPA buffer before suFFT",
        "measured_at": timestamp,
    }
    print(json.dumps(report, indent=2))

    summary = json.loads(SUMMARY_PATH.read_text())
    summary.setdefault("fno_ns", {})["chain_consistency"] = report
    summary.setdefault("meta", {})["updated_at"] = timestamp
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / "fno_chain_consistency_2026-07-25.md"
    log_path.write_text(
        "\n".join(
            [
                "# FNO CPU/SUPA chain consistency",
                "",
                f"- time_utc: {timestamp}",
                f"- threshold: {RELATIVE_ERROR_THRESHOLD}",
                f"- random_model_rel: {random_result['relative_error']}",
                f"- checkpoint_model_rel: {checkpoint_result['relative_error']}",
                f"- ok: {all_ok}",
                "- fallback: SUPA-resident fused input D2D-copied into host-seeded SUPA buffer before suFFT",
                "",
            ]
        )
    )
    if not all_ok:
        raise AssertionError(f"FNO chain consistency failed: {report}")


if __name__ == "__main__":
    main()
