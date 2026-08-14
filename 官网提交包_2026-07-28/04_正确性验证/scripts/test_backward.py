#!/usr/bin/env python3
"""Extension: spectral_mul autograd vs pure torch einsum reference grads."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

from spectral_conv_ops import spectral_mul_autograd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"
GRAD_REL_THRESHOLD = 1.0e-4


def _rel_err(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    diff = torch.linalg.norm((prediction - reference).reshape(-1))
    ref = torch.linalg.norm(reference.reshape(-1)).clamp_min(1.0e-12)
    return float((diff / ref).item())


def run_case(batch: int, cin: int, cout: int, m1: int, m2: int) -> dict:
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(batch, cin, m1, m2, generator=generator, dtype=torch.cfloat)
    w = torch.randn(cin, cout, m1, m2, generator=generator, dtype=torch.cfloat)
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    y_ref = torch.einsum("bixy,ioxy->boxy", x_ref, w_ref)
    loss_ref = (y_ref.real.pow(2) + y_ref.imag.pow(2)).sum()
    loss_ref.backward()

    y = spectral_mul_autograd(x, w)
    loss = (y.real.pow(2) + y.imag.pow(2)).sum()
    loss.backward()

    fwd_rel = _rel_err(y.detach(), y_ref.detach())
    gx_rel = _rel_err(x.grad, x_ref.grad)
    gw_rel = _rel_err(w.grad, w_ref.grad)
    ok = max(fwd_rel, gx_rel, gw_rel) <= GRAD_REL_THRESHOLD
    return {
        "shape": f"B{batch}_Cin{cin}_Cout{cout}_{m1}x{m2}",
        "fwd_rel": fwd_rel,
        "grad_x_rel": gx_rel,
        "grad_w_rel": gw_rel,
        "ok": ok,
    }


def main() -> None:
    print({"task": "spectral_mul_backward", "threshold": GRAD_REL_THRESHOLD})
    cases = [
        run_case(2, 2, 3, 4, 4),
        run_case(2, 4, 4, 8, 8),
        run_case(2, 4, 4, 12, 12),
    ]
    worst = max(max(c["fwd_rel"], c["grad_x_rel"], c["grad_w_rel"]) for c in cases)
    ok = all(c["ok"] for c in cases)
    for case in cases:
        print(case)
    print({"worst_grad_rel": worst, "ok": ok})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    summary.setdefault("meta", {})["updated_at"] = stamp
    sc = summary.setdefault("spectral_conv_combo", {})
    sc["backward"] = {
        "status": "pass" if ok else "fail",
        "threshold": GRAD_REL_THRESHOLD,
        "worst_rel": worst,
        "cases": cases,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
