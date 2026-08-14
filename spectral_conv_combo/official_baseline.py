#!/usr/bin/env python3
"""Reproduce the official Spectral Convolution baseline from the track brief.

Source: 赛题文档/官网-赛道五-模型与算子详情页.md §3.1–3.2
- Full SpectralConv2d (weights1 + weights2)
- Correctness smoke + multi-resolution performance table

Device policy on this machine:
- Algorithmic baseline (correctness + perf numbers) runs on **CPU**.
  Official snippet uses CUDA; here `torch.cuda.is_available()` is False.
- SUPA is probed separately: native `torch.fft.rfft2` on `device=supa` is
  numerically inconsistent with CPU (observed large error), so it is NOT used
  as the accuracy baseline. Team SUPA work should keep FFT on CPU (or suFFT)
  and accelerate the complex multiply — see spectral_conv_ops.py.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_br  # noqa: F401

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"
SUMMARY_PATH = RESULTS_DIR / "summary.json"

CORRECTNESS_B, CORRECTNESS_C_IN, CORRECTNESS_C_OUT = 4, 32, 64
CORRECTNESS_H, CORRECTNESS_W = 128, 128
CORRECTNESS_MODES1, CORRECTNESS_MODES2 = 16, 16

PERF_RESOLUTIONS = [(64, 64), (128, 128), (256, 256)]
PERF_B, PERF_C_IN, PERF_C_OUT = 4, 32, 64
PERF_MODES1, PERF_MODES2 = 16, 16
PERF_WARMUP = 10
PERF_ITERS = 100
REL_ERROR_THRESHOLD = 1.0e-4


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution — FNO core (official reference)."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input_tensor, weights):
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x):
        batch_size, _channels_in, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


def synchronize(device: torch.device) -> None:
    if device.type == "supa":
        torch.supa.synchronize()


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "supa":
        return torch.supa.max_memory_allocated() / 1024**2
    return 0.0


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "supa":
        torch.supa.reset_peak_memory_stats()


def test_correctness(device: torch.device) -> dict:
    model = SpectralConv2d(
        CORRECTNESS_C_IN, CORRECTNESS_C_OUT, CORRECTNESS_MODES1, CORRECTNESS_MODES2
    ).to(device)
    x = torch.randn(
        CORRECTNESS_B, CORRECTNESS_C_IN, CORRECTNESS_H, CORRECTNESS_W, device=device
    )

    with torch.no_grad():
        y = model(x)
    synchronize(device)

    print(f"输入 shape: {tuple(x.shape)}")
    print(f"输出 shape: {tuple(y.shape)}")
    print(f"输出范围: [{y.min().item():.6f}, {y.max().item():.6f}]")

    x_grad = x.detach().clone().requires_grad_(True)
    y_grad = model(x_grad)
    loss = y_grad.sum()
    loss.backward()
    synchronize(device)

    assert x_grad.grad is not None
    print(f"梯度 shape: {tuple(x_grad.grad.shape)}")
    print(f"梯度范围: [{x_grad.grad.min().item():.6f}, {x_grad.grad.max().item():.6f}]")
    print("正确性验证通过!")

    return {
        "ok": True,
        "device": str(device),
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "output_min": float(y.min().item()),
        "output_max": float(y.max().item()),
        "grad_shape": list(x_grad.grad.shape),
        "grad_min": float(x_grad.grad.min().item()),
        "grad_max": float(x_grad.grad.max().item()),
        "modes": [CORRECTNESS_MODES1, CORRECTNESS_MODES2],
    }


def benchmark_performance(device: torch.device) -> list[dict]:
    results = []
    for height, width in PERF_RESOLUTIONS:
        model = SpectralConv2d(PERF_C_IN, PERF_C_OUT, PERF_MODES1, PERF_MODES2).to(device)
        x = torch.randn(PERF_B, PERF_C_IN, height, width, device=device)

        for _ in range(PERF_WARMUP):
            _ = model(x)
        synchronize(device)

        start = time.time()
        for _ in range(PERF_ITERS):
            _ = model(x)
        synchronize(device)
        forward_time_ms = (time.time() - start) / PERF_ITERS * 1000.0

        reset_peak_memory(device)
        _ = model(x)
        synchronize(device)
        memory_mb = peak_memory_mb(device)

        row = {
            "resolution": f"{height}x{width}",
            "forward_time_ms": f"{forward_time_ms:.3f}",
            "memory_MB": f"{memory_mb:.1f}",
        }
        results.append(row)
        print(f"分辨率 {height}x{width}: 前向 {forward_time_ms:.3f}ms, 显存 {memory_mb:.1f}MB")
    return results


def probe_supa_fft() -> dict:
    """Document native torch.fft on SUPA vs CPU (known risk for this stack)."""
    if torch.supa.device_count() < 1:
        return {"ok": False, "reason": "no_supa_device"}

    torch.manual_seed(0)
    x_cpu = torch.randn(2, 4, 32, 32)
    x_supa = x_cpu.to("supa")
    ft_cpu = torch.fft.rfft2(x_cpu)
    ft_supa = torch.fft.rfft2(x_supa).cpu()
    max_abs = float((ft_cpu - ft_supa).abs().max().item())
    ref = float(ft_cpu.abs().max().item()) + 1.0e-12
    rel = max_abs / ref
    ok = rel <= REL_ERROR_THRESHOLD
    print(f"SUPA torch.fft.rfft2 vs CPU: max_abs={max_abs:.6e}, rel={rel:.6e}, ok={ok}")
    return {
        "ok": ok,
        "max_abs": max_abs,
        "rel_vs_cpu_peak": rel,
        "threshold": REL_ERROR_THRESHOLD,
        "note": "If false, do not put full SpectralConv (incl. FFT) on device=supa via torch.fft",
    }


def write_artifacts(correctness: dict, perf_rows: list[dict], fft_probe: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = datetime.now().strftime("%Y-%m-%d")

    payload = {
        "task": "official_spectral_conv_baseline",
        "time_utc": stamp,
        "correctness": correctness,
        "performance": perf_rows,
        "supa_fft_probe": fft_probe,
        "ok": bool(correctness.get("ok")),
    }

    out_json = RUN_LOG_DIR / f"official_baseline_{day}.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Official SpectralConv baseline (PyTorch reference)",
        "",
        f"- time_utc: {stamp}",
        f"- baseline_device: {correctness.get('device')}",
        f"- correctness_ok: {correctness.get('ok')}",
        f"- supa_fft_probe_ok: {fft_probe.get('ok')}",
        "",
        "## Correctness (official shapes / forward / backward)",
        "```json",
        json.dumps(correctness, indent=2),
        "```",
        "",
        "## Performance",
        "",
        "| 分辨率 | 前向耗时 (ms) | 显存 (MB) |",
        "|---|---:|---:|",
    ]
    for row in perf_rows:
        lines.append(f"| {row['resolution']} | {row['forward_time_ms']} | {row['memory_MB']} |")
    lines.extend(
        [
            "",
            "## SUPA torch.fft probe",
            "```json",
            json.dumps(fft_probe, indent=2),
            "```",
            "",
            f"Artifacts: `{out_json}`",
            "",
        ]
    )
    out_md = RUN_LOG_DIR / f"official_baseline_{day}.md"
    out_md.write_text("\n".join(lines))

    summary = {}
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
    summary.setdefault("meta", {})
    summary["meta"]["updated_at"] = stamp
    summary["official_baseline"] = {
        "status": "pass" if payload["ok"] else "fail",
        "device": correctness.get("device"),
        "supa_fft_ok": fft_probe.get("ok"),
        "perf": perf_rows,
        "run_log": str(out_md),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print({"summary": str(SUMMARY_PATH), "run_log": str(out_md), "json": str(out_json)})
    return out_md


def main() -> None:
    device = torch.device("cpu")
    print({"torch": torch.__version__, "baseline_device": str(device), "numpy": np.__version__})

    correctness = test_correctness(device)
    print("\n--- 性能测试 (CPU baseline) ---")
    perf_rows = benchmark_performance(device)
    print("\n--- SUPA torch.fft probe ---")
    fft_probe = probe_supa_fft()
    write_artifacts(correctness, perf_rows, fft_probe)
    print({"task": "official_spectral_conv_baseline", "ok": True})


if __name__ == "__main__":
    main()
