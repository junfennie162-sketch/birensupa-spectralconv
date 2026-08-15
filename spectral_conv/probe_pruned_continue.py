#!/usr/bin/env python3
"""Continue testing default pruned vs suFFT. Does not write summary.json / formal ms."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch_br  # noqa: F401

from reference_pytorch import make_random_weights, spectral_conv2d
from spectral_conv_ops import spectral_conv2d_supa

THRESHOLD = 1.0e-4
OUT_JSON = Path(__file__).resolve().parents[1] / "results" / "run_logs" / "pruned_continue_test_2026-08-15.json"
OUT_MD = Path(__file__).resolve().parents[1] / "results" / "run_logs" / "pruned_continue_test_2026-08-15.md"

ACC_CASES = (
    ("tiny_8x8", 2, 2, 3, 8, 8, 2, 2, 100),
    ("small_32x32", 2, 4, 4, 32, 32, 8, 8, 200),
    ("target_64x64", 2, 4, 4, 64, 64, 12, 12, 300),
)

PATHS = (
    ("default_pruned", {}),
    ("sufft_trunc", {"SPECTRAL_PRUNED_FFT": "0", "SPECTRAL_PRUNED_INV": "0"}),
)


def rel_err(expected, actual) -> float:
    actual = actual.detach().cpu()
    denom = float(expected.norm().item())
    diff = float((expected - actual).norm().item())
    return diff if denom < 1.0e-12 else diff / denom


def apply_env(extra: dict) -> None:
    for key in ("SPECTRAL_PRUNED_FFT", "SPECTRAL_PRUNED_INV"):
        os.environ.pop(key, None)
    os.environ["SPECTRAL_TRUNC_COL"] = "auto"
    os.environ.update(extra)


def make_input(batch, cin, h, w, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(batch, cin, h, w)
    x.uniform_(-0.5, 0.5, generator=g)
    return x


def time_ms(fn, warmup=10, iters=100) -> float:
    for _ in range(warmup):
        _ = fn()
    torch.supa.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    torch.supa.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> int:
    acc = []
    for name, extra in PATHS:
        apply_env(extra)
        for case, b, cin, cout, h, w, m1, m2, seed in ACC_CASES:
            x = make_input(b, cin, h, w, seed)
            w1 = make_random_weights(cin, cout, m1, m2, seed + 17)
            w2 = make_random_weights(cin, cout, m1, m2, seed + 23)
            y_ref = spectral_conv2d(x, w1, m1, m2, weights2=w2)
            y = spectral_conv2d_supa(x, w1, w2, m1, m2)
            r = rel_err(y_ref, y)
            rec = {"path": name, "case": case, "rel": r, "ok": r <= THRESHOLD}
            print(rec)
            acc.append(rec)

    timing = []
    for name, extra in PATHS:
        apply_env(extra)
        row = {"path": name}
        for res, seed in ((64, 42), (128, 43), (256, 44)):
            x = make_input(4, 32, res, res, seed)
            w1 = torch.nn.Parameter(make_random_weights(32, 64, 16, 16, seed + 1))
            w2 = torch.nn.Parameter(make_random_weights(32, 64, 16, 16, seed + 2))
            apply_env(extra)
            ms = time_ms(lambda: spectral_conv2d_supa(x, w1, w2, 16, 16))
            row[f"ms{res}"] = ms
            print({"path": name, "res": res, "ms": ms})
        timing.append(row)

    all_ok = all(z["ok"] for z in acc)
    payload = {
        "task": "pruned_continue_test",
        "protocol": "accuracy=official 3 cases; perf=B4 Cin32 Cout64 modes16 warmup10 iters100 CPU-in",
        "all_ok": all_ok,
        "accuracy": acc,
        "timing": timing,
        "writes_formal_perf": False,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    lines = [
        "# pruned 续测（2026-08-15）",
        "",
        "> 旁注。**未写** `summary.json` / 正式 idle。协议对齐官网：正确性三案 + 性能 warmup=10/iters=100。",
        "",
        "## 正确性",
        "",
        "| 路径 | tiny 8×8 | small 32×32 | target 64×64 |",
        "|------|----------:|------------:|-------------:|",
    ]
    by = {}
    for rec in acc:
        by.setdefault(rec["path"], {})[rec["case"]] = rec
    for name, _ in PATHS:
        c = by[name]
        lines.append(
            f"| {name} | {c['tiny_8x8']['rel']:.3e} {'PASS' if c['tiny_8x8']['ok'] else 'FAIL'} "
            f"| {c['small_32x32']['rel']:.3e} {'PASS' if c['small_32x32']['ok'] else 'FAIL'} "
            f"| {c['target_64x64']['rel']:.3e} {'PASS' if c['target_64x64']['ok'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## 非正式计时（未 promote）",
        "",
        "| 路径 | 64 ms | 128 ms | 256 ms |",
        "|------|------:|-------:|-------:|",
    ]
    for row in timing:
        lines.append(
            f"| {row['path']} | {row['ms64']:.3f} | {row['ms128']:.3f} | {row['ms256']:.3f} |"
        )
    lines += [
        "",
        f"all_ok: **{str(all_ok).lower()}**",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(payload)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
