#!/usr/bin/env python3
"""Wave-1: segment timing for CURRENT fused path (旁注；不写 spectral_conv.perf)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_br  # noqa: F401

import spectral_conv_ext
from reference_pytorch import make_random_weights
from spectral_conv_ops import (
    _host_out_buffer,
    _out_freq_buffer,
    _roundtrip_supa_input,
    _weights_to_supa_cached,
    spectral_conv2d_fused,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RUN_LOG_DIR = RESULTS_DIR / "run_logs"
SUMMARY_PATH = RESULTS_DIR / "summary.json"

B, CIN, COUT = 4, 32, 64
MODES = 16
ITERS = 30
WARMUP = 8


def sync() -> None:
    torch.supa.synchronize()


def profile_fused_segments(height: int, width: int) -> dict:
    x = torch.randn(B, CIN, height, width)
    w1 = make_random_weights(CIN, COUT, MODES, MODES, 0)
    w2 = make_random_weights(CIN, COUT, MODES, MODES, 1)
    # warmup e2e
    for _ in range(WARMUP):
        _ = spectral_conv2d_fused(x, w1, w2, MODES, MODES, to_cpu=True)
        sync()

    t_h2d = t_r2c = t_mul = t_c2r = t_d2h = t_e2e = 0.0
    width_freq = width // 2 + 1
    trunc_col = (MODES / max(width_freq, 1)) <= 0.50

    for _ in range(ITERS):
        sync()
        t0 = time.perf_counter()
        y = spectral_conv2d_fused(x, w1, w2, MODES, MODES, to_cpu=True)
        sync()
        t_e2e += time.perf_counter() - t0
        _ = y

        # segmented (same kernels as fused hot path)
        sync()
        t0 = time.perf_counter()
        x_supa = x.detach().to("supa", torch.float32).contiguous()
        sync()
        t_h2d += time.perf_counter() - t0

        w1_supa = _weights_to_supa_cached(w1)
        w2_supa = _weights_to_supa_cached(w2)

        sync()
        t0 = time.perf_counter()
        if trunc_col:
            x_freq = spectral_conv_ext.rfft2_sufft_trunc(x_supa, MODES)
            freq_w = MODES
        else:
            x_freq = spectral_conv_ext.rfft2_sufft(x_supa)
            freq_w = width_freq
        sync()
        t_r2c += time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()
        out_freq = _out_freq_buffer(
            B, COUT, height, freq_w, x_freq.device, zero=False, modes1=MODES, modes2=MODES
        )
        spectral_conv_ext.spectral_mul_dual_full_scatter_out(
            x_freq, w1_supa, w2_supa, out_freq, MODES, MODES, False
        )
        sync()
        t_mul += time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()
        if trunc_col:
            y_supa = spectral_conv_ext.irfft2_sufft_trunc(out_freq, height, width, MODES)
        else:
            y_supa = spectral_conv_ext.irfft2_sufft(out_freq, height, width)
        sync()
        t_c2r += time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()
        host = _host_out_buffer(y_supa)
        host.copy_(y_supa.detach(), non_blocking=False)
        sync()
        t_d2h += time.perf_counter() - t0

    scale = 1000.0 / ITERS
    return {
        "path": "fused_formal",
        "resolution": f"{height}x{width}",
        "e2e_ms": round(t_e2e * scale, 3),
        "h2d_ms": round(t_h2d * scale, 3),
        "r2c_ms": round(t_r2c * scale, 3),
        "mul_scatter_ms": round(t_mul * scale, 3),
        "c2r_ms": round(t_c2r * scale, 3),
        "d2h_ms": round(t_d2h * scale, 3),
        "trunc_col": trunc_col,
        "note": "旁注分段；不得覆盖 summary.spectral_conv.perf",
    }


def main() -> None:
    rows = []
    for h, w in [(64, 64), (128, 128), (256, 256)]:
        print(f"fused segments {h}x{w}...")
        row = profile_fused_segments(h, w)
        print(row)
        rows.append(row)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = datetime.now().strftime("%Y-%m-%d")
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / f"spectral_fused_segments_{day}.md"

    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    formal = ((summary.get("spectral_conv") or {}).get("perf") or {}).get("rows") or []
    cpu = (summary.get("official_baseline") or {}).get("perf") or []

    def ms(rows_list, res_prefix):
        for r in rows_list:
            if str(r.get("resolution", "")).startswith(res_prefix):
                return float(r.get("forward_time_ms", 0) or 0)
        return None

    speedup_lines = []
    for res in ("64", "128", "256"):
        f = ms(formal, res)
        c = None
        for r in cpu:
            if str(r.get("resolution", "")).startswith(res):
                c = float(r.get("forward_time_ms", 0) or 0)
        if f and c:
            speedup_lines.append(f"- {res}: CPU {c:.3f} ms / fused {f:.3f} ms → **{c/f:.1f}×**（相对官网 CPU 参考，非竞品 GPU）")

    c2r_share = []
    for r in rows:
        total = r["r2c_ms"] + r["mul_scatter_ms"] + r["c2r_ms"] + 1e-9
        c2r_share.append(
            f"- {r['resolution']}: C2R {r['c2r_ms']} ms / (R2C+mul+C2R)={total:.3f} → share≈{100*r['c2r_ms']/total:.1f}%"
        )

    log_path.write_text(
        "\n".join(
            [
                "# Spectral fused segment profile（旁注）",
                "",
                f"- time_utc: {stamp}",
                f"- config: B={B} Cin={CIN} Cout={COUT} modes={MODES} warmup={WARMUP} iters={ITERS}",
                "- **不写** `summary.spectral_conv.perf`（formal 主表冻结）",
                "",
                "## Formal idle 主表（只读）",
                "",
                f"```json\n{json.dumps(formal, indent=2)}\n```",
                "",
                "## vs 官网 CPU 加速比",
                "",
                *speedup_lines,
                "",
                "## Fused 分段（本脚本）",
                "",
                "```json",
                json.dumps(rows, indent=2),
                "```",
                "",
                "## C2R 墙占比（设备段）",
                "",
                *c2r_share,
                "",
                "## 结论",
                "",
                "- mul/scatter 已非主耗时；墙在 C2R（irfft）与端到端同步边界。",
                "- 加速比叙事锚定 official_baseline CPU，禁止写成官方 GPU/SOL 榜。",
                "",
            ]
        )
    )

    opt = summary.setdefault("optimization", {})
    opt["fused_segments_2026_08_01"] = {
        "status": "done",
        "run_log": str(log_path.relative_to(RESULTS_DIR.parent)),
        "rows": rows,
        "note": "旁注；formal perf 未改",
    }
    summary.setdefault("meta", {})["updated_at"] = stamp
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print({"ok": True, "run_log": str(log_path)})


if __name__ == "__main__":
    main()
