"""Auto-tuner for SpectralConv combo knobs.

Scans the (path, buffer_max, fused_block) search space at a handful of
representative shapes, picks the Pareto-best config per `min(H, W)`, and
writes the result into `spectral_conv_ops._AUTO_TUNE_TABLE`. The kernel
reads that table on every call (zero overhead when empty).

Usage:
    python3 tune.py                       # scan + persist
    python3 tune.py --shape 64 128 256    # custom resolutions
    python3 tune.py --quick              # 3 iters per cell (smoke)
    python3 tune.py --dry-run            # scan, don't mutate global table
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import spectral_conv_ops as ops  # noqa: E402


# ---------------------------------------------------------------------------
# Probe utilities
# ---------------------------------------------------------------------------

def _time_one(op, *, warmup: int, iters: int) -> float:
    """Median wall-clock (ms) for one kernel call after `warmup` warmups."""
    for _ in range(warmup):
        op()
    if hasattr(torch_br := __import__("torch_br"), "supa"):
        torch_br.supa.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        op()
        if hasattr(torch_br := __import__("torch_br"), "supa"):
            torch_br.supa.synchronize()
        else:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "supa":
        # torch_br SUPA caching allocator exposes its own counter via reset
        # hook (we mirror test_perf.py).
        try:
            import torch_br  # noqa: F401
            torch_br.supa.reset_peak_memory_stats()
            return 0.0  # filled in by the runner below
        except Exception:
            return 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    return 0.0


def _reset_peak(device: torch.device) -> None:
    if device.type == "supa":
        try:
            import torch_br  # noqa: F401
            torch_br.supa.reset_peak_memory_stats()
        except Exception:
            pass
    elif torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _read_peak_mb(device: torch.device) -> float:
    if device.type == "supa":
        try:
            import torch_br
            return torch_br.supa.max_memory_allocated() / (1024 * 1024)
        except Exception:
            return 0.0
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0


def _make_inputs(
    shape: tuple[int, int, int, int],
    modes: tuple[int, int],
    device: torch.device,
):
    B, Cin, H, W = shape
    M1, M2 = modes
    x = torch.randn(*shape, dtype=torch.float32, device=device)
    w1 = torch.randn(Cin, 2 * Cin, M1, M2, dtype=torch.float32)
    w2 = torch.randn(Cin, 2 * Cin, M1, M2, dtype=torch.float32)
    return x, w1, w2


def _bench_config(
    shape: tuple[int, int, int, int],
    modes: tuple[int, int],
    *,
    use_sufft: bool,
    buffer_max: int,
    fused_block: int | None,
    warmup: int,
    iters: int,
) -> dict:
    """Run one config and return {forward_ms, peak_mb}."""
    ops.clear_weight_supa_cache()
    ops._AUTO_TUNE_TABLE["__global__"] = {"buffer_max": buffer_max}
    ops._AUTO_TUNE_TABLE.clear()
    ops._AUTO_TUNE_TABLE["__global__"] = {"buffer_max": buffer_max}
    if fused_block is not None:
        ops._AUTO_TUNE_TABLE["__global__"]["fused_block"] = fused_block
    x, w1, w2 = _make_inputs(shape, modes, torch.device("cpu"))
    # Force a fresh SUPA tensor on the input for the fused path.
    if use_sufft:
        x_dev = x.to("supa")
    else:
        x_dev = x

    def _call():
        if use_sufft:
            ops.spectral_conv2d_fused(
                x_dev, w1, w2, modes[0], modes[1],
                to_cpu=True, synchronize=False,
            )
        else:
            ops.spectral_conv2d_v1(
                x, w1, w2, modes[0], modes[1],
            )

    _reset_peak(x_dev.device if use_sufft else x.device)
    try:
        ms = _time_one(_call, warmup=warmup, iters=iters)
    except Exception as e:
        return {"forward_ms": float("inf"), "peak_mb": 0.0, "error": str(e)}
    peak = _read_peak_mb(x_dev.device if use_sufft else x.device)
    return {"forward_ms": round(ms, 3), "peak_mb": round(peak, 1)}


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

PATH_CHOICES = ("v1", "fused")
BUFFER_CHOICES = (2, 4, 8)
FUSED_BLOCK_CHOICES = (None, 64, 128, 256)


def _pareto_best(rows: list[dict]) -> dict:
    """Pick fastest (then lowest peak) among rows that all succeeded."""
    ok = [r for r in rows if r["forward_ms"] != float("inf")]
    if not ok:
        return {"path": "fused", "buffer_max": 4, "fused_block": None}
    ok.sort(key=lambda r: (r["forward_ms"], r["peak_mb"]))
    return ok[0]


def _scan_one_shape(
    shape: tuple[int, int, int, int],
    modes: tuple[int, int],
    *,
    warmup: int,
    iters: int,
) -> dict:
    rows: list[dict] = []
    for path in PATH_CHOICES:
        for buf_max in BUFFER_CHOICES:
            block_pool = FUSED_BLOCK_CHOICES if path == "fused" else (None,)
            for blk in block_pool:
                row = _bench_config(
                    shape, modes,
                    use_sufft=(path == "fused"),
                    buffer_max=buf_max,
                    fused_block=blk,
                    warmup=warmup, iters=iters,
                )
                row.update({"path": path, "buffer_max": buf_max, "fused_block": blk})
                rows.append(row)
    best = _pareto_best(rows)
    return {
        "shape": f"{shape[2]}x{shape[3]}",
        "min_dim": min(shape[2], shape[3]),
        "rows": rows,
        "best": best,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_SHAPES: list[tuple[int, int, int, int]] = [
    # (B, Cin, H, W)
    (4, 32, 64, 64),
    (4, 32, 128, 128),
    (4, 32, 256, 256),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs="+", type=int, default=None,
                        help="Override resolutions: e.g. --shape 64 128 256")
    parser.add_argument("--quick", action="store_true", help="3 warmup/3 iters")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't mutate the global auto-tune table")
    parser.add_argument("--out", default=str(THIS_DIR / "tune_results.json"),
                        help="Where to dump the full sweep")
    parser.add_argument("--modes", type=int, default=16)
    args = parser.parse_args()

    warmup, iters = (3, 3) if args.quick else (5, 10)

    if args.shape:
        shapes = [(4, 32, s, s) for s in args.shape]
    else:
        shapes = DEFAULT_SHAPES

    print(f"[tune] scanning {len(shapes)} shapes, warmup={warmup} iters={iters}")
    sweep = []
    for shape in shapes:
        result = _scan_one_shape(shape, (args.modes, args.modes),
                                 warmup=warmup, iters=iters)
        sweep.append(result)
        best = result["best"]
        print(f"[tune] {result['shape']}: best path={best['path']} "
              f"buf={best['buffer_max']} block={best['fused_block']} "
              f"-> {best.get('forward_ms', 'n/a')} ms, {best.get('peak_mb', 'n/a')} MB")

    table: dict[int, dict] = {}
    for entry in sweep:
        if entry["best"].get("forward_ms", float("inf")) == float("inf"):
            continue
        md = entry["min_dim"]
        use_sufft = entry["best"]["path"] == "fused"
        table[md] = {
            "use_sufft": use_sufft,
            "buffer_max": entry["best"]["buffer_max"],
            "fused_block": entry["best"]["fused_block"],
        }

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"sweep": sweep, "table": table}, indent=2))
    print(f"[tune] wrote {out_path}")

    if not args.dry_run:
        ops._AUTO_TUNE_TABLE.clear()
        for k, v in table.items():
            ops._AUTO_TUNE_TABLE[k] = v
        print(f"[tune] applied {len(table)} entries to ops._AUTO_TUNE_TABLE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())