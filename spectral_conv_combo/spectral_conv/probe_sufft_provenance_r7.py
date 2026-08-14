#!/usr/bin/env python3
"""R7 probe: suFFT pointer-provenance hypotheses for SUPA-origin activations.

Compares relative error vs CPU FFT reference and wall time for three policies:
  1) baseline: per-call pinned host round-trip (_roundtrip_supa_input)
  2) host_pool_overwrite: preallocated host→SUPA buffer; copy_ before rfft
  3) writeback_pool: spectral out written into host-origin buffer; skip RT if
     storage is known host-originated (simulated by always writing into pool)

Hard rule: rel > 1e-4 => FAIL. Only PASS + faster than baseline is mergeable.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch
import torch_br  # noqa: F401

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "fno_ns"))

from reference_pytorch import make_random_weights, spectral_conv2d as spectral_conv2d_reference  # noqa: E402
from spectral_conv_ops import (  # noqa: E402
    _roundtrip_supa_input,
    clear_weight_supa_cache,
    spectral_conv2d_fused,
    spectral_conv2d_supa,
)
import spectral_conv_ext  # noqa: E402

THRESHOLD = 1.0e-4
WARMUP = 8
ITERS = 30


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu().reshape(-1)
    b = b.detach().float().cpu().reshape(-1)
    denom = max(float(torch.linalg.vector_norm(b).item()), 1e-12)
    return float(torch.linalg.vector_norm(a - b).item() / denom)


def _host_origin_supa(shape, dtype=torch.float32) -> torch.Tensor:
    """Allocate SUPA storage that was first filled from host (CPU→SUPA)."""
    host = torch.zeros(shape, dtype=dtype, pin_memory=True)
    return host.to("supa", non_blocking=False).contiguous()


def fused_with_policy(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    modes: int,
    *,
    policy: str,
    pool: torch.Tensor | None,
) -> torch.Tensor:
    """Minimal fused path with selectable SUPA-input materialization."""
    if policy == "baseline":
        x_supa = _roundtrip_supa_input(x) if x.device.type == "supa" else x.to("supa")
    elif policy == "host_pool_overwrite":
        assert pool is not None
        if x.device.type == "supa":
            pool.copy_(x.detach(), non_blocking=False)
            x_supa = pool
        else:
            x_supa = x.to("supa").contiguous()
    elif policy == "writeback_pool":
        # Caller is expected to already place x in a host-origin buffer.
        # If not, fall back to pool overwrite (same as policy 2).
        assert pool is not None
        if x.data_ptr() == pool.data_ptr():
            x_supa = pool
        else:
            pool.copy_(x.detach(), non_blocking=False)
            x_supa = pool
    else:
        raise ValueError(policy)

    x_supa = x_supa.contiguous()
    b, _c, h, w = x_supa.shape
    cout = int(w1.shape[1])
    wf = w // 2 + 1
    from spectral_conv_ops import _weights_to_supa_cached, _out_freq_buffer, _y_freq_buffer

    w1s = _weights_to_supa_cached(w1)
    w2s = _weights_to_supa_cached(w2)
    x_freq = spectral_conv_ext.rfft2_sufft(x_supa)
    out_freq = _out_freq_buffer(b, cout, h, wf, x_freq.device)
    c1 = x_freq[:, :, :modes, :modes, :].contiguous()
    c2 = x_freq[:, :, -modes:, :modes, :].contiguous()
    y1 = _y_freq_buffer(b, cout, modes, modes, c1.device, 0)
    y2 = _y_freq_buffer(b, cout, modes, modes, c2.device, 1)
    spectral_conv_ext.spectral_mul_dual_out(c1, w1s, c2, w2s, y1, y2)
    out_freq[:, :, :modes, :modes, :] = y1
    out_freq[:, :, -modes:, :modes, :] = y2
    return spectral_conv_ext.irfft2_sufft(out_freq, h, w)


def bench_single_layer(policy: str) -> dict:
    torch.manual_seed(0)
    b, cin, cout, h, modes = 16, 32, 32, 64, 16
    # Device-produced activation (simulates FNO layer input after GELU).
    x0 = torch.randn(b, cin, h, h, device="supa")
    # Force a device kernel write so storage is device-originated.
    x = torch.nn.functional.gelu(x0)
    w1 = make_random_weights(cin, cout, modes, modes, 11)
    w2 = make_random_weights(cin, cout, modes, modes, 22)
    ref = spectral_conv2d_reference(x.cpu(), w1, modes, modes, weights2=w2)

    pool = _host_origin_supa(x.shape)

    clear_weight_supa_cache()
    # Correctness
    if policy == "writeback_pool":
        # Start from host-origin pool filled with x; after fused, write y back.
        pool.copy_(x)
        y = fused_with_policy(pool, w1, w2, modes, policy=policy, pool=pool)
        # Simulate next-layer writeback into a second host-origin buffer.
        out_pool = _host_origin_supa(y.shape)
        out_pool.copy_(y)
        y_out = out_pool
    else:
        y_out = fused_with_policy(x, w1, w2, modes, policy=policy, pool=pool)

    rel = _rel(y_out, ref)
    ok = rel <= THRESHOLD and bool(torch.isfinite(y_out.detach().float().cpu()).all())

    out_pool = _host_origin_supa((b, cout, h, h))
    # Timing (same policy)
    for _ in range(WARMUP):
        if policy == "writeback_pool":
            pool.copy_(x)
            y = fused_with_policy(pool, w1, w2, modes, policy=policy, pool=pool)
            out_pool.copy_(y)
        else:
            fused_with_policy(x, w1, w2, modes, policy=policy, pool=pool)
    torch_br.supa.synchronize()
    xs = []
    for _ in range(ITERS):
        torch_br.supa.synchronize()
        t0 = time.perf_counter()
        if policy == "writeback_pool":
            pool.copy_(x)
            y = fused_with_policy(pool, w1, w2, modes, policy=policy, pool=pool)
            out_pool.copy_(y)
        else:
            fused_with_policy(x, w1, w2, modes, policy=policy, pool=pool)
        torch_br.supa.synchronize()
        xs.append((time.perf_counter() - t0) * 1000)
    return {
        "policy": policy,
        "rel": rel,
        "ok": ok,
        "median_ms": statistics.median(xs),
        "mean_ms": statistics.mean(xs),
    }


def bench_chain_writeback() -> dict:
    """Full 4-layer micro-chain: write each spectral out into host-origin pool
    before IN/GELU-like ops, measuring whether subsequent layers can skip RT.
    """
    from model import FNO2d

    torch.manual_seed(0)
    m = FNO2d(16, 16, 32, 4, 10, 1).eval()
    x_cpu = torch.randn(4, 10, 64, 64)
    with torch.no_grad():
        y_cpu = m.forward(x_cpu, use_supa=False)
    m.prepare_supa_eval()
    x = x_cpu.to("supa")
    with torch.no_grad():
        y_base = m.forward_supa_chain(x, use_sufft="auto")
    rel_base = _rel(y_base, y_cpu)

    # Manual chain with host-origin ring between layers for spectral input only.
    from model import FourierLayer

    def forward_ring(model: FNO2d, inp: torch.Tensor) -> torch.Tensor:
        device = inp.device
        grid = model._cached_grid(inp.shape, device)
        h = model.lift(torch.cat([inp, grid], dim=1))
        # Host-origin activation pool matching width features.
        pool = _host_origin_supa(h.shape)
        pool.copy_(h)
        h = pool
        for layer in model.fourier_layers:
            # Force spectral to see host-origin storage via copy into pool.
            if h.data_ptr() != pool.data_ptr():
                if tuple(pool.shape) != tuple(h.shape):
                    pool = _host_origin_supa(h.shape)
                pool.copy_(h)
                h_in = pool
            else:
                h_in = h
            y = spectral_conv2d_supa(
                h_in,
                layer.spectral.weights1,
                layer.spectral.weights2,
                layer.modes1,
                layer.modes2,
                use_sufft="auto",
                to_cpu=False,
            )
            # y is device-produced from irfft — write back into a host-origin buf
            # before skip/IN/GELU so the *next* spectral input might be safe.
            # For this probe we still use baseline RT inside fused; the win we
            # test is: after writeback+IN+GELU into host pool, can we skip RT?
            y_host = _host_origin_supa(y.shape)
            y_host.copy_(y)
            skip = layer.conv(h_in)
            out = y_host + skip
            out = layer.norm(out)
            out = torch.nn.functional.gelu(out)
            # Materialize activation into host-origin for next layer.
            if tuple(pool.shape) != tuple(out.shape):
                pool = _host_origin_supa(out.shape)
            pool.copy_(out)
            h = pool
        return model.project(h).cpu()

    # Variant that skips RT when ptr is host-pool (monkeypatch).
    import spectral_conv_ops as ops

    orig = ops._roundtrip_supa_input
    skip_count = {"n": 0, "rt": 0}
    host_ptrs: set[int] = set()

    def tracking_roundtrip(x: torch.Tensor) -> torch.Tensor:
        if x.data_ptr() in host_ptrs:
            skip_count["n"] += 1
            return x.contiguous()
        skip_count["rt"] += 1
        return orig(x)

    # Rebuild chain registering pool ptrs
    def forward_ring_skip(model: FNO2d, inp: torch.Tensor) -> tuple[torch.Tensor, dict]:
        host_ptrs.clear()
        skip_count["n"] = 0
        skip_count["rt"] = 0
        ops._roundtrip_supa_input = tracking_roundtrip
        try:
            device = inp.device
            grid = model._cached_grid(inp.shape, device)
            h = model.lift(torch.cat([inp, grid], dim=1))
            pool = _host_origin_supa(h.shape)
            pool.copy_(h)
            host_ptrs.add(pool.data_ptr())
            h = pool
            for layer in model.fourier_layers:
                if h.data_ptr() not in host_ptrs:
                    pool = _host_origin_supa(h.shape)
                    pool.copy_(h)
                    host_ptrs.add(pool.data_ptr())
                    h_in = pool
                else:
                    h_in = h
                y = spectral_conv2d_supa(
                    h_in,
                    layer.spectral.weights1,
                    layer.spectral.weights2,
                    layer.modes1,
                    layer.modes2,
                    use_sufft="auto",
                    to_cpu=False,
                )
                y_host = _host_origin_supa(y.shape)
                y_host.copy_(y)
                host_ptrs.add(y_host.data_ptr())
                skip = layer.conv(h_in)
                out = y_host + skip
                out = layer.norm(out)
                out = torch.nn.functional.gelu(out)
                pool = _host_origin_supa(out.shape)
                pool.copy_(out)
                host_ptrs.add(pool.data_ptr())
                h = pool
            return model.project(h).cpu(), dict(skip_count)
        finally:
            ops._roundtrip_supa_input = orig

    with torch.no_grad():
        y_skip, counts = forward_ring_skip(m, x)
    rel_skip = _rel(y_skip, y_cpu)

    # Timing baseline vs skip-ring
    def time_fn(fn, n=ITERS):
        for _ in range(WARMUP):
            fn()
        torch_br.supa.synchronize()
        xs = []
        for _ in range(n):
            torch_br.supa.synchronize()
            t0 = time.perf_counter()
            fn()
            torch_br.supa.synchronize()
            xs.append((time.perf_counter() - t0) * 1000)
        return statistics.median(xs)

    with torch.no_grad():
        med_base = time_fn(lambda: m.forward_supa_chain(x, use_sufft="auto"))
        med_skip = time_fn(lambda: forward_ring_skip(m, x)[0])

    return {
        "rel_baseline_chain": rel_base,
        "rel_writeback_skip_rt": rel_skip,
        "ok_skip": rel_skip <= THRESHOLD,
        "median_ms_baseline": med_base,
        "median_ms_writeback_skip": med_skip,
        "skip_counts_last": counts,
    }


def main() -> int:
    rows = []
    for policy in ("baseline", "host_pool_overwrite", "writeback_pool"):
        row = bench_single_layer(policy)
        rows.append(row)
        print(row)
    chain = bench_chain_writeback()
    print("chain", chain)

    base = next(r for r in rows if r["policy"] == "baseline")
    mergeable = []
    for r in rows:
        if r["policy"] == "baseline":
            continue
        if r["ok"] and r["median_ms"] < base["median_ms"] * 0.98:
            mergeable.append(r["policy"])
    # Chain-level skip is the interesting merge.
    if chain["ok_skip"] and chain["median_ms_writeback_skip"] < chain["median_ms_baseline"] * 0.98:
        mergeable.append("chain_writeback_skip_rt")
    else:
        print(
            "MERGE_DECISION: keep R6 pinned baseline; "
            f"chain_ok={chain['ok_skip']} "
            f"base={chain['median_ms_baseline']:.3f} "
            f"skip={chain['median_ms_writeback_skip']:.3f}"
        )
    print("MERGEABLE", mergeable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
