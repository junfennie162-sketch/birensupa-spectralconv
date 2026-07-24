"""Combo SpectralConv (official-aligned).

v1: CPU rFFT/iFFT + SUPA mul (dual-corner, one sync, weight cache)
fused: suFFT on SUPA + on-device dual-corner mul (>=256 / use_sufft=True)

Optional: fused path can keep output on SUPA (`to_cpu=False`) to cut Host copies.
Buffer reuse: recycle fused spectrum + host staging tensors (TurboFNO traffic idea on Biren).
"""

from __future__ import annotations

import hashlib
import weakref

import torch
import torch.nn as nn
import torch_br  # noqa: F401

import spectral_conv_ext

_WEIGHT_SUPA_CACHE: dict[tuple, torch.Tensor] = {}
# id -> (weakref, version, cached_supa). Re-validate object identity to avoid id reuse.
_PARAM_CACHE: dict[int, tuple] = {}
_WEIGHT_CACHE_MAX = 64
# Reuse fused spectrum / host output buffers to cut allocator + D2H overhead on hot path.
_OUT_FREQ_CACHE: dict[tuple, torch.Tensor] = {}
_HOST_OUT_CACHE: dict[tuple, torch.Tensor] = {}
_OUT_FREQ_CPU_CACHE: dict[tuple, torch.Tensor] = {}
_BUFFER_CACHE_MAX = 4
# Per-corner output buffer reuse: `spectral_mul` allocates a fresh
# (B, Cout, M1, M2, 2) SUPA tensor each call; for hot loops that's
# ~20 ms/call of pure allocator churn on Biren. Cache it.
_Y_FREQ_CACHE: dict[tuple, torch.Tensor] = {}
# NOTE: caching the per-call corner slices (`x_freq[:, :, :M1, :M2, :]`)
# did NOT help — the SUPA `.copy_()` from a strided slice costs more than
# `.contiguous()` because PyTorch's strided->contig path on SUPA is a
# direct `cudaMemcpy2D` while `.copy_()` round-trips through a kernel.
# Kept as a placeholder so future experiments can re-enable.

# --- Auto-tuning knobs --------------------------------------------------------
# Per-shape cached decision: min(H,W) -> {use_sufft: bool, buffer_max: int,
# fused_block: int|None}. Populated by `tune.py` (or online during first call
# if `auto_tune=True` is passed). Empty by default so behaviour is identical
# to the hand-tuned defaults below.
_AUTO_TUNE_TABLE: dict[int, dict] = {}
_AUTO_TUNE_DEFAULTS: dict = {
    "use_sufft": "auto",   # path selector; one of {"auto","v1","fused"}
    "buffer_max": 4,       # per-cache entry cap
    "fused_block": None,   # block size hint for the on-device mul (None=auto)
}


def _auto_tune_decision(min_dim: int) -> dict:
    """Return the cached decision for `min_dim` (or the default)."""
    return _AUTO_TUNE_TABLE.get(min_dim, _AUTO_TUNE_DEFAULTS)


def _buffer_cache_max() -> int:
    """Effective buffer cache cap. Tunable per-shape via auto-tune table."""
    override = _AUTO_TUNE_TABLE.get("__global__")
    if override is not None and "buffer_max" in override:
        return max(1, int(override["buffer_max"]))
    return _BUFFER_CACHE_MAX


def _out_freq_buffer(
    batch_size: int,
    channels_out: int,
    height: int,
    width_freq: int,
    device: torch.device,
) -> torch.Tensor:
    """Reusable SUPA-side spectrum buffer (peak-memory optimisation).

    Hold a single allocations worth of `(B, Cout, H, Wf, 2)` so the fused
    path doesn't re-allocate on every call, which keeps peak memory bounded
    and avoids `cudaMalloc` overhead. The next caller of the same key
    overwrites it via `zero_()`. The cap is read from
    `_buffer_cache_max()` so the auto-tuner can shrink it under memory
    pressure without code changes.
    """
    key = (batch_size, channels_out, height, width_freq, str(device))
    buf = _OUT_FREQ_CACHE.get(key)
    if buf is None:
        buf = torch.zeros(
            batch_size,
            channels_out,
            height,
            width_freq,
            2,
            device=device,
            dtype=torch.float32,
        )
        cap = _buffer_cache_max()
        if len(_OUT_FREQ_CACHE) >= cap:
            _OUT_FREQ_CACHE.pop(next(iter(_OUT_FREQ_CACHE)))
        _OUT_FREQ_CACHE[key] = buf
    else:
        buf.zero_()
    return buf


def _host_out_buffer(sample: torch.Tensor) -> torch.Tensor:
    """Host staging for D2H; pinned when available. Returns a buffer we own."""
    key = (tuple(sample.shape), sample.dtype)
    buf = _HOST_OUT_CACHE.get(key)
    if buf is None:
        try:
            buf = torch.empty(sample.shape, dtype=sample.dtype, pin_memory=True)
        except (RuntimeError, TypeError):
            buf = torch.empty(sample.shape, dtype=sample.dtype)
        if len(_HOST_OUT_CACHE) >= _buffer_cache_max():
            _HOST_OUT_CACHE.pop(next(iter(_HOST_OUT_CACHE)))
        _HOST_OUT_CACHE[key] = buf
    return buf


def _complex_to_interleaved(tensor_complex: torch.Tensor) -> torch.Tensor:
    tensor_complex = tensor_complex.to(torch.complex64).contiguous()
    return torch.view_as_real(tensor_complex).contiguous()


def _interleaved_to_complex(tensor_real_imag: torch.Tensor) -> torch.Tensor:
    return torch.view_as_complex(tensor_real_imag.contiguous())


def spectral_mul_supa(
    x_freq_modes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Legacy single-corner bridge (returns CPU complex). Prefer dual v1 path."""
    x_cpu = x_freq_modes.detach().to("cpu", torch.complex64).contiguous()
    w_cpu = weights.detach().to("cpu", torch.complex64).contiguous()
    x_supa = _complex_to_interleaved(x_cpu).to("supa").contiguous()
    w_supa = _weights_to_supa_cached(w_cpu)
    y_supa = spectral_conv_ext.spectral_mul(x_supa, w_supa)
    torch_br.supa.synchronize()
    return _interleaved_to_complex(y_supa.cpu())


def spectral_mul_supa_device(
    x_freq_modes_interleaved: torch.Tensor,
    weights_interleaved: torch.Tensor,
    synchronize: bool = False,
) -> torch.Tensor:
    y = spectral_conv_ext.spectral_mul(
        x_freq_modes_interleaved.contiguous(),
        weights_interleaved.contiguous(),
    )
    if synchronize:
        torch_br.supa.synchronize()
    return y


def _weights_to_supa_cached(weights: torch.Tensor) -> torch.Tensor:
    """Cache SUPA weights.

    Parameters: keyed by id + weakref identity check (avoids data_ptr/id reuse bugs).
    Other tensors: content hash.
    """
    if isinstance(weights, nn.Parameter):
        wid = id(weights)
        version = int(getattr(weights, "_version", -1))
        entry = _PARAM_CACHE.get(wid)
        if entry is not None:
            ref, ver, cached = entry
            obj = ref()
            if obj is weights and ver == version:
                return cached
        w_cpu = weights.detach().to("cpu", torch.complex64).contiguous()
        cached = _complex_to_interleaved(w_cpu).to("supa").contiguous()
        _PARAM_CACHE[wid] = (weakref.ref(weights), version, cached)
        if len(_PARAM_CACHE) > _WEIGHT_CACHE_MAX * 2:
            dead = [k for k, (r, _, _) in _PARAM_CACHE.items() if r() is None]
            for k in dead:
                _PARAM_CACHE.pop(k, None)
        return cached

    w_cpu = weights.detach().to("cpu", torch.complex64).contiguous()
    digest = hashlib.blake2b(w_cpu.numpy().tobytes(), digest_size=16).hexdigest()
    key = (tuple(w_cpu.shape), digest)
    cached = _WEIGHT_SUPA_CACHE.get(key)
    if cached is None:
        cached = _complex_to_interleaved(w_cpu).to("supa").contiguous()
        if len(_WEIGHT_SUPA_CACHE) >= _WEIGHT_CACHE_MAX:
            _WEIGHT_SUPA_CACHE.pop(next(iter(_WEIGHT_SUPA_CACHE)))
        _WEIGHT_SUPA_CACHE[key] = cached
    return cached


def clear_weight_supa_cache() -> None:
    _WEIGHT_SUPA_CACHE.clear()
    _PARAM_CACHE.clear()
    _OUT_FREQ_CACHE.clear()
    _HOST_OUT_CACHE.clear()
    _OUT_FREQ_CPU_CACHE.clear()


def _y_freq_buffer(
    batch_size: int,
    channels_out: int,
    modes1: int,
    modes2: int,
    device: torch.device,
    corner_id: int = 0,
) -> torch.Tensor:
    """Reusable per-corner `(B, Cout, M1, M2, 2)` SUPA spectrum buffer.

    `spectral_mul_out` writes into this directly so the fused path doesn't
    pay the per-call `cudaMalloc` cost of returning a fresh SUPA tensor.
    Backed by `spectral_conv_ext.spectral_mul_out` (added 2026-07-24).
    `corner_id` keeps the two angular buffers separate — sharing one tensor
    across corner1 + corner2 silently overlaps writes and corrupts output.
    Cap is `_BUFFER_CACHE_MAX`.
    """
    key = (batch_size, channels_out, modes1, modes2, str(device), corner_id)
    buf = _Y_FREQ_CACHE.get(key)
    if buf is None:
        buf = torch.zeros(
            batch_size, channels_out, modes1, modes2, 2,
            device=device, dtype=torch.float32,
        )
        cap = _buffer_cache_max()
        if len(_Y_FREQ_CACHE) >= cap:
            _Y_FREQ_CACHE.pop(next(iter(_Y_FREQ_CACHE)))
        _Y_FREQ_CACHE[key] = buf
    # R4: drop the cache-hit `buf.zero_()`. `spectral_mul_out` writes every
    # element of `(B, Cout, M1, M2, 2)`, so the prior call's contents are
    # fully overwritten. Removing the zero saves a D2D-zero kernel launch
    # per call (≈0.1-0.3 ms × 2 corners per fused call).
    return buf


def _out_freq_cpu_buffer(
    batch_size: int,
    channels_out: int,
    height: int,
    width_freq: int,
) -> torch.Tensor:
    """Reuse host spectrum buffer on v1 path to cut allocator churn / peak spikes."""
    key = (batch_size, channels_out, height, width_freq)
    buf = _OUT_FREQ_CPU_CACHE.get(key)
    if buf is None:
        buf = torch.zeros(
            batch_size,
            channels_out,
            height,
            width_freq,
            dtype=torch.complex64,
            device="cpu",
        )
        if len(_OUT_FREQ_CPU_CACHE) >= _buffer_cache_max():
            _OUT_FREQ_CPU_CACHE.pop(next(iter(_OUT_FREQ_CPU_CACHE)))
        _OUT_FREQ_CPU_CACHE[key] = buf
    else:
        buf.zero_()
    return buf


class SpectralMulFunction(torch.autograd.Function):
    """SUPA spectral_mul forward + conjugate-transpose complex einsum backward."""

    @staticmethod
    def forward(ctx, x_freq_modes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        x_cpu = x_freq_modes.to("cpu", torch.complex64).contiguous()
        w_cpu = weights.to("cpu", torch.complex64).contiguous()
        ctx.save_for_backward(x_cpu, w_cpu)
        with torch.no_grad():
            y_cpu = spectral_mul_supa(x_cpu, w_cpu)
        return y_cpu

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_cpu, w_cpu = ctx.saved_tensors
        grad_y = grad_output.to("cpu", torch.complex64).contiguous()
        grad_x = torch.einsum("boxy,ioxy->bixy", grad_y, w_cpu.conj())
        grad_w = torch.einsum("bixy,boxy->ioxy", x_cpu.conj(), grad_y)
        return grad_x, grad_w


def spectral_mul_autograd(
    x_freq_modes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Differentiable wrapper around SUPA spectral_mul (extension score)."""
    return SpectralMulFunction.apply(x_freq_modes, weights)


def resolve_use_sufft(
    height: int,
    width: int,
    use_sufft: bool | str,
) -> bool:
    """Pick FFT path. Auto: consult per-shape cache, fall back to default.

    Selection precedence (highest first):
      1. Explicit bool from caller.
      2. Per-shape entry in `_AUTO_TUNE_TABLE` for `min(H, W)` (populated
         by `tune.py` — see `skills/auto_tune.md`).
      3. Default `auto` rule: fused when `min(H, W) >= 64`.

    The auto rule uses two thresholds:
      - `min(H, W) < 64`  → v1 (cold-start safe, ~5 MB peak)
      - `64 <= min(H, W)` → fused (full on-device FFT/mul/iFFT chain)

    The 64 cutoff (down from 128) came from a `profile_segments_v2.py` sweep
    on 2026-07-23: with proper warmup + Parameter-cached weights, the fused
    suFFT path beats v1 (CPU FFT + bridged SUPA mul) at *every* resolution
    >= 64, both in latency and in peak device memory. Earlier measurements
    that pushed 64 to v1 had a cold weight cache which made v1 look
    artificially cheap. Tiny resolutions (< 64) stay on v1 because the
    SUPA allocator overhead dominates at small sizes.
    """
    if isinstance(use_sufft, bool):
        return use_sufft
    if isinstance(use_sufft, str):
        key = use_sufft.strip().lower()
        if key in {"v1", "cpu"}:
            return False
        if key in {"fused", "sufft", "supa"}:
            return True
        if key in {"auto", "adaptive"}:
            cached = _AUTO_TUNE_TABLE.get(min(height, width))
            if cached is not None and "use_sufft" in cached:
                return bool(cached["use_sufft"])
            return min(height, width) >= 64
        raise ValueError(f"use_sufft string must be auto/v1/fused, got {use_sufft!r}")
    raise TypeError(f"use_sufft must be bool or str, got {type(use_sufft).__name__}")


def spectral_conv2d_v1(
    x: torch.Tensor,
    weights1: torch.Tensor,
    weights2: torch.Tensor,
    modes1: int,
    modes2: int,
) -> torch.Tensor:
    """CPU FFT + dual-corner SUPA mul (one stacked launch, one sync)."""
    x_cpu = x.detach().to("cpu", torch.float32).contiguous()
    batch_size, _cin, height, width = x_cpu.shape
    channels_out = int(weights1.shape[1])
    width_freq = width // 2 + 1

    x_freq = torch.fft.rfft2(x_cpu)
    corner1 = x_freq[:, :, :modes1, :modes2].contiguous()
    corner2 = x_freq[:, :, -modes1:, :modes2].contiguous()

    x1_supa = _complex_to_interleaved(corner1).to("supa").contiguous()
    x2_supa = _complex_to_interleaved(corner2).to("supa").contiguous()
    w1_supa = _weights_to_supa_cached(weights1)
    w2_supa = _weights_to_supa_cached(weights2)

    y1_supa = spectral_mul_supa_device(x1_supa, w1_supa, synchronize=False)
    y2_supa = spectral_mul_supa_device(x2_supa, w2_supa, synchronize=False)
    torch_br.supa.synchronize()

    out_freq = _out_freq_cpu_buffer(batch_size, channels_out, height, width_freq)
    out_freq[:, :, :modes1, :modes2] = _interleaved_to_complex(y1_supa.cpu())
    out_freq[:, :, -modes1:, :modes2] = _interleaved_to_complex(y2_supa.cpu())
    return torch.fft.irfft2(out_freq, s=(height, width))


def spectral_conv2d_fused(
    x: torch.Tensor,
    weights1: torch.Tensor,
    weights2: torch.Tensor,
    modes1: int,
    modes2: int,
    *,
    to_cpu: bool = True,
    synchronize: bool | None = None,
) -> torch.Tensor:
    """Device-resident suFFT path. Set to_cpu=False to keep output on SUPA."""
    if x.device.type == "supa":
        x_supa = x.detach().to(torch.float32).contiguous()
    else:
        x_supa = x.detach().to("supa", torch.float32).contiguous()
    batch_size, _channels_in, height, width = x_supa.shape
    channels_out = int(weights1.shape[1])
    width_freq = width // 2 + 1

    w1_supa = _weights_to_supa_cached(weights1)
    w2_supa = _weights_to_supa_cached(weights2)

    x_freq = spectral_conv_ext.rfft2_sufft(x_supa)
    out_freq = _out_freq_buffer(
        batch_size, channels_out, height, width_freq, x_freq.device
    )

    # Reuse pre-allocated corner buffers (cut per-call SUPA allocator).
    corner1 = x_freq[:, :, :modes1, :modes2, :].contiguous()
    corner2 = x_freq[:, :, -modes1:, :modes2, :].contiguous()
    y1_buf = _y_freq_buffer(batch_size, channels_out, modes1, modes2, corner1.device, corner_id=0)
    y2_buf = _y_freq_buffer(batch_size, channels_out, modes1, modes2, corner2.device, corner_id=1)
    # R5: dual-corner pybind dispatch — one C++ entry point that runs both
    # `spectral_mul` launches with a single pybind boundary. Saves ~0.04 ms
    # per fused call vs two single pybind calls (≈ 0.16 ms / chain @ L=4).
    spectral_conv_ext.spectral_mul_dual_out(
        corner1, w1_supa,
        corner2, w2_supa,
        y1_buf, y2_buf,
    )
    out_freq[:, :, :modes1, :modes2, :] = y1_buf
    out_freq[:, :, -modes1:, :modes2, :] = y2_buf

    y = spectral_conv_ext.irfft2_sufft(out_freq, height, width)
    do_sync = to_cpu if synchronize is None else synchronize
    if do_sync:
        torch_br.supa.synchronize()
    if to_cpu:
        # Staging buffer speeds device→host. We *return* the buffer itself
        # (NOT a clone) so peak memory stays bounded by one D2H copy. The
        # buffer will be overwritten by the next call into the same shape.
        host = _host_out_buffer(y)
        host.copy_(y.detach(), non_blocking=False)
        return host
    return y


def spectral_conv3d_supa(
    x: torch.Tensor,
    weights: torch.Tensor,
    modes1: int,
    modes2: int,
    modes3: int,
) -> torch.Tensor:
    """3D spectral conv: CPU rFFT3 + SUPA spectral_mul on reshaped corner modes."""
    if x.dim() != 5:
        raise ValueError(f"x must be [B,C,D,H,W], got {tuple(x.shape)}")
    x_cpu = x.detach().to("cpu", torch.float32).contiguous()
    batch_size, channels_in, depth, height, width = x_cpu.shape
    channels_out = int(weights.shape[1])

    x_freq = torch.fft.rfftn(x_cpu, dim=(-3, -2, -1))
    out_freq = torch.zeros(
        batch_size,
        channels_out,
        depth,
        height,
        width // 2 + 1,
        dtype=torch.complex64,
        device="cpu",
    )
    corner = x_freq[:, :, :modes1, :modes2, :modes3].contiguous()
    corner_2d = corner.reshape(batch_size, channels_in, modes1, modes2 * modes3)
    weights_2d = (
        weights.detach()
        .to("cpu", torch.complex64)
        .contiguous()
        .reshape(channels_in, channels_out, modes1, modes2 * modes3)
    )
    y_2d = spectral_mul_supa(corner_2d, weights_2d)
    out_freq[:, :, :modes1, :modes2, :modes3] = y_2d.reshape(
        batch_size, channels_out, modes1, modes2, modes3
    )
    return torch.fft.irfftn(out_freq, s=(depth, height, width), dim=(-3, -2, -1))


def warmup_spectral_plans(
    height: int = 256,
    width: int = 256,
    channels_in: int = 32,
    channels_out: int = 64,
    modes1: int = 16,
    modes2: int = 16,
    batch_size: int = 4,
) -> None:
    """Touch fused path once so suFFT plans / allocator settle before timing.

    Defaults match官网 §3.2 (B=4, Cin=32, Cout=64, modes=16).
    """
    x = torch.randn(batch_size, channels_in, height, width, dtype=torch.float32)
    scale = 1.0 / (channels_in * channels_out)
    w1 = scale * torch.rand(channels_in, channels_out, modes1, modes2, dtype=torch.cfloat)
    w2 = scale * torch.rand(channels_in, channels_out, modes1, modes2, dtype=torch.cfloat)
    _ = spectral_conv2d_fused(x, w1, w2, modes1, modes2, to_cpu=True)
    torch_br.supa.synchronize()


def spectral_conv2d_supa(
    x: torch.Tensor,
    weights1: torch.Tensor,
    weights2: torch.Tensor,
    modes1: int,
    modes2: int,
    use_sufft: bool | str = "auto",
    *,
    to_cpu: bool = True,
    synchronize: bool | None = None,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"x must be [B,C,H,W], got {tuple(x.shape)}")
    height, width = int(x.shape[-2]), int(x.shape[-1])
    if resolve_use_sufft(height, width, use_sufft):
        return spectral_conv2d_fused(
            x,
            weights1,
            weights2,
            modes1,
            modes2,
            to_cpu=to_cpu,
            synchronize=synchronize,
        )
    # v1 always returns CPU (FFT/iFFT on host)
    return spectral_conv2d_v1(x, weights1, weights2, modes1, modes2)


class SpectralConv2dSupa(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(
        self,
        x: torch.Tensor,
        use_sufft: bool | str = "auto",
        *,
        to_cpu: bool = True,
    ) -> torch.Tensor:
        return spectral_conv2d_supa(
            x,
            self.weights1,
            self.weights2,
            self.modes1,
            self.modes2,
            use_sufft=use_sufft,
            to_cpu=to_cpu,
        )
