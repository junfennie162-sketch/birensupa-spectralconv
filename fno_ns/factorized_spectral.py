"""Factorized SpectralConv (F-FNO style): sequential 1D spectral along W then H.

Math differs from corner-2D SpectralConv — needs retrain. This module is for
A/B and optional FNO experiments; contest SpectralConv path stays 2D corners.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedSpectralConv2d(nn.Module):
    """Cascaded 1D complex multiplies on low modes (W then H)."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        # After rFFT on W: keep first modes2 bins
        self.weights_w = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes2, dtype=torch.cfloat)
        )
        # After rFFT on H: keep first modes1 bins (channels already Cout)
        self.weights_h = nn.Parameter(
            scale * torch.rand(out_channels, out_channels, modes1, dtype=torch.cfloat)
        )

    def _mul_w(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Cin, H, W] → [B, Cout, H, W]
        b, c, h, w = x.shape
        m2 = min(self.modes2, w // 2 + 1)
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            b, self.out_channels, h, w // 2 + 1, dtype=torch.complex64, device=x.device
        )
        # einsum over channel + broadcast H: [B,Cin,H,m] x [Cin,Cout,m] → [B,Cout,H,m]
        out_ft[:, :, :, :m2] = torch.einsum(
            "bchw,com->bohw",
            x_ft[:, :, :, :m2],
            self.weights_w[:, :, :m2],
        )
        return torch.fft.irfft(out_ft, n=w, dim=-1)

    def _mul_h(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → same channels (Cout→Cout)
        b, c, h, w = x.shape
        m1 = min(self.modes1, h // 2 + 1)
        # rFFT along H: treat as [B,C,W,H] then rfft last
        xt = x.permute(0, 1, 3, 2).contiguous()  # [B,C,W,H]
        x_ft = torch.fft.rfft(xt, dim=-1)
        out_ft = torch.zeros(
            b, c, w, h // 2 + 1, dtype=torch.complex64, device=x.device
        )
        out_ft[:, :, :, :m1] = torch.einsum(
            "bcwm,com->bowm",
            x_ft[:, :, :, :m1],
            self.weights_h[:, :, :m1],
        )
        y = torch.fft.irfft(out_ft, n=h, dim=-1)
        return y.permute(0, 1, 3, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu":
            x = x.cpu()
        x = x.to(torch.float32)
        return self._mul_h(self._mul_w(x))


class FactorizedFourierLayer(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral_conv = FactorizedSpectralConv2d(width, width, modes1, modes2)
        self.conv = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral_conv(x) + self.conv(x)))


def bench_factorized_vs_standard(
    *,
    batch: int = 4,
    width: int = 32,
    height: int = 64,
    modes: int = 12,
    warmup: int = 3,
    iters: int = 10,
) -> dict:
    """CPU wall-clock compare of one layer (torch path only)."""
    from model import SpectralConv2d

    torch.manual_seed(0)
    x = torch.randn(batch, width, height, height)
    std = SpectralConv2d(width, width, modes, modes)
    fac = FactorizedSpectralConv2d(width, width, modes, modes)

    def _time(fn) -> float:
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) * 1000.0 / iters

    std_ms = _time(lambda: std(x, use_supa=False))
    fac_ms = _time(lambda: fac(x))
    return {
        "shape": f"B{batch}_C{width}_{height}x{height}_m{modes}",
        "standard_torch_ms": round(std_ms, 3),
        "factorized_torch_ms": round(fac_ms, 3),
        "factorized_speedup": round(std_ms / max(fac_ms, 1e-9), 3),
        "factorized_params": sum(p.numel() for p in fac.parameters()),
        "standard_params": sum(p.numel() for p in std.parameters()),
    }


if __name__ == "__main__":
    for h in (64, 128):
        for m in (12, 16):
            print(bench_factorized_vs_standard(height=h, modes=m))
