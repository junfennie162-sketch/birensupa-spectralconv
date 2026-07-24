"""Official-aligned PyTorch SpectralConv2d reference (CPU).

Matches 官网-赛道五-模型与算子详情页.md §3.1 (weights1 + weights2).
Also provides 3D reference helpers for extension-score tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
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

    def compl_mul2d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def spectral_conv3d(
    x: torch.Tensor,
    weights: torch.Tensor,
    modes1: int,
    modes2: int,
    modes3: int,
) -> torch.Tensor:
    """3D spectral conv (single low-frequency corner), CPU reference."""
    if x.dim() != 5:
        raise ValueError(f"x must be [B,C,D,H,W], got {tuple(x.shape)}")
    if weights.dim() != 5:
        raise ValueError(
            f"weights must be [C_in,C_out,m1,m2,m3], got {tuple(weights.shape)}"
        )
    batch_size, _channels_in, depth, height, width = x.shape
    channels_out = weights.shape[1]
    x_freq = torch.fft.rfftn(x, dim=(-3, -2, -1))
    out_freq = torch.zeros(
        batch_size,
        channels_out,
        depth,
        height,
        width // 2 + 1,
        dtype=torch.complex64,
        device=x.device,
    )
    out_freq[:, :, :modes1, :modes2, :modes3] = torch.einsum(
        "bixyz,ioxyz->boxyz",
        x_freq[:, :, :modes1, :modes2, :modes3].to(torch.complex64),
        weights.to(torch.complex64),
    )
    return torch.fft.irfftn(out_freq, s=(depth, height, width), dim=(-3, -2, -1))


def make_random_weights(
    channels_in: int,
    channels_out: int,
    modes1: int,
    modes2: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    real_part = torch.empty(
        (channels_in, channels_out, modes1, modes2),
        dtype=torch.float32,
        device="cpu",
    )
    imag_part = torch.empty_like(real_part)
    real_part.uniform_(-0.5, 0.5, generator=generator)
    imag_part.uniform_(-0.5, 0.5, generator=generator)
    return torch.complex(real_part, imag_part)
