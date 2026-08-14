"""CPU / PyTorch reference for 2D Spectral Convolution (FNO-style, dual corners).

x: [B, C_in, H, W]
weights1/weights2: [C_in, C_out, modes1, modes2] complex64
y: [B, C_out, H, W]

Positive low modes use weights1; negative freq rows use weights2 (official FNO).
"""

from __future__ import annotations

import torch


def spectral_conv2d(
    x: torch.Tensor,
    weights: torch.Tensor,
    modes1: int,
    modes2: int,
    weights2: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"x must be [B,C,H,W], got {tuple(x.shape)}")
    if weights.dim() != 4:
        raise ValueError(f"weights must be [C_in,C_out,modes1,modes2], got {tuple(weights.shape)}")
    if not torch.is_complex(weights):
        raise ValueError("weights must be complex64/complex128")

    batch_size, channels_in, height, width = x.shape
    channels_out = weights.shape[1]
    if weights.shape[0] != channels_in:
        raise ValueError("weights C_in must match x C_in")
    if weights.shape[2] != modes1 or weights.shape[3] != modes2:
        raise ValueError("weights trailing dims must equal modes1/modes2")
    if modes1 > height or modes2 > (width // 2 + 1):
        raise ValueError("modes exceed rfft2 spectrum size")

    x_freq = torch.fft.rfft2(x)
    out_freq = torch.zeros(
        batch_size,
        channels_out,
        height,
        width // 2 + 1,
        dtype=torch.complex64,
        device=x.device,
    )
    out_freq[:, :, :modes1, :modes2] = torch.einsum(
        "bixy,ioxy->boxy",
        x_freq[:, :, :modes1, :modes2].to(torch.complex64),
        weights.to(torch.complex64),
    )
    if weights2 is not None:
        if weights2.shape != weights.shape:
            raise ValueError("weights2 shape must match weights1")
        out_freq[:, :, -modes1:, :modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_freq[:, :, -modes1:, :modes2].to(torch.complex64),
            weights2.to(torch.complex64),
        )
    return torch.fft.irfft2(out_freq, s=(height, width))


def compl_mul3d(input_freq: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bixyz,ioxyz->boxyz", input_freq, weights)


def spectral_conv3d(
    x: torch.Tensor,
    weights1: torch.Tensor,
    weights2: torch.Tensor,
    weights3: torch.Tensor,
    weights4: torch.Tensor,
    modes1: int,
    modes2: int,
    modes3: int,
) -> torch.Tensor:
    """3D spectral conv (official 4-corner), CPU reference.

    x: [B, C_in, D, H, W]
    weights*: [C_in, C_out, modes1, modes2, modes3] complex
    Corners: [:m1,:m2,:m3], [-m1:,:m2,:m3], [:m1,-m2:,:m3], [-m1:,-m2:,:m3]
    """
    if x.dim() != 5:
        raise ValueError(f"x must be [B,C,D,H,W], got {tuple(x.shape)}")
    batch_size, _channels_in, depth, height, width = x.shape
    channels_out = weights1.shape[1]
    x_freq = torch.fft.rfftn(x, dim=(-3, -2, -1)).to(torch.complex64)
    out_freq = torch.zeros(
        batch_size,
        channels_out,
        depth,
        height,
        width // 2 + 1,
        dtype=torch.complex64,
        device=x.device,
    )
    w1 = weights1.to(torch.complex64)
    w2 = weights2.to(torch.complex64)
    w3 = weights3.to(torch.complex64)
    w4 = weights4.to(torch.complex64)
    out_freq[:, :, :modes1, :modes2, :modes3] = compl_mul3d(
        x_freq[:, :, :modes1, :modes2, :modes3], w1
    )
    out_freq[:, :, -modes1:, :modes2, :modes3] = compl_mul3d(
        x_freq[:, :, -modes1:, :modes2, :modes3], w2
    )
    out_freq[:, :, :modes1, -modes2:, :modes3] = compl_mul3d(
        x_freq[:, :, :modes1, -modes2:, :modes3], w3
    )
    out_freq[:, :, -modes1:, -modes2:, :modes3] = compl_mul3d(
        x_freq[:, :, -modes1:, -modes2:, :modes3], w4
    )
    return torch.fft.irfftn(out_freq, s=(depth, height, width), dim=(-3, -2, -1))


def make_random_weights3d(
    channels_in: int,
    channels_out: int,
    modes1: int,
    modes2: int,
    modes3: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    real_part = torch.empty(
        (channels_in, channels_out, modes1, modes2, modes3),
        dtype=torch.float32,
        device="cpu",
    )
    imag_part = torch.empty_like(real_part)
    real_part.uniform_(-0.5, 0.5, generator=generator)
    imag_part.uniform_(-0.5, 0.5, generator=generator)
    return torch.complex(real_part, imag_part)


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
