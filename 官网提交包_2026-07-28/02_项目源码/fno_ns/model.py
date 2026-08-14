"""FNO-2d for Navier-Stokes vorticity — reuses SUPA SpectralConv mul.

Aligned with 赛题文档/官网-赛道五-模型与算子详情页.md §进阶题 C.
- Fourier Layer count >= 4
- Spectral path (eval): resolution-adaptive SUPA SpectralConv (<256 → v1, >=256 → fused)
- `use_supa=False` keeps a pure-torch differentiable path for CPU training
  (SUPA mul backward is unit-tested in `spectral_conv/test_backward.py`)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_SPECTRAL_DIR = Path(__file__).resolve().parents[1] / "spectral_conv"
if str(_SPECTRAL_DIR) not in sys.path:
    sys.path.insert(0, str(_SPECTRAL_DIR))

from spectral_conv_ops import spectral_conv2d_supa  # noqa: E402


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution with two frequency corners (official FNO style)."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale
            * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale
            * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor, use_supa: bool = True) -> torch.Tensor:
        if use_supa:
            # Adaptive formal path: v1 on small maps, fused on >=256
            return spectral_conv2d_supa(
                x,
                self.weights1.detach(),
                self.weights2.detach(),
                self.modes1,
                self.modes2,
                use_sufft="auto",
            )
        # Train path: pure torch (CPU FFT + einsum). SUPA backward is covered by
        # spectral_conv/test_backward.py — do NOT call spectral_mul_autograd here;
        # each step would do 4 layers × 2 corners of Host↔Device and crawl.
        if x.device.type != "cpu":
            x = x.cpu()
        batch_size, _channels, height, width = x.shape
        x_ft = torch.fft.rfft2(x.to(torch.float32))
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.complex64,
            device="cpu",
        )
        modes1, modes2 = self.modes1, self.modes2
        out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
            x_ft[:, :, :modes1, :modes2].to(torch.complex64),
            self.weights1.to(torch.complex64),
        )
        out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(
            x_ft[:, :, -modes1:, :modes2].to(torch.complex64),
            self.weights2.to(torch.complex64),
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


class FourierLayer(nn.Module):
    """spectral conv + 1x1 skip + InstanceNorm + GELU."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.conv = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x: torch.Tensor, use_supa: bool = True) -> torch.Tensor:
        spectral = self.spectral_conv(x, use_supa=use_supa)
        skip = self.conv(x)
        return F.gelu(self.norm(spectral + skip))

    def forward_supa(self, x: torch.Tensor, *, use_sufft: bool | str = "auto") -> torch.Tensor:
        """Device-resident forward: keeps all intermediates on x.device.

        Differs from `forward` in two ways:
        1. Spectral branch pipes `to_cpu=False` end-to-end (skips the per-layer
           host staging in `spectral_conv2d_supa`).
        2. Conv1x1 weights and IN parameters are explicitly placed on
           x.device here so the residual `spectral + skip` stays on SUPA.

        Call `FNO2d.prepare_supa_eval()` once before timing to also move
        InstanceNorm running stats to SUPA. The optional `use_gn_substitute`
        kwarg of `FNO2d.forward_supa_chain` swaps InstanceNorm for
        parameter-free GroupNorm to remove the moving-stats dependency.
        """
        device = x.device
        # keep branches on the same device as x; harmless on CPU
        self.conv.to(device)
        self.norm.to(device)
        # R4: pass `nn.Parameter` raw (no `.detach()`). With `.detach()` the
        # `_weights_to_supa_cached` lookup misses the O(1) id-keyed branch
        # and falls back to a D2H + numpy + blake2b hash round-trip; 2
        # weights × 4 layers × ~1 ms ≈ 8 ms/layer of overhead. Removing
        # `.detach()` keeps the parameter object identical across calls.
        y = spectral_conv2d_supa(
            x,
            self.spectral_conv.weights1,
            self.spectral_conv.weights2,
            self.spectral_conv.modes1,
            self.spectral_conv.modes2,
            use_sufft=use_sufft,
            to_cpu=False,
        )
        skip = self.conv(x)
        return F.gelu(self.norm(y + skip))


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2D Navier-Stokes vorticity."""

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        n_layers: int = 4,
        in_channels: int = 10,
        out_channels: int = 1,
    ):
        super().__init__()
        if n_layers < 4:
            raise ValueError("contest requires Fourier Layer >= 4")
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels

        self.lift = nn.Conv2d(in_channels + 2, width, 1)
        self.fourier_layers = nn.ModuleList(
            [FourierLayer(width, modes1, modes2) for _ in range(n_layers)]
        )
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1),
        )

    @staticmethod
    def get_grid(shape, device):
        batch_size, _channels, height, width = shape
        grid_x = (
            torch.linspace(0, 1, height, device=device)
            .reshape(1, 1, height, 1)
            .expand(batch_size, 1, height, width)
        )
        grid_y = (
            torch.linspace(0, 1, width, device=device)
            .reshape(1, 1, 1, width)
            .expand(batch_size, 1, height, width)
        )
        return torch.cat([grid_x, grid_y], dim=1)

    def forward(self, x: torch.Tensor, use_supa: bool = True) -> torch.Tensor:
        """
        Args:
            x: [B, T_in, H, W] vorticity frames
        Returns:
            [B, 1, H, W] predicted vorticity
        """
        if x.dim() != 4:
            raise ValueError(f"x must be [B,T,H,W], got {tuple(x.shape)}")
        x = x.to(torch.float32).cpu()
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat([x, grid], dim=1)
        x = self.lift(x)
        for layer in self.fourier_layers:
            x = layer(x, use_supa=use_supa)
        return self.project(x)

    def prepare_supa_eval(self) -> None:
        """Move all parameters + InstanceNorm running stats to SUPA.

        Idempotent. Mirrors `ai4s-n`'s `prepare_supa_eval`; the IN running
        stats must be moved explicitly because `nn.InstanceNorm2d.to(...)`
        does NOT recurse into `running_mean` / `running_var`.
        """
        self.to("supa")
        for layer in self.fourier_layers:
            n = layer.norm
            rm = getattr(n, "running_mean", None)
            rv = getattr(n, "running_var", None)
            if rm is not None:
                n.running_mean = rm.to("supa")
                n.running_var = rv.to("supa")
                tracked = getattr(n, "num_batches_tracked", None)
                if tracked is not None:
                    n.num_batches_tracked = tracked.to("supa")

    def forward_supa_chain(
        self,
        x: torch.Tensor,
        *,
        use_sufft: bool | str = "auto",
    ) -> torch.Tensor:
        """Device-resident forward: stays on SUPA end-to-end (only final D2H).

        Matches `ai4s-n`'s `forward_supa_chain` semantics:
        - `lift`, every `FourierLayer.forward_supa`, and `project` run on SUPA.
        - `FourierLayer.forward_supa` is `to_cpu=False` all the way through
          `spectral_conv2d_supa`, so no per-layer D2H sync is issued.
        - The single final `.cpu()` is for compatibility with the existing
          CPU-side eval harness (`relative_l2` runs on CPU).

        R5: the previous `use_gn_substitute` flag was semantically broken:
        it applied a `GroupNorm(1, C)` *on top of* the InstanceNorm that
        `forward_supa` already includes — so the chain ran two norms + two
        GELUs per layer, which (a) is wrong by construction and (b) was
        silently hiding perf numbers from the R3 / R4 chain benchmarks. The
        kwarg has been removed; the InstanceNorm path is the single canonical
        implementation. `use_sufft="auto"` selects `fused` for shapes with
        `min(H, W) >= 64`, matching the auto rule in
        `spectral_conv_ops.resolve_use_sufft`.
        """
        if x.dim() != 4:
            raise ValueError(f"x must be [B,T,H,W], got {tuple(x.shape)}")
        if x.device.type != "supa":
            x = x.to("supa").to(torch.float32)
        else:
            x = x.to(torch.float32)
        # R4: `prepare_supa_eval()` is now caller responsibility (it ran
        # ~0.3-1 ms per forward; chain perf dropped in benchmarks).
        device = x.device
        grid = self.get_grid(x.shape, device)
        h = self.lift.to(device)(torch.cat([x, grid], dim=1))
        for layer in self.fourier_layers:
            # R5: take the post-InstanceNorm+GELU output directly; the IN
            # norm is already inside `forward_supa`. Do not double-norm.
            h = layer.forward_supa(h, use_sufft=use_sufft)
        return self.project.to(device)(h).cpu()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
