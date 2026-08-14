"""FNO-NS datasets: offline NS-like generator + optional .pt loader.

Public HDF5/HF downloads are often blocked in this contest Docker; we ship a
reproducible spectral vorticity generator inspired by FNO-style NS on a torus
(viscosity damping + forcing + random initial spectrum). When a real file is
present under data/, it is preferred.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CACHE = DATA_DIR / "ns_like_v1e-3_N512_T20_64.pt"
V2_CACHE = DATA_DIR / "ns_like_v2_N1024_T30_64.pt"


class SequenceVorticityDataset(Dataset):
    """x: [T_in,H,W], y: [T_out,H,W] from a tensor [N,T,H,W]."""

    def __init__(self, data: torch.Tensor, t_in: int, t_out: int = 1):
        if data.dim() != 4:
            raise ValueError(f"expect [N,T,H,W], got {tuple(data.shape)}")
        self.data = data.contiguous()
        self.t_in = t_in
        self.t_out = t_out
        if self.data.shape[1] < t_in + t_out:
            raise ValueError("not enough time steps in data")

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int):
        sample = self.data[index]
        return sample[: self.t_in], sample[self.t_in : self.t_in + self.t_out]


def generate_ns_like_vorticity(
    n_samples: int = 512,
    resolution: int = 64,
    n_times: int = 20,
    viscosity: float = 1.0e-3,
    seed: int = 0,
    nonlinear_strength: float = 0.02,
) -> torch.Tensor:
    """Generate [N,T,H,W] vorticity with spectral viscosity + fixed forcing."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = torch.device("cpu")

    ky = torch.fft.fftfreq(resolution, d=1.0 / resolution, device=device)
    kx = torch.fft.rfftfreq(resolution, d=1.0 / resolution, device=device)
    kay, kax = torch.meshgrid(ky, kx, indexing="ij")
    k2 = kay**2 + kax**2
    k2 = torch.clamp(k2, min=0.0)
    k2_safe = k2.clone()
    k2_safe[0, 0] = 1.0

    grid = torch.linspace(0, 1, resolution, device=device)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    forcing = 0.1 * (torch.sin(2 * torch.pi * (xx + yy)) + torch.cos(2 * torch.pi * xx))

    noise = torch.randn(
        n_samples, resolution, resolution, generator=generator, device=device
    )
    noise_ft = torch.fft.rfft2(noise)
    spectrum = 1.0 / (1.0 + k2_safe) ** 1.5
    spectrum[0, 0] = 0.0
    w_ft = noise_ft * spectrum
    w = torch.fft.irfft2(w_ft, s=(resolution, resolution))

    dt = 1.0e-2
    frames = [w]
    for step in range(n_times - 1):
        w_ft = torch.fft.rfft2(w)
        damp = torch.exp(-viscosity * k2 * dt)
        w_ft = w_ft * damp
        w = torch.fft.irfft2(w_ft, s=(resolution, resolution))
        w = w + dt * forcing
        # slightly stronger structured nonlinearity for v2 sequences
        strength = nonlinear_strength * (1.0 + 0.15 * (step % 5))
        w = w + strength * dt * torch.sin(w) * torch.cos(0.5 * w)
        frames.append(w)

    data = torch.stack(frames, dim=1).contiguous()
    mean = data.mean(dim=(1, 2, 3), keepdim=True)
    std = data.std(dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-6)
    return ((data - mean) / std).to(torch.float32)


def load_or_build_ns_like(
    cache_path: Path | None = None,
    n_samples: int = 512,
    resolution: int = 64,
    n_times: int = 20,
    seed: int = 0,
    rebuild: bool = False,
    version: str = "v2",
) -> tuple[torch.Tensor, str]:
    """Return [N,T,H,W] and a short source tag.

    version=\"v2\" prefers stronger offline NS-like cache (more samples / longer T).
    """
    if cache_path is None:
        cache_path = V2_CACHE if version == "v2" else DEFAULT_CACHE

    for candidate in sorted(DATA_DIR.glob("*.pt")):
        if candidate.name.startswith("ns_like"):
            continue
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        tensor = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
        if tensor.dim() == 4 and tensor.shape[-1] != tensor.shape[-2]:
            if tensor.shape[1] == tensor.shape[2]:
                tensor = tensor.permute(0, 3, 1, 2).contiguous()
        return tensor.to(torch.float32), f"file:{candidate.name}"

    if cache_path.exists() and not rebuild:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            return payload["data"].to(torch.float32), payload.get("source", "cache:ns_like")
        return payload.to(torch.float32), "cache:ns_like"

    nonlinear = 0.035 if version == "v2" else 0.02
    source_tag = "generated_ns_like_v2" if version == "v2" else "generated_ns_like_v1e-3"
    data = generate_ns_like_vorticity(
        n_samples=n_samples,
        resolution=resolution,
        n_times=n_times,
        seed=seed,
        nonlinear_strength=nonlinear,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "data": data,
            "source": source_tag,
            "viscosity": 1.0e-3,
            "resolution": resolution,
            "n_times": n_times,
            "seed": seed,
            "version": version,
        },
        cache_path,
    )
    return data, source_tag


def split_train_test(
    data: torch.Tensor, n_train: int, n_test: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    n_total = data.shape[0]
    if n_train + n_test > n_total:
        raise ValueError(f"need {n_train + n_test} samples, have {n_total}")
    perm = torch.randperm(n_total, generator=generator)
    train_idx = perm[:n_train]
    test_idx = perm[n_train : n_train + n_test]
    return data[train_idx], data[test_idx]
