#!/usr/bin/env python3
"""Emit 128 fused inv with 8 height-n1 per block + four sequential irfft waves."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
src = (ROOT / "pruned_inv_fused128_n4.su").read_text()
src = src.replace(
    "pruned_irfft2_fused_mixed128_n4_kernel", "pruned_irfft2_fused_mixed128_n8_kernel"
)
src = src.replace(
    "launch_pruned_irfft2_fused_mixed128_n4", "launch_pruned_irfft2_fused_mixed128_n8"
)
src = src.replace("__shared__ float sr[32][16];", "__shared__ float sr[64][16];")
src = src.replace("__shared__ float si[32][16];", "__shared__ float si[64][16];")
src = src.replace("blockIdx.x) % 4", "blockIdx.x) % 2")
src = src.replace("blockIdx.x) / 4", "blockIdx.x) / 2")
src = src.replace("n1_pair * 4", "n1_pair * 8")
src = src.replace("if (tid < 64)", "if (tid < 128)")
src = src.replace("batch_size * channels * 4", "batch_size * channels * 2")
src = src.replace("for (int wave = 0; wave < 2; ++wave)", "for (int wave = 0; wave < 4; ++wave)")
out = ROOT / "pruned_inv_fused128_n8.su"
out.write_text(src)
print(f"wrote {out} bytes={out.stat().st_size}")
