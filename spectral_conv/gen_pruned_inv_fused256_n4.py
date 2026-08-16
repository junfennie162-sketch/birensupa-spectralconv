#!/usr/bin/env python3
"""Emit 256 fused inv with 4 height-n1 per block + two sequential irfft waves."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
src = (ROOT / "pruned_inv_fused256.su").read_text()
src = src.replace(
    "pruned_irfft2_fused_mixed256_kernel", "pruned_irfft2_fused_mixed256_n4_kernel"
)
src = src.replace(
    "launch_pruned_irfft2_fused_mixed256", "launch_pruned_irfft2_fused_mixed256_n4"
)
src = src.replace("__shared__ float sr[16][16];", "__shared__ float sr[32][16];")
src = src.replace("__shared__ float si[16][16];", "__shared__ float si[32][16];")
src = src.replace("blockIdx.x) % 16", "blockIdx.x) % 8")
src = src.replace("blockIdx.x) / 16", "blockIdx.x) / 8")
src = src.replace("n1_pair * 2", "n1_pair * 4")
src = src.replace("if (tid < 32)", "if (tid < 64)")
src = src.replace("batch_size * channels * 16", "batch_size * channels * 8")
old = """    __syncthreads();
    const int g = tid >> 4;
    const int n1_w = tid & 15;
    const int which_row = g >> 3;
    const int g8 = g & 7;
    const int h = n1_a + which_row + g8 * 32;
    const int out_base = ((b * channels + ch) * 256 + h) * 256;
"""
new = """    __syncthreads();
#pragma unroll 1
    for (int wave = 0; wave < 2; ++wave) {
    const int g = (tid >> 4) + wave * 16;
    const int n1_w = tid & 15;
    const int which_row = g >> 3;
    const int g8 = g & 7;
    const int h = n1_a + which_row + g8 * 32;
    const int out_base = ((b * channels + ch) * 256 + h) * 256;
"""
if old not in src:
    raise SystemExit("irfft header not found")
src = src.replace(old, new, 1)
src = src.replace(
    "    spatial[out_base + n1_w + 240] = (2.0f * b15r - r0) * invw;\n}",
    "    spatial[out_base + n1_w + 240] = (2.0f * b15r - r0) * invw;\n    }\n}",
    1,
)
out = ROOT / "pruned_inv_fused256_n4.su"
out.write_text(src)
print(f"wrote {out} bytes={out.stat().st_size}")
