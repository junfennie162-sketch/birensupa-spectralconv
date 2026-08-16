#!/usr/bin/env python3
"""Emit 64 n8 and 128 n4 fused-inverse variants (two sequential irfft waves)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def emit_128_n4() -> None:
    src = (ROOT / "pruned_inv_fused128.su").read_text()
    src = src.replace(
        "pruned_irfft2_fused_mixed128_kernel", "pruned_irfft2_fused_mixed128_n4_kernel"
    )
    src = src.replace(
        "launch_pruned_irfft2_fused_mixed128", "launch_pruned_irfft2_fused_mixed128_n4"
    )
    src = src.replace("__shared__ float sr[16][16];", "__shared__ float sr[32][16];")
    src = src.replace("__shared__ float si[16][16];", "__shared__ float si[32][16];")
    src = src.replace("blockIdx.x) % 8", "blockIdx.x) % 4")
    src = src.replace("blockIdx.x) / 8", "blockIdx.x) / 4")
    src = src.replace("n1_pair * 2", "n1_pair * 4")
    src = src.replace("if (tid < 32)", "if (tid < 64)")
    src = src.replace("batch_size * channels * 8", "batch_size * channels * 4")
    old = """    __syncthreads();
    const int g = tid >> 4;
    const int n1_w = tid & 15;
    const int which_row = g >> 3;
    const int g8 = g & 7;
    const int h = n1_a + which_row + g8 * 16;
    const int out_base = ((b * channels + ch) * 128 + h) * 128;
"""
    new = """    __syncthreads();
#pragma unroll 1
    for (int wave = 0; wave < 2; ++wave) {
    const int g = (tid >> 4) + wave * 16;
    const int n1_w = tid & 15;
    const int which_row = g >> 3;
    const int g8 = g & 7;
    const int h = n1_a + which_row + g8 * 16;
    const int out_base = ((b * channels + ch) * 128 + h) * 128;
"""
    if old not in src:
        raise SystemExit("128 irfft header not found")
    src = src.replace(old, new, 1)
    src = src.replace(
        "    spatial[out_base + n1_w + 112] = (2.0f * b7r - r0) * invw;\n}",
        "    spatial[out_base + n1_w + 112] = (2.0f * b7r - r0) * invw;\n    }\n}",
        1,
    )
    out = ROOT / "pruned_inv_fused128_n4.su"
    out.write_text(src)
    print(f"wrote {out} bytes={out.stat().st_size}")


def emit_64_n8() -> None:
    src = (ROOT / "pruned_inv_fused64.su").read_text()
    src = src.replace(
        "pruned_irfft2_fused_mixed64_kernel", "pruned_irfft2_fused_mixed64_n8_kernel"
    )
    src = src.replace(
        "launch_pruned_irfft2_fused_mixed64", "launch_pruned_irfft2_fused_mixed64_n8"
    )
    src = src.replace("__shared__ float sr[16][16];", "__shared__ float sr[32][16];")
    src = src.replace("__shared__ float si[16][16];", "__shared__ float si[32][16];")
    src = src.replace("blockIdx.x) % 4", "blockIdx.x) % 2")
    src = src.replace("blockIdx.x) / 4", "blockIdx.x) / 2")
    src = src.replace("n1_pair * 4", "n1_pair * 8")
    src = src.replace("if (tid < 64)", "if (tid < 128)")
    src = src.replace("batch_size * channels * 4", "batch_size * channels * 2")
    old = """    __syncthreads();
    const int g = tid >> 4;
    const int n1_w = tid & 15;
    const int h = n1_a + (g >> 2) + (g & 3) * 16;
    const int out_base = ((b * channels + ch) * 64 + h) * 64;
"""
    new = """    __syncthreads();
#pragma unroll 1
    for (int wave = 0; wave < 2; ++wave) {
    const int g = (tid >> 4) + wave * 16;
    const int n1_w = tid & 15;
    const int h = n1_a + (g >> 2) + (g & 3) * 16;
    const int out_base = ((b * channels + ch) * 64 + h) * 64;
"""
    if old not in src:
        raise SystemExit("64 irfft header not found")
    src = src.replace(old, new, 1)
    src = src.replace(
        "    spatial[out_base + n1_w + 48] = (2.0f * b3r - r0) * invw;\n}",
        "    spatial[out_base + n1_w + 48] = (2.0f * b3r - r0) * invw;\n    }\n}",
        1,
    )
    out = ROOT / "pruned_inv_fused64_n8.su"
    out.write_text(src)
    print(f"wrote {out} bytes={out.stat().st_size}")


if __name__ == "__main__":
    emit_128_n4()
    emit_64_n8()
