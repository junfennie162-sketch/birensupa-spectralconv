#!/usr/bin/env python3
"""Emit 16-row/256-thread fused inverse for 64 (ifft 16x4 + irfft 16x4)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pruned_inv_fused64.su"


def extract_between(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


def transform_ifft_body(body: str) -> str:
    for group, off in enumerate(range(0, 64, 16)):
        marker = (
            f"        const int h = n1 + {off};\n"
            "        const int out = plane + (h * 16 + m2) * 2;\n"
        )
        if marker not in body:
            raise SystemExit(f"missing store marker for h=n1+{off}")
        k = body.index(marker)
        rest = body[k + len(marker) :]
        line1, _, rest2 = rest.partition("\n")
        line2, _, _ = rest2.partition("\n")
        re_expr = line1.split(" = ", 1)[1]
        im_expr = line2.split(" = ", 1)[1]
        new = (
            f"        sr[which * 4 + {group}][m2] = {re_expr}\n"
            f"        si[which * 4 + {group}][m2] = {im_expr}\n"
        )
        body = body[:k] + new + body[k + len(marker) + len(line1) + 1 + len(line2) + 1 :]
    if "row_freq" in body:
        raise SystemExit("ifft body still references row_freq")
    return body


def main() -> None:
    ifft_src = (ROOT / "pruned_ifft_h_fact.su").read_text()
    irfft_src = (ROOT / "pruned_irfft_w64.su").read_text()
    header = extract_between(ifft_src, "#include <supa.h>", "static __device__ void dft4_plus")
    dft4 = extract_between(ifft_src, "static __device__ void dft4_plus", "static __device__ void dft8_plus")

    kname = "__global__ void pruned_ifft_h_m16_fact16x4_h64_kernel"
    chunk = ifft_src[ifft_src.index(kname) :]
    kstart = chunk.index("    float u0r = 0.0f;")
    kend = chunk.index("\n}\n", kstart)
    ifft_body = transform_ifft_body(chunk[kstart:kend])
    ifft_indented = "\n".join(
        ("    " + line if line.strip() else line) for line in ifft_body.splitlines()
    )

    loads = ["    const float r0 = sr[g][0];"]
    for k in range(1, 16):
        loads.append(f"    const float r{k} = sr[g][{k}];")
        loads.append(f"    const float i{k} = si[g][{k}];")
    istart = irfft_src.index("    float b0r = 0.0f;")
    iend = irfft_src.index("    const float invw = 1.0f / 64.0f;")
    # first kernel only
    irfft_body = irfft_src[istart:iend]
    irfft_body = irfft_body.replace("static_cast<float>(n1)", "static_cast<float>(n1_w)")
    stores = "    const float invw = 1.0f / 64.0f;\n" + "\n".join(
        f"    spatial[out_base + n1_w + {k * 16}] = (2.0f * b{k}r - r0) * invw;"
        for k in range(4)
    )

    text = f"""{header}{dft4}
__global__ void pruned_irfft2_fused_mixed64_kernel(int batch_size, int channels,
                       const float *__restrict__ in_freq, float *__restrict__ spatial) {{
    __shared__ float sr[16][16];
    __shared__ float si[16][16];
    const int n1_pair = static_cast<int>(blockIdx.x) % 4;
    int rest = static_cast<int>(blockIdx.x) / 4;
    const int ch = rest % channels;
    const int b = rest / channels;
    const int n1_a = n1_pair * 4;
    const int tid = static_cast<int>(threadIdx.x);
    const int plane = ((b * channels + ch) * 64) * 32;
    if (tid < 64) {{
        const int which = tid >> 4;
        const int m2 = tid & 15;
        const int n1 = n1_a + which;
{ifft_indented}
    }}
    __syncthreads();
    const int g = tid >> 4;
    const int n1_w = tid & 15;
    const int h = n1_a + (g >> 2) + (g & 3) * 16;
    const int out_base = ((b * channels + ch) * 64 + h) * 64;
{chr(10).join(loads)}
{irfft_body}
{stores}
}}

extern "C" suError_t launch_pruned_irfft2_fused_mixed64(const float *in_freq, float *spatial,
                                                       int batch_size, int channels,
                                                       suStream_t stream) {{
    dim3 block(256);
    dim3 grid(static_cast<unsigned>(batch_size * channels * 4));
    pruned_irfft2_fused_mixed64_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, in_freq, spatial);
    return suGetLastError();
}}
"""
    OUT.write_text(text)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
