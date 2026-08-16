#!/usr/bin/env python3
"""Emit smem-plane fused inverse for 256.

16 rows / 256 threads: two height-n1 groups per block so the irfft phase
matches KEEP occupancy (block=128 was slightly slower).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pruned_inv_fused256.su"


def extract_between(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


def transform_ifft_body(body: str) -> str:
    for group, off in enumerate(range(0, 256, 32)):
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
            f"        sr[which * 8 + {group}][m2] = {re_expr}\n"
            f"        si[which * 8 + {group}][m2] = {im_expr}\n"
        )
        body = body[:k] + new + body[k + len(marker) + len(line1) + 1 + len(line2) + 1 :]
    if "row_freq" in body:
        raise SystemExit("ifft body still references row_freq")
    return body


def irfft_from_smem() -> str:
    loads = ["    const float r0 = sr[g][0];"]
    for k in range(1, 16):
        loads.append(f"    const float r{k} = sr[g][{k}];")
        loads.append(f"    const float i{k} = si[g][{k}];")
    acc_decl = "\n".join(
        f"    float b{k}r = 0.0f;\n    float b{k}i = 0.0f;" for k in range(16)
    )
    twiddles = [
        "    b0r = r0;",
        "    float step_s, step_c;",
        "    sincosf((2.0f * M_PIf) * static_cast<float>(n1_w) / 256.0f, &step_s, &step_c);",
        "    float wr = step_c;",
        "    float wi = step_s;",
    ]
    for k in range(1, 16):
        twiddles.append(f"    b{k}r += r{k} * wr - i{k} * wi;")
        twiddles.append(f"    b{k}i += r{k} * wi + i{k} * wr;")
        if k < 15:
            twiddles.extend(
                [
                    "    {",
                    "        const float nr = wr * step_c - wi * step_s;",
                    "        const float ni = wr * step_s + wi * step_c;",
                    "        wr = nr;",
                    "        wi = ni;",
                    "    }",
                ]
            )
    stores = ["    const float invw = 1.0f / 256.0f;"]
    for k in range(16):
        stores.append(
            f"    spatial[out_base + n1_w + {k * 16}] = (2.0f * b{k}r - r0) * invw;"
        )
    return "\n".join(
        loads
        + [acc_decl]
        + twiddles
        + [
            "    dft16_plus(b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i, b4r, b4i, b5r, b5i, b6r, b6i, b7r, b7i, b8r, b8i, b9r, b9i, b10r, b10i, b11r, b11i, b12r, b12i, b13r, b13i, b14r, b14i, b15r, b15i);"
        ]
        + stores
    )


def main() -> None:
    ifft_src = (ROOT / "pruned_ifft_h_fact256.su").read_text()
    irfft_src = (ROOT / "pruned_irfft_w256_pair.su").read_text()

    dft8 = extract_between(ifft_src, "static __device__ void dft8_plus", "\n__global__ void")
    dft16 = extract_between(irfft_src, "static __device__ void dft16_plus", "\n__global__ void")
    header = extract_between(ifft_src, "#include <supa.h>", "static __device__ void dft8_plus")

    kstart = ifft_src.index("    float u0r = 0.0f;")
    kend = ifft_src.index("\n}\n", kstart)
    ifft_body = transform_ifft_body(ifft_src[kstart:kend])
    ifft_indented = "\n".join(
        ("    " + line if line.strip() else line) for line in ifft_body.splitlines()
    )

    text = f"""{header}{dft8}
{dft16}
__global__ void pruned_irfft2_fused_mixed256_kernel(int batch_size, int channels,
                       const float *__restrict__ in_freq, float *__restrict__ spatial) {{
    __shared__ float sr[16][16];
    __shared__ float si[16][16];
    const int n1_pair = static_cast<int>(blockIdx.x) % 16;
    int rest = static_cast<int>(blockIdx.x) / 16;
    const int ch = rest % channels;
    const int b = rest / channels;
    const int n1_a = n1_pair * 2;
    const int tid = static_cast<int>(threadIdx.x);
    const int plane = ((b * channels + ch) * 256) * 32;
    if (tid < 32) {{
        const int which = tid >> 4;
        const int m2 = tid & 15;
        const int n1 = n1_a + which;
{ifft_indented}
    }}
    __syncthreads();
    const int g = tid >> 4;
    const int n1_w = tid & 15;
    const int which_row = g >> 3;
    const int g8 = g & 7;
    const int h = n1_a + which_row + g8 * 32;
    const int out_base = ((b * channels + ch) * 256 + h) * 256;
{irfft_from_smem()}
}}

extern "C" suError_t launch_pruned_irfft2_fused_mixed256(const float *in_freq, float *spatial,
                                                        int batch_size, int channels,
                                                        suStream_t stream) {{
    dim3 block(256);
    dim3 grid(static_cast<unsigned>(batch_size * channels * 16));
    pruned_irfft2_fused_mixed256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, in_freq, spatial);
    return suGetLastError();
}}
"""
    OUT.write_text(text)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {text.count(chr(10))+1} lines)")


if __name__ == "__main__":
    main()
