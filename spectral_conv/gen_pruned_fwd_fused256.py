#!/usr/bin/env python3
"""Emit 256 fused forward: KEEP 8-row rfft tiles + register fft_h.

Does not store the 256-row plane in smem (that is a hard No-Go).
Smem = Z[8][16][16] + row8[8][16] = 17 KB.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "pruned_fft_h_fact256.su"
OUT = ROOT / "pruned_fwd_fused256.su"

head = SRC.read_text().split("__global__")[0]
if "static __device__ void dft16_minus" not in head:
    raise SystemExit("dft16_minus missing")

z_rfft = "\n".join(
    f"            Z[local_row][n1][{k}].x = u{k}r;\n"
    f"            Z[local_row][n1][{k}].y = u{k}i;"
    for k in range(16)
)
z_fft = "\n".join(
    f"        Z[m2_local][n1h][{k}].x = u{k}r;\n"
    f"        Z[m2_local][n1h][{k}].y = u{k}i;"
    for k in range(16)
)
loads = "\n".join(
    f"            u{k}r = x[x_base + n1 + {k * 16}];\n"
    f"            u{k}i = 0.0f;"
    for k in range(16)
)
u_args = ", ".join(f"u{k}r, u{k}i" for k in range(16))
u_decl = "\n".join(f"    float u{k}r = 0.0f, u{k}i = 0.0f;" for k in range(16))
v_decl = "\n".join(f"    float v{k}r = 0.0f, v{k}i = 0.0f;" for k in range(16))
v_from_u = "\n".join(f"    u{k}r = v{k}r;\n    u{k}i = v{k}i;" for k in range(16))
v_assign = "\n".join(
    f"            if (k == {k}) {{ v{k}r = row8[r][m2].x; v{k}i = row8[r][m2].y; }}"
    for k in range(16)
)

twiddle = """        float acc_r = 0.0f;
        float acc_i = 0.0f;
        float wr = 1.0f;
        float wi = 0.0f;
#pragma unroll
        for (int n1s = 0; n1s < 16; ++n1s) {
            const float zr = Z[m2_local][n1s][bin].x;
            const float zi = Z[m2_local][n1s][bin].y;
            acc_r += zr * wr - zi * wi;
            acc_i += zr * wi + zi * wr;
            const float nr = wr * step_c - wi * step_s;
            const float ni = wr * step_s + wi * step_c;
            wr = nr;
            wi = ni;
        }"""

body = f"""
__global__ void pruned_rfft2_fused_mixed256_kernel(int batch_size, int channels,
                       const float *__restrict__ x, float *__restrict__ packed) {{
    __shared__ float2 Z[8][16][16];
    __shared__ float2 row8[8][16];
    const int tid = static_cast<int>(threadIdx.x);
    const int n1 = tid & 15;
    const int local_row = tid >> 4;
    const int plane_idx = static_cast<int>(blockIdx.x);
    const int ch = plane_idx % channels;
    const int b = plane_idx / channels;
    const int plane = b * channels + ch;
{u_decl}
{v_decl}

#pragma unroll 1
    for (int tile = 0; tile < 32; ++tile) {{
        if (tid < 128) {{
            const int h = tile * 8 + local_row;
            const int x_base = (plane * 256 + h) * 256;
{loads}
            dft16_minus({u_args});
{z_rfft}
        }}
        __syncthreads();
        if (tid < 128) {{
            const int m2 = n1;
            const int bin = m2 & 15;
            float acc_r = 0.0f;
            float acc_i = 0.0f;
            float step_s, step_c;
            sincosf(-(2.0f * M_PIf) * static_cast<float>(m2) / 256.0f, &step_s, &step_c);
            float wr = 1.0f;
            float wi = 0.0f;
#pragma unroll
            for (int n1s = 0; n1s < 16; ++n1s) {{
                const float zr = Z[local_row][n1s][bin].x;
                const float zi = Z[local_row][n1s][bin].y;
                acc_r += zr * wr - zi * wi;
                acc_i += zr * wi + zi * wr;
                const float nr = wr * step_c - wi * step_s;
                const float ni = wr * step_s + wi * step_c;
                wr = nr;
                wi = ni;
            }}
            row8[local_row][m2].x = acc_r;
            row8[local_row][m2].y = acc_i;
        }}
        __syncthreads();
        {{
            const int n1h = tid >> 4;
            const int m2 = tid & 15;
            const int h0 = tile * 8;
#pragma unroll
            for (int r = 0; r < 8; ++r) {{
                const int h = h0 + r;
                if ((h & 15) == n1h) {{
                    const int k = h >> 4;
{v_assign}
                }}
            }}
        }}
        __syncthreads();
    }}

#if 0
    {{
        const int n1h = tid >> 4;
        const int m2 = tid & 15;
        packed[(plane * 256 + (0 * 16 + n1h)) * 32 + m2 * 2] = v0r;
        packed[(plane * 256 + (0 * 16 + n1h)) * 32 + m2 * 2 + 1] = v0i;
        packed[(plane * 256 + (1 * 16 + n1h)) * 32 + m2 * 2] = v1r;
        packed[(plane * 256 + (1 * 16 + n1h)) * 32 + m2 * 2 + 1] = v1i;
        packed[(plane * 256 + (2 * 16 + n1h)) * 32 + m2 * 2] = v2r;
        packed[(plane * 256 + (2 * 16 + n1h)) * 32 + m2 * 2 + 1] = v2i;
        packed[(plane * 256 + (3 * 16 + n1h)) * 32 + m2 * 2] = v3r;
        packed[(plane * 256 + (3 * 16 + n1h)) * 32 + m2 * 2 + 1] = v3i;
        packed[(plane * 256 + (4 * 16 + n1h)) * 32 + m2 * 2] = v4r;
        packed[(plane * 256 + (4 * 16 + n1h)) * 32 + m2 * 2 + 1] = v4i;
        packed[(plane * 256 + (5 * 16 + n1h)) * 32 + m2 * 2] = v5r;
        packed[(plane * 256 + (5 * 16 + n1h)) * 32 + m2 * 2 + 1] = v5i;
        packed[(plane * 256 + (6 * 16 + n1h)) * 32 + m2 * 2] = v6r;
        packed[(plane * 256 + (6 * 16 + n1h)) * 32 + m2 * 2 + 1] = v6i;
        packed[(plane * 256 + (7 * 16 + n1h)) * 32 + m2 * 2] = v7r;
        packed[(plane * 256 + (7 * 16 + n1h)) * 32 + m2 * 2 + 1] = v7i;
        packed[(plane * 256 + (8 * 16 + n1h)) * 32 + m2 * 2] = v8r;
        packed[(plane * 256 + (8 * 16 + n1h)) * 32 + m2 * 2 + 1] = v8i;
        packed[(plane * 256 + (9 * 16 + n1h)) * 32 + m2 * 2] = v9r;
        packed[(plane * 256 + (9 * 16 + n1h)) * 32 + m2 * 2 + 1] = v9i;
        packed[(plane * 256 + (10 * 16 + n1h)) * 32 + m2 * 2] = v10r;
        packed[(plane * 256 + (10 * 16 + n1h)) * 32 + m2 * 2 + 1] = v10i;
        packed[(plane * 256 + (11 * 16 + n1h)) * 32 + m2 * 2] = v11r;
        packed[(plane * 256 + (11 * 16 + n1h)) * 32 + m2 * 2 + 1] = v11i;
        packed[(plane * 256 + (12 * 16 + n1h)) * 32 + m2 * 2] = v12r;
        packed[(plane * 256 + (12 * 16 + n1h)) * 32 + m2 * 2 + 1] = v12i;
        packed[(plane * 256 + (13 * 16 + n1h)) * 32 + m2 * 2] = v13r;
        packed[(plane * 256 + (13 * 16 + n1h)) * 32 + m2 * 2 + 1] = v13i;
        packed[(plane * 256 + (14 * 16 + n1h)) * 32 + m2 * 2] = v14r;
        packed[(plane * 256 + (14 * 16 + n1h)) * 32 + m2 * 2 + 1] = v14i;
        packed[(plane * 256 + (15 * 16 + n1h)) * 32 + m2 * 2] = v15r;
        packed[(plane * 256 + (15 * 16 + n1h)) * 32 + m2 * 2 + 1] = v15i;
        return;
    }}
#endif
{v_from_u}
    dft16_minus({u_args});
    const int n1h = tid >> 4;
    const int m2 = tid & 15;
    const int m2_local = m2 & 7;
    const int m1 = n1h;
    const int bin = m1 & 15;
    float step_s, step_c;

    if (m2 < 8) {{
{z_fft}
    }}
    __syncthreads();
    if (m2 < 8) {{
        sincosf(-(2.0f * M_PIf) * static_cast<float>(m1) / 256.0f, &step_s, &step_c);
        {{
{twiddle}
            const int out_idx = (plane * 256 + m1) * 32 + m2 * 2;
            packed[out_idx] = acc_r;
            packed[out_idx + 1] = acc_i;
        }}
        sincosf(-(2.0f * M_PIf) * static_cast<float>(240 + m1) / 256.0f, &step_s, &step_c);
        {{
{twiddle}
            const int out_idx = (plane * 256 + 240 + m1) * 32 + m2 * 2;
            packed[out_idx] = acc_r;
            packed[out_idx + 1] = acc_i;
        }}
    }}
    __syncthreads();
    if (m2 >= 8) {{
{z_fft}
    }}
    __syncthreads();
    if (m2 >= 8) {{
        sincosf(-(2.0f * M_PIf) * static_cast<float>(m1) / 256.0f, &step_s, &step_c);
        {{
{twiddle}
            const int out_idx = (plane * 256 + m1) * 32 + m2 * 2;
            packed[out_idx] = acc_r;
            packed[out_idx + 1] = acc_i;
        }}
        sincosf(-(2.0f * M_PIf) * static_cast<float>(240 + m1) / 256.0f, &step_s, &step_c);
        {{
{twiddle}
            const int out_idx = (plane * 256 + 240 + m1) * 32 + m2 * 2;
            packed[out_idx] = acc_r;
            packed[out_idx + 1] = acc_i;
        }}
    }}
}}

extern "C" suError_t launch_pruned_rfft2_fused_mixed256(const float *x, float *packed,
                                                       int batch_size, int channels,
                                                       suStream_t stream) {{
    dim3 block(256);
    dim3 grid(static_cast<unsigned>(batch_size * channels));
    pruned_rfft2_fused_mixed256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, x, packed);
    return suGetLastError();
}}
"""

OUT.write_text(head + body)
print(f"wrote {OUT} bytes={OUT.stat().st_size}")
