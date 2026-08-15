#!/usr/bin/env python3
"""Generate register-unrolled packed FFT + 1-row fused inverse (not loc[] serial FFT)."""

from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).with_name("pruned_geo.su")
OUT_W64 = Path(__file__).with_name("pruned_irfft_w64.su")
OUT_HFACT = Path(__file__).with_name("pruned_ifft_h_fact.su")
OUT_RFACT = Path(__file__).with_name("pruned_rfft_w_fact.su")
OUT_FFTH = Path(__file__).with_name("pruned_fft_h_fact.su")
OUT_FFTH256 = Path(__file__).with_name("pruned_fft_h_fact256.su")
OUT_HFACT256 = Path(__file__).with_name("pruned_ifft_h_fact256.su")
OUT_IRFFT256_SMEM = Path(__file__).with_name("pruned_irfft_w256_smem.su")
OUT_FACT64 = Path(__file__).with_name("pruned_fwd_fact64.su")
OUT_IRFFT256_PAIR = Path(__file__).with_name("pruned_irfft_w256_pair.su")
OUT_IRFFT128_VEC4 = Path(__file__).with_name("pruned_irfft_w128_vec4.su")
OUT_IRFFT64_VEC4 = Path(__file__).with_name("pruned_irfft_w64_vec4.su")
OUT_HFACT256_64 = Path(__file__).with_name("pruned_ifft_h_fact256_64x4.su")
OUT_RFACT256_R4 = Path(__file__).with_name("pruned_rfft_w256_r4.su")

SU_HEADER = """#include <supa.h>
#include <supa_device.h>
#include <math.h>

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

#define BF(ar, ai, br, bi, cs, sn) \\
    do { \\
        const float _tr = (br) * (cs) - (bi) * (sn); \\
        const float _ti = (br) * (sn) + (bi) * (cs); \\
        const float _ar = (ar); \\
        const float _ai = (ai); \\
        (br) = _ar - _tr; \\
        (bi) = _ai - _ti; \\
        (ar) = _ar + _tr; \\
        (ai) = _ai + _ti; \\
    } while (0)
"""


def bitrev(x: int, bits: int) -> int:
    y = 0
    for _ in range(bits):
        y = (y << 1) | (x & 1)
        x >>= 1
    return y


def fmt_f(v: float) -> str:
    if abs(v) < 1e-12:
        return "0.0f"
    if abs(v - 1.0) < 1e-12:
        return "1.0f"
    if abs(v + 1.0) < 1e-12:
        return "-1.0f"
    return f"{v:.9f}f"


def emit_cs_lut(n: int) -> str:
    rows = []
    for w in range(n):
        ang = 2.0 * math.pi * w / n
        comma = "," if w + 1 < n else ""
        rows.append(f"    {{{fmt_f(math.cos(ang))}, {fmt_f(math.sin(ang))}}}{comma}")
    return (
        f"__device__ __constant__ float pruned_cs_lut_{n}[{n}][2] = {{\n"
        + "\n".join(rows)
        + "\n};\n"
    )


def emit_fft32() -> str:
    lines = [
        "static __device__ void fft32_regs(",
        "    float &r0, float &i0, float &r1, float &i1, float &r2, float &i2, float &r3, float &i3,",
        "    float &r4, float &i4, float &r5, float &i5, float &r6, float &i6, float &r7, float &i7,",
        "    float &r8, float &i8, float &r9, float &i9, float &r10, float &i10, float &r11, float &i11,",
        "    float &r12, float &i12, float &r13, float &i13, float &r14, float &i14, float &r15, float &i15,",
        "    float &r16, float &i16, float &r17, float &i17, float &r18, float &i18, float &r19, float &i19,",
        "    float &r20, float &i20, float &r21, float &i21, float &r22, float &i22, float &r23, float &i23,",
        "    float &r24, float &i24, float &r25, float &i25, float &r26, float &i26, float &r27, float &i27,",
        "    float &r28, float &i28, float &r29, float &i29, float &r30, float &i30, float &r31, float &i31) {",
    ]
    n = 32
    logn = 5
    for stage in range(1, logn + 1):
        length = 1 << stage
        half = length >> 1
        lines.append(f"    // stage {stage} len={length}")
        for group in range(0, n, length):
            for j in range(half):
                i0 = group + j
                i1 = i0 + half
                ang = -2.0 * math.pi * j / length
                cs = math.cos(ang)
                sn = math.sin(ang)
                lines.append(
                    f"    BF(r{i0}, i{i0}, r{i1}, i{i1}, {fmt_f(cs)}, {fmt_f(sn)});"
                )
    lines.append("}")
    return "\n".join(lines)


def emit_pack_store_k(k: int) -> str:
    if k == 0:
        return "    out[0] = r0 + i0;\n    out[1] = 0.0f;"
    nk = 32 - k
    ang = -2.0 * math.pi * k / 64.0
    cs = math.cos(ang)
    sn = math.sin(ang)
    return f"""    {{
        const float zr = r{k};
        const float zi = i{k};
        const float znr = r{nk};
        const float zni = i{nk};
        const float hr = 0.5f * (zr + znr);
        const float hi = 0.5f * (zi - zni);
        const float gr = 0.5f * (zi + zni);
        const float gi = 0.5f * (znr - zr);
        out[{k * 2}] = hr + gr * {fmt_f(cs)} - gi * {fmt_f(sn)};
        out[{k * 2 + 1}] = hi + gr * {fmt_f(sn)} + gi * {fmt_f(cs)};
    }}"""


def emit_warp_pack32() -> str:
    return r"""
static __device__ int bitrev5(int x) {
    int y = 0;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    return y;
}

// One row per warp. Butterflies go through smem — Biren shfl_xor was numerically wrong
// (CPU sim of the same DIT matches numpy; GPU shuffle path had rel~5).
__global__ void pruned_rfft_w_pack32_warp_kernel(int batch_size, int channels, int height,
                                                 int modes2, const float *__restrict__ x,
                                                 float *__restrict__ row_freq) {
    __shared__ float sr[8][32];
    __shared__ float si[8][32];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x * (blockDim.x >> 5) + warp);
    const int total_rows = batch_size * channels * height;
    const int valid = row < total_rows;
    int b = 0;
    int ch = 0;
    int h = 0;
    if (valid) {
        h = row % height;
        int rem = row / height;
        ch = rem % channels;
        b = rem / channels;
        const int x_base = ((b * channels + ch) * height + h) * 64;
        const int dst = bitrev5(lane);
        sr[warp][dst] = x[x_base + lane * 2];
        si[warp][dst] = x[x_base + lane * 2 + 1];
    } else {
        sr[warp][lane] = 0.0f;
        si[warp][lane] = 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int stage = 1; stage <= 5; ++stage) {
        const int length = 1 << stage;
        const int half = length >> 1;
        const int j = lane & (half - 1);
        const int i0 = (lane & ~(length - 1)) | j;
        const int i1 = i0 + half;
        const float ar = sr[warp][i0];
        const float ai = si[warp][i0];
        const float br = sr[warp][i1];
        const float bi = si[warp][i1];
        float sn, cs;
        sincosf(-(2.0f * M_PIf) * static_cast<float>(j) / static_cast<float>(length), &sn, &cs);
        const float tr = br * cs - bi * sn;
        const float ti = br * sn + bi * cs;
        __syncthreads();
        if ((lane & half) == 0) {
            sr[warp][lane] = ar + tr;
            si[warp][lane] = ai + ti;
        } else {
            sr[warp][lane] = ar - tr;
            si[warp][lane] = ai - ti;
        }
        __syncthreads();
    }
    if (!valid || lane >= modes2) {
        return;
    }
    float *out = row_freq + ((b * channels + ch) * height + h) * modes2 * 2;
    if (lane == 0) {
        out[0] = sr[warp][0] + si[warp][0];
        out[1] = 0.0f;
        return;
    }
    const float zr = sr[warp][lane];
    const float zi = si[warp][lane];
    const float znr = sr[warp][32 - lane];
    const float zni = si[warp][32 - lane];
    const float hr = 0.5f * (zr + znr);
    const float hi = 0.5f * (zi - zni);
    const float gr = 0.5f * (zi + zni);
    const float gi = 0.5f * (znr - zr);
    float sn, cs;
    sincosf(-(2.0f * M_PIf) * static_cast<float>(lane) / 64.0f, &sn, &cs);
    out[lane * 2] = hr + gr * cs - gi * sn;
    out[lane * 2 + 1] = hi + gr * sn + gi * cs;
}
"""


def emit_pack64_kernel() -> str:
    loads = []
    for j in range(32):
        dst = bitrev(j, 5)
        loads.append(f"    float r{dst} = x[x_base + {2 * j}];")
        loads.append(f"    float i{dst} = x[x_base + {2 * j + 1}];")
    stores = [emit_pack_store_k(k) for k in range(16)]
    args = ", ".join(f"r{i}, i{i}" for i in range(32))
    return f"""
__global__ void pruned_rfft_w_pack32_row_kernel(int batch_size, int channels, int height,
                                                int modes2, const float *__restrict__ x,
                                                float *__restrict__ row_freq) {{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total_rows = batch_size * channels * height;
    if (row >= total_rows) {{
        return;
    }}
    const int h = row % height;
    int rem = row / height;
    const int ch = rem % channels;
    const int b = rem / channels;
    const int x_base = ((b * channels + ch) * height + h) * 64;
{chr(10).join(loads)}
    fft32_regs({args});
    float *out = row_freq + ((b * channels + ch) * height + h) * modes2 * 2;
{chr(10).join(stores)}
}}
"""


def emit_coop_rfft(width: int) -> str:
    pad = width + 4
    name = f"pruned_rfft_w_m16_w{width}_coop_kernel"
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ x, float *__restrict__ row_freq) {{
    __shared__ float xs[16][{pad}];
    const int local_row = static_cast<int>(threadIdx.x) >> 4;
    const int m2 = static_cast<int>(threadIdx.x) & 15;
    const int row = static_cast<int>(blockIdx.x) * 16 + local_row;
    const int total_rows = batch_size * channels * height;
    int b = 0;
    int ch = 0;
    int h = 0;
    if (row < total_rows) {{
        h = row % height;
        int rem = row / height;
        ch = rem % channels;
        b = rem / channels;
        const int x_base = ((b * channels + ch) * height + h) * {width};
#pragma unroll 4
        for (int i = m2; i < {width}; i += 16) {{
            xs[local_row][i] = x[x_base + i];
        }}
    }}
    __syncthreads();
    if (row >= total_rows) {{
        return;
    }}
    const float two_pi_k = -(2.0f * M_PIf) * static_cast<float>(m2) / static_cast<float>({width});
    float step_s, step_c;
    sincosf(two_pi_k, &step_s, &step_c);
    float acc_r = 0.0f;
    float acc_i = 0.0f;
    float wr = 1.0f;
    float wi = 0.0f;
#pragma unroll 8
    for (int w = 0; w < {width}; ++w) {{
        const float xv = xs[local_row][w];
        acc_r += xv * wr;
        acc_i += xv * wi;
        const float nr = wr * step_c - wi * step_s;
        const float ni = wr * step_s + wi * step_c;
        wr = nr;
        wi = ni;
    }}
    const int out = ((b * channels + ch) * height + h) * 32 + m2 * 2;
    row_freq[out] = acc_r;
    row_freq[out + 1] = acc_i;
}}
"""


def emit_coop_fft_h(height: int) -> str:
    pad = height + 1
    name = f"pruned_fft_h_m16_n{height}_coop_kernel"
    return f"""
__global__ void {name}(int batch_size, int channels,
                       const float *__restrict__ row_freq, float *__restrict__ packed) {{
    __shared__ float cr[8][{pad}];
    __shared__ float ci[8][{pad}];
    const int local_col = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int total_cols = batch_size * channels * 16;
    const int col = static_cast<int>(blockIdx.x) * 8 + local_col;
    int b = 0;
    int ch = 0;
    int m2 = 0;
    if (col < total_cols) {{
        m2 = col & 15;
        int rem = col >> 4;
        ch = rem % channels;
        b = rem / channels;
        const int row_plane = (b * channels + ch) * {height} * 32;
#pragma unroll 4
        for (int hh = lane; hh < {height}; hh += 32) {{
            const int in = row_plane + (hh * 16 + m2) * 2;
            cr[local_col][hh] = row_freq[in];
            ci[local_col][hh] = row_freq[in + 1];
        }}
    }}
    __syncthreads();
    if (col >= total_cols) {{
        return;
    }}
    const int m1 = lane & 15;
    const int corner = lane >> 4;
    const int kh = (corner == 0) ? m1 : ({height} - 16 + m1);
    const float two_pi_k = -(2.0f * M_PIf) * static_cast<float>(kh) / static_cast<float>({height});
    float step_s, step_c;
    sincosf(two_pi_k, &step_s, &step_c);
    float acc_r = 0.0f;
    float acc_i = 0.0f;
    float wr = 1.0f;
    float wi = 0.0f;
#pragma unroll 8
    for (int hh = 0; hh < {height}; ++hh) {{
        const float xr = cr[local_col][hh];
        const float xi = ci[local_col][hh];
        acc_r += xr * wr - xi * wi;
        acc_i += xr * wi + xi * wr;
        const float nr = wr * step_c - wi * step_s;
        const float ni = wr * step_s + wi * step_c;
        wr = nr;
        wi = ni;
    }}
    const int out = ((b * channels + ch) * {height} + kh) * 32 + m2 * 2;
    packed[out] = acc_r;
    packed[out + 1] = acc_i;
}}
"""


def emit_dft_rfft(width: int, vec2: bool = False) -> str:
    name = f"pruned_rfft_w_m16_w{width}_kernel"
    unroll = " 8"
    if vec2:
        body = f"""    const float2 *row = reinterpret_cast<const float2 *>(x + x_base);
    const float two_pi_k = -(2.0f * M_PIf) * static_cast<float>(m2) / static_cast<float>({width});
    float step_s, step_c;
    sincosf(two_pi_k, &step_s, &step_c);
    float acc_r = 0.0f;
    float acc_i = 0.0f;
    float wr = 1.0f;
    float wi = 0.0f;
#pragma unroll{unroll}
    for (int i = 0; i < {width // 2}; ++i) {{
        const float2 v = row[i];
        acc_r += v.x * wr;
        acc_i += v.x * wi;
        {{
            const float nr = wr * step_c - wi * step_s;
            const float ni = wr * step_s + wi * step_c;
            wr = nr;
            wi = ni;
        }}
        acc_r += v.y * wr;
        acc_i += v.y * wi;
        {{
            const float nr = wr * step_c - wi * step_s;
            const float ni = wr * step_s + wi * step_c;
            wr = nr;
            wi = ni;
        }}
    }}"""
    else:
        body = f"""    const float two_pi_k = -(2.0f * M_PIf) * static_cast<float>(m2) / static_cast<float>({width});
    float step_s, step_c;
    sincosf(two_pi_k, &step_s, &step_c);
    float acc_r = 0.0f;
    float acc_i = 0.0f;
    float wr = 1.0f;
    float wi = 0.0f;
#pragma unroll{unroll}
    for (int w = 0; w < {width}; ++w) {{
        const float xv = x[x_base + w];
        acc_r += xv * wr;
        acc_i += xv * wi;
        const float nr = wr * step_c - wi * step_s;
        const float ni = wr * step_s + wi * step_c;
        wr = nr;
        wi = ni;
    }}"""
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ x, float *__restrict__ row_freq) {{
    const int total = batch_size * channels * height * 16;
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int m2 = index & 15;
    int rest = index >> 4;
    const int h = rest % height;
    rest /= height;
    const int ch = rest % channels;
    const int b = rest / channels;
    const int x_base = ((b * channels + ch) * height + h) * {width};
{body}
    const int out = ((b * channels + ch) * height + h) * 32 + m2 * 2;
    row_freq[out] = acc_r;
    row_freq[out + 1] = acc_i;
}}
"""


def emit_dft_rfft_row2(width: int, vec2: bool = False) -> str:
    name = f"pruned_rfft_w_m16_w{width}_row2_kernel"
    unroll = " 8"
    if vec2:
        inner = f"""    const float2 *row0 = reinterpret_cast<const float2 *>(x + x0);
    const float2 *row1 = reinterpret_cast<const float2 *>(x + x1);
#pragma unroll{unroll}
    for (int i = 0; i < {width // 2}; ++i) {{
        const float2 v0 = row0[i];
        const float2 v1 = row1[i];
        acc0r += v0.x * wr;
        acc0i += v0.x * wi;
        acc1r += v1.x * wr;
        acc1i += v1.x * wi;
        {{
            const float nr = wr * step_c - wi * step_s;
            const float ni = wr * step_s + wi * step_c;
            wr = nr;
            wi = ni;
        }}
        acc0r += v0.y * wr;
        acc0i += v0.y * wi;
        acc1r += v1.y * wr;
        acc1i += v1.y * wi;
        {{
            const float nr = wr * step_c - wi * step_s;
            const float ni = wr * step_s + wi * step_c;
            wr = nr;
            wi = ni;
        }}
    }}"""
    else:
        inner = f"""#pragma unroll{unroll}
    for (int w = 0; w < {width}; ++w) {{
        const float xv0 = x[x0 + w];
        const float xv1 = x[x1 + w];
        acc0r += xv0 * wr;
        acc0i += xv0 * wi;
        acc1r += xv1 * wr;
        acc1i += xv1 * wi;
        const float nr = wr * step_c - wi * step_s;
        const float ni = wr * step_s + wi * step_c;
        wr = nr;
        wi = ni;
    }}"""
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ x, float *__restrict__ row_freq) {{
    const int half_h = height >> 1;
    const int total = batch_size * channels * half_h * 16;
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int m2 = index & 15;
    int rest = index >> 4;
    const int hpair = rest % half_h;
    rest /= half_h;
    const int ch = rest % channels;
    const int b = rest / channels;
    const int h0 = hpair << 1;
    const int h1 = h0 + 1;
    const int plane = (b * channels + ch) * height;
    const int x0 = (plane + h0) * {width};
    const int x1 = (plane + h1) * {width};
    const float two_pi_k = -(2.0f * M_PIf) * static_cast<float>(m2) / static_cast<float>({width});
    float step_s, step_c;
    sincosf(two_pi_k, &step_s, &step_c);
    float acc0r = 0.0f;
    float acc0i = 0.0f;
    float acc1r = 0.0f;
    float acc1i = 0.0f;
    float wr = 1.0f;
    float wi = 0.0f;
{inner}
    const int out0 = (plane + h0) * 32 + m2 * 2;
    const int out1 = (plane + h1) * 32 + m2 * 2;
    row_freq[out0] = acc0r;
    row_freq[out0 + 1] = acc0i;
    row_freq[out1] = acc1r;
    row_freq[out1 + 1] = acc1i;
}}
"""


def emit_dft_rfft_dual(width: int) -> str:
    name = f"pruned_rfft_w_m16_w{width}_dual_kernel"
    unroll = " 8"
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ x, float *__restrict__ row_freq) {{
    const int total = batch_size * channels * height * 8;
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int pair = index & 7;
    int rest = index >> 3;
    const int h = rest % height;
    rest /= height;
    const int ch = rest % channels;
    const int b = rest / channels;
    const int k0 = pair << 1;
    const int k1 = k0 + 1;
    const int x_base = ((b * channels + ch) * height + h) * {width};
    const float scale = -(2.0f * M_PIf) / static_cast<float>({width});
    float step0s, step0c, step1s, step1c;
    sincosf(scale * static_cast<float>(k0), &step0s, &step0c);
    sincosf(scale * static_cast<float>(k1), &step1s, &step1c);
    float acc0r = 0.0f;
    float acc0i = 0.0f;
    float acc1r = 0.0f;
    float acc1i = 0.0f;
    float wr0 = 1.0f;
    float wi0 = 0.0f;
    float wr1 = 1.0f;
    float wi1 = 0.0f;
#pragma unroll{unroll}
    for (int w = 0; w < {width}; ++w) {{
        const float xv = x[x_base + w];
        acc0r += xv * wr0;
        acc0i += xv * wi0;
        acc1r += xv * wr1;
        acc1i += xv * wi1;
        const float n0r = wr0 * step0c - wi0 * step0s;
        const float n0i = wr0 * step0s + wi0 * step0c;
        wr0 = n0r;
        wi0 = n0i;
        const float n1r = wr1 * step1c - wi1 * step1s;
        const float n1i = wr1 * step1s + wi1 * step1c;
        wr1 = n1r;
        wi1 = n1i;
    }}
    const int out = ((b * channels + ch) * height + h) * 32;
    row_freq[out + k0 * 2] = acc0r;
    row_freq[out + k0 * 2 + 1] = acc0i;
    row_freq[out + k1 * 2] = acc1r;
    row_freq[out + k1 * 2 + 1] = acc1i;
}}
"""


def emit_dft_rfft_goertzel(width: int) -> str:
    name = f"pruned_rfft_w_m16_w{width}_gzt_kernel"
    half = width // 2
    unroll = " 8"
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ x, float *__restrict__ row_freq) {{
    const int total = batch_size * channels * height * 16;
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int m2 = index & 15;
    int rest = index >> 4;
    const int h = rest % height;
    rest /= height;
    const int ch = rest % channels;
    const int b = rest / channels;
    const int x_base = ((b * channels + ch) * height + h) * {width};
    const float2 *row = reinterpret_cast<const float2 *>(x + x_base);
    const int out = ((b * channels + ch) * height + h) * 32 + m2 * 2;
    if (m2 == 0) {{
        float acc = 0.0f;
#pragma unroll{unroll}
        for (int i = 0; i < {half}; ++i) {{
            const float2 v = row[i];
            acc += v.x + v.y;
        }}
        row_freq[out] = acc;
        row_freq[out + 1] = 0.0f;
        return;
    }}
    const float omega = (2.0f * M_PIf) * static_cast<float>(m2) / static_cast<float>({width});
    float sn, cs;
    sincosf(omega, &sn, &cs);
    const float coeff = 2.0f * cs;
    float s1 = 0.0f;
    float s2 = 0.0f;
#pragma unroll{unroll}
    for (int i = 0; i < {half}; ++i) {{
        const float2 v = row[i];
        float t = v.x + coeff * s1 - s2;
        s2 = s1;
        s1 = t;
        t = v.y + coeff * s1 - s2;
        s2 = s1;
        s1 = t;
    }}
    row_freq[out] = s1 * cs - s2;
    row_freq[out + 1] = s1 * sn;
}}
"""


def emit_dft_fft_h(height: int) -> str:
    name = f"pruned_fft_h_m16_n{height}_kernel"
    unroll = " 8"
    return f"""
__global__ void {name}(int batch_size, int channels,
                       const float *__restrict__ row_freq, float *__restrict__ packed) {{
    const int single = batch_size * channels * 16 * 16;
    const int total = single * 2;
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int corner = index / single;
    const int local = index - corner * single;
    const int m2 = local & 15;
    int rem = local >> 4;
    const int m1 = rem & 15;
    rem >>= 4;
    const int ch = rem % channels;
    const int b = rem / channels;
    const int kh = (corner == 0) ? m1 : ({height} - 16 + m1);
    const float two_pi_k = -(2.0f * M_PIf) * static_cast<float>(kh) / static_cast<float>({height});
    float step_s, step_c;
    sincosf(two_pi_k, &step_s, &step_c);
    float acc_r = 0.0f;
    float acc_i = 0.0f;
    float wr = 1.0f;
    float wi = 0.0f;
    const int row_plane = (b * channels + ch) * {height} * 32;
#pragma unroll{unroll}
    for (int hh = 0; hh < {height}; ++hh) {{
        const int in = row_plane + (hh * 16 + m2) * 2;
        const float xr = row_freq[in];
        const float xi = row_freq[in + 1];
        acc_r += xr * wr - xi * wi;
        acc_i += xr * wi + xi * wr;
        const float nr = wr * step_c - wi * step_s;
        const float ni = wr * step_s + wi * step_c;
        wr = nr;
        wi = ni;
    }}
    const int out = ((b * channels + ch) * {height} + kh) * 32 + m2 * 2;
    packed[out] = acc_r;
    packed[out + 1] = acc_i;
}}
"""


def emit_dft_fft_h_dual(height: int) -> str:
    name = f"pruned_fft_h_m16_n{height}_dual_kernel"
    unroll = " 8"
    return f"""
__global__ void {name}(int batch_size, int channels,
                       const float *__restrict__ row_freq, float *__restrict__ packed) {{
    const int total = batch_size * channels * 16 * 16;
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int m2 = index & 15;
    int rem = index >> 4;
    const int m1 = rem & 15;
    rem >>= 4;
    const int ch = rem % channels;
    const int b = rem / channels;
    const int kh_bot = {height} - 16 + m1;
    const float two_pi = (2.0f * M_PIf) / static_cast<float>({height});
    float step_ts, step_tc, step_bs, step_bc;
    sincosf(-two_pi * static_cast<float>(m1), &step_ts, &step_tc);
    sincosf(-two_pi * static_cast<float>(kh_bot), &step_bs, &step_bc);
    float acc_tr = 0.0f;
    float acc_ti = 0.0f;
    float acc_br = 0.0f;
    float acc_bi = 0.0f;
    float wr_t = 1.0f;
    float wi_t = 0.0f;
    float wr_b = 1.0f;
    float wi_b = 0.0f;
    const int row_plane = (b * channels + ch) * {height} * 32;
#pragma unroll{unroll}
    for (int hh = 0; hh < {height}; ++hh) {{
        const int in = row_plane + (hh * 16 + m2) * 2;
        const float xr = row_freq[in];
        const float xi = row_freq[in + 1];
        acc_tr += xr * wr_t - xi * wi_t;
        acc_ti += xr * wi_t + xi * wr_t;
        acc_br += xr * wr_b - xi * wi_b;
        acc_bi += xr * wi_b + xi * wr_b;
        const float ntr = wr_t * step_tc - wi_t * step_ts;
        const float nti = wr_t * step_ts + wi_t * step_tc;
        wr_t = ntr;
        wi_t = nti;
        const float nbr = wr_b * step_bc - wi_b * step_bs;
        const float nbi = wr_b * step_bs + wi_b * step_bc;
        wr_b = nbr;
        wi_b = nbi;
    }}
    const int top = ((b * channels + ch) * {height} + m1) * 32 + m2 * 2;
    packed[top] = acc_tr;
    packed[top + 1] = acc_ti;
    const int bot = ((b * channels + ch) * {height} + kh_bot) * 32 + m2 * 2;
    packed[bot] = acc_br;
    packed[bot + 1] = acc_bi;
}}
"""


def emit_dft_fft_h_dual_goertzel(height: int) -> str:
    name = f"pruned_fft_h_m16_n{height}_dual_gzt_kernel"
    unroll = " 8"
    return f"""
__global__ void {name}(int batch_size, int channels,
                       const float *__restrict__ row_freq, float *__restrict__ packed) {{
    const int total = batch_size * channels * 16 * 16;
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int m2 = index & 15;
    int rem = index >> 4;
    const int m1 = rem & 15;
    rem >>= 4;
    const int ch = rem % channels;
    const int b = rem / channels;
    const int kh_bot = {height} - 16 + m1;
    const int row_plane = (b * channels + ch) * {height} * 32;
    const int top = ((b * channels + ch) * {height} + m1) * 32 + m2 * 2;
    const int bot = ((b * channels + ch) * {height} + kh_bot) * 32 + m2 * 2;
    const float two_pi = (2.0f * M_PIf) / static_cast<float>({height});
    if (m1 == 0) {{
        float acc_tr = 0.0f;
        float acc_ti = 0.0f;
        float omega = two_pi * static_cast<float>(kh_bot);
        float sn, cs;
        sincosf(omega, &sn, &cs);
        const float coeff = 2.0f * cs;
        float sr1 = 0.0f, sr2 = 0.0f, si1 = 0.0f, si2 = 0.0f;
#pragma unroll{unroll}
        for (int hh = 0; hh < {height}; ++hh) {{
            const int in = row_plane + (hh * 16 + m2) * 2;
            const float2 xv = *reinterpret_cast<const float2 *>(row_freq + in);
            acc_tr += xv.x;
            acc_ti += xv.y;
            const float sr0 = xv.x + coeff * sr1 - sr2;
            const float si0 = xv.y + coeff * si1 - si2;
            sr2 = sr1;
            sr1 = sr0;
            si2 = si1;
            si1 = si0;
        }}
        packed[top] = acc_tr;
        packed[top + 1] = acc_ti;
        packed[bot] = sr1 * cs - sr2 - si1 * sn;
        packed[bot + 1] = sr1 * sn + si1 * cs - si2;
        return;
    }}
    float sn_t, cs_t, sn_b, cs_b;
    sincosf(two_pi * static_cast<float>(m1), &sn_t, &cs_t);
    sincosf(two_pi * static_cast<float>(kh_bot), &sn_b, &cs_b);
    const float coeff_t = 2.0f * cs_t;
    const float coeff_b = 2.0f * cs_b;
    float tr1 = 0.0f, tr2 = 0.0f, ti1 = 0.0f, ti2 = 0.0f;
    float br1 = 0.0f, br2 = 0.0f, bi1 = 0.0f, bi2 = 0.0f;
#pragma unroll{unroll}
    for (int hh = 0; hh < {height}; ++hh) {{
        const int in = row_plane + (hh * 16 + m2) * 2;
        const float2 xv = *reinterpret_cast<const float2 *>(row_freq + in);
        const float tr0 = xv.x + coeff_t * tr1 - tr2;
        const float ti0 = xv.y + coeff_t * ti1 - ti2;
        tr2 = tr1;
        tr1 = tr0;
        ti2 = ti1;
        ti1 = ti0;
        const float br0 = xv.x + coeff_b * br1 - br2;
        const float bi0 = xv.y + coeff_b * bi1 - bi2;
        br2 = br1;
        br1 = br0;
        bi2 = bi1;
        bi1 = bi0;
    }}
    packed[top] = tr1 * cs_t - tr2 - ti1 * sn_t;
    packed[top + 1] = tr1 * sn_t + ti1 * cs_t - ti2;
    packed[bot] = br1 * cs_b - br2 - bi1 * sn_b;
    packed[bot + 1] = br1 * sn_b + bi1 * cs_b - bi2;
}}
"""


def emit_irfft_w_xn(width: int, pix: int) -> str:
    pairs = width // pix
    shift = pairs.bit_length() - 1
    name = f"pruned_irfft_w_m16_x{pix}_w{width}_kernel"
    loads = [
        "    const float2 z0 = *reinterpret_cast<const float2 *>(row_freq + row_base);",
        "    const float r0 = z0.x;",
    ]
    for k in range(1, 16):
        loads.append(
            f"    const float2 z{k} = *reinterpret_cast<const float2 *>(row_freq + row_base + {2 * k});"
        )
        loads.append(f"    const float r{k} = z{k}.x;")
        loads.append(f"    const float i{k} = z{k}.y;")
    acc = [
        "        float acc = r0;",
        f"        const float two_pi_w = (2.0f * M_PIf) * static_cast<float>(w) / {width}.0f;",
        "        float step_s, step_c;",
        "        sincosf(two_pi_w, &step_s, &step_c);",
        "        float wr = step_c;",
        "        float wi = step_s;",
    ]
    for k in range(1, 16):
        acc.append(f"        acc += 2.0f * (r{k} * wr - i{k} * wi);")
        if k < 15:
            acc.extend(
                [
                    "        {",
                    "            const float nr = wr * step_c - wi * step_s;",
                    "            const float ni = wr * step_s + wi * step_c;",
                    "            wr = nr;",
                    "            wi = ni;",
                    "        }",
                ]
            )
    acc.append(f"        spatial[out_base + w] = acc / {width}.0f;")
    loads_s = "\n".join(loads)
    acc_s = "\n".join(acc)
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ row_freq, float *__restrict__ spatial) {{
    const int total = batch_size * channels * height * {pairs};
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int pair = index & {pairs - 1};
    int rest = index >> {shift};
    const int h = rest % height;
    rest /= height;
    const int ch = rest % channels;
    const int b = rest / channels;
    const int row_base = ((b * channels + ch) * height + h) * 32;
{loads_s}
    const int out_base = ((b * channels + ch) * height + h) * {width};
#pragma unroll
    for (int dw = 0; dw < {pix}; ++dw) {{
        const int w = pair * {pix} + dw;
{acc_s}
    }}
}}
"""


def emit_irfft_w_x2(width: int) -> str:
    return emit_irfft_w_xn(width, 2)


def emit_dft_plus_regs(n: int) -> str:
    return emit_dft_regs(n, sign=1)


def emit_dft_regs(n: int, sign: int = 1) -> str:
    """Unnormalized DFT. sign=+1 is +i (irfft); sign=-1 is -i (forward FFT)."""
    suffix = "plus" if sign > 0 else "minus"
    bits = n.bit_length() - 1
    args = ", ".join(f"float &r{i}, float &i{i}" for i in range(n))
    lines = [f"static __device__ void dft{n}_{suffix}({args}) {{"]
    for i in range(n):
        j = bitrev(i, bits)
        if i < j:
            lines.append(
                "    { "
                f"const float tr = r{i}; r{i} = r{j}; r{j} = tr; "
                f"const float ti = i{i}; i{i} = i{j}; i{j} = ti; "
                "}"
            )
    logn = bits
    for stage in range(1, logn + 1):
        length = 1 << stage
        half = length >> 1
        lines.append(f"    // stage {stage} len={length}")
        for group in range(0, n, length):
            for j in range(half):
                i0 = group + j
                i1 = i0 + half
                ang = sign * 2.0 * math.pi * j / length
                lines.append(
                    f"    BF(r{i0}, i{i0}, r{i1}, i{i1}, {fmt_f(math.cos(ang))}, {fmt_f(math.sin(ang))});"
                )
    lines.append("}")
    return "\n".join(lines)


def emit_rfft_w_fact_smem(width: int, rows_per_block: int = 8) -> str:
    """Mixed-radix real FFT: 16 n1 threads, N2-pt -i DFT, smem reduce to 16 bins."""
    n1_count = 16
    n2 = width // n1_count
    if n2 not in (4, 8, 16):
        raise ValueError(f"unsupported rfft fact n2={n2}")
    name = f"pruned_rfft_w_m16_fact16x{n2}_w{width}_kernel"
    if rows_per_block != 8:
        name = f"pruned_rfft_w_m16_fact16x{n2}_r{rows_per_block}_w{width}_kernel"
    loads = []
    for n2i in range(n2):
        loads.append(
            f"        u{n2i}r = x[x_base + n1 + {n1_count * n2i}];"
        )
        loads.append(f"        u{n2i}i = 0.0f;")
    udecl = "\n".join(f"    float u{m}r, u{m}i;" for m in range(n2))
    loads_s = "\n".join(loads)
    dft_args = ", ".join(f"u{m}r, u{m}i" for m in range(n2))
    stores_z = "\n".join(
        f"        Z[local_row][n1][{m}].x = u{m}r;\n"
        f"        Z[local_row][n1][{m}].y = u{m}i;"
        for m in range(n2)
    )
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ x, float *__restrict__ row_freq) {{
    __shared__ float2 Z[{rows_per_block}][16][{n2}];
    const int tid = static_cast<int>(threadIdx.x);
    const int n1 = tid & 15;
    const int local_row = tid >> 4;
    const int row0 = static_cast<int>(blockIdx.x) * {rows_per_block} + local_row;
    const int total_rows = batch_size * channels * height;
    const int valid = row0 < total_rows;
{udecl}
    int b = 0, ch = 0, h = 0, x_base = 0;
    if (valid) {{
        h = row0 % height;
        int rest = row0 / height;
        ch = rest % channels;
        b = rest / channels;
        x_base = ((b * channels + ch) * height + h) * {width};
{loads_s}
        dft{n2}_minus({dft_args});
{stores_z}
    }}
    __syncthreads();
    if (!valid) {{
        return;
    }}
    const int m2 = n1;
    const int bin = m2 & {n2 - 1};
    float acc_r = 0.0f;
    float acc_i = 0.0f;
    float step_s, step_c;
    sincosf(-(2.0f * M_PIf) * static_cast<float>(m2) / {width}.0f, &step_s, &step_c);
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
    const int out = ((b * channels + ch) * height + h) * 32 + m2 * 2;
    row_freq[out] = acc_r;
    row_freq[out + 1] = acc_i;
}}
"""


def emit_fft_h_fact_smem(height: int, m2_per_block: int = 0) -> str:
    """Mixed-radix column FFT: 16 n1 × m2 threads, N2-pt -i DFT, smem reduce two corners."""
    n1_count = 16
    n2 = height // n1_count
    if n2 not in (4, 8, 16):
        raise ValueError(f"unsupported fft_h fact n2={n2}")
    if m2_per_block == 0:
        m2_per_block = 8 if height == 256 else 16
    if 16 % m2_per_block != 0:
        raise ValueError("m2_per_block must divide 16")
    m2_mask = m2_per_block - 1
    n1_shift = m2_per_block.bit_length() - 1
    name = f"pruned_fft_h_m16_fact16x{n2}_h{height}_kernel"
    kh_bot = height - 16
    udecl = "\n".join(f"    float u{m}r, u{m}i;" for m in range(n2))
    loads = []
    for n2i in range(n2):
        loads.append("    {")
        loads.append(f"        const int h = n1 + {n1_count * n2i};")
        loads.append("        const int in = row_plane + (h * 16 + m2) * 2;")
        loads.append(f"        u{n2i}r = row_freq[in];")
        loads.append(f"        u{n2i}i = row_freq[in + 1];")
        loads.append("    }")
    loads_s = "\n".join(loads)
    stores_z = "\n".join(
        f"    Z[m2_local][n1][{m}].x = u{m}r;\n"
        f"    Z[m2_local][n1][{m}].y = u{m}i;"
        for m in range(n2)
    )
    dft_args = ", ".join(f"u{m}r, u{m}i" for m in range(n2))
    reduce_loop = """    {
        float acc_r = 0.0f;
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
        }
        packed[out_idx] = acc_r;
        packed[out_idx + 1] = acc_i;
    }"""
    return f"""
__global__ void {name}(int batch_size, int channels,
                       const float *__restrict__ row_freq, float *__restrict__ packed) {{
    __shared__ float2 Z[{m2_per_block}][16][{n2}];
    const int tid = static_cast<int>(threadIdx.x);
    const int m2_local = tid & {m2_mask};
    const int n1 = tid >> {n1_shift};
    const int m2 = static_cast<int>(blockIdx.y) * {m2_per_block} + m2_local;
    const int plane_idx = static_cast<int>(blockIdx.x);
    const int ch = plane_idx % channels;
    const int b = plane_idx / channels;
    const int row_plane = (b * channels + ch) * {height} * 32;
{udecl}
{loads_s}
    dft{n2}_minus({dft_args});
{stores_z}
    __syncthreads();
    const int m1 = n1;
    const int bin = m1 & {n2 - 1};
    float step_s, step_c;
    int out_idx;
    sincosf(-(2.0f * M_PIf) * static_cast<float>(m1) / {height}.0f, &step_s, &step_c);
    out_idx = ((b * channels + ch) * {height} + m1) * 32 + m2 * 2;
{reduce_loop}
    sincosf(-(2.0f * M_PIf) * static_cast<float>({kh_bot} + m1) / {height}.0f, &step_s, &step_c);
    out_idx = ((b * channels + ch) * {height} + {kh_bot} + m1) * 32 + m2 * 2;
{reduce_loop}
}}
"""


def emit_irfft_w_fact(width: int, n1_count: int) -> str:
    """Mixed-radix irfft: n = n1 + n1_count*n2, n2-pt +i DFT of folded 16 bins."""
    n2 = width // n1_count
    if n2 not in (4, 8, 16) or 16 % n2 != 0:
        raise ValueError(f"unsupported fact {n1_count}x{n2}")
    name = f"pruned_irfft_w_m16_fact{n1_count}x{n2}_w{width}_kernel"
    shift = n1_count.bit_length() - 1
    loads = [
        "    const float2 z0 = *reinterpret_cast<const float2 *>(row_freq + row_base);",
        "    const float r0 = z0.x;",
    ]
    for k in range(1, 16):
        loads.append(
            f"    const float2 z{k} = *reinterpret_cast<const float2 *>(row_freq + row_base + {2 * k});"
        )
        loads.append(f"    const float r{k} = z{k}.x;")
        loads.append(f"    const float i{k} = z{k}.y;")
    body = []
    for m in range(n2):
        body.append(f"    float b{m}r = 0.0f;")
        body.append(f"    float b{m}i = 0.0f;")
    body.append("    b0r = r0;")
    body.append("    float step_s, step_c;")
    body.append(
        f"    sincosf((2.0f * M_PIf) * static_cast<float>(n1) / {width}.0f, &step_s, &step_c);"
    )
    body.append("    float wr = step_c;")
    body.append("    float wi = step_s;")
    for k in range(1, 16):
        m = k % n2
        body.append(f"    b{m}r += r{k} * wr - i{k} * wi;")
        body.append(f"    b{m}i += r{k} * wi + i{k} * wr;")
        if k < 15:
            body.extend(
                [
                    "    {",
                    "        const float nr = wr * step_c - wi * step_s;",
                    "        const float ni = wr * step_s + wi * step_c;",
                    "        wr = nr;",
                    "        wi = ni;",
                    "    }",
                ]
            )
    dft_args = ", ".join(f"b{m}r, b{m}i" for m in range(n2))
    body.append(f"    dft{n2}_plus({dft_args});")
    body.append(f"    const float invw = 1.0f / {width}.0f;")
    for n2i in range(n2):
        body.append(
            f"    spatial[out_base + n1 + {n1_count * n2i}] = (2.0f * b{n2i}r - r0) * invw;"
        )
    loads_s = "\n".join(loads)
    body_s = "\n".join(body)
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ row_freq, float *__restrict__ spatial) {{
    const int total = batch_size * channels * height * {n1_count};
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int n1 = index & {n1_count - 1};
    int rest = index >> {shift};
    const int h = rest % height;
    rest /= height;
    const int ch = rest % channels;
    const int b = rest / channels;
    const int row_base = ((b * channels + ch) * height + h) * 32;
{loads_s}
    const int out_base = ((b * channels + ch) * height + h) * {width};
{body_s}
}}
"""


def emit_irfft_w_fact_smem(width: int, rows_per_block: int = 8) -> str:
    """Mixed-radix irfft with one shared load of 16 bins per row."""
    n1_count = 16
    n2 = width // n1_count
    if n2 not in (8, 16):
        raise ValueError(f"unsupported irfft smem n2={n2}")
    name = f"pruned_irfft_w_m16_fact16x{n2}_smem_w{width}_kernel"
    body = []
    for m in range(n2):
        body.append(f"    float b{m}r = 0.0f;")
        body.append(f"    float b{m}i = 0.0f;")
    body.append("    const float r0 = bins[local_row][0].x;")
    body.append("    b0r = r0;")
    body.append("    float step_s, step_c;")
    body.append(
        f"    sincosf((2.0f * M_PIf) * static_cast<float>(n1) / {width}.0f, &step_s, &step_c);"
    )
    body.append("    float wr = step_c;")
    body.append("    float wi = step_s;")
    for k in range(1, 16):
        m = k % n2
        body.append("    {")
        body.append(f"        const float rk = bins[local_row][{k}].x;")
        body.append(f"        const float ik = bins[local_row][{k}].y;")
        body.append(f"        b{m}r += rk * wr - ik * wi;")
        body.append(f"        b{m}i += rk * wi + ik * wr;")
        body.append("    }")
        if k < 15:
            body.extend(
                [
                    "    {",
                    "        const float nr = wr * step_c - wi * step_s;",
                    "        const float ni = wr * step_s + wi * step_c;",
                    "        wr = nr;",
                    "        wi = ni;",
                    "    }",
                ]
            )
    dft_args = ", ".join(f"b{m}r, b{m}i" for m in range(n2))
    body.append(f"    dft{n2}_plus({dft_args});")
    body.append(f"    const float invw = 1.0f / {width}.0f;")
    for n2i in range(n2):
        body.append(
            f"    spatial[out_base + n1 + {n1_count * n2i}] = (2.0f * b{n2i}r - r0) * invw;"
        )
    body_s = "\n".join(body)
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ row_freq, float *__restrict__ spatial) {{
    __shared__ float2 bins[{rows_per_block}][16];
    const int tid = static_cast<int>(threadIdx.x);
    const int n1 = tid & 15;
    const int local_row = tid >> 4;
    const int row0 = static_cast<int>(blockIdx.x) * {rows_per_block} + local_row;
    const int total_rows = batch_size * channels * height;
    const int valid = row0 < total_rows;
    int b = 0, ch = 0, h = 0, row_base = 0, out_base = 0;
    if (valid) {{
        h = row0 % height;
        int rest = row0 / height;
        ch = rest % channels;
        b = rest / channels;
        row_base = ((b * channels + ch) * height + h) * 32;
        out_base = ((b * channels + ch) * height + h) * {width};
        bins[local_row][n1] = *reinterpret_cast<const float2 *>(row_freq + row_base + n1 * 2);
    }}
    __syncthreads();
    if (!valid) {{
        return;
    }}
{body_s}
}}
"""


def _emit_irfft_one_n1(n2: int, width: int, n1_count: int, n1_var: str) -> str:
    lines = []
    for m in range(n2):
        lines.append(f"    b{m}r = 0.0f;")
        lines.append(f"    b{m}i = 0.0f;")
    lines.append("    b0r = r0;")
    lines.append(
        f"    sincosf((2.0f * M_PIf) * static_cast<float>({n1_var}) / {width}.0f, &step_s, &step_c);"
    )
    lines.append("    wr = step_c;")
    lines.append("    wi = step_s;")
    for k in range(1, 16):
        m = k % n2
        lines.append(f"    b{m}r += r{k} * wr - i{k} * wi;")
        lines.append(f"    b{m}i += r{k} * wi + i{k} * wr;")
        if k < 15:
            lines.extend(
                [
                    "    {",
                    "        const float nr = wr * step_c - wi * step_s;",
                    "        const float ni = wr * step_s + wi * step_c;",
                    "        wr = nr;",
                    "        wi = ni;",
                    "    }",
                ]
            )
    dft_args = ", ".join(f"b{m}r, b{m}i" for m in range(n2))
    lines.append(f"    dft{n2}_plus({dft_args});")
    for n2i in range(n2):
        lines.append(
            f"    spatial[out_base + {n1_var} + {n1_count * n2i}] = (2.0f * b{n2i}r - r0) * invw;"
        )
    return "\n".join(lines)


def emit_irfft_w_fact_pair(width: int = 256) -> str:
    """8 threads/row: load 16 bins once (float4), sequential n1 and n1+1."""
    n1_count = 16
    n2 = width // n1_count
    if n2 != 16:
        raise ValueError(f"pair kernel only for n2=16, got {n2}")
    name = f"pruned_irfft_w_m16_fact16x{n2}_pair_w{width}_kernel"
    load_pairs = []
    for i in range(8):
        load_pairs.append(f"    const float4 p{i} = bp[{i}];")
    unpack = [
        "    const float r0 = p0.x;",
        "    const float r1 = p0.z;",
        "    const float i1 = p0.w;",
    ]
    for k in range(2, 16):
        pi = k // 2
        if k % 2 == 0:
            unpack.append(f"    const float r{k} = p{pi}.x;")
            unpack.append(f"    const float i{k} = p{pi}.y;")
        else:
            unpack.append(f"    const float r{k} = p{pi}.z;")
            unpack.append(f"    const float i{k} = p{pi}.w;")
    bdecl = "\n".join(f"    float b{m}r, b{m}i;" for m in range(n2))
    body_a = _emit_irfft_one_n1(n2, width, n1_count, "n1a")
    body_b = _emit_irfft_one_n1(n2, width, n1_count, "n1b")
    loads_s = "\n".join(load_pairs)
    unpack_s = "\n".join(unpack)
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ row_freq, float *__restrict__ spatial) {{
    const int total = batch_size * channels * height * 8;
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int pair = index & 7;
    const int n1a = pair << 1;
    const int n1b = n1a + 1;
    int rest = index >> 3;
    const int h = rest % height;
    rest /= height;
    const int ch = rest % channels;
    const int b = rest / channels;
    const int row_base = ((b * channels + ch) * height + h) * 32;
    const int out_base = ((b * channels + ch) * height + h) * {width};
    const float4 *bp = reinterpret_cast<const float4 *>(row_freq + row_base);
{loads_s}
{unpack_s}
{bdecl}
    float step_s, step_c, wr, wi;
    const float invw = 1.0f / {width}.0f;
{body_a}
{body_b}
}}
"""


def emit_irfft_pair256_launches() -> str:
    return r"""
extern "C" suError_t launch_pruned_irfft_w_fact16x16_pair_w256(const float *row_freq, float *spatial,
                                                              int batch_size, int channels, int height,
                                                              suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 8);
    dim3 block(256);
    dim3 grid((nthreads + 255u) / 256u);
    pruned_irfft_w_m16_fact16x16_pair_w256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}
"""


def emit_irfft_w_fact_vec4(width: int) -> str:
    """Mixed-radix irfft with 8× float4 spectrum loads. One dftN, 16 n1 threads."""
    n1_count = 16
    n2 = width // n1_count
    if n2 not in (4, 8, 16):
        raise ValueError(f"unsupported vec4 n2={n2}")
    name = f"pruned_irfft_w_m16_fact16x{n2}_vec4_w{width}_kernel"
    loads = ["    const float4 *bp = reinterpret_cast<const float4 *>(row_freq + row_base);"]
    for i in range(8):
        loads.append(f"    const float4 p{i} = bp[{i}];")
    loads.append("    const float r0 = p0.x;")
    loads.append("    const float r1 = p0.z;")
    loads.append("    const float i1 = p0.w;")
    for k in range(2, 16):
        pi = k // 2
        if k % 2 == 0:
            loads.append(f"    const float r{k} = p{pi}.x;")
            loads.append(f"    const float i{k} = p{pi}.y;")
        else:
            loads.append(f"    const float r{k} = p{pi}.z;")
            loads.append(f"    const float i{k} = p{pi}.w;")
    body = []
    for m in range(n2):
        body.append(f"    float b{m}r = 0.0f;")
        body.append(f"    float b{m}i = 0.0f;")
    body.append("    b0r = r0;")
    body.append("    float step_s, step_c;")
    body.append(
        f"    sincosf((2.0f * M_PIf) * static_cast<float>(n1) / {width}.0f, &step_s, &step_c);"
    )
    body.append("    float wr = step_c;")
    body.append("    float wi = step_s;")
    for k in range(1, 16):
        m = k % n2
        body.append(f"    b{m}r += r{k} * wr - i{k} * wi;")
        body.append(f"    b{m}i += r{k} * wi + i{k} * wr;")
        if k < 15:
            body.extend(
                [
                    "    {",
                    "        const float nr = wr * step_c - wi * step_s;",
                    "        const float ni = wr * step_s + wi * step_c;",
                    "        wr = nr;",
                    "        wi = ni;",
                    "    }",
                ]
            )
    dft_args = ", ".join(f"b{m}r, b{m}i" for m in range(n2))
    body.append(f"    dft{n2}_plus({dft_args});")
    body.append(f"    const float invw = 1.0f / {width}.0f;")
    for n2i in range(n2):
        body.append(
            f"    spatial[out_base + n1 + {n1_count * n2i}] = (2.0f * b{n2i}r - r0) * invw;"
        )
    loads_s = "\n".join(loads)
    body_s = "\n".join(body)
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ row_freq, float *__restrict__ spatial) {{
    const int total = batch_size * channels * height * {n1_count};
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int n1 = index & 15;
    int rest = index >> 4;
    const int h = rest % height;
    rest /= height;
    const int ch = rest % channels;
    const int b = rest / channels;
    const int row_base = ((b * channels + ch) * height + h) * 32;
    const int out_base = ((b * channels + ch) * height + h) * {width};
{loads_s}
{body_s}
}}
"""


def emit_irfft_vec4_launches(width: int) -> str:
    n2 = width // 16
    return f"""
extern "C" suError_t launch_pruned_irfft_w_fact16x{n2}_vec4_w{width}(const float *row_freq, float *spatial,
                                                              int batch_size, int channels, int height,
                                                              suStream_t stream) {{
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 16);
    dim3 block(256);
    dim3 grid((nthreads + 255u) / 256u);
    pruned_irfft_w_m16_fact16x{n2}_vec4_w{width}_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}}
"""


def emit_irfft_smem256_launches() -> str:
    return r"""
extern "C" suError_t launch_pruned_irfft_w_fact16x16_smem_w256(const float *row_freq, float *spatial,
                                                              int batch_size, int channels, int height,
                                                              suStream_t stream) {
    const unsigned rows = static_cast<unsigned>(batch_size * channels * height);
    dim3 block(128);
    dim3 grid((rows + 7u) / 8u);
    pruned_irfft_w_m16_fact16x16_smem_w256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}
"""


def _emit_ifft_h_fold(n2: int, row0: int) -> str:
    lines = [
        f"    {{ const float2 z = *reinterpret_cast<const float2 *>("
        f"in_freq + plane + ({row0} * 16 + m2) * 2);",
        "      u0r += z.x; u0i += z.y; }",
        "    wr = step_c;",
        "    wi = step_s;",
    ]
    for k in range(1, 16):
        m = k % n2
        row = row0 + k
        lines.append(
            f"    {{ const float2 z = *reinterpret_cast<const float2 *>("
            f"in_freq + plane + ({row} * 16 + m2) * 2);"
        )
        lines.append(f"      u{m}r += z.x * wr - z.y * wi;")
        lines.append(f"      u{m}i += z.x * wi + z.y * wr; }}")
        if k < 15:
            lines.extend(
                [
                    "    {",
                    "        const float nr = wr * step_c - wi * step_s;",
                    "        const float ni = wr * step_s + wi * step_c;",
                    "        wr = nr;",
                    "        wi = ni;",
                    "    }",
                ]
            )
    return "\n".join(lines)


def emit_ifft_h_fact(height: int, n1_count: int = 16) -> str:
    """Mixed-radix ifft along H: two FNO corners, n = n1 + n1_count*n2."""
    n2 = height // n1_count
    if n2 not in (4, 8, 16) or 16 % n2 != 0:
        raise ValueError(f"unsupported ifft_h fact {n1_count}x{n2} h{height}")
    name = f"pruned_ifft_h_m16_fact{n1_count}x{n2}_h{height}_kernel"
    n1_bits = n1_count.bit_length() - 1
    rest_shift = 4 + n1_bits
    kh0 = height - 16
    body = []
    for m in range(n2):
        body.append(f"    float u{m}r = 0.0f;")
        body.append(f"    float u{m}i = 0.0f;")
    body.append("    float step_s, step_c, wr, wi;")
    body.append(
        f"    sincosf((2.0f * M_PIf) * static_cast<float>(n1) / {height}.0f, &step_s, &step_c);"
    )
    body.append(_emit_ifft_h_fold(n2, 0))
    dft_args = ", ".join(f"u{m}r, u{m}i" for m in range(n2))
    body.append(f"    dft{n2}_plus({dft_args});")
    for m in range(n2):
        body.append(f"    const float p{m}r = u{m}r;")
        body.append(f"    const float p{m}i = u{m}i;")
        body.append(f"    u{m}r = 0.0f;")
        body.append(f"    u{m}i = 0.0f;")
    body.append(_emit_ifft_h_fold(n2, kh0))
    body.append(f"    dft{n2}_plus({dft_args});")
    body.append("    float ps, pc;")
    body.append(
        f"    sincosf((2.0f * M_PIf) * 16.0f * static_cast<float>(n1) / {height}.0f, &ps, &pc);"
    )
    body.append(f"    const float inv_h = 1.0f / {height}.0f;")
    for n2i in range(n2):
        body.append("    {")
        body.append(f"        const float br = u{n2i}r * pc + u{n2i}i * ps;")
        body.append(f"        const float bi = u{n2i}i * pc - u{n2i}r * ps;")
        body.append(f"        const int h = n1 + {n1_count * n2i};")
        body.append("        const int out = plane + (h * 16 + m2) * 2;")
        body.append(f"        row_freq[out] = (p{n2i}r + br) * inv_h;")
        body.append(f"        row_freq[out + 1] = (p{n2i}i + bi) * inv_h;")
        body.append("    }")
    body_s = "\n".join(body)
    return f"""
__global__ void {name}(int batch_size, int channels,
                       const float *__restrict__ in_freq, float *__restrict__ row_freq) {{
    const int total = batch_size * channels * 16 * {n1_count};
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int m2 = index & 15;
    const int n1 = (index >> 4) & {n1_count - 1};
    int rest = index >> {rest_shift};
    const int ch = rest % channels;
    const int b = rest / channels;
    const int plane = ((b * channels + ch) * {height}) * 32;
{body_s}
}}
"""


def emit_irfft_w_x2_ilp(width: int) -> str:
    pairs = width // 2
    shift = pairs.bit_length() - 1
    name = f"pruned_irfft_w_m16_x2_ilp_w{width}_kernel"
    loads = [
        "    const float2 z0 = *reinterpret_cast<const float2 *>(row_freq + row_base);",
        "    const float r0 = z0.x;",
    ]
    for k in range(1, 16):
        loads.append(
            f"    const float2 z{k} = *reinterpret_cast<const float2 *>(row_freq + row_base + {2 * k});"
        )
        loads.append(f"    const float r{k} = z{k}.x;")
        loads.append(f"    const float i{k} = z{k}.y;")
    body = [
        "    float acc0 = r0;",
        "    float acc1 = r0;",
        "    const int w0 = pair << 1;",
        "    const int w1 = w0 + 1;",
        f"    const float scale = (2.0f * M_PIf) / {width}.0f;",
        "    float s0, c0, s1, c1;",
        "    sincosf(scale * static_cast<float>(w0), &s0, &c0);",
        "    sincosf(scale * static_cast<float>(w1), &s1, &c1);",
        "    float wr0 = c0, wi0 = s0, wr1 = c1, wi1 = s1;",
    ]
    for k in range(1, 16):
        body.append(f"    acc0 += 2.0f * (r{k} * wr0 - i{k} * wi0);")
        body.append(f"    acc1 += 2.0f * (r{k} * wr1 - i{k} * wi1);")
        if k < 15:
            body.extend(
                [
                    "    {",
                    "        const float n0r = wr0 * c0 - wi0 * s0;",
                    "        const float n0i = wr0 * s0 + wi0 * c0;",
                    "        wr0 = n0r; wi0 = n0i;",
                    "        const float n1r = wr1 * c1 - wi1 * s1;",
                    "        const float n1i = wr1 * s1 + wi1 * c1;",
                    "        wr1 = n1r; wi1 = n1i;",
                    "    }",
                ]
            )
    body.append(f"    spatial[out_base + w0] = acc0 / {width}.0f;")
    body.append(f"    spatial[out_base + w1] = acc1 / {width}.0f;")
    loads_s = "\n".join(loads)
    body_s = "\n".join(body)
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ row_freq, float *__restrict__ spatial) {{
    const int total = batch_size * channels * height * {pairs};
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int pair = index & {pairs - 1};
    int rest = index >> {shift};
    const int h = rest % height;
    rest /= height;
    const int ch = rest % channels;
    const int b = rest / channels;
    const int row_base = ((b * channels + ch) * height + h) * 32;
{loads_s}
    const int out_base = ((b * channels + ch) * height + h) * {width};
{body_s}
}}
"""


def emit_irfft_w_xn_ilp(width: int, pix: int, reps: int = 1) -> str:
    groups = width // pix
    n_idx = groups // reps
    shift = n_idx.bit_length() - 1
    suffix = f"_r{reps}" if reps > 1 else ""
    name = f"pruned_irfft_w_m16_x{pix}_ilp{suffix}_w{width}_kernel"
    loads = [
        "    const float2 z0 = *reinterpret_cast<const float2 *>(row_freq + row_base);",
        "    const float r0 = z0.x;",
    ]
    for k in range(1, 16):
        loads.append(
            f"    const float2 z{k} = *reinterpret_cast<const float2 *>(row_freq + row_base + {2 * k});"
        )
        loads.append(f"    const float r{k} = z{k}.x;")
        loads.append(f"    const float i{k} = z{k}.y;")
    theta = 2.0 * math.pi / width
    inner = [
        f"        const int w0 = (pair * {reps} + g) * {pix};",
        f"        const float scale = (2.0f * M_PIf) / {width}.0f;",
        f"        const float base_c = {fmt_f(math.cos(theta))};",
        f"        const float base_s = {fmt_f(math.sin(theta))};",
        "        float s0, c0;",
        "        sincosf(scale * static_cast<float>(w0), &s0, &c0);",
    ]
    for p in range(1, pix):
        prev = p - 1
        inner.append(f"        const float c{p} = c{prev} * base_c - s{prev} * base_s;")
        inner.append(f"        const float s{p} = c{prev} * base_s + s{prev} * base_c;")
    for p in range(pix):
        inner.append(f"        float acc{p} = r0;")
        inner.append(f"        float wr{p} = c{p}, wi{p} = s{p};")
    for k in range(1, 16):
        for p in range(pix):
            inner.append(f"        acc{p} += 2.0f * (r{k} * wr{p} - i{k} * wi{p});")
        if k < 15:
            inner.append("        {")
            for p in range(pix):
                inner.append(f"            const float n{p}r = wr{p} * c{p} - wi{p} * s{p};")
                inner.append(f"            const float n{p}i = wr{p} * s{p} + wi{p} * c{p};")
                inner.append(f"            wr{p} = n{p}r; wi{p} = n{p}i;")
            inner.append("        }")
    inner.append(f"        const float invw = 1.0f / {width}.0f;")
    if pix == 4:
        inner.extend(
            [
                "        float4 o;",
                "        o.x = acc0 * invw;",
                "        o.y = acc1 * invw;",
                "        o.z = acc2 * invw;",
                "        o.w = acc3 * invw;",
                "        *reinterpret_cast<float4 *>(spatial + out_base + w0) = o;",
            ]
        )
    elif pix == 2:
        inner.extend(
            [
                "        float2 o;",
                "        o.x = acc0 * invw;",
                "        o.y = acc1 * invw;",
                "        *reinterpret_cast<float2 *>(spatial + out_base + w0) = o;",
            ]
        )
    else:
        for p in range(pix):
            inner.append(f"        spatial[out_base + w0 + {p}] = acc{p} * invw;")
    loads_s = "\n".join(loads)
    inner_s = "\n".join(inner)
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ row_freq, float *__restrict__ spatial) {{
    const int total = batch_size * channels * height * {n_idx};
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int pair = index & {n_idx - 1};
    int rest = index >> {shift};
    const int h = rest % height;
    rest /= height;
    const int ch = rest % channels;
    const int b = rest / channels;
    const int row_base = ((b * channels + ch) * height + h) * 32;
{loads_s}
    const int out_base = ((b * channels + ch) * height + h) * {width};
#pragma unroll
    for (int g = 0; g < {reps}; ++g) {{
{inner_s}
    }}
}}
"""


def emit_irfft_w_smem_row(width: int, pix: int, rows_per_block: int = 4) -> str:
    groups = width // pix
    row_shift = groups.bit_length() - 1
    name = f"pruned_irfft_w_m16_x{pix}_smem_w{width}_kernel"
    theta = 2.0 * math.pi / width
    decls = ["        const float r0 = sb[local_row][0].x;"]
    for k in range(1, 16):
        decls.append(f"        const float r{k} = sb[local_row][{k}].x;")
        decls.append(f"        const float i{k} = sb[local_row][{k}].y;")
    body = [
        f"        const int w0 = pair * {pix};",
        f"        const float scale = (2.0f * M_PIf) / {width}.0f;",
        f"        const float base_c = {fmt_f(math.cos(theta))};",
        f"        const float base_s = {fmt_f(math.sin(theta))};",
        "        float s0, c0;",
        "        sincosf(scale * static_cast<float>(w0), &s0, &c0);",
    ]
    for p in range(1, pix):
        prev = p - 1
        body.append(f"        const float c{p} = c{prev} * base_c - s{prev} * base_s;")
        body.append(f"        const float s{p} = c{prev} * base_s + s{prev} * base_c;")
    for p in range(pix):
        body.append(f"        float acc{p} = r0;")
        body.append(f"        float wr{p} = c{p}, wi{p} = s{p};")
    for k in range(1, 16):
        for p in range(pix):
            body.append(f"        acc{p} += 2.0f * (r{k} * wr{p} - i{k} * wi{p});")
        if k < 15:
            body.append("        {")
            for p in range(pix):
                body.append(f"            const float n{p}r = wr{p} * c{p} - wi{p} * s{p};")
                body.append(f"            const float n{p}i = wr{p} * s{p} + wi{p} * c{p};")
                body.append(f"            wr{p} = n{p}r; wi{p} = n{p}i;")
            body.append("        }")
    body.append(f"        const float invw = 1.0f / {width}.0f;")
    if pix == 4:
        body.extend(
            [
                "        float4 o;",
                "        o.x = acc0 * invw;",
                "        o.y = acc1 * invw;",
                "        o.z = acc2 * invw;",
                "        o.w = acc3 * invw;",
                "        *reinterpret_cast<float4 *>(spatial + out_base + w0) = o;",
            ]
        )
    else:
        for p in range(pix):
            body.append(f"        spatial[out_base + w0 + {p}] = acc{p} * invw;")
    decls_s = "\n".join(decls)
    body_s = "\n".join(body)
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ row_freq, float *__restrict__ spatial) {{
    __shared__ float2 sb[{rows_per_block}][16];
    const int local_row = static_cast<int>(threadIdx.x) >> {row_shift};
    const int pair = static_cast<int>(threadIdx.x) & {groups - 1};
    const int total_rows = batch_size * channels * height;
    const int row = static_cast<int>(blockIdx.x) * {rows_per_block} + local_row;
    const bool valid = row < total_rows;
    int h = 0;
    int ch = 0;
    int b = 0;
    if (valid) {{
        h = row % height;
        int rem = row / height;
        ch = rem % channels;
        b = rem / channels;
    }}
    const int row_base = ((b * channels + ch) * height + h) * 32;
    if (valid && pair < 16) {{
        sb[local_row][pair] = *reinterpret_cast<const float2 *>(row_freq + row_base + pair * 2);
    }}
    __syncthreads();
    if (!valid) {{
        return;
    }}
    const int out_base = ((b * channels + ch) * height + h) * {width};
{decls_s}
{body_s}
}}
"""


def emit_irfft_w_x4_stride_lut() -> str:
    width = 256
    pix = 4
    theta = 2.0 * math.pi / width
    decls = ["            const float r0 = *reinterpret_cast<const float2 *>(row_freq + row_base);"]
    # r0 from float2 .x - actually the load above is wrong type. Use named like xn_ilp.
    decls = [
        "            const float2 z0 = *reinterpret_cast<const float2 *>(row_freq + row_base);",
        "            const float r0 = z0.x;",
    ]
    for k in range(1, 16):
        decls.append(
            f"            const float2 z{k} = *reinterpret_cast<const float2 *>(row_freq + row_base + {2 * k});"
        )
        decls.append(f"            const float r{k} = z{k}.x;")
        decls.append(f"            const float i{k} = z{k}.y;")
    body = [
        "            const int w0 = pair << 2;",
        "            const float c0 = tw[w0].x;",
        "            const float s0 = tw[w0].y;",
        "            const float c1 = tw[w0 + 1].x;",
        "            const float s1 = tw[w0 + 1].y;",
        "            const float c2 = tw[w0 + 2].x;",
        "            const float s2 = tw[w0 + 2].y;",
        "            const float c3 = tw[w0 + 3].x;",
        "            const float s3 = tw[w0 + 3].y;",
    ]
    for p in range(4):
        body.append(f"            float acc{p} = r0;")
        body.append(f"            float wr{p} = c{p}, wi{p} = s{p};")
    for k in range(1, 16):
        for p in range(4):
            body.append(f"            acc{p} += 2.0f * (r{k} * wr{p} - i{k} * wi{p});")
        if k < 15:
            body.append("            {")
            for p in range(4):
                body.append(f"                const float n{p}r = wr{p} * c{p} - wi{p} * s{p};")
                body.append(f"                const float n{p}i = wr{p} * s{p} + wi{p} * c{p};")
                body.append(f"                wr{p} = n{p}r; wi{p} = n{p}i;")
            body.append("            }")
    body.append("            const float invw = 1.0f / 256.0f;")
    body.extend(
        [
            "            float4 o;",
            "            o.x = acc0 * invw;",
            "            o.y = acc1 * invw;",
            "            o.z = acc2 * invw;",
            "            o.w = acc3 * invw;",
            "            *reinterpret_cast<float4 *>(spatial + out_base + w0) = o;",
        ]
    )
    decls_s = "\n".join(decls)
    body_s = "\n".join(body)
    return f"""
__global__ void pruned_irfft_w_m16_x4_stride_lut_w256_kernel(
    int batch_size, int channels, int height,
    const float *__restrict__ row_freq, float *__restrict__ spatial) {{
    __shared__ float2 tw[256];
    const int tid = static_cast<int>(threadIdx.x);
    {{
        float s, c;
        sincosf((2.0f * M_PIf) * static_cast<float>(tid) / 256.0f, &s, &c);
        tw[tid].x = c;
        tw[tid].y = s;
    }}
    __syncthreads();
    const int local_row = tid >> 6;
    const int pair = tid & 63;
    const int total_rows = batch_size * channels * height;
    const int grid_stride = static_cast<int>(gridDim.x) * 4;
    for (int row0 = static_cast<int>(blockIdx.x) * 4; row0 < total_rows; row0 += grid_stride) {{
        const int row = row0 + local_row;
        if (row >= total_rows) {{
            continue;
        }}
        const int h = row % height;
        int rem = row / height;
        const int ch = rem % channels;
        const int b = rem / channels;
        const int row_base = ((b * channels + ch) * height + h) * 32;
        const int out_base = ((b * channels + ch) * height + h) * 256;
{decls_s}
{body_s}
    }}
}}
"""


def emit_ifft_h_x2_named() -> str:
    loads = []
    for m1 in range(16):
        loads.append(
            f"    const float2 t{m1} = *reinterpret_cast<const float2 *>("
            f"in_freq + plane + ({m1} * 16 + m2) * 2);"
        )
    for m1 in range(16):
        loads.append(
            f"    const float2 b{m1} = *reinterpret_cast<const float2 *>("
            f"in_freq + plane + ((kh0 + {m1}) * 16 + m2) * 2);"
        )

    def macs(var: str) -> str:
        lines = []
        for m1 in range(16):
            lines.append(f"        acc_r += {var}{m1}.x * wr - {var}{m1}.y * wi;")
            lines.append(f"        acc_i += {var}{m1}.x * wi + {var}{m1}.y * wr;")
            if m1 < 15:
                lines.append(
                    "        { const float nr = wr * step_c - wi * step_s; "
                    "const float ni = wr * step_s + wi * step_c; wr = nr; wi = ni; }"
                )
        return "\n".join(lines)

    loads_s = "\n".join(loads)
    top_macs = macs("t")
    bot_macs = macs("b")
    return f"""
__global__ void pruned_ifft_h_m16_x2_named_kernel(int batch_size, int channels, int height,
                                                  const float *__restrict__ in_freq,
                                                  float *__restrict__ row_freq) {{
    const int pairs = height >> 1;
    const int total = batch_size * channels * pairs * 16;
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int m2 = index & 15;
    int rem = index >> 4;
    const int pair = rem % pairs;
    rem /= pairs;
    const int ch = rem % channels;
    const int b = rem / channels;
    const int plane = ((b * channels + ch) * height) * 32;
    const int kh0 = height - 16;
{loads_s}
    const float inv_h = 1.0f / static_cast<float>(height);
#pragma unroll
    for (int dw = 0; dw < 2; ++dw) {{
        const int h = (pair << 1) + dw;
        const float two_pi_h = (2.0f * M_PIf) * static_cast<float>(h) / static_cast<float>(height);
        float step_s, step_c;
        sincosf(two_pi_h, &step_s, &step_c);
        float acc_r = 0.0f;
        float acc_i = 0.0f;
        float wr = 1.0f;
        float wi = 0.0f;
{top_macs}
        sincosf(two_pi_h * static_cast<float>(kh0), &wi, &wr);
{bot_macs}
        const int out = plane + (h * 16 + m2) * 2;
        row_freq[out] = acc_r * inv_h;
        row_freq[out + 1] = acc_i * inv_h;
    }}
}}
"""


def emit_ifft_h_x2_h(height: int) -> str:
    loads = []
    for m1 in range(16):
        loads.append(
            f"    const float2 t{m1} = *reinterpret_cast<const float2 *>("
            f"in_freq + plane + ({m1} * 16 + m2) * 2);"
        )
    for m1 in range(16):
        loads.append(
            f"    const float2 b{m1} = *reinterpret_cast<const float2 *>("
            f"in_freq + plane + (({height - 16} + {m1}) * 16 + m2) * 2);"
        )

    def macs(var: str) -> str:
        lines = []
        for m1 in range(16):
            lines.append(f"        acc_r += {var}{m1}.x * wr - {var}{m1}.y * wi;")
            lines.append(f"        acc_i += {var}{m1}.x * wi + {var}{m1}.y * wr;")
            if m1 < 15:
                lines.append(
                    "        { const float nr = wr * step_c - wi * step_s; "
                    "const float ni = wr * step_s + wi * step_c; wr = nr; wi = ni; }"
                )
        return "\n".join(lines)

    loads_s = "\n".join(loads)
    top_macs = macs("t")
    bot_macs = macs("b")
    name = f"pruned_ifft_h_m16_x2_h{height}_kernel"
    lut = f"pruned_cs_lut_{height}"
    kh0 = height - 16
    mask = height - 1
    inv = fmt_f(1.0 / height)
    return f"""
__global__ void {name}(int batch_size, int channels,
                       const float *__restrict__ in_freq, float *__restrict__ row_freq) {{
    const int pairs = {height >> 1};
    const int total = batch_size * channels * pairs * 16;
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total) {{
        return;
    }}
    const int m2 = index & 15;
    int rem = index >> 4;
    const int pair = rem % pairs;
    rem /= pairs;
    const int ch = rem % channels;
    const int b = rem / channels;
    const int plane = ((b * channels + ch) * {height}) * 32;
{loads_s}
#pragma unroll
    for (int dw = 0; dw < 2; ++dw) {{
        const int h = (pair << 1) + dw;
        const float step_c = {lut}[h][0];
        const float step_s = {lut}[h][1];
        float acc_r = 0.0f;
        float acc_i = 0.0f;
        float wr = 1.0f;
        float wi = 0.0f;
{top_macs}
        const int bot_i = (h * {kh0}) & {mask};
        wr = {lut}[bot_i][0];
        wi = {lut}[bot_i][1];
{bot_macs}
        const int out = plane + (h * 16 + m2) * 2;
        float2 o;
        o.x = acc_r * {inv};
        o.y = acc_i * {inv};
        *reinterpret_cast<float2 *>(row_freq + out) = o;
    }}
}}
"""


def emit_fused_inv_occ(width: int) -> str:
    pix = width // 32
    name = f"pruned_irfft2_fused_occ_w{width}_kernel"
    loads = "\n".join(
        f"    const float r{k} = sr[local_row][{k}];\n    const float i{k} = si[local_row][{k}];"
        if k
        else f"    const float r0 = sr[local_row][0];"
        for k in range(16)
    )
    acc = [
        "        float acc = r0;",
        f"        const float two_pi_w = (2.0f * M_PIf) * static_cast<float>(w) / {width}.0f;",
        "        float step_s, step_c;",
        "        sincosf(two_pi_w, &step_s, &step_c);",
        "        float wr = step_c;",
        "        float wi = step_s;",
    ]
    for k in range(1, 16):
        acc.append(f"        acc += 2.0f * (r{k} * wr - i{k} * wi);")
        if k < 15:
            acc.extend(
                [
                    "        {",
                    "            const float nr = wr * step_c - wi * step_s;",
                    "            const float ni = wr * step_s + wi * step_c;",
                    "            wr = nr;",
                    "            wi = ni;",
                    "        }",
                ]
            )
    acc.append(f"        spatial[out_base + w] = acc * inv_w;")
    acc_s = "\n".join(acc)
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ in_freq, float *__restrict__ spatial) {{
    __shared__ float sr[8][16];
    __shared__ float si[8][16];
    const int local_row = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row8 = static_cast<int>(blockIdx.x) * 8 + local_row;
    const int total_rows = batch_size * channels * height;
    const bool valid = row8 < total_rows;
    int h = 0;
    int ch = 0;
    int b = 0;
    int plane = 0;
    if (valid) {{
        h = row8 % height;
        int rem = row8 / height;
        ch = rem % channels;
        b = rem / channels;
        plane = ((b * channels + ch) * height) * 32;
    }}
    if (valid && lane < 16) {{
        const int m2 = lane;
        const float two_pi_h = (2.0f * M_PIf) * static_cast<float>(h) / static_cast<float>(height);
        float step_s, step_c;
        sincosf(two_pi_h, &step_s, &step_c);
        float acc_r = 0.0f;
        float acc_i = 0.0f;
        float wr = 1.0f;
        float wi = 0.0f;
#pragma unroll
        for (int m1 = 0; m1 < 16; ++m1) {{
            const int top = plane + (m1 * 16 + m2) * 2;
            const float xr = in_freq[top];
            const float xi = in_freq[top + 1];
            acc_r += xr * wr - xi * wi;
            acc_i += xr * wi + xi * wr;
            const float nr = wr * step_c - wi * step_s;
            const float ni = wr * step_s + wi * step_c;
            wr = nr;
            wi = ni;
        }}
        const int kh0 = height - 16;
        sincosf(two_pi_h * static_cast<float>(kh0), &wi, &wr);
#pragma unroll
        for (int m1 = 0; m1 < 16; ++m1) {{
            const int bot = plane + ((kh0 + m1) * 16 + m2) * 2;
            const float xr = in_freq[bot];
            const float xi = in_freq[bot + 1];
            acc_r += xr * wr - xi * wi;
            acc_i += xr * wi + xi * wr;
            const float nr = wr * step_c - wi * step_s;
            const float ni = wr * step_s + wi * step_c;
            wr = nr;
            wi = ni;
        }}
        const float inv_h = 1.0f / static_cast<float>(height);
        sr[local_row][m2] = acc_r * inv_h;
        si[local_row][m2] = acc_i * inv_h;
    }}
    __syncthreads();
    if (!valid) {{
        return;
    }}
{loads}
    const float inv_w = 1.0f / static_cast<float>({width});
    const int out_base = ((b * channels + ch) * height + h) * {width};
#pragma unroll
    for (int dw = 0; dw < {pix}; ++dw) {{
        const int w = lane * {pix} + dw;
{acc_s}
    }}
}}
"""



def emit_fused_inv_occ16(width: int) -> str:
    pix = width // 16
    name = f"pruned_irfft2_fused_occ16_w{width}_kernel"
    loads = "\n".join(
        (
            f"    const float r{k} = sr[local_row][{k}];\n    const float i{k} = si[local_row][{k}];"
            if k
            else "    const float r0 = sr[local_row][0];"
        )
        for k in range(16)
    )
    acc = [
        "        float acc = r0;",
        f"        const float two_pi_w = (2.0f * M_PIf) * static_cast<float>(w) / {width}.0f;",
        "        float step_s, step_c;",
        "        sincosf(two_pi_w, &step_s, &step_c);",
        "        float wr = step_c;",
        "        float wi = step_s;",
    ]
    for k in range(1, 16):
        acc.append(f"        acc += 2.0f * (r{k} * wr - i{k} * wi);")
        if k < 15:
            acc.extend(
                [
                    "        {",
                    "            const float nr = wr * step_c - wi * step_s;",
                    "            const float ni = wr * step_s + wi * step_c;",
                    "            wr = nr;",
                    "            wi = ni;",
                    "        }",
                ]
            )
    acc.append("        spatial[out_base + w] = acc * inv_w;")
    acc_s = "\n".join(acc)
    return f"""
__global__ void {name}(int batch_size, int channels, int height,
                       const float *__restrict__ in_freq, float *__restrict__ spatial) {{
    __shared__ float sr[16][16];
    __shared__ float si[16][16];
    const int local_row = static_cast<int>(threadIdx.x) >> 4;
    const int m2 = static_cast<int>(threadIdx.x) & 15;
    const int row16 = static_cast<int>(blockIdx.x) * 16 + local_row;
    const int total_rows = batch_size * channels * height;
    const bool valid = row16 < total_rows;
    int h = 0;
    int ch = 0;
    int b = 0;
    int plane = 0;
    if (valid) {{
        h = row16 % height;
        int rem = row16 / height;
        ch = rem % channels;
        b = rem / channels;
        plane = ((b * channels + ch) * height) * 32;
        const float two_pi_h = (2.0f * M_PIf) * static_cast<float>(h) / static_cast<float>(height);
        float step_s, step_c;
        sincosf(two_pi_h, &step_s, &step_c);
        float acc_r = 0.0f;
        float acc_i = 0.0f;
        float wr = 1.0f;
        float wi = 0.0f;
#pragma unroll
        for (int m1 = 0; m1 < 16; ++m1) {{
            const int top = plane + (m1 * 16 + m2) * 2;
            const float xr = in_freq[top];
            const float xi = in_freq[top + 1];
            acc_r += xr * wr - xi * wi;
            acc_i += xr * wi + xi * wr;
            const float nr = wr * step_c - wi * step_s;
            const float ni = wr * step_s + wi * step_c;
            wr = nr;
            wi = ni;
        }}
        const int kh0 = height - 16;
        sincosf(two_pi_h * static_cast<float>(kh0), &wi, &wr);
#pragma unroll
        for (int m1 = 0; m1 < 16; ++m1) {{
            const int bot = plane + ((kh0 + m1) * 16 + m2) * 2;
            const float xr = in_freq[bot];
            const float xi = in_freq[bot + 1];
            acc_r += xr * wr - xi * wi;
            acc_i += xr * wi + xi * wr;
            const float nr = wr * step_c - wi * step_s;
            const float ni = wr * step_s + wi * step_c;
            wr = nr;
            wi = ni;
        }}
        const float inv_h = 1.0f / static_cast<float>(height);
        sr[local_row][m2] = acc_r * inv_h;
        si[local_row][m2] = acc_i * inv_h;
    }}
    __syncthreads();
    if (!valid) {{
        return;
    }}
{loads}
    const float inv_w = 1.0f / static_cast<float>({width});
    const int out_base = ((b * channels + ch) * height + h) * {width};
#pragma unroll
    for (int dw = 0; dw < {pix}; ++dw) {{
        const int w = m2 * {pix} + dw;
{acc_s}
    }}
}}
"""


def emit_fused_row() -> str:
    # 1 row / thread, modes1=modes2=16. Named bins to avoid loc[] spills.
    assigns = []
    decls = "    float " + ", ".join(f"br{k}, bi{k}" for k in range(16)) + ";"
    calls = []
    for m2 in range(16):
        calls.append(
            f"    idft_h_m16_col(in_freq, plane, height, kh0, two_pi_h, step_c, step_s, inv_h, {m2}, br{m2}, bi{m2});"
        )
    w_acc = [
        "        const float c0 = pix_c;",
        "        const float s0 = pix_s;",
        "        const float c1 = pix_c * base_c - pix_s * base_s;",
        "        const float s1 = pix_c * base_s + pix_s * base_c;",
        "        float acc0 = br0;",
        "        float acc1 = br0;",
        "        float wr0 = c0, wi0 = s0, wr1 = c1, wi1 = s1;",
    ]
    for k in range(1, 16):
        w_acc.append(f"        acc0 += 2.0f * (br{k} * wr0 - bi{k} * wi0);")
        w_acc.append(f"        acc1 += 2.0f * (br{k} * wr1 - bi{k} * wi1);")
        if k < 15:
            w_acc.append("        {")
            w_acc.append("            const float n0r = wr0 * c0 - wi0 * s0;")
            w_acc.append("            const float n0i = wr0 * s0 + wi0 * c0;")
            w_acc.append("            wr0 = n0r; wi0 = n0i;")
            w_acc.append("            const float n1r = wr1 * c1 - wi1 * s1;")
            w_acc.append("            const float n1i = wr1 * s1 + wi1 * c1;")
            w_acc.append("            wr1 = n1r; wi1 = n1i;")
            w_acc.append("        }")
    w_acc.append("        spatial[out_base + w] = acc0 * inv_w;")
    w_acc.append("        spatial[out_base + w + 1] = acc1 * inv_w;")
    w_acc.append("        pix_c = c1 * base_c - s1 * base_s;")
    w_acc.append("        pix_s = c1 * base_s + s1 * base_c;")
    return f"""
static __device__ void idft_h_m16_col(const float *__restrict__ in_freq, int plane, int height,
                                      int kh0, float two_pi_h, float step_c, float step_s,
                                      float inv_h, int m2, float &br, float &bi) {{
    float acc_r = 0.0f;
    float acc_i = 0.0f;
    float wr = 1.0f;
    float wi = 0.0f;
#pragma unroll
    for (int m1 = 0; m1 < 16; ++m1) {{
        const int top = plane + (m1 * 16 + m2) * 2;
        const float xr = in_freq[top];
        const float xi = in_freq[top + 1];
        acc_r += xr * wr - xi * wi;
        acc_i += xr * wi + xi * wr;
        const float nr = wr * step_c - wi * step_s;
        const float ni = wr * step_s + wi * step_c;
        wr = nr;
        wi = ni;
    }}
    sincosf(two_pi_h * static_cast<float>(kh0), &wi, &wr);
#pragma unroll
    for (int m1 = 0; m1 < 16; ++m1) {{
        const int bot = plane + ((kh0 + m1) * 16 + m2) * 2;
        const float xr = in_freq[bot];
        const float xi = in_freq[bot + 1];
        acc_r += xr * wr - xi * wi;
        acc_i += xr * wi + xi * wr;
        const float nr = wr * step_c - wi * step_s;
        const float ni = wr * step_s + wi * step_c;
        wr = nr;
        wi = ni;
    }}
    br = acc_r * inv_h;
    bi = acc_i * inv_h;
}}

__global__ void pruned_irfft2_fused_row_m16_kernel(int batch_size, int channels, int height,
                                                   int width, const float *__restrict__ in_freq,
                                                   float *__restrict__ spatial) {{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = batch_size * channels * height;
    if (row >= total) {{
        return;
    }}
    const int h = row % height;
    int rem = row / height;
    const int ch = rem % channels;
    const int b = rem / channels;
    const int plane = ((b * channels + ch) * height) * 32;
    const float two_pi_h = (2.0f * M_PIf) * static_cast<float>(h) / static_cast<float>(height);
    float step_s, step_c;
    sincosf(two_pi_h, &step_s, &step_c);
    const float inv_h = 1.0f / static_cast<float>(height);
    const int kh0 = height - 16;
{decls}
{chr(10).join(calls)}
    const float inv_w = 1.0f / static_cast<float>(width);
    const int out_base = ((b * channels + ch) * height + h) * width;
    float base_s, base_c;
    sincosf((2.0f * M_PIf) / static_cast<float>(width), &base_s, &base_c);
    float pix_c = 1.0f;
    float pix_s = 0.0f;
    for (int w = 0; w < width; w += 2) {{
{chr(10).join(w_acc)}
    }}
}}
"""


def emit_launches() -> str:
    return r"""
static void launch_1d(unsigned nthreads, dim3 &block, dim3 &grid) {
    block = dim3(256);
    grid = dim3((nthreads + 255u) / 256u);
}

extern "C" suError_t launch_pruned_rfft_w_pack32(const float *x, float *row_freq,
                                                 int batch_size, int channels, int height,
                                                 int modes2, suStream_t stream) {
    const unsigned warps = static_cast<unsigned>(batch_size * channels * height);
    dim3 block(256);
    dim3 grid((warps + 7u) / 8u);
    pruned_rfft_w_pack32_warp_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, modes2, x, row_freq);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_rfft_w_geo(const float *x, float *row_freq,
                                              int batch_size, int channels, int height,
                                              int width, int modes2, suStream_t stream) {
    if (modes2 != 16 || (width != 64 && width != 128 && width != 256)) {
        return (suError_t)1;
    }
    dim3 block, grid;
    if (width == 64 && (height & 1) == 0) {
        const unsigned nthreads =
            static_cast<unsigned>(batch_size * channels * (height >> 1) * 16);
        launch_1d(nthreads, block, grid);
        pruned_rfft_w_m16_w64_row2_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, height, x, row_freq);
        return suGetLastError();
    }
    const unsigned nthreads =
        static_cast<unsigned>(batch_size * channels * height * 16);
    launch_1d(nthreads, block, grid);
    if (width == 64) {
        pruned_rfft_w_m16_w64_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, height, x, row_freq);
    } else if (width == 128) {
        pruned_rfft_w_m16_w128_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, height, x, row_freq);
    } else {
        pruned_rfft_w_m16_w256_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, height, x, row_freq);
    }
    return suGetLastError();
}

extern "C" suError_t launch_pruned_rfft_w_coop(const float *x, float *row_freq,
                                               int batch_size, int channels, int height,
                                               int width, int modes2, suStream_t stream) {
    if (modes2 != 16 || (width != 64 && width != 128 && width != 256)) {
        return (suError_t)1;
    }
    const unsigned rows = static_cast<unsigned>(batch_size * channels * height);
    dim3 block(256);
    dim3 grid((rows + 15u) / 16u);
    if (width == 64) {
        pruned_rfft_w_m16_w64_coop_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, height, x, row_freq);
    } else if (width == 128) {
        pruned_rfft_w_m16_w128_coop_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, height, x, row_freq);
    } else {
        pruned_rfft_w_m16_w256_coop_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, height, x, row_freq);
    }
    return suGetLastError();
}

extern "C" suError_t launch_pruned_fft_h_geo(const float *row_freq, float *packed,
                                             int batch_size, int channels, int height,
                                             int modes1, int modes2, suStream_t stream) {
    if (modes1 != 16 || modes2 != 16) {
        return (suError_t)1;
    }
    const unsigned nthreads =
        static_cast<unsigned>(batch_size * channels * 16 * 16);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    if (height == 64) {
        pruned_fft_h_m16_n64_dual_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, row_freq, packed);
        return suGetLastError();
    }
    if (height == 128) {
        pruned_fft_h_m16_n128_dual_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, row_freq, packed);
        return suGetLastError();
    }
    if (height == 256) {
        pruned_fft_h_m16_n256_dual_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, row_freq, packed);
        return suGetLastError();
    }
    return (suError_t)1;
}

extern "C" suError_t launch_pruned_fft_h_coop(const float *row_freq, float *packed,
                                              int batch_size, int channels, int height,
                                              int modes1, int modes2, suStream_t stream) {
    if (modes1 != 16 || modes2 != 16) {
        return (suError_t)1;
    }
    const unsigned cols = static_cast<unsigned>(batch_size * channels * 16);
    dim3 block(256);
    dim3 grid((cols + 7u) / 8u);
    if (height == 64) {
        pruned_fft_h_m16_n64_coop_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, row_freq, packed);
        return suGetLastError();
    }
    if (height == 128) {
        pruned_fft_h_m16_n128_coop_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, row_freq, packed);
        return suGetLastError();
    }
    if (height == 256) {
        pruned_fft_h_m16_n256_coop_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, row_freq, packed);
        return suGetLastError();
    }
    return (suError_t)1;
}

extern "C" suError_t launch_pruned_irfft2_fused(const float *in_freq, float *spatial,
                                                int batch_size, int channels, int height,
                                                int width, int modes1, int modes2,
                                                suStream_t stream) {
    if (modes1 != 16 || modes2 != 16) {
        return (suError_t)1;
    }
    if (width == 64) {
        const unsigned row_groups =
            (static_cast<unsigned>(batch_size * channels * height) + 15u) / 16u;
        dim3 block(256);
        dim3 grid(row_groups);
        pruned_irfft2_fused_occ16_w64_kernel<<<grid, block, 0, stream>>>(
            batch_size, channels, height, in_freq, spatial);
        return suGetLastError();
    }
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft2_fused_row_m16_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, width, in_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_ifft_h_x2_named(const float *in_freq, float *row_freq,
                                                   int batch_size, int channels, int height,
                                                   suStream_t stream) {
    const unsigned nthreads =
        static_cast<unsigned>(batch_size * channels * (height >> 1) * 16);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_ifft_h_m16_x2_named_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, in_freq, row_freq);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_x2_w128(const float *row_freq, float *spatial,
                                                   int batch_size, int channels, int height,
                                                   suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 64);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_x2_w128_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_x2_ilp_w64(const float *row_freq, float *spatial,
                                                      int batch_size, int channels, int height,
                                                      suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 32);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_x2_ilp_w64_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_x2_ilp_w128(const float *row_freq, float *spatial,
                                                       int batch_size, int channels, int height,
                                                       suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 64);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_x2_ilp_w128_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_x4_w64(const float *row_freq, float *spatial,
                                                  int batch_size, int channels, int height,
                                                  suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 16);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_x4_w64_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_x2_w256(const float *row_freq, float *spatial,
                                                   int batch_size, int channels, int height,
                                                   suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 128);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_x2_w256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_x2_ilp_w256(const float *row_freq, float *spatial,
                                                       int batch_size, int channels, int height,
                                                       suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 128);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_x2_ilp_w256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_x4_ilp_w256(const float *row_freq, float *spatial,
                                                       int batch_size, int channels, int height,
                                                       suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 64);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_x4_ilp_w256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_x4_smem_w256(const float *row_freq, float *spatial,
                                                        int batch_size, int channels, int height,
                                                        suStream_t stream) {
    const unsigned rows = static_cast<unsigned>(batch_size * channels * height);
    dim3 block(256);
    dim3 grid((rows + 3u) / 4u);
    pruned_irfft_w_m16_x4_smem_w256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_x4_stride_lut_w256(const float *row_freq, float *spatial,
                                                              int batch_size, int channels, int height,
                                                              suStream_t stream) {
    const unsigned rows = static_cast<unsigned>(batch_size * channels * height);
    dim3 block(256);
    unsigned groups = (rows + 3u) / 4u;
    unsigned grid_n = groups < 1024u ? groups : 1024u;
    if (grid_n < 1u) {
        grid_n = 1u;
    }
    dim3 grid(grid_n);
    pruned_irfft_w_m16_x4_stride_lut_w256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_x4_ilp_r2_w256(const float *row_freq, float *spatial,
                                                         int batch_size, int channels, int height,
                                                         suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 32);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_x4_ilp_r2_w256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_x4_ilp_w128(const float *row_freq, float *spatial,
                                                       int batch_size, int channels, int height,
                                                       suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 32);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_x4_ilp_w128_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_x8_ilp_w256(const float *row_freq, float *spatial,
                                                       int batch_size, int channels, int height,
                                                       suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 32);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_x8_ilp_w256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_fact16x16_w256(const float *row_freq, float *spatial,
                                                          int batch_size, int channels, int height,
                                                          suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 16);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_fact16x16_w256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_fact16x8_w128(const float *row_freq, float *spatial,
                                                         int batch_size, int channels, int height,
                                                         suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 16);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_fact16x8_w128_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}
"""


def emit_w64_launches() -> str:
    return r"""
static void launch_1d(unsigned nthreads, dim3 &block, dim3 &grid) {
    block = dim3(256);
    grid = dim3((nthreads + 255u) / 256u);
}

extern "C" suError_t launch_pruned_irfft_w_fact16x4_w64(const float *row_freq, float *spatial,
                                                        int batch_size, int channels, int height,
                                                        suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 16);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_fact16x4_w64_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_irfft_w_fact8x8_w64(const float *row_freq, float *spatial,
                                                       int batch_size, int channels, int height,
                                                       suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * height * 8);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_irfft_w_m16_fact8x8_w64_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, row_freq, spatial);
    return suGetLastError();
}
"""


def emit_hfact_launches() -> str:
    return r"""
static void launch_1d(unsigned nthreads, dim3 &block, dim3 &grid) {
    block = dim3(256);
    grid = dim3((nthreads + 255u) / 256u);
}

extern "C" suError_t launch_pruned_ifft_h_fact16x4_h64(const float *in_freq, float *row_freq,
                                                       int batch_size, int channels,
                                                       suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * 16 * 16);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_ifft_h_m16_fact16x4_h64_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, in_freq, row_freq);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_ifft_h_fact16x8_h128(const float *in_freq, float *row_freq,
                                                        int batch_size, int channels,
                                                        suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * 16 * 16);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_ifft_h_m16_fact16x8_h128_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, in_freq, row_freq);
    return suGetLastError();
}
"""


def emit_rfft_fact_launches() -> str:
    return r"""
extern "C" suError_t launch_pruned_rfft_w_fact16x8_w128(const float *x, float *row_freq,
                                                        int batch_size, int channels, int height,
                                                        suStream_t stream) {
    const unsigned rows = static_cast<unsigned>(batch_size * channels * height);
    dim3 block(128);
    dim3 grid((rows + 7u) / 8u);
    pruned_rfft_w_m16_fact16x8_w128_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, x, row_freq);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_rfft_w_fact16x16_w256(const float *x, float *row_freq,
                                                         int batch_size, int channels, int height,
                                                         suStream_t stream) {
    const unsigned rows = static_cast<unsigned>(batch_size * channels * height);
    dim3 block(128);
    dim3 grid((rows + 7u) / 8u);
    pruned_rfft_w_m16_fact16x16_w256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, x, row_freq);
    return suGetLastError();
}
"""


def emit_rfft_r4_256_launches() -> str:
    return r"""
extern "C" suError_t launch_pruned_rfft_w_fact16x16_r4_w256(const float *x, float *row_freq,
                                                            int batch_size, int channels, int height,
                                                            suStream_t stream) {
    const unsigned rows = static_cast<unsigned>(batch_size * channels * height);
    dim3 block(64);
    dim3 grid((rows + 3u) / 4u);
    pruned_rfft_w_m16_fact16x16_r4_w256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, x, row_freq);
    return suGetLastError();
}
"""


def emit_fact64_launches() -> str:
    return r"""
extern "C" suError_t launch_pruned_rfft_w_fact16x4_w64(const float *x, float *row_freq,
                                                       int batch_size, int channels, int height,
                                                       suStream_t stream) {
    const unsigned rows = static_cast<unsigned>(batch_size * channels * height);
    dim3 block(128);
    dim3 grid((rows + 7u) / 8u);
    pruned_rfft_w_m16_fact16x4_w64_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, height, x, row_freq);
    return suGetLastError();
}

extern "C" suError_t launch_pruned_fft_h_fact16x4_h64(const float *row_freq, float *packed,
                                                      int batch_size, int channels,
                                                      suStream_t stream) {
    dim3 block(256);
    dim3 grid(static_cast<unsigned>(batch_size * channels));
    pruned_fft_h_m16_fact16x4_h64_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, row_freq, packed);
    return suGetLastError();
}
"""


def emit_fft_h_fact128_launches() -> str:
    return r"""
extern "C" suError_t launch_pruned_fft_h_fact16x8_h128(const float *row_freq, float *packed,
                                                       int batch_size, int channels,
                                                       suStream_t stream) {
    dim3 block(256);
    dim3 grid(static_cast<unsigned>(batch_size * channels));
    pruned_fft_h_m16_fact16x8_h128_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, row_freq, packed);
    return suGetLastError();
}
"""


def emit_fft_h_fact256_launches() -> str:
    return r"""
extern "C" suError_t launch_pruned_fft_h_fact16x16_h256(const float *row_freq, float *packed,
                                                        int batch_size, int channels,
                                                        suStream_t stream) {
    dim3 block(128);
    dim3 grid(static_cast<unsigned>(batch_size * channels), 2u);
    pruned_fft_h_m16_fact16x16_h256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, row_freq, packed);
    return suGetLastError();
}
"""


def emit_hfact256_launches() -> str:
    return r"""
static void launch_1d(unsigned nthreads, dim3 &block, dim3 &grid) {
    block = dim3(256);
    grid = dim3((nthreads + 255u) / 256u);
}

extern "C" suError_t launch_pruned_ifft_h_fact32x8_h256(const float *in_freq, float *row_freq,
                                                        int batch_size, int channels,
                                                        suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * 16 * 32);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_ifft_h_m16_fact32x8_h256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, in_freq, row_freq);
    return suGetLastError();
}
"""


def emit_hfact256_64_launches() -> str:
    return r"""
static void launch_1d(unsigned nthreads, dim3 &block, dim3 &grid) {
    block = dim3(256);
    grid = dim3((nthreads + 255u) / 256u);
}

extern "C" suError_t launch_pruned_ifft_h_fact64x4_h256(const float *in_freq, float *row_freq,
                                                        int batch_size, int channels,
                                                        suStream_t stream) {
    const unsigned nthreads = static_cast<unsigned>(batch_size * channels * 16 * 64);
    dim3 block, grid;
    launch_1d(nthreads, block, grid);
    pruned_ifft_h_m16_fact64x4_h256_kernel<<<grid, block, 0, stream>>>(
        batch_size, channels, in_freq, row_freq);
    return suGetLastError();
}
"""


def main() -> None:
    parts = [
        SU_HEADER,
        emit_fft32(),
        emit_pack64_kernel(),
        emit_warp_pack32(),
        emit_dft_rfft(64),
        emit_dft_rfft(128, True),
        emit_dft_rfft(256, True),
        emit_dft_rfft_row2(64),
        emit_dft_rfft_row2(128, True),
        emit_dft_rfft_row2(256, True),
        emit_dft_rfft_dual(64),
        emit_dft_rfft_dual(128),
        emit_dft_rfft_dual(256),
        emit_dft_rfft_goertzel(64),
        emit_dft_rfft_goertzel(128),
        emit_dft_rfft_goertzel(256),
        emit_coop_rfft(64),
        emit_coop_rfft(128),
        emit_coop_rfft(256),
        emit_dft_fft_h(64),
        emit_dft_fft_h(128),
        emit_dft_fft_h(256),
        emit_dft_fft_h_dual(64),
        emit_dft_fft_h_dual(128),
        emit_dft_fft_h_dual(256),
        emit_dft_fft_h_dual_goertzel(64),
        emit_dft_fft_h_dual_goertzel(128),
        emit_dft_fft_h_dual_goertzel(256),
        emit_coop_fft_h(64),
        emit_coop_fft_h(128),
        emit_coop_fft_h(256),
        emit_dft_plus_regs(8),
        emit_dft_plus_regs(16),
        emit_irfft_w_fact(256, 16),
        emit_irfft_w_fact(128, 16),
        emit_irfft_w_x2(128),
        emit_irfft_w_x2_ilp(64),
        emit_irfft_w_x2_ilp(128),
        emit_irfft_w_x2_ilp(256),
        emit_irfft_w_xn_ilp(128, 4),
        emit_irfft_w_xn_ilp(256, 4),
        emit_irfft_w_smem_row(256, 4),
        emit_irfft_w_x4_stride_lut(),
        emit_irfft_w_xn_ilp(256, 4, 2),
        emit_irfft_w_xn_ilp(256, 8),
        emit_irfft_w_xn(64, 4),
        emit_irfft_w_x2(256),
        emit_ifft_h_x2_named(),
        emit_fused_inv_occ16(64),
        emit_fused_inv_occ(64),
        emit_fused_inv_occ(128),
        emit_fused_inv_occ(256),
        emit_fused_row(),
        emit_launches(),
    ]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    w64_parts = [
        SU_HEADER,
        emit_dft_plus_regs(4),
        emit_dft_plus_regs(8),
        emit_irfft_w_fact(64, 16),
        emit_irfft_w_fact(64, 8),
        emit_w64_launches(),
    ]
    OUT_W64.write_text("\n".join(w64_parts), encoding="utf-8")
    print(f"wrote {OUT_W64} ({OUT_W64.stat().st_size} bytes)")
    hfact_parts = [
        SU_HEADER,
        emit_dft_plus_regs(4),
        emit_dft_plus_regs(8),
        emit_ifft_h_fact(64),
        emit_ifft_h_fact(128),
        emit_hfact_launches(),
    ]
    OUT_HFACT.write_text("\n".join(hfact_parts), encoding="utf-8")
    print(f"wrote {OUT_HFACT} ({OUT_HFACT.stat().st_size} bytes)")
    rfact_parts = [
        SU_HEADER,
        emit_dft_regs(8, sign=-1),
        emit_dft_regs(16, sign=-1),
        emit_rfft_w_fact_smem(128),
        emit_rfft_w_fact_smem(256),
        emit_rfft_fact_launches(),
    ]
    OUT_RFACT.write_text("\n".join(rfact_parts), encoding="utf-8")
    print(f"wrote {OUT_RFACT} ({OUT_RFACT.stat().st_size} bytes)")
    ffth_parts = [
        SU_HEADER,
        emit_dft_regs(8, sign=-1),
        emit_fft_h_fact_smem(128),
        emit_fft_h_fact128_launches(),
    ]
    OUT_FFTH.write_text("\n".join(ffth_parts), encoding="utf-8")
    print(f"wrote {OUT_FFTH} ({OUT_FFTH.stat().st_size} bytes)")
    ffth256_parts = [
        SU_HEADER,
        emit_dft_regs(16, sign=-1),
        emit_fft_h_fact_smem(256),
        emit_fft_h_fact256_launches(),
    ]
    OUT_FFTH256.write_text("\n".join(ffth256_parts), encoding="utf-8")
    print(f"wrote {OUT_FFTH256} ({OUT_FFTH256.stat().st_size} bytes)")
    hfact256_parts = [
        SU_HEADER,
        emit_dft_plus_regs(8),
        emit_ifft_h_fact(256, 32),
        emit_hfact256_launches(),
    ]
    OUT_HFACT256.write_text("\n".join(hfact256_parts), encoding="utf-8")
    print(f"wrote {OUT_HFACT256} ({OUT_HFACT256.stat().st_size} bytes)")
    irfft256_smem_parts = [
        SU_HEADER,
        emit_dft_plus_regs(16),
        emit_irfft_w_fact_smem(256),
        emit_irfft_smem256_launches(),
    ]
    OUT_IRFFT256_SMEM.write_text("\n".join(irfft256_smem_parts), encoding="utf-8")
    print(f"wrote {OUT_IRFFT256_SMEM} ({OUT_IRFFT256_SMEM.stat().st_size} bytes)")
    fact64_parts = [
        SU_HEADER,
        emit_dft_regs(4, sign=-1),
        emit_rfft_w_fact_smem(64),
        emit_fft_h_fact_smem(64),
        emit_fact64_launches(),
    ]
    OUT_FACT64.write_text("\n".join(fact64_parts), encoding="utf-8")
    print(f"wrote {OUT_FACT64} ({OUT_FACT64.stat().st_size} bytes)")
    pair_parts = [
        SU_HEADER,
        emit_dft_plus_regs(16),
        emit_irfft_w_fact_vec4(256),
        emit_irfft_vec4_launches(256),
    ]
    OUT_IRFFT256_PAIR.write_text("\n".join(pair_parts), encoding="utf-8")
    print(f"wrote {OUT_IRFFT256_PAIR} ({OUT_IRFFT256_PAIR.stat().st_size} bytes)")
    v128_parts = [
        SU_HEADER,
        emit_dft_plus_regs(8),
        emit_irfft_w_fact_vec4(128),
        emit_irfft_vec4_launches(128),
    ]
    OUT_IRFFT128_VEC4.write_text("\n".join(v128_parts), encoding="utf-8")
    print(f"wrote {OUT_IRFFT128_VEC4} ({OUT_IRFFT128_VEC4.stat().st_size} bytes)")
    v64_parts = [
        SU_HEADER,
        emit_dft_plus_regs(4),
        emit_irfft_w_fact_vec4(64),
        emit_irfft_vec4_launches(64),
    ]
    OUT_IRFFT64_VEC4.write_text("\n".join(v64_parts), encoding="utf-8")
    print(f"wrote {OUT_IRFFT64_VEC4} ({OUT_IRFFT64_VEC4.stat().st_size} bytes)")
    h64_parts = [
        SU_HEADER,
        emit_dft_plus_regs(4),
        emit_ifft_h_fact(256, 64),
        emit_hfact256_64_launches(),
    ]
    OUT_HFACT256_64.write_text("\n".join(h64_parts), encoding="utf-8")
    print(f"wrote {OUT_HFACT256_64} ({OUT_HFACT256_64.stat().st_size} bytes)")
    r4_parts = [
        SU_HEADER,
        emit_dft_regs(16, sign=-1),
        emit_rfft_w_fact_smem(256, 4),
        emit_rfft_r4_256_launches(),
    ]
    OUT_RFACT256_R4.write_text("\n".join(r4_parts), encoding="utf-8")
    print(f"wrote {OUT_RFACT256_R4} ({OUT_RFACT256_R4.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
