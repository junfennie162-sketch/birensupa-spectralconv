# 工程融合 vs 真融合 · 诚实对照卡

> 评审易抓「假融合」。本卡固定口径：我们做的是 **设备常驻链路 + 稀疏 scatter 逻辑融合**，不是 cuFFT-callback 式 FFT⊗mul 真融合。

## KEEP（已落地）

| 项 | 含义 |
|----|------|
| suFFT R2C/C2R 设备常驻 | 频谱不落 CPU 桥接 mul |
| dual_scatter / dual_out | 零填 + 双角 mul/scatter 合一 |
| packed trunc / 列截断 | 只搬/算低频有效区 |
| SUPA spectral_mul | 复数矩阵乘在 Biren kernel |

## ABORT / 永久 No-Go

| 项 | 原因 |
|----|------|
| Plan2d / PlanMany / BuildPlan3d | `libsufft` 无导出 / stub |
| NVIDIA TurboFNO 真融合热路径 | 与 Biren suFFT 硬冲突 |
| `torch.fft@SUPA` 热路径 | 正确性/平台风险 |
| strided pack CopyD2D 跳 permute | ABI 不支持 |
| R13 ping-pong / R14 fused IN+GELU | 已否决微 opt |

## 评委金句

「mul 不是墙；C2R 是墙。我们融合的是链路与稀疏写回，不是改写 FFT 库。」
