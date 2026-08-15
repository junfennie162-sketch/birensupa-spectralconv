# 裁剪 DFT 探针计划（2026-08-15）

> **旁注 / 未 promote。** 正式 idle ms 冻结；禁止默认 `test_perf.py`。  
> 现行主报：公开 NS64 L2 **0.035012** · Spectral **3.797 / 8.037 / 29.295 ms**。

**Goal:** 用同赛道「只算保留频点」的方向，接到咱们已有的 fused suFFT + SUPA 双角乘上；先小样本证明数学对，再看 64 档能不能快过现行 fused。

**Architecture:** 不抄对方 radix-2 整栈。分三层：① 分解 DFT 只算两个官方角；② 逆变换先 scatter+`irfft2` 保正确，再试 GEMM 逆；③ 混合路径 = 咱们 `rfft2_sufft_trunc` + 现有 mul + 裁剪逆。默认热路径不改，除非某条样本又对又快。

**Tech stack:** PyTorch einsum / `torch.fft` 作金标准；已有 `spectral_conv_ext`；BIREN `supa`。不写 formal `summary.json.spectral_conv.perf`。

## 红线

- 正确性：相对官网双角 reference ≤ **1e-4**（目标看是否接近咱们现有 2e-7）
- 正式三档 ms 不覆写
- 单卡；样本先 8/32，再官方 64；128/256 本轮不做除非 64 明显赢
- 未过线不改 `spectral_conv2d_supa` 默认

## 文件

| 文件 | 职责 |
|------|------|
| `spectral_conv/pruned_dft.py` | 分解正变换、scatter 逆、GEMM 逆（探针库） |
| `spectral_conv/probe_pruned_dft_accuracy.py` | 小样本正确性 |
| `spectral_conv/probe_pruned_dft_sample.py` | 官方 64 形状：对 + 计时（不写 formal） |
| `results/run_logs/pruned_dft_probe_2026-08-15.md` | 当次结果旁注 |

## 样本阶梯

1. **S0 角点** — 8×8 / 32×32：分解 DFT 两角 vs `torch.fft.rfft2` 切片  
2. **S1 端到端** — 同上形状：pruned vs `reference_pytorch.spectral_conv2d`  
3. **S2 逆变换** — GEMM 逆 vs scatter+`irfft2`  
4. **S3 官方 64** — `B=4,Cin=32,Cout=64,modes=16` 正确性  
5. **S4 计时** — 同形状，warmup 5 / iters 20：现行 fused vs pruned(CPU/SUPA) vs 混合（trunc rfft + mul + 裁剪逆）

S4 只有「相对 fused 更快且 S3 过线」才考虑改 dispatch；否则记 No-Go，热路径不动。

## 已有基础（不要重复造）

列截断 C2C（`rfft2_sufft_trunc`）已在 fused 里：64 档曾经更慢，128/256 有收益。对方赢在 **不用 suFFT、小 FFT 融在 kernel 里**。咱们第一枪是分解 DFT / 换掉 C2R，不是重写一套 radix-2。

## 本轮结果（2026-08-15 样本）

S0–S3 **PASS**（最差 e2e 约 6e-6）。S4 官方 64：fused **3.797 ms** 最快；pruned/混合慢 6–13×。**No-Go**，不改 `spectral_conv2d_supa` 默认。详见 [`pruned_dft_probe_2026-08-15.md`](pruned_dft_probe_2026-08-15.md)。
