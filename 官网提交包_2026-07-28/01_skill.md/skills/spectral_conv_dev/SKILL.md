# Skill: spectral_conv_dev

在 BIREN 上开发 FNO Spectral Convolution（混合路线）。

## 何时使用

- 实现/调试 SUPA `spectral_mul`
- 与官网双角 PyTorch reference 对比（rel ≤ 1e-4）
- 将算子接入 FNO Fourier Layer

## 步骤摘要

见仓库根提交包 [`../skill.md`](../skill.md)。

## 注意

- 不要用 `torch.fft` 直接跑在 `supa` 做正确性
- 方式一与 Extension **共用同一算法的 `.su` kernel**
