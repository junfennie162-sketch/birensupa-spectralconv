# P3 正式性能对比表冻结（2026-07-22）

## 正式路径

- **正式提交路径**：`spectral_conv2d_fused`（suFFT R2C → SUPA 角点切片 → `spectral_mul` 留 SUPA → suFFT C2R）
- **对照路径**：v1 CPU rFFT + 桥接 `spectral_mul_supa` + CPU iFFT
- **配置**：官网 §3.2（B=4, C_in=32, C_out=64, modes=16×16, warmup=10, iters=100）

## 四列对比（同机 Biren106B）

| 分辨率 | 官网 PyTorch-on-SUPA* | 旧 v1 (P0 冻结) | 旧 suFFT+桥接 (P0) | **正式 fused (P3)** |
|---|---:|---:|---:|---:|
| 64×64 | ~5.3 ms | 14.064 ms | ~49 ms | **6.252 ms** |
| 128×128 | ~8.7 ms | 23.033 ms | ~49 ms | **15.944 ms** |
| 256×256 | ~27.7 ms | 214.949 ms | ~120 ms | **87.605 ms** |

\*官网基线数字来自选手手册/本机早期 `official_baseline` 对照叙述；本机 CPU 跑官方脚本见 `official_baseline_2026-07-21.md`。

同日对照重测（`test_sufft_perf.py` 内 v1 列）：64/128/256 ≈ 15.9 / 82.0 / 254.9 ms（v1）；fused 如上。

## 相对 P0 旧 suFFT 桥接路径

| 分辨率 | P0 旧 suFFT | P3 fused | 改善 |
|---|---:|---:|---:|
| 64×64 | ~49 ms | 6.25 ms | ~87% ↓ |
| 128×128 | ~49 ms | 15.94 ms | ~67% ↓ |
| 256×256 | ~120 ms | 87.61 ms | ~27% ↓ |

相对旧 v1：三档均更快；64 已接近官网量级。

## 微优化停止条件

- 本轮未再做低于 5% ROI 的微改（plan 缓存已在、mul 前后无整谱 D2H）。
- **性能叙事冻结**；后续只在新 phase 单目标再压。

## 命令

```bash
cd submission/spectral_conv
python3 test_sufft_perf.py   # rows_sufft = formal fused
python3 test_perf.py         # v1 对照
```
