# Spectral Convolution（必选）

FNO 核心 2D Spectral Convolution。默认热路径是 **裁剪 DFT**（只变换保留的低频双角），不是整幅厂商 FFT。

```text
x: [B, C_in, H, W]
  → 宽度混合基 rFFT（只算 modes2 个 bin）
  → 高度两角 DFT（只算顶/底 modes1）
  → 频域复数乘（SUPA）
  → 逆高度 + 逆宽度裁剪 iFFT
y: [B, C_out, H, W]
```

`SPECTRAL_PRUNED_FFT=0 SPECTRAL_PRUNED_INV=0` 可退回 suFFT trunc。

## 验收

- 相对误差 ≤ `1e-4`（相对 `reference_pytorch.py`）
- 官方三案 + 性能 64/128/256（`B=4, Cin=32, Cout=64, modes=16`）

## 一键复现（不写正式 idle）

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd /workspace/ai4s-f/submission
./scripts/reproduce.sh
```

只编译并跑 `probe_pruned_continue.py`。正式 idle 仍以 `results/summary.json` 为准，需空闲独占再跑 `python3 test_perf.py`。

## 文件（现行）

| 文件 | 作用 |
|------|------|
| `spectral_conv_ops.py` | 组装：裁剪 DFT / suFFT 回退 + SUPA mul |
| `spectral_conv_ext.cpp` / `.su` | pybind、workspace、复数乘 |
| `pruned_*.su` | 热路径 kernel（混合基 rfft / fft_h / ifft_h / irfft） |
| `build.sh` | 编译并链接 `spectral_conv_ext*.so` |
| `reference_pytorch.py` | CPU 标准答案 |
| `test_accuracy.py` | 官方三案正确性 |
| `test_perf.py` | 正式 64/128/256 idle（会写 summary） |
| `probe_pruned_continue.py` | 非正式复现：三案 + warmup=10/iters=100 |
| `test_backward.py` / `test_3d_accuracy.py` / `test_irregular_shapes.py` | 加分项 |
| `tune.py` | auto 路径决策（不是得分句） |

## 状态

- 主表：**0.764 / 1.827 / 6.504 ms**（`pipe_b_r1` · v12 · 2026-08-16）
- 上一主表：0.762 / 1.981 / 7.324 ms（`fused_keep_r1` · v11）；再上一板 0.961 / 2.207 / 7.870
- worst rel（默认裁剪路径官方三案）**7.16e-6**
