# Spectral Convolution（必选）· 工作区 ai4s-f

FNO 核心 2D Spectral Convolution 前向：

```text
x: [B, C_in, H, W]
  → 2D rFFT（v1：CPU torch）
  → 截断低频 modes
  → 频域复数乘（SUPA kernel `spectral_mul`）
  → iFFT（v1：CPU torch）
y: [B, C_out, H, W]
```

## 验收

- 相对误差 ≤ `1e-4`（相对 `reference_pytorch.py`）
- 核心自定义计算：SUPA Extension 频域复数乘
- 优先支持 `64×64`（已含小尺寸冒烟）

## 文件

| 文件 | 作用 |
|------|------|
| `official_baseline.py` | 官网 §3.1–3.2 完整 SpectralConv2d 基准（CPU 复现 + SUPA FFT 探测） |
| `reference_pytorch.py` | 队内简化 CPU 标准答案（单角 modes） |
| `spectral_conv_ext.su` / `.cpp` | SUPA 复数乘 + suFFT 1D 拼 2D + pybind |
| `build.sh` | 链接 `libsufft`；按 GEMV Extension 方式编译 |
| `spectral_conv_ops.py` | v1 CPU FFT / v2 suFFT + SUPA mul 组装 |
| `test_accuracy.py` | v1 正确性；写入 `../results/summary.json` |
| `test_perf.py` | 自研路径性能（64/128/256）；写入 `spectral_conv.perf` |
| `test_sufft_accuracy.py` | 加强：suFFT + SUPA mul 正确性 |
| `test_sufft_perf.py` | 加强：suFFT vs v1 性能对比 |

模板来源：`/workspace/ai4s/gemv/torch_extension/`（勿改坏官方 `gemv/`）。

## 复现官网基准

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
cd /workspace/ai4s-f/submission/spectral_conv
python3 official_baseline.py
```

日志：`../results/run_logs/official_baseline_*.md`

## 编译与 SUPA 正确性 / 性能

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd /workspace/ai4s-f/submission/spectral_conv
./build.sh
python3 test_accuracy.py
python3 test_sufft_accuracy.py   # 加强路径
python3 test_perf.py
```

## 状态

- 官网基准已在 CPU 复现（正确性 + 64/128/256 性能表）。
- v1：`FFT@CPU + SUPA 复数乘`（默认正式路径）。
- 加强 v2：`suFFT(batched 1D×2 + plan 缓存) + SUPA 复数乘`；正确性已过；256 上快于 v1，小分辨率仍慢。
- 本机 `libsufft` **无** `BuildPlan2d` 导出，故用 1D plan 拼 2D。
