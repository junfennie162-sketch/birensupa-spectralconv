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

## 状态（2026-08-14）

- 官网基准已在 CPU 复现（正确性 + 64/128/256 性能表）。
- **正式热路径**：`use_sufft="auto"` → fused suFFT（1D×2）+ SUPA gather-scatter mul + trunc/pack（P2–P8b）。
- **提交主表（idle）**：**3.797 / 8.037 / 29.295 ms** @64/128/256（08-14 复测；07-31 板 3.811/8.054/29.560）；worst rel ≈ 2.17e-7。
- v1（CPU FFT + SUPA mul）保留作对照/可微训练路径，**不是**正式性能主表。
- 本机 `libsufft` **无** `BuildPlan2d`/`PlanMany` 导出；Plan2d / strided pack / `torch.fft@SUPA` **已封死**。
- Spectral formal ms：**冻结**（见 `results/run_logs/OPT_MASTER_PLAN_2026-07-31.md`）；仅允许 idle 护栏复测。
