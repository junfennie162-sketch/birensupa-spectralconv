# 测试结果

## 环境

- Docker / SDK：`/usr/local/birensupa/sdk/1.11.0.0.rc2`
- SUPA_BASE：`/usr/local/birensupa/sdk/1.11.0.0.rc2`
- Python / torch / torch_br：`torch 2.9.0+cu128` + `torch_br`（`device=supa`）
- 运行设备：BIREN 单卡 Biren106B（`brsmi` 可见；`torch.cuda.is_available()` = False 属预期）
- 基线日期：2026-07-21
- 机器摘要：`results/summary.json` → `env` / `gemv_baseline`

### 环境各组件 → 后续用途（Agent 必读）

| 组件 | 实测结论 | 后续能做什么 |
|------|----------|----------------|
| `brsw_set_env.sh` + SDK | 可正确 source | 编译/链接 SUPA 与 `torch_br` 全套依赖 |
| `brcc` | 可用 | 写并编译 SpectralConv `.su` |
| `brsmi` / Biren106B | 单卡可用 | 正确性/性能与提交用运行日志 |
| `torch` + `torch_br` / `supa` | 张量可上 `supa:0` | Extension 封装、FNO 组装、CPU reference 对比 |
| GEMV 方式一 accuracy | 3/3 通过 | SUPA/C++ 直连链路可用（备选） |
| GEMV 方式二 Extension | `ok=True` | **本队正式路线模板已通**，迁改 SpectralConv |

说明：以上验证的是**服务器竞赛环境**，不是 SpectralConv/FNO 作品本身。

## 编译

- 命令：`cd submission/spectral_conv && ./build.sh`（`ai4s-f`）
- 结果：已生成 `spectral_conv_ext*.so`（对照 GEMV Extension 链接方式）

## 正确性

### 官方基准 · GEMV（环境冒烟）

- 方式一：`make run-accuracy` → `accuracy_ok: true`（passed 3/3）
- 方式二：`torch_extension` → `{'task': 'gemv_supa_pytorch_extension', 'ok': True}`
- 结论：环境门槛已过，可进入必选算子开发

### 必选 · Spectral Convolution（ai4s-f · 2026-07-21）

- 命令：`./build.sh && python3 test_accuracy.py`
- Reference：`reference_pytorch.py`（CPU rFFT + einsum + iFFT）
- SUPA 核心：`spectral_mul` 频域复数乘（`.su`）；v1 FFT 仍在 CPU，后续可迁 suFFT
- 输出：worst Frobenius 相对误差 ≈ `1.24e-7`（阈值 ≤ `1e-4`）
- 用例：`8×8` / `32×32` / `64×64` 均 `ok=True`
- 日志：`results/run_logs/spectral_accuracy_2026-07-21.md`
- 结论：**正确性通过**

### 进阶 · FNO-Navier-Stokes

- 命令：`cd submission/fno_ns && python3 test_forward.py && python3 visualize.py`
- 模型：`model.py`（4 层 Fourier Layer；双角 SpectralConv 与必选共用 fused API）
- 数据：离线 NS-like 64×64（`dataset.py` / `generated_ns_like_v1e-3`；外网不可用时的 FNO 风格替代）
- 训练：256 样本 × 40 epoch（CPU torch 可微）；SUPA 评估走 fused，跑满 test loader
- 指标：相对 L2（SUPA）≈ **0.0173**（相对初版 0.0506 / 加强版 0.0359 明显下降）
- 可视化路径：`results/figures/fno_ns_pred_vs_gt_2026-07-22.png`
- 日志：`results/run_logs/fno_forward_2026-07-22.md`
- 结论：**加长训达标**；公开 HDF5/HF 有网后可替换数据再训

### 扩展 · spectral_mul 反向（P4）

- 命令：`python3 test_backward.py`
- 实现：`SpectralMulFunction`（SUPA 前向 + 共轭转置 einsum 反向）
- 指标：worst grad 相对误差 ≈ **6.25e-8**（阈值 1e-4）
- 日志：`results/run_logs/spectral_backward_2026-07-22.md`

### 扩展 · SpectralConv3d（P6）

- 命令：`python3 test_3d_accuracy.py`
- 路径：CPU `rfftn` + SUPA `spectral_mul`（modes 维 reshape 复用 2D kernel）
- 指标：worst_rel ≈ **1.07e-7**
- 日志：`results/run_logs/spectral_3d_accuracy_2026-07-22.md`
- 说明：仅算子前向扩展；未做完整 3D FNO

## 性能（正式路径 · 分辨率自适应 · 2026-07-22 晚）

- **正式路径 `use_sufft=\"auto\"`**：`min(H,W)<256` → v1；`>=256` → fused（SOL 严格测法下小图 fused 更慢，故分流）
- 交叉评测：`python3 test_sol_style_perf.py`（warmup=10 / iters=50 / trials=3 / 每轮 clone；**median**）

| 分辨率 | v1 median | fused median | **auto 正式** | auto vs v1 |
|---|---:|---:|---:|---:|
| 64×64 | 17.9 ms | 45.9 ms | **17.9 ms** | ≈1.0× |
| 128×128 | 23.8 ms | 46.2 ms | **28.0 ms** | ≈0.85×（方差内等同 v1） |
| 256×256 | 268.0 ms | 61.4 ms | **63.9 ms** | **≈4.2×** |

旧 P3 冻结表（同输入热路径、偏乐观）仍见 `opt_perf_freeze_2026-07-22.md`；对外讲性能以 **auto + SOL 风格** 为准。

### SOL-ExecBench 风格交叉评测（优化决策用）

- 参考：[NVIDIA/SOL-ExecBench](https://github.com/nvidia/sol-execbench)（正确性优先、warmup=10 / iters=50 / trials=3、每轮 clone 输入、相对固定 baseline 的 gap）
- 命令：`cd spectral_conv && python3 test_sol_style_perf.py` 或 `./scripts/run_tests.sh sol-perf`
- 说明：Biren 无 SOLAR/B200 理论 SOL 界，报告 **proxy_sol_score**（相对队内 v1 baseline + 手册叙述 ref）；官网正式表仍以 §3.2 iters=100 为准
- 日志：`results/run_logs/spectral_sol_style_perf_*.md`

（GEMV Extension 顺带：`perf_4096x1024` avg ≈ 2.94 ms，仅作环境参考）

## 可视化或附加产物

- 图片 / 日志 / checkpoint：`results/`
- FNO 对比图：`results/figures/fno_ns_pred_vs_gt_2026-07-21.png`
- 环境冒烟日志：`results/run_logs/env_baseline_2026-07-21.md`

## 已知限制

- **正式路径**为 fused（`use_sufft=True`）；v1（CPU FFT）保留为对照与可微训练后备。
- 本 SDK suFFT 仅导出 1D plan，2D 由两次 1D + plan 缓存拼成；fused 正确性 worst_rel ≈ `2.12e-7`（单角）/ `2.16e-7`（双角）。
- FNO 当前用离线 NS-like 64×64；公开 NS HDF5 有网后替换。`ai4s-n` 勿与本区并发占 GPU。
