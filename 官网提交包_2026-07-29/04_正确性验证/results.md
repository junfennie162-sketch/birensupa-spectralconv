# 测试结果

## 1. 环境

- SDK：`/usr/local/birensupa/sdk/1.11.0.0.rc2`
- PyTorch：`2.9.0+cu128` + `torch_br`
- 设备：BIREN 单卡 Biren106B，PyTorch 设备名 `supa`
- `torch.cuda.is_available() == False` 属预期
- 每个新终端先执行：

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
```

## 2. 必选 Spectral Convolution

### 2.1 实现

正式实现位于 `spectral_conv/`：

- `spectral_conv_ext.su`：SUPA 频域复数乘 kernel
- `spectral_conv_ext.cpp`：PyTorch Extension、suFFT 1D plan 拼接、plan/workspace cache、`spectral_mul_out` / `dual_out`
- `spectral_conv_ops.py`：CPU FFT v1 与 suFFT fused 双路径，`min(H,W)>=64` 时 auto 选择 fused
- Reference：`reference_pytorch.py`

### 2.2 正确性

2026-07-25 只读同协议复测，误差采用 Frobenius 相对误差：

| case | shape | modes | relative error | 结果 |
|---|---|---|---:|---|
| tiny_8x8 | B2/Cin2/Cout3/8×8 | 2×2 | 4.675e-8 | PASS |
| small_32x32 | B2/Cin4/Cout4/32×32 | 8×8 | 1.137e-7 | PASS |
| target_64x64 | B2/Cin4/Cout4/64×64 | 12×12 | 2.170e-7 | PASS |
| official_128 | B4/Cin32/Cout64/128×128 | 16×16 | 2.714e-7 | PASS |
| official_256_fused | B4/Cin32/Cout64/256×256 | 16×16 | 2.846e-7 | PASS |

- worst relative error：`2.846e-7`
- 阈值：`1e-4`
- 结论：5/5 PASS

### 2.3 性能

2026-07-29 正式复测（R11 + P4 packed trunc + P7 dual-launch + P8b packed scale）：`B=4, Cin=32, Cout=64, modes=16×16`，`use_sufft="auto"`，warmup=10，iters=100，同步 wall-clock，计时范围包含 CPU 输入到 CPU 输出。

| 分辨率 | forward ms | peak MB |
|---|---:|---:|
| 64×64 | 3.807 | 225.3 |
| 128×128 | 8.001 | 253.3 |
| 256×256 | 29.162 | 353.3 |

脚本：`spectral_conv/test_perf.py`。历史 v1/fused/SOL-style 结果保留在 `results/summary.json` 的 legacy/optimization 字段和 `results/run_logs/`，不再作为主表。

### 2.4 扩展

| 项目 | 结果 |
|---|---|
| Backward | 3/3 PASS，worst relative error `6.253e-8` |
| SpectralConv3d | 2/2 PASS，worst relative error `1.073e-7` |
| irregular shapes | 9/9 PASS，worst relative error `3.202e-7` |
| suFFT 独立验证 | 有独立 accuracy / perf 脚本 |
| SOL-style proxy | warmup=10、iters=50、trials=3；只作为队内 proxy，不冒充官方理论 SOL |

## 3. 进阶 FNO-Navier-Stokes

### 3.1 模型

- 4 层 Fourier Layer
- width：32
- modes：16×16
- 输入：10 个 64×64 涡度时间步
- 输出：1 个 64×64 涡度时间步
- 每层复用必选 SpectralConv Extension
- checkpoint：`fno_ns/checkpoints/fno_ns_demo.pt`（约 17 MB）

### 3.2 数据披露

正式结果使用 `generated_ns_like_v2`，这是项目内可复现的自生成 NS-like 数据，不是公开 NS64 benchmark：

- 总样本：1024
- 训练/验证/测试：768 / 0 / 128
- 时间步：30
- 分辨率：64×64
- 粘度：`1e-3`
- seed：`20260722`

完整说明见 `results/data_disclosure.md`。当前 `0.008768` 只表述为项目内 NS-like v2 指标。

### 3.3 训练量与 L2

- 训练 history：150 epoch（110 主训 + R7 侧车 40 epoch，`lr=5e-5` cosine）
- batch size：8
- 等价 optimizer step：`150 × ceil(768/8) = 14400`
- 达到最新评测标准的“≥6000 step 或 ≥100 epoch”推荐训练档
- Torch 测试 relative L2：`0.008534692373359576`（2026-07-29 sidecar promote）
- 先前正式 L2：`0.008768`（已备份为 promote 前 demo）
- 更早备份：`fno_ns_demo.pt.pre_r7_backup`（promote 时覆盖为上一版 demo）

现有 checkpoint 可复评；主训未保存 optimizer/scheduler state。R7 侧车细调后 test L2 优于当时 demo 才覆盖。

### 3.4 FNO chain 状态

2026-07-26 正式门禁（R7 host-seeded D2D 物化 + promote 后 ckpt）：

- checkpoint chain vs CPU：相对误差 **4.824e-5**，通过 `1e-4`
- 随机模型：**6.580e-5**，通过
- 历史诊断：修复前曾出现 rel=`0.01655` / 未过门禁的 16.112 ms，**不得**作为正式性能

脚本：`fno_ns/test_chain_cpu_supa_consistency.py`。

### 3.5 FNO batch=16 性能

2026-07-26 正式 benchmark（R7 host-seeded D2D）：BIREN 单卡、64×64、batch=16、warmup=10、iters=50；测速前 checkpoint chain 相对 CPU 为 `4.800e-5`，通过 `1e-4`。

| scope | grid_points/s | samples/s | ms/sample | ms/batch | peak MB |
|---|---:|---:|---:|---:|---:|
| pure forward | 1,366,849.278 | 333.703 | 2.996673 | 47.947 | 171.6 |
| with DataLoader | 1,330,242.866 | 324.766 | 3.079137 | 49.266 | 171.6 |

脚本：`fno_ns/benchmark_fno_batch16.py`；日志：`results/run_logs/fno_batch16_benchmark_2026-07-25.md`（同文件覆盖最新正式数）。grid point 按单通道 `H×W` 计算。

### 3.5b 训练吞吐（加分项）

同量纲 `grid_points/s`，**明确包含** forward + relative-L2 loss + backward + Adam step。路径与提交 checkpoint 一致：CPU / `use_supa=False`。

| metric | value |
|---|---:|
| grid_points/s | 34,711.585 |
| samples/s | 8.475 |
| ms/sample | 118.001 |
| ms/batch (step) | 944.008 |

脚本：`fno_ns/benchmark_train_throughput.py`；日志：`results/run_logs/fno_train_throughput_2026-07-25.md`。这不是推理 batch=16 主表。

### 3.6 可视化

`fno_ns/visualize.py` 生成：

1. 主图：Input / GT / Pred / `|Error|` / 相对误差图（Pred/GT 共用对称色标）
2. 多样本条带：best / median / worst（按 sample relative L2）

图注含 `data`、`sample`、`target_t`、`sample_rel_L2`。实体写入 `results/figures/` 并同步 `demo/media/`。

## 4. Agent / Skill

- `development_log.md`：≥23 段有效记录，覆盖 kernel、性能、模型/超参、数据、可视化和 BIREN 平台适配
- `skill.md`：SpectralConv + FNO 工作流及性能方法论
- `skills/spectral_chain_optimization.md`：R3–R7 技术沉淀
- `skills/fno_eval_protocol.md`：batch16 / chain 门禁 / 数据披露 / 训练吞吐
- `skills/sol_gap_analysis.md` + `spectral_conv/bench_sol_proxy.py`：本地 SOL-style 差距分析（非官方 SOL）
- 自动调优：`spectral_conv/tune.py` 已落地，`tune_results.json` 可跨进程加载

## 5. 已知限制

- FNO 数据为自生成 NS-like v2，不是公开 NS64 benchmark。
- FNO chain 的 SUPA-resident 输入需 correctness fallback；未通过一致性门禁的快速结果不作为正式性能。
- SDK 仅导出 suFFT 1D plan，2D 由两次 1D + transpose 拼接；没有可用的 `sufftBuildPlan2d/Many` ABI。
- SpectralConv3d 是算子扩展，不是完整 3D FNO。
- 单卡 GPU 禁止 f/n 并发测试。
