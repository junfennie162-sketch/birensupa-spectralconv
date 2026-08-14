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
| 64×64 | 3.811 | 225.3 |
| 128×128 | 8.054 | 253.3 |
| 256×256 | 29.560 | 353.3 |

主表对齐 `summary.json` → `spectral_conv.perf`（idle recheck 2026-07-31T02:54:15Z；与 P8b 3.807/8.001/29.162 噪声内一致）。脚本：`spectral_conv/test_perf.py`。历史 v1/fused/SOL-style 结果保留在 legacy/optimization 字段与 `results/run_logs/`，不再作为主表。

相对官网 CPU 参考（`official_baseline`，本机）：约 **19.5× / 11.1× / 10.0×** @64/128/256（非竞品 GPU / 非 SOL）。fused 分段旁注见 `results/run_logs/spectral_fused_segments_2026-08-01.md`（C2R 为主墙；**未改** formal 主表）。

### 2.4 扩展

| 项目 | 结果 |
|---|---|
| Backward | 3/3 PASS，worst relative error `6.253e-8` |
| SpectralConv3d | 2/2 PASS（官网四角），worst relative error `≈1.19e-7` |
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
- 主报 checkpoint：`fno_ns/checkpoints/fno_ns_public_demo.pt`（约 17 MB）
- 旁注 checkpoint：`fno_ns/checkpoints/fno_ns_demo.pt`（自建 v2，非公开分）

### 3.2 数据披露

**正式主报使用公开 NS64**（`navier_stokes_v1e-3_N1200_T20.pt`，1000/128）。自建 `generated_ns_like_v2` 仅为工程对照，见 `results/data_disclosure.md` 附录。

### 3.3 训练量与 L2（公开 NS64 为主）

**正式公开集成绩（2026-08-06 · dualview_r2 · 评测报告 v9）**

| 项 | 值 |
|---|---|
| 数据 | `fno_ns/data/navier_stokes_v1e-3_N1200_T20.pt`（HF 公开 NS64） |
| 划分 | n_train=1000 / n_test=128，seed=`20260722` |
| 训练 | … → freeze_r9/r11 → long_push（qt/thaw/dualview）→ **dualview_r2** |
| test relative L2 | **0.03511497611179948**（`dualview_r2` promote） |
| checkpoint | `fno_ns/checkpoints/fno_ns_public_demo.pt` |

对照（勿与公开分混报）：

| 场景 | relative L2 | 说明 |
|---|---:|---|
| 公开集主报（dualview_r2） | **0.035115** | 正式公开成绩 · v9 |
| freeze_r9（历史 v8） | 0.035302 | 上一正式版本 |
| sched_samp_r5 | 0.035725 | 历史主报 · v7 |
| sched_samp_r3 | 0.035855 | 历史主报 |
| sched_samp_r2 | 0.036092 | 历史主报 |
| multistep_probe | 0.036576 | 历史主报 |
| sq3b_freeze | 0.037520 | 历史主报 |
| boostC / boostA | 0.037820 / 0.039612 | 轨迹中间点 |
| 公开集 continue 基线 | 0.041835 | 历史中间点 |
| continue3 零样本→公开集 | 0.411508 | 未重训，域偏移 |
| 自建 v2 continue3（1000/128） | 0.005144 | 工程对照，非公开集 |

- 自建 v2 历史：150 epoch / 14400 step（768 划分）等见 `data_disclosure.md` 与归档包
- Spectral idle **3.811 / 8.054 / 29.560 ms**（与 NS 数据无关）
- 可视化：`results/figures/fno_ns_pred_vs_gt_2026-08-02.png` + sample_strip（对齐 public demo；字段见 `summary.fno_ns.visualization`）

公开集 checkpoint 可复评；一键链：`scripts/run_public_ns64_autochain.sh`。

### 3.4 FNO chain 状态

2026-07-26 正式门禁（R7 host-seeded D2D 物化 + promote 后 ckpt）：

- checkpoint chain vs CPU：相对误差 **4.758e-5**，通过 `1e-4`
- 随机模型：**6.580e-5**，通过
- 历史诊断：修复前曾出现 rel=`0.01655` / 未过门禁的 16.112 ms，**不得**作为正式性能

脚本：`fno_ns/test_chain_cpu_supa_consistency.py`。

### 3.5 FNO batch=16 性能

2026-08-03 协议合规复测（freeze_r9 ckpt）：公开 NS64（1000/128）+ `fno_ns_public_demo.pt`；BIREN 单卡、64×64、batch=16、warmup=10、iters=50；chain 门禁 B=4 rel≈`8.80e-5` PASS；B=16 旁注 rel≈`9.55e-5` PASS。

| scope | grid_points/s | samples/s | ms/sample | ms/batch | peak MB |
|---|---:|---:|---:|---:|---:|
| pure forward | 1,600,295.313 | 390.697 | 2.559528 | 40.952 | 202.2 |
| with DataLoader | 1,439,739.645 | 351.499 | 2.844959 | 45.519 | 202.2 |

脚本：`fno_ns/benchmark_fno_batch16.py`（默认 public）；日志：`results/run_logs/fno_batch16_benchmark_public_ns64_2026-08-03.md`。历史 v2 旁注可用 `--legacy-v2`。grid point 按单通道 `H×W` 计算。

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

- `development_log.md`：≥34 段编号记录（精品抽查 24–34），覆盖 kernel、性能、模型/超参、数据、可视化和 BIREN 平台适配
- `skill.md`：SpectralConv + FNO 工作流及性能方法论
- `skills/spectral_chain_optimization.md`：R3–R7 技术沉淀
- `skills/fno_eval_protocol.md`：batch16 / chain 门禁 / 数据披露 / 训练吞吐
- `skills/sol_gap_analysis.md` + `spectral_conv/bench_sol_proxy.py`：本地 SOL-style 差距分析（非官方 SOL）
- 自动调优：`spectral_conv/tune.py` 已落地，`tune_results.json` 可跨进程加载

## 5. 已知限制

- FNO **精度主报**已切换公开 NS64；自建 v2 数字仅旁注，禁止混报。
- FNO chain 的 SUPA-resident 输入需 correctness fallback；未通过一致性门禁的快速结果不作为正式性能。
- SDK 仅导出 suFFT 1D plan，2D 由两次 1D + transpose 拼接；没有可用的 `sufftBuildPlan2d/Many` ABI；Spectral ms 已冻结（见 OPT_MASTER_PLAN）。
- SpectralConv3d 是算子扩展，不是完整 3D FNO。
- 单卡 GPU 禁止 f/n 并发测试；正式 perf 禁止与重训争用。
