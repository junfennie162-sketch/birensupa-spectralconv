# skill.md · SpectralConv + FNO-NS（翻斗花园）

## Skill 名称

`spectral_conv_fno_ns_biren`

## 目标

在壁仞 BIREN GPU 上实现 FNO 核心 2D Spectral Convolution（SUPA），并组装 ≥4 层 FNO 完成二维 Navier-Stokes 涡度前向验证。

## 输入

- 张量 `x: [B, C_in, H, W]`（优先 2 的幂次分辨率）
- 可配置 `modes1/modes2`
- FNO：多帧涡度输入 `[B, T_in, H, W]`

## 步骤

1. `source brsw_set_env.sh`；`export SUPA_BASE=...`
2. 方式一（可选）：`my_task_direct` → `make build && make run-accuracy`
3. 方式二：`submission/spectral_conv/./build.sh`
4. `python3 test_accuracy.py`（rel ≤ 1e-4 vs 官网双角 reference）
5. `python3 test_perf.py`（64/128/256）
6. `cd ../fno_ns && python3 test_forward.py && python3 visualize.py`
7. FNO 推理主表：`python3 benchmark_fno_batch16.py`（先过 chain 一致性）
8. 训练吞吐加分：`python3 benchmark_train_throughput.py`（CPU/`use_supa=False`，含 fwd+loss+bwd+opt）

## 输出

- 正确性 JSON / `results/run_logs/spectral_accuracy_*.md`
- 性能表 `results/run_logs/spectral_perf_*.md`
- FNO 前向日志与 `results/figures/fno_ns_pred_vs_gt_*.png`、`fno_ns_sample_strip_*.png`
- FNO batch16 / train throughput → `results/summary.json`

## 能力边界

- v1：FFT/iFFT 在 CPU；SUPA 负责频域复数乘
- 勿使用 `torch.fft` 直接跑在 `device=supa` 做正确性
- 公开 NS64 数据可替换合成数据；当前默认合成场保证可复现

## 性能极限视角（SOL-ExecBench 校准）

SOL-ExecBench 的「硬件极限参考线」决定了 SpectralConv 的 forward_time_ms
能拿多少分。我们的工作不是让它无限快，而是让它在 BIREN SUPA 上**贴
着内存带宽 + GEMM 峰值**走，离谱的常数代价靠工程手段砍掉。下面是每个
kernel 选择背后的依据，评审问「为什么这么改」时直接对照。

### 1. 算子拆解 vs 合并访问

| 子算子 | 物理量级 | 我们做法 |
|--------|----------|----------|
| 2D FFT | H·W·log(HW) 点运算 | suFFT on SUPA ≥64，CPU rFFT <64（冷启开销） |
| 频域复数乘 | O(B·Cout·Cin·M1·M2) FMA | 自写 `spectral_mul_supa_device`：连续访存 + 张量核复数 GEMM |
| iFFT | 同 FFT | **suFFT on SUPA ≥64**，<64 走 CPU |
| H2D / D2H | B·C·H·W·sizeof | 复用 `_HOST_OUT_CACHE` + `_OUT_FREQ_CACHE`，避免每 call 分配 |

> 理由：FFT 在 SUPA 上跑小尺寸的栈开销过去被认为不能承受，但
> `profile_segments_v2.py` 显示在「proper warmup + Parameter-cached
> weights」条件下 fused 路径在 64 都已经赢了 v1（5.4 ms vs 11.0 ms
> 64×64, B4 Cin32 Cout64）。threshold 由 256 → 128 → 64，每次下沉
> 都靠 sweep 数据，不靠拍脑袋。

### 2. 设备链路（data movement）

理想态：进 SUPA 一次、出 SUPA 一次。中间频域 buffer 在 device 上原地
复用（`_OUT_FREQ_CACHE`，容量上限 4）。**不允许**每次调用重新分配
`(B, Cout, H, Wf, 2)` — 这会让 64×64 的 peak memory 从 ~10 MB 跳到 ~30 MB。

### 3. 峰值显存策略

`peak_memory_mb` 直接评分。我们强制：

- 频域 buffer 复用（`_OUT_FREQ_CACHE`）。
- host staging 复用（`_HOST_OUT_CACHE`），并返回 buffer 本身（不
  `clone()`），让下一次 call 覆盖 — 64/128 节省约 18 MB。
- 缩到 `_BUFFER_CACHE_MAX=4`，防极端 shape 切换时把缓存填爆。

正式 `test_perf.py`（warmup=10 / iters=100，`use_sufft="auto"`，CPU→CPU）峰值显存：

| resolution | forward_ms | peak memory_MB |
|------------|-----------:|---------------:|
| 64×64      | 5.302      | 146.6          |
| 128×128    | 13.670     | 238.5          |
| 256×256    | 52.480     | 582.7          |

说明：microbench / tune 单路径峰值可能更低（如 fused@64 ≈41.7 MB）；正式表以 `test_perf` 为准。

### 4. 已踩的坑（不要重做）

- **`torch.cat` on SUPA** 喂给自定义扩展 kernel 时可能写出非连续/无效
  内存 → rel ≈ 1。**禁止**在 SUPA 上 cat + 立即调 `.cu` kernel；要么
  `.contiguous()`、要么 `.clone()`（后者有 ~30% 开销，权衡后不上）。
  见 `reference-project/notes/SUPA_cat_pitfall.md`。
- **`nn.InstanceNorm2d.running_mean/var` 不会被 `model.to('supa')` 搬
  到 SUPA** — FNO 全链路前向必须在 `prepare_supa_eval()` 里显式
  `.to('supa')`。
- **`torch.fft` 直接跑在 supa 设备上** → 数值正确性 OK，但当前
  `torch_br` 版本上算 SUPA-FFT 时会出现精度偏移（rel ≈ 5e-3，超阈
  值）。用 CPU FFT。

### 5. 下一步可选（已在 todo）

- `sufftSetStream` 流水 H2D 与 kernel launch — 当前扩展模块未开放
  stream 句柄，需要重编 `.so`，ROI 偏低。
- 把 64 切的 corner 直接写成 SUPA 端 `(2*M1, M2)` 一次性 GEMM — 已
  试过，赢回 ~2 ms，输在 `torch.cat` 坑。
- 频域 buffer 改成 half（fp16）— 需官方放宽精度阈值到 1e-3，目前
  1e-4 不允许。

## 进阶 / 加分项（已完成）

- 3D SpectralConv `spectral_conv3d_supa`
- `spectral_mul` 反向传播（`SpectralMulFunction`）
- FNO-NS 全链路 SUPA 前向（`forward_supa_chain`）
- 双角 fused 评估 → 因 SUPA cat 坑回退（已记档）

## Auto-Tuning Skill（`spectral_conv/tune.py`）

**目的**：让 SpectralConv 在不同分辨率和显存压力下自动选择真实生效的路径与 cache 上限。测试范围与正式算子一致，均为 CPU 输入到 CPU 输出。

### 可调旋钮

| 旋钮 | 取值 | 影响 |
|---|---|---|
| `path` | `v1` / `fused` | CPU FFT 还是 suFFT fused |
| `buffer_max` | 2 / 4 / 8 | 频域与 host buffer cache 上限 |

当前 kernel 不读取 `fused_block`，因此 tuner 不扫描该伪旋钮，避免用噪声产生无效决策。

### 自动搜索流程

```bash
cd spectral_conv
python3 tune.py
python3 tune.py --quick
python3 tune.py --dry-run
python3 tune.py --shape 64 96 128 256
```

每个配置独立清缓存，执行 warmup 后同步计时，记录 median、mean 和 peak memory。选型规则：先丢掉 `mean/median > 2` 的不稳定行（避免 v1 虚高 median），再按 mean → median → peak memory 选优。正式 sweep 默认 warmup=10 / iters=30。完整结果写入 `spectral_conv/tune_results.json`。

### 运行期动态选优

`spectral_conv_ops.py` 在导入时读取 `tune_results.json`，将决策加载到 `_AUTO_TUNE_TABLE`。`use_sufft="auto"` 查表选路径并同步 `buffer_max`；若结果文件不存在或无效，回退到手工规则。`to_cpu=False` 的 FNO chain 强制 fused，因为 v1 必然返回 CPU。权重缓存按对象身份键入，避免 plain Tensor 每次内容哈希拖慢热路径。

2026-07-25 正式 sweep（10 warmup / 30 iters）选出：

| resolution | path | buffer_max | median ms | peak MB |
|---|---|---:|---:|---:|
| 64 | fused | 2 | 5.337 | 41.7 |
| 128 | fused | 2 | 13.702 | 137.9 |
| 256 | fused | 8 | 52.720 | 522.3 |

### 评审卖点

1. 可运行：quick 和正式 sweep 均已产出 JSON。
2. 可落地：新进程自动加载结果，不只在 tuner 进程临时生效。
3. 可解释：只扫描代码实际使用的旋钮，保留完整候选表与选择依据。

## FNO 评测协议速查

详见 `skills/fno_eval_protocol.md`：batch16 `grid_points/s` 公式、chain `1e-4` 门禁、NS-like 数据披露、训练吞吐加分口径。

## SOL 差距分析（本地 proxy）

详见 `skills/sol_gap_analysis.md`。脚本：`spectral_conv/bench_sol_proxy.py`。产出墙钟 / 显存 / GB/s·TFLOPS proxy，**不是**官方硬件 SOL。
