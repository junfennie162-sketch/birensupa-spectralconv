# skill.md · SpectralConv + FNO-NS（翻斗花园）

## Skill 名称

`spectral_conv_fno_ns_biren`（总入口）

## 子 Skill 索引

| Skill | 路径 | 用途 |
|-------|------|------|
| spectral-conv-dev | [`skills/spectral_conv_dev/`](skills/spectral_conv_dev/) | 算子构建、正确性与误差排查 |
| fno-experiment | [`skills/fno_experiment/`](skills/fno_experiment/) | FNO 前向、指标与可视化 |
| operator-opt-loop | [`skills/operator_opt_loop/`](skills/operator_opt_loop/) | OPT 规范闭环 P0–P6（dry-run / `--strict`）；见 `LOOP_PROCESS.md` |
| fno-eval-protocol | [`skills/fno_eval_protocol.md`](skills/fno_eval_protocol.md) | 公开集评测/吞吐协议 |
| spectral-chain-opt | [`skills/spectral_chain_optimization.md`](skills/spectral_chain_optimization.md) | fused/chain 优化叙事 |
| sol-gap（proxy） | [`skills/sol_gap_analysis.md`](skills/sol_gap_analysis.md) | 队内 SOL proxy，**非得分句** |

## 目标

在壁仞 BIREN GPU 上实现 FNO 核心 2D Spectral Convolution（SUPA），并组装 ≥4 层 FNO 完成二维 Navier-Stokes 涡度前向验证（主报：公开 NS64）。

## 输入

- 张量 `x: [B, C_in, H, W]`（优先 2 的幂次分辨率）
- 可配置 `modes1/modes2`
- FNO：多帧涡度输入 `[B, T_in, H, W]`；数据优先 `fno_ns/data/navier_stokes_v1e-3_N1200_T20.pt`

## 步骤

1. `source brsw_set_env.sh`；`export SUPA_BASE=...`
2. 方式一（可选）：`my_task_direct` → `make build && make run-accuracy`
3. 方式二：`cd spectral_conv && ./build.sh`
4. `python3 test_accuracy.py`（rel ≤ 1e-4 vs 官网双角 reference）
5. `python3 test_perf.py`（64/128/256；idle；写 formal 须无争用）
6. `cd ../fno_ns`：公开集评测 / `visualize.py`（demo_batch 须 public_ns64）
7. **Agent 必须项抽查**：提交根 [`AGENT_OFFICIAL.md`](AGENT_OFFICIAL.md)（≥5 段有效交互、≥3 类场景）+ [`development_log.md`](development_log.md)
8. FNO 推理主表：`python3 benchmark_fno_batch16.py`（先过 chain 一致性；默认公开 NS64）
9. FNO 精度（选修）：主报已 **v10 `spec_ref_r2`**；机制见 `train_public_spectral_refiner_probe.py` + wave4 链（Spectral-Refiner lite · 仅训 spectral + Sobolev H⁻¹ 损失）
10. 训练吞吐加分：`python3 benchmark_train_throughput.py`（CPU/`use_supa=False`）
11. Agent 回放：`python3 skills/operator_opt_loop/run_loop.py --dry-run --strict`（流程见 `skills/operator_opt_loop/LOOP_PROCESS.md`）

## 输出

- 正确性 / 性能日志与 `results/summary.json`
- FNO 图：`results/figures/fno_ns_pred_vs_gt_*.png`、sample_strip
- 主报：公开 L2 **0.035012**（`spec_ref_r2` · **v10**）；上一正式 v9 `dualview_r2` 0.035115；Spectral 本次 idle **3.797 / 8.037 / 29.295 ms**

## 能力边界

- 正式热路径：fused suFFT + SUPA mul（`use_sufft="auto"`）；v1 为对照/可微训练
- 勿使用 `torch.fft` 直接跑在 `device=supa` 做正确性
- FNO **正式主报**用公开 NS64（1000/128 · eval 10→1 · residual）；ckpt `fno_ns/checkpoints/fno_ns_public_demo.pt`；自建 v2 仅工程旁注

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

正式 `test_perf.py` idle 主表（warmup=10 / iters=100，`use_sufft="auto"`，CPU→CPU；**2026-08-14 复测**）：

| resolution | forward_ms | peak memory_MB |
|------------|-----------:|---------------:|
| 64×64      | **3.797**  | 225.3          |
| 128×128    | **8.037**  | 253.3          |
| 256×256    | **29.295** | 353.3          |

说明：vs 官网 CPU 参考加速比约 **19.5× / 11.1× / 10.0×**（非竞品 GPU）。历史中间板（R7 时代约 5.3/13.7/52）仅作演进痕迹，**不得**覆盖 formal 主表。口径见 `results/run_logs/tune_skill_disclaimer_2026-08-01.md`。

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

### 5. Formal ms 冻结后（勿重开）

- Spectral formal 三档已冻结；日常只做护栏复测与旁注叙事（C2R 墙）。  
- Plan2d / `torch.fft@SUPA` / NVIDIA 真融合 / strided pack：**永久 No-Go**（见 OPT_ROUND2）。  
- 历史可选项（stream 流水、half 权重等）ROI 不足或破坏正确性，保持关闭。

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

历史 sweep（2026-07-25，10 warmup / 30 iters；**HISTORICAL**，勿当 formal ms）：

| resolution | path | buffer_max | median ms | peak MB |
|---|---|---:|---:|---:|
| 64 | fused | 2 | 5.337 | 41.7 |
| 128 | fused | 2 | 13.702 | 137.9 |
| 256 | fused | 8 | 52.720 | 522.3 |

> Disclaimer：tune JSON 仅驱动 `use_sufft="auto"` 路径选择；formal 主表以 idle **3.797 / 8.037 / 29.295** 为准（08-14 复测；见 `_history/tune_skill_disclaimer_2026-08-01.md`）。

### 评审卖点

1. 可运行：quick / dry-run / 正式 sweep 均已产出 JSON。
2. 可落地：新进程自动加载结果，不只在 tuner 进程临时生效。
3. 可解释：只扫描代码实际使用的旋钮；SOL/tune **不得**冒充 formal 得分句。

## FNO 评测协议速查

详见 `skills/fno_eval_protocol.md`：batch16 `grid_points/s` 公式、chain `1e-4` 门禁、公开 NS64 主报、训练吞吐加分口径。

## SOL 差距分析（本地 proxy）

详见 `skills/sol_gap_analysis.md`。脚本：`spectral_conv/bench_sol_proxy.py`。产出墙钟 / 显存 / GB/s·TFLOPS proxy，**不是**官方硬件 SOL / SOL-ExecBench。
