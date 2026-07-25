# Skills 汇总 · 2026-07-25

> 把 ai4s-f 和 ai4s-n 两个仓库里所有 SKILL 文档合并到一份，
> 给评委/队友一份完整的 Skill 体系视图。
> 关联：根目录 `skill.md` / `SKILL.md` + `skills/*/SKILL.md`

---

## 0. 概述

本仓库一共包含 **5 个 Skill 文档**（4 个 SKILL.md + 1 个 skill.md），分为 2 层：

| 层 | 文档 | 数量 | 用途 |
|----|------|------|------|
| **L1 · 主 skill** | 根目录 `skill.md` / `SKILL.md` | 2 | 单一项目级入口，覆盖整个 SpectralConv + FNO-NS 工作流 |
| **L2 · 子 skill** | `skills/spectral_conv_dev/SKILL.md`、`skills/fno_experiment/SKILL.md` | 2 | 子任务级入口，单点突破 |

总字数约 **600 行**，按内容来源分布：

| 来源仓库 | 文档 |
|---------|------|
| `ai4s-f/submission/skill.md` | 主 skill（早期版，154 行） |
| `ai4s-f/submission/skills/spectral_conv_dev/SKILL.md` | 子 skill · 算子开发（26 行） |
| `ai4s-f/submission/skills/fno_experiment/SKILL.md` | 子 skill · FNO 实验（26 行） |
| `ai4s-n/submission/skills/fno_experiment/SKILL.md` | 同上副本（n 侧同样存在） |
| `ai4s/submission/skill.md` + `ai4s/submission/SKILL.md` | **本次合并的最终版**（157 行） |

---

## 1. 主 Skill · `翻斗花园_FNO_SpectralConv_BirenSUPA`

> 文件：`skill.md` / `SKILL.md`（根目录）
> 来源：`ai4s-f/skill.md`（早期）→ `ai4s/skill.md`（最终版）
> 全限定 ID：`fandou-garden/fno-spectral-conv-biren-supa`
> 团队：翻斗花园（赛道 5 · 模型与算子）
> 目标硬件：BIREN SUPA GPU（Biren106B）
> 目标框架：PyTorch + torch_br + 自研 C++/SUPA Extension
> Skill 版本：v1.0（2026-07-25）

### 1.1 Skill 一句话描述

在 BIREN SUPA GPU 上构建 FNO 核心 2D Spectral Convolution（FFT + 频域复数乘 + iFFT 全链路 SUPA 化），组装 ≥4 层 FNO 完成二维 Navier-Stokes 涡度前向验证，并通过 Auto-Tuning 在不同分辨率下自动选最佳路径与显存策略。

### 1.2 Skill 适用场景

| 场景 | 描述 |
|------|------|
| **必选算子评测** | 提交 BIREN 平台 SpectralConv 单算子（必选题）|
| **进阶模型评测** | 提交 FNO-2D Navier-Stokes 完整链路（进阶 C）|
| **Auto-Tuning 集成** | 自动扫描 `path × buffer_max` 选 Pareto-best 配置 |
| **跨分辨率性能调优** | 64/128/256 不同路径切换策略 |
| **SUPA 平台 bug 排查** | rfft2_sufft SUPA-input + SUDNN nn.Conv2d crash + 2D FFT ABI |

### 1.3 Skill 不适用场景

- ❌ 非 BIREN 平台（NVIDIA CUDA / AMD ROCm 不适用）
- ❌ 3D FNO（本 Skill 仅覆盖 2D FNO；3D SpectralConv 仅作算子扩展演示）
- ❌ 长时间训练（建议训练在 CPU 路径或调小规模）

### 1.4 目标

在壁仞 BIREN GPU 上实现 FNO 核心 2D Spectral Convolution（SUPA），并组装 ≥4 层 FNO 完成二维 Navier-Stokes 涡度前向验证。

### 1.5 输入

- 张量 `x: [B, C_in, H, W]`（优先 2 的幂次分辨率）
- 可配置 `modes1/modes2`
- FNO：多帧涡度输入 `[B, T_in, H, W]`

### 1.6 步骤

1. `source brsw_set_env.sh`；`export SUPA_BASE=...`
2. 方式一（可选）：`my_task_direct` → `make build && make run-accuracy`
3. 方式二：`submission/spectral_conv/./build.sh`
4. `python3 test_accuracy.py`（rel ≤ 1e-4 vs 官网双角 reference）
5. `python3 test_perf.py`（64/128/256）
6. `cd ../fno_ns && python3 test_forward.py && python3 visualize.py`

### 1.7 输出

- 正确性 JSON / `results/run_logs/spectral_accuracy_*.md`
- 性能表 `results/run_logs/spectral_perf_*.md` + `spectral_grid_points.json`
- FNO 前向日志与 `results/figures/fno_ns_pred_vs_gt.png`

### 1.8 能力边界

- v1：FFT/iFFT 在 CPU；SUPA 负责频域复数乘
- 勿使用 `torch.fft` 直接跑在 `device=supa` 做正确性
- 公开 NS64 数据可替换合成数据；当前默认合成场保证可复现

---

## 2. 性能极限视角（SOL-ExecBench 校准）

SOL-ExecBench 的「硬件极限参考线」决定了 SpectralConv 的 `forward_time_ms` 能拿多少分。我们的工作不是让它无限快，而是让它在 BIREN SUPA 上**贴着内存带宽 + GEMM 峰值**走，离谱的常数代价靠工程手段砍掉。下面是每个 kernel 选择背后的依据，评审问「为什么这么改」时直接对照。

### 2.1 算子拆解 vs 合并访问

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

### 2.2 设备链路（data movement）

理想态：进 SUPA 一次、出 SUPA 一次。中间频域 buffer 在 device 上原地
复用（`_OUT_FREQ_CACHE`，容量上限 4）。**不允许**每次调用重新分配
`(B, Cout, H, Wf, 2)` — 这会让 64×64 的 peak memory 从 ~10 MB 跳到 ~30 MB。

### 2.3 峰值显存策略

`peak_memory_mb` 直接评分。我们强制：

- 频域 buffer 复用（`_OUT_FREQ_CACHE`）
- host staging 复用（`_HOST_OUT_CACHE`），并返回 buffer 本身（不
  `clone()`），让下一次 call 覆盖 — 64/128 节省约 18 MB
- 缩到 `_BUFFER_CACHE_MAX=4`，防极端 shape 切换时把缓存填爆

实测：`B4_Cin32_Cout64` 下

| resolution | memory_MB |
|------------|-----------|
| 64×64      | 9.5       |
| 128×128    | 137.9     |
| 256×256    | 522.2     |

### 2.4 已踩的坑（不要重做）

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

### 2.5 下一步可选（已在 todo）

- `sufftSetStream` 流水 H2D 与 kernel launch — 当前扩展模块未开放
  stream 句柄，需要重编 `.so`，ROI 偏低。
- 把 64 切的 corner 直接写成 SUPA 端 `(2*M1, M2)` 一次性 GEMM — 已
  试过，赢回 ~2 ms，输在 `torch.cat` 坑。
- 频域 buffer 改成 half（fp16）— 需官方放宽精度阈值到 1e-3，目前
  1e-4 不允许。

---

## 3. 进阶 / 加分项（已完成）

- 3D SpectralConv `spectral_conv3d_supa`
- `spectral_mul` 反向传播（`SpectralMulFunction`）
- FNO-NS 全链路 SUPA 前向（`forward_supa_chain`）
- 双角 fused 评估 → 因 SUPA cat 坑回退（已记档）

---

## 4. Auto-Tuning Skill（`spectral_conv/tune.py`）

**目的**：让 SpectralConv 在不同分辨率和显存压力下自动选择真实生效的路径与 cache 上限。测试范围与正式算子一致，均为 CPU 输入到 CPU 输出。

### 4.1 可调旋钮

| 旋钮 | 取值 | 影响 |
|---|---|---|
| `path` | `v1` / `fused` | CPU FFT 还是 suFFT fused |
| `buffer_max` | 2 / 4 / 8 | 频域与 host buffer cache 上限 |

当前 kernel 不读取 `fused_block`，因此 tuner 不扫描该伪旋钮，避免用噪声产生无效决策。

### 4.2 自动搜索流程

```bash
cd spectral_conv
python3 tune.py
python3 tune.py --quick
python3 tune.py --dry-run
python3 tune.py --shape 64 96 128 256
```

每个配置独立清缓存，执行 warmup 后同步计时，记录 median、mean 和 peak memory；先按 median，再按 peak memory 选择 Pareto-best。完整结果写入 `spectral_conv/tune_results.json`。

### 4.3 运行期动态选优

`spectral_conv_ops.py` 在导入时读取 `tune_results.json`，将决策加载到 `_AUTO_TUNE_TABLE`。`use_sufft="auto"` 查表选路径并同步 `buffer_max`；若结果文件不存在或无效，回退到手工规则。`to_cpu=False` 的 FNO chain 强制 fused，因为 v1 必然返回 CPU。

2026-07-25 正式 sweep（5 warmup / 10 iters）选出：

| resolution | path | buffer_max | median ms | peak MB |
|---|---|---:|---:|---:|
| 64 | v1 | 8 | 1.511 | 9.5 |
| 128 | v1 | 4 | 1.843 | 10.5 |
| 256 | fused | 8 | 52.486 | 522.3 |

### 4.4 评审卖点

1. 可运行：quick 和正式 sweep 均已产出 JSON。
2. 可落地：新进程自动加载结果，不只在 tuner 进程临时生效。
3. 可解释：只扫描代码实际使用的旋钮，保留完整候选表与选择依据。

---

## 5. 子 Skill · `spectral-conv-dev`

> 文件：`skills/spectral_conv_dev/SKILL.md`
> 来自：`ai4s-f`

### 5.1 目标

在 BIREN / SUPA 上构建并验证 2D Spectral Convolution 扩展。

### 5.2 典型自然语言输入

- 「编译 SpectralConv 扩展并与 PyTorch reference 对比，误差要小于 1e-4」

### 5.3 步骤

1. `source` 环境：`submission/scripts/setup_env.sh`
2. 进入 `submission/spectral_conv/`，执行 `./build.sh`
3. 运行 `python3 test_accuracy.py`，记录相对误差到 `submission/results/`
4. 若失败：缩小 B/C/H/W 与 modes，固定种子，对比 CPU reference

### 5.4 输出

- 控制台 JSON / 日志
- `submission/results/run_logs/` 下的运行记录

### 5.5 边界

- 仅覆盖前向正确性与基础性能；反向传播为加分项
- 禁止与其它 GPU 任务并发

### 5.6 注意（最终版新增）

- 不要用 `torch.fft` 直接跑在 `supa` 做正确性
- 方式一与 Extension **共用同一算法的 `.su` kernel**

---

## 6. 子 Skill · `fno-experiment`

> 文件：`skills/fno_experiment/SKILL.md`
> 来自：`ai4s-f` 和 `ai4s-n`（两份内容一致）

### 6.1 目标

基于已验证的 SpectralConv 扩展，组织 FNO-NS 单次前向与结果可视化。

### 6.2 典型自然语言输入

- 「用 64×64 Navier-Stokes 样例跑 FNO 前向，并画出预测与真值对比」

### 6.3 步骤

1. 确认 `spectral_conv` 扩展已构建且 accuracy 通过
2. 按 `fno_ns/data/README.md` 准备数据
3. 运行 `python3 test_forward.py`，记录相对 L2
4. 运行 `python3 visualize.py`，图写入 `submission/results/figures/`

### 6.4 输出

- 指标摘要、流场对比图、日志路径

### 6.5 边界

- 海选以可复现前向为主；长时间训练非必须
- Fourier Layer 不少于 4 层

### 6.6 扩展（本项目）

- 训练脚本 `fno_ns/train_official.py`（按官方 n_train=1000, bs=16, 100 epoch）
- 续训脚本 `fno_ns/resume_train.py`（增量训练 + 最佳检查点保存）
- Smoke test 结果：L2=0.0696（2 epoch, gate=晋阶）

---

## 7. Skill 体系拓扑

```
                        ┌────────────────────────────────────────────┐
                        │  主 Skill: spectral_conv_fno_ns_biren       │
                        │  （根目录 skill.md / SKILL.md）              │
                        │                                            │
                        │  • 目标 / 输入 / 步骤 / 输出 / 能力边界      │
                        │  • SOL-ExecBench 性能极限视角               │
                        │  • Auto-Tuning Skill（tune.py）            │
                        └───────────────┬────────────────────────────┘
                                        │
                                        ▼
              ┌─────────────────────────┴─────────────────────────┐
              │                                                   │
              ▼                                                   ▼
  ┌───────────────────────────────┐         ┌───────────────────────────────┐
  │ 子 Skill: spectral-conv-dev   │         │ 子 Skill: fno-experiment       │
  │ (skills/spectral_conv_dev/)   │         │ (skills/fno_experiment/)        │
  │                               │         │                               │
  │ 单算子开发                    │         │ FNO 全链路                    │
  │ 编译/accuracy/调试            │         │ 前向/可视化/训练              │
  │ 边界：前向 + 基础 perf        │         │ 边界：≥4 Fourier Layer        │
  └───────────────────────────────┘         └───────────────────────────────┘
```

---

## 8. 文件位置（相对路径）

```
submission/
├── SKILL.md                  ← 主 skill（大写，官方要求）
├── skill.md                  ← 主 skill（小写，同内容备份）
├── SKILLS_SUMMARY_2026-07-25.md  ← 本文件
└── skills/
    ├── README.md
    ├── spectral_conv_dev/SKILL.md   ← 子 skill 1
    └── fno_experiment/SKILL.md       ← 子 skill 2
```

---

## 9. 对照官方要求

| 官方要求 | 本仓库覆盖 |
|---------|-----------|
| **`skill.md` 文件（必须提交）** | ✅ `SKILL.md` + `skill.md` |
| Auto-Tuning Skill（官方评测图中提到）| ✅ §4（tune.py + _AUTO_TUNE_TABLE）|
| 性能极限视角（SOL-ExecBench 校准）| ✅ §2 |
| 子 Skill 文档 | ✅ `skills/spectral_conv_dev/SKILL.md` + `skills/fno_experiment/SKILL.md` |

---

## 10. 来源仓库对照表

| 文档 | ai4s-f | ai4s-n | ai4s（最终）|
|------|--------|--------|------------|
| 根 `skill.md` | ✅ 154 行（含 SOL 视角 + Auto-Tune）| ❌ | ✅ 157 行（同步自 f） |
| `SKILL.md` 大写副本 | ❌ | ❌ | ✅ 157 行（同上）|
| `skills/spectral_conv_dev/SKILL.md` | ✅ 26 行 | ❌ | ✅ 20 行（含「方式一与 Extension 共用 .su kernel」补充）|
| `skills/fno_experiment/SKILL.md` | ✅ 26 行 | ✅ 26 行（复制）| ✅ 26 行（复制）|

> 所有 5 个 Skill 文档均已合并进本仓库，**最终版本统一在 `/workspace/ai4s/submission/`**。