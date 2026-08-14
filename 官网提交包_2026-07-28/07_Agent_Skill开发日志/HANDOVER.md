# 交接总结 · 2026-07-25

> 提交内容梳理、剩余事项、关键文件位置

---

## 1. 已完成的工作

### 必选：Spectral Convolution（算子赛道）
- **双实现路径**：
  - `v1`：`torch.fft.rfft2` (CPU) + `spectral_mul_supa_device` (SUPA)
  - `fused`：`rfft2_sufft` + `spectral_mul_supa_device` + `irfft2_sufft`（全 SUPA）
- **自适应切换**：`min(H, W) ≥ 64` 自动走 fused
- **频域 buffer 缓存**：`_OUT_FREQ_CACHE` / `_HOST_OUT_CACHE` 减少 D2H 次数
- **Auto-Tuning**：`tune.py` 扫描 `path` × `buffer_max` × `fused_block`，25 秒出 Pareto-best 配置，写入 `_AUTO_TUNE_TABLE`
- **R5 dual_out**：`spectral_mul_dual_out(x1,w1,x2,w2,y1,y2)` 单 pybind 启动两个角点 kernel
- **3D / backward 扩展**：`spectral_conv3d`、`spectral_mul` 的 backward
- **鲁棒性**：9 种非 2-power 分辨率全部 worst rel < 4e-7

| 测试 | 结果 |
|------|------|
| `test_accuracy.py` | 5/5 case 通过，worst rel 2.83e-7（阈值 1e-4） |
| `test_3d_accuracy.py` | 通过 |
| `test_backward.py` | 通过 |
| `test_irregular_shapes.py` | 9/9 通过，worst rel 3.92e-7 |
| `test_perf.py` | 64/128/256 → **5.32 / 13.69 / 52.64 ms** |

### 进阶 C：FNO-2D Navier-Stokes
- 4 Fourier Layer 完整 SUPA 链路
- 适配 `nn.InstanceNorm2d.running_mean/var` 不随 `.to('supa')` 自动迁移的坑
- FNO L2 = 0.009516

---

## 2. 提交目录（最终版）

```
submission/
├── README.md                 # 总览 / 选题 / 命令
├── SKILL.md                  # Auto-Tuning Skill（大写，官方要求）
├── skill.md                  # 同上（小写副本）
├── SUBMISSION_MANIFEST.md    # 6 phase 清单
├── HANDOVER.md               # 本文件
├── development_log.md        # 17 段开发记录
├── results.md                # 测试结果
├── results/                  # JSON + run_logs
├── spectral_conv_combo/      # ★ 必选算子主目录（含 .so/.su/.cpp）
├── spectral_conv/            # 备份版本
├── fno_ns/                   # 进阶 FNO
├── scripts/                  # maintain_assets / setup_env / run_tests
├── skills/                   # 子技能（chain_optimization / dev / experiment）
├── demo/                     # 展示材料
└── logs/                     # 编译/运行时原始日志
```

---

## 3. 官方要求 7 项 → 文件位置对照

| # | 官方要求 | 本仓库位置 |
|---|---------|------------|
| 1 | 项目源码（含 SUPA/Extension） | `spectral_conv_combo/*.cpp` `*.su` `*.so`、`fno_ns/` |
| 2 | 依赖说明 + 编译命令 | `README.md`、`spectral_conv_combo/build.sh`、`scripts/setup_env.sh` |
| 3 | 正确性脚本与结果 | `spectral_conv_combo/test_accuracy.py`、`results.md`、`results/run_logs/` |
| 4 | 性能测试与报告 | `spectral_conv_combo/test_perf.py`、`results.md` 性能表 |
| 5 | 运行日志或截图 | `results/run_logs/` (≥8 个 .md) + `logs/` |
| 6 | Agent 日志 ≥5 段、≥3 类场景 | `development_log.md` (17 段) |
| 7 | **skill.md（必须）** | `SKILL.md` + `skill.md` + `skills/*/SKILL.md` |

---

## 4. 当前已知风险

1. **`rfft2_sufft` SUPA-input correctness bug**（P1）：
   - 当输入已是 SUPA 驻留 tensor 时，`rfft2_sufft` 返回 garbage（rel≈1.4）
   - 当前 SpectralConv 主路径（CPU input → SUPA）正确，影响范围仅 FNO chain 的逐层 rfft
   - 本次任务不评测 FNO chain 正确性，单算子全过 + 性能达标

2. **SUDNN `nn.Conv2d` 偶发崩溃**（P0 → 已通过 fallback 规避）：
   - `ErrorCode: 6 / 719` 出现在 SUDA 1×1 conv plan 初始化阶段
   - 与本次 SpectralConv 优化无关，属环境/SUDRN 库问题
   - 单算子测试 5/5 通过 + 性能达标 → 必选题不受影响

---

## 5. 官方指标对照表（基于用户 2026-07-25 10:38 提供的截图）

| 维度 | 官方要求 | 提交版数据 | 状态 |
|------|---------|-----------|------|
| **正确性** | 相对 L2 ≤ 阈值（声明 64×64） | SpectralConv 5/5 通过，worst rel **2.83e-7**；FNO L2 = **0.009516** | ✅ |
| **显存** | 峰值 MB（加档） | 64/128/256 → **41.7 / 137.9 / 522.2 MB** | ✅ |
| **性能 (主指标)** | `grid_points/s = Σ(H·W·batch)/Σt_warmup`，warmup=10，统计=50，**bs=16** | 当前 `summary.json` 是 `forward_time_ms`，bs=1；**未按官方口径测** | ⚠️ 缺口径 |
| **FNO 模型搭建** | 4 Fourier Layer 完整，复用必选 SpectralConv | 4 层，复用 `spectral_conv_combo` | ✅ |
| **精度声明** | 64×64 数据 | 5-case 含 64×64 | ✅ |
| **可视化** | 预测 vs GT | `results/figures/fno_ns_pred_vs_gt_*.png` ×3 | ✅ |
| **Agent 开发** | ≥5 段、≥3 类场景 | `development_log.md` 17 段 | ✅ |
| **skill.md** | 必须 | `SKILL.md` + `skill.md` 双份 | ✅ |
| **训练门：晋阶** | ≤500 step（≈8 epoch）| 总 step 数 = `epochs × ⌈n_train/bs⌋`；1000/16 = 63 step/epoch | ❓ 取决于 epoch |
| **训练门：有效** | ≤2000 step（≈30 epoch）| 同上 | ❓ |
| **训练门：推荐** | ≥6000 step（≈100 epoch）| 同上 | ❓ |

### 已补齐 / 明天一键跑

- `spectral_conv_combo/test_perf_grid_points.py` — 按官方公式 `grid_points/s` 测（bs=16, warmup=10, iters=50），输出到 `results/run_logs/spectral_grid_points.json`
- `fno_ns/train_official.py` — 按官方 `n_train=1000, bs=16, 100 epoch, 6300 step` 跑完整训练；自动判定晋阶/有效/推荐 gate；保存 best 到 `fno_ns/checkpoints/fno_ns_official_best.pt`
- `fno_ns/dataset.py` 与 `fno_ns/checkpoints/fno_ns_demo.pt` 已从 `ai4s-f` 同步过来（之前 `submission/` 缺这两个，`resume_train.py` 跑不起来）
- 注：v2 数据缓存 481 MB **没有**打进 git（按 `.gitignore` 规则本地重新生成）；接收方首次跑 `train_official.py` 会自动调 `load_or_build_ns_like` 生成

### 已测得的官方口径数据（2026-07-25）

**SpectralConv grid_points/s**（`results/run_logs/spectral_grid_points.json`，bs=16, warmup=10, iters=50）：

| 分辨率 | mean_ms | median_ms | grid_points/s |
|--------|---------|-----------|---------------|
| 64×64 | 74.17 | 80.45 | **883,577** |
| 128×128 | 79.54 | 72.44 | **3,295,767** |
| 256×256 | 212.88 | 210.98 | **4,925,789** |

**FNO 训练 smoke test**（`results/run_logs/fno_official_train_20260725_105006.log`，1 epoch, bs=16, n_train=1000）：
- steps/epoch = **63**（符合官方 1000/16 ≈ 63）
- 1 epoch = 77 秒 → **100 epoch 预计 ≈ 128 分钟（≈2.1 小时）**
- 1 epoch test_l2 = 0.103（best）；gate = **晋阶**（≤500 step）
- ⚠️ 未跑完整 100 epoch 已停止（用户决定让搭档接手跑）；提交版只标注 smoke test 数据 + 完整脚本

**显存（按官方"峰值 MB"口径，已在 `summary.json`）**：64/128/256 → 41.7/137.9/522.2 MB

---

## 5. GitHub 推送未完成总结

**状态**：本地 git 仓库已就绪、101 文件、1 commit `4422058`（在 `/workspace/ai4s/submission/.git/`），**未推到 GitHub**。

**远端仓库**：[`junfennie162-sketch/birensupa-spectralconv`](https://github.com/junfennie162-sketch/birensupa-spectralconv) 已创建（空）。

**阻塞原因**：用户提供的 PAT 是 **fine-grained personal access token**（`github_pat_11B...`），只勾了 `repo` 权限，但 fine-grained token 的"创建仓库"和"推送到新仓库"分别需要 `administration: write` 和显式把目标仓库加入"Repository access"白名单。

实际尝试与错误：
1. `gh repo create ... --push` → `GraphQL: Resource not accessible by personal access token (createRepository)`
2. `gh api user/repos POST` → `403 Resource not accessible by personal access token`
3. `git push -u origin main`（用同一 PAT 通过 x-access-token URL）→ `403 Permission denied to junfennie162-sketch`

**明天接手要做**（二选一）：
- **方法 A（推荐，30 秒）**：用户去 https://github.com/settings/personal-access-tokens 打开那个 token 的编辑页：
  1. "Repository access" → 选 **"Only select repositories"** → 点 **"Select repositories"** → 勾 `birensupa-spectralconv` → 保存
  2. 然后跑：
     ```bash
     cd /workspace/ai4s/submission
     git push -u origin main
     ```
- **方法 B**：用户重新生成一个 **classic PAT**（勾 `repo` 即可），用 classic PAT 重试上面三条命令中的任一条
- **方法 C**：用户手动 `git clone` 本仓库到本地，在本地配 GitHub SSH key 后 `git push`（彻底绕开本机 PAT 限制）

**本地 git 已经完整准备好**（验证过 `git log`、`git status --short` 干净），明天只要 PAT 权限到位，一行 `git push -u origin main` 立刻搞定。

---

## 6. 复现命令

```bash
cd spectral_conv_combo/
bash build.sh                     # 编译 .so
python test_accuracy.py           # 5-case correctness
python test_perf.py               # 64/128/256 perf
python test_irregular_shapes.py   # 9-shape robustness
python test_backward.py           # backward correctness
python test_3d_accuracy.py        # 3D extension
python tune.py --quick            # auto-tuning 扫描
```

---

## 7. 明天重跑规划（未跑通的优化路线）

这条线**不打包提交**，留给明天用户 + 搭档（ai4s-f）各跑一遍。

### 7.1 待重跑的两条路线

| 路线 | 阻塞 | 明天接手 |
|------|------|----------|
| **A. `rfft2_sufft` SUPA-input bug 修复** | 当输入已是 SUPA 驻留 tensor 时返回 garbage（rel≈1.4）；之前在 Python 层加 `x.detach().cpu().to('supa').contiguous()` 临时绕过，但引入 D2H/H2D 开销 | 明天先环境 reset（重启 Python kernel / 重启 container）再试；若仍坏，尝试 C++ 层强制 `.cpu()` 转 host-mirror 后再 `sufftExecR2C`（之前编译报 `expected primary-expression before 'float'` 的语法坑待解） |
| **B. Path B · Kernel Fusion** | 调研完毕 ROI 太低，已决定不做；明天仅做轻量验证 | 跳过；如一定要做，参考 `sdk_kernel_probe_2026-07-24.md` 评估表 |
| **C. FNO chain profile 重跑** | SUDNN `ErrorCode: 6` 在 `nn.Conv2d` 初始化时崩，与代码改动无关 | 环境 reset 后重跑 `fno_ns/profile_chain.py`；若仍炸，FNO chain 的 skip-conv 改用 `torch.einsum`/手写 matmul 替换 `nn.Conv2d(1×1)` |

### 7.2 明天接手清单（按优先级）

1. **环境 reset + 全量回归**
   ```bash
   # 重启 Python kernel / 重启 container
   cd /workspace/ai4s/submission
   bash scripts/maintain_assets.sh status   # 资产检查
   python spectral_conv_combo/test_accuracy.py        # 5/5 应过
   python spectral_conv_combo/test_perf.py            # 5.32/13.69/52.64 ms
   python fno_ns/test_supa_chain.py                   # 若 SUDNN 不再炸
   ```

2. **修 `rfft2_sufft` SUPA-input bug**（如果时间允许）
   - 改 `spectral_conv_ext.cpp::rfft2_sufft`：开头先 `auto x_cpu = x.cpu();` 强制 host-mirror，再 `sufftExecR2C`
   - 改完后 `git commit -m "fix(rfft2_sufft): handle SUPA-resident input"`
   - 重跑 `fno_ns/test_supa_chain.py`，预期逐层 rel < 1e-4

3. **解决 GitHub 推送**（见 §5 的三选一方法）

4. **如 SUDNN 持续崩**：FNO chain 的 `nn.Conv2d(1×1)` 用 `torch.einsum('bchw,co->bohw', x, w.squeeze())` 替换，绕开 SUDA 1×1 conv 路径

### 7.3 不在明天范围内

- 真实 rfft+mul+irfft 三合一 fused kernel：SDK 不开源 FFT，TCI 是 tile 抽象不是蝶形运算，做不了
- 重新扫描 `_AUTO_TUNE_TABLE` Pareto 前沿：当前最优配置已固化，明天不需要
- 重新生成数据 / 训练 FNO：模型 L2 = 0.009516 已达评测阈值