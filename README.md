# 翻斗花园 · Spectral Convolution + FNO-NS

<p align="left">
  <b>书生国智科探挑战赛</b> · 壁仞飞翔杯 · 赛道五（模型与算子）<br/>
  队伍：<b>翻斗花园</b> · 中北大学<br/>
  提交仓库：<a href="https://github.com/junfennie162-sketch/birensupa-spectralconv">junfennie162-sketch/birensupa-spectralconv</a>（clone 后即提交根）<br/>
  工作区镜像：<a href="https://github.com/Aafff623/fandou-ai4s">Aafff623/fandou-ai4s</a>
</p>

在壁仞 **Biren106B** 单卡上，以 **SUPA + PyTorch Extension（方式二）** 实现 FNO 核心算子 **Spectral Convolution**，并组装 ≥4 层 FNO，完成二维不可压 Navier-Stokes **涡度场** 的公开集预测与可视化。开发过程由 Cursor / 壁仞 Agent 全程辅助，日志可抽查复现。

---

## 现行主报（一眼）

| 主报项 | 现行值 |
|--------|--------|
| 公开 NS64 相对 L2 | **0.035012** · tag `spec_ref_r2` · **v10** |
| Spectral idle（64 / 128 / 256） | **3.797 / 8.037 / 29.295 ms**（2026-08-14 idle 复测；历史板 3.811/8.054/29.560） |
| Spectral 正确性 worst rel | ≈ **2.17×10⁻⁷**（阈值 `1e-4`） |
| Checkpoint | `fno_ns/checkpoints/fno_ns_public_demo.pt` |
| Phase | `submit_gate` **done** |
| 精度姿态 | **v10 promote**（wave4 · Spectral-Refiner lite） |

> **GitHub 不含**：公开 NS `.pt`（约 376MB）与 2.9G 完整 tar。FNO 复评请自备 `fno_ns/data/navier_stokes_v1e-3_N1200_T20.pt`；权重已入库 `fno_ns/checkpoints/fno_ns_public_demo.pt`。

> **真源指针**
>
> 1. 数字：[`results/summary.json`](results/summary.json)
> 2. 行动：[`results/run_logs/CURRENT.md`](results/run_logs/CURRENT.md)
> 3. 官方对照：[`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) · [`skill.md`](skill.md) · **[`AGENT_OFFICIAL.md`](AGENT_OFFICIAL.md)**（Agent 必须项抽查）

---

## 交卷必跑（编译 / 正确性 / 性能）

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd spectral_conv && ./build.sh   # clone 后已在提交根；本机路径也可能是 …/submission

# 1) 编译必选算子（SUPA + PyTorch Extension）
cd spectral_conv && ./build.sh

# 2) 正确性：相对误差 ≤ 1e-4（现行 worst ≈ 2.17e-7）
python3 test_accuracy.py

# 3) 性能：64/128/256 idle（现行 3.797 / 8.037 / 29.295 ms）
python3 test_perf.py

# 4) FNO 公开集复评（期望 tag=spec_ref_r2，L2≈0.035012）
cd ../fno_ns && python3 test_forward.py
```

报告落点：[`results.md`](results.md) · [`results/run_logs/正确性验证报告_2026-08-14.md`](results/run_logs/正确性验证报告_2026-08-14.md) · [`results/run_logs/性能检测报告_2026-08-14.md`](results/run_logs/性能检测报告_2026-08-14.md)

Agent 必须项：[`AGENT_OFFICIAL.md`](AGENT_OFFICIAL.md)（≥5 段、≥3 类场景）+ [`development_log.md`](development_log.md) + [`skill.md`](skill.md)

```bash
# 一键自检（材料 / 口径硬门禁）
python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
```

---

## 评委 3 分钟路径

建议按序打开；第 0 步为一页包。

1. **一页包 + PPT** → [`JUDGE_3MIN_PACK_2026-08-04.md`](results/run_logs/JUDGE_3MIN_PACK_2026-08-04.md) · [`PPT答辩冻结稿_2026-08-04.md`](results/PPT答辩冻结稿_2026-08-04.md)
2. **主报数字** → [`summary.json`](results/summary.json) · L2 **0.035012**（`spec_ref_r2` · v10）
3. **Spectral 口播** → [`results.md`](results.md) · [六轴口播](results/run_logs/SPECTRAL_SIX_AXIS_ORAL_2026-08-04.md)
4. **扩展抽查** → [`extension_showcase.md`](results/run_logs/extension_showcase.md) · [`SPECTRAL_BONUS_AUDIT_CARD.md`](results/run_logs/SPECTRAL_BONUS_AUDIT_CARD.md)
5. **瓶颈解剖** → [`ERROR_AUTOPSY_VERDICT_2026-08-04.md`](results/run_logs/ERROR_AUTOPSY_VERDICT_2026-08-04.md) · [`demo/media/README.md`](demo/media/README.md)
6. **Agent 精品** → [`development_log.md`](development_log.md) · 记录 `26 / 30 / 32 / 35 / 36 / 37 / 38`
7. **Demo / SCP** → [`demo/scp_description.md`](demo/scp_description.md) · `demo/media/`（旧图见 `archive_history/`）

---

<details open>
<summary><b>1. 赛道与选题</b>（官方 README 必填）</summary>

<br/>

| 项 | 说明 |
|----|------|
| 必选题 | **Spectral Convolution**：2D 频域卷积前向；可配置 `modes1/modes2`；与官网双角 reference 对比 |
| 进阶题 | **进阶 C · FNO-NS**：≥4 层 Fourier Neural Operator，二维 NS 涡度预测 |
| 开发路线 | **方式二**：SUPA kernel + PyTorch C++/pybind Extension |
| 衔接加分 | FNO 推理 **复用** 必选 fused SpectralConv（suFFT + SUPA `spectral_mul`） |
| 运行约束 | 单卡 BIREN；无多卡 / 分布式依赖 |

手册要求 README 至少覆盖：赛题与路线、环境、核心算子、编译 / 正确性 / 性能命令、入口与预期、已知限制。  
官方为**最低清单，无字数上限**；下文在必填项之外补充方法、数据协议与工程纪律。

</details>

---

<details open>
<summary><b>2. 环境与依赖</b>（官方 README 必填）</summary>

<br/>

### 2.1 硬件与 SDK

| 项 | 值 |
|----|-----|
| GPU | Biren106B · 单卡 |
| SDK | `1.11.0.0.rc2` |
| `SUPA_BASE` | `/usr/local/birensupa/sdk/1.11.0.0.rc2` |
| 设备名 | `device="supa"`（须先 `import torch_br`） |

> **注意：** `torch.cuda.is_available()` 返回 `False` 为平台预期，不代表故障。

### 2.2 每次新开终端

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2

# 确认 GPU 空闲；禁止与搭档区 ai4s-n 并发占卡
brsmi
```

一键辅助（在 `submission/` 下）：

```bash
./scripts/setup_env.sh
```

### 2.3 软件栈

- **构建工具**
  - `brcc` / `g++` / `brcc --supa-link`
  - 入口：`spectral_conv/build.sh`
- **运行时**
  - `torch` + `torch_br`
  - 复数频域乘：SUPA `spectral_mul`
  - 正式 FFT：`suFFT`
- **训练旁路**
  - `use_supa=False` 纯 torch 可微路径（CPU）
  - 用于公开集训练与梯度安全

</details>

---

<details open>
<summary><b>3. 核心算子 · Spectral Convolution</b>（官方 README 必填）</summary>

<br/>

### 3.1 问题与正式路径

FNO 每层需在频域对低模态复数权重做乘法，再回到空间域做局部分支与非线性。平台上 **FFT / 复数乘 / 布局** 是正确性与性能的交汇点。

**正式路径（fused / auto）：**

```text
输入（常自 CPU 构造）
  → suFFT R2C
  → SUPA spectral_mul（权重常驻 device）
  → suFFT C2R
  → 输出
```

设计要点：

1. 双角（two-corner）布局与官网 FNO reference 对齐（`reference_pytorch.py`）
2. `modes1/modes2` 可配置；主测板见 `summary.spectral_conv`
3. **v1** 保留为对照与可微训练后备
4. **正式得分板以 fused idle 为准**

### 3.2 正确性

```bash
cd spectral_conv
python3 test_accuracy.py
```

- **Reference：** 官网风格双角 spectral conv（`reference_pytorch`）
- **阈值：** 相对误差 ≤ `1e-4`
- **现行 formal：** worst rel ≈ **2.17e−7**（64×64 目标 case）

### 3.3 性能

```bash
cd spectral_conv
python3 test_perf.py
```

| 分辨率 | formal idle |
|--------|-------------|
| 64×64 | **3.797 ms** |
| 128×128 | **8.037 ms** |
| 256×256 | **29.295 ms** |

配置：warmup=`10` · iters=`100` · 同步 wall-clock。

> **纪律：** formal ms **已冻结**。无争用证明前，禁止默认重跑 `test_perf` 覆写 `summary.json`。

<details>
<summary>分段旁注（非第二套正式得分）</summary>

<br/>

- 应用层观测：C2R 约占 device 段 **60%–70%**
- `mul_scatter` 已近噪声量级
- CPU 加速比等仅作口播旁注
- **禁止**写成 SOL / 官方得分句
- 口播卡片：[六轴口播](results/run_logs/SPECTRAL_SIX_AXIS_ORAL_2026-08-04.md)

</details>

### 3.4 扩展抽查（加分项，不替代主报）

| 扩展 | 命令 | 现行 |
|------|------|------|
| Backward | `python3 test_backward.py` | worst grad rel ≈ 6.3e−8 |
| SpectralConv3d 四角 | `python3 test_3d_accuracy.py` | worst ≈ 1.19e−7（**≠** 完整 3D FNO） |
| 非规则分辨率 | `python3 test_irregular_shapes.py` | 9/9 pass |
| 自动调优 | `tune.py` + `tune_results.json` | 决策可复现 |

抽查入口：[`SPECTRAL_BONUS_AUDIT_CARD.md`](results/run_logs/SPECTRAL_BONUS_AUDIT_CARD.md)

</details>

---

<details open>
<summary><b>4. 进阶 FNO-NS</b>（进阶 C）</summary>

<br/>

### 4.1 模型结构

- 实现：`fno_ns/model.py`
- 层数：**4** Fourier Layer
- 宽度 / 模态：width=`32` · modes=`16`
- 张量：`[B, T_in=10, H, W]` → `[B, 1, H, W]`
- 头：persistence **residual**（相对最后一帧的增量）
- 推理：可走 SUPA fused SpectralConv
- 训练公开 ckpt：多用 CPU 可微路径（`use_supa=False`）

### 4.2 公开主报协议

| 项 | 设定 |
|----|------|
| 数据文件 | `fno_ns/data/navier_stokes_v1e-3_N1200_T20.pt`（HF 离线缓存） |
| 物理 / 网格 | ν=`1e-3` · 64×64 |
| 划分 | n_train=`1000` / n_test=`128` · seed=`20260722` |
| 任务 | clean **10→1**（前 10 帧 GT → 预测第 11 帧） |
| 指标 | 相对 L2：`‖pred−gt‖ / ‖gt‖` |
| 主报 | **0.035011906176805496**（`spec_ref_r2` · **v10**） |

披露全文：[`results/data_disclosure.md`](results/data_disclosure.md)

> **答辩口径：** 经典 FNO 论文 ≈0.0128（ν=1e−3、64×64）属于 **T=50 递归轨迹** 设定，与本队 **T20 文件上 clean 单步 10→1** **不可直接横比**。  
> 详见 [`ERROR_AUTOPSY_VERDICT_2026-08-04.md`](results/run_logs/ERROR_AUTOPSY_VERDICT_2026-08-04.md)。

### 4.3 有效训练配方

```text
scratch / boost
  → squeeze
  → multistep TF + soft
  → scheduled sampling
  → soft
  → freeze spectral 抛光
  → freeze_r9（现行主报）
```

已纳入轨迹的辅助手段：

- 高频 FFT 幅度损失
- 周期 roll 增强
- residual 头

<details>
<summary>已证伪 / 停用路线（摘要）</summary>

<br/>

1. 盲目 `modes=20` / `width=48`（test 变差）
2. 同构 sched deepen（贴墙 / NO_SIGNAL）
3. 单独 hard-example reweight（Δ=0）
4. 未破 gate 的弱 promote（如 0.035252 → **已回滚**）

全表：[`results/experiment_matrix.md`](results/experiment_matrix.md)

</details>

### 4.4 工程对照（非公开分）

自建 `ns_like_v2` continue3 相对 L2 ≈ **0.005144**：

- 仅证明管线与优化能力
- **不得**写入公开主报或冒充榜单分

### 4.5 可视化与 Demo

| 材料 | 路径 |
|------|------|
| Pred / GT / error | `demo/media/fno_ns_pred_vs_gt_2026-08-02.png` |
| best / median / worst strip | `demo/media/fno_ns_sample_strip_2026-08-02.png` |
| 协议对照图 | `demo/media/protocol_vs_fno_paper_0128.png` |
| 误差分解图 | `demo/media/error_decomp_e1_tf_ar.png` |
| 频谱 / 涡结构 | `demo/media/spectrum_best_median_worst.png`（+ heatmaps） |
| 单卡日志 | `demo/media/brsmi_snapshot.txt` |
| 指标快照 | `demo/media/metrics_snapshot.md` |

</details>

---

<details open>
<summary><b>5. 编译 · 正确性 · 性能 · 入口</b>（官方 README 必填）</summary>

<br/>

### 5.1 编译必选算子

```bash
cd /workspace/ai4s-f/submission
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2

cd spectral_conv
./build.sh
# 预期：Extension 链接成功，进程 exit 0
```

### 5.2 正确性（Spectral）

```bash
cd spectral_conv
python3 test_accuracy.py
# 预期：全部 case ok；max_rel ≤ 1e-4
# 现行 formal worst ≈ 2.17e-7
```

可选加强抽查：

```bash
python3 test_sufft_accuracy.py
python3 test_backward.py
python3 test_3d_accuracy.py
python3 test_irregular_shapes.py
```

### 5.3 性能（Spectral）

```bash
cd spectral_conv
python3 test_perf.py
# 三档 64/128/256；正式板已冻结，交卷勿无故覆写
```

现行正式板：**3.797 / 8.037 / 29.295 ms**（2026-08-14 idle 复测；07-31 板 3.811/8.054/29.560，噪声内）。

### 5.4 FNO 前向与可视化

```bash
cd /workspace/ai4s-f/submission/fno_ns
python3 test_forward.py     # ≥4 层；报告相对 L2
python3 visualize.py        # 刷新 figures / demo media
```

### 5.5 推荐总回归

```bash
cd /workspace/ai4s-f/submission

./scripts/run_tests.sh              # 主链路（推荐）
./scripts/run_tests.sh all          # 更全串行回归
./scripts/run_tests.sh sufft        # suFFT 正确性
./scripts/run_tests.sh fno-chain    # CPU vs SUPA，rel ≤ 1e-4
./scripts/run_tests.sh fno-batch16  # batch=16 吞吐
./scripts/run_demo.sh

python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
./scripts/maintain_assets.sh check submit_gate
```

| # | 场景 | 命令 | 通过标准 |
|---|------|------|----------|
| 1 | 改 `.su` / `.cpp` / `build.sh` | `build.sh` + `test_accuracy.py` | rel ≤ `1e-4` |
| 2 | 关心算子速度 | `test_perf.py`（慎写 formal） | 三档 ms 可复现 |
| 3 | 改 FNO 模型 | `test_forward.py` / `fno-chain` | 前向成功；chain ≤ `1e-4` |
| 4 | 交卷前 | `run_tests.sh` + `run_loop --strict` + `maintain check` | exit 0 / PASS |

**预期输出落点：**

- `results/summary.json`
- `results.md`
- `results/run_logs/`
- `demo/media/`

### 5.6 FNO 吞吐（工程旁注）

| 项 | 现行 |
|----|------|
| batch=16 纯前向 | ≈ **1.60M** grid_points/s · ≈391 samp/s · peak≈202 MB |
| 训练吞吐（加分） | ≈ **3.47×10⁴** grid_points/s（CPU / `use_supa=False`，含 loss+bwd+opt） |

须先通过 chain 一致性；未过门禁的快速数字不进正式叙述。

</details>

---

<details>
<summary><b>6. 目录结构</b></summary>

<br/>

```text
submission/
├── README.md                 # 本文件
├── skill.md                  # 官方必须 Skill 入口
├── SUBMISSION_CHECKLIST.md   # 官方条目 ↔ 路径对照
├── FILE_CONVENTIONS.md       # 现行 / 历史 / 归档约定
├── development_log.md        # Agent 日志（必须 ≥5 段）
├── results.md                # 正确性与性能汇总正文
├── spectral_conv/            # 必选算子：.su / .cpp / build / tests
├── fno_ns/                   # FNO 模型、数据、训练、可视化、诊断
├── scripts/                  # setup / run_tests / run_demo / maintain_assets
├── skills/                   # Agent Skills + operator_opt_loop
├── demo/                     # SCP + media（评委展示）
└── results/                  # summary · phase_status · run_logs · figures
```

| 路径 | 职责 |
|------|------|
| `spectral_conv/` | SUPA kernel、Extension、正确性 / 性能 / 扩展测试 |
| `fno_ns/` | FNO-NS、公开集协议、promote 门禁、Autopsy / PF 探针 |
| `results/run_logs/` | 轮次计划、对齐卡、JUDGE 包、实验摘要 |
| `skills/operator_opt_loop/` | P0–P6 规范闭环（`--dry-run --strict`） |

</details>

---

<details>
<summary><b>7. 关键实测摘要</b></summary>

<br/>

| 项 | 结果 |
|----|------|
| SpectralConv rel（formal） | ≈ **2.17e−7** ≤ `1e-4` |
| Spectral idle 64/128/256 | **3.797 / 8.037 / 29.295 ms** |
| spectral_mul backward | worst ≈ 6.3e−8 |
| SpectralConv3d（四角） | worst ≈ 1.19e−7 |
| FNO 层数 | **4** |
| FNO 公开 NS64 L2 | **0.035012**（`spec_ref_r2` · v10） |
| FNO batch16 | ≈1.60M gps · peak≈202 MB |
| FNO 训练吞吐（旁注） | ≈3.47×10⁴ gps |
| 自建 v2（非公开） | ≈0.005144 |

表格化正文：[`results.md`](results.md)

</details>

---

<details open>
<summary><b>8. 精度线与工程纪律</b>（本队现状）</summary>

<br/>

### 8.1 Promote 门禁

```text
gate = live_best − 1e−4
例：0.035012 → gate 0.034912

仅当同时满足：
  (1) best < gate
  (2) 人工确认
才可 promote / 编评测报告新 v

自动链默认禁止无 gate promote
（promote_guard.py + ALLOW_AUTO_PROMOTE）
```

### 8.2 近期收口事实

| 实验 | 数字 | 处置 |
|------|------|------|
| `spec_ref_r2` | **0.035012** | **KEEP · 现行主报 v10** |
| `dualview_r2` | 0.035115 | 历史 v9 |
| `freeze_r9` | 0.035302 | 历史 v8 |
| freeze_r10 手跑 / autochain | 0.035287 / 0.035252 | 未破 gate；弱写入 **已回滚** |
| soft / hybrid / modes20 / A1 | Δ≈0 或差于主报 | NO_SIGNAL / KILL |
| Error Autopsy D（epochs=0） | ρ(e1,g)≈0.80；worst16∩=10 | 产出裁决与答辩三图 |
| `pf_clean_r1`（clean-anchor PF） | best **0.035216** | **NO_SIGNAL** · 差 gate≈1.4e−5 · **未 promote** |

### 8.3 红线

1. 禁止把 SOL / proxy / tune median 写成正式得分句
2. 禁止未破 gate 探针编入评测报告正式 v 号
3. 禁止 TTA、自建 v2、异构算子污染公开主报
4. 禁止 `ai4s-f` 与 `ai4s-n` 并发占用同一 GPU
5. 禁止无必要长挂训练；探针用 `nohup` + `--stop-on-gate`

</details>

---

<details>
<summary><b>9. Agent / Skills</b>（官方必须 · 约 15%）</summary>

<br/>

| 资源 | 说明 |
|------|------|
| [`development_log.md`](development_log.md) | ≥5 段有效交互；覆盖 kernel / 超参 / 平台 / 数据 / 可视化 / 瓶颈；精品含回滚门禁、材料闭环、Autopsy、PF |
| [`skill.md`](skill.md) | 官方必须总入口 |
| `skills/spectral_conv_dev` | 算子构建与排错 |
| `skills/fno_experiment` | FNO 实验与指标 |
| `skills/operator_opt_loop` | OPT P0–P6；见 `LOOP_PROCESS.md` |

```bash
python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
```

</details>

---

<details open>
<summary><b>10. 已知限制</b>（官方 README 必填）</summary>

<br/>

1. **正式 Spectral 路径为 fused**；v1 仅对照 / 训练后备，不以 v1 性能板替代 formal idle。
2. **Formal idle ms 冻结**；分段占比、CPU 加速比为旁注叙事。
3. **公开主报协议**为 HF `N1200_T20` 上 clean 单步 10→1（1000/128）；与论文 T=50 轨迹指标不可横比。
4. **自建 v2** 仅工程对照，不计公开分。
5. **SpectralConv3d / irregular** 为算子扩展抽查，不代表完整 3D FNO 求解器。
6. **FNO device-resident 性能**须先通过 CPU↔SUPA chain 门禁（rel≤`1e-4`）。
7. **精度线已永久停训**；后续以答辩材料与可复现回归为主，不再默认开同构抛光 / STLW / 扩 modes。
8. 完整 `run_tests.sh` 为数分钟量级（**不含**无故重跑 formal perf）。

</details>

---

<details>
<summary><b>11. 展示材料</b>（建议项 · 已准备）</summary>

<br/>

| 材料 | 路径 |
|------|------|
| 评委一页包 | `results/run_logs/JUDGE_3MIN_PACK_2026-08-04.md` |
| PPT 页冻结稿 | `results/PPT答辩冻结稿_2026-08-04.md` |
| 六轴口播 | `results/run_logs/SPECTRAL_SIX_AXIS_ORAL_2026-08-04.md` |
| 流场与解剖图 | `demo/media/` |
| 90s 分镜 | `results/run_logs/demo_storyboard_90s.md` |
| 资产对齐卡 | `results/run_logs/OFFICIAL_ASSET_ALIGNMENT_2026-08-04.md` |

</details>

---

<details>
<summary><b>12. 外部文档</b></summary>

<br/>

- 《算子与模型赛道选手手册》  
  `/workspace/赛题文档/算子与模型赛道选手手册.md`  
  （README 最低字段见「提交要求 · README.md」）
- 官网赛道五详情  
  `/workspace/赛题文档/官网-赛道五-模型与算子详情页.md`  
  （统一要求 / 提交规范）
- 工作区约定  
  `/workspace/ai4s-f/AGENTS.md` · `/workspace/AGENTS.md`

</details>

---

## 附录

<details>
<summary><b>A. 一页命令速查</b></summary>

<br/>

```bash
# ---------- 环境 ----------
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2

cd /workspace/ai4s-f/submission
./scripts/setup_env.sh

# ---------- 必选 ----------
cd spectral_conv && ./build.sh && python3 test_accuracy.py
# python3 test_perf.py   # formal 已冻结，交卷勿默认覆写

# ---------- 进阶 ----------
cd ../fno_ns && python3 test_forward.py && python3 visualize.py

# ---------- 总回归 + 材料 ----------
cd ..
./scripts/run_tests.sh
./scripts/run_demo.sh
python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
./scripts/maintain_assets.sh check submit_gate
```

</details>

<details>
<summary><b>B. GPU / SUPA 职责划分</b></summary>

<br/>

| 环节 | 承担 |
|------|------|
| SpectralConv kernel | 频域复数乘（SUPA Extension） |
| 正式前向 | suFFT R2C → SUPA mul（常驻）→ suFFT C2R |
| FNO Fourier Layer | 复用 fused 双角 API |
| 训练路径 | CPU torch 可微（`use_supa=False`） |
| FFT v1 对照 | CPU torch FFT |

</details>

<details>
<summary><b>C. 官方 README 必填项对照</b></summary>

<br/>

手册「提交要求 · README.md」逐条对照：

| # | 官方要求 | 本文件位置 |
|---|----------|------------|
| 1 | 赛道与赛题（必选 + 进阶） | §1 |
| 2 | 开发路线 | §1 · 文首 |
| 3 | 运行设备与环境版本 | §2 |
| 4 | 核心自定义算子 / SUPA kernel | §3 |
| 5 | 编译命令 | §5.1 |
| 6 | 正确性测试与 reference | §3.2 · §5.2 |
| 7 | 性能测试命令 | §3.3 · §5.3 |
| 8 | 运行入口与预期输出 | §5 · 附录 A |
| 9 | 已知限制 | §10 |

</details>
