# GPT 调研简报 · FNO-NS 精度瓶颈与 AR 分布错配

> **用途**：把本文件整份拖进 ChatGPT / 其它模型对话框，请其针对「最大卡点」做调研分析与可执行建议。  
> **工作区**：壁仞飞翔杯 · 赛道五（模型与算子）· 队伍「翻斗花园」· 开发区 `/workspace/ai4s-f/submission`  
> **日期**：2026-08-04  
> **请 GPT 用中文回答**；结论须带出处（论文 arXiv / 官方文档 URL）；区分「已证实 / 假说 / 需我们本地实验」。

---

## 0. 你要扮演的角色

你是 AI4S / 神经算子（Neural Operator）方向的技术顾问。请基于下文**队内实测约束**，不要给「再加宽一点 / 再训久一点」这类已被我们证伪的空泛建议。

输出结构请固定为：

1. **协议对齐**：我们的评测与经典 FNO 论文是否同构  
2. **瓶颈定性**：最大卡点是什么（按优先级）  
3. **机制推荐**：最多 2 条可落地新训练机制（含伪代码级步骤）  
4. **否决清单**：哪些方向不该再做  
5. **实验设计**：单卡、短探针（≤4 epoch）、gate 纪律下的最小实验  
6. **不确定项**：还需要我们提供哪些本地日志/曲线才能定论  

---

## 1. 让 GPT 着重调研的部分（核心任务）

请按优先级调研并回答：

### 1.1 协议是否对齐（最优先）

经典文献（Li et al., *Fourier Neural Operator for Parametric PDEs*, arXiv:2010.08895）在 Navier-Stokes、ν=1e−3、64×64 上报告 FNO-2D 相对 L2 ≈ **0.0128**。

我们的正式协议是：

| 项 | 我们的设定 |
|----|------------|
| 数据文件 | `navier_stokes_v1e-3_N1200_T20.pt`（HF `abelsr1710/navier-stokes-2d-fno` 离线缓存） |
| 粘度 | ν = 1e−3 |
| 分辨率 | 64×64 |
| 划分 | n_train=**1000** / n_test=**128**，seed=**20260722** |
| 任务 | 输入前 **T_in=10** 个涡度场 → 预测下一时刻 **T_out=1** |
| 指标 | 相对 L2：`‖pred−gt‖ / ‖gt‖`（按 batch/样本约定与 `train_public_ns64_boost.evaluate` 一致） |
| 模型 | 4 层 FNO，width=**32**，modes=**16**，residual 头 + 可选 hf 损失 + 周期 roll aug |
| 主报 L2 | **0.035302**（tag `freeze_r9`，评测报告 v8） |

**请调研并明确**：

- 论文 0.0128 的设定（数据生成、T、是否 FNO-2D+RNN 逐步、N、modes/width、损失）与我们差在哪里？  
- 在「我们这份公开数据 + 单步 10→1」协议下，0.035 算偏高、合理，还是明显有空间？  
- 若空间存在，更可能来自**数据/协议差**还是**训练机制差**？

### 1.2 AR / teacher-forcing 分布偏移（主嫌疑瓶颈）

队内判断：当前设计最大难点不在 Spectral mul，而在：

> **单步 GT 监督训练 vs（潜在）自回归推理/误差累积的分布错配**；同构抛光已熄火。

我们已做过：scheduled sampling（`p_ar` 缓升）、soft-α、freeze spectral 抛光、hard-example reweight（A1，Δ=0）。  
我们**没做过真·Pushforward**（第 1 步 stopgrad 自噪声、只对第 2 步反传）和 **STLW**（多步损失按时间步加权再退火）。

**请调研**：

- Pushforward（Brandstetter et al. / MP-PDE 等）在「**评测仍是单步 T_out=1**」时，是否仍可能显著降单步 L2？还是主要改善多步 rollout 旁注？  
- STLW、Recurrent Neural Operator、input noising、PDE-Refiner 等，谁最贴合我们的协议与算力（单卡、CPU 训练路径为主、epochs≤4 快探针）？  
- scheduled sampling 与 pushforward 的本质差别是什么？我们已有 sched 却仍贴墙，说明什么？

### 1.3 误差结构与容量饱和

- 为何 **modes=20 / width=48** 重训后 best≈0.0367 / 0.0361，**差于**主报 0.0353？文献上「盲目加 modes」是否常失败？  
- ν=1e−3 时 modes=16 是否已接近谱截断饱和？  
- 请建议：在固定 `freeze_r9` ckpt 上，我们应做哪些**离线诊断**（频谱误差分箱、worst-k 样本形态、逐步 rollout 误差曲线）来确认瓶颈是低频/高频/涡结构/时间相关？

### 1.4 算子侧 C2R 墙（次要，但需定性）

Spectral fused 分段（旁注，非 formal 覆写）：

| 分辨率 | e2e≈ | R2C | mul_scatter | C2R | C2R/(R2C+mul+C2R) |
|--------|------|-----|-------------|-----|-------------------|
| 64 | 3.84 ms | 1.06 | 0.23 | 2.13 | ≈62% |
| 128 | 8.89 | 2.16 | 0.25 | 4.56 | ≈66% |
| 256 | 29.3 | 6.75 | 0.26 | 16.5 | ≈70% |

Formal idle 已冻结：**3.811 / 8.054 / 29.560 ms**。平台：Biren106B + suFFT + 自研 SUPA `spectral_mul`。

**请调研**：在应用层（不改厂商 FFT 库）是否还有现实突破；或结论应为「应用层不可破，只做叙事」。

### 1.5 竞赛评分语境（约束建议边界）

赛道权重大意（必选 Spectral + 进阶 FNO + Agent≈15%）：

- Spectral：正确性 35% / 性能 25% / 扩展 15% / Agent 15%  
- FNO：搭建 30% / 精度 25% / 可视化 20% / 性能 10% / Agent 15%  

我们正确性已 ≪1e−4；formal ms 冻结；精度线官方姿态「已停」；材料/答辩包刚补齐。  
**请不要建议**：解冻 formal ms 死磕 C2R、用 SOL 当官方得分、F-FNO 换主报算子、TTA 计主报、自建 v2 L2 冒充公开分。

---

## 2. 我的困惑点（请直接回应）

1. **主矛盾到底是谁？**  
   是「AR 分布偏移」真能解释我们贴在 0.0353、破不了 0.0352 吗？还是其实是数据协议与论文不可比，0.035 已接近该公开集单步任务的饱和区？

2. **单步评测下，Pushforward 还有没有 ROI？**  
   若评委只看单步相对 L2，投入 4 epoch 探针值不值得？有没有「只改善 rollout、不改善单步」的高概率陷阱？

3. **gate=1e−4 是否过严？**  
   队内规定须 `best < baseline−1e−4` 才算 SIGNAL。这会导致 +0.14% 级别提升永远进不了版本链。从科研/竞赛哪边看更合理？有没有更稳的 promote 规则建议（在不灌水前提下）？

4. **modes/width 变差如何解释？**  
   容量增大反而远离主报，是优化失败、过拟合 128 test，还是归纳偏置被破坏？

5. **下一步唯一最优动作是什么？**  
   在「单卡、禁长训、未破 gate 不编 v、精度已停」纪律下：  
   - A) 坚持停精度，只答辩；  
   - B) 开 1 条 Pushforward；  
   - C) 开 1 条 STLW；  
   - D) 先做离线误差解剖再决定。  
   **请只选一个主推，并说明为什么否决其它。**

---

## 3. 最重要的上下文与前置信息

### 3.1 项目一句话

在壁仞 BIREN GPU 上用 SUPA/PyTorch Extension 实现 FNO 核心 **Spectral Convolution**，并搭建 ≥4 层 FNO 做二维 Navier-Stokes 涡度预测；全程 Agent 辅助开发；提交需可复现。

### 3.2 正式主报（不可改口径）

| 项 | 值 |
|----|-----|
| 公开 NS64 相对 L2 | **0.03530218452215195** |
| tag / 版本 | `freeze_r9` / 评测报告 **v8** |
| Spectral idle | **3.811 / 8.054 / 29.560 ms**（冻结） |
| 正确性 worst rel | ≈2.17e−7 ≪ 1e−4 |
| batch16 吞吐（旁注） | ≈1.60M grid_points/s |
| 设备 | Biren106B · `device=supa` · SDK `1.11.0.0.rc2` |
| 训练路径备注 | 提交 ckpt 多为 CPU 路径训练（`use_supa=False`）；推理可走 SUPA fused |

### 3.3 有效配方（已 KEEP）

轨迹（公开集）：

`scratch/boost → squeeze → multistep TF+soft → sched-sampling → soft → freeze_r9`

卖点卡片：

- 双角 fused SpectralConv（suFFT + SUPA mul）  
- residual + 高频损失 + 周期 roll  
- sched → soft → freeze spectral 抛光  
- 公开集与自建 v2 **严格分栏**（v2≈0.005 不进主报）

### 3.4 已证伪 / 禁止再建议的方向

| 方向 | 结果摘要 |
|------|----------|
| sched deepen r4/r6/r7 | 近失或 Δ=0 |
| soft_r10 | Δ=0，early_stop（test 变差） |
| freeze_r10 | 近失 0.035287；autochain 0.035252 **未破 gate**，已回滚 |
| hybrid / soup | 无提升或弱于主报 |
| modes20 / width48 | best≈0.0367 / 0.0361，**差于主报** |
| hard_reweight A1 | Δ=0 NO_SIGNAL |
| 解冻 Spectral formal / 挖 C2R | 平台墙；No-Go |
| R13 ping-pong / R14 fused IN+GELU | 正确性 FAIL 或更慢 → ROLLBACK |
| F-FNO 换主报、TTA 计主报 | 口径/衔接 No-Go |

### 3.5 队内门禁纪律（建议必须遵守）

```
gate = live_best - 1e-4
例：0.035302 → gate 0.035202

仅当 best < gate 且人工确认 → 才可 promote / 编评测报告新 v 号
探针：nohup + --stop-on-gate；epochs≤4；patience≤2
单卡串行；禁止默认 test_perf 覆写 formal ms
自动 promote 须破 gate + ALLOW_AUTO_PROMOTE=1
```

### 3.6 可视化旁证（误差并不均匀）

冻结 ckpt 上 strip（数量级）：

- best sample L2 ≈ 0.020  
- median ≈ 0.030  
- worst ≈ 0.088–0.089（难例尾巴很长）  

说明平均 L2 可能被**尾部难例**拉高；A1 难例重加权已试过但 Δ=0。

### 3.7 外部灵感（我们已扫过、待你裁定）

| 机制 | 状态 |
|------|------|
| Pushforward / MP-PDE | 未落地；与现有 sched **机理正交**（候选） |
| STLW（时间步损失课程） | 未落地；与等权多步损失正交（候选） |
| iMODE（modes 8→16 渐进） | 后备；勿滑向 modes20 |
| LOGLO / 分带谱损 | 与现有 `highfreq_rel_loss` 近亲，默认低优先 |

### 3.8 仓库关键路径（便于你引用，无需我粘贴代码）

- 模型：`fno_ns/model.py`  
- 公开集训练：`fno_ns/train_public_ns64_boost.py`、`train_public_sched_sampling.py`  
- 难例探针：`fno_ns/train_public_hard_reweight_probe.py`  
- Promote 门禁：`fno_ns/promote_guard.py`  
- 失败矩阵：`results/experiment_matrix.md`  
- 现行指针：`results/run_logs/CURRENT.md`  
- 主报：`results/summary.json` → `fno_ns.public_ns64`  
- 算子分段：`results/run_logs/spectral_fused_segments_2026-08-01.md`  

---

## 4. 请 GPT 直接回答的「最终问题」

> 在遵守我们门禁与已证伪清单的前提下：  
> **若只能再做一件事来冲击公开 NS64 单步相对 L2（目标破 0.035202），应该做什么？若你认为不该再冲击精度，请明确说「停精度，只答辩」，并给出评分向替代动作。**

请给出：

- 主推动作名称  
- 为什么它打在「AR 错配 / 协议差 / 难例尾」中的哪一个  
- 最小实验参数（epochs、是否 freeze spectral、t_out、损失公式要点）  
- 成功/失败判据（对照 gate）  
- 失败后的永久停止条件  

---

## 5. 附件级数字速查

```
主报 L2:     0.03530218452215195   (freeze_r9, v8)
gate:        0.03520218452215195   (baseline - 1e-4)
近失:        0.035287              (freeze_r10 手跑)
弱提升回滚:  0.035252              (未破 gate, 已 demote)
A1:          0.035302 Δ=0
modes20:     ~0.03675              (差于主报)
width48:     ~0.03613              (差于主报)
Spectral:    3.811 / 8.054 / 29.560 ms formal idle FROZEN
```

---

**文件结束。** 请从 §1 调研任务与 §2 困惑点开始回答；§4 给唯一主推决策。
