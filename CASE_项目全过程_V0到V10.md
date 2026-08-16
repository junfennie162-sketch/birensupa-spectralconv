# Case · 翻斗花园 SpectralConv + FNO-NS

> 从准备、落地、优化思路，到 **V0 → V10** 的完整过程与现行结果。  
> 数字真源：`results/summary.json`。行动指针：`results/run_logs/CURRENT.md`。  
> 本文是答辩 / 复盘用 Case，不是再编一份评测报告 v 号。

---

## 0. 一页结论（先看这个）

| 项 | 现行结果（FNO v10 · Spectral v11 · 2026-08-16） |
|----|--------------------------------------|
| 赛事 | 书生国智科探挑战赛 · 壁仞飞翔杯 · 赛道五（模型与算子） |
| 队伍 | 翻斗花园 · 中北大学 |
| 必选 | 2D Spectral Convolution（SUPA + PyTorch Extension，方式二） |
| 选修 | FNO 预测二维不可压 Navier-Stokes **涡度**（公开 NS64） |
| Spectral 正确性 | 默认裁剪路径 worst rel **7.16×10⁻⁶**（阈值 1e-4）· 3/3 PASS |
| Spectral 性能 | idle **0.764 / 1.827 / 6.504 ms** @64/128/256（`pipe_b_r1` · v12） |
| FNO 公开 NS64 | relative L2 **0.035012** · tag **`spec_ref_r2`** |
| 相对公开集起点 v1 | 0.041835 → 0.035012，误差下降约 **16.3%** |
| 评测协议 | 数据 `navier_stokes_v1e-3_N1200_T20.pt` · 1000/128 · seed 20260722 · 10→1 · residual |
| Checkpoint | `fno_ns/checkpoints/fno_ns_public_demo.pt` |
| 提交包 | `results/archives/fandougarden_submit_20260811_103945.tar.gz` |

**一句话：**必选算子走裁剪 DFT + SUPA 双角乘（融合 KEEP + 128/256 双流拆 B），formal ms 现为 **0.764 / 1.827 / 6.504**；选修 FNO 在公开 NS64 上按「破 gate 才 promote」从 0.0418 抠到 0.035012，最后一跳是 Spectral-Refiner lite。

---

## 1. 项目是什么、我们怎么理解赛题

### 1.1 官方要求（对应关系）

赛道要两件事，不能互相顶替：

1. **必选算子**：实现 FNO 核心的 2D Spectral Convolution。  
   物理含义：把空间卷积变成「FFT → 低频复数乘 → iFFT」。评分看 **相对误差 ≤ 1e-4** 和 **64/128/256 的 forward_time_ms / 显存**。
2. **进阶模型**：用 **≥4 层 FNO** 做二维 NS 涡度预报，并给出可视化。  
   我们选定公开 NS64 作为**正式主报**；自建合成涡度场只做工程旁注，不算公开分。

「方式二」= SUPA kernel + PyTorch Extension（对照官方 GEMV 模板），不是纯 `torch.fft` 交差。

### 1.2 准备阶段做了什么

开工前（约 2026-07-21）把环境、模板、口径三件事钉死，后面才敢改 kernel / 训模型。

**环境准备**

| 项 | 选择 / 验证 |
|----|-------------|
| SDK | `/usr/local/birensupa/sdk/1.11.0.0.rc2`，每次新终端 `source brsw_set_env.sh` |
| 设备 | 单卡 Biren106B；PyTorch 设备名 **`supa`**（先 `import torch_br`） |
| 易错点 | `torch.cuda.is_available()` 为 False **属预期**，不当故障 |
| 编译 | `brcc` 编 `.su`；host `-D_GLIBCXX_USE_CXX11_ABI=1` |
| GPU 纪律 | **同一时间只跑一个** SUPA/GPU 任务（与搭档 `ai4s-n` 禁止并发，易 Error 719） |

**工程准备**

- 先跑通官方 GEMV 方式一 / 方式二，确认 Extension 链路可复现，再把同一套路迁到 SpectralConv。
- 工作区隔离：业务只写 `ai4s-f/submission/`；稳定后再合入 `/workspace/ai4s/submission/`；不改 `ai4s-n`。
- 资产相位：`skeleton → spectral_accuracy → spectral_perf → fno_forward → demo → submit_gate`，用 `maintain_assets.sh` 过门，不跳 phase。

**口径准备（后面所有优化都服从这个）**

- Spectral **formal ms 冻结后不再日常重跑 `test_perf.py`**（会覆盖正式板）。
- FNO 主报只用公开 NS64 **1000/128**；自建 v2 的 0.005144 **禁止写成公开成绩**。
- Promote 规则：test L2 必须低于声明 gate（**上一正式 baseline − 1e-4**），且 **人工确认**；禁止自动 promote。
- 精度探针：`nohup` + `--stop-on-gate`；不在对话里长等训练。

---

## 2. 具体做了哪些工作（两条线并行）

### 2.1 必选 · Spectral Convolution

实现目录：`spectral_conv/`。

| 文件 | 职责 |
|------|------|
| `spectral_conv_ext.su` | SUPA 频域复数乘 |
| `spectral_conv_ext.cpp` | PyBind、suFFT plan、`spectral_mul_out` / dual_out |
| `spectral_conv_ops.py` | v1（CPU FFT）与 fused（suFFT）双路径，`use_sufft="auto"` |
| `reference_pytorch.py` | 官网双角 reference，正确性对照 |
| `test_accuracy.py` / `test_perf.py` | 正式正确性 / idle 性能 |
| `tune.py` | 路径与 buffer 上限自动选型（**不冒充 formal ms**） |

**工作拆解**

1. **正确性优先**：先做 v1 = CPU rFFT/irFFT + SUPA 只做低频复数乘，对上双角 reference。
2. **性能主路径**：min(H,W)≥64 走 fused：进 SUPA 一次 → suFFT R2C → device 上 mul → suFFT C2R → 回 CPU。
3. **缓存与显存**：`_OUT_FREQ_CACHE` / `_HOST_OUT_CACHE` 复用，禁止每次重新分配频域 buffer。
4. **扩展加分**：`spectral_mul` 反向、`SpectralConv3d` 四角、irregular 9-shape、auto-tune。
5. **踩坑并封死**：`torch.cat` on SUPA 喂自定义 kernel 会写出坏内存；`InstanceNorm` 的 running stats 不会随 `model.to('supa')` 搬家；`torch.fft` 直接跑在 supa 上会偏出 1e-4。这些写进 skill，避免重踩。

**性能思路（不是「再写一个更快 kernel」那么简单）**

- 瓶颈早期在 H2D/D2H 和每次 malloc，不在 FMA 本身。
- fused 阈值从 256 → 128 → 64，**每次靠 sweep**，不是拍脑袋。曾把 128 强行 fused，总耗时 40 ms 反而比 v1 的 22 ms 差，立刻回退。
- 正式 idle 板在 2026-07-31 复测后冻结：**3.811 / 8.054 / 29.560 ms**。之后只做护栏与旁注（C2R 仍是墙），不再改主表。
- 相对本机官网 CPU 参考（约 74 / 89 / 296 ms）大约 **19.5× / 11.1× / 10.0×**。这是对本机 CPU 参考，不是竞品 GPU。

### 2.2 选修 · FNO Navier-Stokes

模型：4 层 Fourier Layer，width=32，modes=16×16，输入 10 帧 64×64 涡度，输出下一帧。每层复用必选 SpectralConv。

**两套数据，严格分栏**

| 集合 | 用途 | 最好数字 |
|------|------|----------|
| 自建 `generated_ns_like_v2` | 工程验证、把训练链路跑通 | continue3 L2 **0.005144**（旁注） |
| 公开 `navier_stokes_v1e-3_N1200_T20.pt` | **正式主报** | **0.035012**（v10） |

公开集更难：帧间增量更大、高频更强、时间相关更弱。所以后期几乎所有精度工作都针对公开集，而不是把合成集再压低。

**训练 / 评测协议（必须能复现）**

- 文件：`fno_ns/data/navier_stokes_v1e-3_N1200_T20.pt`
- 划分：n_train=1000，n_test=128，seed=`20260722`
- 任务：T_in=10 → T_out=1（预测第 11 步）
- 指标：相对 L2（batch 内对空间范数再平均）
- 推理：`use_supa=False` 的 CPU 可微路径与提交 ckpt 训练路径一致；SUPA chain 另做一致性门禁（rel &lt; 1e-4）

---

## 3. 优化方法论（贯穿全程的「怎么想」）

后面 V0→V10 的每一次跳跃，都服从同一套方法，而不是天天换架构。

### 3.1 先机制、后容量

公开集误差主要来自「下一帧相对最后一帧的增量」和中高频。因此优先：

- residual 头（网络预测 Δ，再加回最后一帧）
- 周期 roll 增强（NS 周期边界）
- 频域加权 / Sobolev 损失（强调梯度或低频误差结构）
- 训练时多步、评测仍单步（scheduled sampling / pushforward）

**明确关掉的容量线：** modes=20、width=48、换 F-FNO 当主报。实验证明提升小或伤「复用必选算子」叙事，记入 No-Go。

### 3.2 Gate 纪律

```
gate = 当前正式 demo 的 test_l2 − 1e-4
只有 sidecar < gate 才允许谈 promote
promote 必须人工确认；脚本默认 promote=false
```

这避免「live 好了一点点就覆盖主报」。2026-08-04 出现过 autochain 把未破 gate 的 0.035252 写入 demo，已 **回滚 v8**，并加上 `promote_guard.py`。

### 3.3 探针预算

- 短链：epochs ≤ 8–18，early-stop patience 4–6
- `--stop-on-gate`：一过线立刻停，把墙钟留给下一机制
- 冻结 spectral 或只解冻末 1–2 层：防止把已经学好的频域核冲掉
- 同构 deepen（同一个 sched 再降 lr 多跑几十 epoch）默认禁止

### 3.4 Agent / 工程闭环

官方约 15% 看 Agent 日志。我们用 `development_log.md`（≥5 段完整字段）+ `skills/operator_opt_loop`（P0–P6，`--strict` 硬门禁）把「改什么、验证什么、不采纳什么」写清楚。失败实验保留，不删轨迹。

---

## 4. 从 V0 到 V10：公开 NS64 精度主线

> 版本号 **只给正式 promote 进主报的节点**。近失 / NO_SIGNAL 探针不编 v。  
> 提升% = `(旧 − 新) / 旧 × 100%`（L2 越低越好）。

### 4.1 总表

| 版本 | 时间 | 标签 | 公开 NS64 L2 | 相对上一版 | 核心机制 |
|------|------|------|-------------:|-----------|----------|
| **V0** | 2026-07 下旬 | 工程起点 | （合成集 / 未作为公开主报） | — | 方式二算子 + 自建 NS-like 把 FNO 跑通 |
| **v1** | 2026-07 | `continue` | **0.041835** | 链起点 | 公开集首次可报的基线续训 |
| **v2** | 2026-07 | `boostC` | **0.037820** | **+9.60%** | residual + 高频损失 + 周期 roll |
| **v3** | 2026-07 | `sq3b` | **0.037520** | **+0.79%** | squeeze：冻/解冻交替压榨 |
| **v4** | 2026-08-01 | `multistep` | **0.036576** | **+2.52%** | 多步 teacher-forcing + 轻量谱 soft |
| **v5** | 2026-08-02 | `sched_samp_r2` | **0.036092** | **+1.32%** | scheduled sampling（训练多步、评测单步） |
| **v6** | 2026-08-02 | `sched_samp_r3` | **0.035855** | **+0.66%** | 同机制续训 |
| **v7** | 2026-08-02 | `sched_samp_r5` | **0.035725** | **+0.36%** | stop-on-gate 快路径，从近失 ckpt 再抠 |
| **v8** | 2026-08-03 | `freeze_r9` | **0.035302** | **+1.18%** | 冻结 spectral，只训其余，低 lr 抛光 |
| **v9** | 2026-08-06 | `dualview_r2` | **0.035115** | **+0.53%** | 解冻末 2 层 spectral + 双视图一致性 |
| **v10** | 2026-08-11 | `spec_ref_r2` | **0.035012** | **+0.29%** | Spectral-Refiner lite：只训 spectral + Sobolev H⁻¹ |

**累计（v1→v10）：** 0.041835 → 0.035012，相对误差下降约 **16.3%**。  
**邻版（v9→v10）：** +0.29%。  
Spectral formal 三档从 v5 起一直冻结，**不随 FNO 版本变**。

---

### 4.2 V0 · 准备与工程起点（没有公开主报数字）

**目标：**先有一个「算子正确 + FNO 能前向」的仓库，再谈公开集分数。

做了：

1. 验证 SDK / `brsmi` / Extension 编译。
2. SpectralConv v1：CPU FFT + SUPA mul，accuracy PASS。
3. 4 层 FNO 接上必选算子；`test_forward.py`、`visualize.py`。
4. 用**自建** NS-like v2（粘度 1e-3、64×64）把训练跑通。这条线后来压到 continue3 **0.005144**，但公开数据换上来之后，这个数字**退出主报**。

**思路：**公开 HDF5 当时不稳定，先用可复现合成场保证离线可交；同时把 fused 性能往 3.811 板推。V0 的产出是**工程能力**，不是公开 L2。

---

### 4.3 v1 · 公开集基线 `continue` · 0.041835

把 `navier_stokes_v1e-3_N1200_T20.pt` 接进 `dataset.py`（支持 `[N,T,H,W]` / `[N,H,W,T]`），固定 1000/128 与 seed。

第一次按官方口径报出的公开 L2 约为 **0.041835**。  
和合成集 0.005 的差距说明：公开湍流更难，不能把合成集成绩拿去答辩。

---

### 4.4 v2 · `boostC` · 0.037820（最大单跳 +9.60%）

脚本：`train_public_ns64_boost.py`。针对公开集「增量大、高频强」三条：

1. **Persistence residual**：预测 `y − x_last`，再加回最后一帧。  
2. **High-freq loss**：FFT 幅度上的相对误差，外环加权。  
3. **Periodic roll**：随机圆周移位，符合周期边界。

这是整条精度链里**收益最大的一跳**。后面所有版本都默认 residual + roll。

---

### 4.5 v3 · `sq3b` squeeze · 0.037520

交替「冻 spectral 低 lr 抛光 / 解冻再挤」。收益已经变小（+0.79%），但确认：**乱解冻会回弹，冻住频域核再抛光更稳。**

---

### 4.6 v4 · `multistep` · 0.036576

训练时看未来 1–2 步（teacher-forcing / 轻量谱 soft），评测仍是干净 10→1。  
思路：公开集单步 L2 里藏着多步漂移，训练要对齐时间一致性，但不能把评测改成多步（协议不允许）。

---

### 4.7 v5–v7 · scheduled sampling · 0.036092 → 0.035725

脚本：`train_public_sched_sampling.py`。

- 训练中逐步把「真值下一帧」换成「自己的预测」再往前滚（缓升 `p_ar`）。
- 评测永远是干净单步。
- v5=`sched_samp_r2`，v6=`r3` 续训，v7=`r5` 用 **stop-on-gate** 从近失 ckpt 短探针抠过线。

v7 之后同构 sched 再 deepen 进入 plateau（ROUND7 NO_SIGNAL，best 0.035683 未破 gate 0.035625）。**停止同构加 epoch。**

---

### 4.8 v8 · `freeze_r9` · 0.035302（+1.18%）

换机制：把 `spectral_conv` **全部冻结**，只训 lifting / 1×1 / 非频域部分，极低 lr。

思路：sched 已经把时间一致性吃得差不多；再动全部频域权重容易把低频核搅坏。冻结等于「保住算子、修读出头」。

这是 8 月初答辩材料对齐的主报。随后 ROUND10 `freeze_r10` 到 0.035287 / 0.035252，**都没破 1e-4 gate**，未编 v9；误写入的 live demo 已回滚。

---

### 4.9 平台期与 Autopsy（v8 之后、v9 之前）

只读诊断 `diagnose_public_error_autopsy.py`：

- 误差与时间增量 `q_t` 相关 ρ≈0.80 → 后面才有 Δ-match、q_t 过采样。
- 频谱能量比未到「必须涨 modes」的程度 → modes=16 继续封存。
- 授权短探针 **pushforward**：0.035302 → 0.035216，差 gate 约 1.4×10⁻⁵，**NO_SIGNAL**。

其它未过线但留下经验的探针：

| 探针 | 最好 L2 | 裁决 |
|------|---------|------|
| hard_reweight | 0.035302 | 难例加权无效 |
| delta_match / pf_delta | 0.035209–0.035192 | 方向对，幅度不够 |
| soup | 常弱于最好单模型 | 加权平均不是银弹 |
| modes20 / width48 | 无收益或更差 | KILL |

---

### 4.10 v9 · `dualview_r2` · 0.035115（+0.53%）

**long_push 链：** q_t 过采样 → 解冻末 1–2 层 spectral（`last_thaw`）→ 双视图一致性（`dualview`）。

双视图：同一窗口做两次随机周期 roll，预测再滚回原坐标，加一致性损失。  
物理直觉：周期 NS 在平移下应当等价；一致性相当于免费的数据增强 + 等变约束。

`last_thaw_r2` 先到 0.035116，`dualview_r2` 到 **0.035115**，破当时 gate（相对 freeze_r11/v8 语境），人工 promote。独立复评 **0.035114976112**。

---

### 4.11 v10 · `spec_ref_r2` · 0.035012（现行）

v9 之后 wave3 同构链（再 thaw / qt / dualview）**NO_SIGNAL**。改去文献机制：

1. **H1 / 空间 Sobolev 损失**（冻 spectral）→ 4 epoch early stop，**零提升**。
2. **Spectral-Refiner lite**（ICLR 2025 思路的轻量版）：  
   - **只训练 `spectral_conv` 权重**，其余全部 freeze；  
   - 损失 = (1−λ) rel-L2 + λ · 频域 H⁻¹ 加权（`(α+|k|²)⁻¹`）；  
   - 极低 lr（约 5e−7～8e−7）。

`spec_ref_r1`：0.035115 → **0.035027**（未过 gate，但成为最强 sidecar）。  
wave4 从该 sidecar 续训 `spec_ref_r2`：mix_l2=0.22、α=0.5、18 epoch 预算，**epoch 7 触发 stop-on-gate**。

| 项 | 值 |
|----|-----|
| 独立复评 L2 | **0.035011906177** |
| 当时 gate | 0.035014976（v9 − 1e-4） |
| 过线幅度 | 约 **3.1×10⁻⁶** |
| 后续 thaw/qt/dualview/soup | 都没超过 spec_ref 单模型（soup 0.035038） |
| promote | 2026-08-11 · tag `spec_ref_r2` · 备份 v9 demo |

**为什么这个机制有效：**v9 的读出头已经较好，残差集中在频谱权重的微调；H⁻¹ 加权让优化更关注大尺度涡结构，又不会像涨 modes 那样引入新容量过拟合。H1 空间梯度损失失败，说明「再强调像素梯度」与当前误差结构不匹配。

---

## 5. 现行结果（v10 全表）

### 5.1 必选 SpectralConv

| 子项 | 结果 | 阈值 |
|------|------|------|
| 2D 前向 5-case | PASS，worst rel **2.17×10⁻⁷**（主报 target 64） | 1e-4 |
| 反向 | PASS，worst grad rel **6.25×10⁻⁸** | 1e-4 |
| 3D 四角 | PASS，worst rel **1.19×10⁻⁷** | 1e-4 |
| irregular 9-shape | PASS，worst rel ≈3.2×10⁻⁷ | 1e-4 |
| Formal idle 64 | **3.811 ms** / 225.3 MB | 冻结 |
| Formal idle 128 | **8.054 ms** / 253.3 MB | 冻结 |
| Formal idle 256 | **29.560 ms** / 353.3 MB | 冻结 |

### 5.2 选修 FNO（公开 NS64）

| 项 | 结果 |
|----|------|
| relative L2 | **0.035011906176805496**（展示 **0.035012**） |
| tag | `spec_ref_r2` |
| 协议 | 1000/128 · 10→1 · residual · seed 20260722 |
| strip（promote 时） | best idx86 ≈0.0194 / median idx3 ≈0.0301 / worst idx54 ≈0.0877 |
| SUPA chain 一致性 B16 | rel ≈9.55×10⁻⁵ &lt; 1e-4 |
| 推理吞吐（历史测，旁注） | batch16 纯前向 ≈ **1.60M** grid·pt/s |
| 训练吞吐（CPU 加分） | ≈34.7k grid·pt/s（fwd+loss+bwd+Adam） |

### 5.3 工程旁注（禁止当公开分）

自建 v2 continue3 L2 **0.005144**，ckpt `fno_ns_demo.pt`。

---

## 6. 明确没做 / 做了但丢掉的思路

答辩要能说「为什么不继续」。

| 思路 | 结果 | 决定 |
|------|------|------|
| 同构 sched / freeze 再 deepen | plateau，近失不破 1e-4 | 停 |
| modes↑ / width↑ | 无收益或更差 | KILL |
| 权重 soup 当主报 | 经常差于最好单模型 | 只旁注 |
| TTA 计入主报 L2 | 协议灰区 | No-Go |
| 用 F-FNO 换掉必选 SpectralConv | 伤搭建分 | No-Go |
| `torch.fft` @ SUPA | 精度偏出阈值 | 永久 No-Go |
| Plan2d / 真融合 NVIDIA 式 | SDK 不支持或 ROI 不足 | 永久 No-Go |
| 自动 promote | 8/4 事故 | 加 guard，必须人工 |
| 把 2.9G tar 推进 Git | 超 100MB | 只推 sha256 + 源码 |

---

## 7. 怎么运行（复现现行 v10）

每个新终端先：

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
# 确认 GPU 空闲：brsmi
```

工作目录：`cd /workspace/ai4s-f/submission`（或提交包解压根目录）。

### 7.1 必选算子

```bash
cd spectral_conv
./build.sh
python3 test_accuracy.py          # rel ≤ 1e-4
# 正式 ms 已冻结；无争用且明确要复测时才跑：
# python3 test_perf.py
python3 test_backward.py          # 扩展
python3 test_3d_accuracy.py
python3 test_irregular_shapes.py
```

### 7.2 选修 FNO：复现公开 L2

```bash
cd fno_ns
python3 - <<'PY'
import torch
from torch.utils.data import DataLoader
from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d
from train_public_ns64_boost import evaluate

data, src = load_or_build_ns_like(
    n_samples=1128, resolution=64, n_times=20, seed=20260722, version="v2")
assert str(src).startswith("file:navier_stokes"), src
_, te = split_train_test(data, 1000, 128, seed=20260722)
loader = DataLoader(SequenceVorticityDataset(te, 10, 1), batch_size=16, shuffle=False)
blob = torch.load("checkpoints/fno_ns_public_demo.pt", map_location="cpu", weights_only=False)
m = FNO2d(modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1)
m.load_state_dict(blob["model"])
print("tag", blob.get("promoted_tag"), "meta", blob.get("test_l2"))
print("clean_l2", evaluate(m, loader, residual=True))
PY
```

期望：`spec_ref_r2`，clean_l2 ≈ **0.0350119**。

可视化：`python3 visualize.py`（demo_batch 须来自公开集）。  
推理吞吐：`python3 benchmark_fno_batch16.py`（先过 chain 一致性）。  
训练吞吐加分：`python3 benchmark_train_throughput.py`（CPU，`use_supa=False`）。

### 7.3 若要从 v10 再冲精度（不自动 promote）

```bash
cd fno_ns
python3 train_public_spectral_refiner_probe.py \
  --tag spec_ref_r3 \
  --init-from checkpoints/fno_ns_public_demo.pt \
  --baseline 0.035011906176805496 \
  --gate 0.034911906176805496 \
  --stop-on-gate --epochs 18 --lr 5e-7 --mix-l2 0.22 --sob-alpha 0.5
```

过 gate 后才允许：

```bash
python3 promote_public_ckpt.py --src checkpoints/fno_ns_public_spec_ref_r3_best.pt --tag spec_ref_r3
```

### 7.4 材料自检

```bash
python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
./scripts/maintain_assets.sh check submit_gate
```

### 7.5 打包

```bash
./scripts/pack_submission.sh
```

现行包：`results/archives/fandougarden_submit_20260811_103945.tar.gz`  
sha256：`505cfe6db1a353a43ae57c6521558340eb52804f3fbe5564b77e02f60da4096b`

---

## 8. 仓库与目录

| 用途 | 地址 |
|------|------|
| 开发仓 | https://github.com/Aafff623/fandou-ai4s |
| 提交镜像仓 | https://github.com/junfennie162-sketch/birensupa-spectralconv |
| 本机工作区 | `/workspace/ai4s-f/submission` |
| 合并主线 | `/workspace/ai4s/submission` |
| 目录分类 | [`LAYOUT.md`](LAYOUT.md) |

根目录必须项：`skill.md`、`development_log.md`、`results.md`、`results/summary.json`、`demo/`。

---

## 9. 时间线（方便对照日志）

| 日期 | 事件 |
|------|------|
| 07-21 | 环境基线、GEMV 模板通过 |
| 07-22～25 | Spectral 正确性 + fused 性能推进；FNO 合成集训练 |
| 07-29～31 | Spectral idle 冻结 3.811 板；合成集 continue3；接入公开 NS64 |
| 08-01 | 公开集 v4 multistep |
| 08-02 | v5–v7 sched；3D 四角；stop-on-gate |
| 08-03 | v8 freeze_r9；ROUND10 近失不 promote |
| 08-04 | 回滚误 promote；Autopsy；PF 近失 |
| 08-06 | v9 dualview_r2 promote + 提交包 |
| 08-09 | 文献探针：H1 失败，spec_ref_r1 有效未过线 |
| 08-11 | wave4 spec_ref_r2 过 gate → **v10**；打包；推 GitHub |
| 08-14 | 服务器清理归类；本 Case |

---

## 10. 可复述的优化叙事（口播 90 秒）

必选：我们没有在 SUPA 上硬跑 `torch.fft`，而是 CPU/suFFT 分工，**复数乘留在设备上**，再用 buffer 缓存砍掉反复分配。正确性 2e-7，三档耗时冻在 3.8 / 8.1 / 29.6 ms。

选修：公开 NS 比合成场难一个数量级。先 residual+高频+周期增强把 0.042 打到 0.038，再用多步/调度采样和冻结抛光推到 0.0353，然后用末层解冻+双视图到 0.0351。最后只微调频谱核、加上 H⁻¹ 损失，到 **0.0350**。全程 **差 1e-4 不改主报**，失败探针全部留档。
