# 项目上下文（翻斗花园 · 给云端 Agent）

## 1. 身份与赛题

| 项 | 值 |
|----|-----|
| 赛事 | 书生国智科探挑战赛 · 壁仞飞翔杯 |
| 赛道 | 赛道五 · 模型与算子（基于 Agent 的模型和算子开发） |
| 队伍 | **翻斗花园** · 中北大学 |
| 工作区标识 | `ai4s-f`（队长侧）；提交根 `submission/` |
| 代码仓（参考） | https://github.com/Aafff623/fandou-ai4s |
| 选题 | 必选 **Spectral Convolution** + 进阶 C **FNO-NS** |

## 2. 硬件与软件环境

| 项 | 值 |
|----|-----|
| GPU | 壁仞 **Biren106B** · **单卡** |
| SDK | `1.11.0.0.rc2` |
| `SUPA_BASE` | `/usr/local/birensupa/sdk/1.11.0.0.rc2` |
| 设备名 | PyTorch `device="supa"`（须先 `import torch_br`） |
| 注意 | `torch.cuda.is_available() == False` 为平台预期，不是故障 |
| 开发工具 | Cursor Agent（SSH · 竞赛 Docker） |

每个新终端环境（复现页可放）：

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
```

## 3. 实现路线（官方「方式二」）

- **方式二**：SUPA kernel + PyTorch C++/pybind **Extension**  
- 核心文件概念：  
  - `spectral_conv/*.su`：频域复数乘 SUPA kernel  
  - `spectral_conv/*.cpp` + `build.sh`：Extension / suFFT 计划与缓存  
  - `spectral_conv_ops.py`：CPU-FFT v1 对照路径 + **fused** 正式热路径  
- FNO：`fno_ns/model.py`，**4 层** Fourier，`width=32`，`modes=16×16`，`T_in=10` → 预测 1 步 64×64 涡度  
- FNO **正式推理复用**必选 fused SpectralConv（`use_sufft`），不是另写一套原生 FFT  

### 热路径一句话（架构页用）

`suFFT R2C → SUPA 双角 gather/scatter 复数乘 → suFFT C2R`；频谱尽量设备常驻；小图阈值 `min(H,W)≥64` 走 fused。

## 4. 数据与协议（精度页必须写清）

| 项 | 正式主报 | 旁注（勿混） |
|----|----------|--------------|
| 数据 | 公开 `navier_stokes_v1e-3_N1200_T20.pt`（HF NS64） | 自建 `generated_ns_like_v2` |
| 划分 | n_train=**1000** / n_test=**128** | 其它 split |
| seed | **20260722** | — |
| 主报 L2 | **0.035302**（`freeze_r9` · 评测报告 **v8**） | 顶层 legacy `rel_l2≈0.00249` **不是**公开主报 |
| Checkpoint | `fno_ns/checkpoints/fno_ns_public_demo.pt` | `fno_ns_demo.pt`（v2） |

## 5. 工程阶段与姿态

| 项 | 值 |
|----|-----|
| Phase | `submit_gate` **done** |
| 精度姿态 | **永久停**（近失未破 gate；不再开同构 deepen / PF / STLW） |
| Spectral formal ms | **冻结**（禁止默认重跑 `test_perf` 写主表） |
| 评测报告 | 唯一最新稿锚定 **v8**；本轮相对上一正式版提升 **0%**（无新 promote） |

## 6. 有效优化叙事（可上 PPT，勿编造额外数字）

**算子侧轨迹（概念）：**  
v1 CPU-FFT bridge → fused suFFT+SUPA mul → dual_scatter / packed → 正式 idle 冻结。

**FNO 精度轨迹（公开集）：**  
boost → squeeze → multistep → sched sampling → soft → **freeze_r9 = 0.035302（v8）**  
邻版历史：v7 0.035725 → v8 0.035302（约 **+1.18%** 误差下降）。  
有效配方关键词：residual + high-freq + aug + sched→soft→freeze。

## 7. 失败与诚实（必须有一页）

| 裁决 | 例子 | 数字要点 |
|------|------|----------|
| KEEP | `freeze_r9` | L2 **0.035302** · v8 |
| ABORT / 回滚 | freeze_r10 autochain 弱结果 | 0.035252 **未破** gate 0.035202 → **已回滚** |
| NO_SIGNAL | A1 hard_reweight；soft_r10；`pf_clean_r1` | PF best **0.035216** 仍未破 gate |
| KILL | 同构 deepen / modes20 远差 / 解冻 formal | 不再做 |

## 8. Agent / Skills 上下文

- 日志：`development_log.md`，记录 **1–38**（远超官方 ≥5）  
- 场景索引覆盖：kernel / bottleneck / hyperparam / platform / data / viz  
- 精品抽查锚点（PPT 可列）：**26 / 30 / 32 / 33 / 35**（可加 36–38 材料与 PF）  
- `skill.md` + 可复用 `skills/operator_opt_loop`（P0–P6）  
- 工程判别：`promote_guard`——须破 gate 且显式允许才可自动 promote  

## 9. 扩展证据（扩展分 15%）

| 项 | 结果（圆整） |
|----|--------------|
| Backward | PASS，worst grad rel ≈ 6.25e-8 |
| SpectralConv3d 四角 | PASS，worst ≈ 1.19e-7；**声明 ≠ 3D FNO** |
| irregular shapes | 9/9 PASS |
| chain CPU vs SUPA | random ≈ 6.58e-5；ckpt ≈ 4.76e-5（均 < 1e-4） |

## 10. 当前材料状态（给 Agent 的「为什么需要 PPT」）

- 必交源码 / 日志 / skill / 结果 **已齐**  
- 展示材料：有 **Markdown 页稿** 与图，**缺可投屏真 PPTX**  
- 你的任务：把冻结内容落成正式答辩 PPT，对齐官方评分轴  

## 11. 仓库路径提示（若评委口头问「在哪」）

云端 Agent **不必**访问这些路径，但备注页可写：

- 提交根：`submission/`  
- 算子：`submission/spectral_conv/`  
- FNO：`submission/fno_ns/`  
- 结果：`submission/results.md` · `results/summary.json`  
- 评委一页包：`results/run_logs/JUDGE_3MIN_PACK_2026-08-04.md`  
- Demo 图：`demo/media/fno_ns_pred_vs_gt_2026-08-02.png`（现行）  
