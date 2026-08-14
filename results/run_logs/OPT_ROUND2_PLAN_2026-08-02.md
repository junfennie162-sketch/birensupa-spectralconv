# OPT_ROUND2 · 2026-08-02 新一轮优化总方针

> **历史**：已被后续 Round 接棒。  
> **现行**见 [`CURRENT.md`](CURRENT.md) · [`OPT_ROUND7_PLAN_2026-08-02.md`](OPT_ROUND7_PLAN_2026-08-02.md)。  
> 本文档保留为 R0–R2 历史执行副本；下链曾为 [`OPT_ROUND3_PLAN_2026-08-02.md`](OPT_ROUND3_PLAN_2026-08-02.md)。  
> 依据：评测报告 + 四路并发 agent 召回 review。  
> 工作区：`/workspace/ai4s-f` · 合入：`/workspace/ai4s/submission` · **勿改 `ai4s-n`**

## 0. 基线（不可漂移）

| 角色 | 指标 | 数值 |
|------|------|------|
| FNO 精度 | public NS64 rel-L2 | **0.036092**（`sched_samp_r2`） |
| Spectral 性能 | idle 64/128/256 | **3.811 / 8.054 / 29.560 ms**（冻结） |
| Spectral 正确性 | worst rel | ~2.17e-7 |
| 相位 | submit_gate | done |

真源：`summary.json` → `fno_ns.public_ns64` / `spectral_conv.perf`。

## 1. 主攻序

```text
Wave-R0  零 GPU：口径清扫 + 叙事专页 + Agent 抽查     ← 立即
Wave-R1  低 GPU：batch16 公开语义 + irregular/bwd/3d  ← ≤45 min idle
Wave-R2  精度探针：soup → sched-sampling → flip/noise  ← CPU 串行
Wave-RX  永久 No-Go（见 §4）
```

## 2. Wave 状态（滚动）

| Wave | 状态 | 备注 |
|------|------|------|
| R0 | **done** | 口径 + 叙事 + dry-run 全绿 |
| R1 | **done** | public batch16 + irregular/bwd/3d；formal ms 未改 |
| R2 | **done / promoted** | soup 无信号；sched-sampling → L2 **0.036092**；跳过 R2-2/R2-3 |
| 合入 | **done** | `maintain check submit_gate` PASS；pack `fandougarden_submit_20260802_090910`；synced `/workspace/ai4s/submission` |

## 3. Wave-R2 精度纪律

- eval：1000/128 / seed=20260722 / step-1；架构默认 width32 modes16 4 层
- promote gate：test L2 **&lt; 0.036476**
- 串行：R2-0 EMA/soup → R2-1 scheduled sampling → R2-2 flip/noise →（条件）R2-3 width48→KD
- 默认不做：multiwin / F-FNO 主报 / 同构 squeeze / TTA 计主报

## 4. 全局 No-Go

1. 同构再 squeeze / 解冻 Spectral formal ms  
2. `torch.fft@SUPA` / Plan2d / 真融合热路径  
3. F-FNO/AFNO 替换必选算子进主报  
4. TTA 计主报 L2；v2/0.005144 伪官方  
5. SOL/proxy 得分句；SUPA fused 接长训  
6. 与 `ai4s-n` 或长训并发写 formal GPU  

## 5. 条件项（评审点名）

- A4 SUPA 训吞吐旁注；A1 微 epoch SUPA 训 smoke（禁 promote）  
- 官网级 3D 四角补全；B 站/成片视频  

## 6. 成功标准

- R0：口径唯一 + 抽查卡齐全  
- R1：batch16 公开语义 + irregular 机读字段  
- R2：有 promote 则 L2 &lt; 0.036476；无信号书面停精度线  
- Spectral formal 三档与正确性相对本轮零回退  
