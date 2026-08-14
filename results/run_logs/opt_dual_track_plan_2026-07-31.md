# 双题压榨路线 · 2026-07-31（状态已按 OPT_MASTER_PLAN 回写）

前提：赛题合规（SUPA 核心、正确性门禁、官方划分评测、不混用 768/1000 L2）。  
主方针：[`OPT_MASTER_PLAN_2026-07-31.md`](OPT_MASTER_PLAN_2026-07-31.md)。  
**主报**：公开 NS64 1000/128；旁注 v2 不得冒充公开分。

## 阶段 A · 选修 FNO-NS（相对 L2）← 当前主战场

| # | 动作 | 状态 | 备注 |
|---|------|------|------|
| A0 | v2 global-continue | **done（旁注）** | 0.005470→0.005268；非公开主报 |
| A1–A2 | v2 freeze/continue3 | **done / FINAL（旁注）** | continue3 L2=0.005144；同构停挖 |
| A3 | width=48 从零（v2） | **v2 abort** | 公开集未测；仅 P2c 条件触发 |
| A4 | modes=20（v2） | **v2 abort** | 同上 |
| A5 | F-FNO | **可选对照** | CPU 探针有；不替换必选 SpectralConv |
| **P_pub** | 公开 NS64 scratch+continue | **done** | 基线曾 L2=0.041835 |
| **P2a** | boostA hf+aug → B residual → C continue | **done** | A **0.039612** → B 0.038575 → C **0.037820** |
| **P2b / squeeze** | residual+hf+freeze 多轮 | **done / plateau** | 终值 **0.037520**；sq4a 无提升后停算力 |
| A7 | 停条件 | **触发** | 连续无 promote；已 ≤0.038；进 P3 |

已否决（无新证据不重开）：matched-loss、modes=12（L2）、multiwin、naive wavg、v2 再同构压榨。

## 阶段 B · 必选 SpectralConv（ms）

| # | 动作 | 状态 |
|---|------|------|
| B1 | 复测 P7+P8b 正式三档 | **done** idle 3.811/8.054/29.560 |
| B2 | 封死清单 | **生效**（见 OPT_MASTER_PLAN §3.3） |
| B3 | 微 opt | **停挖**（irfft/C2R 墙） |
| B4 | 正确性护栏 | 保持；idle 复测 |

## 阶段 C · 双题总检 → 提交

见 OPT_MASTER_PLAN P0/P3；材料口径闸优先于微秒级挖空。

## 执行纪律

- 评测只用 **公开 1000/128** 决定 promote；768 / v2 仅旁注
- 同时只跑一条 CPU 重训；GPU formal 与训练/ai4s-n 互斥
- 有提升才合入 demo；否则只留日志
